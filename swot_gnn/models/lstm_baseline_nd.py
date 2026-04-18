"""
LSTM-only baseline for multi-day lake WSE forecasting.

Architecture mirrors SWOT-GNN (InputEncoder → StepTypeEmbedding → LSTMBlock × N → ForecastHead)
but omits the GraphGPS spatial propagation step entirely.  Each lake is
processed independently; no message passing between neighbours.

Use this model as a benchmark to isolate the contribution of graph
connectivity in the full SWOT-GNN model.

Classes exported:
    LSTMBaselineMultiStep         — deterministic point forecast
    LSTMBaselineMultiStepGauss    — probabilistic Gaussian forecast (mean + log_std),
                                    trained with CRPS over multiple lead days
"""
import torch
import torch.nn as nn
from typing import Optional

from .swot_gnn import InputEncoder, StaticEncoder, ForecastHead


class LSTMBlock(nn.Module):
    """
    Temporal-only block: bidirectional 2-layer LSTM with static feature injection.

    Mirrors the temporal half of STBlock (models/st_block.py) but removes the
    GraphGPS spatial step.  Each node (lake) is processed independently along
    the time axis.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        static_embed_dim: int = 0,
    ):
        super().__init__()
        self.static_embed_dim = static_embed_dim
        # Bi-LSTM input: dynamic features + static embedding (concatenated per timestep)
        lstm_input_size = in_dim + static_embed_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim // 2,   # concat of both directions → hidden_dim
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:      (num_nodes, seq_len, in_dim)
            static: (num_nodes, static_embed_dim)
        Returns:
            (num_nodes, seq_len, hidden_dim)
        """
        _, seq_len, _ = x.size()
        if static is not None and self.static_embed_dim > 0:
            static_exp = static.unsqueeze(1).expand(-1, seq_len, -1)   # (N, T, static_embed_dim)
            x = torch.cat([x, static_exp], dim=-1)                     # (N, T, in_dim + static_embed_dim)
        lstm_out, _ = self.lstm(x)                                      # (N, T, hidden_dim)
        return self.norm(self.lstm_dropout(lstm_out))


class LSTMBaselineMultiStep(nn.Module):
    """
    LSTM-only baseline for multi-day lake WSE forecasting (no graph).

    Architecture:
        InputEncoder → [LSTMBlock × st_blocks] → ForecastHead

    Each lake is processed independently across time; there is no spatial
    message passing between lakes.  The forward interface is intentionally
    compatible with SWOTGNN (minus edge_index / batch), so the same
    training and inference infrastructure can be reused with minimal changes.

    Parameters mirror SWOTGNN.__init__ exactly, omitting gps_layers_per_block
    and gps_heads which are not applicable without GraphGPS.
    """

    def __init__(
        self,
        swot_dim: int = 8,
        climate_dim: int = 13,
        embed_dim: int = 64,
        hidden_dim: int = 32,
        st_blocks: int = 2,
        dropout: float = 0.5,
        static_dim: int = 33,
        static_embed_dim: int = 32,
        forecast_horizon: int = 1,
    ):
        super().__init__()
        # Separate MLPs for SWOT features and climate forcing → embed_dim
        self.encoder = InputEncoder(
            swot_dim=swot_dim,
            climate_dim=climate_dim,
            embed_dim=embed_dim,
        )
        # Encode static attributes to a flat embedding injected into every LSTMBlock
        self.static_encoder = StaticEncoder(
            static_dim=static_dim,
            embed_dim=static_embed_dim,
        )
        self.static_embed_dim = static_embed_dim
        self.forecast_horizon = forecast_horizon
        # Learned step-type embedding: 0 = historical step, 1 = forecast step.
        # Added to the encoded features before the first LSTMBlock so the LSTM
        # receives an explicit signal about the observation regime switch.
        self.step_type_embed = nn.Embedding(2, embed_dim)
        # Stack of LSTM-only blocks: first takes embed_dim, rest take hidden_dim
        self.lstm_blocks = nn.ModuleList()
        for i in range(st_blocks):
            in_d = embed_dim if i == 0 else hidden_dim
            self.lstm_blocks.append(
                LSTMBlock(
                    in_dim=in_d,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    static_embed_dim=static_embed_dim,
                )
            )
        # Maps the forecast-step hidden states → WSE per node (scalar or multi-step vector)
        self.forecast_head = ForecastHead(
            hidden_dim=hidden_dim, dropout=dropout, forecast_horizon=forecast_horizon
        )

    def _encode(
        self,
        x: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Shared encoder: InputEncoder → step-type embedding → LSTMBlocks.

        Factored out so subclasses can call it without duplicating logic.

        Returns:
            h: (num_nodes, seq_len + forecast_horizon, hidden_dim)
        """
        h = self.encoder(x)
        T = h.size(1)
        step_types = torch.zeros(T, dtype=torch.long, device=h.device)
        step_types[-self.forecast_horizon:] = 1
        h = h + self.step_type_embed(step_types).unsqueeze(0)
        static_emb = self.static_encoder(static_features) if static_features is not None else None
        for block in self.lstm_blocks:
            h = block(h, static=static_emb)
        return h

    def forward(
        self,
        x: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:               (num_nodes, seq_len + forecast_horizon, swot_dim + climate_dim)
                             History window (seq_len steps) + forecast_horizon forecast steps.
                             SWOT features (indices 0–swot_dim-1) are zeroed on forecast steps.
            static_features: (num_nodes, static_dim) — time-invariant lake attributes
        Returns:
            forecast_horizon == 1: (num_nodes,)
            forecast_horizon >  1: (num_nodes, forecast_horizon)
        """
        h = self._encode(x, static_features)                                     # (N, T, hidden_dim)
        return self.forecast_head(h)                                             # (N,) or (N, H)


class LSTMBaselineMultiStepGauss(LSTMBaselineMultiStep):
    """
    Probabilistic LSTM-only baseline: same backbone as LSTMBaselineMultiStep,
    dual Gaussian output heads trained with multi-step CRPS.

    Architecture:
        InputEncoder → [LSTMBlock × st_blocks] → mean_head
                                                → log_std_head

    forward() returns (mean, log_std), each (num_nodes, forecast_horizon),
    matching the interface expected by ObservedGaussianCRPSLossMultiStep and
    the existing train_lstm_nd._run_epoch_lstm_nd infrastructure (tuple branch).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hidden_dim       = kwargs.get("hidden_dim", 32)
        dropout          = kwargs.get("dropout", 0.5)
        forecast_horizon = kwargs.get("forecast_horizon", 1)

        del self.forecast_head
        self.mean_head    = ForecastHead(hidden_dim, dropout, forecast_horizon)
        self.log_std_head = ForecastHead(hidden_dim, dropout, forecast_horizon)

    def forward(
        self,
        x: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:               (num_nodes, seq_len + forecast_horizon, swot_dim + climate_dim)
            static_features: (num_nodes, static_dim)
        Returns:
            mean:    (num_nodes, forecast_horizon) — predicted mean WSE (normalised)
            log_std: (num_nodes, forecast_horizon) — predicted log standard deviation
        """
        h = self._encode(x, static_features)                                     # (N, T, hidden_dim)
        return self.mean_head(h), self.log_std_head(h)