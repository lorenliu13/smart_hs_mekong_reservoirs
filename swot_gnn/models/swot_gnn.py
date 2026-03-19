"""
SWOT-GNN: Temporal Graph Neural Network for river discharge simulation.
Replication of Osanlou et al. NeurIPS 2024.

Architecture: InputEncoder -> STBlock x2 -> ForecastHead
- InputEncoder:   project raw features to embed_dim
- STBlock:        temporal (LSTM) + spatial (GraphGPS) processing
- ForecastHead:   map the final time-step hidden state to a single scalar
                  (1-day-ahead WSE) per node
"""
import torch
import torch.nn as nn
from typing import Optional

from .st_block import STBlock


class InputEncoder(nn.Module):
    """
    Encodes dynamic node features to embed_dim using separate MLPs for SWOT and climate features.

    Expected feature order (matches DYNAMIC_FEATURE_VARS in feature_assembler.py):
        SWOT    [0 : swot_dim]:    obs_mask, latest_wse, days_since_last_obs, time_doy_sin, time_doy_cos
        Climate [swot_dim :]:      LWd, P, Pres, RelHum, SWd, Temp, Wind
    Each group is projected to embed_dim // 2 then concatenated → embed_dim.
    """

    def __init__(
        self,
        swot_dim: int = 5,
        climate_dim: int = 7,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.swot_dim = swot_dim
        chunk = embed_dim // 2
        self.mlp_swot = nn.Sequential(
            nn.LayerNorm(swot_dim),
            nn.Linear(swot_dim, chunk),
            nn.ReLU(),
        )
        self.mlp_climate = nn.Sequential(
            nn.LayerNorm(climate_dim),
            nn.Linear(climate_dim, chunk),
            nn.ReLU(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, seq_len, swot_dim + climate_dim)
        Returns:
            (num_nodes, seq_len, embed_dim)
        """
        swot_out = self.mlp_swot(x[:, :, :self.swot_dim])
        climate_out = self.mlp_climate(x[:, :, self.swot_dim:])
        return self.out_norm(torch.cat([swot_out, climate_out], dim=-1))


class StaticEncoder(nn.Module):
    """
    Encodes time-invariant static attributes into a dense embedding vector.
    The embedding is used as auxiliary input features for both the bi-LSTM and
    GraphGPS in every STBlock.
    """

    def __init__(self, static_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(static_dim),
            nn.Linear(static_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (num_nodes, static_dim)
        Returns:
            (num_nodes, embed_dim)
        """
        return self.mlp(s)


class ForecastHead(nn.Module):
    """
    Maps the hidden representation at the last time step to WSE forecast(s) per node.

    Two fully-connected layers with a bottleneck:
        hidden_dim  →  hidden_dim // 2  →  forecast_horizon

    When forecast_horizon=1 (default), output is squeezed to (num_nodes,) for
    backward compatibility with 1-day-ahead training code.
    When forecast_horizon>1 (e.g. 10 for lake multi-step), output is (num_nodes, horizon).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, forecast_horizon: int = 1):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        bottleneck = max(hidden_dim // 2, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, forecast_horizon),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (num_nodes, seq_len + forecast_horizon, hidden_dim) — full STBlock output
        Returns:
            forecast_horizon == 1: (num_nodes,)              — backward compatible
            forecast_horizon >  1: (num_nodes, horizon)      — multi-step direct
        """
        h_last = h[:, -1, :]           # (num_nodes, hidden_dim)
        out = self.net(h_last)         # (num_nodes, forecast_horizon)
        if self.forecast_horizon == 1:
            return out.squeeze(-1)     # (num_nodes,) — backward compatible
        return out                     # (num_nodes, horizon)


class SWOTGNN(nn.Module):
    """
    SWOT-GNN: InputEncoder -> STBlock x N -> ForecastHead.

    Default (forecast_horizon=1): returns a single scalar per node (1-day-ahead WSE).
    With forecast_horizon=10: returns (num_nodes, 10) for lake multi-step forecasting.
    """

    def __init__(
        self,
        swot_dim: int = 5,
        climate_dim: int = 7,
        embed_dim: int = 64,
        hidden_dim: int = 32,
        st_blocks: int = 2,
        gps_layers_per_block: int = 3,
        gps_heads: int = 4,
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
        # Encode static attributes to a flat embedding injected into every STBlock
        self.static_encoder = StaticEncoder(
            static_dim=static_dim,
            embed_dim=static_embed_dim,
        )
        self.static_embed_dim = static_embed_dim
        # Stack of ST-blocks: first takes embed_dim, rest take hidden_dim
        self.st_blocks = nn.ModuleList()
        for i in range(st_blocks):
            in_d = embed_dim if i == 0 else hidden_dim
            self.st_blocks.append(
                STBlock(
                    in_dim=in_d,
                    hidden_dim=hidden_dim,
                    gps_layers=gps_layers_per_block,
                    gps_heads=gps_heads,
                    dropout=dropout,
                    static_embed_dim=static_embed_dim,
                )
            )
        # Dedicated head: maps the forecast-step hidden state → WSE per node (scalar or multi-step)
        self.forecast_head = ForecastHead(
            hidden_dim=hidden_dim, dropout=dropout, forecast_horizon=forecast_horizon
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, seq_len + 1, swot_dim + climate_dim)
               History window (seq_len steps) + 1 forecast step.
               WSE features are zeroed on the forecast step to prevent leakage.
            edge_index: (2, num_edges)
            static_features: (num_nodes, static_dim) - time-invariant attributes
            batch: Optional PyG batch vector for multi-graph batching
        Returns:
            forecast_horizon=1: (num_nodes,)        — 1-day-ahead WSE (backward compatible)
            forecast_horizon>1: (num_nodes, horizon) — multi-step direct WSE forecast
        """
        # 1. Encode raw features to embed_dim for every node and time step
        h = self.encoder(x)                                          # (N, T, embed_dim)

        # 2. Encode static attributes once; injected into each STBlock
        static_emb = self.static_encoder(static_features) if static_features is not None else None

        # 3. Pass through the stack of ST-blocks (LSTM + GraphGPS)
        for st in self.st_blocks:
            h = st(h, edge_index, batch=batch, static=static_emb)   # (N, T, hidden_dim)

        # 4. Project the final time step to a single WSE scalar per node
        return self.forecast_head(h)                                 # (N,)
