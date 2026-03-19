"""
Baseline models for SWOT-GNN comparison.
- Drainage-area ratio (Archfield & Vogel)
- GPS-GNN: GNN only, no temporal LSTM
- LSTM: 4-layer bi-LSTM, no graph
- PersistenceBaseline: predict WSE(t+1) = WSE(t) (last observed value)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# PyTorch Geometric optional: needed for GATConv and GraphGPSLayer
try:
    from torch_geometric.nn import GATConv
    from .graph_gps_layer import GraphGPSLayer
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def drainage_area_ratio(
    ungauged_darea: np.ndarray,
    gauged_darea: np.ndarray,
    gauged_discharge: np.ndarray,
    gauge_to_ungauged: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Drainage-area ratio method: Q_ungauged = Q_gauged * (A_ungauged / A_gauged).

    Args:
        ungauged_darea: Drainage area at ungauged locations (num_ungauged,)
        gauged_darea: Drainage area at gauged locations (num_gauged,)
        gauged_discharge: Discharge at gauged locations (num_gauged,) or (num_gauged, num_dates)
        gauge_to_ungauged: Optional (num_ungauged,) index of nearest gauge for each ungauged

    Returns:
        Predicted discharge at ungauged locations
    """
    # Default: use gauge 0 for all ungauged sites
    if gauge_to_ungauged is None:
        gauge_to_ungauged = np.zeros(len(ungauged_darea), dtype=int)
    # Support both single timestep and time series
    if gauged_discharge.ndim == 1:
        gauged_discharge = gauged_discharge[:, None]
    out = np.zeros((len(ungauged_darea), gauged_discharge.shape[1]))
    # Q_ungauged = Q_gauged * (A_ungauged / A_gauged) for each ungauged site
    for i in range(len(ungauged_darea)):
        g = gauge_to_ungauged[i]
        ratio = ungauged_darea[i] / (gauged_darea[g] + 1e-10)
        out[i, :] = gauged_discharge[g, :] * ratio
    return out.squeeze()


class PersistenceBaseline:
    """
    Persistence (naive) baseline: predict WSE(t+1) = WSE(t).

    Uses the most recent observed WSE for each lake as the forecast.
    Serves as the simplest possible benchmark — any useful model should beat this.

    Usage:
        baseline = PersistenceBaseline()
        preds = baseline.predict(wse)   # wse: (num_lakes, seq_len) or (num_lakes,)
    """

    def predict(self, wse: np.ndarray) -> np.ndarray:
        """
        Args:
            wse: (num_lakes, seq_len) or (num_lakes,) — observed WSE time series.
                 The last value along the time axis is used as the prediction.

        Returns:
            (num_lakes,) — predicted WSE at t+1 for each lake.
        """
        wse = np.asarray(wse)
        if wse.ndim == 1:
            # Already a single value per lake
            return wse.copy()
        # Take last timestep: (num_lakes, seq_len) -> (num_lakes,)
        return wse[:, -1].copy()

    def evaluate(
        self,
        wse: np.ndarray,
        targets: np.ndarray,
    ) -> dict:
        """
        Compute RMSE and MAE of the persistence forecast against true targets.

        Args:
            wse:     (num_lakes, seq_len) or (num_lakes,) — input WSE.
            targets: (num_lakes,) — true WSE at t+1.

        Returns:
            dict with 'rmse' and 'mae'.
        """
        preds = self.predict(wse)
        targets = np.asarray(targets)
        diff = preds - targets
        return {
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "mae": float(np.mean(np.abs(diff))),
        }


class GPSGNN(nn.Module):
    """
    GNN-only baseline: 6 GraphGPS layers, no temporal LSTM.
    Applied to single-day graph (or mean over sequence).
    Captures spatial structure (river network) but ignores temporal dynamics.
    """

    def __init__(
        self,
        feat_dim: int = 15,
        hidden_dim: int = 32,
        gps_layers: int = 6,
        gps_heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required")
        # Project raw features to hidden dim
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        # Stack of GraphGPS layers: each does local GAT + global attention
        self.gps_stack = nn.ModuleList([
            GraphGPSLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                heads=gps_heads,
                dropout=dropout,
            )
            for _ in range(gps_layers)
        ])
        # Final projection to scalar output (e.g. discharge)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, feat_dim) or (num_nodes, seq_len, feat_dim) - use last timestep
        """
        # If temporal sequence given, use only last timestep (no temporal modeling)
        if x.dim() == 3:
            x = x[:, -1, :]
        h = self.encoder(x)
        # Residual connections: each GPS layer adds to previous hidden state
        for gps in self.gps_stack:
            h = gps(h, edge_index, batch) + h
        return self.readout(h).squeeze(-1)


class LSTMBaseline(nn.Module):
    """
    LSTM baseline: 4-layer bi-LSTM on reach features, no graph.
    Each node (reach) is processed independently; no message passing between neighbors.
    Captures temporal dynamics but ignores spatial (upstream-downstream) structure.
    """

    def __init__(
        self,
        feat_dim: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        # Bi-LSTM: hidden_dim//2 per direction, concat -> hidden_dim
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        # MLP to map LSTM output to scalar
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, seq_len, feat_dim)
        Returns:
            (num_nodes,) - predicted discharge for last timestep
        """
        out, _ = self.lstm(x)
        h = out[:, -1, :]  # Take last timestep hidden state
        return self.readout(h).squeeze(-1)
