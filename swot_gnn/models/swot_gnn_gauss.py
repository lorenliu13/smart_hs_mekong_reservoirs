"""
SWOTGNNGauss: Probabilistic extension of SWOT-GNN using a Gaussian output head.

Inherits the full backbone (InputEncoder -> STBlock x N) from SWOTGNN and
replaces the single ForecastHead with two parallel heads that predict the
mean and log-standard-deviation of a Gaussian distribution over the
next-day normalised WSE per lake.

Training loss: ObservedGaussianNLLLoss (see training/train.py).
Inference:     Returns (mean, log_std) tuple; denormalise mean with lake_mean/std,
               denormalise std by scaling only (no shift).
"""
import torch
import torch.nn as nn
from typing import Optional

from .swot_gnn import SWOTGNN, ForecastHead


class SWOTGNNGauss(SWOTGNN):
    """
    Probabilistic SWOT-GNN: same backbone as SWOTGNN, dual Gaussian output head.

    Replaces the deterministic ForecastHead with two parallel heads:
        mean_head    → μ  (predicted mean WSE, normalised)
        log_std_head → log σ  (predicted log-std; clamp to [-6, 6] at loss/inference time)

    forward() returns a tuple (mean, log_std), each of shape (num_nodes,).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Read head parameters from kwargs (already set as attributes by SWOTGNN.__init__)
        hidden_dim       = kwargs.get("hidden_dim", 32)
        dropout          = kwargs.get("dropout", 0.1)
        forecast_horizon = kwargs.get("forecast_horizon", 1)

        # Replace the deterministic head with two parallel Gaussian heads
        del self.forecast_head
        self.mean_head    = ForecastHead(hidden_dim, dropout, forecast_horizon)
        self.log_std_head = ForecastHead(hidden_dim, dropout, forecast_horizon)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        static_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (num_nodes, seq_len + forecast_horizon, swot_dim + climate_dim)
            edge_index: (2, num_edges)
            static_features: (num_nodes, static_dim)
            batch: Optional PyG batch vector for multi-graph batching
        Returns:
            mean:    (num_nodes,) — predicted mean WSE (normalised)
            log_std: (num_nodes,) — predicted log standard deviation (normalised)
        """
        # Shared backbone — identical to SWOTGNN.forward; heads handle last-step extraction internally
        h = self.encoder(x)
        static_emb = self.static_encoder(static_features) if static_features is not None else None
        for st in self.st_blocks:
            h = st(h, edge_index, batch=batch, static=static_emb)   # (N, T, hidden_dim)

        return self.mean_head(h), self.log_std_head(h)
