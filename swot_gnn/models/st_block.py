"""
ST-Block: Spatio-Temporal block for SWOT-GNN.
Processes (1) Temporal: LSTM over time per node, (2) Spatial: GraphGPS over graph per timestep.
Order: temporal first, then spatial at each timestep.

Static features are concatenated as input features to the LSTM (every timestep).
The GraphGPS stack receives only the LSTM hidden states.
"""
import torch
import torch.nn as nn
from typing import Optional

from .graph_gps_layer import GraphGPSLayer


class STBlock(nn.Module):
    """
    ST-Block: (1) Temporal processing via LSTM, (2) Spatial processing via GraphGPS.
    Each node gets a temporal sequence -> LSTM encodes it -> GraphGPS propagates across graph
    at each timestep. Output is full (num_nodes, seq_len, hidden_dim) for next block.

    Static features (time-invariant node attributes) are concatenated to x before the LSTM only.
    The GraphGPS stack receives only the LSTM hidden states.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        gps_layers: int = 3,
        gps_heads: int = 4,
        dropout: float = 0.1,
        static_embed_dim: int = 0,
    ):
        super().__init__()
        self.static_embed_dim = static_embed_dim
        # LSTM input: dynamic features + static embedding (concatenated per timestep)
        lstm_input_size = in_dim + static_embed_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        # GraphGPS stack: applied per timestep to mix info across the river network
        self.gps_stack = nn.ModuleList([
            GraphGPSLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                heads=gps_heads,
                dropout=dropout,
            )
            for _ in range(gps_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, seq_len, in_dim) - node features over time
            edge_index: (2, num_edges)
            batch: Optional batch vector
            static: (num_nodes, static_embed_dim) - static node embedding, concatenated as
                input features to the LSTM at every timestep.
        Returns:
            (num_nodes, seq_len, hidden_dim) - full temporal sequence for next ST-block
        """
        num_nodes, seq_len, _ = x.size()

        # Step 1: Temporal - concatenate static to every timestep, then run LSTM
        # The LSTM treats each node independently as a batch item (batch_first=True,
        # so num_nodes acts as the batch dimension). It processes (seq_len, lstm_input_size)
        # and outputs (seq_len, hidden_dim).
        if static is not None and self.static_embed_dim > 0:
            static_exp = static.unsqueeze(1).expand(-1, seq_len, -1)  # (num_nodes, seq_len, static_embed_dim)
            x = torch.cat([x, static_exp], dim=-1)                    # (num_nodes, seq_len, in_dim + static_embed_dim)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        h = self.norm(lstm_out)

        # Step 2: Spatial - at each timestep t, run GraphGPS across graph.
        # GraphGPS mixes information across the river network at each snapshot in time,
        # letting upstream/downstream context flow between nodes.
        out_list = []
        for t in range(seq_len):
            ht = h[:, t, :]  # (num_nodes, hidden_dim)
            for gps in self.gps_stack:
                ht = gps(ht, edge_index, batch) + ht  # Residual connection
            out_list.append(ht)
        return torch.stack(out_list, dim=1)  # (num_nodes, seq_len, hidden_dim)
