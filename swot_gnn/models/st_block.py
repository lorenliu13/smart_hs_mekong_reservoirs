"""
ST-Block: Spatio-Temporal block for SWOT-GNN.
Processes (1) Temporal: bi-LSTM over time per node, (2) Spatial: GraphGPS over graph per timestep.
Order: temporal first, then spatial at each timestep.

Static features are concatenated as input features to both the bi-LSTM (every timestep)
and the GraphGPS (every timestep, projected back to hidden_dim before the GPS stack).
"""
import torch
import torch.nn as nn
from typing import Optional

from .graph_gps_layer import GraphGPSLayer


class STBlock(nn.Module):
    """
    ST-Block: (1) Temporal processing via bi-LSTM, (2) Spatial processing via GraphGPS.
    Each node gets a temporal sequence -> LSTM encodes it -> GraphGPS propagates across graph
    at each timestep. Output is full (num_nodes, seq_len, hidden_dim) for next block.

    Static features (time-invariant node attributes) are injected as input features:
    - LSTM: static embedding concatenated to x at every timestep
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        gps_layers: int = 3,
        gps_heads: int = 4,
        dropout: float = 0.5,
        static_embed_dim: int = 0,
    ):
        super().__init__()
        self.static_embed_dim = static_embed_dim
        # Bi-LSTM input: dynamic features + static embedding (concatenated per timestep)
        lstm_input_size = in_dim + static_embed_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
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

        # Step 1: Temporal - concatenate static to every timestep, then run bi-LSTM
        # The bi-LSTM treats each node independently as a batch item (batch_first=True,
        # so num_nodes acts as the batch dimension). It processes (seq_len, lstm_input_size)
        # and outputs (seq_len, hidden_dim).
        if static is not None and self.static_embed_dim > 0:
            static_exp = static.unsqueeze(1).expand(-1, seq_len, -1)  # (num_nodes, seq_len, static_embed_dim)
            x = torch.cat([x, static_exp], dim=-1)                    # (num_nodes, seq_len, in_dim + static_embed_dim)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        h = self.norm(lstm_out)

        # Step 2: Spatial - GraphGPS across all timesteps at once (vectorized).
        # We treat each (sample, timestep) pair as an independent graph so GATConv and
        # global attention see no cross-timestep or cross-sample edges/nodes.
        #
        # Layout after batching:
        #   num_nodes  = n_graphs_in * n_per_graph    (n_graphs_in ≥ 1 when sample-batched)
        #   n_super    = n_graphs_in * seq_len         (one "super-graph" per sample × timestep)
        #
        # h is reshaped (N, T, H) → (G, n, T, H) → (G, T, n, H) → (G*T*n, H) so that
        # nodes in the same super-graph are contiguous, matching edge_index_gps offsets.

        # Nodes per individual graph (same for all samples by construction)
        n_graphs_in = 1 if batch is None else int(batch.max().item()) + 1
        n_per_graph  = num_nodes // n_graphs_in

        # Extract the base edge_index for a single graph (graph-0 edges, indices < n_per_graph)
        mask0       = edge_index[0] < n_per_graph
        ei_base     = edge_index[:, mask0]              # (2, e)

        # Build edge_index for all n_super = n_graphs_in * seq_len super-graphs
        n_super  = n_graphs_in * seq_len
        offsets  = torch.arange(n_super, device=edge_index.device) * n_per_graph  # (n_super,)
        ei_exp   = ei_base.unsqueeze(0) + offsets.view(-1, 1, 1)  # (n_super, 2, e)
        edge_index_gps = ei_exp.permute(1, 0, 2).reshape(2, -1)   # (2, n_super * e)

        # Rearrange h: (N, T, H) → (G, n, T, H) → (G, T, n, H) → (n_super * n_per_graph, H)
        H = h.size(-1)
        h_4d   = h.view(n_graphs_in, n_per_graph, seq_len, H)
        h_4d   = h_4d.permute(0, 2, 1, 3).contiguous()  # (G, T, n, H)
        h_flat = h_4d.view(n_super * n_per_graph, H)     # (n_super * n_per_graph, H)

        # Batch vector for global attention: each super-graph is an independent group
        t_batch = torch.arange(n_super, device=h.device).repeat_interleave(n_per_graph)

        for gps in self.gps_stack:
            h_flat = gps(h_flat, edge_index_gps, t_batch) + h_flat  # Residual connection

        # Reshape back: (n_super * n_per_graph, H) → (G, T, n, H) → (G, n, T, H) → (N, T, H)
        h_out = h_flat.view(n_graphs_in, seq_len, n_per_graph, H)
        h_out = h_out.permute(0, 2, 1, 3).contiguous()
        return h_out.view(num_nodes, seq_len, H)  # (num_nodes, seq_len, hidden_dim)
