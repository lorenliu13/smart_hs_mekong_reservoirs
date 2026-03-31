"""
GraphGPS layer: local GAT message passing + global attention (FAVOR+ style).
Per Osanlou et al. NeurIPS 2024 / Rampasek et al. NeurIPS 2022.

Combines:
- Local: GAT aggregates info from graph neighbors (upstream/downstream reaches)
- Global: Attention lets each node attend to all others (captures long-range dependencies)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import add_self_loops
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GraphGPSLayer(nn.Module):
    """
    Single GraphGPS layer: local GAT + global attention.
    Output = LayerNorm(local_out + global_out) with residual from input.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.5,
        use_linear_attn: bool = True,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for GraphGPSLayer")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.use_linear_attn = use_linear_attn

        # Local: Graph Attention Network - aggregates from edge neighbors
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
        )
        # Project input for residual when in_dim != out_dim
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm_local = nn.LayerNorm(out_dim)
        self.norm_global = nn.LayerNorm(out_dim)
        # Global attention: Q, K, V projections for self-attention over all nodes
        self.proj_q = nn.Linear(out_dim, out_dim)
        self.proj_k = nn.Linear(out_dim, out_dim)
        self.proj_v = nn.Linear(out_dim, out_dim)
        self.proj_out = nn.Linear(out_dim, out_dim)

    def _global_attention(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Global attention over all nodes. Three modes:
        - Batched (batch provided): vectorized FAVOR+ per graph-group via bmm — O(G·n·d²)
          Nodes are assumed contiguous within each graph (guaranteed by our batching code).
        - Large n (>256, no batch): FAVOR+ linear attention — O(n·d²)
        - Small n (no batch): standard softmax attention — O(n²·d)
        """
        n = x.size(0)
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        if batch is not None:
            # Vectorized FAVOR+ across all graph groups — no Python loop.
            # Nodes within each group are contiguous, so reshape is valid.
            n_graphs = int(batch.max().item()) + 1
            n_per = n // n_graphs          # nodes per group (uniform by construction)
            scale = self.out_dim ** -0.25
            q_f = (F.elu(q) + 1) * scale  # (N, d)
            k_f = (F.elu(k) + 1) * scale  # (N, d)
            # Reshape: (N, d) -> (G, n_per, d)
            q3 = q_f.view(n_graphs, n_per, -1)
            k3 = k_f.view(n_graphs, n_per, -1)
            v3 = v.view(n_graphs, n_per, -1)
            kv  = torch.bmm(k3.transpose(1, 2), v3)          # (G, d, d)
            k_sum = k3.sum(dim=1)                              # (G, d)
            denom = torch.bmm(q3, k_sum.unsqueeze(-1)) + 1e-8 # (G, n_per, 1)
            out3  = torch.bmm(q3, kv) / denom                 # (G, n_per, d)
            out   = out3.view(n, -1)                           # (N, d)
        elif self.use_linear_attn and n > 256:
            # FAVOR+ style: kernel feature map phi(x)=elu(x)+1 makes attention linearizable
            # out_i = sum_j phi(q_i)^T phi(k_j) v_j / sum_j phi(q_i)^T phi(k_j)
            # Implemented via: q @ (k^T @ v) / (q @ sum(k))
            scale = self.out_dim ** -0.25
            q = F.elu(q) + 1
            k = F.elu(k) + 1
            q = q * scale
            k = k * scale
            kv  = torch.einsum("nd,nv->dv", k, v)  # (out_dim, out_dim)
            out = torch.einsum("nd,dv->nv", q, kv) / (torch.einsum("nd,d->n", q, k.sum(0)) + 1e-8).unsqueeze(-1)
        else:
            # Standard attention: softmax(QK^T / sqrt(d)) @ V
            scale = self.out_dim ** -0.5
            attn = torch.mm(q, k.t()) * scale
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out  = torch.mm(attn, v)
        return self.proj_out(out)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, in_dim)
            edge_index: (2, num_edges)
            batch: Optional batch vector for batched graphs
        Returns:
            (num_nodes, out_dim)
        """
        # Step 1: Local - GAT aggregates from graph neighbors, residual from input
        h_local = self.gat(x, edge_index)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)
        h_local = self.norm_local(h_local + self.res_proj(x))

        # Step 2: Global - self-attention over all nodes, residual from local
        h_global = self._global_attention(h_local, batch)
        h_global = F.dropout(h_global, p=self.dropout, training=self.training)
        out = self.norm_global(h_local + h_global)
        return out
