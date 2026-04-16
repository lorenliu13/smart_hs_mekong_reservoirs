"""
GraphGPS layer: local GAT message passing + global attention (FAVOR+ style).
Per Osanlou et al. NeurIPS 2024 / Rampasek et al. NeurIPS 2022.

Combines:
- Local: GAT aggregates info from graph neighbors (upstream/downstream reaches)
- Global: Multi-head attention lets each node attend to all others (captures long-range
  dependencies). Each head operates in a subspace of dimension out_dim // heads,
  allowing different heads to specialize in different relational patterns
  (e.g. basin proximity, WSE similarity, climate zone).
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
        dropout: float = 0.1,
        use_linear_attn: bool = True,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for GraphGPSLayer")
        assert out_dim % heads == 0, f"out_dim ({out_dim}) must be divisible by heads ({heads})"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
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
        Multi-head global attention over all nodes.

        Q, K, V are projected to out_dim then split into `heads` subspaces of size
        head_dim = out_dim // heads. Attention is computed independently per head,
        outputs are concatenated, and a final linear projection mixes them back to out_dim.

        Three execution modes (same as before, now applied per head):
        - Batched (batch vector provided): each sample attends only within itself.
        - Large n (>256), unbatched: FAVOR+ linear attention, O(n) per head.
        - Small n, unbatched: standard softmax attention.
        """
        n = x.size(0)
        H, d_h = self.heads, self.head_dim

        # Project then split into heads: (N, out_dim) -> (N, H, d_h)
        q = self.proj_q(x).view(n, H, d_h)
        k = self.proj_k(x).view(n, H, d_h)
        v = self.proj_v(x).view(n, H, d_h)

        if batch is not None:
            # Batch-aware path: each sample attends only within itself.
            # Reshape to (B, N, H, d_h) so nodes from different samples never interact.
            B = int(batch.max().item()) + 1
            N = n // B
            q = q.view(B, N, H, d_h)
            k = k.view(B, N, H, d_h)
            v = v.view(B, N, H, d_h)
            if self.use_linear_attn and N > 256:
                # FAVOR+ per head per sample
                # kv: (B, H, d_h, d_h)  norm: (B, N, H)  out: (B, N, H, d_h)
                scale = d_h ** -0.25
                q = (F.elu(q) + 1) * scale
                k = (F.elu(k) + 1) * scale
                kv   = torch.einsum("bnhd,bnhv->bhdv", k, v)
                norm = torch.einsum("bnhd,bhd->bnh",   q, k.sum(dim=1)) + 1e-8
                out  = torch.einsum("bnhd,bhdv->bnhv", q, kv) / norm.unsqueeze(-1)
            else:
                # Batched softmax multi-head: (B, H, N, N) attention matrices
                # Permute to (B, H, N, d_h) for batched matmul
                scale = d_h ** -0.5
                q = q.permute(0, 2, 1, 3)                                           # (B, H, N, d_h)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale                 # (B, H, N, N)
                attn = F.softmax(attn, dim=-1)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                out  = torch.matmul(attn, v).permute(0, 2, 1, 3)                   # (B, N, H, d_h)
            out = out.reshape(n, self.out_dim)

        elif self.use_linear_attn and n > 256:
            # FAVOR+ multi-head unbatched
            # kv: (H, d_h, d_h)  norm: (N, H)  out: (N, H, d_h)
            scale = d_h ** -0.25
            q = (F.elu(q) + 1) * scale
            k = (F.elu(k) + 1) * scale
            kv   = torch.einsum("nhd,nhv->hdv", k, v)
            norm = torch.einsum("nhd,hd->nh",   q, k.sum(dim=0)) + 1e-8
            out  = torch.einsum("nhd,hdv->nhv", q, kv) / norm.unsqueeze(-1)
            out  = out.reshape(n, self.out_dim)

        else:
            # Standard softmax multi-head unbatched
            # Permute to (H, N, d_h) for batched matmul across heads
            scale = d_h ** -0.5
            q = q.permute(1, 0, 2)                                                  # (H, N, d_h)
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale                     # (H, N, N)
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out  = torch.matmul(attn, v).permute(1, 0, 2).reshape(n, self.out_dim)  # (N, out_dim)

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
