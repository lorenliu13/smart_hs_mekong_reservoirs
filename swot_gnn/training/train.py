"""
Training primitives for SWOT-GNN.

Exports:
    ObservedMSELoss          — masked MSE for 1-step forecasting
    ObservedMSELossMultiStep — masked MSE for multi-step forecasting
    ObservedGaussianNLLLoss  — masked Gaussian NLL for probabilistic forecasting
    ObservedGaussianCRPSLoss — masked closed-form Gaussian CRPS for probabilistic forecasting
    _run_epoch               — one forward pass over a DataLoader (train or eval)
"""
import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader


class ObservedMSELossMultiStep(nn.Module):
    """MSE loss for multi-step forecasting, computed only at observed nodes.

    Operates on (n_nodes, horizon) tensors (e.g. lake 10-day forecasting).
    Loss is averaged over all (node, day) pairs where obs_mask=1.
    Equivalent to ObservedMSELoss when horizon=1 with squeezed tensors.
    """

    def forward(
        self,
        pred: torch.Tensor,    # (n_nodes, horizon)
        target: torch.Tensor,  # (n_nodes, horizon)
        mask: torch.Tensor,    # (n_nodes, horizon) — 1 if observed, 0 if not
    ) -> torch.Tensor:
        loss = (pred - target) ** 2          # element-wise squared error
        return (loss * mask).sum() / (mask.sum() + 1e-8)


class ObservedMSELoss(nn.Module):
    """MSE loss computed only at observed nodes (obs_mask=1).

    Unobserved nodes (SWOT passes where no measurement exists) are excluded
    from the loss so the model is not penalized for predicting at masked locations.
    The small epsilon (1e-8) prevents division by zero when all nodes are masked.
    """

    def forward(
        self,
        pred: torch.Tensor,   # (n_seg,) — predicted WSE for forecast day
        target: torch.Tensor, # (n_seg,) — ground-truth WSE
        mask: torch.Tensor,   # (n_seg,) — 1 if observed, 0 if unobserved
    ) -> torch.Tensor:
        loss = (pred - target) ** 2          # element-wise squared error
        masked_loss = loss * mask            # zero-out unobserved segments
        return masked_loss.sum() / (mask.sum() + 1e-8)  # mean over observed nodes


class ObservedGaussianNLLLoss(nn.Module):
    """Gaussian NLL loss, computed only at observed nodes (obs_mask=1).

    Inputs:
        mean:    (n_seg,) — predicted mean WSE (normalised)
        log_std: (n_seg,) — predicted log standard deviation (normalised)
        target:  (n_seg,) — ground-truth WSE (normalised)
        mask:    (n_seg,) — 1 if observed, 0 if not

    Loss per element: 0.5 * (2·log_std + ((target - mean) / exp(log_std))²)
    which equals -log N(target | mean, exp(log_std)²) up to a constant.
    log_std is clamped to [-6, 6] to prevent numerical explosion.
    """

    def forward(
        self,
        mean:    torch.Tensor,  # (n_seg,)
        log_std: torch.Tensor,  # (n_seg,)
        target:  torch.Tensor,  # (n_seg,)
        mask:    torch.Tensor,  # (n_seg,)
    ) -> torch.Tensor:
        log_std = log_std.clamp(-6, 6)
        nll = 0.5 * (2 * log_std + ((target - mean) / log_std.exp()) ** 2)
        return (nll * mask).sum() / (mask.sum() + 1e-8)


class ObservedGaussianCRPSLoss(nn.Module):
    """Closed-form Gaussian CRPS loss, computed only at observed nodes (obs_mask=1).

    For a Gaussian predictive distribution N(μ, σ²), CRPS has the closed form:
        CRPS = σ · [z·(2Φ(z) − 1) + 2φ(z) − 1/√π]
    where z = (target − mean) / σ, Φ = standard normal CDF, φ = standard normal PDF.

    Compared with Gaussian NLL, CRPS is a proper scoring rule that is less sensitive
    to extreme observations and penalises over-dispersed forecasts more gently.

    Inputs:
        mean:    (n_seg,) — predicted mean WSE (normalised)
        log_std: (n_seg,) — predicted log standard deviation (normalised)
        target:  (n_seg,) — ground-truth WSE (normalised)
        mask:    (n_seg,) — 1 if observed, 0 if not

    log_std is clamped to [-6, 6] to prevent numerical explosion.
    """

    _STANDARD_NORMAL = Normal(0.0, 1.0)
    _INV_SQRT_PI = 1.0 / math.sqrt(math.pi)

    def forward(
        self,
        mean:    torch.Tensor,  # (n_seg,)
        log_std: torch.Tensor,  # (n_seg,)
        target:  torch.Tensor,  # (n_seg,)
        mask:    torch.Tensor,  # (n_seg,)
    ) -> torch.Tensor:
        log_std = log_std.clamp(-6, 6)
        std = log_std.exp()
        z   = (target - mean) / std

        # Re-create standard normal on the correct device
        standard_normal = Normal(
            torch.zeros(1, device=mean.device),
            torch.ones(1,  device=mean.device),
        )
        phi = standard_normal.log_prob(z).exp()   # φ(z)
        Phi = standard_normal.cdf(z)              # Φ(z)

        crps = std * (z * (2 * Phi - 1) + 2 * phi - self._INV_SQRT_PI)
        return (crps * mask).sum() / (mask.sum() + 1e-8)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    grad_clip: float = 1.0,
    spatial_mask: torch.Tensor = None,
) -> float:
    """Run one forward pass over `loader` using true batching.

    All samples in each DataLoader batch are merged into a single forward pass:
    node tensors are concatenated, edge_index is tiled with per-sample offsets,
    and a batch vector isolates attention within each sample.

    When `optimizer` is provided (training mode) the function back-propagates
    and steps the optimiser.  Without it the function runs in eval / no-grad mode.

    Args:
        spatial_mask: Optional (n_lakes,) float tensor with 1 for active lakes and 0
            for held-out lakes. When provided (spatial cross-validation), the loss is
            computed only where both SWOT observation mask AND spatial_mask are 1.
            All lake nodes still participate in message passing regardless of this mask.

    Returns the mean per-batch loss over the full loader.
    """
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss, n_batches = 0.0, 0
    with ctx:
        for data_lists, static_feats, labels, masks in loader:
            B = len(data_lists)
            if B == 0:
                continue

            # Stack node features across time for each sample, then concatenate
            x_list  = [torch.stack([d.x for d in data_lists[b]], dim=1) for b in range(B)]
            n_lakes = x_list[0].size(0)
            x_batch = torch.cat(x_list, dim=0).to(device)             # (B*N, T, feat)

            # All samples share the same graph topology — tile edge_index with node offsets
            # so each sample's subgraph uses non-overlapping node indices
            edge_index = data_lists[0][0].edge_index.to(device)       # (2, n_edges)
            offsets    = torch.arange(B, device=device) * n_lakes
            edge_index_tiled = (
                edge_index.unsqueeze(0) + offsets.view(B, 1, 1)
            ).permute(1, 0, 2).reshape(2, -1)                         # (2, B*n_edges)

            # batch vector tells _global_attention which nodes belong to which sample
            batch_vec    = torch.arange(B, device=device).repeat_interleave(n_lakes)
            static_batch = static_feats.reshape(B * n_lakes, -1).to(device)
            lab          = labels.reshape(B * n_lakes).to(device)   # (B*N,) — flat 1-D
            msk          = masks.reshape(B * n_lakes).to(device)    # (B*N,) — flat 1-D

            # Apply spatial mask: tile (n_lakes,) → (B*n_lakes,) and AND with obs mask.
            # Message passing uses all nodes; only the loss is gated by this mask.
            if spatial_mask is not None:
                sm  = spatial_mask.to(device).float()                  # (n_lakes,)
                sm  = sm.unsqueeze(0).expand(B, -1).reshape(-1)        # (B*n_lakes,)
                msk = msk * sm

            if is_train:
                optimizer.zero_grad()

            pred = model(x_batch, edge_index_tiled,
                         static_features=static_batch, batch=batch_vec)
            # Gaussian models return (mean, log_std); point models return a tensor
            if isinstance(pred, tuple):
                loss = criterion(*pred, lab, msk)  # mean, log_std, target, mask
            else:
                loss = criterion(pred, lab, msk)

            if is_train:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)
