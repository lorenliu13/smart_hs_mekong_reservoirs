"""
Training primitive for multi-day (N-day) SWOT-GNN forecasting.

Exports:
    _run_epoch_nd — one forward pass over a DataLoader for multi-step forecasting.

Key difference from training/train.py:
    Labels and masks have shape (n_nodes, forecast_horizon) instead of (n_nodes,).
    The epoch function reshapes them to (B*N, H) and applies ObservedMSELossMultiStep,
    which averages the loss over all (lake, lead_day) pairs where obs_mask=1.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _run_epoch_nd(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    grad_clip: float = 1.0,
    spatial_mask: torch.Tensor = None,
) -> float:
    """Run one forward pass over `loader` for multi-day forecasting.

    Mirrors training/train.py::_run_epoch but handles multi-step labels and masks
    of shape (B, n_lakes, forecast_horizon) from the DataLoader.

    All samples in each DataLoader batch are merged into a single forward pass:
    node tensors are concatenated, edge_index is tiled with per-sample offsets,
    and a batch vector isolates attention within each sample.

    When `optimizer` is provided (training mode) the function back-propagates
    and steps the optimiser.  Without it the function runs in eval / no-grad mode.

    Args:
        model:        SWOT-GNN model with forecast_horizon > 1 configured.
        loader:       DataLoader yielding (data_lists, static_feats, labels, masks)
                      where labels and masks are (B, n_lakes, forecast_horizon).
        criterion:    Loss function for multi-step forecasting (e.g.
                      ObservedMSELossMultiStep, ObservedGaussianCRPSLossMultiStep).
                      Point models: expects (pred, target, mask) each of shape
                      (B*n_lakes, forecast_horizon). Gaussian models: expects
                      (mean, log_std, target, mask) each of shape (B*n_lakes, forecast_horizon).
        device:       Torch device (cpu or cuda).
        optimizer:    If provided, performs a gradient update step (train mode).
        grad_clip:    Max gradient norm for clipping (0 disables clipping).
        spatial_mask: Optional (n_lakes,) float tensor with 1 for active lakes and 0
                      for held-out lakes. When provided, is broadcast to
                      (B*n_lakes, forecast_horizon) and AND-ed with the obs mask so
                      the loss is only computed for active-lake, observed-day cells.
                      All lake nodes still participate in message passing.

    Returns:
        Mean per-batch loss over the full loader.
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

            # Stack node features across time steps for each sample
            x_list  = [torch.stack([d.x for d in data_lists[b]], dim=1) for b in range(B)]
            n_lakes = x_list[0].size(0)
            x_batch = torch.cat(x_list, dim=0).to(device)             # (B*N, T, feat)

            # Tile edge_index with per-sample node offsets so subgraphs are disjoint
            edge_index = data_lists[0][0].edge_index.to(device)       # (2, n_edges)
            offsets    = torch.arange(B, device=device) * n_lakes
            edge_index_tiled = (
                edge_index.unsqueeze(0) + offsets.view(B, 1, 1)
            ).permute(1, 0, 2).reshape(2, -1)                         # (2, B*n_edges)

            batch_vec    = torch.arange(B, device=device).repeat_interleave(n_lakes)
            static_batch = static_feats.reshape(B * n_lakes, -1).to(device)

            # Multi-step: labels and masks are (B, n_lakes, H) → (B*N, H)
            H   = labels.shape[-1]
            lab = labels.reshape(B * n_lakes, H).to(device)           # (B*N, H)
            msk = masks.reshape(B * n_lakes, H).to(device)            # (B*N, H)

            # Apply spatial mask: broadcast (n_lakes,) → (B*n_lakes, H)
            # and AND with the per-(lake, day) obs mask.
            if spatial_mask is not None:
                sm  = spatial_mask.to(device).float()                  # (n_lakes,)
                sm  = sm.unsqueeze(0).expand(B, -1).reshape(-1)        # (B*n_lakes,)
                sm  = sm.unsqueeze(-1).expand(-1, H)                   # (B*n_lakes, H)
                msk = msk * sm

            if is_train:
                optimizer.zero_grad()

            pred = model(x_batch, edge_index_tiled,
                         static_features=static_batch, batch=batch_vec)
            # Point models return (B*N, H); Gaussian models return two (B*N, H) tensors
            if isinstance(pred, tuple):
                loss = criterion(*pred, lab, msk)
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
