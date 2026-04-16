"""
Training primitive for multi-day LSTM-only lake WSE forecasting.

Mirrors training/train_nd.py::_run_epoch_nd but removes all graph-specific
operations (edge_index tiling, batch vector construction).  The LSTM model
processes each lake independently, so no graph connectivity data is needed.

Exports:
    _run_epoch_lstm_nd
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _run_epoch_lstm_nd(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    grad_clip: float = 1.0,
    spatial_mask: torch.Tensor = None,
) -> float:
    """Run one forward pass over `loader` for multi-day LSTM forecasting.

    Mirrors _run_epoch_nd from training/train_nd.py but omits edge_index
    tiling and batch vector construction — the LSTM model takes only
    (x_batch, static_features) without graph arguments.

    When `optimizer` is provided (training mode) the function back-propagates
    and steps the optimiser.  Without it the function runs in eval / no-grad mode.

    Args:
        model:        LSTMBaselineMultiStep with forecast_horizon > 1 configured.
        loader:       DataLoader yielding (data_lists, static_feats, labels, masks)
                      where labels and masks are (B, n_lakes, forecast_horizon).
        criterion:    Loss function for multi-step forecasting (e.g.
                      ObservedMSELossMultiStep).  Point models: expects
                      (pred, target, mask) each of shape (B*n_lakes, forecast_horizon).
        device:       Torch device (cpu or cuda).
        optimizer:    If provided, performs a gradient update step (train mode).
        grad_clip:    Max gradient norm for clipping (0 disables clipping).
        spatial_mask: Optional (n_lakes,) float tensor with 1 for active lakes and 0
                      for held-out lakes.  When provided, is broadcast to
                      (B*n_lakes, forecast_horizon) and AND-ed with the obs mask so
                      the loss is only computed for active-lake, observed-day cells.

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

            static_batch = static_feats.reshape(B * n_lakes, -1).to(device)  # (B*N, static_dim)

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

            pred = model(x_batch, static_features=static_batch)

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