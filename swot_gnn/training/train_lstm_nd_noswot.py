"""
Ablation epoch function: identical to train_lstm_nd._run_epoch_lstm_nd but
zeros out SWOT observation-based input features (indices 0-5) before the
forward pass.  Indices 6-7 (doy_sin/cos) and 8-20 (ERA5 climate) are kept.

This tests whether the model has genuine climate-driven forecasting skill
beyond SWOT-persistence (latest_wse forward-fill).

Exported:
    _run_epoch_lstm_nd_noswot
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# obs_mask, latest_wse, latest_wse_u, latest_wse_std, latest_area_total,
# days_since_last_obs — zeroed to remove SWOT state initialisation signal.
# doy_sin (6) and doy_cos (7) are date-derived and intentionally kept.
_SWOT_OBS_FEAT_DIM = 6


def _run_epoch_lstm_nd_noswot(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    grad_clip: float = 1.0,
    spatial_mask: torch.Tensor = None,
) -> float:
    """Run one epoch with SWOT observation features zeroed out.

    Drop-in replacement for _run_epoch_lstm_nd.  All arguments and return
    value are identical; the only difference is that x_batch[:, :, :6] is
    set to zero before the model forward pass so the model cannot use
    obs_mask, latest_wse, or any other SWOT-derived state.

    Args:
        model:        LSTMBaselineMultiStep (or Gauss variant).
        loader:       DataLoader yielding (data_lists, static_feats, labels, masks).
        criterion:    Multi-step loss (ObservedMSELossMultiStep or CRPS variant).
        device:       Torch device.
        optimizer:    If provided, performs a gradient update (train mode).
        grad_clip:    Max gradient norm for clipping (0 disables).
        spatial_mask: Optional (n_lakes,) float tensor; 1 for active lakes.

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

            x_list  = [torch.stack([d.x for d in data_lists[b]], dim=1) for b in range(B)]
            n_lakes = x_list[0].size(0)
            x_batch = torch.cat(x_list, dim=0).to(device)             # (B*N, T, feat)

            # Zero out SWOT observation-based features on all timesteps
            x_batch[:, :, :_SWOT_OBS_FEAT_DIM] = 0.0

            static_batch = static_feats.reshape(B * n_lakes, -1).to(device)

            H   = labels.shape[-1]
            lab = labels.reshape(B * n_lakes, H).to(device)
            msk = masks.reshape(B * n_lakes, H).to(device)

            if spatial_mask is not None:
                sm  = spatial_mask.to(device).float()
                sm  = sm.unsqueeze(0).expand(B, -1).reshape(-1)
                sm  = sm.unsqueeze(-1).expand(-1, H)
                msk = msk * sm

            if is_train:
                optimizer.zero_grad()

            pred = model(x_batch, static_features=static_batch)

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
