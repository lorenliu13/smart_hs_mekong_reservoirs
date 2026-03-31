"""
Training loop for SWOT-GNN.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.swot_gnn import SWOTGNN


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


def _build_batched_inputs(data_lists, static_feats, labels, masks, device):
    """Stack all samples in a collated batch into a single set of tensors for one forward pass.

    WHY: Running the model once on a large tensor is much faster than running it B times
    on small tensors, because each model call has overhead (kernel launches, memory
    allocation) and small tensors leave the GPU underutilised.

    HOW — the graph batching trick:
    All samples share the same graph topology (same lakes, same edges).  We stack their
    node features into one long tensor and replicate edge_index B times, each copy shifted
    by `n` (the number of nodes per graph).  The result is B disjoint sub-graphs sitting
    inside one big disconnected graph — the standard PyG mini-batch format.

    Example with B=2 samples, n=3 nodes, e=2 edges:
        Sample 0 nodes: [0, 1, 2]      edge_index: [[0,1],[1,2]]
        Sample 1 nodes: [3, 4, 5]      edge_index: [[3,4],[4,5]]  ← original + n
        Combined nodes: [0,1,2, 3,4,5] edge_index: [[0,1,3,4],[1,2,4,5]]  shape (2, B*e)

    Returns:
        x_flat      : (B*n, T, n_feat)  — node features for all samples
        ei_batched  : (2, B*e)          — B offset copies of the base edge_index
        static_flat : (B*n, static_dim) — static attributes for all nodes
        lab_flat    : (B*n,)            — target WSE labels (horizon=1) or (B*n, horizon)
        msk_flat    : (B*n,)            — observation mask matching lab_flat
        batch_vec   : (B*n,)            — integer in [0, B) mapping each node to its sample;
                                          used by STBlock to keep GPS attention within samples
    """
    bs = len(data_lists)

    # ── Node features ────────────────────────────────────────────────────────────
    # Each sample is a list of T PyG Data objects (one graph snapshot per day).
    # Stack the per-day node feature matrices along a new time dimension to get
    # (n, T, n_feat), then stack across samples to get (B, n, T, n_feat).
    def _to_x(sample):
        if isinstance(sample, list):
            return torch.stack([d.x for d in sample], dim=1)   # (n, T, n_feat)
        return sample.x.unsqueeze(1)                            # (n, 1, n_feat) when seq_len=1

    x_all  = torch.stack([_to_x(data_lists[b]) for b in range(bs)]).to(device)
    B, n, T, F = x_all.shape
    x_flat = x_all.view(B * n, T, F)   # flatten B and n → (B*n, T, F)

    # ── Edge index ───────────────────────────────────────────────────────────────
    # All samples use the same lake graph, so the base edge_index is taken from
    # sample 0.  We then create B copies, each shifted by n*b so that sample b's
    # nodes start at index b*n — preventing any cross-sample edges.
    sample0 = data_lists[0]
    ei_base = (sample0[0].edge_index if isinstance(sample0, list) else sample0.edge_index).to(device)
    # offsets shape: (B, 1, 1) so it broadcasts over (B, 2, e)
    offsets    = (torch.arange(B, device=device) * n).view(B, 1, 1)
    ei_batched = (ei_base.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
    # Result shape: (2, B*e) — B disjoint copies of the graph edge list

    # ── Static features ──────────────────────────────────────────────────────────
    # static_feats may arrive as a pre-stacked tensor (B, n, static_dim) from the
    # lake collate_fn, or as a plain list when called from older code paths.
    if isinstance(static_feats, torch.Tensor):
        static_flat = static_feats.view(B * n, -1).to(device)      # (B*n, static_dim)
    else:
        static_flat = torch.stack([static_feats[b] for b in range(bs)]).view(B * n, -1).to(device)

    # ── Labels and masks ─────────────────────────────────────────────────────────
    # labels/masks may be (B, n, horizon) tensors or lists of (n,) tensors.
    # squeeze(-1) collapses the horizon dimension when forecast_horizon=1 so the
    # loss function receives a flat (B*n,) vector.
    if isinstance(labels, torch.Tensor):
        lab_flat = labels.view(B * n, -1).squeeze(-1).to(device)   # (B*n,) or (B*n, horizon)
        msk_flat = masks.view(B * n, -1).squeeze(-1).to(device)
    else:
        lab_flat = torch.stack([labels[b] for b in range(bs)]).view(B * n, -1).squeeze(-1).to(device)
        msk_flat = torch.stack([masks[b]  for b in range(bs)]).view(B * n, -1).squeeze(-1).to(device)

    # ── Batch vector ─────────────────────────────────────────────────────────────
    # batch_vec[i] = which sample node i belongs to.
    # Example B=2, n=3: [0, 0, 0, 1, 1, 1]
    # STBlock uses this to split nodes back into per-sample groups before the
    # GPS timestep vectorisation, and GraphGPSLayer uses it so global attention
    # does not mix nodes from different samples.
    batch_vec = torch.arange(B, device=device).repeat_interleave(n)  # (B*n,)

    return x_flat, ei_batched, static_flat, lab_flat, msk_flat, batch_vec


def _compute_loader_loss(loader: DataLoader, model: nn.Module, criterion: nn.Module,
                         device: torch.device) -> float:
    """Compute average loss over an entire DataLoader (used for validation).

    Runs in eval mode with no_grad to save memory.  Each collated batch is processed
    in a single batched forward pass instead of sample-by-sample.
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            data_lists, static_feats, labels, masks = batch
            if not data_lists:
                continue

            # Build all tensors for a single batched forward pass
            x_flat, ei_batched, static_flat, lab_flat, msk_flat, batch_vec = \
                _build_batched_inputs(data_lists, static_feats, labels, masks, device)

            pred = model(x_flat, ei_batched, static_features=static_flat, batch=batch_vec)
            loss = criterion(pred, lab_flat, msk_flat)

            bs = len(data_lists)
            # loss is already averaged over observed nodes; multiply by bs so we
            # can later divide by total n_samples to get a true per-sample mean.
            total_loss += loss.item() * bs
            n_samples  += bs

    return total_loss / max(n_samples, 1)


def train_swot_gnn(
    train_loader: DataLoader,
    model: SWOTGNN,
    device: torch.device,
    scaler: StandardScaler,
    lr: float = 0.001,
    num_epochs: int = 100,
    save_path: Optional[Path] = None,
    val_loader: Optional[DataLoader] = None,
    patience: int = 25,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """
    Train SWOT-GNN with MSE loss at observed nodes. Best model saved by validation loss.

    Early stopping halts training when the monitored loss (val if available, else train)
    has not improved for `patience` consecutive epochs.

    Args:
        train_loader: DataLoader yielding (data_lists, labels, masks)
        model: SWOTGNN model
        device: torch device
        scaler: StandardScaler fit on WSE values (for inverse transform)
        lr: Learning rate
        num_epochs: Maximum number of epochs
        save_path: Path to save best model
        val_loader: Optional validation DataLoader. If provided, save best by val loss.
        patience: Number of epochs without improvement before stopping early.
                  Only active when val_loader is provided (defaults to 25).
        grad_clip: Max gradient norm for gradient clipping (default 1.0).

    Returns:
        Dict with train_losses, val_losses, best_epoch, best_loss, stopped_early
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Halve LR when val loss hasn't improved for 8 epochs; min LR floor = 1e-5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    criterion = ObservedMSELoss()

    # ── Automatic Mixed Precision (AMP) ──────────────────────────────────────────
    # On CUDA, AMP runs most ops in float16 (half precision), which halves memory
    # bandwidth and uses Tensor Cores → roughly 1.5–2× faster with no code changes
    # to the model.  On CPU, use_amp=False makes autocast and GradScaler no-ops so
    # behaviour is identical to full float32 training.
    use_amp    = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler(enabled=use_amp)
    # GradScaler is needed because float16 has a small dynamic range: gradients
    # can underflow to zero.  The scaler multiplies the loss by a large factor
    # before backward(), then divides the gradients back before optimizer.step().

    train_losses = []   # per-epoch average training loss
    val_losses = []     # per-epoch average validation loss (empty if no val_loader)
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0  # early-stopping counter
    stopped_early = False

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # Each batch is a tuple: (data_lists, static_feats, labels, masks)
            data_lists, static_feats, labels, masks = batch
            if not data_lists:
                continue

            bs = len(data_lists)
            optimizer.zero_grad()

            # ── Build one big tensor for all B samples ────────────────────────
            # Instead of looping over samples and calling model B times, we stack
            # everything into (B*n, ...) tensors and run a single forward pass.
            # See _build_batched_inputs for the graph-offset trick that keeps
            # samples isolated from each other inside the shared edge_index.
            x_flat, ei_batched, static_flat, lab_flat, msk_flat, batch_vec = \
                _build_batched_inputs(data_lists, static_feats, labels, masks, device)

            # ── Forward pass under AMP ────────────────────────────────────────
            # autocast selects float16 for matrix multiplies and convolutions
            # but keeps sensitive ops (softmax, layer norm) in float32.
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x_flat, ei_batched, static_features=static_flat,
                             batch=batch_vec)          # (B*n,) — WSE per node
                loss = criterion(pred, lab_flat, msk_flat)

            # ── Backward + gradient clip + optimizer step ─────────────────────
            if use_amp:
                # Scale loss up → backward (prevents float16 gradient underflow)
                amp_scaler.scale(loss).backward()
                # Unscale gradients back to true magnitudes before clipping
                amp_scaler.unscale_(optimizer)
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # Step only if gradients are finite; otherwise skip and reduce scale
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    # Clip gradient norm to prevent LSTM exploding gradients
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Accumulate loss scaled back to a per-sample sum
            epoch_loss += loss.item() * bs
            n_batches  += bs

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # ── Validation & model selection ──────────────────────────────────────
        # Prefer validation loss for model selection when a val_loader is provided;
        # fall back to training loss when running without a validation split.
        if val_loader is not None:
            avg_val = _compute_loader_loss(val_loader, model, criterion, device)
            val_losses.append(avg_val)
            loss_to_compare = avg_val
            scheduler.step(avg_val)
        else:
            loss_to_compare = avg_train
            scheduler.step(avg_train)

        # Save the model whenever it achieves a new best loss; reset patience counter
        if loss_to_compare < best_loss:
            best_loss = loss_to_compare
            best_epoch = epoch
            epochs_without_improvement = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            # No improvement this epoch — advance the early-stopping counter
            epochs_without_improvement += 1

        # Print a progress summary every epoch
        current_lr = optimizer.param_groups[0]["lr"]
        msg = f"Epoch {epoch+1}/{num_epochs} | Train: {avg_train:.4f}"
        if val_loader is not None:
            msg += f" | Val: {avg_val:.4f}"
        msg += f" | Best: {best_epoch+1} ({best_loss:.4f}) | LR: {current_lr:.2e}"
        print(msg)

        # ── Early stopping ────────────────────────────────────────────────────
        # Halt training when val loss has not improved for `patience` consecutive
        # epochs — prevents overfitting and saves compute.
        # Only active when a val_loader is provided; without one, train loss is a
        # poor early-stopping signal (it will keep decreasing even while overfitting).
        if val_loader is not None and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}: val loss did not improve for {patience} epochs.")
            stopped_early = True
            break

    result = {
        "train_losses": train_losses,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "stopped_early": stopped_early,
    }
    if val_loader is not None:
        result["val_losses"] = val_losses
    return result
