"""
Training loop for SWOT-GNN.
"""
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
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


def _compute_loader_loss(loader: DataLoader, model: nn.Module, criterion: nn.Module, device: torch.device) -> float:
    """Compute average loss over an entire DataLoader (used for validation).

    Runs in eval mode with no_grad to save memory.  Iterates every sample
    in every batch and accumulates the per-sample loss, then returns the mean.
    Uses autocast for float16 inference (matches training precision).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            # Unpack the 4-element tuple produced by collate_fn
            data_lists, static_feats, labels, masks = batch
            if not data_lists:
                continue

            batch_size = len(data_lists)
            for b in range(batch_size):
                sample = data_lists[b]

                # sample is either a list of T PyG Data objects (one per time step)
                # or a single Data object when seq_len == 1.
                if isinstance(sample, list):
                    # Stack node features across time: (n_seg, T, n_feat)
                    x = torch.stack([d.x for d in sample], dim=1)
                    # All time steps share the same graph topology
                    edge_index = sample[0].edge_index
                else:
                    x = sample.x.unsqueeze(1)   # add time dimension: (n_seg, 1, n_feat)
                    edge_index = sample.edge_index

                x = x.to(device)
                edge_index = edge_index.to(device)
                static = static_feats[b].to(device)   # (n_seg, static_dim)
                lab = labels[b].to(device)             # (n_seg,) — target WSE
                msk = masks[b].to(device)              # (n_seg,) — observation mask

                # Forward pass in float16; ForecastHead already extracts the last time step
                # pred shape: (n_seg,) — 1-day-ahead WSE per node
                with autocast(device_type=device.type):
                    pred = model(x, edge_index, static_features=static)
                    loss = criterion(pred, lab, msk)
                total_loss += loss.item()
            n_batches += batch_size

    return total_loss / max(n_batches, 1)


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
    # GradScaler prevents float16 gradient underflow by scaling loss up before
    # backward and scaling gradients back down before the optimizer step.
    # Disabled automatically when device is CPU (float16 not accelerated on CPU).
    use_amp = device.type == "cuda"
    grad_scaler = GradScaler(enabled=use_amp)

    train_losses = []   # per-epoch average training loss
    val_losses = []     # per-epoch average validation loss (empty if no val_loader)
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0  # early-stopping counter
    stopped_early = False

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0   # counts individual samples, not batch groups

        for batch in train_loader:
            # Each batch is a tuple: (data_lists, static_feats, labels, masks)
            # data_lists[b] — list of T PyG Data objects for sample b
            # static_feats[b] — static node features tensor for sample b
            # labels[b]       — target WSE at forecast day for sample b
            # masks[b]        — observation mask at forecast day for sample b
            data_lists, static_feats, labels, masks = batch
            if not data_lists:
                continue

            batch_size = len(data_lists)
            # Accumulate gradients across all samples before stepping (true batch update)
            optimizer.zero_grad()
            batch_loss = 0.0
            for b in range(batch_size):
                sample = data_lists[b]

                # Build the temporal node-feature tensor x: (n_seg, T, n_feat)
                if isinstance(sample, list):
                    x = torch.stack([d.x for d in sample], dim=1)
                    edge_index = sample[0].edge_index
                else:
                    x = sample.x.unsqueeze(1)
                    edge_index = sample.edge_index

                x = x.to(device)
                edge_index = edge_index.to(device)
                static = static_feats[b].to(device)
                lab = labels[b].to(device)
                msk = masks[b].to(device)

                # Forward pass in float16; ForecastHead extracts the last time step
                # and projects it to a scalar — pred shape: (n_seg,)
                with autocast(device_type=device.type, enabled=use_amp):
                    pred = model(x, edge_index, static_features=static)
                    loss = criterion(pred, lab, msk)

                # Scale loss to prevent float16 gradient underflow, then accumulate.
                # Divide by batch_size so gradient scale matches a single averaged loss.
                grad_scaler.scale(loss / batch_size).backward()
                batch_loss += loss.item()

            # Unscale before clipping so the clip threshold is in true gradient units,
            # then step (skipped automatically if gradients contain inf/nan from overflow).
            if grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += batch_loss
            n_batches += batch_size

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Decide which loss to use for model selection:
        # prefer validation loss when a val_loader is provided.
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

        # Early stopping: halt when val loss has not improved for `patience` epochs.
        # Only triggered when a val_loader is provided (watching val loss is meaningful).
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
