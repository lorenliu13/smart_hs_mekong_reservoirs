"""
Ablation: temporal expanding-window CV for multi-day lake WSE forecasting
using an LSTM-only baseline with SWOT observation features zeroed out.

Identical to wsend_lstm_training/run_training_lake_lstm_wsend_temporal_cv.py
except that _run_epoch_lstm_nd_noswot is used in place of _run_epoch_lstm_nd.
SWOT input features (obs_mask, latest_wse, wse_u, wse_std, area_total,
days_since_last_obs — indices 0-5) are set to zero on every timestep before
the model forward pass.  DOY encoding (indices 6-7) and ERA5 climate
(indices 8-20) are unchanged.

Purpose:
    Quantify how much model skill comes from SWOT state initialisation
    (latest_wse persistence) vs. genuine climate-driven forecasting.
    If this ablation performs similarly to the full model, the model has
    learned little beyond orbit-cycle persistence.

Usage:
  python ablation_noswot/run_training_lake_lstm_wsend_temporal_cv_noswot.py \\
    --config          ../configs/lstm/exp02_mekong_lstm_wsend_era5_ifshres_gritv06_202312_202602_temporalcv.yaml \\
    --wse-datacube    /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube   /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube  /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --lake-graph      /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --run-name        ablation_noswot_lstm_temporalcv \\
    --fold-idx        0
"""
import argparse
import csv
import json
import shutil
import sys
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve package root (swot_gnn/) two levels up from this file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.temporal_cv import (
    build_temporal_cv_fold,
    N_TEMPORAL_FOLDS,
    TEMPORAL_FOLD_DATES,
)
from data.temporal_graph_dataset_lake import collate_temporal_graph_batch_lake
from models.registry import MODEL_REGISTRY
from training.train_lstm_nd_noswot import _run_epoch_lstm_nd_noswot


# ── Main training routine ──────────────────────────────────────────────────────

def train(cfg, args):
    device = torch.device(args.device)

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    forecast_horizon = cfg["training"]["forecast_horizon"]
    fold_def = TEMPORAL_FOLD_DATES[args.fold_idx]

    # ── Build temporal CV datasets ────────────────────────────────────────────
    train_ds, val_ds, test_ds, norm_stats = build_temporal_cv_fold(
        wse_datacube_path            = args.wse_datacube,
        era5_climate_datacube_path   = args.era5_datacube,
        ecmwf_forecast_datacube_path = args.ecmwf_datacube,
        static_datacube_path         = args.static_datacube,
        lake_graph_path              = args.lake_graph,
        fold_idx                     = args.fold_idx,
        seq_len                      = cfg["training"]["seq_len"],
        forecast_horizon             = forecast_horizon,
        val_frac                     = cfg["training"].get("val_frac", 0.15),
        require_obs_on_any_forecast_day = True,
    )

    static_dim = int(train_ds.static_features.shape[-1])
    cfg["model"]["static_dim"] = static_dim
    print(f"Static features: {static_dim} attributes per lake (auto-detected)")
    print("Ablation: SWOT observation features (indices 0-5) will be zeroed.")

    train_spatial_mask = train_ds.spatial_mask
    val_spatial_mask   = val_ds.spatial_mask
    test_spatial_mask  = test_ds.spatial_mask

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size = cfg["training"]["batch_size"],
        collate_fn = collate_temporal_graph_batch_lake,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = torch.utils.data.DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg  = dict(cfg["model"])
    model_type = model_cfg.pop("model_type", "LSTMBaselineMultiStep")
    spec       = MODEL_REGISTRY[model_type]
    model      = spec.model_cls(**model_cfg).to(device)
    criterion  = spec.loss_cls()
    optimizer  = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    patience  = cfg["training"].get("patience", 20)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    # ── Output directory ──────────────────────────────────────────────────────
    run_dir = Path(args.save_dir) / args.base_run_name / f"fold_{args.fold_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = run_dir / "_best_model_tmp.pt"
    print(f"\nRun: {args.run_name}  →  {run_dir}\n")

    shutil.copy2(args.config, run_dir / "config.yaml")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_epoch       = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    lr_history, ts_history   = [], []
    stopped_early = False
    training_start = time.time()

    for epoch in range(cfg["training"]["num_epochs"]):
        epoch_start = time.time()
        avg_train = _run_epoch_lstm_nd_noswot(model, train_loader, criterion, device,
                                              optimizer=optimizer, grad_clip=grad_clip,
                                              spatial_mask=train_spatial_mask)
        avg_val   = _run_epoch_lstm_nd_noswot(model, val_loader,   criterion, device,
                                              spatial_mask=val_spatial_mask)
        epoch_secs = time.time() - epoch_start

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        lr = optimizer.param_groups[0]["lr"]
        lr_history.append(lr)
        ts_history.append(datetime.now(timezone.utc).isoformat(timespec="seconds"))

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), tmp_ckpt)
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:>3}/{cfg['training']['num_epochs']} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"Best: ep{best_epoch+1} ({best_val_loss:.4f}) | LR: {lr:.2e} | "
              f"Time: {epoch_secs:.1f}s")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}: "
                  f"val loss did not improve for {patience} epochs.")
            stopped_early = True
            break

    total_training_secs = time.time() - training_start
    h, rem = divmod(int(total_training_secs), 3600)
    m, s   = divmod(rem, 60)
    print(f"\nTotal training time: {h:02d}:{m:02d}:{s:02d} "
          f"({total_training_secs:.1f}s, {len(train_losses)} epochs)")

    # ── Rename checkpoint ─────────────────────────────────────────────────────
    final_ckpt_name = f"best_epoch{best_epoch + 1:03d}.pt"
    final_ckpt      = run_dir / final_ckpt_name
    tmp_ckpt.rename(final_ckpt)

    # ── Evaluate on held-out test window ──────────────────────────────────────
    print(
        f"\nEvaluating on held-out test window "
        f"[{fold_def['test_start']} → {fold_def['test_end']}] …"
    )
    state_dict = torch.load(final_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    test_loss = _run_epoch_lstm_nd_noswot(model, test_loader, criterion, device,
                                          spatial_mask=test_spatial_mask)
    print(f"Test loss (held-out window): {test_loss:.4f}")

    # ── Re-save checkpoint with full training history ─────────────────────────
    torch.save({
        "model_state_dict": state_dict,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "best_epoch":       best_epoch,
        "best_val_loss":    best_val_loss,
        "test_loss":        test_loss,
        "stopped_early":    stopped_early,
        "norm_stats":       norm_stats,
        "config":           cfg,
        "ablation":         "noswot_input",
    }, final_ckpt)

    # ── Save training_log.csv ─────────────────────────────────────────────────
    with open(run_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "timestamp"])
        for i, (tl, vl, lr_i, ts_i) in enumerate(
                zip(train_losses, val_losses, lr_history, ts_history)):
            writer.writerow([i + 1, f"{tl:.8f}", f"{vl:.8f}", f"{lr_i:.2e}", ts_i])

    # ── Save test_metrics.json ────────────────────────────────────────────────
    test_metrics = {
        "ablation":         "noswot_input",
        "fold_idx":         args.fold_idx,
        "train_start":      fold_def["train_start"],
        "train_end":        fold_def["train_end"],
        "test_start":       fold_def["test_start"],
        "test_end":         fold_def["test_end"],
        "forecast_horizon": forecast_horizon,
        "n_lakes":          int(norm_stats["n_lakes"]),
        "n_train_dates":    int(norm_stats["n_train_dates"]),
        "n_val_dates":      int(norm_stats["n_val_dates"]),
        "n_test_dates":     int(norm_stats["n_test_dates"]),
        "test_loss":        float(test_loss),
        "best_val_loss":    float(best_val_loss),
    }
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ── Append to registry.csv ────────────────────────────────────────────────
    registry_path = Path(args.save_dir) / "registry_lstm_wsend_temporal_cv_noswot.csv"
    exp_id        = args.run_name.split("_")[0]
    reg_header = [
        "run_name", "exp_id", "seed", "fold_idx",
        "train_start", "train_end", "test_start", "test_end",
        "forecast_horizon", "n_lakes", "n_train_dates", "n_test_dates",
        "best_epoch", "best_val_loss", "test_loss",
        "total_epochs", "stopped_early", "runtime_min",
        "lr", "seq_len", "st_blocks", "ablation", "status",
    ]
    reg_row = [
        args.run_name,
        exp_id,
        args.seed,
        args.fold_idx,
        fold_def["train_start"],
        fold_def["train_end"],
        fold_def["test_start"],
        fold_def["test_end"],
        forecast_horizon,
        norm_stats["n_lakes"],
        norm_stats["n_train_dates"],
        norm_stats["n_test_dates"],
        best_epoch + 1,
        f"{best_val_loss:.6f}",
        f"{test_loss:.6f}",
        len(train_losses),
        stopped_early,
        round(total_training_secs / 60, 2),
        cfg["training"]["lr"],
        cfg["training"]["seq_len"],
        cfg["model"].get("st_blocks", ""),
        "noswot_input",
        "done",
    ]
    write_header = not registry_path.exists()
    with open(registry_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(reg_header)
        writer.writerow(reg_row)

    # ── Save training_curve.png ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses,   label="Val loss")
    ax.axhline(test_loss, color="red", linestyle=":", linewidth=1.5,
               label=f"Test loss (held-out window): {test_loss:.4f}")
    ax.axvline(best_epoch + 1, color="gray", linestyle="--",
               label=f"Best (epoch {best_epoch + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss — {forecast_horizon}-day forecast (no SWOT input)")
    ax.set_title(
        f"{args.run_name}  —  fold {args.fold_idx} "
        f"[test {fold_def['test_start']} → {fold_def['test_end']}]\n"
        f"Ablation: SWOT observation features zeroed"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "training_curve.png", dpi=120)
    plt.close(fig)

    # ── Save run_config.yaml ──────────────────────────────────────────────────
    run_config = {
        "run_name": args.run_name,
        "exp_id":   args.run_name.split("_")[0],
        "seed":     args.seed,
        "ablation": "noswot_input",
        "temporal_cv": {
            "fold_idx":      args.fold_idx,
            "train_start":   fold_def["train_start"],
            "train_end":     fold_def["train_end"],
            "test_start":    fold_def["test_start"],
            "test_end":      fold_def["test_end"],
            "n_train_dates": int(norm_stats["n_train_dates"]),
            "n_val_dates":   int(norm_stats["n_val_dates"]),
            "n_test_dates":  int(norm_stats["n_test_dates"]),
            "n_lakes":       int(norm_stats["n_lakes"]),
        },
        "data": {
            "wse_datacube":    args.wse_datacube,
            "era5_datacube":   args.era5_datacube,
            "ecmwf_datacube":  args.ecmwf_datacube,
            "static_datacube": args.static_datacube,
            "lake_graph":      args.lake_graph,
        },
        "model":    cfg["model"],
        "training": cfg["training"],
        "result": {
            "best_epoch":         best_epoch + 1,
            "best_val_loss":      float(best_val_loss),
            "test_loss":          float(test_loss),
            "total_epochs":       len(train_losses),
            "stopped_early":      stopped_early,
            "training_time_secs": round(total_training_secs, 1),
            "training_time_hms":  f"{h:02d}:{m:02d}:{s:02d}",
            "checkpoint":         final_ckpt_name,
        },
    }
    with open(run_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nOutputs saved to {run_dir}/")
    print(f"  {final_ckpt_name:<26} — model weights + full training history")
    print(f"  config.yaml                — frozen copy of source config")
    print(f"  training_log.csv           — per-epoch train/val loss, lr, timestamp")
    print(f"  test_metrics.json          — held-out window test loss for this fold")
    print(f"  training_curve.png         — loss plot with test loss reference line")
    print(f"  run_config.yaml            — hyperparameters + data paths + full result summary")
    print(f"Registry row appended → {registry_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(
        description="Ablation: temporal CV LSTM with SWOT observation features zeroed"
    )
    parser.add_argument(
        "--config",
        default=str(script_dir / "configs" / "lstm" /
                    "exp02_mekong_lstm_wsend_era5_ifshres_gritv06_202312_202602_temporalcv.yaml"),
        help="Path to YAML config (same config as full model)",
    )
    parser.add_argument("--wse-datacube",    required=True)
    parser.add_argument("--era5-datacube",   required=True)
    parser.add_argument("--ecmwf-datacube",  required=True)
    parser.add_argument("--static-datacube", required=True)
    parser.add_argument("--lake-graph",      required=True)
    parser.add_argument("--save-dir",  default="checkpoints",
                        help="Root directory for all run outputs")
    parser.add_argument("--run-name",  required=True)
    parser.add_argument("--fold-idx",  type=int, required=True,
                        help="Which temporal fold to run (0, 1, or 2)")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--patience",   type=int, default=None)
    parser.add_argument("--seed",       type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.patience is not None:
        cfg["training"]["patience"] = args.patience
    if args.seed is None:
        args.seed = cfg.get("seed", 42)

    forecast_horizon = cfg["training"]["forecast_horizon"]
    assert forecast_horizon > 1, (
        f"Designed for forecast_horizon > 1, got {forecast_horizon}."
    )

    if not 0 <= args.fold_idx < N_TEMPORAL_FOLDS:
        parser.error(f"--fold-idx must be 0, 1, or 2 (got {args.fold_idx})")

    model_type = cfg["model"].get("model_type", "LSTMBaselineMultiStep")
    model_slug = MODEL_REGISTRY[model_type].slug

    args.base_run_name = args.run_name
    args.run_name = (
        f"{args.run_name}"
        f"_fold{args.fold_idx}of{N_TEMPORAL_FOLDS}"
        f"_h{forecast_horizon}"
        f"_{model_slug}"
        f"_s{args.seed}"
    )

    train(cfg, args)


if __name__ == "__main__":
    main()
