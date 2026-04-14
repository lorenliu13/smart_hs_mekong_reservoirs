"""
Spatial cross-validation for 1-day-ahead lake WSE forecasting.

Lakes are randomly split into n_folds (default 5) groups.  One fold of lakes is
held out as the test set; the remaining lakes are used for training and
validation.  Run one fold at a time and submit each as a separate cluster job.

Key differences from run_training_lake_wse1d.py (temporal CV):
  - Dataset built with build_spatial_cv_fold instead of
    build_temporal_dataset_from_lake_datacubes.
  - ALL lake nodes remain in the graph so message passing spans the full
    network; spatial_mask gates which nodes contribute
    to the loss.
  - Normalization statistics are derived from training lakes only.
  - After training, the best checkpoint is evaluated on the held-out test lakes
    across all time steps; test_metrics.json is written to the run directory.

Usage (single fold):
  python run_spatial_cv_wse1d.py \\
    --config     configs/exp01_nextday_wse.yaml \\
    --wse-datacube   /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube  /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --lake-graph  /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --run-name   exp01_spatialcv \\
    --fold-idx   0

Launch all folds (bash example):
  for i in 0 1 2 3 4; do
    python run_spatial_cv_wse1d.py ... --fold-idx $i &
  done
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.temporal_graph_dataset_lake import (
    build_spatial_cv_fold,
    collate_temporal_graph_batch_lake,
)
from models.registry import MODEL_REGISTRY
from training.train import _run_epoch


# ── Main training routine ──────────────────────────────────────────────────────

def train(cfg, args):
    device = torch.device(args.device)

    # Set random seeds for reproducibility; this affects the lake split and model initialization
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # ── Build spatial CV datasets ───────────────────────────────────────────
    train_ds, val_ds, test_ds, norm_stats = build_spatial_cv_fold(
        wse_datacube_path            = args.wse_datacube,
        era5_climate_datacube_path   = args.era5_datacube,
        ecmwf_forecast_datacube_path = args.ecmwf_datacube,
        static_datacube_path         = args.static_datacube,
        lake_graph_path              = args.lake_graph,
        seq_len                      = cfg["training"]["seq_len"],
        forecast_horizon             = cfg["training"]["forecast_horizon"],
        n_folds                      = args.n_folds,
        fold_idx                     = args.fold_idx,
        spatial_split_seed           = args.spatial_seed,
        val_frac                     = cfg["training"].get("val_frac", 0.15),
        val_method                   = args.val_method,
        spatial_val_frac             = args.spatial_val_frac,
    )

    # Auto-detect static_dim from the loaded datacube and override config placeholder
    static_dim = int(train_ds.static_features.shape[-1])
    cfg["model"]["static_dim"] = static_dim
    print(f"Static features: {static_dim} attributes per lake (auto-detected)")

    # Spatial masks gate the loss to the relevant lake subset in each phase
    train_spatial_mask = train_ds.spatial_mask   # (n_lakes,) — train lakes
    val_spatial_mask   = val_ds.spatial_mask     # (n_lakes,) — spatial-val lakes
    test_spatial_mask  = test_ds.spatial_mask    # (n_lakes,) — held-out test lakes

    # ── DataLoaders ─────────────────────────────────────────────────────────
    # Create PyTorch DataLoaders for train, val, and test sets; use the same collate_fn for all since they share the same graph structure
    loader_kwargs = dict(
        batch_size = cfg["training"]["batch_size"],
        collate_fn = collate_temporal_graph_batch_lake,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = torch.utils.data.DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Model ───────────────────────────────────────────────────────────────
    model_cfg  = dict(cfg["model"])
    model_type = model_cfg.pop("model_type", "SWOTGNN")
    spec       = MODEL_REGISTRY[model_type]
    model      = spec.model_cls(**model_cfg).to(device)
    criterion  = spec.loss_cls()
    optimizer  = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    patience  = cfg["training"].get("patience", 20)

    # Get the number of trainable parameters for reference
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    # ── Output directory ────────────────────────────────────────────────────
    # Outputs are saved under save_dir / base_run_name / fold_{fold_idx}/ so
    # all folds of the same experiment are grouped under one parent directory.
    run_dir  = Path(args.save_dir) / args.base_run_name / f"fold_{args.fold_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = run_dir / "_best_model_tmp.pt"
    print(f"\nRun: {args.run_name}  →  {run_dir}\n")

    shutil.copy2(args.config, run_dir / "config.yaml")

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_epoch       = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    lr_history, ts_history   = [], []
    stopped_early = False
    training_start = time.time()

    for epoch in range(cfg["training"]["num_epochs"]):
        epoch_start = time.time()
        # Training loss: observed MSE on train lakes only
        avg_train = _run_epoch(model, train_loader, criterion, device,
                               optimizer=optimizer, grad_clip=grad_clip,
                               spatial_mask=train_spatial_mask)
        # Validation loss: observed MSE on spatial-val lakes (unseen during training)
        avg_val   = _run_epoch(model, val_loader,   criterion, device,
                               spatial_mask=val_spatial_mask)
        epoch_secs = time.time() - epoch_start

        train_losses.append(avg_train) # track train loss for this epoch
        val_losses.append(avg_val) # track val loss for this epoch
        scheduler.step(avg_val) # adjust learning rate based on val loss plateau

        lr = optimizer.param_groups[0]["lr"] # get current learning rate for logging
        lr_history.append(lr) # track learning rate for this epoch
        ts_history.append(datetime.now(timezone.utc).isoformat(timespec="seconds")) # track timestamp for this epoch
        
        if avg_val < best_val_loss:
            # If validation loss improved, update best_val_loss and best_epoch, reset patience counter, and save model checkpoint.
            best_val_loss    = avg_val
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), tmp_ckpt)
        else:
            # If validation loss did not improve, increment patience counter. If patience counter exceeds the patience threshold, we will stop training early.
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
    
    # Calcluate the total training time
    total_training_secs = time.time() - training_start
    h, rem = divmod(int(total_training_secs), 3600)
    m, s   = divmod(rem, 60)
    print(f"\nTotal training time: {h:02d}:{m:02d}:{s:02d} "
          f"({total_training_secs:.1f}s, {len(train_losses)} epochs)")

    # ── Rename checkpoint to encode best epoch ───────────────────────────────
    final_ckpt_name = f"best_epoch{best_epoch + 1:03d}.pt"
    final_ckpt      = run_dir / final_ckpt_name
    tmp_ckpt.rename(final_ckpt)

    # ── Evaluate on held-out test lakes ──────────────────────────────────────
    print("\nEvaluating on held-out test lakes …")
    state_dict = torch.load(final_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    test_loss = _run_epoch(model, test_loader, criterion, device,
                           spatial_mask=test_spatial_mask)
    print(f"Test loss (held-out lakes): {test_loss:.4f}")

    # ── Re-save checkpoint enriched with training history + test result ──────
    torch.save({
        "model_state_dict": state_dict,
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss":     test_loss,
        "stopped_early": stopped_early,
        "norm_stats":    norm_stats,
        "config":        cfg,
    }, final_ckpt)

    # ── Save training_log.csv ────────────────────────────────────────────────
    with open(run_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "timestamp"])
        for i, (tl, vl, lr_i, ts_i) in enumerate(
                zip(train_losses, val_losses, lr_history, ts_history)):
            writer.writerow([i + 1, f"{tl:.8f}", f"{vl:.8f}", f"{lr_i:.2e}", ts_i])

    # ── Save test_metrics.json ───────────────────────────────────────────────
    test_metrics = {
        "fold_idx":      args.fold_idx,
        "n_folds":       args.n_folds,
        "n_test_lakes":  int(norm_stats["n_test_lakes"]),
        "test_loss":     float(test_loss),
        "best_val_loss": float(best_val_loss),
    }
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ── Append to registry.csv ───────────────────────────────────────────────
    registry_path = Path(args.save_dir) / "registry_spatial_cv.csv"
    exp_id        = args.run_name.split("_")[0]
    reg_header = [
        "run_name", "exp_id", "seed", "fold_idx", "n_folds",
        "n_train_lakes", "n_test_lakes",
        "best_epoch", "best_val_loss", "test_loss",
        "total_epochs", "stopped_early", "runtime_min",
        "lr", "seq_len", "st_blocks", "status",
    ]
    reg_row = [
        args.run_name,
        exp_id,
        args.seed,
        args.fold_idx,
        args.n_folds,
        norm_stats["n_train_lakes"],
        norm_stats["n_test_lakes"],
        best_epoch + 1,
        f"{best_val_loss:.6f}",
        f"{test_loss:.6f}",
        len(train_losses),
        stopped_early,
        round(total_training_secs / 60, 2),
        cfg["training"]["lr"],
        cfg["training"]["seq_len"],
        cfg["model"].get("st_blocks", ""),
        "done",
    ]
    write_header = not registry_path.exists()
    with open(registry_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(reg_header)
        writer.writerow(reg_row)

    # ── Save training_curve.png ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss (train lakes)")
    ax.plot(epochs, val_losses,   label="Val loss (train lakes)")
    ax.axhline(test_loss, color="red", linestyle=":", linewidth=1.5,
               label=f"Test loss (held-out lakes): {test_loss:.4f}")
    ax.axvline(best_epoch + 1, color="gray", linestyle="--",
               label=f"Best (epoch {best_epoch + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (observed lakes)")
    ax.set_title(f"{args.run_name}  —  fold {args.fold_idx}/{args.n_folds}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "training_curve.png", dpi=120)
    plt.close(fig)

    # ── Save run_config.yaml ────────────────────────────────────────────────
    run_config = {
        "run_name": args.run_name,
        "exp_id":   args.run_name.split("_")[0],
        "seed":     args.seed,
        "spatial_cv": {
            "fold_idx":           args.fold_idx,
            "n_folds":            args.n_folds,
            "spatial_split_seed": args.spatial_seed,
            "n_train_lakes":      int(norm_stats["n_train_lakes"]),
            "n_test_lakes":       int(norm_stats["n_test_lakes"]),
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
            "best_epoch":          best_epoch + 1,
            "best_val_loss":       float(best_val_loss),
            "test_loss":           float(test_loss),
            "total_epochs":        len(train_losses),
            "stopped_early":       stopped_early,
            "training_time_secs":  round(total_training_secs, 1),
            "training_time_hms":   f"{h:02d}:{m:02d}:{s:02d}",
            "checkpoint":          final_ckpt_name,
        },
    }
    with open(run_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nOutputs saved to {run_dir}/")
    print(f"  {final_ckpt_name:<26} — model weights + full training history")
    print(f"  config.yaml                — frozen copy of source config")
    print(f"  training_log.csv           — per-epoch train/val loss, lr, timestamp")
    print(f"  test_metrics.json          — held-out lake test loss for this fold")
    print(f"  training_curve.png         — loss plot with test loss reference line")
    print(f"  run_config.yaml            — hyperparameters + data paths + full result summary")
    print(f"Registry row appended → {registry_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # __file__ is the path to this script; resolve to absolute path and get parent dir
    script_dir = Path(__file__).resolve().parent.parent
    # Set up argument parser with config path defaulting to configs/default.yaml
    parser = argparse.ArgumentParser(
        description="Spatial CV — 1-day-ahead lake WSE forecasting (SWOT-GNN)"
    ) # description shown in python script.py --help
    # This allows user to specifiy a custom settings file 
    # help: provides a description of this argument for the help menu
    parser.add_argument(
        "--config", default=str(script_dir / "configs" / "default.yaml"),
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument("--wse-datacube",    required=True)
    parser.add_argument("--era5-datacube",   required=True)
    parser.add_argument("--ecmwf-datacube",  required=True)
    parser.add_argument("--static-datacube", required=True)
    parser.add_argument("--lake-graph",      required=True)
    parser.add_argument("--save-dir",  default="checkpoints",
                        help="Root directory for all run outputs")
    parser.add_argument("--run-name",  required=True,
                        help="Base run name; fold index and model slug are appended automatically")
    parser.add_argument("--fold-idx",  type=int, required=True,
                        help="Which spatial fold to use as the test set (0-indexed)")
    parser.add_argument("--n-folds",   type=int, default=5,
                        help="Total number of spatial folds (default 5)")
    parser.add_argument("--spatial-seed", type=int, default=42,
                        help="RNG seed for the lake shuffle (default 42)")
    parser.add_argument(
        "--val-method", default="temporal", choices=["temporal", "spatial"],
        help=(
            "Validation strategy. "
            "'temporal' (default): hold out last val_frac of dates from train-fold lakes. "
            "'spatial': use all dates; hold out spatial_val_frac of train-fold lakes for val."
        ),
    )
    parser.add_argument(
        "--spatial-val-frac", type=float, default=0.1,
        help="Fraction of train-fold lakes used as spatial val set (only with --val-method spatial, default 0.1)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--patience",   type=int, default=None)
    parser.add_argument("--seed",       type=int, default=None,
                        help="Model training seed (overrides config)")
    # create args from the command line input
    args = parser.parse_args()

    with open(args.config) as f:
        # Load config from YAML file; this should include all hyperparameters and settings
        cfg = yaml.safe_load(f)

    # if the user specified num_epochs or patience on the command line, override the config values
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.patience is not None:
        cfg["training"]["patience"] = args.patience

    # if the user did not specify a seed on the command line, use the one from the config or default to 42
    if args.seed is None:
        args.seed = cfg.get("seed", 42)

    # look up model slug for the specified model type; default to "SWOTGNN" if not specified in config
    model_type = cfg["model"].get("model_type", "SWOTGNN")
    # the MODEL_REGISTRY maps model type strings to their corresponding classes and metadata; we extract the slug for naming purposes
    model_slug = MODEL_REGISTRY[model_type].slug # slug is a short identifier for the model, e.g. "swotgnn"

    # Encode fold info, val method, and seeds in the run folder name for traceability
    # If it is temporal, run name will look like: exp01_spatialcv_fold0of5_spseed42_valt_swotgnn_s42
    # If it is spatial, run name will look like: exp01_spatialcv_fold0of5_spseed42_valsp0.1_swotgnn_s42
    # valsp 0.1 means 10% of the training lakes are held out for validation
    val_tag = (
        f"_valsp{args.spatial_val_frac}" if args.val_method == "spatial"
        else "_valt"   # temporal
    )
    # Save base name before augmentation; used to group folds under one parent directory
    args.base_run_name = args.run_name
    args.run_name = (
        f"{args.run_name}"
        f"_fold{args.fold_idx}of{args.n_folds}"
        f"_sp{args.spatial_seed}"
        f"{val_tag}"
        f"_{model_slug}"
        f"_s{args.seed}"
    )

    # This script is specifically designed for 1-day-ahead forecasting; enforce this to avoid confusion
    assert cfg["training"]["forecast_horizon"] == 1, (
        "run_spatial_cv_wse1d.py is designed for forecast_horizon=1."
    )

    train(cfg, args)


if __name__ == "__main__":
    main()
