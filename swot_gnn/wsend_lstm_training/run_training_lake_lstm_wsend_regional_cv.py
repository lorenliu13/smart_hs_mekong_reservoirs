"""
Regional spatial cross-validation for multi-day lake WSE forecasting
using an LSTM-only baseline (no graph).

Architecture: InputEncoder → LSTMBlock × N → ForecastHead.
Each lake is processed independently; no spatial message passing.
Use results to benchmark the contribution of graph connectivity in SWOT-GNN.

Lakes are split into 5 geographic regions (see data/regional_cv.py).  One
region is held out as the test set; the remaining four are used for training
and validation.  Run one fold at a time and submit each as a separate job.

Region → fold index:
    0: Upper Mekong + Northern Highlands          (~21%)
    1: Red River Basin + Pearl River Tributaries  (~13%)
    2: Vietnam Coastal Basins + Mekong Delta      (~20%)
    3: Khorat Plateau                             (~25%)
    4: Tonle Sap Basin + 3S Basin                 (~19%)

Multi-day forecasting strategy:
    A training sample is valid if at least one SWOT observation exists for any
    lake on any forecast day within [init_date, init_date + forecast_horizon - 1].
    The model predicts WSE at all forecast_horizon lead days simultaneously
    (direct multi-step).  Loss is computed over all (lake, lead_day) pairs where
    obs_mask=1, averaged across the batch (ObservedMSELossMultiStep).

Key differences from wsend_training/run_training_lake_wsend_regional_cv.py:
  - Imports _run_epoch_lstm_nd from training.train_lstm_nd (no edge_index / batch).
  - Model call omits edge_index and batch vector.
  - Registry CSV is registry_lstm_wsend_regional_cv.csv.

Usage:
  python run_training_lake_lstm_wsend_regional_cv.py \\
    --config          configs/wsend/lstm_baseline_wsend.yaml \\
    --wse-datacube    /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube   /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube  /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --lake-graph      /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --run-name        lstm_baseline_wsend_regionalcv \\
    --fold-idx        0

Launch all folds (bash example):
  for i in 0 1 2 3 4; do
    python run_training_lake_lstm_wsend_regional_cv.py ... --fold-idx $i &
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

from data.regional_cv import (
    build_regional_cv_fold,
    N_REGIONAL_FOLDS,
    REGION_NAMES,
)
from data.temporal_graph_dataset_lake import collate_temporal_graph_batch_lake
from models.registry import MODEL_REGISTRY
from training.train_lstm_nd import _run_epoch_lstm_nd


# ── Main training routine ──────────────────────────────────────────────────────

def train(cfg, args):
    device = torch.device(args.device)

    # Set random seeds for reproducibility
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    forecast_horizon = cfg["training"]["forecast_horizon"]

    # ── Build regional CV datasets ──────────────────────────────────────────
    train_ds, val_ds, test_ds, norm_stats = build_regional_cv_fold(
        wse_datacube_path            = args.wse_datacube,
        era5_climate_datacube_path   = args.era5_datacube,
        ecmwf_forecast_datacube_path = args.ecmwf_datacube,
        static_datacube_path         = args.static_datacube,
        lake_graph_path              = args.lake_graph,
        seq_len                      = cfg["training"]["seq_len"],
        forecast_horizon             = forecast_horizon,
        fold_idx                     = args.fold_idx,
        val_frac                     = cfg["training"].get("val_frac", 0.15),
        val_method                   = args.val_method,
        spatial_val_frac             = args.spatial_val_frac,
        spatial_val_seed             = args.spatial_val_seed,
        hybas_col                    = args.hybas_col,
        require_obs_on_any_forecast_day = True,
    )

    # Auto-detect static_dim from the loaded datacube and override config placeholder
    static_dim = int(train_ds.static_features.shape[-1])
    cfg["model"]["static_dim"] = static_dim
    print(f"Static features: {static_dim} attributes per lake (auto-detected)")

    # Spatial masks gate the loss to the relevant lake subset in each phase
    train_spatial_mask = train_ds.spatial_mask
    val_spatial_mask   = val_ds.spatial_mask
    test_spatial_mask  = test_ds.spatial_mask

    # ── DataLoaders ─────────────────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size = cfg["training"]["batch_size"],
        collate_fn = collate_temporal_graph_batch_lake,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = torch.utils.data.DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Model ───────────────────────────────────────────────────────────────
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

    # ── Output directory ────────────────────────────────────────────────────
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
        avg_train = _run_epoch_lstm_nd(model, train_loader, criterion, device,
                                       optimizer=optimizer, grad_clip=grad_clip,
                                       spatial_mask=train_spatial_mask)
        avg_val   = _run_epoch_lstm_nd(model, val_loader,   criterion, device,
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

    # ── Rename checkpoint ───────────────────────────────────────────────────
    final_ckpt_name = f"best_epoch{best_epoch + 1:03d}.pt"
    final_ckpt      = run_dir / final_ckpt_name
    tmp_ckpt.rename(final_ckpt)

    # ── Evaluate on held-out test region ─────────────────────────────────────
    print(f"\nEvaluating on held-out test region [{REGION_NAMES[args.fold_idx]}] …")
    state_dict = torch.load(final_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    test_loss = _run_epoch_lstm_nd(model, test_loader, criterion, device,
                                   spatial_mask=test_spatial_mask)
    print(f"Test loss (held-out region): {test_loss:.4f}")

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
        "fold_idx":          args.fold_idx,
        "region_name":       REGION_NAMES[args.fold_idx],
        "forecast_horizon":  forecast_horizon,
        "n_test_lakes":      int(norm_stats["n_test_lakes"]),
        "n_unassigned":      int(norm_stats["n_unassigned"]),
        "test_loss":         float(test_loss),
        "best_val_loss":     float(best_val_loss),
    }
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ── Append to registry.csv ───────────────────────────────────────────────
    registry_path = Path(args.save_dir) / "registry_lstm_wsend_regional_cv.csv"
    exp_id        = args.run_name.split("_")[0]
    reg_header = [
        "run_name", "exp_id", "seed", "fold_idx", "region_name",
        "forecast_horizon", "n_train_lakes", "n_test_lakes",
        "best_epoch", "best_val_loss", "test_loss",
        "total_epochs", "stopped_early", "runtime_min",
        "lr", "seq_len", "st_blocks", "status",
    ]
    reg_row = [
        args.run_name,
        exp_id,
        args.seed,
        args.fold_idx,
        REGION_NAMES[args.fold_idx],
        forecast_horizon,
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
               label=f"Test loss (held-out region): {test_loss:.4f}")
    ax.axvline(best_epoch + 1, color="gray", linestyle="--",
               label=f"Best (epoch {best_epoch + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss — {forecast_horizon}-day forecast (observed lakes)")
    ax.set_title(
        f"{args.run_name}  —  fold {args.fold_idx} [{REGION_NAMES[args.fold_idx]}]"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "training_curve.png", dpi=120)
    plt.close(fig)

    # ── Save run_config.yaml ────────────────────────────────────────────────
    run_config = {
        "run_name": args.run_name,
        "exp_id":   args.run_name.split("_")[0],
        "seed":     args.seed,
        "regional_cv": {
            "fold_idx":      args.fold_idx,
            "region_name":   REGION_NAMES[args.fold_idx],
            "val_method":    args.val_method,
            "hybas_col":     args.hybas_col,
            "n_train_lakes": int(norm_stats["n_train_lakes"]),
            "n_test_lakes":  int(norm_stats["n_test_lakes"]),
            "n_unassigned":  int(norm_stats["n_unassigned"]),
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
    print(f"  test_metrics.json          — held-out region test loss for this fold")
    print(f"  training_curve.png         — loss plot with test loss reference line")
    print(f"  run_config.yaml            — hyperparameters + data paths + full result summary")
    print(f"Registry row appended → {registry_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Regional CV — multi-day lake WSE forecasting (LSTM-only baseline)"
    )
    parser.add_argument(
        "--config", default=str(script_dir / "configs" / "wsend" / "lstm_baseline_wsend.yaml"),
        help="Path to YAML config (default: configs/wsend/lstm_baseline_wsend.yaml)",
    )
    parser.add_argument("--wse-datacube",    required=True)
    parser.add_argument("--era5-datacube",   required=True)
    parser.add_argument("--ecmwf-datacube",  required=True)
    parser.add_argument("--static-datacube", required=True)
    parser.add_argument("--lake-graph",      required=True,
                        help="Path to GRIT PLD lake graph CSV (must contain "
                             "lake_id and hybasin_level_4 columns)")
    parser.add_argument(
        "--hybas-col", default="hybasin_level_4",
        help="Column in the lake graph CSV holding the HYBAS Level-4 sub-basin ID "
             "(default: hybasin_level_4)",
    )
    parser.add_argument("--save-dir",  default="checkpoints",
                        help="Root directory for all run outputs")
    parser.add_argument("--run-name",  required=True,
                        help="Base run name; fold, horizon, and model slug are appended automatically")
    parser.add_argument("--fold-idx",  type=int, required=True,
                        help="Which region to hold out as the test set (0–4)")
    parser.add_argument(
        "--val-method", default="temporal", choices=["temporal", "spatial"],
        help=(
            "Validation strategy. "
            "'temporal' (default): hold out last val_frac of dates from train-region lakes. "
            "'spatial': use all dates; hold out spatial_val_frac of train-region lakes for val."
        ),
    )
    parser.add_argument(
        "--spatial-val-frac", type=float, default=0.1,
        help="Fraction of train-region lakes used as spatial val (only with --val-method spatial, default 0.1)",
    )
    parser.add_argument(
        "--spatial-val-seed", type=int, default=43,
        help="RNG seed for the spatial val lake draw (default 43)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--patience",   type=int, default=None)
    parser.add_argument("--seed",       type=int, default=None,
                        help="Model training seed (overrides config)")
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
        f"run_training_lake_lstm_wsend_regional_cv.py is designed for forecast_horizon > 1, "
        f"got {forecast_horizon}. Use wse1d_training/ scripts for 1-day-ahead forecasting."
    )

    model_type = cfg["model"].get("model_type", "LSTMBaselineMultiStep")
    model_slug = MODEL_REGISTRY[model_type].slug

    # Encode fold, horizon, val method, and seed in the run folder name.
    # Example: lstm_baseline_wsend_regionalcv_fold0of5_h5_valt_lstm_nd_s42
    val_tag = (
        f"_valsp{args.spatial_val_frac}" if args.val_method == "spatial"
        else "_valt"
    )
    args.base_run_name = args.run_name
    args.run_name = (
        f"{args.run_name}"
        f"_fold{args.fold_idx}of{N_REGIONAL_FOLDS}"
        f"_h{forecast_horizon}"
        f"{val_tag}"
        f"_{model_slug}"
        f"_s{args.seed}"
    )

    train(cfg, args)


if __name__ == "__main__":
    main()