"""
Experiment 01: 1-day-ahead lake WSE forecasting.

Given 30 days of ERA5 reanalysis + SWOT WSE history and the ECMWF IFS
forecast for the next day, predict the next-day Water Surface Elevation
(WSE) for every lake node in the Mekong reservoir graph.

Architecture: SWOT-GNN (InputEncoder -> STBlock x N -> ForecastHead)
Loss:         ObservedMSELoss — MSE at lakes with a SWOT observation only.

Usage:
  python run_training_lake.py \\
    --config     configs/exp01_nextday_wse.yaml \\
    --wse-datacube   /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube  /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --lake-graph  /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --run-name   exp01_v1
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
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.temporal_graph_dataset_lake import (
    build_temporal_dataset_from_lake_datacubes,
    collate_temporal_graph_batch_lake,
)
from models.swot_gnn import SWOTGNN
from training.train import ObservedMSELoss


# ── Training helpers ───────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=1.0):
    """
    Run one forward pass over `loader`.

    When `optimizer` is provided (training mode) the function back-propagates
    and steps the optimiser.  Without it the function runs in eval / no-grad mode.

    Returns the mean per-sample loss over the full loader.
    """
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    total_loss, n_samples = 0.0, 0
    with ctx:
        for data_lists, static_feats, labels, masks in loader:
            bs = len(data_lists)
            if is_train:
                optimizer.zero_grad()

            batch_loss = 0.0
            for b in range(bs):
                # Stack per-timestep node feature matrices → (n_lakes, T, n_feat)
                x = torch.stack([d.x for d in data_lists[b]], dim=1).to(device)
                edge_index = data_lists[b][0].edge_index.to(device)
                static = static_feats[b].to(device)      # (n_lakes, static_dim)

                # Labels/mask: (n_lakes, 1) for forecast_horizon=1 → squeeze to (n_lakes,)
                lab = labels[b].squeeze(-1).to(device)   # (n_lakes,)
                msk = masks[b].squeeze(-1).to(device)    # (n_lakes,)

                # Forward pass — ForecastHead returns (n_lakes,) when horizon=1
                pred = model(x, edge_index, static_features=static)

                loss = criterion(pred, lab, msk)
                if is_train:
                    (loss / bs).backward()
                batch_loss += loss.item()

            if is_train:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += batch_loss
            n_samples  += bs

    return total_loss / max(n_samples, 1)


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

    # ── Build datasets ──────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, norm_stats = build_temporal_dataset_from_lake_datacubes(
        wse_datacube_path        = args.wse_datacube,
        era5_climate_datacube_path = args.era5_datacube,
        ecmwf_forecast_datacube_path = args.ecmwf_datacube,
        static_datacube_path     = args.static_datacube,
        lake_graph_path          = args.lake_graph,
        seq_len                  = cfg["training"]["seq_len"],
        forecast_horizon         = cfg["training"]["forecast_horizon"],
        train_frac               = cfg["training"]["train_frac"],
        val_frac                 = cfg["training"]["val_frac"],
        test_frac                = cfg["training"]["test_frac"],
    )

    # Auto-detect static_dim from the loaded datacube and override config placeholder
    static_dim = int(train_ds.static_features.shape[-1])
    cfg["model"]["static_dim"] = static_dim
    print(f"Static features: {static_dim} attributes per lake (auto-detected)")

    # ── DataLoaders ─────────────────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size = cfg["training"]["batch_size"],
        collate_fn = collate_temporal_graph_batch_lake,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Model ───────────────────────────────────────────────────────────────
    model     = SWOTGNN(**cfg["model"]).to(device)
    criterion = ObservedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    patience  = cfg["training"].get("patience", 20)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    # ── Output directory ────────────────────────────────────────────────────
    run_dir  = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = run_dir / "_best_model_tmp.pt"
    print(f"\nRun: {args.run_name}  →  {run_dir}\n")

    # Freeze a copy of the source config at run time
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
        avg_train = _run_epoch(model, train_loader, criterion, device,
                               optimizer=optimizer, grad_clip=grad_clip)
        avg_val   = _run_epoch(model, val_loader,   criterion, device)
        epoch_secs = time.time() - epoch_start

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        lr = optimizer.param_groups[0]["lr"]
        lr_history.append(lr)
        ts_history.append(datetime.now(timezone.utc).isoformat(timespec="seconds"))

        print(f"Epoch {epoch+1:>3}/{cfg['training']['num_epochs']} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"Best: ep{best_epoch+1} ({best_val_loss:.4f}) | LR: {lr:.2e} | "
              f"Time: {epoch_secs:.1f}s")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), tmp_ckpt)
        else:
            patience_counter += 1

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

    # ── Rename checkpoint to encode best epoch ───────────────────────────────
    final_ckpt_name = f"best_epoch{best_epoch + 1:03d}.pt"
    final_ckpt      = run_dir / final_ckpt_name
    tmp_ckpt.rename(final_ckpt)

    # ── Re-save checkpoint enriched with training history ───────────────────
    state_dict = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    torch.save({
        "model_state_dict": state_dict,
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
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

    # ── Save summary.json ────────────────────────────────────────────────────
    summary = {
        "run_name":         args.run_name,
        "seed":             args.seed,
        "best_epoch":       best_epoch + 1,
        "best_val_loss":    float(best_val_loss),
        "total_epochs_run": len(train_losses),
        "stopped_early":    stopped_early,
        "runtime_seconds":  round(total_training_secs, 1),
        "checkpoint":       final_ckpt_name,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Append to registry.csv ───────────────────────────────────────────────
    registry_path = Path(args.save_dir) / "registry.csv"
    exp_id        = args.run_name.split("_")[0]
    reg_header = [
        "run_name", "exp_id", "seed", "best_epoch", "best_val_loss",
        "total_epochs", "stopped_early", "runtime_min",
        "lr", "seq_len", "st_blocks", "status",
    ]
    reg_row = [
        args.run_name,
        exp_id,
        args.seed,
        best_epoch + 1,
        f"{best_val_loss:.6f}",
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
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses,   label="Val loss")
    ax.axvline(best_epoch + 1, color="gray", linestyle="--",
               label=f"Best (epoch {best_epoch + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (observed lakes)")
    ax.set_title(f"{args.run_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "training_curve.png", dpi=120)
    plt.close(fig)

    # ── Save run_config.yaml ────────────────────────────────────────────────
    run_config = {
        "run_name": args.run_name,
        "exp_id":   args.run_name.split("_")[0],
        "seed":     args.seed,
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
            "best_epoch":          best_epoch + 1,    # 1-indexed
            "best_val_loss":       float(best_val_loss),
            "total_epochs":        len(train_losses),
            "stopped_early":       stopped_early,
            "training_time_secs":  round(total_training_secs, 1),
            "training_time_hms":   f"{h:02d}:{m:02d}:{s:02d}",
        },
    }
    with open(run_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nOutputs saved to {run_dir}/")
    print(f"  {final_ckpt_name:<26} — model weights + full training history")
    print(f"  config.yaml                — frozen copy of source config")
    print(f"  training_log.csv           — per-epoch train/val loss, lr, timestamp")
    print(f"  summary.json               — lightweight result summary")
    print(f"  training_curve.png         — loss plot")
    print(f"  run_config.yaml            — hyperparameters + data paths + result summary")
    print(f"Registry row appended → {registry_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="1-day-ahead lake WSE forecasting (SWOT-GNN)"
    )
    parser.add_argument(
        "--config", default=str(script_dir / "configs" / "default.yaml"),
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument("--wse-datacube",    required=True,
                        help="swot_lake_wse_datacube_*.nc")
    parser.add_argument("--era5-datacube",   required=True,
                        help="swot_lake_era5_climate_datacube.nc")
    parser.add_argument("--ecmwf-datacube",  required=True,
                        help="swot_lake_ecmwf_forecast_datacube.nc")
    parser.add_argument("--static-datacube", required=True,
                        help="swot_lake_static_datacube.nc")
    parser.add_argument("--lake-graph",      required=True,
                        help="GRIT PLD lake graph CSV")
    parser.add_argument("--save-dir",  default="checkpoints",
                        help="Root directory for all run outputs")
    parser.add_argument("--run-name",  required=True,
                        help="Versioned run subfolder name (e.g. exp03_mekong_wse1d_era5ifs_gritv06_202312_202602_swotgnn_s42)")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=None,
        help="Override num_epochs from config (e.g. --num-epochs 1 for a smoke test)",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Override early-stop patience from config",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed — overrides the seed in the config file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides — useful for quick smoke tests without editing the YAML
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.patience is not None:
        cfg["training"]["patience"] = args.patience

    # Seed priority: CLI --seed > config seed > fallback 42
    if args.seed is None:
        args.seed = cfg.get("seed", 42)

    # Bake seed into the run name so the folder is always self-describing
    args.run_name = f"{args.run_name}_s{args.seed}"

    assert cfg["training"]["forecast_horizon"] == 1, (
        "run_lake_exp01.py is designed for forecast_horizon=1. "
        "Use the lake_datacube config for multi-step forecasting."
    )

    train(cfg, args)


if __name__ == "__main__":
    main()
