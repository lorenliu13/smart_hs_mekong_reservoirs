"""
Run SWOT-GNN training for 1-day-ahead WSE forecasting.
Usage:
  Segment-based: python run_training.py --dynamic-datacube ... --static-datacube ... --segment-based --segment-mapping segment_reach_mapping.csv
  Reach-based:   python run_training.py --grit-path ... --dynamic-datacube ... --static-datacube ...
"""
import argparse
import csv
from pathlib import Path
import yaml
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for subprocess / Colab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.temporal_graph_dataset import (
    build_temporal_dataset_from_datacubes,
    build_temporal_dataset_from_datacubes_segment_based,
    save_dataset_cache,
    load_dataset_cache,
)
from models.swot_gnn import SWOTGNN
from training.train import train_swot_gnn


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    # Path to YAML config; defaults to datacube.yaml when datacube inputs are provided
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--config", default=str(script_dir / "configs" / "default.yaml"))
    # GRIT reach CSV: required in reach-based mode to define the river network graph
    parser.add_argument("--grit-path", help="Path to GRIT reach CSV (required for reach-based mode)")
    # Dynamic NetCDF datacube: time-varying features (WSE, climate) per segment/reach × time
    parser.add_argument("--dynamic-datacube", help="Path to dynamic NetCDF datacube (from training_data_processing_20260202.py)")
    # Static NetCDF datacube: time-invariant features (33 attributes) per segment/reach
    parser.add_argument("--static-datacube", help="Path to static NetCDF datacube")
    # Flag to switch from reach-based to segment-based graph construction
    parser.add_argument("--segment-based", action="store_true", help="Use segment-based datacubes (from training_data_processing_segment_based_20260222.py)")
    # Segment downstream-area CSV/shapefile: defines graph topology for segment mode
    parser.add_argument("--segment-darea-path", help="Path to segment darea CSV/shapefile with downstream links (required for --segment-based)")
    # Base output directory; each run is saved in a versioned subfolder inside it
    parser.add_argument("--save-dir", default="checkpoints")
    # Run name used as the subfolder under --save-dir (e.g. model_v1_segment_default).
    # Each run folder contains: best_model.pt, losses.csv, training_curve.png, run_config.yaml
    parser.add_argument("--run-name", default="model_v1",
                        help="Versioned name for this run (e.g. model_v1_segment_default). "
                             "Creates {save-dir}/{run-name}/ with all outputs.")
    # Automatically use GPU if available, otherwise fall back to CPU
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Optional path to a .npz dataset cache. If the file exists, skip NetCDF loading and
    # normalization entirely. If it does not exist, build the datasets normally then save
    # the cache so future runs can use it.
    parser.add_argument("--cache-path", default=None,
                        help="Path to .npz dataset cache (created on first run, reused on subsequent runs)")
    args = parser.parse_args()

    # --- Validate Required Arguments ---
    use_datacube = bool(args.dynamic_datacube and args.static_datacube)
    use_segment = bool(args.segment_based)
    if not use_datacube:
        raise ValueError("--dynamic-datacube and --static-datacube are required")
    if use_segment and not args.segment_darea_path:
        raise ValueError("--segment-based requires --segment-darea-path (path to segment darea CSV/shapefile)")
    if not use_segment and not args.grit_path:
        raise ValueError("reach-based mode requires --grit-path (path to GRIT reach CSV)")

    # Override config path: datacube inputs warrant their own config (feat_dim, static_dim, etc.)
    if use_datacube and args.config == str(script_dir / "configs" / "default.yaml"):
        args.config = str(script_dir / "configs" / "datacube.yaml")

    # --- Load YAML Config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Extract train/val/test split fractions from config (defaults: 70/15/15)
    train_frac = cfg["training"].get("train_frac", 0.7)
    val_frac = cfg["training"].get("val_frac", 0.15)
    test_frac = cfg["training"].get("test_frac", 0.15)

    # --- Build Temporal Graph Datasets ---
    cache_path = Path(args.cache_path) if args.cache_path else None

    if cache_path and cache_path.exists():
        # Fast path: load pre-computed normalized arrays from .npz cache
        print(f"Loading dataset from cache: {cache_path}")
        train_ds, val_ds, test_ds, norm_stats = load_dataset_cache(
            cache_path, seq_len=cfg["training"]["seq_len"]
        )
    elif use_segment:
        # Segment-based: graph topology from segment darea CSV; returns norm_stats for diagnostics
        train_ds, val_ds, test_ds, norm_stats = build_temporal_dataset_from_datacubes_segment_based(
            dynamic_datacube_path=args.dynamic_datacube,
            static_datacube_path=args.static_datacube,
            segment_darea_path=args.segment_darea_path,
            start_date="2023-10-01",
            end_date="2025-12-01",
            seq_len=cfg["training"]["seq_len"],
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        # Report which dynamic feature indices received log1p+z-score vs z-score-only normalization
        print(f"Normalization applied — dynamic indices (log1p+z): {norm_stats['log1p_dynamic_indices']}, "
              f"(z only): {[i for i in norm_stats['zscore_dynamic_indices'] if i not in norm_stats['log1p_dynamic_indices']]}")
        if cache_path:
            save_dataset_cache(cache_path, train_ds, val_ds, test_ds, norm_stats)
    else:
        # Reach-based: graph topology from GRIT reach CSV
        train_ds, val_ds, test_ds = build_temporal_dataset_from_datacubes(
            dynamic_datacube_path=args.dynamic_datacube,
            static_datacube_path=args.static_datacube,
            grit_reach_path=args.grit_path,
            start_date="2023-10-01",
            end_date="2025-12-01",
            seq_len=cfg["training"]["seq_len"],
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )

    def collate_fn(x):
        # Collate_fn controls how individual samples are combined into a batch. 
        # Given N individual samples from x, tell the code how to merge them into one batch
        # Each dataset sample is a 4-tuple: (data_list, static, labels, mask)
        # data_list: list of PyG Data objects (one per timestep in the sequence)
        # static: (n_nodes, static_dim) static node features
        # labels: (n_nodes,) target WSE on the forecast day
        # mask: (n_nodes,) binary — 1 where SWOT observed, 0 where unobserved
        return (
            [s[0] for s in x],          # list of data_lists (one per batch item)
            torch.stack([s[1] for s in x]),  # (batch, n_nodes, static_dim)
            torch.stack([s[2] for s in x]),  # (batch, n_nodes)
            torch.stack([s[3] for s in x]),  # (batch, n_nodes)
        )

    # --- Fit Label Scaler on Training Set ---
    # Collect observed WSE label values from up to the first 100 training samples
    # (obs mask = 1 nodes only) to fit a StandardScaler for optional rescaling.
    all_labels = []
    for i in range(min(100, len(train_ds))):
        _, _, labels, mask = train_ds[i]
        all_labels.extend(labels[mask > 0].numpy().tolist())
    scaler = StandardScaler()
    if all_labels:
        scaler.fit([[v] for v in all_labels])

    # --- DataLoaders ---
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,           # Shuffle training samples each epoch
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,          # Keep validation order deterministic
        collate_fn=collate_fn,
    )

    # --- Versioned run folder ---
    # All outputs for this run go into {save_dir}/{run_name}/
    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "best_model.pt"
    print(f"\nRun: {args.run_name}  →  {run_dir}\n")

    # --- Model & Training ---
    # Instantiate SWOT-GNN from config: InputEncoder → STBlock × N → Readout
    model = SWOTGNN(**cfg["model"])

    result = train_swot_gnn(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=torch.device(args.device),
        scaler=scaler,
        lr=cfg["training"]["lr"],
        num_epochs=cfg["training"]["num_epochs"],
        save_path=save_path,
        patience=cfg["training"].get("patience", 25),
        grad_clip=cfg["training"].get("grad_clip", 1.0),
    )

    # --- Re-save checkpoint enriched with loss history ---
    # train_swot_gnn saves only model.state_dict() mid-training; reload and embed result.
    state_dict = torch.load(save_path, map_location="cpu")
    torch.save({
        "model_state_dict": state_dict,
        "train_losses":  result["train_losses"],
        "val_losses":    result.get("val_losses", []),
        "best_epoch":    result["best_epoch"],
        "best_loss":     result["best_loss"],
        "stopped_early": result["stopped_early"],
    }, save_path)

    # --- Save losses.csv ---
    train_losses = result["train_losses"]
    val_losses   = result.get("val_losses", [])
    losses_path  = run_dir / "losses.csv"
    with open(losses_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"] + (["val_loss"] if val_losses else []))
        for i, tl in enumerate(train_losses):
            row = [i + 1, f"{tl:.8f}"]
            if val_losses:
                row.append(f"{val_losses[i]:.8f}")
            writer.writerow(row)

    # --- Save training_curve.png ---
    best_epoch  = result["best_epoch"]
    curve_path  = run_dir / "training_curve.png"
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss")
    if val_losses:
        ax.plot(epochs, val_losses, label="Val loss")
    ax.axvline(best_epoch + 1, color="gray", linestyle="--", linewidth=1,
               label=f"Best (epoch {best_epoch + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (observed nodes)")
    ax.set_title(f"Training Curves — {args.run_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)

    # --- Save run_config.yaml ---
    run_config = {
        "run_name": args.run_name,
        "data": {
            "dynamic_datacube":   str(args.dynamic_datacube),
            "static_datacube":    str(args.static_datacube),
            "segment_based":      args.segment_based,
            "segment_darea_path": str(args.segment_darea_path) if args.segment_darea_path else None,
            "cache_path":         str(args.cache_path) if args.cache_path else None,
        },
        "model":    cfg["model"],
        "training": cfg["training"],
        "result": {
            "best_epoch":    result["best_epoch"] + 1,   # 1-indexed for readability
            "best_loss":     float(result["best_loss"]),
            "total_epochs":  len(train_losses),
            "stopped_early": result["stopped_early"],
        },
    }
    with open(run_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

    print(f"\nOutputs saved to {run_dir}/")
    print(f"  best_model.pt      — model weights + loss history")
    print(f"  losses.csv         — per-epoch train/val loss")
    print(f"  training_curve.png — loss plot")
    print(f"  run_config.yaml    — hyperparameters + data paths + result summary")


if __name__ == "__main__":
    main()
