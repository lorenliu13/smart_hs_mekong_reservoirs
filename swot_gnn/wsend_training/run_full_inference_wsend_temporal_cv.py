"""
Full inference script for the GNN temporal CV model.

Unlike run_inference_wsend_temporal_cv.py (which only runs on init dates
that have at least one SWOT observation in the forecast window), this script
produces predictions for EVERY valid init date in the dataset regardless of
whether any observation is available.

Best-model selection:
    Scans all fold directories under <save-dir>/<run-name>/fold_*/run_config.yaml,
    reads the best_val_loss from each, and loads the checkpoint with the lowest
    validation loss.  Use --fold-idx to override and pin a specific fold.

Output:
    <save-dir>/<run-name>/full_inference/
        full_predictions.csv          — all lakes × all init_dates × all lead_days
        full_inference_config.json    — provenance (fold used, date range, n_lakes …)

CSV columns:
    lake_id, fold_used, init_date, forecast_date, lead_day,
    pred_norm, pred_m, pred_std_norm, pred_std_m,
    target_norm, target_m,   ← NaN when no observation is available
    obs_available,           ← bool flag
    days_since_last_swot     ← NaN when no prior SWOT obs exists for that lake

Usage:
  python run_full_inference_wsend_temporal_cv.py \\
    --config          configs/wsend/exp08_wsend.yaml \\
    --wse-datacube    /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube   /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube  /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --wse-stats-csv   /path/to/lake_wse_norm_stats.csv \\
    --lake-graph      /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --save-dir        checkpoints \\
    --run-name        exp08_wsend_temporalcv \\
    --seed            42
"""
import argparse
import json
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.temporal_cv import (
    build_temporal_cv_fold,
    N_TEMPORAL_FOLDS,
    TEMPORAL_FOLD_DATES,
)
from models.registry import MODEL_REGISTRY


# ── Dataset builder (no obs filter) ───────────────────────────────────────────

def _build_full_dataset(args, cfg_training, fold_idx: int):
    """
    Rebuild the same temporal CV fold used during training, but with
    require_obs_on_any_forecast_day=False so that every valid init date
    (i.e. one with sufficient ERA5 history) is included.

    Returns (full_ds, norm_stats) where full_ds covers ALL init dates in the
    fold's full date range (train + val + test windows combined).
    """
    from data.temporal_cv import (
        TEMPORAL_FOLD_DATES, N_TEMPORAL_FOLDS, _LOG1P_DYN, _ZSCORE_DYN,
        ECMWF_CLIMATE_VARS, SWOT_DIM,
    )
    from data.temporal_graph_dataset_lake import (
        assemble_lake_features_from_datacubes,
        TemporalGraphDatasetLake,
        ECMWF_CLIMATE_VARS,
        SWOT_DIM,
    )
    from data.graph_builder import build_graph_from_lake_graph

    seq_len          = cfg_training["seq_len"]
    forecast_horizon = cfg_training["forecast_horizon"]

    fold_def    = TEMPORAL_FOLD_DATES[fold_idx]
    train_start = pd.Timestamp(fold_def["train_start"])
    train_end   = pd.Timestamp(fold_def["train_end"])
    test_end    = pd.Timestamp(fold_def["test_end"])

    (
        era5_dynamic,
        ecmwf_forecast,
        static_features,
        wse_labels,
        obs_mask,
        lake_ids_out,
        era5_dates,
        ecmwf_init_dates,
    ) = assemble_lake_features_from_datacubes(
        wse_datacube_path=args.wse_datacube,
        era5_climate_datacube_path=args.era5_datacube,
        ecmwf_forecast_datacube_path=args.ecmwf_datacube,
        static_datacube_path=args.static_datacube,
    )

    n_lakes_total = len(lake_ids_out)

    edge_index, _, _, _ = build_graph_from_lake_graph(
        lake_graph_csv=args.lake_graph,
        lake_ids=lake_ids_out,
    )

    era5_date_to_idx = {d: i for i, d in enumerate(era5_dates)}

    # Collect ALL init dates in the fold's full date range
    all_valid: list[int] = []
    for j, init_date in enumerate(ecmwf_init_dates):
        last_hist_day = init_date - pd.Timedelta(days=1)
        era5_idx = era5_date_to_idx.get(last_hist_day)
        if era5_idx is None or era5_idx < seq_len - 1:
            continue
        if train_start <= init_date <= test_end:
            all_valid.append(j)

    all_valid_arr = np.array(all_valid, dtype=np.int64)
    print(f"Full inference: {len(all_valid_arr)} init dates "
          f"({train_start.date()} → {test_end.date()}, no obs filter)")

    # ── Normalization (identical to build_temporal_cv_fold) ───────────────────
    era5_dynamic   = era5_dynamic.copy()
    ecmwf_forecast = ecmwf_forecast.copy()

    for i in _LOG1P_DYN:
        era5_dynamic[:, :, i] = np.log1p(np.clip(era5_dynamic[:, :, i], 0, None))

    norm_start = train_start - pd.Timedelta(days=seq_len)
    norm_era5_mask = np.array(
        [norm_start <= d <= train_end for d in era5_dates], dtype=bool
    )
    norm_era5_slice = era5_dynamic[:, norm_era5_mask, :]

    n_dyn    = era5_dynamic.shape[-1]
    dyn_mean = np.zeros(n_dyn, dtype=np.float32)
    dyn_std  = np.ones(n_dyn,  dtype=np.float32)

    for i in _ZSCORE_DYN:
        vals        = norm_era5_slice[:, :, i].ravel()
        dyn_mean[i] = float(vals.mean())
        dyn_std[i]  = float(vals.std()) + 1e-8

    for i in _ZSCORE_DYN:
        era5_dynamic[:, :, i] = (era5_dynamic[:, :, i] - dyn_mean[i]) / dyn_std[i]

    ecmwf_log1p_indices  = [k for k, _ in enumerate(ECMWF_CLIMATE_VARS) if (SWOT_DIM + k) in _LOG1P_DYN]
    ecmwf_zscore_indices = [k for k, _ in enumerate(ECMWF_CLIMATE_VARS) if (SWOT_DIM + k) in _ZSCORE_DYN]

    for k in ecmwf_log1p_indices:
        ecmwf_forecast[:, :, :, k] = np.log1p(np.clip(ecmwf_forecast[:, :, :, k], 0, None))
    for k in ecmwf_zscore_indices:
        era5_idx_k = SWOT_DIM + k
        ecmwf_forecast[:, :, :, k] = (
            ecmwf_forecast[:, :, :, k] - dyn_mean[era5_idx_k]
        ) / dyn_std[era5_idx_k]

    stat_mean = static_features.mean(axis=0).astype(np.float32)
    stat_std  = static_features.std(axis=0).astype(np.float32) + 1e-8
    static_features = (static_features - stat_mean) / stat_std

    norm_stats = {
        "fold_idx":    fold_idx,
        "train_start": str(train_start.date()),
        "train_end":   str(train_end.date()),
        "test_start":  fold_def["test_start"],
        "test_end":    fold_def["test_end"],
        "n_all_dates": len(all_valid_arr),
        "n_lakes":     n_lakes_total,
    }

    full_ds = TemporalGraphDatasetLake(
        era5_dynamic=era5_dynamic,
        ecmwf_forecast=ecmwf_forecast,
        static_features=static_features,
        edge_index=edge_index,
        era5_dates=era5_dates,
        ecmwf_init_dates=ecmwf_init_dates,
        wse_labels=wse_labels,
        obs_mask=obs_mask,
        lake_ids=lake_ids_out,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        indices=all_valid_arr,
    )
    full_ds.spatial_mask = torch.ones(n_lakes_total, dtype=torch.float32)

    return full_ds, norm_stats


# ── Inference (all lakes, all lead days, no mask filter) ──────────────────────

def run_full_inference(ds, model, device, fold_used: int) -> pd.DataFrame:
    """
    Run GNN over every sample in ds.  Outputs a row for EVERY lake × lead_day,
    regardless of whether an observation is available.

    Columns:
        lake_id, fold_used, init_date, forecast_date, lead_day,
        pred_norm, pred_std_norm, target_norm, obs_available,
        days_since_last_swot
    """
    # Precompute running "last obs era5 index" per lake at every era5 date.
    obs_mask_np = ds.obs_mask                          # (n_lakes, n_era5_dates)
    n_lk, n_dt  = obs_mask_np.shape
    last_obs_era5_idx = np.full((n_lk, n_dt), -1, dtype=np.int32)
    last_obs_era5_idx[:, 0] = np.where(obs_mask_np[:, 0] > 0, 0, -1)
    for _t in range(1, n_dt):
        last_obs_era5_idx[:, _t] = np.where(
            obs_mask_np[:, _t] > 0, _t, last_obs_era5_idx[:, _t - 1]
        )

    records = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(ds)):
            data_list, static, labels, mask = ds[idx]

            x          = torch.stack([d.x for d in data_list], dim=1).to(device)
            edge_index = data_list[0].edge_index.to(device)
            static_t   = static.to(device)

            pred_out = model(x, edge_index, static_features=static_t)

            if isinstance(pred_out, tuple):
                mean_cpu = pred_out[0].cpu().numpy()
                std_cpu  = pred_out[1].clamp(-6, 6).exp().cpu().numpy()
            else:
                mean_cpu = pred_out.cpu().numpy()
                std_cpu  = None

            labels_np = labels.numpy()
            mask_np   = mask.numpy()
            if labels_np.ndim == 1:
                labels_np = labels_np[:, np.newaxis]
                mask_np   = mask_np[:, np.newaxis]
            if mean_cpu.ndim == 1:
                mean_cpu = mean_cpu[:, np.newaxis]
            if std_cpu is not None and std_cpu.ndim == 1:
                std_cpu = std_cpu[:, np.newaxis]

            j              = int(ds.valid_starts[idx])
            init_date      = pd.Timestamp(ds.ecmwf_init_dates[j])
            last_hist_day  = init_date - pd.Timedelta(days=1)
            last_hist_idx  = ds.era5_date_to_idx.get(last_hist_day)
            horizon        = mean_cpu.shape[1]

            last_obs_per_lake = (
                last_obs_era5_idx[:, last_hist_idx]
                if last_hist_idx is not None
                else np.full(n_lk, -1, dtype=np.int32)
            )

            for d in range(horizon):
                forecast_date = init_date + pd.Timedelta(days=d)
                for lake_i, lake_id in enumerate(ds.lake_ids):
                    has_obs = bool(mask_np[lake_i, d] > 0)
                    loi     = int(last_obs_per_lake[lake_i])
                    days_since_swot = (
                        (forecast_date - ds.era5_dates[loi]).days
                        if loi >= 0 else float("nan")
                    )
                    records.append({
                        "lake_id"             : int(lake_id),
                        "fold_used"           : fold_used,
                        "init_date"           : init_date,
                        "forecast_date"       : forecast_date,
                        "lead_day"            : d,
                        "pred_norm"           : float(mean_cpu[lake_i, d]),
                        "pred_std_norm"       : float(std_cpu[lake_i, d]) if std_cpu is not None else float("nan"),
                        "target_norm"         : float(labels_np[lake_i, d]) if has_obs else float("nan"),
                        "obs_available"       : has_obs,
                        "days_since_last_swot": days_since_swot,
                    })

    df = pd.DataFrame(records)
    n_dates = df["init_date"].nunique() if len(df) > 0 else 0
    n_obs   = int(df["obs_available"].sum()) if len(df) > 0 else 0
    print(f"  Full inference: {len(df):>9} total rows  |  "
          f"{n_dates} init dates  |  {n_obs} rows with obs")
    return df


# ── Denormalize ────────────────────────────────────────────────────────────────

def denormalize(df: pd.DataFrame, lake_stats: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(lake_stats[["lake_id", "lake_mean", "lake_std"]],
                  on="lake_id", how="left")
    std_eps = df["lake_std"].fillna(1.0) + 1e-8
    mean_   = df["lake_mean"].fillna(0.0)
    df["pred_m"]   = df["pred_norm"] * std_eps + mean_
    df["target_m"] = df["target_norm"] * std_eps + mean_
    if "pred_std_norm" in df.columns:
        df["pred_std_m"] = df["pred_std_norm"] * std_eps
    return df


# ── Best-fold selection ────────────────────────────────────────────────────────

def find_best_fold(save_dir: Path, base_run_name: str) -> tuple[int, Path, dict]:
    """
    Scan fold_0 … fold_{N-1} under save_dir/base_run_name, read best_val_loss
    from run_config.yaml in each, and return (best_fold_idx, run_dir, run_config).
    """
    best_fold_idx  = None
    best_val_loss  = float("inf")
    best_run_dir   = None
    best_run_cfg   = None

    for fold_idx in range(N_TEMPORAL_FOLDS):
        run_dir  = save_dir / base_run_name / f"fold_{fold_idx}"
        cfg_path = run_dir / "run_config.yaml"
        if not cfg_path.exists():
            print(f"  fold_{fold_idx}: run_config.yaml not found — skipping")
            continue
        with open(cfg_path) as f:
            rc = yaml.safe_load(f)
        val_loss = rc.get("result", {}).get("best_val_loss", float("inf"))
        print(f"  fold_{fold_idx}: best_val_loss = {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_fold_idx  = fold_idx
            best_run_dir   = run_dir
            best_run_cfg   = rc

    if best_fold_idx is None:
        raise FileNotFoundError(
            f"No completed fold found under {save_dir / base_run_name}. "
            "Run training first."
        )
    print(f"  → Best fold: fold_{best_fold_idx}  (val_loss={best_val_loss:.6f})")
    return best_fold_idx, best_run_dir, best_run_cfg


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Full inference: run the best GNN checkpoint on ALL init dates, "
            "regardless of SWOT obs availability."
        )
    )
    parser.add_argument("--wse-datacube",    required=True)
    parser.add_argument("--era5-datacube",   required=True)
    parser.add_argument("--ecmwf-datacube",  required=True)
    parser.add_argument("--static-datacube", required=True)
    parser.add_argument("--wse-stats-csv",   required=True)
    parser.add_argument("--lake-graph",      required=True)
    parser.add_argument("--config",          required=True)
    parser.add_argument("--save-dir",  default="checkpoints")
    parser.add_argument("--run-name",  required=True)
    parser.add_argument(
        "--fold-idx", type=int, default=None,
        help="Pin a specific fold (0/1/2). Omit to auto-select the best fold.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is None:
        args.seed = cfg.get("seed", 42)

    forecast_horizon = cfg["training"]["forecast_horizon"]
    save_dir         = Path(args.save_dir)

    # ── Select fold ───────────────────────────────────────────────────────────
    print("\nScanning fold checkpoints ...")
    if args.fold_idx is not None:
        if not 0 <= args.fold_idx < N_TEMPORAL_FOLDS:
            parser.error(f"--fold-idx must be 0, 1, or 2 (got {args.fold_idx})")
        fold_idx = args.fold_idx
        run_dir  = save_dir / args.run_name / f"fold_{fold_idx}"
        cfg_path = run_dir / "run_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"run_config.yaml not found: {cfg_path}")
        with open(cfg_path) as f:
            run_config = yaml.safe_load(f)
        val_loss = run_config.get("result", {}).get("best_val_loss", float("nan"))
        print(f"  Pinned fold_{fold_idx}: best_val_loss = {val_loss:.6f}")
    else:
        fold_idx, run_dir, run_config = find_best_fold(save_dir, args.run_name)

    fold_def     = TEMPORAL_FOLD_DATES[fold_idx]
    cfg_model    = run_config["model"]
    cfg_training = run_config["training"]

    print(f"\nUsing fold {fold_idx} — "
          f"train {fold_def['train_start']} → {fold_def['train_end']}  |  "
          f"test  {fold_def['test_start']} → {fold_def['test_end']}")
    print(f"Best epoch : {run_config['result']['best_epoch']}  |  "
          f"best val loss : {run_config['result']['best_val_loss']:.6f}")

    ckpt_path = run_dir / run_config["result"]["checkpoint"]

    # ── Build full dataset (all init dates, no obs filter) ───────────────────
    print("\nBuilding full-inference dataset ...")
    full_ds, norm_stats = _build_full_dataset(args, cfg_training, fold_idx)
    print(f"Lakes : {full_ds.n_lakes}")

    # ── Load model ────────────────────────────────────────────────────────────
    device         = torch.device(args.device)
    cfg_model_load = dict(cfg_model)
    model_type_    = cfg_model_load.pop("model_type", "SWOTGNNMultiStep")
    spec           = MODEL_REGISTRY[model_type_]
    model          = spec.model_cls(**cfg_model_load).to(device)

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Checkpoint loaded : {ckpt_path.name}  |  device: {device}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\nRunning full inference ...")
    raw_df = run_full_inference(full_ds, model, device, fold_used=fold_idx)

    # ── Denormalize ───────────────────────────────────────────────────────────
    lake_stats = pd.read_csv(args.wse_stats_csv)
    lake_stats = lake_stats[["lake_id", "lake_mean", "lake_std"]].drop_duplicates("lake_id")
    full_df    = denormalize(raw_df, lake_stats)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = run_dir.parent / "full_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    col_order = [
        "lake_id", "fold_used", "init_date", "forecast_date", "lead_day",
        "pred_norm", "pred_m",
        "pred_std_norm", "pred_std_m",
        "target_norm", "target_m",
        "obs_available",
        "days_since_last_swot",
    ]
    for col in ("pred_std_norm", "pred_std_m", "target_m", "lake_mean", "lake_std",
                "days_since_last_swot"):
        if col not in full_df.columns:
            full_df[col] = float("nan")

    csv_path = out_dir / "full_predictions.csv"
    full_df[col_order].to_csv(csv_path, index=False)
    print(f"\nFull predictions saved → {csv_path}")
    print(f"  {len(full_df):,} rows  |  "
          f"{full_df['init_date'].nunique()} init dates  |  "
          f"{full_df['lake_id'].nunique()} lakes  |  "
          f"{int(full_df['obs_available'].sum()):,} rows with obs")

    # ── Save provenance JSON ──────────────────────────────────────────────────
    config_out = {
        "fold_used":         fold_idx,
        "best_val_loss":     run_config["result"]["best_val_loss"],
        "best_epoch":        run_config["result"]["best_epoch"],
        "checkpoint":        str(ckpt_path),
        "train_start":       fold_def["train_start"],
        "train_end":         fold_def["train_end"],
        "test_start":        fold_def["test_start"],
        "test_end":          fold_def["test_end"],
        "forecast_horizon":  forecast_horizon,
        "n_init_dates":      int(full_df["init_date"].nunique()),
        "n_lakes":           int(full_df["lake_id"].nunique()),
        "n_rows_total":      len(full_df),
        "n_rows_with_obs":   int(full_df["obs_available"].sum()),
    }
    json_path = out_dir / "full_inference_config.json"
    with open(json_path, "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"Provenance saved  → {json_path}\n")


if __name__ == "__main__":
    main()
