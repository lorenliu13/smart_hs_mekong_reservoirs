"""
Inference script for regional cross-validation multi-day lake WSE forecasting.

Mirrors wse1d_training/run_inference_regional_cv.py but for checkpoints produced by
run_training_lake_wsend_regional_cv.py.  Key differences from the 1-day version:

  - Run name slug includes _h{forecast_horizon} (matches training slug logic).
  - forecast_horizon > 1 is asserted.
  - Metrics are computed independently for each lead day (0 … H-1).
  - Per-lake metric CSVs are saved per lead day:
      lake_metrics_{split}_lead{d}.csv  (d = lead day index, 0-based)
  - An aggregate CSV (all lead days stacked) is also saved per split:
      lake_metrics_{split}.csv  (contains a lead_day column)
  - The printed summary table shows median metrics broken down by lead day.

Dataset reconstruction mirrors training exactly:
  - build_regional_cv_fold with the same fold_idx / val_method / spatial_val_frac /
    spatial_val_seed arguments.
  - require_obs_on_any_forecast_day=True (same default as training).

Usage:
  python run_inference_wsend_regional_cv.py \\
    --config          configs/wsend/exp08_wsend.yaml \\
    --wse-datacube    /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube   /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube  /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --wse-stats-csv   /path/to/lake_wse_norm_stats.csv \\
    --lake-graph      /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --save-dir        checkpoints \\
    --run-name        exp08_wsend_regionalcv \\
    --fold-idx        0 \\
    --val-method      temporal \\
    --seed            42 \\
    --device          cuda

Note: --config, --fold-idx, --val-method, --spatial-val-frac, --spatial-val-seed,
and --seed must match the values used during training so that the run directory
name is resolved correctly (identical slug logic to run_training_lake_wsend_regional_cv.py).
"""
import argparse
import json
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.regional_cv import (
    build_regional_cv_fold,
    N_REGIONAL_FOLDS,
    REGION_NAMES,
)
from models.registry import MODEL_REGISTRY
from training.evaluate import compute_kge


# ── Metrics ───────────────────────────────────────────────────────────────────

def nse(obs, pred):
    """Nash-Sutcliffe Efficiency. Range: -inf to 1 (perfect)."""
    obs, pred = np.array(obs), np.array(pred)
    ok = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[ok], pred[ok]
    if len(obs) == 0:
        return np.nan
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return 1.0 if np.allclose(obs, pred) else np.nan
    return 1 - np.sum((obs - pred) ** 2) / denom


def crps_gaussian(obs, mu, sigma):
    """
    Analytical CRPS for Gaussian predictive distribution.

    CRPS(N(μ,σ), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ.
    Lower is better; equals MAE when σ→0.
    """
    obs, mu, sigma = np.array(obs), np.array(mu), np.array(sigma)
    ok = ~(np.isnan(obs) | np.isnan(mu) | np.isnan(sigma) | (sigma <= 0))
    if ok.sum() == 0:
        return np.nan
    z    = (obs[ok] - mu[ok]) / sigma[ok]
    crps = sigma[ok] * (z * (2 * scipy_stats.norm.cdf(z) - 1)
                        + 2 * scipy_stats.norm.pdf(z)
                        - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def pi_coverage(obs, mu, sigma, alpha=0.90):
    """
    Fraction of observations inside the (1-alpha) central prediction interval.
    E.g. alpha=0.90 → 90% PI: μ ± 1.645σ.
    """
    obs, mu, sigma = np.array(obs), np.array(mu), np.array(sigma)
    ok = ~(np.isnan(obs) | np.isnan(mu) | np.isnan(sigma) | (sigma <= 0))
    if ok.sum() == 0:
        return np.nan
    z_crit = scipy_stats.norm.ppf(0.5 + alpha / 2)   # e.g. 1.645 for 90%
    inside = (obs[ok] >= mu[ok] - z_crit * sigma[ok]) & \
             (obs[ok] <= mu[ok] + z_crit * sigma[ok])
    return float(inside.mean())


def pi_width(sigma, alpha=0.90):
    """Mean width of the (1-alpha) central prediction interval: 2 * z_crit * σ."""
    sigma = np.array(sigma)
    ok    = ~(np.isnan(sigma) | (sigma <= 0))
    if ok.sum() == 0:
        return np.nan
    z_crit = scipy_stats.norm.ppf(0.5 + alpha / 2)
    return float(np.mean(2 * z_crit * sigma[ok]))


def residual_autocorr_lag1(obs, pred, dates=None):
    """Pearson lag-1 autocorrelation of residuals in temporal order."""
    obs, pred = np.array(obs), np.array(pred)
    ok = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[ok], pred[ok]
    if dates is not None:
        dates_ok = np.array(dates)[ok]
        order    = np.argsort(dates_ok)
        obs, pred = obs[order], pred[order]
    residuals = obs - pred
    if len(residuals) < 3:
        return np.nan
    r0, r1 = residuals[:-1], residuals[1:]
    if np.std(r0) == 0 or np.std(r1) == 0:
        return np.nan
    return float(np.corrcoef(r0, r1)[0, 1])


# ── Core functions ────────────────────────────────────────────────────────────

def denormalize(df: pd.DataFrame, lake_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Add pred_m / target_m columns (raw WSE in metres).
    For probabilistic models, also adds pred_std_m (physical-unit std).

    lake_stats must have columns: lake_id, lake_mean, lake_std
    WSE normalisation: wse_norm = (wse - lake_mean) / (lake_std + 1e-8)
    Inverse:           wse      = wse_norm * (lake_std + 1e-8) + lake_mean
    Std inverse:       std      = std_norm  * (lake_std + 1e-8)   (scale only, no shift)
    """
    df = df.merge(lake_stats[["lake_id", "lake_mean", "lake_std"]],
                  on="lake_id", how="left")
    std_eps = df["lake_std"].fillna(1.0) + 1e-8
    mean_   = df["lake_mean"].fillna(0.0)
    df["pred_m"]   = df["pred_norm"]   * std_eps + mean_
    df["target_m"] = df["target_norm"] * std_eps + mean_
    if "pred_std_norm" in df.columns:
        df["pred_std_m"] = df["pred_std_norm"] * std_eps  # std scales, does not shift
    return df


def run_inference(ds, model, device, test_lake_ids: set, split_name=''):
    """Run model over all samples; return DataFrame of observed-lake predictions.

    Supports forecast_horizon >= 1. Each row records:
        lake_id, init_date, forecast_date, lead_day, pred_norm, target_norm,
        pred_std_norm (NaN for deterministic / point models), is_test_lake

    is_test_lake is True for regionally held-out test lakes, False for train lakes.

    Inputs:
        ds: TemporalGraphDatasetLake with spatial masks (train_ds, val_ds, test_ds
            from build_regional_cv_fold)
        model: trained PyTorch model for inference
        device: torch device to run inference on
        test_lake_ids: set of lake IDs that are in the regionally held-out test set
            (derived from ds.spatial_mask)
        split_name: string label for the dataset split (e.g. 'train', 'val', 'test')
            used for logging purposes

    Outputs:
        DataFrame with columns: lake_id, init_date, forecast_date, lead_day,
            pred_norm, target_norm, pred_std_norm, is_test_lake
    """
    records = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(ds)):
            data_list, static, labels, mask = ds[idx]

            x          = torch.stack([d.x for d in data_list], dim=1).to(device)
            edge_index = data_list[0].edge_index.to(device)
            static_t   = static.to(device)

            pred_out = model(x, edge_index, static_features=static_t)

            # Gaussian models return (mean, log_std); point models return a tensor
            if isinstance(pred_out, tuple):
                mean_cpu = pred_out[0].cpu().numpy()
                std_cpu  = pred_out[1].clamp(-6, 6).exp().cpu().numpy()  # σ_norm
            else:
                mean_cpu = pred_out.cpu().numpy()
                std_cpu  = None

            # Normalise to 2-D (n_lakes, horizon) regardless of forecast_horizon
            labels_np = labels.numpy()
            mask_np   = mask.numpy()
            if labels_np.ndim == 1:          # forecast_horizon == 1 backward compat
                labels_np = labels_np[:, np.newaxis]
                mask_np   = mask_np[:, np.newaxis]
            if mean_cpu.ndim == 1:
                mean_cpu = mean_cpu[:, np.newaxis]
            if std_cpu is not None and std_cpu.ndim == 1:
                std_cpu = std_cpu[:, np.newaxis]

            j         = int(ds.valid_starts[idx])
            init_date = pd.Timestamp(ds.ecmwf_init_dates[j])
            horizon   = labels_np.shape[1]

            for d in range(horizon):
                forecast_date = init_date + pd.Timedelta(days=d)
                for lake_i, lake_id in enumerate(ds.lake_ids):
                    if mask_np[lake_i, d] > 0:
                        records.append({
                            'lake_id'      : int(lake_id),
                            'init_date'    : init_date,
                            'forecast_date': forecast_date,
                            'lead_day'     : d,
                            'pred_norm'    : float(mean_cpu[lake_i, d]),
                            'target_norm'  : float(labels_np[lake_i, d]),
                            'pred_std_norm': float(std_cpu[lake_i, d]) if std_cpu is not None else float('nan'),
                            'is_test_lake' : int(lake_id) in test_lake_ids,
                        })

    df = pd.DataFrame(records)
    n_test_rows  = int(df['is_test_lake'].sum()) if len(df) > 0 else 0
    n_train_rows = len(df) - n_test_rows
    print(f'  {split_name:<6}: {len(df):>6} observed-lake rows  |  '
          f'{df["init_date"].nunique() if len(df) > 0 else 0} forecast dates  |  '
          f'test-lake rows: {n_test_rows}  train-lake rows: {n_train_rows}')
    return df


def compute_lake_metrics(df: pd.DataFrame, lead_day: int | None = None) -> pd.DataFrame:
    """
    Compute per-lake metrics (physical units) for a single lead day or all lead days.

    If lead_day is not None, only rows with df['lead_day'] == lead_day are used.
    The returned DataFrame includes a 'lead_day' column.

    For deterministic models, NSE/KGE/RMSE/MAE are computed on the predicted mean.
    For probabilistic models (pred_std_m present and finite), additional metrics are
    computed:
      - crps_m   : mean CRPS (proper scoring rule; lower = better)
      - cov90    : empirical 90% PI coverage (ideal = 0.90)
      - piw90_m  : mean 90% PI width in metres (lower = sharper)
    """
    if lead_day is not None:
        df = df[df['lead_day'] == lead_day]
    ld_label = lead_day  # may be None when computing across all lead days

    has_std = "pred_std_m" in df.columns
    lake_rows = []
    for lake_id, sub in df.groupby('lake_id'):
        sub = sub.sort_values('forecast_date')
        if len(sub) < 2:
            continue
        obs  = sub['target_m'].values
        pred = sub['pred_m'].values
        mse  = float(np.mean((obs - pred) ** 2))
        row = {
            'lead_day'    : ld_label if ld_label is not None else -1,
            'lake_id'     : lake_id,
            'is_test_lake': bool(sub['is_test_lake'].iloc[0]),
            'n_obs'       : len(sub),
            'mse_m2'      : mse,
            'rmse_m'      : float(np.sqrt(mse)),
            'mae_m'       : float(np.mean(np.abs(obs - pred))),
            'nse'         : nse(obs, pred),
            'kge'         : compute_kge(obs, pred),
            'autocorr'    : residual_autocorr_lag1(obs, pred, sub['forecast_date'].values),
        }
        if has_std:
            sigma = sub['pred_std_m'].values
            row['crps_m']  = crps_gaussian(obs, pred, sigma)
            row['cov90']   = pi_coverage(obs, pred, sigma, alpha=0.90)
            row['piw90_m'] = pi_width(sigma, alpha=0.90)
        lake_rows.append(row)
    return pd.DataFrame(lake_rows)


def compute_metrics_by_lead_day(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """
    Compute per-lake metrics separately for each lead day present in df.

    Returns a dict mapping lead_day index → per-lake metrics DataFrame.
    """
    lead_days = sorted(df['lead_day'].unique())
    return {d: compute_lake_metrics(df, lead_day=d) for d in lead_days}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Regional-CV multi-step inference — load checkpoint, run predictions, save CSVs"
    )
    parser.add_argument("--wse-datacube",    required=True,
                        help="swot_lake_wse_datacube_wse_norm.nc")
    parser.add_argument("--era5-datacube",   required=True,
                        help="swot_lake_era5_climate_datacube.nc")
    parser.add_argument("--ecmwf-datacube",  required=True,
                        help="swot_lake_ecmwf_forecast_datacube.nc")
    parser.add_argument("--static-datacube", required=True,
                        help="swot_lake_static_datacube.nc")
    parser.add_argument("--wse-stats-csv",   required=True,
                        help="lake_wse_norm_stats.csv (used for denormalisation)")
    parser.add_argument("--lake-graph",      required=True,
                        help="GRIT PLD lake graph CSV (must contain lake_id and "
                             "hybasin_level_4 columns)")
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML config used during training (same as --config passed to training)",
    )
    parser.add_argument("--save-dir",  default="checkpoints",
                        help="Root directory for run outputs (same as training)")
    parser.add_argument("--run-name",  required=True,
                        help="Base run name passed to training (without fold/horizon/seed suffixes)")
    parser.add_argument("--fold-idx",  type=int, required=True,
                        help="Fold index (region) used during training (0–4)")
    parser.add_argument(
        "--hybas-col", default="hybasin_level_4",
        help="Column in the lake graph CSV holding the HYBAS Level-4 sub-basin ID "
             "(default: hybasin_level_4)",
    )
    parser.add_argument(
        "--val-method", default="temporal", choices=["temporal", "spatial"],
        help="Validation strategy used during training (default: temporal)",
    )
    parser.add_argument(
        "--spatial-val-frac", type=float, default=0.1,
        help="Fraction of train-region lakes used as spatial val set "
             "(only with --val-method spatial, default 0.1)",
    )
    parser.add_argument(
        "--spatial-val-seed", type=int, default=43,
        help="RNG seed for the spatial val lake draw used during training (default 43)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed used during training (default: from config or 42)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Mirror run-name construction from run_training_lake_wsend_regional_cv.py
    if args.seed is None:
        args.seed = cfg.get("seed", 42)

    forecast_horizon = cfg["training"]["forecast_horizon"]
    assert forecast_horizon > 1, (
        f"run_inference_wsend_regional_cv.py is designed for forecast_horizon > 1, "
        f"got {forecast_horizon}. Use wse1d_training/ scripts for 1-day-ahead forecasting."
    )

    model_type = cfg["model"].get("model_type", "SWOTGNNMultiStep")
    model_slug = MODEL_REGISTRY[model_type].slug
    val_tag = (
        f"_valsp{args.spatial_val_frac}" if args.val_method == "spatial"
        else "_valt"
    )
    # Save base name before augmentation; matches the parent directory created during training
    base_run_name = args.run_name
    args.run_name = (
        f"{args.run_name}"
        f"_fold{args.fold_idx}of{N_REGIONAL_FOLDS}"
        f"_h{forecast_horizon}"
        f"{val_tag}"
        f"_{model_slug}"
        f"_s{args.seed}"
    )

    # Mirrors the training output path: save_dir / base_run_name / fold_{fold_idx}/
    run_dir  = Path(args.save_dir) / base_run_name / f"fold_{args.fold_idx}"
    cfg_path = run_dir / "run_config.yaml"

    region_name = REGION_NAMES.get(args.fold_idx, f"fold_{args.fold_idx}")
    print(f"Resolved run directory: {run_dir}")
    print(f"Region [{args.fold_idx}]: {region_name}")
    print(f"Forecast horizon: {forecast_horizon} days")

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Check that --run-name, --fold-idx, --val-method, --spatial-val-frac, "
            f"--spatial-val-seed, and --seed all match the training run.\n"
            f"Expected slug: {args.run_name}"
        )
    if not cfg_path.exists():
        existing = sorted(run_dir.iterdir())
        existing_str = "\n  ".join(p.name for p in existing) or "(empty)"
        raise FileNotFoundError(
            f"run_config.yaml not found in {run_dir}\n"
            f"Training may not have completed — run_config.yaml is written at the "
            f"very end of training.\n"
            f"Files present in run_dir:\n  {existing_str}"
        )

    # ── Load run config saved during training ─────────────────────────────────
    with open(cfg_path) as f:
        run_config = yaml.safe_load(f)
    cfg_model    = run_config['model']
    cfg_training = run_config['training']
    print(f'Loaded config: best epoch={run_config["result"]["best_epoch"]}, '
          f'best val loss={run_config["result"]["best_val_loss"]:.6f}')

    # Resolve checkpoint path from run_config
    ckpt_path = run_dir / run_config["result"]["checkpoint"]

    # ── Rebuild the same regional CV split ────────────────────────────────────
    train_ds, val_ds, test_ds, norm_stats = build_regional_cv_fold(
        wse_datacube_path            = args.wse_datacube,
        era5_climate_datacube_path   = args.era5_datacube,
        ecmwf_forecast_datacube_path = args.ecmwf_datacube,
        static_datacube_path         = args.static_datacube,
        lake_graph_path              = args.lake_graph,
        seq_len                      = cfg_training['seq_len'],
        forecast_horizon             = cfg_training['forecast_horizon'],
        fold_idx                     = args.fold_idx,
        val_frac                     = cfg_training.get('val_frac', 0.15),
        val_method                   = args.val_method,
        spatial_val_frac             = args.spatial_val_frac,
        spatial_val_seed             = args.spatial_val_seed,
        hybas_col                    = args.hybas_col,
        require_obs_on_any_forecast_day = True,
    )
    print(f'Splits: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test samples')
    print(f'Lakes : {test_ds.n_lakes}   Forecast horizon: {test_ds.forecast_horizon}')
    print(f'Fold  : {args.fold_idx}/{N_REGIONAL_FOLDS}  [{region_name}]  '
          f'({norm_stats["n_test_lakes"]} test lakes / '
          f'{norm_stats["n_train_lakes"]} train lakes)')

    # Derive the set of held-out test lake IDs from the spatial mask
    test_mask_np  = test_ds.spatial_mask.numpy()  # (n_lakes,) 1 for test lake, 0 for train lake
    test_lake_ids = set(int(lid) for lid, m in zip(test_ds.lake_ids, test_mask_np) if m > 0)
    print(f'Test lake IDs resolved: {len(test_lake_ids)} lakes')

    # ── Load model checkpoint ─────────────────────────────────────────────────
    device         = torch.device(args.device)
    cfg_model_load = dict(cfg_model)
    model_type     = cfg_model_load.pop("model_type", "SWOTGNNMultiStep")
    spec           = MODEL_REGISTRY[model_type]
    model          = spec.model_cls(**cfg_model_load).to(device)

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    best_epoch = ckpt.get('best_epoch', '?')
    best_loss  = ckpt.get('best_val_loss', float('nan'))
    print(f'\nCheckpoint loaded — best epoch: '
          f'{best_epoch + 1 if isinstance(best_epoch, int) else best_epoch}'
          f'   best val loss: {best_loss:.6f}')
    print(f'Device: {device}')

    # ── Run inference on all splits ───────────────────────────────────────────
    print("\nRunning inference ...")
    # All three datasets share the same full lake graph.  test_ds always covers
    # the full date range regardless of val_method, so running inference once on
    # test_ds is sufficient.  Rows are then partitioned into train/val/test
    # without re-running the model.
    full_raw = run_inference(test_ds, model, device, test_lake_ids, "all")

    # Build lake_id → spatial group from the three spatial_mask tensors.
    # For val_method="spatial": each lake belongs to exactly one of train/val/test.
    # For val_method="temporal": train_mask == val_mask (same train-region lakes),
    #   so all train-region lakes map to 'train_fold' here; date decides train vs val.
    val_mask_np_ = val_ds.spatial_mask.numpy()
    lake_to_group: dict[int, str] = {}
    for lake_i, lake_id in enumerate(test_ds.lake_ids):
        lid = int(lake_id)
        if test_mask_np[lake_i] > 0:
            lake_to_group[lid] = 'test'
        elif val_mask_np_[lake_i] > 0:
            lake_to_group[lid] = 'val'
        else:
            lake_to_group[lid] = 'train'

    if args.val_method == "temporal":
        # train-region lakes (currently labelled 'train' or 'val' identically above)
        # are split by init_date: dates in train_ds.valid_starts → 'train', else → 'val'.
        train_init_dates = {
            pd.Timestamp(test_ds.ecmwf_init_dates[j]) for j in train_ds.valid_starts
        }

        def _assign_split(row) -> str:
            grp = lake_to_group[row['lake_id']]
            if grp == 'test':
                return 'test'
            return 'train' if row['init_date'] in train_init_dates else 'val'

        full_raw['_split'] = full_raw.apply(_assign_split, axis=1)
    else:
        # spatial: lake_to_group already assigns each lake to one split
        full_raw['_split'] = full_raw['lake_id'].map(lake_to_group)

    train_raw = full_raw[full_raw['_split'] == 'train'].drop(columns='_split').copy()
    val_raw   = full_raw[full_raw['_split'] == 'val'].drop(columns='_split').copy()
    test_raw  = full_raw[full_raw['_split'] == 'test'].drop(columns='_split').copy()
    print(f'  Partitioned: {len(train_raw)} train rows | '
          f'{len(val_raw)} val rows | {len(test_raw)} test rows')
    print()

    # ── Denormalize ───────────────────────────────────────────────────────────
    lake_stats = pd.read_csv(args.wse_stats_csv)
    lake_stats = lake_stats[["lake_id", "lake_mean", "lake_std"]].drop_duplicates("lake_id")

    train_df = denormalize(train_raw, lake_stats)
    val_df   = denormalize(val_raw,   lake_stats)
    test_df  = denormalize(test_raw,  lake_stats)

    # ── Save result CSVs (all lakes, all splits, all lead days) ──────────────
    col_order = [
        "lake_id", "is_test_lake", "init_date", "forecast_date", "lead_day",
        "pred_norm", "target_norm", "pred_m", "target_m",
        "pred_std_norm", "pred_std_m",
    ]
    # pred_std_norm / pred_std_m are NaN for deterministic models; keep columns consistent
    for col in ("pred_std_norm", "pred_std_m"):
        for df in (train_df, val_df, test_df):
            if col not in df.columns:
                df[col] = float("nan")
    train_df[col_order].to_csv(run_dir / "train_result_df.csv", index=False)
    val_df[col_order].to_csv(run_dir   / "val_result_df.csv",   index=False)
    test_df[col_order].to_csv(run_dir  / "test_result_df.csv",  index=False)
    print(f"Result CSVs saved to {run_dir}/")
    print(f"  train: {len(train_df)} rows | val: {len(val_df)} | test: {len(test_df)}\n")

    # ── Per-lake metrics broken down by lead day ──────────────────────────────
    # For each split, compute metrics per lead day; stack all lead days into one
    # aggregate CSV (with lead_day column) and also save per-lead-day CSVs.
    lead_days = sorted(full_raw['lead_day'].unique())

    splits_info = [
        ("train", train_df),
        ("val",   val_df),
        ("test",  test_df),
    ]

    for split_name, split_df in splits_info:
        by_lead = compute_metrics_by_lead_day(split_df)

        # Aggregate CSV: all lead days stacked
        agg_df = pd.concat(by_lead.values(), ignore_index=True)
        agg_csv = run_dir / f"lake_metrics_{split_name}.csv"
        agg_df.to_csv(agg_csv, index=False)

        # Per-lead-day CSVs
        for d, mdf in by_lead.items():
            mdf.to_csv(run_dir / f"lake_metrics_{split_name}_lead{d}.csv", index=False)

        n_lakes = len(split_df['lake_id'].unique()) if len(split_df) > 0 else 0
        print(f'Per-lake metrics saved ({split_name}, {n_lakes} lakes, '
              f'{len(lead_days)} lead days):')
        print(f'  -> {agg_csv.name}  (all lead days stacked)')
        for d in lead_days:
            print(f'  -> lake_metrics_{split_name}_lead{d}.csv')

    # ── Median summary broken down by lead day ────────────────────────────────
    metric_cols = ['mse_m2', 'rmse_m', 'mae_m', 'nse', 'kge', 'autocorr',
                   'crps_m', 'cov90', 'piw90_m']

    def _print_lead_summary(label: str, split_df: pd.DataFrame) -> None:
        print(f'\n=== {label} — Median Per-Lake Metrics by Lead Day (5th–95th CI) ===')
        by_lead = compute_metrics_by_lead_day(split_df)
        for d, mdf in sorted(by_lead.items()):
            cols = [c for c in metric_cols if c in mdf.columns]
            parts = []
            for col in cols:
                med = mdf[col].median()
                lo  = mdf[col].quantile(0.05)
                hi  = mdf[col].quantile(0.95)
                parts.append(f'{col}: {med:.3f} ({lo:.3f}–{hi:.3f})')
            print(f'  lead_day={d}  |  ' + '  |  '.join(parts))

    _print_lead_summary("Train lakes", train_df)
    _print_lead_summary("Val lakes",   val_df)
    _print_lead_summary(f"Test lakes [{region_name}]", test_df)

    # ── Save inference_metrics.json (aggregate median per split × lead day) ───
    def _agg_metrics_json(split_df: pd.DataFrame) -> dict:
        result = {}
        by_lead = compute_metrics_by_lead_day(split_df)
        for d, mdf in sorted(by_lead.items()):
            cols = [c for c in metric_cols if c in mdf.columns]
            result[f"lead{d}"] = {
                col: {
                    "median": float(mdf[col].median()),
                    "p05":    float(mdf[col].quantile(0.05)),
                    "p95":    float(mdf[col].quantile(0.95)),
                }
                for col in cols
                if not mdf[col].isna().all()
            }
        return result

    inference_metrics = {
        "fold_idx":         args.fold_idx,
        "region_name":      region_name,
        "forecast_horizon": forecast_horizon,
        "n_lead_days":      len(lead_days),
        "lead_days":        lead_days,
        "train": _agg_metrics_json(train_df),
        "val":   _agg_metrics_json(val_df),
        "test":  _agg_metrics_json(test_df),
    }
    inf_metrics_path = run_dir / "inference_metrics.json"
    with open(inf_metrics_path, "w") as f:
        json.dump(inference_metrics, f, indent=2)
    print(f'\nAggregate inference metrics saved → {inf_metrics_path}')


if __name__ == "__main__":
    main()
