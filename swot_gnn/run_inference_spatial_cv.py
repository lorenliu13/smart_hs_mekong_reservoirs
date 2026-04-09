"""
Inference script for spatial cross-validation lake WSE forecasting.

Mirrors run_inference_lake.py but for checkpoints produced by
run_spatial_cv_wse1d.py.  Key differences:

  - Datasets are rebuilt with build_spatial_cv_fold (same fold / seed args
    as training) so the spatial masks are available.
  - Inference is run on all three splits (train, val, test).  Every row is
    tagged with a boolean `is_test_lake` column derived from the fold's
    spatial_mask.
  - Per-lake metrics (physical units) are computed ONLY for held-out test
    lakes.  The full result CSVs are saved for all lakes / all splits.

Usage:
  python run_inference_spatial_cv.py \\
    --config          configs/exp02_mekong_wse1d.yaml \\
    --wse-datacube    /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube   /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube  /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --wse-stats-csv   /path/to/lake_wse_norm_stats.csv \\
    --lake-graph      /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --save-dir        checkpoints \\
    --run-name        exp02_mekong_wse1d_era5_ifshres_gritv06_202312_202602 \\
    --fold-idx        0 \\
    --n-folds         5 \\
    --spatial-seed    42 \\
    --val-method      temporal \\
    --seed            42 \\
    --device          cuda

Note: --config, --fold-idx, --n-folds, --spatial-seed, --val-method, and
--seed must match the values used during training so that the run directory
name is resolved correctly (identical slug logic to run_spatial_cv_wse1d.py).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.temporal_graph_dataset_lake import (
    build_spatial_cv_fold,
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

    is_test_lake is True for spatially held-out test lakes, False for train lakes.
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


def compute_lake_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-lake metrics (physical units).

    For deterministic models, NSE/KGE/RMSE/MAE are computed on the predicted mean.
    For probabilistic models (pred_std_m present and finite), additional metrics are
    computed:
      - crps_m   : mean CRPS (proper scoring rule; lower = better)
      - cov90    : empirical 90% PI coverage (ideal = 0.90)
      - piw90_m  : mean 90% PI width in metres (lower = sharper)
    """
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spatial-CV inference — load checkpoint, run predictions, save CSVs"
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
                        help="GRIT PLD lake graph CSV")
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML config used during training (same as --config passed to training)",
    )
    parser.add_argument("--save-dir",  default="checkpoints",
                        help="Root directory for run outputs (same as training)")
    parser.add_argument("--run-name",  required=True,
                        help="Base run name passed to training (without fold/seed suffixes)")
    parser.add_argument("--fold-idx",  type=int, required=True,
                        help="Fold index used during training (0-indexed)")
    parser.add_argument("--n-folds",   type=int, default=5,
                        help="Total number of spatial folds used during training (default 5)")
    parser.add_argument("--spatial-seed", type=int, default=42,
                        help="RNG seed for the lake shuffle used during training (default 42)")
    parser.add_argument(
        "--val-method", default="temporal", choices=["temporal", "spatial"],
        help="Validation strategy used during training (default: temporal)",
    )
    parser.add_argument(
        "--spatial-val-frac", type=float, default=0.1,
        help="Fraction of train-fold lakes used as spatial val set (only with --val-method spatial, default 0.1)",
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

    # Mirror run-name construction from run_spatial_cv_wse1d.py
    if args.seed is None:
        args.seed = cfg.get("seed", 42)

    model_type = cfg["model"].get("model_type", "SWOTGNN")
    model_slug = MODEL_REGISTRY[model_type].slug
    val_tag = (
        f"_valsp{args.spatial_val_frac}" if args.val_method == "spatial"
        else "_valt"
    )
    args.run_name = (
        f"{args.run_name}"
        f"_fold{args.fold_idx}of{args.n_folds}"
        f"_sp{args.spatial_seed}"
        f"{val_tag}"
        f"_{model_slug}"
        f"_s{args.seed}"
    )

    run_dir  = Path(args.save_dir) / args.run_name
    cfg_path = run_dir / "run_config.yaml"

    print(f"Resolved run directory: {run_dir}")

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Check that --run-name, --fold-idx, --n-folds, --spatial-seed, "
            f"--val-method, --spatial-val-frac, and --seed all match the training run.\n"
            f"Expected slug: {args.run_name}"
        )
    if not cfg_path.exists():
        # List what IS in the directory to help diagnose
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

    # ── Rebuild the same spatial CV split ─────────────────────────────────────
    train_ds, val_ds, test_ds, norm_stats = build_spatial_cv_fold(
        wse_datacube_path            = args.wse_datacube,
        era5_climate_datacube_path   = args.era5_datacube,
        ecmwf_forecast_datacube_path = args.ecmwf_datacube,
        static_datacube_path         = args.static_datacube,
        lake_graph_path              = args.lake_graph,
        seq_len                      = cfg_training['seq_len'],
        forecast_horizon             = cfg_training['forecast_horizon'],
        n_folds                      = args.n_folds,
        fold_idx                     = args.fold_idx,
        spatial_split_seed           = args.spatial_seed,
        val_frac                     = cfg_training.get('val_frac', 0.15),
        val_method                   = args.val_method,
        spatial_val_frac             = args.spatial_val_frac,
    )
    print(f'Splits: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test samples')
    print(f'Lakes : {test_ds.n_lakes}   Forecast horizon: {test_ds.forecast_horizon}')
    print(f'Fold  : {args.fold_idx}/{args.n_folds}  '
          f'({norm_stats["n_test_lakes"]} test lakes / '
          f'{norm_stats["n_train_lakes"]} train lakes)')

    # Derive the set of held-out test lake IDs from the spatial mask
    test_mask_np    = test_ds.spatial_mask.numpy()  # (n_lakes,)
    test_lake_ids   = set(int(lid) for lid, m in zip(test_ds.lake_ids, test_mask_np) if m > 0)
    print(f'Test lake IDs resolved: {len(test_lake_ids)} lakes')

    # ── Load model checkpoint ─────────────────────────────────────────────────
    device         = torch.device(args.device)
    cfg_model_load = dict(cfg_model)
    model_type     = cfg_model_load.pop("model_type", "SWOTGNN")
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
    train_raw = run_inference(train_ds, model, device, test_lake_ids, "train")
    val_raw   = run_inference(val_ds,   model, device, test_lake_ids, "val")
    test_raw  = run_inference(test_ds,  model, device, test_lake_ids, "test")
    print()

    # ── Denormalize ───────────────────────────────────────────────────────────
    lake_stats = pd.read_csv(args.wse_stats_csv)
    lake_stats = lake_stats[["lake_id", "lake_mean", "lake_std"]].drop_duplicates("lake_id")

    train_df = denormalize(train_raw, lake_stats)
    val_df   = denormalize(val_raw,   lake_stats)
    test_df  = denormalize(test_raw,  lake_stats)

    # ── Save result CSVs (all lakes, all splits) ──────────────────────────────
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

    # ── Per-lake metrics — test lakes only ────────────────────────────────────
    # Compute metrics on the held-out test lakes only, using each split and
    # combined views.  is_test_lake filters to the spatially held-out lakes.
    def _filter_test(df: pd.DataFrame) -> pd.DataFrame:
        return df[df['is_test_lake']].copy()

    # Dedup keys: same (lake, init_date, lead_day) can appear in multiple splits
    # when val_method="spatial" (all three datasets share the same date range) or
    # in temporal mode (test_idx = all_valid overlaps both train and val ranges).
    _DEDUP_KEYS = ["lake_id", "init_date", "lead_day"]

    test_only_from_test   = _filter_test(test_df)
    test_only_from_val    = _filter_test(val_df)
    test_only_full        = (
        _filter_test(pd.concat([train_df, val_df, test_df], ignore_index=True))
        .drop_duplicates(subset=_DEDUP_KEYS)
        .reset_index(drop=True)
    )
    val_test_combined     = (
        _filter_test(pd.concat([val_df, test_df], ignore_index=True))
        .drop_duplicates(subset=_DEDUP_KEYS)
        .reset_index(drop=True)
    )

    lake_metrics_test_df     = compute_lake_metrics(test_only_from_test)
    lake_metrics_val_test_df = compute_lake_metrics(val_test_combined)
    lake_metrics_full_df     = compute_lake_metrics(test_only_full)

    lake_metrics_test_df.to_csv(run_dir     / "lake_metrics_test.csv",     index=False)
    lake_metrics_val_test_df.to_csv(run_dir / "lake_metrics_val_test.csv", index=False)
    lake_metrics_full_df.to_csv(run_dir     / "lake_metrics_full.csv",     index=False)
    val_mode_note = "(all dates — same as test for spatial val)" if args.val_method == "spatial" else "(val+test date range)"
    print(f'Per-lake metrics saved (test lakes only, {len(lake_metrics_test_df)} lakes):')
    print(f'  -> {run_dir}/lake_metrics_test.csv       (test split, test lakes)')
    print(f'  -> {run_dir}/lake_metrics_val_test.csv   {val_mode_note}, test lakes)')
    print(f'  -> {run_dir}/lake_metrics_full.csv       (all splits deduped, test lakes)')

    # ── Median summary with 5th-95th percentile CI ────────────────────────────
    metric_cols = ['mse_m2', 'rmse_m', 'mae_m', 'nse', 'kge', 'autocorr',
                   'crps_m', 'cov90', 'piw90_m']

    def _print_summary(label, mdf):
        cols = [c for c in metric_cols if c in mdf.columns]
        print(f'\n=== {label} — Median Per-Lake Metrics (5th–95th CI) [test lakes only] ===')
        for col in cols:
            med = mdf[col].median()
            lo  = mdf[col].quantile(0.05)
            hi  = mdf[col].quantile(0.95)
            print(f'  {col:<12}: {med:6.3f}  ({lo:.3f} – {hi:.3f})')

    _print_summary("Test split",      lake_metrics_test_df)
    _print_summary("Val + Test",      lake_metrics_val_test_df)
    _print_summary("Full record",     lake_metrics_full_df)


if __name__ == "__main__":
    main()
