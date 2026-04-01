"""
Inference script for 1-day-ahead lake WSE forecasting.

Loads the best checkpoint saved by run_training_lake_wse1d.py, re-builds the
exact train / val / test splits from the original datacubes, runs forward
passes, denormalises predictions, and writes result CSVs + per-lake metrics.

Usage:
  python run_inference_lake.py \\
    --wse-datacube    /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube   /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube  /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --wse-stats-csv   /path/to/lake_wse_norm_stats.csv \\
    --lake-graph      /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --save-dir        checkpoints \\
    --run-name        exp02_mekong_wse1d_era5_ifshres_gritv06_202312_202602 \\
    --seed            42 \\
    --device          cuda

Note: --run-name should be the base name passed to training (without the
_s{seed} suffix).  The seed suffix is appended automatically to match the
directory created by run_training_lake_wse1d.py.
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

from data.temporal_graph_dataset_lake import build_temporal_dataset_from_lake_datacubes
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


def run_inference(ds, model, device, split_name=''):
    """Run model over all samples; return DataFrame of observed-lake predictions.

    Supports forecast_horizon >= 1. Each row records:
        lake_id, init_date, forecast_date, lead_day, pred_norm, target_norm,
        pred_std_norm (NaN for deterministic / point models)
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
                        })

    df = pd.DataFrame(records)
    print(f'  {split_name:<6}: {len(df):>6} observed-lake rows  |  '
          f'{df["init_date"].nunique()} forecast dates')
    return df


def compute_lake_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-lake metrics (physical units) on the test split.

    For deterministic models, NSE/KGE/RMSE/MAE are computed on the predicted mean.
    For probabilistic models (pred_std_m present and finite), additional metrics are
    computed:
      - crps_m   : mean CRPS (proper scoring rule; lower = better)
      - cov90    : empirical 90% PI coverage (ideal = 0.90)
      - piw90_m  : mean 90% PI width in metres (lower = sharper)
    """
    has_std = "pred_std_m" in test_df.columns
    lake_rows = []
    for lake_id, sub in test_df.groupby('lake_id'):
        sub = sub.sort_values('forecast_date')
        if len(sub) < 2:
            continue
        obs  = sub['target_m'].values
        pred = sub['pred_m'].values
        mse  = float(np.mean((obs - pred) ** 2))
        row = {
            'lake_id' : lake_id,
            'n_obs'   : len(sub),
            'mse_m2'  : mse,
            'rmse_m'  : float(np.sqrt(mse)),
            'mae_m'   : float(np.mean(np.abs(obs - pred))),
            'nse'     : nse(obs, pred),
            'kge'     : compute_kge(obs, pred),
            'autocorr': residual_autocorr_lag1(obs, pred, sub['forecast_date'].values),
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
        description="Inference — load checkpoint, run predictions, save CSVs"
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
    parser.add_argument("--save-dir",  default="checkpoints",
                        help="Root directory for run outputs (same as training)")
    parser.add_argument("--run-name",  required=True,
                        help="Base run name passed to training (without _s{seed} suffix)")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed used during training (default: 42, same as training fallback)",
    )
    args = parser.parse_args()

    # Mirror the seed-suffix logic from run_training_lake_wse1d.py
    if args.seed is None:
        args.seed = 42
    run_name_full = f"{args.run_name}_s{args.seed}"

    run_dir  = Path(args.save_dir) / run_name_full
    cfg_path = run_dir / "run_config.yaml"

    # Resolve checkpoint name from summary.json (training writes best_epoch{NNN}.pt)
    summary_path = run_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    ckpt_path = run_dir / summary["checkpoint"]

    # ── Load run config saved during training ─────────────────────────────────
    with open(cfg_path) as f:
        run_config = yaml.safe_load(f)
    cfg_model    = run_config['model']
    cfg_training = run_config['training']
    print(f'Loaded config: best epoch={run_config["result"]["best_epoch"]}, '
          f'best val loss={run_config["result"]["best_val_loss"]:.6f}')

    # ── Rebuild the same train / val / test splits ─────────────────────────────
    train_ds, val_ds, test_ds, _ = build_temporal_dataset_from_lake_datacubes(
        wse_datacube_path            = args.wse_datacube,
        era5_climate_datacube_path   = args.era5_datacube,
        ecmwf_forecast_datacube_path = args.ecmwf_datacube,
        static_datacube_path         = args.static_datacube,
        lake_graph_path              = args.lake_graph,
        seq_len                      = cfg_training['seq_len'],
        forecast_horizon             = cfg_training['forecast_horizon'],
        train_frac                   = cfg_training['train_frac'],
        val_frac                     = cfg_training['val_frac'],
        test_frac                    = cfg_training['test_frac'],
    )
    print(f'Splits: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test samples')
    print(f'Lakes : {test_ds.n_lakes}   Forecast horizon: {test_ds.forecast_horizon}')

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
    train_raw = run_inference(train_ds, model, device, "train")
    val_raw   = run_inference(val_ds,   model, device, "val")
    test_raw  = run_inference(test_ds,  model, device, "test")
    print()

    # ── Denormalize ───────────────────────────────────────────────────────────
    lake_stats = pd.read_csv(args.wse_stats_csv)
    lake_stats = lake_stats[["lake_id", "lake_mean", "lake_std"]].drop_duplicates("lake_id")

    train_df = denormalize(train_raw, lake_stats)
    val_df   = denormalize(val_raw,   lake_stats)
    test_df  = denormalize(test_raw,  lake_stats)

    # ── Save result CSVs ──────────────────────────────────────────────────────
    col_order = [
        "lake_id", "init_date", "forecast_date", "lead_day",
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

    # ── Per-lake metrics (physical units) ─────────────────────────────────────
    lake_metrics_df = compute_lake_metrics(test_df)
    lake_metrics_df.to_csv(run_dir / "lake_metrics_test.csv", index=False)
    print(f'Per-lake metrics saved: {len(lake_metrics_df)} lakes')
    print(f'  -> {run_dir}/lake_metrics_test.csv')

    # ── Median summary with 5th-95th percentile CI ────────────────────────────
    metric_cols = ['mse_m2', 'rmse_m', 'mae_m', 'nse', 'kge', 'autocorr',
                   'crps_m', 'cov90', 'piw90_m']
    metric_cols = [c for c in metric_cols if c in lake_metrics_df.columns]
    print('\n=== Median Per-Lake Metrics (5th-95th CI) ===')
    for col in metric_cols:
        med = lake_metrics_df[col].median()
        lo  = lake_metrics_df[col].quantile(0.05)
        hi  = lake_metrics_df[col].quantile(0.95)
        print(f'  {col:<12}: {med:6.3f}  ({lo:.3f} - {hi:.3f})')


if __name__ == "__main__":
    main()
