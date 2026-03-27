"""
Baseline: Last-observation persistence for next-day lake WSE forecasting.

For every sample, predicts:
    WSE(init_date + d) = latest_wse(init_date - 1)     (d = 0 … forecast_horizon-1)

i.e. the forward-filled normalised WSE from the last history step is propagated
unchanged across all forecast lead days.  No model is trained; results are
computed analytically from the input datacubes.

Result CSVs and per-lake metrics are written in the same format as
run_inference_lake.py so that all experiments can be compared directly.

Usage:
  python run_baseline_last_obs_lake.py \\
    --wse-datacube    /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube   /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube  /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --wse-stats-csv   /path/to/lake_wse_norm_stats.csv \\
    --lake-graph      /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --save-dir        checkpoints \\
    --run-name        exp00_mekong_wse1d_last_obs_gritv06_202312_202512
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.temporal_graph_dataset_lake import assemble_lake_features_from_datacubes
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


# ── Core functions ─────────────────────────────────────────────────────────────

def denormalize(df: pd.DataFrame, lake_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Add pred_m / target_m columns (raw WSE in metres).

    lake_stats must have columns: lake_id, lake_mean, lake_std
    WSE normalisation: wse_norm = (wse - lake_mean) / (lake_std + 1e-8)
    Inverse:           wse      = wse_norm * (lake_std + 1e-8) + lake_mean
    """
    df = df.merge(lake_stats[["lake_id", "lake_mean", "lake_std"]],
                  on="lake_id", how="left")
    std_eps = df["lake_std"].fillna(1.0) + 1e-8
    mean_   = df["lake_mean"].fillna(0.0)
    df["pred_m"]   = df["pred_norm"]   * std_eps + mean_
    df["target_m"] = df["target_norm"] * std_eps + mean_
    return df


def run_last_obs_baseline(
    era5_dynamic: np.ndarray,          # (n_lakes, n_dates, n_features)
    wse_labels: np.ndarray,            # (n_lakes, n_dates) — normalised WSE, NaN where unobserved
    obs_mask: np.ndarray,              # (n_lakes, n_dates)
    lake_ids: np.ndarray,              # (n_lakes,)
    era5_dates: pd.DatetimeIndex,
    ecmwf_init_dates: pd.DatetimeIndex,
    valid_starts: np.ndarray,          # positions in ecmwf_init_dates for this split
    seq_len: int,
    forecast_horizon: int,
    split_name: str = '',
) -> pd.DataFrame:
    """
    Persistence baseline: pred_norm = latest_wse at last history step.

    Feature index 1 in era5_dynamic is 'latest_wse' — the forward-filled
    per-lake normalised WSE.  It is not re-normalised by the dataset (it is
    pre-normalised per lake), so it is directly comparable to wse_labels.

    Returns a DataFrame with columns:
        lake_id, init_date, forecast_date, lead_day, pred_norm, target_norm
    """
    era5_date_to_idx = {d: i for i, d in enumerate(era5_dates)}
    LATEST_WSE_IDX = 1  # feature index for 'latest_wse' in era5_dynamic

    records = []
    for j in valid_starts:
        init_date    = ecmwf_init_dates[j]
        last_hist_day = init_date - pd.Timedelta(days=1)
        era5_end_idx  = era5_date_to_idx.get(last_hist_day)
        if era5_end_idx is None or era5_end_idx < seq_len - 1:
            continue

        # Persistence prediction: last observed (forward-filled) normalised WSE
        pred_norm = era5_dynamic[:, era5_end_idx, LATEST_WSE_IDX]  # (n_lakes,)

        for d in range(forecast_horizon):
            forecast_date = init_date + pd.Timedelta(days=d)
            t_idx = era5_date_to_idx.get(forecast_date)
            if t_idx is None:
                continue

            mask_d   = obs_mask[:, t_idx]    # (n_lakes,)
            target_d = wse_labels[:, t_idx]  # (n_lakes,)

            for lake_i, lake_id in enumerate(lake_ids):
                if mask_d[lake_i] > 0:
                    records.append({
                        'lake_id'      : int(lake_id),
                        'init_date'    : init_date,
                        'forecast_date': forecast_date,
                        'lead_day'     : d,
                        'pred_norm'    : float(pred_norm[lake_i]),
                        'target_norm'  : float(target_d[lake_i]),
                    })

    df = pd.DataFrame(records)
    print(f'  {split_name:<6}: {len(df):>6} observed-lake rows  |  '
          f'{df["init_date"].nunique() if len(df) else 0} forecast dates')
    return df


def compute_lake_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-lake metrics (physical units) on a split."""
    lake_rows = []
    for lake_id, sub in test_df.groupby('lake_id'):
        sub = sub.sort_values('forecast_date')
        if len(sub) < 2:
            continue
        obs  = sub['target_m'].values
        pred = sub['pred_m'].values
        mse  = float(np.mean((obs - pred) ** 2))
        lake_rows.append({
            'lake_id' : lake_id,
            'n_obs'   : len(sub),
            'mse_m2'  : mse,
            'rmse_m'  : float(np.sqrt(mse)),
            'mae_m'   : float(np.mean(np.abs(obs - pred))),
            'nse'     : nse(obs, pred),
            'kge'     : compute_kge(obs, pred),
            'autocorr': residual_autocorr_lag1(obs, pred, sub['forecast_date'].values),
        })
    return pd.DataFrame(lake_rows)


def _find_valid_starts(
    ecmwf_init_dates: pd.DatetimeIndex,
    era5_date_to_idx: dict,
    obs_mask: np.ndarray,
    seq_len: int,
    forecast_horizon: int,
    require_obs: bool = True,
) -> np.ndarray:
    """Replicate the valid-start logic from build_temporal_dataset_from_lake_datacubes."""
    valid = []
    for j, init_date in enumerate(ecmwf_init_dates):
        last_hist_day = init_date - pd.Timedelta(days=1)
        era5_idx = era5_date_to_idx.get(last_hist_day)
        if era5_idx is None or era5_idx < seq_len - 1:
            continue
        if require_obs:
            has_obs = False
            for d in range(forecast_horizon):
                t_idx = era5_date_to_idx.get(init_date + pd.Timedelta(days=d))
                if t_idx is not None and obs_mask[:, t_idx].sum() > 0:
                    has_obs = True
                    break
            if not has_obs:
                continue
        valid.append(j)
    return np.array(valid, dtype=np.int64)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "exp00 baseline: last-observation persistence for next-day WSE — "
            "no model, no training"
        )
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
                        help="Root directory for run outputs")
    parser.add_argument("--run-name",
                        default="exp00_mekong_wse1d_last_obs_gritv06_202312_202512",
                        help="Run subfolder name")
    parser.add_argument("--seq-len",          type=int, default=30,
                        help="ERA5 history window length (default: 30)")
    parser.add_argument("--forecast-horizon", type=int, default=1,
                        help="Number of forecast lead days (default: 1)")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac",   type=float, default=0.15)
    parser.add_argument("--test-frac",  type=float, default=0.15)
    args = parser.parse_args()

    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seq_len          = args.seq_len
    forecast_horizon = args.forecast_horizon

    # ── Load raw arrays from datacubes ────────────────────────────────────────
    print("Loading datacubes …")
    (
        era5_dynamic,
        _ecmwf_forecast,   # not used by the persistence baseline
        _static_features,  # not used by the persistence baseline
        wse_labels,
        obs_mask,
        lake_ids,
        era5_dates,
        ecmwf_init_dates,
    ) = assemble_lake_features_from_datacubes(
        wse_datacube_path            = args.wse_datacube,
        era5_climate_datacube_path   = args.era5_datacube,
        ecmwf_forecast_datacube_path = args.ecmwf_datacube,
        static_datacube_path         = args.static_datacube,
    )
    print(f'Lakes: {len(lake_ids)}  |  ERA5 dates: {len(era5_dates)}  |  '
          f'ECMWF init dates: {len(ecmwf_init_dates)}')

    # ── Replicate the same train / val / test splits ───────────────────────────
    era5_date_to_idx = {d: i for i, d in enumerate(era5_dates)}

    all_valid = _find_valid_starts(
        ecmwf_init_dates, era5_date_to_idx, obs_mask,
        seq_len, forecast_horizon, require_obs=True,
    )
    n_valid   = len(all_valid)
    train_end = int(n_valid * args.train_frac)
    val_end   = int(n_valid * (args.train_frac + args.val_frac))
    train_idx = all_valid[:train_end]
    val_idx   = all_valid[train_end:val_end]
    test_idx  = all_valid[val_end:]
    print(f'Splits: {len(train_idx)} train | {len(val_idx)} val | {len(test_idx)} test samples')
    print(f'Forecast horizon: {forecast_horizon} day(s)\n')

    # ── Run persistence baseline on all splits ─────────────────────────────────
    print("Running last-observation baseline …")
    train_raw = run_last_obs_baseline(
        era5_dynamic, wse_labels, obs_mask, lake_ids,
        era5_dates, ecmwf_init_dates, train_idx,
        seq_len, forecast_horizon, split_name='train',
    )
    val_raw = run_last_obs_baseline(
        era5_dynamic, wse_labels, obs_mask, lake_ids,
        era5_dates, ecmwf_init_dates, val_idx,
        seq_len, forecast_horizon, split_name='val',
    )
    test_raw = run_last_obs_baseline(
        era5_dynamic, wse_labels, obs_mask, lake_ids,
        era5_dates, ecmwf_init_dates, test_idx,
        seq_len, forecast_horizon, split_name='test',
    )
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
    ]
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
    metric_cols = ['mse_m2', 'rmse_m', 'mae_m', 'nse', 'kge', 'autocorr']
    print('\n=== Median Per-Lake Metrics (5th-95th CI) ===')
    for col in metric_cols:
        med = lake_metrics_df[col].median()
        lo  = lake_metrics_df[col].quantile(0.05)
        hi  = lake_metrics_df[col].quantile(0.95)
        print(f'  {col:<12}: {med:6.3f}  ({lo:.3f} - {hi:.3f})')


if __name__ == "__main__":
    main()
