"""
Inference script for Exp01: 1-day-ahead lake WSE forecasting.

Loads the best checkpoint saved by run_lake_exp01.py, re-builds the exact
train / val / test splits from the original datacubes, runs forward passes,
denormalises predictions, and writes result CSVs + per-lake metrics.

Usage:
  python run_inference_lake.py \\
    --wse-datacube    /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube   /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube  /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --wse-stats-csv   /path/to/lake_wse_norm_stats.csv \\
    --lake-graph      /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --save-dir        checkpoints \\
    --run-name        exp01_mekong_wse1d_era5_ifshres_gritv06_202312_202512_v01 \\
    --device          cuda
"""
import argparse
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.temporal_graph_dataset_lake import build_temporal_dataset_from_lake_datacubes
from models.swot_gnn import SWOTGNN
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


# ── Core functions ────────────────────────────────────────────────────────────

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


def run_inference(ds, model, device, split_name=''):
    """Run model over all samples; return DataFrame of observed-lake predictions.

    Supports forecast_horizon >= 1. Each row records:
        lake_id, init_date, forecast_date, lead_day, pred_norm, target_norm
    """
    records = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(ds)):
            data_list, static, labels, mask = ds[idx]

            x          = torch.stack([d.x for d in data_list], dim=1).to(device)
            edge_index = data_list[0].edge_index.to(device)
            static_t   = static.to(device)

            pred     = model(x, edge_index, static_features=static_t)
            pred_cpu = pred.cpu().numpy()

            # Normalise to 2-D (n_lakes, horizon) regardless of forecast_horizon
            labels_np = labels.numpy()
            mask_np   = mask.numpy()
            if labels_np.ndim == 1:          # forecast_horizon == 1 backward compat
                labels_np = labels_np[:, np.newaxis]
                mask_np   = mask_np[:, np.newaxis]
                pred_cpu  = pred_cpu[:, np.newaxis]

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
                            'pred_norm'    : float(pred_cpu[lake_i, d]),
                            'target_norm'  : float(labels_np[lake_i, d]),
                        })

    df = pd.DataFrame(records)
    print(f'  {split_name:<6}: {len(df):>6} observed-lake rows  |  '
          f'{df["init_date"].nunique()} forecast dates')
    return df


def compute_lake_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-lake metrics (physical units) on the test split."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp01: inference — load checkpoint, run predictions, save CSVs"
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
    parser.add_argument("--run-name",  default="exp01_nextday_wse_v1",
                        help="Run subfolder name (must match the training run)")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    run_dir   = Path(args.save_dir) / args.run_name
    ckpt_path = run_dir / "best_model.pt"
    cfg_path  = run_dir / "run_config.yaml"

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
    device = torch.device(args.device)
    model  = SWOTGNN(**cfg_model).to(device)

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
