"""
Ablation inference: temporal expanding-window CV multi-day lake WSE forecasting
with SWOT observation features zeroed out.

Mirrors wsend_lstm_training/run_inference_lstm_wsend_temporal_cv.py exactly,
except that input features at indices 0-5 (obs_mask, latest_wse, wse_u,
wse_std, area_total, days_since_last_obs) are set to zero before every model
forward pass.  Indices 6-7 (doy_sin/cos) and 8-20 (ERA5 climate) are kept.

Must be paired with checkpoints trained by
ablation_noswot/run_training_lake_lstm_wsend_temporal_cv_noswot.py so that
the zeroing is consistent between training and inference.

Usage:
  python ablation_noswot/run_inference_lstm_wsend_temporal_cv_noswot.py \\
    --config          ../configs/lstm/exp02_mekong_lstm_wsend_era5_ifshres_gritv06_202312_202602_temporalcv.yaml \\
    --wse-datacube    /path/to/swot_lake_wse_datacube_wse_norm.nc \\
    --era5-datacube   /path/to/swot_lake_era5_climate_datacube.nc \\
    --ecmwf-datacube  /path/to/swot_lake_ecmwf_forecast_datacube.nc \\
    --static-datacube /path/to/swot_lake_static_datacube.nc \\
    --wse-stats-csv   /path/to/lake_wse_norm_stats.csv \\
    --lake-graph      /path/to/gritv06_great_mekong_pld_lake_graph_0sqkm.csv \\
    --save-dir        checkpoints \\
    --run-name        ablation_noswot_lstm_temporalcv \\
    --fold-idx        0 \\
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
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.temporal_cv import (
    build_temporal_cv_fold,
    N_TEMPORAL_FOLDS,
    TEMPORAL_FOLD_DATES,
)
from models.registry import MODEL_REGISTRY
from training.evaluate import compute_kge

# SWOT observation-based feature indices zeroed during training — must match
# _SWOT_OBS_FEAT_DIM in training/train_lstm_nd_noswot.py
_SWOT_OBS_FEAT_DIM = 6


# ── Metrics ───────────────────────────────────────────────────────────────────

def nse(obs, pred):
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
    obs, mu, sigma = np.array(obs), np.array(mu), np.array(sigma)
    ok = ~(np.isnan(obs) | np.isnan(mu) | np.isnan(sigma) | (sigma <= 0))
    if ok.sum() == 0:
        return np.nan
    z_crit = scipy_stats.norm.ppf(0.5 + alpha / 2)
    inside = (obs[ok] >= mu[ok] - z_crit * sigma[ok]) & \
             (obs[ok] <= mu[ok] + z_crit * sigma[ok])
    return float(inside.mean())


def pi_width(sigma, alpha=0.90):
    sigma  = np.array(sigma)
    ok     = ~(np.isnan(sigma) | (sigma <= 0))
    if ok.sum() == 0:
        return np.nan
    z_crit = scipy_stats.norm.ppf(0.5 + alpha / 2)
    return float(np.mean(2 * z_crit * sigma[ok]))


def residual_autocorr_lag1(obs, pred, dates=None):
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
    df = df.merge(lake_stats[["lake_id", "lake_mean", "lake_std"]],
                  on="lake_id", how="left")
    std_eps = df["lake_std"].fillna(1.0) + 1e-8
    mean_   = df["lake_mean"].fillna(0.0)
    df["pred_m"]   = df["pred_norm"]   * std_eps + mean_
    df["target_m"] = df["target_norm"] * std_eps + mean_
    if "pred_std_norm" in df.columns:
        df["pred_std_m"] = df["pred_std_norm"] * std_eps
    return df


def run_inference(ds, model, device, fold_idx: int, split_name: str = "") -> pd.DataFrame:
    """Run model over all samples with SWOT features zeroed; return observed-lake predictions."""
    records = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(ds)):
            data_list, static, labels, mask = ds[idx]

            x        = torch.stack([d.x for d in data_list], dim=1).to(device)
            static_t = static.to(device)

            # Zero SWOT observation features — must match training zeroing
            x[:, :, :_SWOT_OBS_FEAT_DIM] = 0.0

            pred_out = model(x, static_features=static_t)

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

            j         = int(ds.valid_starts[idx])
            init_date = pd.Timestamp(ds.ecmwf_init_dates[j])
            horizon   = labels_np.shape[1]

            for d in range(horizon):
                forecast_date = init_date + pd.Timedelta(days=d)
                for lake_i, lake_id in enumerate(ds.lake_ids):
                    if mask_np[lake_i, d] > 0:
                        records.append({
                            "lake_id"      : int(lake_id),
                            "fold_idx"     : fold_idx,
                            "init_date"    : init_date,
                            "forecast_date": forecast_date,
                            "lead_day"     : d,
                            "pred_norm"    : float(mean_cpu[lake_i, d]),
                            "target_norm"  : float(labels_np[lake_i, d]),
                            "pred_std_norm": float(std_cpu[lake_i, d]) if std_cpu is not None else float("nan"),
                        })

    df = pd.DataFrame(records)
    n_dates = df["init_date"].nunique() if len(df) > 0 else 0
    print(f"  {split_name:<6}: {len(df):>7} observed-lake rows  |  "
          f"{n_dates} forecast dates")
    return df


def compute_lake_metrics(df: pd.DataFrame, lead_day: int | None = None) -> pd.DataFrame:
    if lead_day is not None:
        df = df[df["lead_day"] == lead_day]

    has_std = "pred_std_m" in df.columns
    lake_rows = []
    for lake_id, sub in df.groupby("lake_id"):
        sub = sub.sort_values("forecast_date")
        if len(sub) < 2:
            continue
        obs  = sub["target_m"].values
        pred = sub["pred_m"].values
        mse  = float(np.mean((obs - pred) ** 2))
        row = {
            "lead_day": lead_day if lead_day is not None else -1,
            "lake_id" : lake_id,
            "n_obs"   : len(sub),
            "mse_m2"  : mse,
            "rmse_m"  : float(np.sqrt(mse)),
            "mae_m"   : float(np.mean(np.abs(obs - pred))),
            "nse"     : nse(obs, pred),
            "kge"     : compute_kge(obs, pred),
            "autocorr": residual_autocorr_lag1(obs, pred, sub["forecast_date"].values),
        }
        if has_std:
            sigma = sub["pred_std_m"].values
            row["crps_m"]  = crps_gaussian(obs, pred, sigma)
            row["cov90"]   = pi_coverage(obs, pred, sigma, alpha=0.90)
            row["piw90_m"] = pi_width(sigma, alpha=0.90)
        lake_rows.append(row)
    return pd.DataFrame(lake_rows)


def compute_metrics_by_lead_day(df: pd.DataFrame) -> dict:
    lead_days = [int(d) for d in sorted(df["lead_day"].unique())]
    return {d: compute_lake_metrics(df, lead_day=d) for d in lead_days}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation inference: temporal-CV LSTM with SWOT features zeroed"
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
    parser.add_argument("--fold-idx",  type=int, required=True)
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
    assert forecast_horizon > 1, (
        f"Designed for forecast_horizon > 1, got {forecast_horizon}."
    )

    if not 0 <= args.fold_idx < N_TEMPORAL_FOLDS:
        parser.error(f"--fold-idx must be 0, 1, or 2 (got {args.fold_idx})")

    model_type = cfg["model"].get("model_type", "LSTMBaselineMultiStep")
    model_slug = MODEL_REGISTRY[model_type].slug

    fold_def = TEMPORAL_FOLD_DATES[args.fold_idx]

    base_run_name = args.run_name
    run_name = (
        f"{args.run_name}"
        f"_fold{args.fold_idx}of{N_TEMPORAL_FOLDS}"
        f"_h{forecast_horizon}"
        f"_{model_slug}"
        f"_s{args.seed}"
    )

    run_dir  = Path(args.save_dir) / base_run_name / f"fold_{args.fold_idx}"
    cfg_path = run_dir / "run_config.yaml"

    print(f"Resolved run directory : {run_dir}")
    print(f"Fold [{args.fold_idx}]  : "
          f"train {fold_def['train_start']} → {fold_def['train_end']}  |  "
          f"test {fold_def['test_start']} → {fold_def['test_end']}")
    print(f"Forecast horizon       : {forecast_horizon} days")
    print(f"Ablation               : SWOT observation features (0-5) zeroed")

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Expected slug: {run_name}"
        )
    if not cfg_path.exists():
        existing     = sorted(run_dir.iterdir())
        existing_str = "\n  ".join(p.name for p in existing) or "(empty)"
        raise FileNotFoundError(
            f"run_config.yaml not found in {run_dir}\n"
            f"Files present:\n  {existing_str}"
        )

    with open(cfg_path) as f:
        run_config = yaml.safe_load(f)
    cfg_model    = run_config["model"]
    cfg_training = run_config["training"]
    print(f"Loaded config: best epoch={run_config['result']['best_epoch']}, "
          f"best val loss={run_config['result']['best_val_loss']:.6f}")

    ckpt_path = run_dir / run_config["result"]["checkpoint"]

    train_ds, val_ds, test_ds, norm_stats = build_temporal_cv_fold(
        wse_datacube_path            = args.wse_datacube,
        era5_climate_datacube_path   = args.era5_datacube,
        ecmwf_forecast_datacube_path = args.ecmwf_datacube,
        static_datacube_path         = args.static_datacube,
        lake_graph_path              = args.lake_graph,
        fold_idx                     = args.fold_idx,
        seq_len                      = cfg_training["seq_len"],
        forecast_horizon             = cfg_training["forecast_horizon"],
        val_frac                     = cfg_training.get("val_frac", 0.15),
        require_obs_on_any_forecast_day = True,
    )
    print(f"Splits : {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test init_dates")
    print(f"Lakes  : {test_ds.n_lakes} (all active in every fold)")

    device         = torch.device(args.device)
    cfg_model_load = dict(cfg_model)
    model_type_    = cfg_model_load.pop("model_type", "LSTMBaselineMultiStep")
    spec           = MODEL_REGISTRY[model_type_]
    model          = spec.model_cls(**cfg_model_load).to(device)

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    best_epoch = ckpt.get("best_epoch", "?")
    best_loss  = ckpt.get("best_val_loss", float("nan"))
    print(f"\nCheckpoint loaded — best epoch: "
          f"{best_epoch + 1 if isinstance(best_epoch, int) else best_epoch}"
          f"   best val loss: {best_loss:.6f}")
    print(f"Device: {device}")

    print("\nRunning inference (SWOT features zeroed) ...")
    train_raw = run_inference(train_ds, model, device, args.fold_idx, "train")
    val_raw   = run_inference(val_ds,   model, device, args.fold_idx, "val")
    test_raw  = run_inference(test_ds,  model, device, args.fold_idx, "test")
    print()

    lake_stats = pd.read_csv(args.wse_stats_csv)
    lake_stats = lake_stats[["lake_id", "lake_mean", "lake_std"]].drop_duplicates("lake_id")

    train_df = denormalize(train_raw, lake_stats)
    val_df   = denormalize(val_raw,   lake_stats)
    test_df  = denormalize(test_raw,  lake_stats)

    col_order = [
        "lake_id", "fold_idx", "init_date", "forecast_date", "lead_day",
        "pred_norm", "target_norm", "pred_m", "target_m",
        "pred_std_norm", "pred_std_m",
    ]
    for col in ("pred_std_norm", "pred_std_m"):
        for df in (train_df, val_df, test_df):
            if col not in df.columns:
                df[col] = float("nan")

    train_df[col_order].to_csv(run_dir / "train_result_df.csv", index=False)
    val_df[col_order].to_csv(run_dir   / "val_result_df.csv",   index=False)
    test_df[col_order].to_csv(run_dir  / "test_result_df.csv",  index=False)
    print(f"Result CSVs saved to {run_dir}/")
    print(f"  train: {len(train_df)} rows | val: {len(val_df)} | test: {len(test_df)}\n")

    lead_days = [int(d) for d in sorted(test_raw["lead_day"].unique())] if len(test_raw) > 0 else []

    splits_info = [("train", train_df), ("val", val_df), ("test", test_df)]
    for split_name, split_df in splits_info:
        if len(split_df) == 0:
            print(f"No rows for split '{split_name}', skipping metrics.")
            continue
        by_lead = compute_metrics_by_lead_day(split_df)
        agg_df  = pd.concat(by_lead.values(), ignore_index=True)
        agg_csv = run_dir / f"lake_metrics_{split_name}.csv"
        agg_df.to_csv(agg_csv, index=False)
        for d, mdf in by_lead.items():
            mdf.to_csv(run_dir / f"lake_metrics_{split_name}_lead{d}.csv", index=False)
        n_lakes_split = len(split_df["lake_id"].unique())
        print(f"Per-lake metrics saved ({split_name}, {n_lakes_split} lakes, "
              f"{len(by_lead)} lead days):")
        print(f"  -> {agg_csv.name}  (all lead days stacked)")
        for d in sorted(by_lead):
            print(f"  -> lake_metrics_{split_name}_lead{d}.csv")

    metric_cols = ["mse_m2", "rmse_m", "mae_m", "nse", "kge", "autocorr",
                   "crps_m", "cov90", "piw90_m"]

    def _print_lead_summary(label: str, split_df: pd.DataFrame) -> None:
        if len(split_df) == 0:
            return
        print(f"\n=== {label} — Median Per-Lake Metrics by Lead Day (5th–95th CI) ===")
        by_lead = compute_metrics_by_lead_day(split_df)
        for d, mdf in sorted(by_lead.items()):
            cols  = [c for c in metric_cols if c in mdf.columns]
            parts = []
            for col in cols:
                med = mdf[col].median()
                lo  = mdf[col].quantile(0.05)
                hi  = mdf[col].quantile(0.95)
                parts.append(f"{col}: {med:.3f} ({lo:.3f}–{hi:.3f})")
            print(f"  lead_day={d}  |  " + "  |  ".join(parts))

    _print_lead_summary("Train", train_df)
    _print_lead_summary("Val",   val_df)
    _print_lead_summary(
        f"Test [fold {args.fold_idx}: {fold_def['test_start']} → {fold_def['test_end']}]",
        test_df,
    )

    def _agg_metrics_json(split_df: pd.DataFrame) -> dict:
        if len(split_df) == 0:
            return {}
        result  = {}
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
        "ablation":         "noswot_input",
        "fold_idx":         args.fold_idx,
        "train_start":      fold_def["train_start"],
        "train_end":        fold_def["train_end"],
        "test_start":       fold_def["test_start"],
        "test_end":         fold_def["test_end"],
        "forecast_horizon": forecast_horizon,
        "n_lakes":          int(norm_stats["n_lakes"]),
        "n_lead_days":      len(lead_days),
        "lead_days":        lead_days,
        "train": _agg_metrics_json(train_df),
        "val":   _agg_metrics_json(val_df),
        "test":  _agg_metrics_json(test_df),
    }
    inf_metrics_path = run_dir / "inference_metrics.json"
    with open(inf_metrics_path, "w") as f:
        json.dump(inference_metrics, f, indent=2)
    print(f"\nAggregate inference metrics saved → {inf_metrics_path}")


if __name__ == "__main__":
    main()
