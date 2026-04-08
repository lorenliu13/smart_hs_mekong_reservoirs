"""
Build the SWOT lake WSE dynamic datacube for the lake-based SWOT-GNN.

Output:
  swot_lake_wse_datacube_{wse_option}.nc
      dims (lake, time) — SWOT WSE model input features + target variable
      vars: obs_mask, latest_wse, latest_wse_u, latest_wse_std,
            latest_area_total, days_since_last_obs, time_doy_sin,
            time_doy_cos,
            wse  ← target: WSE in wse_option form, NaN where not observed

Side file: lake_wse_norm_stats.csv (per-lake lake_mean, lake_std).

Usage:
    python build_wse_datacube_lake_based_20260319.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from datacube_utils import load_lake_ids_from_graph

# ─── Configuration ─────────────────────────────────────────────────────────────

SWOT_LAKE_WSE_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/swot/great_mekong_river_basin/lakes_daily"
    "/swot_lake_2023_12_2026_02_daily_wse_xtrk10_60km_dark50pct_qf01_daily_final.csv"
)
LAKE_GRAPH_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    "/gritv06_great_mekong_pld_lake_graph_0sqkm.csv"
)
SAVE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/mekong_lakes_swotpld_era5_ifshres10d_gritv06_202312_202602_qc"
)

# Specify as YYYY-MM; start expands to the 1st of the month, end to the last day (inclusive).
START_MONTH = "2023-10"
END_MONTH   = "2026-02"

# Which form of WSE to use as the "latest_wse" model input:
#   "wse_norm"    — (wse - per-lake mean) / per-lake std  ← default, dimensionless
#   "wse_anomaly" — wse - per-lake mean                   (same units as wse, m)
#   "wse"         — raw WSE in metres
WSE_OPTION = "wse_norm"   # "wse_norm" | "wse_anomaly" | "wse"

# ───────────────────────────────────────────────────────────────────────────────



def build_swot_wse_arrays(
    swot_csv: Path,
    lake_ids: np.ndarray,
    all_dates: pd.DatetimeIndex,
    wse_option: str = "wse_norm",
):
    """
    Load SWOT lake daily WSE and build per-lake model input and target arrays.

    Returns:
        mask_cube:               (n_lakes, n_dates) – 1 where SWOT observed, 0 otherwise
        latest_wse_cube:         (n_lakes, n_dates) – forward-filled WSE (wse_option form); 0 before first obs
        latest_wse_u_cube:       (n_lakes, n_dates) – forward-filled WSE uncertainty; 0 before first obs
        latest_wse_std_cube:     (n_lakes, n_dates) – forward-filled within-pass WSE std; 0 before first obs
        latest_area_total_cube:  (n_lakes, n_dates) – forward-filled total water area (m²); 0 before first obs
        lag_cube:                (n_lakes, n_dates) – days since last SWOT observation; 0 before first obs
        doy_sin_cube:            (n_lakes, n_dates) – sin(2π × doy / 365.25)
        doy_cos_cube:            (n_lakes, n_dates) – cos(2π × doy / 365.25)
        norm_stats_df:           DataFrame with columns lake_id, lake_mean, lake_std
        wse_cube:                (n_lakes, n_dates) – target: WSE in wse_option form, NaN where not observed
    """
    print("Loading SWOT lake WSE data …")
    swot_df = pd.read_csv(swot_csv)
    swot_df["date"] = pd.to_datetime(swot_df["date"])
    swot_df["lake_id"] = swot_df["lake_id"].astype(np.int64)
    # only keep the lakes that are in the lake graph (and thus in the other datacubes)
    swot_df = swot_df[swot_df["lake_id"].isin(lake_ids)]

    # ── Compute per-lake normalization statistics ────────────────────────────
    grp = swot_df.groupby("lake_id")["wse"]
    wse_mean = grp.mean().rename("lake_mean")
    wse_std  = grp.std().rename("lake_std").fillna(1.0).clip(lower=1e-8)
    norm_stats_df = pd.DataFrame({"lake_mean": wse_mean, "lake_std": wse_std}).reset_index()

    swot_df = swot_df.merge(norm_stats_df, on="lake_id", how="left")
    swot_df["wse_anomaly"] = swot_df["wse"] - swot_df["lake_mean"]
    swot_df["wse_norm"]    = swot_df["wse_anomaly"] / (swot_df["lake_std"] + 1e-8)

    # ── Allocate output arrays ───────────────────────────────────────────────
    n_lakes = len(lake_ids)
    n_dates = len(all_dates)
    shape   = (n_lakes, n_dates)

    mask_cube                = np.zeros(shape,        dtype=np.int8)
    latest_wse_cube          = np.zeros(shape,        dtype=np.float32)
    latest_wse_u_cube        = np.zeros(shape,        dtype=np.float32)
    latest_wse_std_cube      = np.zeros(shape,        dtype=np.float32)
    latest_area_total_cube   = np.zeros(shape,        dtype=np.float32)
    lag_cube                 = np.zeros(shape,        dtype=np.float32)
    wse_cube                 = np.full(shape, np.nan, dtype=np.float32)

    # ── Day-of-year cyclical encoding ───────────────────────────────────────
    doy = all_dates.dayofyear.to_numpy().astype(np.float32)
    time_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    time_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    doy_sin_cube = np.tile(time_sin, (n_lakes, 1))
    doy_cos_cube = np.tile(time_cos, (n_lakes, 1))

    lake_set    = set(lake_ids.tolist())
    lake_to_idx = {lid: i for i, lid in enumerate(lake_ids)}

    for lake_id, lake_df in tqdm(swot_df.groupby("lake_id"), desc="Building SWOT WSE arrays"):
        if lake_id not in lake_set:
            continue
        i = lake_to_idx[lake_id]
        lake_df = lake_df.set_index("date")

        full_series = lake_df[wse_option].reindex(all_dates)
        mask       = (~full_series.isna()).astype(np.int8).values
        latest_wse = full_series.ffill().fillna(0.0).values

        valid = mask.astype(float)
        last_valid_idx = np.where(valid == 1, np.arange(n_dates, dtype=float), np.nan)
        last_valid_idx = pd.Series(last_valid_idx).ffill().to_numpy()
        last_valid_idx = np.where(np.isnan(last_valid_idx), 0, last_valid_idx)
        lag = np.arange(n_dates, dtype=np.float32) - last_valid_idx

        mask_cube[i, :]       = mask
        latest_wse_cube[i, :] = latest_wse.astype(np.float32)
        lag_cube[i, :]        = lag.astype(np.float32)

        wse_u_series = lake_df["wse_u"].reindex(all_dates)
        latest_wse_u_cube[i, :] = wse_u_series.ffill().fillna(0.0).values.astype(np.float32)

        wse_std_series = lake_df["wse_std"].reindex(all_dates)
        latest_wse_std_cube[i, :] = wse_std_series.ffill().fillna(0.0).values.astype(np.float32)

        wse_cube[i, :] = full_series.values.astype(np.float32)  # sparse: NaN where not observed

        area_series = lake_df["area_total"].reindex(all_dates)
        latest_area_total_cube[i, :] = area_series.ffill().fillna(0.0).values.astype(np.float32)

    return (
        mask_cube,
        latest_wse_cube,
        latest_wse_u_cube,
        latest_wse_std_cube,
        latest_area_total_cube,
        lag_cube,
        doy_sin_cube,
        doy_cos_cube,
        norm_stats_df,
        wse_cube,
    )


def build_wse_datacube(
    swot_csv: Path,
    lake_ids: np.ndarray,
    all_dates: pd.DatetimeIndex,
    wse_option: str,
    save_dir: Path,
) -> Path:
    """
    Assemble and save the SWOT lake WSE dynamic datacube.

    Dims: (lake, time)
    Input features: obs_mask, latest_wse, latest_wse_u, latest_wse_std,
                    latest_area_total, days_since_last_obs, time_doy_sin, time_doy_cos
    Target:         wse — WSE in wse_option form, NaN where not observed
    """
    print("\n=== Building SWOT WSE datacube ===")

    (mask_cube, latest_wse_cube, latest_wse_u_cube,
     latest_wse_std_cube, latest_area_total_cube, lag_cube,
     doy_sin_cube, doy_cos_cube, norm_stats_df,
     wse_cube) = build_swot_wse_arrays(
        swot_csv, lake_ids, all_dates, wse_option
    )

    ds = xr.Dataset(
        data_vars={
            "obs_mask":              (["lake", "time"], mask_cube.astype(np.int8)),
            "latest_wse":            (["lake", "time"], latest_wse_cube),
            "latest_wse_u":          (["lake", "time"], latest_wse_u_cube),
            "latest_wse_std":        (["lake", "time"], latest_wse_std_cube),
            "latest_area_total":     (["lake", "time"], latest_area_total_cube),
            "days_since_last_obs":   (["lake", "time"], lag_cube),
            "time_doy_sin":          (["lake", "time"], doy_sin_cube),
            "time_doy_cos":          (["lake", "time"], doy_cos_cube),
            "wse":                    (["lake", "time"], wse_cube),
        },
        coords={
            "lake": lake_ids,
            "time": all_dates,
        },
        attrs={
            "description": "SWOT lake WSE dynamic datacube for lake-SWOT-GNN",
            "wse_option": wse_option,
            "created_by": "build_wse_datacube_lake_based_20260319.py",
        },
    )

    out_path = save_dir / f"swot_lake_wse_datacube_{wse_option}.nc"
    ds.to_netcdf(out_path)
    print(f"WSE datacube saved → {out_path}  shape: {len(lake_ids)} lakes × {len(all_dates)} days")

    stats_path = save_dir / "lake_wse_norm_stats.csv"
    norm_stats_df.to_csv(stats_path, index=False)
    print(f"WSE norm stats saved → {stats_path}")

    return out_path


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading lake IDs from: {LAKE_GRAPH_CSV}")
    lake_ids = load_lake_ids_from_graph(LAKE_GRAPH_CSV)
    print(f"  Found {len(lake_ids)} lakes in GRIT PLD lake graph.")

    lake_graph_save_path = SAVE_DIR / LAKE_GRAPH_CSV.name
    pd.read_csv(LAKE_GRAPH_CSV).to_csv(lake_graph_save_path, index=False)
    print(f"Lake graph CSV saved → {lake_graph_save_path}")

    start_date = pd.Timestamp(START_MONTH + "-01")
    end_date   = pd.Timestamp(END_MONTH   + "-01") + pd.offsets.MonthEnd(0)
    all_dates = pd.date_range(start_date, end_date, freq="D")

    build_wse_datacube(
        swot_csv=SWOT_LAKE_WSE_CSV,
        lake_ids=lake_ids,
        all_dates=all_dates,
        wse_option=WSE_OPTION,
        save_dir=SAVE_DIR,
    )
