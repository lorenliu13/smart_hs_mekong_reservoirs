"""
Prepare SWOT lake WSE training datacubes and assemble lake node features
for the lake-based SWOT-GNN.

Outputs three NetCDF files:
  1. swot_lake_era5_dynamic_datacube_{wse_option}.nc
       dims (lake, time) — SWOT WSE features + ERA5-Land climate (13 vars)
  2. swot_lake_ecmwf_forecast_datacube.nc
       dims (lake, init_time, forecast_day) — ECMWF IFS climate (13 vars)
  3. swot_lake_static_datacube.nc
       dims (lake, feature) — placeholder zeros (static attrs to be added later)

Plus a side file: lake_wse_norm_stats.csv (per-lake lake_mean, lake_std).

Run on the cluster where ERA5/ECMWF per-lake CSVs are available.

Usage:
    python training_data_processing_lake_based_20260319.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────

# --- Cluster data paths ---
# Root directory for ERA5-Land per-lake CSV files (one sub-folder per variable)
ERA5_BASE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data"
    "/mekong_river_basin_reservoirs/era5_land_daily_catchment_level"
    "/era5land_daily_per_pld_lake_0sqkm"
)
# Root directory for ECMWF IFS per-lake CSV files (one sub-folder per variable)
ECMWF_BASE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data"
    "/mekong_river_basin_reservoirs/ecmwf_ifs_daily_catchment_level"
    "/hres/ecmwf_ifs_daily_per_pld_lake_0sqkm"
)
# SWOT daily lake water surface elevation (WSE) observations, pre-filtered
# with cross-track distance ≤10–60 km, dark-water <50 %, and quality flag = 0/1
SWOT_LAKE_WSE_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/swot/mekong_river_basin/swot/lakes_daily"
    "/swot_lake_daily_wse_xtrk10_60km_dark50pct_qf01_lake_graph_0sqkm.csv"
)
# GRIT PLD lake graph CSV — defines which lake nodes (lake_id) exist in the graph
LAKE_GRAPH_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    "/gritv06_pld_lake_graph_0sqkm.csv"
)
# Where the three output NetCDF datacubes (+ norm stats CSV) will be written
SAVE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/training_data_lake_based_20260319"
)
# Lake static attributes CSV — one row per lake
LAKE_STATIC_ATTRS_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/training_data_lake_based_20260319/lake_graph_static_attrs_0sqkm.csv"
)
# Columns to drop from the static attributes CSV before building the datacube
STATIC_EXCLUDE_COLS = [
    "lake_id",
    "domain",
    "mean_annual_discharge",
    "mean_sum_runoff",
    "mean_annual_discharge_da",
    "mean_sum_runoff_da",
]

# Date range for the ERA5 reanalysis and SWOT WSE datacube.
# ECMWF init_dates are not fixed here — they are discovered automatically
# by scanning the ECMWF per-lake CSV directory (see _determine_ecmwf_init_dates).
START_DATE = "2023-10-01"
END_DATE   = "2025-12-31"

# Which form of WSE to use as the "latest_wse" model input:
#   "wse_norm"    — (wse - per-lake mean) / per-lake std  ← default, dimensionless
#   "wse_anomaly" — wse - per-lake mean                   (same units as wse, m)
#   "wse"         — raw WSE in metres
WSE_OPTION = "wse_norm"   # "wse_norm" | "wse_anomaly" | "wse"

# Number of lead-days in the ECMWF IFS forecast per initialisation date.
# Forecasts are stored as forecast_day = 1 … FORECAST_HORIZON.
FORECAST_HORIZON = 10

# Raw ERA5-Land / ECMWF IFS variable names as they appear in the per-lake CSV files.
# These are used to load data; derive_climate_vars() then converts them into
# the 13 model-ready climate variables (unit conversions, wind speed magnitude, etc.)
ERA5_RAW_VARS = [
    "tp",    # total precipitation            (m)
    "t2m",   # 2-m air temperature            (K)
    "d2m",   # 2-m dewpoint temperature       (K)
    "sp",    # surface pressure               (Pa)
    "u10",   # 10-m U-component of wind       (m/s)
    "v10",   # 10-m V-component of wind       (m/s)
    "ssrd",  # surface shortwave radiation↓   (J/m²/day)
    "strd",  # surface longwave  radiation↓   (J/m²/day)
    "sf",    # snowfall                        (m of water equiv.)
    "sd",    # snow depth                      (m of water equiv.)
    "swvl1", # volumetric soil water layer 1  (m³/m³, 0–7 cm)
    "swvl2", # volumetric soil water layer 2  (m³/m³, 7–28 cm)
    "swvl3", # volumetric soil water layer 3  (m³/m³, 28–100 cm)
    "swvl4", # volumetric soil water layer 4  (m³/m³, 100–289 cm)
]

# ─── Feature ordering ──────────────────────────────────────────────────────────

# Ordered list of the 21 model input features extracted from the ERA5 dynamic
# datacube when assembling training tensors (see assemble_lake_features_from_datacubes).
# Features 0–7 are SWOT-derived; features 8–20 are ERA5-Land climate variables.
#
# Two extra variables (wse_std, area_total) are also stored in the datacube
# but are NOT in this list — they are NaN-sparse "support" tensors used for
# loss weighting and evaluation rather than as model inputs.
ERA5_INPUT_VARS: List[str] = [
    "obs_mask",              # 0:  1 where SWOT observed, 0 otherwise
    "latest_wse",            # 1:  forward-filled normalised WSE (0 before first obs)
    "latest_wse_u",          # 2:  forward-filled WSE uncertainty (0 before first obs)
    "latest_wse_std",        # 3:  forward-filled WSE std         (0 before first obs)
    "latest_area_total",     # 4:  forward-filled total water area (0 before first obs)
    "days_since_last_obs",   # 5:  days since last SWOT pass
    "time_doy_sin",          # 6:  sin(2π × doy / 365.25)
    "time_doy_cos",          # 7:  cos(2π × doy / 365.25)
    "LWd",                   # 8:  longwave downward radiation   (W/m²)
    "SWd",                   # 9:  shortwave downward radiation  (W/m²)
    "P",                     # 10: precipitation                 (mm)
    "Pres",                  # 11: surface pressure              (Pa)
    "Temp",                  # 12: 2-m air temperature           (K)
    "Td",                    # 13: 2-m dewpoint temperature      (K)
    "Wind",                  # 14: 10-m wind speed               (m/s)
    "sf",                    # 15: snowfall                      (mm)
    "sd",                    # 16: snow depth                    (mm)
    "swvl1",                 # 17: soil moisture layer 1         (m³/m³)
    "swvl2",                 # 18: soil moisture layer 2         (m³/m³)
    "swvl3",                 # 19: soil moisture layer 3         (m³/m³)
    "swvl4",                 # 20: soil moisture layer 4         (m³/m³)
]

# The 13 climate variables stored in the ECMWF IFS forecast datacube.
# Their order and naming exactly match the climate block at indices 8–20 of
# ERA5_INPUT_VARS so that ERA5 reanalysis and ECMWF forecast tensors can be
# concatenated / substituted directly in the model without re-ordering.
ECMWF_CLIMATE_VARS: List[str] = [
    "LWd", "SWd", "P", "Pres", "Temp", "Td", "Wind",
    "sf", "sd", "swvl1", "swvl2", "swvl3", "swvl4",
]

# Number of SWOT-derived features (obs_mask … time_doy_cos, indices 0–7)
SWOT_DIM: int = 8
# Number of climate features (LWd … swvl4, indices 8–20)
CLIMATE_DIM: int = 13

# Feature indices that must be zeroed when the model is run in forecast mode
# (future timesteps have no SWOT observations, so all SWOT features are 0).
# Indices 6–7 (doy_sin / doy_cos) are kept because the date is still known.
WSE_LAKE_DYNAMIC_INDICES: List[int] = [0, 1, 2, 3, 4, 5]  # obs_mask … days_since_last_obs

# ───────────────────────────────────────────────────────────────────────────────


def load_lake_ids_from_graph(lake_graph_csv: Path) -> np.ndarray:
    """
    Extract unique lake IDs from the GRIT PLD lake graph CSV.
    Excludes the terminal node (-1).

    Returns:
        Sorted int64 array of lake IDs.
    """
    lake_graph = pd.read_csv(lake_graph_csv)
    ids = lake_graph['lake_id'].to_numpy(dtype=np.int64)
    ids = np.unique(ids)
    ids = ids[ids != -1]  # exclude terminal node
    return ids


def derive_climate_vars(raw: dict) -> dict:
    """
    Derive 13 model climate variables from raw ERA5-Land / ECMWF IFS variables.

    Accepts arrays of any shape (2-D or 3-D). Element-wise numpy operations
    broadcast correctly for both ERA5 (n_lakes, n_dates) and ECMWF
    (n_lakes, n_init_dates, forecast_horizon) arrays.

    Returns:
        Dict with keys: LWd, SWd, P, Pres, Temp, Td, Wind, sf, sd,
                        swvl1, swvl2, swvl3, swvl4
    """
    return {
        "LWd":   (raw["strd"]  / 86400.0).astype(np.float32),   # J/m² → W/m²
        "SWd":   (raw["ssrd"]  / 86400.0).astype(np.float32),   # J/m² → W/m²
        "P":     (raw["tp"]    * 1000.0).astype(np.float32),    # m   → mm
        "Pres":  raw["sp"].astype(np.float32),                   # Pa
        "Temp":  raw["t2m"].astype(np.float32),                  # K
        "Td":    raw["d2m"].astype(np.float32),                  # K  (2-m dewpoint)
        "Wind":  np.sqrt(raw["u10"]**2 + raw["v10"]**2).astype(np.float32),  # m/s
        "sf":    (raw["sf"]    * 1000.0).astype(np.float32),    # m   → mm (snowfall)
        "sd":    (raw["sd"]    * 1000.0).astype(np.float32),    # m   → mm (snow depth)
        "swvl1": raw["swvl1"].astype(np.float32),                # m³/m³
        "swvl2": raw["swvl2"].astype(np.float32),                # m³/m³
        "swvl3": raw["swvl3"].astype(np.float32),                # m³/m³
        "swvl4": raw["swvl4"].astype(np.float32),                # m³/m³
    }


def build_swot_wse_arrays(
    swot_csv: Path,
    lake_ids: np.ndarray,
    all_dates: pd.DatetimeIndex,
    wse_option: str = "wse_norm",
):
    """
    Load SWOT lake daily WSE and compute per-lake feature and support arrays.

    Returns:
        mask_cube:               (n_lakes, n_dates) – 1 where observed
        latest_wse_cube:         (n_lakes, n_dates) – forward-filled normalized WSE
        latest_wse_u_cube:       (n_lakes, n_dates) – forward-filled WSE uncertainty
        latest_wse_std_cube:     (n_lakes, n_dates) – forward-filled WSE std
        latest_area_total_cube:  (n_lakes, n_dates) – forward-filled total water area
        lag_cube:                (n_lakes, n_dates) – days since last observation
        doy_sin_cube:            (n_lakes, n_dates) – sin(2π * doy / 365.25)
        doy_cos_cube:            (n_lakes, n_dates) – cos(2π * doy / 365.25)
        norm_stats_df:           DataFrame with columns lake_id, lake_mean, lake_std
        wse_std_cube:            (n_lakes, n_dates) – WSE std, NaN where not observed
        area_total_cube:         (n_lakes, n_dates) – total water area (m²), NaN where not observed
    """
    print("Loading SWOT lake WSE data …")
    swot_df = pd.read_csv(swot_csv)
    swot_df["date"] = pd.to_datetime(swot_df["date"])
    swot_df["lake_id"] = swot_df["lake_id"].astype(np.int64)

    # ── Compute per-lake normalization statistics ────────────────────────────
    # wse_std is clipped to avoid division by zero for lakes with constant WSE.
    grp = swot_df.groupby("lake_id")["wse"]
    wse_mean = grp.mean().rename("lake_mean")
    wse_std  = grp.std().rename("lake_std").fillna(1.0).clip(lower=1e-8)
    norm_stats_df = pd.DataFrame({"lake_mean": wse_mean, "lake_std": wse_std}).reset_index()

    # Add anomaly and normalized WSE columns to the raw dataframe so the
    # caller can choose which form to store via wse_option.
    swot_df = swot_df.merge(norm_stats_df, on="lake_id", how="left")
    swot_df["wse_anomaly"] = swot_df["wse"] - swot_df["lake_mean"]
    swot_df["wse_norm"]    = swot_df["wse_anomaly"] / (swot_df["lake_std"] + 1e-8)

    # ── Allocate output arrays ───────────────────────────────────────────────
    n_lakes = len(lake_ids) # number of lakes
    n_dates = len(all_dates) # number of observation dates
    shape   = (n_lakes, n_dates)

    # Feature arrays — zero-initialised (missing = 0 before first observation)
    mask_cube                = np.zeros(shape,        dtype=np.int8)    # observation indicator
    latest_wse_cube          = np.zeros(shape,        dtype=np.float32) # forward-filled WSE
    latest_wse_u_cube        = np.zeros(shape,        dtype=np.float32) # forward-filled uncertainty
    latest_wse_std_cube      = np.zeros(shape,        dtype=np.float32) # forward-filled within-pass std
    latest_area_total_cube   = np.zeros(shape,        dtype=np.float32) # forward-filled water area
    lag_cube                 = np.zeros(shape,        dtype=np.float32) # days since last pass
    # Support arrays — NaN-initialised so the model/loss function can detect
    # which timesteps actually had a SWOT observation.
    wse_std_cube             = np.full(shape, np.nan, dtype=np.float32)
    area_total_cube          = np.full(shape, np.nan, dtype=np.float32)

    # ── Day-of-year cyclical encoding ───────────────────────────────────────
    # Encode calendar seasonality as (sin, cos) pair so 31-Dec and 1-Jan are
    # spatially close in the feature space (avoids discontinuity at year wrap).
    # The same values apply to every lake, so tile across the lake dimension.
    doy = all_dates.dayofyear.to_numpy().astype(np.float32)
    time_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    time_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    doy_sin_cube = np.tile(time_sin, (n_lakes, 1))  # (n_lakes, n_dates)
    doy_cos_cube = np.tile(time_cos, (n_lakes, 1))

    # Build fast lookup structures for the inner loop
    lake_set    = set(lake_ids.tolist())
    lake_to_idx = {lid: i for i, lid in enumerate(lake_ids)}

    for lake_id, lake_df in tqdm(swot_df.groupby("lake_id"), desc="Building SWOT WSE arrays"):
        # Skip lakes that are in the SWOT CSV but not in the lake graph
        if lake_id not in lake_set:
            continue
        i = lake_to_idx[lake_id]
        lake_df = lake_df.set_index("date")

        # Reindex onto the full daily grid; missing dates become NaN
        full_series = lake_df[wse_option].reindex(all_dates)

        # obs_mask = 1 only on days when SWOT actually observed this lake
        mask       = (~full_series.isna()).astype(np.int8).values
        # Forward-fill carries the last observed WSE forward; pre-observation
        # days default to 0 (fillna), matching how the model treats unobserved lakes.
        latest_wse = full_series.ffill().fillna(0.0).values

        # ── Days since last observation ──────────────────────────────────────
        # Strategy: replace each observed timestep with its date index (integer),
        # forward-fill to propagate to subsequent unobserved days, then subtract
        # from the running date index to get the lag.  Timesteps before the first
        # observation are given lag = 0 (treated as "never seen", not as day 0).
        valid = mask.astype(float)
        last_valid_idx = np.where(valid == 1, np.arange(n_dates, dtype=float), np.nan)
        last_valid_idx = pd.Series(last_valid_idx).ffill().to_numpy()
        last_valid_idx = np.where(np.isnan(last_valid_idx), 0, last_valid_idx)
        lag = np.arange(n_dates, dtype=np.float32) - last_valid_idx

        mask_cube[i, :]       = mask
        latest_wse_cube[i, :] = latest_wse.astype(np.float32)
        lag_cube[i, :]        = lag.astype(np.float32)

        # WSE uncertainty (instrument noise proxy) — forward-filled model feature
        wse_u_series = lake_df["wse_u"].reindex(all_dates)
        latest_wse_u_cube[i, :] = wse_u_series.ffill().fillna(0.0).values.astype(np.float32)

        # Within-pass WSE standard deviation (spatial spread across sub-reaches):
        #   - Forward-filled version  → model input (latest_wse_std_cube)
        #   - Sparse NaN version      → evaluation / loss support (wse_std_cube)
        wse_std_series = lake_df["wse_std"].reindex(all_dates)
        latest_wse_std_cube[i, :]  = wse_std_series.ffill().fillna(0.0).values.astype(np.float32)
        wse_std_cube[i, :]         = wse_std_series.values.astype(np.float32)

        # Total water surface area (m²):
        #   - Forward-filled version  → model input (latest_area_total_cube)
        #   - Sparse NaN version      → evaluation / loss support (area_total_cube)
        area_series = lake_df["area_total"].reindex(all_dates)
        latest_area_total_cube[i, :] = area_series.ffill().fillna(0.0).values.astype(np.float32)
        area_total_cube[i, :]        = area_series.values.astype(np.float32)

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
        wse_std_cube,
        area_total_cube,
    )


def load_era5_climate_arrays(
    era5_base_dir: Path,
    raw_vars: list,
    lake_ids: np.ndarray,
    all_dates: pd.DatetimeIndex,
) -> dict:
    """
    Load ERA5-Land per-lake CSV files and return 13 derived climate arrays,
    each of shape (n_lakes, n_dates).

    CSV format: lake_id, date, {var_name}, extraction_method
    Files:      era5_base_dir/{var}/era5land_per_lake_{var}_YYYY-MM.csv
    """
    lake_id_set = set(lake_ids.tolist())
    all_dates_set = set(all_dates.tolist())

    # ERA5-Land files are split by variable and month:
    #   era5_base_dir/{var}/era5land_per_lake_{var}_{YYYY}-{MM}.csv
    # Collect the unique year-month pairs needed to cover all_dates.
    year_months = sorted({(d.year, d.month) for d in all_dates})

    raw_dict = {}
    for var in raw_vars:
        print(f"  Loading ERA5 variable: {var} …")
        frames = []
        for (year, month) in year_months:
            fpath = era5_base_dir / var / f"era5land_per_lake_{var}_{year}-{month:02d}.csv"
            if not fpath.exists():
                print(f"    [WARN] Missing: {fpath}")
                continue
            df = pd.read_csv(fpath)
            df["date"] = pd.to_datetime(df["date"])
            df["lake_id"] = df["lake_id"].astype(np.int64)
            frames.append(df)

        if not frames:
            # No data at all for this variable — fall back to zeros so the
            # pipeline can still run (model will see zeroed-out features).
            print(f"  [WARN] No files found for variable {var}; filling with zeros.")
            raw_dict[var] = np.zeros((len(lake_ids), len(all_dates)), dtype=np.float32)
            continue

        # Concatenate monthly chunks, then filter to only the lakes and dates
        # we actually need (avoids a huge pivot table with irrelevant rows).
        combined = pd.concat(frames, ignore_index=True)
        combined = combined[
            combined["lake_id"].isin(lake_id_set) &
            combined["date"].isin(all_dates_set)
        ]

        # Pivot to (lake_id × date) matrix, then reindex to guarantee the exact
        # lake and date order matching lake_ids / all_dates.  Missing cells → NaN.
        pivot = (
            combined
            .pivot_table(index="lake_id", columns="date", values=var, aggfunc="first")
            .reindex(index=lake_ids, columns=all_dates)
        )
        arr = pivot.to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)   # replace NaN gaps with 0
        raw_dict[var] = arr

    # Convert raw ERA5 variables to the 13 model-ready climate variables
    return derive_climate_vars(raw_dict)


def build_era5_dynamic_datacube(
    swot_csv: Path,
    era5_base_dir: Path,
    lake_ids: np.ndarray,
    all_dates: pd.DatetimeIndex,
    wse_option: str,
    save_dir: Path,
) -> Path:
    """
    Assemble and save the ERA5 + SWOT WSE dynamic datacube.

    Dims: (lake, time)
    Variables: obs_mask, latest_wse, latest_wse_u, latest_wse_std, latest_area_total,
               days_since_last_obs, time_doy_sin, time_doy_cos (SWOT features),
               wse_std, area_total (support — NaN where not observed),
               LWd, SWd, P, Pres, Temp, Td, Wind, sf, sd,
               swvl1, swvl2, swvl3, swvl4
    """
    print("\n=== Building ERA5 dynamic datacube ===")

    # Step 1: Build all SWOT-derived feature arrays (8 features + 2 support arrays)
    (mask_cube, latest_wse_cube, latest_wse_u_cube,
     latest_wse_std_cube, latest_area_total_cube, lag_cube,
     doy_sin_cube, doy_cos_cube, norm_stats_df,
     wse_std_cube, area_total_cube) = build_swot_wse_arrays(
        swot_csv, lake_ids, all_dates, wse_option
    )

    # Step 2: Load ERA5-Land reanalysis and derive the 13 climate features
    print("Loading ERA5-Land climate data …")
    climate = load_era5_climate_arrays(era5_base_dir, ERA5_RAW_VARS, lake_ids, all_dates)

    # Step 3: Pack everything into an xarray Dataset with named dims and coords
    # so downstream code can use label-based indexing (ds.sel(lake=..., time=...)).
    ds = xr.Dataset(
        data_vars={
            # SWOT features
            "obs_mask":              (["lake", "time"], mask_cube.astype(np.int8)),
            "latest_wse":            (["lake", "time"], latest_wse_cube),
            "latest_wse_u":          (["lake", "time"], latest_wse_u_cube),
            "latest_wse_std":        (["lake", "time"], latest_wse_std_cube),
            "latest_area_total":     (["lake", "time"], latest_area_total_cube),
            "days_since_last_obs":   (["lake", "time"], lag_cube),
            "time_doy_sin":          (["lake", "time"], doy_sin_cube),
            "time_doy_cos":          (["lake", "time"], doy_cos_cube),
            # Support (NaN where not observed)
            "wse_std":               (["lake", "time"], wse_std_cube),
            "area_total":            (["lake", "time"], area_total_cube),
            # ERA5-Land derived climate
            "LWd":    (["lake", "time"], climate["LWd"]),
            "SWd":    (["lake", "time"], climate["SWd"]),
            "P":      (["lake", "time"], climate["P"]),
            "Pres":   (["lake", "time"], climate["Pres"]),
            "Temp":   (["lake", "time"], climate["Temp"]),
            "Td":     (["lake", "time"], climate["Td"]),
            "Wind":   (["lake", "time"], climate["Wind"]),
            "sf":     (["lake", "time"], climate["sf"]),
            "sd":     (["lake", "time"], climate["sd"]),
            "swvl1":  (["lake", "time"], climate["swvl1"]),
            "swvl2":  (["lake", "time"], climate["swvl2"]),
            "swvl3":  (["lake", "time"], climate["swvl3"]),
            "swvl4":  (["lake", "time"], climate["swvl4"]),
        },
        coords={
            "lake": lake_ids,
            "time": all_dates,
        },
        attrs={
            "description": "SWOT lake WSE + ERA5-Land climate dynamic datacube for lake-SWOT-GNN",
            "wse_option": wse_option,
            "created_by": "training_data_processing_lake_based_20260319.py",
        },
    )

    # wse_option is encoded in the filename so multiple normalization variants
    # can coexist in the same save_dir without overwriting each other.
    out_path = save_dir / f"swot_lake_era5_dynamic_datacube_{wse_option}.nc"
    ds.to_netcdf(out_path)
    print(f"ERA5 dynamic datacube saved → {out_path}  shape: {len(lake_ids)} lakes × {len(all_dates)} days")

    # Save per-lake lake_mean / lake_std so that predictions can be
    # de-normalized back to real WSE (metres) during inference / evaluation.
    stats_path = save_dir / "lake_wse_norm_stats.csv"
    norm_stats_df.to_csv(stats_path, index=False)
    print(f"WSE norm stats saved → {stats_path}")

    return out_path


def _determine_ecmwf_init_dates(ecmwf_base_dir: Path, probe_var: str = "tp") -> pd.DatetimeIndex:
    """
    Scan the ECMWF per-lake directory for one variable to determine available init_dates.

    Files are named: ecmwf_per_lake_{var}_YYYY-MM-DD.csv
    """
    var_dir = ecmwf_base_dir / probe_var
    if not var_dir.exists():
        raise FileNotFoundError(f"ECMWF variable directory not found: {var_dir}")

    dates = []
    for fpath in sorted(var_dir.glob(f"ecmwf_per_lake_{probe_var}_*.csv")):
        # File stem looks like "ecmwf_per_lake_tp_2024-01-14".
        # The init_date is always the last "_"-separated token.
        stem = fpath.stem
        date_str = stem.split("_")[-1]
        try:
            dates.append(pd.Timestamp(date_str))
        except Exception:
            pass   # skip files whose names don't end in a valid date

    if not dates:
        raise ValueError(f"No ECMWF init_date files found in {var_dir}")

    return pd.DatetimeIndex(sorted(dates))


def load_ecmwf_climate_arrays(
    ecmwf_base_dir: Path,
    raw_vars: list,
    lake_ids: np.ndarray,
    all_init_dates: pd.DatetimeIndex,
    forecast_horizon: int = 10,
) -> dict:
    """
    Load ECMWF IFS per-lake CSV files and return 13 derived climate arrays,
    each of shape (n_lakes, n_init_dates, forecast_horizon).

    CSV format: lake_id, init_date, forecast_day, valid_date, {var_name}, extraction_method
    Files:      ecmwf_base_dir/{var}/ecmwf_per_lake_{var}_YYYY-MM-DD.csv
    """
    n_lakes     = len(lake_ids)
    n_init      = len(all_init_dates)
    # Pre-build index maps for O(1) lookup inside the inner loop
    lake_to_idx = {lid: i for i, lid in enumerate(lake_ids)}
    init_to_idx = {d: i for i, d in enumerate(all_init_dates)}

    raw_dict = {}
    for var in raw_vars:
        print(f"  Loading ECMWF variable: {var} …")
        # 3-D array: axis 0 = lake, axis 1 = init_date, axis 2 = forecast_day (0-indexed)
        # Zero-initialised; missing files leave that (lake, init_date, day) slice as 0.
        raw_cube = np.zeros((n_lakes, n_init, forecast_horizon), dtype=np.float32)
        n_missing = 0

        for init_date in tqdm(all_init_dates, desc=f"    {var}", leave=False):
            date_str = init_date.strftime("%Y-%m-%d")
            # One CSV per init_date: rows = (lake_id, forecast_day) pairs
            fpath = ecmwf_base_dir / var / f"ecmwf_per_lake_{var}_{date_str}.csv"
            if not fpath.exists():
                n_missing += 1
                continue

            df = pd.read_csv(fpath)
            # Coerce to int64 and drop rows with invalid lake IDs
            df["lake_id"] = pd.to_numeric(df["lake_id"], errors="coerce").dropna().astype(np.int64)
            # Keep only lakes that are in our target lake_ids list
            df = df[df["lake_id"].isin(lake_to_idx)]
            if df.empty:
                continue

            t_idx = init_to_idx[init_date]
            # Pivot so rows = lake_id (matching lake_ids order) and
            # columns = forecast_day (1-indexed integers).
            pivot = (
                df.pivot_table(index="lake_id", columns="forecast_day", values=var, aggfunc="first")
                .reindex(index=lake_ids)
            )
            # Write each available forecast day into the raw cube.
            # forecast_day is 1-indexed in the CSV; we store it 0-indexed in the array.
            for fd in range(1, forecast_horizon + 1):
                if fd in pivot.columns:
                    col_vals = pivot[fd].to_numpy(dtype=np.float32)
                    col_vals = np.nan_to_num(col_vals, nan=0.0)
                    raw_cube[:, t_idx, fd - 1] = col_vals

        if n_missing > 0:
            pct = 100.0 * n_missing / max(n_init, 1)
            print(f"    [WARN] {var}: {n_missing}/{n_init} init_date files missing ({pct:.1f}%)")

        raw_dict[var] = raw_cube

    # Convert raw ECMWF variables to the 13 model-ready climate variables
    # using the same derive_climate_vars() function used for ERA5 — the unit
    # conversions apply identically to the 3-D ECMWF arrays.
    return derive_climate_vars(raw_dict)


def build_ecmwf_forecast_datacube(
    ecmwf_base_dir: Path,
    lake_ids: np.ndarray,
    all_init_dates: pd.DatetimeIndex,
    forecast_horizon: int,
    save_dir: Path,
) -> Path:
    """
    Assemble and save the ECMWF IFS forecast datacube.

    Dims: (lake, init_time, forecast_day)
    Variables: LWd, SWd, P, Pres, Temp, Td, Wind, sf, sd, swvl1, swvl2, swvl3, swvl4
    """
    print("\n=== Building ECMWF forecast datacube ===")

    # Load all 13 derived climate variables as 3-D arrays:
    # shape (n_lakes, n_init_dates, forecast_horizon)
    climate = load_ecmwf_climate_arrays(
        ecmwf_base_dir, ERA5_RAW_VARS, lake_ids, all_init_dates, forecast_horizon
    )

    # forecast_day coordinate is 1-indexed (day 1 = day after init_date)
    forecast_days = np.arange(1, forecast_horizon + 1, dtype=np.int32)

    # Build the data_vars dict dynamically from whatever derive_climate_vars returned
    data_vars = {}
    for var_name, arr in climate.items():
        data_vars[var_name] = (["lake", "init_time", "forecast_day"], arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lake":        lake_ids,
            "init_time":   all_init_dates,
            "forecast_day": forecast_days,
        },
        attrs={
            "description": "ECMWF IFS forecast climate per lake for lake-SWOT-GNN",
            "forecast_horizon": forecast_horizon,
            "created_by": "training_data_processing_lake_based_20260319.py",
        },
    )

    out_path = save_dir / "swot_lake_ecmwf_forecast_datacube.nc"
    ds.to_netcdf(out_path)
    print(
        f"ECMWF forecast datacube saved → {out_path}  "
        f"shape: {len(lake_ids)} lakes × {len(all_init_dates)} init_dates × {forecast_horizon} days"
    )
    return out_path


def build_static_datacube(
    lake_ids: np.ndarray,
    save_dir: Path,
    static_attrs_csv: Path = LAKE_STATIC_ATTRS_CSV,
    exclude_cols: list = STATIC_EXCLUDE_COLS,
) -> Path:
    """
    Build the static attributes datacube from a per-lake CSV file.

    Loads lake_graph_static_attrs_0sqkm.csv, drops the excluded columns,
    aligns rows to lake_ids order (NaN-filled for any missing lakes),
    and saves as swot_lake_static_datacube.nc with dims (lake, feature).
    """
    print("\n=== Building static datacube ===")
    print(f"  Loading static attrs from: {static_attrs_csv}")

    attrs_df = pd.read_csv(static_attrs_csv)
    attrs_df["lake_id"] = attrs_df["lake_id"].astype(np.int64)

    # Drop excluded columns (ignore any that are absent to be safe)
    drop_cols = [c for c in exclude_cols if c in attrs_df.columns]
    attrs_df = attrs_df.drop(columns=drop_cols)

    # Set lake_id as index and reindex to the target lake_ids order.
    # Lakes absent from the CSV receive NaN (then filled with 0).
    attrs_df = attrs_df.set_index("lake_id")
    attrs_df = attrs_df.reindex(lake_ids)

    feature_names = attrs_df.columns.tolist()
    static_cube = attrs_df.to_numpy(dtype=np.float32)          # (n_lakes, n_features)
    static_cube = np.nan_to_num(static_cube, nan=0.0)

    n_lakes, n_features = static_cube.shape
    print(f"  Static features ({n_features}): {feature_names}")

    ds = xr.Dataset(
        data_vars={"static_feature": (["lake", "feature"], static_cube)},
        coords={
            "lake":    lake_ids,
            "feature": feature_names,
        },
        attrs={
            "description": "Lake static attributes datacube for lake-SWOT-GNN",
            "source_csv":  str(static_attrs_csv),
            "excluded_cols": ", ".join(exclude_cols),
            "created_by": "training_data_processing_lake_based_20260319.py",
        },
    )

    out_path = save_dir / "swot_lake_static_datacube.nc"
    ds.to_netcdf(out_path)
    print(f"Static datacube saved → {out_path}  shape: {n_lakes} lakes × {n_features} features")
    return out_path


def assemble_lake_features_from_datacubes(
    era5_dynamic_datacube_path: Union[str, Path],
    ecmwf_forecast_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    lake_ids: Optional[np.ndarray] = None,
    era5_dates: Optional[pd.DatetimeIndex] = None,
    ecmwf_init_dates: Optional[pd.DatetimeIndex] = None,
    era5_input_vars: Optional[List[str]] = None,
    ecmwf_vars: Optional[List[str]] = None,
) -> Tuple[
    np.ndarray,          # era5_dynamic    (n_lakes, n_era5_dates, 21)
    np.ndarray,          # ecmwf_forecast  (n_lakes, n_init_dates, 10, 13)
    np.ndarray,          # static_features (n_lakes, n_static)
    np.ndarray,          # obs_mask        (n_lakes, n_era5_dates)
    np.ndarray,          # lake_ids_out    (n_lakes,)
    pd.DatetimeIndex,    # era5_dates_out
    pd.DatetimeIndex,    # ecmwf_init_dates_out
]:
    """
    Load lake features from the three datacubes into numpy arrays.

    Args:
        era5_dynamic_datacube_path:   Path to swot_lake_era5_dynamic_datacube_*.nc
        ecmwf_forecast_datacube_path: Path to swot_lake_ecmwf_forecast_datacube.nc
        static_datacube_path:         Path to swot_lake_static_datacube.nc
        lake_ids:         Optional subset of lake IDs. If None, use intersection of all cubes.
        era5_dates:       Optional date range for ERA5 data. If None, use full datacube range.
        ecmwf_init_dates: Optional init_date range for ECMWF data. If None, use full range.
        era5_input_vars:  Variables to extract as model input (default: ERA5_INPUT_VARS).
        ecmwf_vars:       Variables to extract from ECMWF cube (default: ECMWF_CLIMATE_VARS).

    Returns:
        era5_dynamic:        (n_lakes, n_era5_dates, 21) — ERA5 model input features
        ecmwf_forecast:      (n_lakes, n_init_dates, 10, 13) — ECMWF climate per init_date
        static_features:     (n_lakes, n_static)            — static attributes (placeholder)
        obs_mask:            (n_lakes, n_era5_dates)         — 1 where SWOT observed
        lake_ids_out:        (n_lakes,)
        era5_dates_out:      DatetimeIndex
        ecmwf_init_dates_out: DatetimeIndex
    """
    era5_path   = Path(era5_dynamic_datacube_path)
    ecmwf_path  = Path(ecmwf_forecast_datacube_path)
    static_path = Path(static_datacube_path)

    for p in (era5_path, ecmwf_path, static_path):
        if not p.exists():
            raise FileNotFoundError(f"Datacube not found: {p}")

    era5_input_vars = era5_input_vars or ERA5_INPUT_VARS
    ecmwf_vars      = ecmwf_vars      or ECMWF_CLIMATE_VARS

    ds_era5   = xr.open_dataset(era5_path)
    ds_ecmwf  = xr.open_dataset(ecmwf_path)
    ds_static = xr.open_dataset(static_path)

    try:
        # ── Resolve lake IDs (intersection across all cubes) ─────────────────
        # Each datacube may have been built from a slightly different lake list.
        # Take the intersection so every returned array covers exactly the same
        # set of lakes in the same order.
        era5_lakes   = ds_era5.coords["lake"].values.astype(np.int64)
        ecmwf_lakes  = ds_ecmwf.coords["lake"].values.astype(np.int64)
        static_lakes = ds_static.coords["lake"].values.astype(np.int64)

        if lake_ids is None:
            # Use all lakes present in every datacube
            lake_ids = np.sort(
                np.intersect1d(np.intersect1d(era5_lakes, ecmwf_lakes), static_lakes)
            )
        else:
            # Keep only the requested lakes that exist in all three cubes
            lake_ids = np.array([
                lid for lid in lake_ids
                if lid in era5_lakes and lid in ecmwf_lakes and lid in static_lakes
            ], dtype=np.int64)

        if len(lake_ids) == 0:
            raise ValueError("No lakes in common across all three datacubes.")

        # ── Resolve ERA5 dates ────────────────────────────────────────────────
        # Filter requested dates to those actually stored in the datacube.
        dyn_times = pd.DatetimeIndex(ds_era5.coords["time"].values)
        if era5_dates is None:
            era5_dates = dyn_times          # use the full datacube range
        else:
            era5_dates = era5_dates[era5_dates.isin(dyn_times)]   # take the intersection
        if len(era5_dates) == 0:
            raise ValueError("No overlapping dates between request and ERA5 datacube.")

        # ── Resolve ECMWF init_dates ──────────────────────────────────────────
        ecmwf_times = pd.DatetimeIndex(ds_ecmwf.coords["init_time"].values)
        if ecmwf_init_dates is None:
            ecmwf_init_dates = ecmwf_times
        else:
            ecmwf_init_dates = ecmwf_init_dates[ecmwf_init_dates.isin(ecmwf_times)]
        if len(ecmwf_init_dates) == 0:
            raise ValueError("No overlapping init_dates between request and ECMWF datacube.")

        # ── ERA5 dynamic features — shape (n_lakes, n_era5_dates, 21) ────────
        # Load each feature variable separately and stack into a single tensor
        # along the last axis.  The variable order follows era5_input_vars,
        # which must match the model's expected feature ordering (ERA5_INPUT_VARS).
        era5_blocks = []
        for v in era5_input_vars:
            if v not in ds_era5.data_vars:
                raise KeyError(
                    f"Variable '{v}' not in ERA5 datacube. "
                    f"Available: {list(ds_era5.data_vars)}"
                )
            arr = ds_era5[v].sel(lake=lake_ids, time=era5_dates).values
            era5_blocks.append(arr)
        era5_dynamic = np.stack(era5_blocks, axis=-1).astype(np.float32)
        era5_dynamic = np.nan_to_num(era5_dynamic, nan=0.0)
        # shape: (n_lakes, n_era5_dates, 21)

        # ── obs_mask — shape (n_lakes, n_era5_dates) ─────────────────────────
        # Extracted separately (not included in era5_input_vars) because it is
        # used as a supervision mask, not as a model input feature.
        obs_mask_arr = ds_era5["obs_mask"].sel(lake=lake_ids, time=era5_dates).values.astype(np.float32)

        # ── ECMWF forecast climate — shape (n_lakes, n_init_dates, n_days, 13) ─
        # Same stacking strategy as ERA5: one block per variable, then stack.
        ecmwf_blocks = []
        for v in ecmwf_vars:
            if v not in ds_ecmwf.data_vars:
                raise KeyError(
                    f"Variable '{v}' not in ECMWF datacube. "
                    f"Available: {list(ds_ecmwf.data_vars)}"
                )
            # Each block: (n_lakes, n_init_dates, n_forecast_days)
            arr = ds_ecmwf[v].sel(lake=lake_ids, init_time=ecmwf_init_dates).values
            ecmwf_blocks.append(arr)
        # Stack along last axis → (n_lakes, n_init_dates, n_forecast_days, 13)
        ecmwf_forecast = np.stack(ecmwf_blocks, axis=-1).astype(np.float32)
        ecmwf_forecast = np.nan_to_num(ecmwf_forecast, nan=0.0)

        # ── Static features — shape (n_lakes, n_static) ──────────────────────
        static_arr = ds_static["static_feature"].sel(lake=lake_ids).values.astype(np.float32)
        static_arr = np.nan_to_num(static_arr, nan=0.0)

        return (
            era5_dynamic,
            ecmwf_forecast,
            static_arr,
            obs_mask_arr,
            lake_ids,
            era5_dates,
            ecmwf_init_dates,
        )

    finally:
        ds_era5.close()
        ds_ecmwf.close()
        ds_static.close()


def generate_lake_datacubes(
    swot_csv: Path = SWOT_LAKE_WSE_CSV,
    lake_graph_csv: Path = LAKE_GRAPH_CSV,
    era5_base_dir: Path = ERA5_BASE_DIR,
    ecmwf_base_dir: Path = ECMWF_BASE_DIR,
    save_dir: Path = SAVE_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    wse_option: str = WSE_OPTION,
    forecast_horizon: int = FORECAST_HORIZON,
    static_attrs_csv: Path = LAKE_STATIC_ATTRS_CSV,
) -> None:
    """
    Main entry point: build all three training datacubes for the lake-based SWOT-GNN.

    Steps:
      1. Load lake IDs from GRIT PLD lake graph
      2. Build ERA5 + SWOT WSE dynamic datacube
      3. Discover available ECMWF init_dates and build forecast datacube
      4. Build static attributes datacube from lake_graph_static_attrs_0sqkm.csv
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lake IDs
    print(f"Loading lake IDs from: {lake_graph_csv}")
    lake_ids = load_lake_ids_from_graph(lake_graph_csv)
    print(f"  Found {len(lake_ids)} lakes in GRIT PLD lake graph.")

    # 2. ERA5 dynamic datacube
    all_dates = pd.date_range(start_date, end_date, freq="D")
    build_era5_dynamic_datacube(
        swot_csv=swot_csv,
        era5_base_dir=era5_base_dir,
        lake_ids=lake_ids,
        all_dates=all_dates,
        wse_option=wse_option,
        save_dir=save_dir,
    )

    # 3. ECMWF forecast datacube
    print(f"\nScanning ECMWF init_dates in: {ecmwf_base_dir}")
    all_init_dates = _determine_ecmwf_init_dates(ecmwf_base_dir, probe_var=ERA5_RAW_VARS[0])
    print(f"  Found {len(all_init_dates)} ECMWF init_dates "
          f"({all_init_dates[0].date()} … {all_init_dates[-1].date()})")

    build_ecmwf_forecast_datacube(
        ecmwf_base_dir=ecmwf_base_dir,
        lake_ids=lake_ids,
        all_init_dates=all_init_dates,
        forecast_horizon=forecast_horizon,
        save_dir=save_dir,
    )

    # 4. Static attributes datacube
    build_static_datacube(lake_ids=lake_ids, save_dir=save_dir, static_attrs_csv=static_attrs_csv)

    print("\n=== Done. Summary ===")
    print(f"  Lakes:       {len(lake_ids)}")
    print(f"  ERA5 dates:  {len(all_dates)} ({start_date} … {end_date})")
    print(f"  ECMWF inits: {len(all_init_dates)}")
    print(f"  Output dir:  {save_dir}")


if __name__ == "__main__":
    generate_lake_datacubes()
