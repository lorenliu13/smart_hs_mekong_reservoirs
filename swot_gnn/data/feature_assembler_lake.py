"""
Assemble lake node features from ERA5 + ECMWF + static datacubes for lake-SWOT-GNN.

Datacubes produced by training_data_processing_lake_based_20260319.py:
  - swot_lake_era5_dynamic_datacube_{wse_option}.nc  (lake, time)
  - swot_lake_ecmwf_forecast_datacube.nc             (lake, init_time, forecast_day)
  - swot_lake_static_datacube.nc                     (lake, feature)
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Optional, Union, List, Tuple

# ─── Feature ordering ──────────────────────────────────────────────────────────

# The 14 model input features from the ERA5 dynamic datacube.
# "wse" is stored in the datacube as the label (NaN on non-observation days)
# but is NOT included here as a model input — latest_wse carries that signal.
ERA5_INPUT_VARS: List[str] = [
    "obs_mask",             # 0: 1 where SWOT observed, 0 otherwise
    "latest_wse",           # 1: forward-filled normalised WSE (0 before first obs)
    "days_since_last_obs",  # 2: days since last SWOT pass
    "time_doy_sin",         # 3: sin(2π × doy / 365.25)
    "time_doy_cos",         # 4: cos(2π × doy / 365.25)
    "LWd",                  # 5: longwave downward radiation  (W/m²)
    "SWd",                  # 6: shortwave downward radiation (W/m²)
    "P",                    # 7: precipitation                (mm)
    "Pres",                 # 8: surface pressure             (Pa)
    "Temp",                 # 9: 2-m air temperature          (K)
    "Wind",                 # 10: 10-m wind speed             (m/s)
    "RelHum",               # 11: relative humidity           (%)
    "sd",                   # 12: snow depth                  (m)
    "swvl1",                # 13: top-layer soil moisture     (m³/m³)
]

# The 9 climate variables in the ECMWF forecast datacube.
# They occupy feature slots 5–13 in the 14-dim input vector (same order as above).
ECMWF_CLIMATE_VARS: List[str] = [
    "LWd", "SWd", "P", "Pres", "Temp", "Wind", "RelHum", "sd", "swvl1",
]

# SWOT feature count (indices 0–4 in ERA5_INPUT_VARS)
SWOT_DIM: int = 5
# Climate feature count (indices 5–13)
CLIMATE_DIM: int = 9

# Indices zeroed out for ECMWF forecast timesteps (no observed WSE in the future)
WSE_LAKE_DYNAMIC_INDICES: List[int] = [0, 1, 2]  # obs_mask, latest_wse, days_since_last_obs

# ───────────────────────────────────────────────────────────────────────────────


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
    np.ndarray,          # era5_dynamic    (n_lakes, n_era5_dates, 14)
    np.ndarray,          # ecmwf_forecast  (n_lakes, n_init_dates, 10, 9)
    np.ndarray,          # static_features (n_lakes, n_static)
    np.ndarray,          # wse_labels      (n_lakes, n_era5_dates)
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
        lake_ids:     Optional subset of lake IDs. If None, use intersection of all cubes.
        era5_dates:   Optional date range for ERA5 data. If None, use full datacube range.
        ecmwf_init_dates: Optional init_date range for ECMWF data. If None, use full range.
        era5_input_vars:  Variables to extract as model input (default: ERA5_INPUT_VARS).
        ecmwf_vars:       Variables to extract from ECMWF cube (default: ECMWF_CLIMATE_VARS).

    Returns:
        era5_dynamic:      (n_lakes, n_era5_dates, 14)  — ERA5 model input features
        ecmwf_forecast:    (n_lakes, n_init_dates, 10, 9) — ECMWF climate per init_date
        static_features:   (n_lakes, n_static)           — static attributes (placeholder)
        wse_labels:        (n_lakes, n_era5_dates)        — raw WSE (NaN → 0 filled)
        obs_mask:          (n_lakes, n_era5_dates)        — 1 where SWOT observed
        lake_ids_out:      (n_lakes,)
        era5_dates_out:    DatetimeIndex
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
        era5_lakes   = ds_era5.coords["lake"].values.astype(np.int64)
        ecmwf_lakes  = ds_ecmwf.coords["lake"].values.astype(np.int64)
        static_lakes = ds_static.coords["lake"].values.astype(np.int64)

        if lake_ids is None:
            lake_ids = np.sort(
                np.intersect1d(np.intersect1d(era5_lakes, ecmwf_lakes), static_lakes)
            )
        else:
            lake_ids = np.array([
                lid for lid in lake_ids
                if lid in era5_lakes and lid in ecmwf_lakes and lid in static_lakes
            ], dtype=np.int64)

        if len(lake_ids) == 0:
            raise ValueError("No lakes in common across all three datacubes.")

        # ── Resolve ERA5 dates ────────────────────────────────────────────────
        dyn_times = pd.DatetimeIndex(ds_era5.coords["time"].values)
        if era5_dates is None:
            era5_dates = dyn_times
        else:
            era5_dates = era5_dates[era5_dates.isin(dyn_times)]
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

        # ── ERA5 dynamic features (14 model inputs) ───────────────────────────
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
        # shape: (n_lakes, n_era5_dates, 14)

        # ── ERA5 labels (wse + obs_mask) ──────────────────────────────────────
        wse_labels = ds_era5["wse"].sel(lake=lake_ids, time=era5_dates).values.astype(np.float32)
        obs_mask_arr = ds_era5["obs_mask"].sel(lake=lake_ids, time=era5_dates).values.astype(np.float32)
        wse_labels = np.nan_to_num(wse_labels, nan=0.0)

        # ── ECMWF forecast climate (9 vars × 10 days) ─────────────────────────
        ecmwf_blocks = []
        forecast_days = ds_ecmwf.coords["forecast_day"].values  # 1-indexed [1..10]
        for v in ecmwf_vars:
            if v not in ds_ecmwf.data_vars:
                raise KeyError(
                    f"Variable '{v}' not in ECMWF datacube. "
                    f"Available: {list(ds_ecmwf.data_vars)}"
                )
            # shape: (n_lakes, n_init_dates, n_forecast_days)
            arr = ds_ecmwf[v].sel(lake=lake_ids, init_time=ecmwf_init_dates).values
            ecmwf_blocks.append(arr)
        # Stack along last axis: (n_lakes, n_init_dates, n_forecast_days, 9)
        ecmwf_forecast = np.stack(ecmwf_blocks, axis=-1).astype(np.float32)
        ecmwf_forecast = np.nan_to_num(ecmwf_forecast, nan=0.0)

        # ── Static features ───────────────────────────────────────────────────
        static_arr = ds_static["static_feature"].sel(lake=lake_ids).values.astype(np.float32)
        static_arr = np.nan_to_num(static_arr, nan=0.0)

        return (
            era5_dynamic,
            ecmwf_forecast,
            static_arr,
            wse_labels,
            obs_mask_arr,
            lake_ids,
            era5_dates,
            ecmwf_init_dates,
        )

    finally:
        ds_era5.close()
        ds_ecmwf.close()
        ds_static.close()
