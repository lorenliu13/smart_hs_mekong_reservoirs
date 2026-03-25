"""
Assemble lake node features for the lake-based SWOT-GNN.

Loads features from four separate pre-built NetCDF datacubes:
  - swot_lake_wse_datacube_{wse_option}.nc      → SWOT WSE input features + target (lake, time)
  - swot_lake_era5_climate_datacube.nc           → ERA5-Land climate features  (lake, time)
  - swot_lake_ecmwf_forecast_datacube.nc         → ECMWF IFS forecast          (lake, init_time, forecast_day)
  - swot_lake_static_datacube.nc                 → static lake attributes       (lake, feature)

Primary entry point: assemble_lake_features_from_datacubes()
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Optional, Tuple, Union

# ─── Feature ordering ──────────────────────────────────────────────────────────

# 8 SWOT-derived input features loaded from the WSE datacube (indices 0–7).
WSE_INPUT_VARS: List[str] = [
    "obs_mask",              # 0:  1 where SWOT observed, 0 otherwise
    "latest_wse",            # 1:  forward-filled normalised WSE (0 before first obs)
    "latest_wse_u",          # 2:  forward-filled WSE uncertainty (0 before first obs)
    "latest_wse_std",        # 3:  forward-filled within-pass WSE std (0 before first obs)
    "latest_area_total",     # 4:  forward-filled total water area (0 before first obs)
    "days_since_last_obs",   # 5:  days since last SWOT pass
    "time_doy_sin",          # 6:  sin(2π × doy / 365.25)
    "time_doy_cos",          # 7:  cos(2π × doy / 365.25)
]

# 13 ERA5-Land climate features loaded from the ERA5 climate datacube (indices 8–20).
# Their order exactly matches the climate block in ECMWF_CLIMATE_VARS so ERA5
# reanalysis and ECMWF forecast tensors can be concatenated / substituted directly.
ERA5_CLIMATE_VARS: List[str] = [
    "LWd",   # 8:  longwave downward radiation   (W/m²)
    "SWd",   # 9:  shortwave downward radiation  (W/m²)
    "P",     # 10: precipitation                 (mm)
    "Pres",  # 11: surface pressure              (Pa)
    "Temp",  # 12: 2-m air temperature           (K)
    "Td",    # 13: 2-m dewpoint temperature      (K)
    "Wind",  # 14: 10-m wind speed               (m/s)
    "sf",    # 15: snowfall                      (mm)
    "sd",    # 16: snow depth                    (mm)
    "swvl1", # 17: soil moisture layer 1         (m³/m³)
    "swvl2", # 18: soil moisture layer 2         (m³/m³)
    "swvl3", # 19: soil moisture layer 3         (m³/m³)
    "swvl4", # 20: soil moisture layer 4         (m³/m³)
]

# Combined ordered list of all 21 model input features (WSE block + climate block).
ERA5_INPUT_VARS: List[str] = WSE_INPUT_VARS + ERA5_CLIMATE_VARS

# ECMWF IFS forecast variables — same 13 as ERA5_CLIMATE_VARS.
ECMWF_CLIMATE_VARS: List[str] = ERA5_CLIMATE_VARS

# Number of SWOT-derived features (indices 0–7)
SWOT_DIM: int = 8
# Number of climate features (indices 8–20)
CLIMATE_DIM: int = 13

# Feature indices zeroed when running in forecast mode (future timesteps have no
# SWOT observations).  Indices 6–7 (doy_sin / doy_cos) are kept because the
# calendar date is still known.
WSE_LAKE_DYNAMIC_INDICES: List[int] = [0, 1, 2, 3, 4, 5]  # obs_mask … days_since_last_obs

# ───────────────────────────────────────────────────────────────────────────────


def assemble_lake_features_from_datacubes(
    wse_datacube_path: Union[str, Path],
    era5_climate_datacube_path: Union[str, Path],
    ecmwf_forecast_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    lake_ids: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    ecmwf_init_dates: Optional[pd.DatetimeIndex] = None,
    wse_input_vars: Optional[List[str]] = None,
    era5_climate_vars: Optional[List[str]] = None,
    ecmwf_vars: Optional[List[str]] = None,
) -> Tuple[
    np.ndarray,          # dynamic_features  (n_lakes, n_dates, 21)
    np.ndarray,          # ecmwf_forecast    (n_lakes, n_init_dates, n_forecast_days, 13)
    np.ndarray,          # static_features   (n_lakes, n_static)
    np.ndarray,          # wse_target        (n_lakes, n_dates)  NaN where not observed
    np.ndarray,          # obs_mask          (n_lakes, n_dates)
    np.ndarray,          # lake_ids_out      (n_lakes,)
    pd.DatetimeIndex,    # dates_out
    pd.DatetimeIndex,    # ecmwf_init_dates_out
]:
    """
    Load lake features from the four separate datacubes into numpy arrays.

    The WSE datacube (SWOT features) and ERA5 climate datacube are loaded
    independently and concatenated into a single dynamic feature tensor
    of shape (n_lakes, n_dates, 21).

    Args:
        wse_datacube_path:            Path to swot_lake_wse_datacube_*.nc
        era5_climate_datacube_path:   Path to swot_lake_era5_climate_datacube.nc
        ecmwf_forecast_datacube_path: Path to swot_lake_ecmwf_forecast_datacube.nc
        static_datacube_path:         Path to swot_lake_static_datacube.nc
        lake_ids:         Optional subset of lake IDs. If None, uses the intersection
                          of all four datacubes.
        dates:            Optional date range. If None, uses the full WSE datacube range.
        ecmwf_init_dates: Optional init_date range for ECMWF data. If None, uses
                          the full range.
        wse_input_vars:   WSE-derived features to extract (default: WSE_INPUT_VARS).
        era5_climate_vars: ERA5 climate features to extract (default: ERA5_CLIMATE_VARS).
        ecmwf_vars:       Variables to extract from ECMWF cube (default: ECMWF_CLIMATE_VARS).

    Returns:
        dynamic_features:     (n_lakes, n_dates, 21) — WSE (8) + ERA5 climate (13) concatenated
        ecmwf_forecast:       (n_lakes, n_init_dates, n_forecast_days, 13)
        static_features:      (n_lakes, n_static)
        wse_target:           (n_lakes, n_dates)  — WSE in datacube form, NaN where not observed
        obs_mask:             (n_lakes, n_dates)  — 1 where SWOT observed
        lake_ids_out:         (n_lakes,)
        dates_out:            DatetimeIndex
        ecmwf_init_dates_out: DatetimeIndex
    """
    wse_path    = Path(wse_datacube_path)
    era5_path   = Path(era5_climate_datacube_path)
    ecmwf_path  = Path(ecmwf_forecast_datacube_path)
    static_path = Path(static_datacube_path)

    for p in (wse_path, era5_path, ecmwf_path, static_path):
        if not p.exists():
            raise FileNotFoundError(f"Datacube not found: {p}")

    wse_input_vars    = wse_input_vars    or WSE_INPUT_VARS
    era5_climate_vars = era5_climate_vars or ERA5_CLIMATE_VARS
    ecmwf_vars        = ecmwf_vars        or ECMWF_CLIMATE_VARS

    ds_wse    = xr.open_dataset(wse_path)
    ds_era5   = xr.open_dataset(era5_path)
    ds_ecmwf  = xr.open_dataset(ecmwf_path)
    ds_static = xr.open_dataset(static_path)

    try:
        # ── Resolve lake IDs (intersection across all four cubes) ─────────────
        wse_lakes    = ds_wse.coords["lake"].values.astype(np.int64)
        era5_lakes   = ds_era5.coords["lake"].values.astype(np.int64)
        ecmwf_lakes  = ds_ecmwf.coords["lake"].values.astype(np.int64)
        static_lakes = ds_static.coords["lake"].values.astype(np.int64)

        all_cube_lakes = np.intersect1d(
            np.intersect1d(wse_lakes, era5_lakes),
            np.intersect1d(ecmwf_lakes, static_lakes),
        )

        if lake_ids is None:
            lake_ids = np.sort(all_cube_lakes)
        else:
            lake_ids = np.array(
                [lid for lid in lake_ids if lid in all_cube_lakes], dtype=np.int64
            )

        if len(lake_ids) == 0:
            raise ValueError("No lakes in common across all four datacubes.")

        # ── Resolve dates (intersection of WSE and ERA5 time axes) ───────────
        wse_times  = pd.DatetimeIndex(ds_wse.coords["time"].values)
        era5_times = pd.DatetimeIndex(ds_era5.coords["time"].values)
        common_times = wse_times.intersection(era5_times)

        if dates is None:
            dates = common_times
        else:
            dates = dates[dates.isin(common_times)]

        if len(dates) == 0:
            raise ValueError("No overlapping dates across WSE and ERA5 datacubes.")

        # ── Resolve ECMWF init_dates ──────────────────────────────────────────
        ecmwf_times = pd.DatetimeIndex(ds_ecmwf.coords["init_time"].values)
        if ecmwf_init_dates is None:
            ecmwf_init_dates = ecmwf_times
        else:
            ecmwf_init_dates = ecmwf_init_dates[ecmwf_init_dates.isin(ecmwf_times)]

        if len(ecmwf_init_dates) == 0:
            raise ValueError("No overlapping init_dates between request and ECMWF datacube.")

        # ── WSE block — shape (n_lakes, n_dates, 8) ───────────────────────────
        wse_blocks = []
        for v in wse_input_vars:
            if v not in ds_wse.data_vars:
                raise KeyError(
                    f"Variable '{v}' not in WSE datacube. "
                    f"Available: {list(ds_wse.data_vars)}"
                )
            arr = ds_wse[v].sel(lake=lake_ids, time=dates).values
            wse_blocks.append(arr)
        wse_feat = np.stack(wse_blocks, axis=-1).astype(np.float32)
        wse_feat = np.nan_to_num(wse_feat, nan=0.0)

        # ── ERA5 climate block — shape (n_lakes, n_dates, 13) ─────────────────
        era5_blocks = []
        for v in era5_climate_vars:
            if v not in ds_era5.data_vars:
                raise KeyError(
                    f"Variable '{v}' not in ERA5 climate datacube. "
                    f"Available: {list(ds_era5.data_vars)}"
                )
            arr = ds_era5[v].sel(lake=lake_ids, time=dates).values
            era5_blocks.append(arr)
        era5_feat = np.stack(era5_blocks, axis=-1).astype(np.float32)
        era5_feat = np.nan_to_num(era5_feat, nan=0.0)

        # Concatenate WSE + ERA5 → (n_lakes, n_dates, 21)
        dynamic_features = np.concatenate([wse_feat, era5_feat], axis=-1)

        # ── WSE target and obs mask — from WSE datacube ───────────────────────
        wse_target = ds_wse["wse"].sel(lake=lake_ids, time=dates).values.astype(np.float32)
        obs_mask   = ds_wse["obs_mask"].sel(lake=lake_ids, time=dates).values.astype(np.float32)

        # ── ECMWF forecast climate — shape (n_lakes, n_init_dates, n_days, 13) ─
        ecmwf_blocks = []
        for v in ecmwf_vars:
            if v not in ds_ecmwf.data_vars:
                raise KeyError(
                    f"Variable '{v}' not in ECMWF datacube. "
                    f"Available: {list(ds_ecmwf.data_vars)}"
                )
            arr = ds_ecmwf[v].sel(lake=lake_ids, init_time=ecmwf_init_dates).values
            ecmwf_blocks.append(arr)
        ecmwf_forecast = np.stack(ecmwf_blocks, axis=-1).astype(np.float32)
        ecmwf_forecast = np.nan_to_num(ecmwf_forecast, nan=0.0)

        # ── Static features — shape (n_lakes, n_static) ──────────────────────
        static_arr = ds_static["static_feature"].sel(lake=lake_ids).values.astype(np.float32)
        static_arr = np.nan_to_num(static_arr, nan=0.0)

        return (
            dynamic_features,
            ecmwf_forecast,
            static_arr,
            wse_target,
            obs_mask,
            lake_ids,
            dates,
            ecmwf_init_dates,
        )

    finally:
        ds_wse.close()
        ds_era5.close()
        ds_ecmwf.close()
        ds_static.close()
