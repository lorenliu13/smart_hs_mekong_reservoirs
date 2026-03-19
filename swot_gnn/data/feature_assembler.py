"""
Assemble node features per reach per day for SWOT-GNN.
Features from datacubes: dynamic (WSE, mask, atmospheric forcing, time encoding)
and static (reach attributes).
"""
# Numerical arrays and DataFrames
import numpy as np
import pandas as pd
# NetCDF / xarray for loading datacubes
import xarray as xr
# Path handling
from pathlib import Path
# Type hints
from typing import Optional, Union, Tuple, List

# Dynamic variable names; must match training_data_processing_20260202.py output
DYNAMIC_FEATURE_VARS = [
    "obs_mask",           # 0: 1 where SWOT observed, 0 otherwise
    "latest_wse",         # 1: WSE at last observation (or 0); model predicts this
    "days_since_last_obs",# 2: Days since last SWOT obs (temporal gap)
    "time_doy_sin",       # 3: sin(2π * day_of_year/365) – seasonal encoding
    "time_doy_cos",       # 4: cos(2π * day_of_year/365)
    "LWd",                # 5: Longwave downward radiation (W/m²)
    "P",                  # 6: Precipitation (mm)
    "Pres",               # 7: Surface pressure (Pa)
    "RelHum",             # 8: Relative humidity (%)
    "SWd",                # 9: Shortwave downward radiation (W/m²)
    "Temp",               # 10: Temperature (K or °C)
    "Wind",               # 11: Wind speed (m/s)
]
# Index of latest_wse in dynamic block; zeroed out at prediction nodes during training
DYNAMIC_WSE_INPUT_IDX = 1
# Indices of WSE-related dynamic features zeroed out for the forecast step (day t+1 has no observed WSE)
# [obs_mask, latest_wse, days_since_last_obs]
WSE_DYNAMIC_INDICES = [0, 1, 2]


def assemble_node_features_from_datacubes(
    dynamic_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    reach_ids: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    dynamic_vars: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Assemble node features from dynamic and static NetCDF datacubes
    produced by training_data_processing_20260202.py.

    Args:
        dynamic_datacube_path: Path to ba_river_swot_dynamic_datacube_*.nc
        static_datacube_path: Path to ba_river_swot_static_datacube_*.nc
        reach_ids: Optional subset of reach IDs (must be in datacube). If None, use all.
        dates: Optional date range. If None, use all dates in dynamic datacube.
        dynamic_vars: Variables to use from dynamic cube (default: DYNAMIC_FEATURE_VARS).

    Returns:
        node_features: (num_reaches, num_dates, feat_dim) - dynamic + static features
        wse_labels: (num_reaches, num_dates) - WSE values for target
        obs_mask: (num_reaches, num_dates) - 1 where SWOT observation available
        reach_ids_out: (num_reaches,) - reach IDs actually used (for graph alignment)
        dates_out: DatetimeIndex - dates actually used
    """
    # Resolve paths and check files exist
    dynamic_path = Path(dynamic_datacube_path)
    static_path = Path(static_datacube_path)
    if not dynamic_path.exists():
        raise FileNotFoundError(f"Dynamic datacube not found: {dynamic_path}")
    if not static_path.exists():
        raise FileNotFoundError(f"Static datacube not found: {static_path}")

    # Open NetCDF datacubes (dynamic: time-varying; static: reach attributes only)
    ds_dyn = xr.open_dataset(dynamic_path)
    ds_static = xr.open_dataset(static_path)

    try:
        # Get coordinate arrays from both cubes
        dyn_reaches = ds_dyn.coords["reach"].values
        dyn_times = pd.DatetimeIndex(ds_dyn.coords["time"].values)
        static_reaches = ds_static.coords["reach"].values

        # Default: use intersection of reaches and full date range
        if reach_ids is None:
            reach_ids = np.sort(np.intersect1d(dyn_reaches, static_reaches))
        if dates is None:
            dates = dyn_times

        # Keep only reaches present in both cubes; keep only dates in dynamic cube
        reach_ids = np.array([r for r in reach_ids if r in dyn_reaches and r in static_reaches])
        dates = dates[dates.isin(dyn_times)]
        if len(dates) == 0:
            raise ValueError("No overlapping dates between request and dynamic datacube")

        # Stack dynamic variables: each var -> (n_reach, n_date), stack on last axis
        dynamic_vars = dynamic_vars or DYNAMIC_FEATURE_VARS
        dynamic_blocks = []
        for v in dynamic_vars:
            if v not in ds_dyn.data_vars:
                raise KeyError(f"Dynamic variable '{v}' not in datacube")
            arr = ds_dyn[v].sel(reach=reach_ids, time=dates).values
            dynamic_blocks.append(arr)
        # Result: (n_reach, n_date, n_dyn) with n_dyn = len(dynamic_vars)
        dynamic_feat = np.stack(dynamic_blocks, axis=-1).astype(np.float32)
        # Model expects no NaN; fill with 0
        dynamic_feat = np.nan_to_num(dynamic_feat, nan=0.0)

        # Static features: (n_reach, n_static) – same every day, broadcast to (n_reach, n_date, n_static)
        static_arr = ds_static["static_feature"].sel(reach=reach_ids).values
        n_reach, n_date, n_dyn = dynamic_feat.shape
        n_static = static_arr.shape[1]
        # [:, None, :] adds time dim; broadcast_to repeats along date axis
        static_feat = np.broadcast_to(
            static_arr[:, None, :], (n_reach, n_date, n_static)
        ).astype(np.float32)
        static_feat = np.nan_to_num(static_feat, nan=0.0)

        # Concatenate: feat_dim = n_dyn + n_static (e.g. 12 + 31 = 43)
        node_features = np.concatenate([dynamic_feat, static_feat], axis=-1)

        # Targets: WSE (NaN where not observed) and binary obs mask
        wse_labels = ds_dyn["wse"].sel(reach=reach_ids, time=dates).values.astype(np.float32)
        obs_mask = ds_dyn["obs_mask"].sel(reach=reach_ids, time=dates).values.astype(np.float32)

        return node_features, wse_labels, obs_mask, reach_ids, dates
    finally:
        ds_dyn.close()
        ds_static.close()


def assemble_node_features_from_datacubes_segment_based(
    dynamic_datacube_path: Union[str, Path],
    static_datacube_path: Union[str, Path],
    segment_ids: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    dynamic_vars: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Assemble node features from segment-based dynamic and static NetCDF datacubes.

    Same as assemble_node_features_from_datacubes but uses "segment" coord
    instead of "reach". Returns dynamic and static features separately so static
    can be encoded independently (e.g. as LSTM initial hidden state).

    Returns:
        dynamic_features: (num_segments, num_dates, n_dyn) - time-varying features only
        static_features: (num_segments, n_static) - time-invariant reach attributes
        wse_labels: (num_segments, num_dates)
        obs_mask: (num_segments, num_dates)
        segment_ids_out: (num_segments,) - segment IDs used
        dates_out: DatetimeIndex
    """
    dynamic_path = Path(dynamic_datacube_path)
    static_path = Path(static_datacube_path)
    if not dynamic_path.exists():
        raise FileNotFoundError(f"Dynamic datacube not found: {dynamic_path}")
    if not static_path.exists():
        raise FileNotFoundError(f"Static datacube not found: {static_path}")

    ds_dyn = xr.open_dataset(dynamic_path)
    ds_static = xr.open_dataset(static_path)

    try:
        # Segment-based: use "segment" coord instead of "reach"
        dyn_segments = ds_dyn.coords["segment"].values
        dyn_times = pd.DatetimeIndex(ds_dyn.coords["time"].values)
        static_segments = ds_static.coords["segment"].values

        if segment_ids is None:
            segment_ids = np.sort(np.intersect1d(dyn_segments, static_segments))
        if dates is None:
            dates = dyn_times

        # Keep only segments in both cubes; restrict dates to dynamic range
        segment_ids = np.array([s for s in segment_ids if s in dyn_segments and s in static_segments])
        dates = dates[dates.isin(dyn_times)]
        if len(dates) == 0:
            raise ValueError("No overlapping dates between request and dynamic datacube")

        # Same logic as reach-based: stack dynamic vars, sel(segment=..., time=...)
        dynamic_vars = dynamic_vars or DYNAMIC_FEATURE_VARS
        dynamic_blocks = []
        for v in dynamic_vars:
            if v not in ds_dyn.data_vars:
                raise KeyError(f"Dynamic variable '{v}' not in datacube")
            arr = ds_dyn[v].sel(segment=segment_ids, time=dates).values
            dynamic_blocks.append(arr)
        dynamic_feat = np.stack(dynamic_blocks, axis=-1).astype(np.float32)
        dynamic_feat = np.nan_to_num(dynamic_feat, nan=0.0)

        # Static: (n_seg, n_static) -- kept separate, NOT broadcast or concatenated
        static_arr = ds_static["static_feature"].sel(segment=segment_ids).values.astype(np.float32)
        static_arr = np.nan_to_num(static_arr, nan=0.0)

        # Targets: WSE (NaN where not observed) and binary obs mask
        wse_labels = ds_dyn["wse"].sel(segment=segment_ids, time=dates).values.astype(np.float32)
        obs_mask = ds_dyn["obs_mask"].sel(segment=segment_ids, time=dates).values.astype(np.float32)

        return dynamic_feat, static_arr, wse_labels, obs_mask, segment_ids, dates
    finally:
        ds_dyn.close()
        ds_static.close()
