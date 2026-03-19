"""
Segment-based training data preparation for SWOT-GNN.

Uses segment-reach mapping from aggregate_wse_by_segments.ipynb. Each segment
gets its WSE, atmospheric forcing, and static features from its corresponding
selected_reach_id (the reach with most SWOT samples or the lake reach).

Segments with no valid selected_reach_id: WSE is treated as no-data (mask=0),
but atmospheric forcing and static attributes are still included using the
first reach in the segment as fallback.

Output: NetCDF datacubes with segment dimension instead of reach dimension,
plus segment-based graph and temporal dataset support.
"""
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple

# --- Configuration ---
wse_option = "wse_norm"
# Options: wse_anomaly, wse_norm, wse

# Paths - configure for your environment
# BASE_DIR: project root (parent of processed_data, raw_data)
BASE_DIR = Path(r"E:\Project_2025_2026\Smart_hs")

DEFAULT_SAVE_FOLDER = BASE_DIR / "processed_data" / "swot_gnn" / "training_data" / "training_data_segment_based_20260222"
SEGMENT_MAPPING_PATH = BASE_DIR / "processed_data" / "swot_gnn" / "training_data" / "segment_mapping_df.csv"
SWOT_WSE_PATH = BASE_DIR / "processed_data" / "swot" / "ba_river_watershed" / "swot_wse_df_with_outlier_removed_2023_2025.csv"
MSWX_FORCING_FOLDER = BASE_DIR / "processed_data" / "swot_gnn" / "training_data" / "mswx_forcing"
REACH_ATTRS_PATH = BASE_DIR / "raw_data" / "grit" / "GRIT_vietnam" / "ba_river_watershed" / "reach_attrs" / "GRITv06_reach_predictors_shared_MEKO.csv"
GRIT_REACH_PATH = BASE_DIR / "raw_data" / "grit" / "GRIT_vietnam" / "ba_river_watershed" / "reaches" / "ba_river_watershed_reaches_gritv06_with_centroid_lake.csv"

# Static feature columns (must match training_data_processing_20260202.py)
STATIC_FEATURE_COLUMNS = [
    "darea",
    "reservoir_capacity",
    "aridity_index",
    "elevation",
    "soil_conduct",
    "soil_thickness",
    "urban_frac",
    "grwl_width",
    "gswe_width_occ_1_scaled",
    "gswe_width_occ_10_scaled",
    "gswe_width_occ_20_scaled",
    "gswe_width_occ_30_scaled",
    "gswe_width_occ_40_scaled",
    "gswe_width_occ_50_scaled",
    "sinuosity",
    "lai",
    "fapar",
    "mswx_prep",
    "mswx_temp",
    "mswx_temp_range",
    "soil_30",
    "soil_60",
    "soil_100",
    "water_table_h",
    "water_table_l",
    "snow_cover_5th",
    "snow_cover_50th",
    "snow_cover_95th",
    "aridity_season",
    "prep_snow",
    "mean_cv",
    "flashiness",
    "BFI",
]
VARIABLE_NAME_LIST = ["LWd", "P", "Pres", "RelHum", "SWd", "Temp", "Wind"]


def load_segment_mapping(
    segment_mapping_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load segment-reach mapping. Include all segments; those without valid
    selected_reach_id will get WSE=no-data but still receive atmospheric and static
    from a fallback reach (first reach in segment).

    Returns:
        mapping_df: Full mapping with segment_id, selected_reach_id (may be NaN)
        segment_ids: Sorted array of all segment IDs
    """
    path = segment_mapping_path or SEGMENT_MAPPING_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Segment mapping not found: {path}. Run aggregate_wse_by_segments.ipynb first.")
    mapping_df = pd.read_csv(path)
    segment_ids = np.sort(mapping_df["segment_id"].unique())
    return mapping_df, segment_ids


def generate_segment_based_datacubes(
    segment_mapping_path: Optional[Path] = None,
    swot_wse_path: Optional[Path] = None,
    mswx_folder: Optional[Path] = None,
    reach_attrs_path: Optional[Path] = None,
    grit_reach_path: Optional[Path] = None,
    save_folder: Optional[Path] = None,
    start_date: str = "2023-10-01",
    end_date: str = "2025-12-01",
) -> None:
    """
    Generate segment-based dynamic, target, and static NetCDF datacubes.
    """
    mapping_path = segment_mapping_path or SEGMENT_MAPPING_PATH
    swot_path = swot_wse_path or SWOT_WSE_PATH
    mswx_path = mswx_folder or MSWX_FORCING_FOLDER
    reach_attrs = reach_attrs_path or REACH_ATTRS_PATH
    grit_reach = grit_reach_path or GRIT_REACH_PATH
    save_dir = Path(save_folder or DEFAULT_SAVE_FOLDER)
    save_dir.mkdir(parents=True, exist_ok=True)

    mapping_df, segment_ids = load_segment_mapping(mapping_path)
    grit_reach_df = pd.read_csv(grit_reach)

    # segment -> reach for atmospheric and static (fallback to first reach in segment if no selected_reach)
    seg_reach = mapping_df.drop_duplicates("segment_id", keep="first").set_index("segment_id")
    fallback_reach = grit_reach_df.groupby("segment_id")["fid"].first()
    segment_to_reach = {}
    segment_has_wse = {}
    for seg_id in segment_ids:
        row = seg_reach.loc[seg_id]
        sel_reach = row["selected_reach_id"]
        if pd.notna(sel_reach):
            segment_to_reach[seg_id] = int(sel_reach)
            segment_has_wse[seg_id] = True
        elif seg_id in fallback_reach.index:
            # Use first reach in segment for atmospheric and static; no WSE data
            segment_to_reach[seg_id] = int(fallback_reach[seg_id])
            segment_has_wse[seg_id] = False

    # Drop segments with no reach at all (no atmospheric/static source)
    segment_ids = np.array([s for s in segment_ids if segment_to_reach.get(s) is not None])
    n_segments = len(segment_ids)
    all_dates = pd.date_range(start_date, end_date, freq="D")

    # Load SWOT and compute WSE features per reach
    swot_node_df = pd.read_csv(swot_path)
    swot_node_df["date"] = pd.to_datetime(swot_node_df["date"])

    wse_mean_per_reach = swot_node_df.groupby("fid")["wse"].mean().rename("reach_mean")
    swot_node_df = swot_node_df.merge(wse_mean_per_reach, on="fid")
    wse_std_per_reach = swot_node_df.groupby("fid")["wse"].std().rename("reach_std")
    swot_node_df = swot_node_df.merge(wse_std_per_reach, on="fid")
    swot_node_df["wse_anomaly"] = swot_node_df["wse"] - swot_node_df["reach_mean"]
    epsilon = 1e-8
    swot_node_df["wse_norm"] = swot_node_df["wse_anomaly"] / (swot_node_df["reach_std"] + epsilon)

    # Build segment x time cubes from reach data
    shape = (n_segments, len(all_dates))
    wse_cube = np.full(shape, np.nan, dtype=np.float32)
    mask_cube = np.zeros(shape, dtype=np.int8)
    latest_wse_cube = np.zeros(shape, dtype=np.float32)
    lag_cube = np.zeros(shape, dtype=np.float32)
    time_doy_sin_cube = np.zeros(shape, dtype=np.float32)
    time_doy_cos_cube = np.zeros(shape, dtype=np.float32)

    doy = all_dates.dayofyear
    time_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    time_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    for seg_idx, segment_id in enumerate(tqdm(segment_ids, desc="Processing segments")):
        if segment_has_wse[segment_id]:
            reach_id = segment_to_reach[segment_id]
            reach_data = swot_node_df[swot_node_df["fid"] == reach_id].set_index("date")
            full_series = reach_data[wse_option].reindex(all_dates)

            mask = (~full_series.isna()).astype(np.int8)
            latest_wse = full_series.ffill().fillna(0)

            valid = mask.values
            last_valid_idx = np.where(valid == 1, np.arange(len(valid)), np.nan)
            last_valid_idx = pd.Series(last_valid_idx).ffill().to_numpy()
            last_valid_idx = np.where(np.isnan(last_valid_idx), 0, last_valid_idx)
            seq_idx = np.arange(len(valid))
            lag = seq_idx - last_valid_idx

            wse_cube[seg_idx, :] = full_series.values
            mask_cube[seg_idx, :] = mask.values
            latest_wse_cube[seg_idx, :] = latest_wse.values
            lag_cube[seg_idx, :] = lag
        else:
            # No WSE data: mask=0, wse=nan, latest_wse=0, lag=large
            wse_cube[seg_idx, :] = np.nan
            mask_cube[seg_idx, :] = 0
            latest_wse_cube[seg_idx, :] = 0.0
            lag_cube[seg_idx, :] = np.arange(len(all_dates), dtype=np.float32)

        time_doy_sin_cube[seg_idx, :] = time_sin
        time_doy_cos_cube[seg_idx, :] = time_cos

    # Atmospheric forcing: map reach-level to segment-level
    # Pre-compute the list of representative reach_ids (one per segment, in order)
    reach_ids_for_segs = [segment_to_reach[s] for s in segment_ids]
    variable_cube_dict = {}
    for variable_name in VARIABLE_NAME_LIST:
        fp = Path(mswx_path) / f"{variable_name}_Past_Daily_combined_catchment_avg_reach_level.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Atmospheric forcing not found: {fp}")
        var_df = pd.read_csv(fp)
        var_df["time"] = pd.to_datetime(var_df["time"])
        if variable_name == "Pres":
            var_df["var"] = var_df["var"] / 100

        # Pivot to (fid × time) matrix, then index rows by each segment's reach
        var_pivot = (
            var_df[var_df["time"].isin(all_dates)]
            .pivot_table(index="fid", columns="time", values="var", aggfunc="first")
            .reindex(columns=all_dates)
        )
        var_cube = np.nan_to_num(
            var_pivot.reindex(reach_ids_for_segs).values.astype(np.float32), nan=0.0
        )
        variable_cube_dict[variable_name] = var_cube

    # Dynamic dataset
    ds = xr.Dataset(
        data_vars={
            "wse": (["segment", "time"], wse_cube),
            "obs_mask": (["segment", "time"], mask_cube),
            "latest_wse": (["segment", "time"], latest_wse_cube),
            "days_since_last_obs": (["segment", "time"], lag_cube),
            "time_doy_sin": (["segment", "time"], time_doy_sin_cube),
            "time_doy_cos": (["segment", "time"], time_doy_cos_cube),
            **{v: (["segment", "time"], variable_cube_dict[v]) for v in VARIABLE_NAME_LIST},
        },
        coords={"segment": segment_ids, "time": all_dates},
    )
    ds.to_netcdf(save_dir / f"ba_river_swot_dynamic_datacube_{wse_option}.nc")
    print(f"Dynamic data saved. Shape: {shape}")

    # Target dataset
    target_ds = xr.Dataset(
        data_vars={
            "wse": (["segment", "time"], wse_cube),
            "obs_mask": (["segment", "time"], mask_cube),
        },
        coords={"segment": segment_ids, "time": all_dates},
    )
    target_ds.to_netcdf(save_dir / f"ba_river_swot_target_datacube_{wse_option}.nc")

    # Static features: from reach attributes, mapped to segments
    reach_attrs_df = pd.read_csv(reach_attrs)
    for reach_id_col in ("reach_id", "fid"):
        if reach_id_col in reach_attrs_df.columns:
            break
    else:
        candidates = [c for c in reach_attrs_df.columns if "reach" in c.lower() or c == "fid"]
        reach_id_col = candidates[0] if candidates else reach_attrs_df.columns[0]

    static_cols = [c for c in STATIC_FEATURE_COLUMNS if c in reach_attrs_df.columns]
    if len(static_cols) < len(STATIC_FEATURE_COLUMNS):
        missing = set(STATIC_FEATURE_COLUMNS) - set(static_cols)
        print(f"Warning: missing static columns {missing}, using {static_cols}")

    attrs_indexed = reach_attrs_df.set_index(reach_id_col)[static_cols]
    static_cube = np.nan_to_num(
        attrs_indexed.reindex(reach_ids_for_segs).values.astype(np.float32), nan=0.0
    )

    static_ds = xr.Dataset(
        data_vars={"static_feature": (["segment", "feature"], static_cube)},
        coords={"segment": segment_ids, "feature": static_cols},
    )
    static_ds.to_netcdf(save_dir / f"ba_river_swot_static_datacube_{wse_option}.nc")
    print(f"Static data saved. Segments: {n_segments}, features: {len(static_cols)}")

    # Save segment-reach mapping for downstream use (fallback reach used when no valid selected_reach)
    segment_reach_df = pd.DataFrame(
        {
            "segment_id": segment_ids,
            "selected_reach_id": [segment_to_reach[s] for s in segment_ids],
            "has_wse": [segment_has_wse[s] for s in segment_ids],
        }
    )
    segment_reach_df.to_csv(save_dir / "segment_reach_mapping.csv", index=False)


if __name__ == "__main__":
    # Edit paths above (BASE_DIR, SEGMENT_MAPPING_PATH, etc.) or set SMART_HS_ROOT env var
    generate_segment_based_datacubes()
