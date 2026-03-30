"""
Build the lake static attributes datacube for the lake-based SWOT-GNN.

Pipeline (end-to-end, no intermediate CSV required)
----------------------------------------------------
  gritv06_pld_lake_upstream_segments_0sqkm.csv
      all_upstream_segments  →  segment IDs (comma-separated)
  gritv06_reaches_mekong_basin_with_pld_lakes.csv
      segment_id  →  reach_id  (one segment can contain many sub-reaches)
  GRITv06_reach_predictors_shared_MEKO.csv
      reach_id  →  reach-level predictor attributes

Aggregation rules
-----------------
* darea  : value of the most downstream reach (= reach with max darea),
           representing the total upstream drainage area of the lake watershed.
* all other attributes : spatial mean across all upstream reaches.

Output
------
  swot_lake_static_datacube.nc
      dims (lake, feature) — static lake attributes

Usage
-----
    python build_static_datacube_lake_based_20260319.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────

LAKE_GRAPH_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    "/gritv06_great_mekong_pld_lake_graph_0sqkm.csv"
)
SAVE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/mekong_lakes_swotpld_era5_ifshres10d_gritv06_202312_202602_qc"
)

# Reach-level predictor attributes  (reach_id == fid in the reaches table)
REACH_ATTRS_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/attrs"
    "/GRITv06_reach_predictors_shared_MEKO_YANG.csv"
)

# Lake → all upstream segment IDs (comma-separated)
LAKE_UPSTREAM_SEGS_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit"
    "/GRIT_mekong_mega_reservoirs/reservoirs"
    "/gritv06_great_mekong_pld_lake_upstream_segments_0sqkm.csv"
)

# Reach table: reach_id = reach ID, segment_id = parent segment
REACHES_WITH_LAKES_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit"
    "/GRIT_mekong_mega_reservoirs/reaches"
    "/gritv06_reaches_great_mekong_with_lake_id.csv"
)

# Columns to drop from the static attributes before building the datacube
STATIC_EXCLUDE_COLS = [
    "domain",
    "mean_annual_discharge",
    "mean_sum_runoff",
    "mean_annual_discharge_da",
    "mean_sum_runoff_da",
]

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
    ids = ids[ids != -1]
    return ids


def prepare_static_attrs(
    lake_upstream_segs_csv: Path,
    reaches_with_lakes_csv: Path,
    reach_attrs_csv: Path,
) -> pd.DataFrame:
    """
    Aggregate reach-level predictor attributes to per-lake static attributes.

    Aggregation rules:
    - darea  : max across all upstream reaches (most downstream = largest darea).
    - others : spatial mean across all upstream reaches.
    - string columns : mode (most common value).

    Returns:
        DataFrame with columns [lake_id, <feature_cols>].
    """
    print("\n=== Preparing static lake attributes ===")

    print("  Loading reach attributes …")
    reach_attrs = pd.read_csv(reach_attrs_csv)
    reach_attrs = reach_attrs.set_index("reach_id")

    feature_cols = reach_attrs.columns.tolist()
    numeric_cols = reach_attrs.select_dtypes(include=[np.number]).columns.tolist()
    string_cols  = reach_attrs.select_dtypes(exclude=[np.number]).columns.tolist()

    print("  Loading reaches table (segment_id → reach_id mapping) …")
    reaches = pd.read_csv(reaches_with_lakes_csv, usecols=["reach_id", "segment_id"])
    seg_to_fids: dict[int, list[int]] = (
        reaches.groupby("segment_id")["reach_id"].apply(list).to_dict()
    )
    print(f"    {len(seg_to_fids)} unique segment_ids → {len(reaches)} reach_ids")

    print("  Loading lake upstream-segment table …")
    lake_upstream = pd.read_csv(lake_upstream_segs_csv)
    print(f"    {len(lake_upstream)} lakes")

    records = []
    for _, row in tqdm(lake_upstream.iterrows(), total=len(lake_upstream),
                       desc="  Aggregating reach attrs per lake"):
        lake_id = row["lake_id"]
        segs_str = str(row["all_upstream_segments"])

        seg_ids = [int(s.strip()) for s in segs_str.split(",")
                   if s.strip() not in ("", "nan")]

        fids = []
        for sid in seg_ids:
            fids.extend(seg_to_fids.get(sid, []))

        valid_fids = [f for f in fids if f in reach_attrs.index]
        if not valid_fids:
            records.append({"lake_id": lake_id})
            continue

        subset = reach_attrs.loc[valid_fids]
        rec = {"lake_id": lake_id}

        rec["darea"] = subset["darea"].max()
        for col in numeric_cols:
            if col == "darea":
                continue
            rec[col] = subset[col].mean()
        for col in string_cols:
            rec[col] = subset[col].mode().iloc[0] if len(subset) > 0 else np.nan

        records.append(rec)

    static_df = pd.DataFrame(records)
    static_df = static_df[["lake_id"] + feature_cols]

    print(f"  Result shape: {static_df.shape}")
    missing = static_df.isnull().sum()
    missing = missing[missing > 0]
    print(f"  Missing values:\n{missing if len(missing) else '  none'}")

    return static_df


def build_static_datacube(
    lake_ids: np.ndarray,
    static_df: pd.DataFrame,
    save_dir: Path,
    exclude_cols: list = STATIC_EXCLUDE_COLS,
) -> Path:
    """
    Build the static attributes datacube from a per-lake DataFrame.

    Drops excluded columns, aligns rows to lake_ids order (NaN-filled for any
    missing lakes), and saves as swot_lake_static_datacube.nc with dims
    (lake, feature).
    """
    print("\n=== Building static datacube ===")

    static_df = static_df.copy()
    static_df["lake_id"] = static_df["lake_id"].astype(np.int64)

    drop_cols = [c for c in exclude_cols if c in static_df.columns]
    static_df = static_df.drop(columns=drop_cols)

    static_df = static_df.set_index("lake_id")
    static_df = static_df.reindex(lake_ids)

    feature_names = static_df.columns.tolist()
    static_cube = static_df.to_numpy(dtype=np.float32)
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
            "excluded_cols": ", ".join(exclude_cols),
            "created_by": "build_static_datacube_lake_based_20260319.py",
        },
    )

    out_path = save_dir / "swot_lake_static_datacube.nc"
    ds.to_netcdf(out_path)
    print(f"Static datacube saved → {out_path}  shape: {n_lakes} lakes × {n_features} features")
    return out_path


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading lake IDs from: {LAKE_GRAPH_CSV}")
    lake_ids = load_lake_ids_from_graph(LAKE_GRAPH_CSV)
    print(f"  Found {len(lake_ids)} lakes in GRIT PLD lake graph.")

    static_df = prepare_static_attrs(
        lake_upstream_segs_csv=LAKE_UPSTREAM_SEGS_CSV,
        reaches_with_lakes_csv=REACHES_WITH_LAKES_CSV,
        reach_attrs_csv=REACH_ATTRS_CSV,
    )

    build_static_datacube(
        lake_ids=lake_ids,
        static_df=static_df,
        save_dir=SAVE_DIR,
    )
