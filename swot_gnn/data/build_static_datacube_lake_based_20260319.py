"""
Build the lake static attributes datacube for the lake-based SWOT-GNN.

Output:
  swot_lake_static_datacube.nc
      dims (lake, feature) — static lake attributes from lake_graph_static_attrs_0sqkm.csv

Usage:
    python build_static_datacube_lake_based_20260319.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────────

LAKE_GRAPH_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    "/gritv06_pld_lake_graph_0sqkm.csv"
)
SAVE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/training_data_lake_based_great_mekong_20260325"
)
LAKE_STATIC_ATTRS_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/training_data_lake_based_great_mekong_20260325/lake_graph_static_attrs_0sqkm.csv"
)

# Columns to drop from the static attributes CSV before building the datacube
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

    drop_cols = [c for c in exclude_cols if c in attrs_df.columns]
    attrs_df = attrs_df.drop(columns=drop_cols)

    attrs_df = attrs_df.set_index("lake_id")
    attrs_df = attrs_df.reindex(lake_ids)

    feature_names = attrs_df.columns.tolist()
    static_cube = attrs_df.to_numpy(dtype=np.float32)
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

    build_static_datacube(
        lake_ids=lake_ids,
        save_dir=SAVE_DIR,
        static_attrs_csv=LAKE_STATIC_ATTRS_CSV,
    )
