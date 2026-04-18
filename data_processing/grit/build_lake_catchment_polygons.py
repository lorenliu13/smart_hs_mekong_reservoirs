"""
Build merged catchment polygons for each lake.

For each lake, all GRIT sub-catchment polygons (one per segment) that drain
into the lake — including the lake body segments and all upstream river
segments up to (but not including) upstream lake boundaries — are dissolved
into a single polygon.

The output is one polygon per lake, with attribute columns taken directly
from the lake graph CSV so the result is self-contained.

Input:
  GRITv06_catchments_mekong.shp
      - One polygon per segment; join key is column ``global_id`` (= segment fid)

  gritv06_pld_lake_upstream_segments_{N}sqkm.csv
      - Output of find_upstream_segments_per_lake.py
      - Columns: lake_id, all_upstream_segments (comma-separated segment fids)

  gritv06_pld_lake_graph_{N}sqkm.csv
      - Lake graph for attribute columns

Output:
  gritv06_pld_lake_catchments_{N}sqkm.shp  (+ .gpkg mirror)
  Columns (attributes from lake graph + geometry):
    lake_id              - unique lake ID
    most_downstream_fid  - fid of the most downstream reach for this lake
    downstream_river_fid - first non-lake reach(es) downstream of lake exit
    upstream_lake_ids    - comma-separated upstream lake IDs
    downstream_lake_ids  - downstream lake ID(s); -1 = basin outlet
    n_segments           - number of sub-catchment polygons merged
    geometry             - dissolved (unioned) polygon
"""

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
AREA_THRESHOLD_SQKM = 0.1  # Must match the lake graph that was produced
OBS_COUNT_THRESHOLD = 30   # Must match the lake graph that was produced

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SUFFIX = rf"area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}"
_RESERVOIRS_DIR = (
    rf"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    rf"\GRIT_mekong_mega_reservoirs\reservoirs\lake_graph_{_SUFFIX}"
)
CATCHMENT_SHP = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\catchments"
    r"\GRITv06_catchments_great_mekong.shp"
)
UPSTREAM_SEGS_CSV = rf"{_RESERVOIRS_DIR}\gritv06_great_mekong_pld_lake_upstream_segments_{_SUFFIX}.csv"
LAKE_GRAPH_CSV   = rf"{_RESERVOIRS_DIR}\gritv06_great_mekong_pld_lake_graph_{_SUFFIX}.csv"
OUTPUT_SHP       = rf"{_RESERVOIRS_DIR}\gritv06_great_mekong_pld_lake_catchments_{_SUFFIX}.shp"
OUTPUT_GPKG      = OUTPUT_SHP.replace(".shp", ".gpkg")

# ---------------------------------------------------------------------------
# 1. Load lake graph (attributes to carry into output)
# ---------------------------------------------------------------------------
lake_graph = pd.read_csv(LAKE_GRAPH_CSV)
# Exclude terminal node (-1)
# lake_graph = lake_graph[lake_graph["lake_id"] != -1].copy()
lake_graph["lake_id"] = lake_graph["lake_id"].astype("int64")
print(f"Lake graph rows (excl. terminal): {len(lake_graph)}")

# ---------------------------------------------------------------------------
# 2. Load upstream segments table
# ---------------------------------------------------------------------------
upstream_df = pd.read_csv(UPSTREAM_SEGS_CSV)
upstream_df["lake_id"] = upstream_df["lake_id"].astype("int64")
print(f"Upstream segments rows: {len(upstream_df)}")


def parse_seg_ids(value) -> list[int]:
    """Parse a comma-separated segment id string into a list of ints."""
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


# Build lake_id → set of segment fids (all_upstream_segments column)
lake_to_segs: dict[int, set[int]] = {
    int(row["lake_id"]): set(parse_seg_ids(row["all_upstream_segments"]))
    for _, row in upstream_df.iterrows()
}

# Collect every segment fid that is needed (union across all lakes)
all_needed_segs: set[int] = set()
for segs in lake_to_segs.values():
    all_needed_segs.update(segs)

print(f"Total unique segment fids needed: {len(all_needed_segs)}")

# ---------------------------------------------------------------------------
# 3. Load catchment shapefile — only keep rows we need
# ---------------------------------------------------------------------------
print("Loading catchment shapefile …")
catchments = gpd.read_file(CATCHMENT_SHP)
print(f"  Total catchment polygons: {len(catchments)}")
print(f"  CRS: {catchments.crs}")
print(f"  Columns: {list(catchments.columns)}")

# The join key is `global_id` (= segment fid)
catchments["global_id"] = catchments["global_id"].astype("int64")

# Filter to only the needed segments (large file — avoid unnecessary work)
catchments = catchments[catchments["global_id"].isin(all_needed_segs)].copy()
print(f"  Catchment polygons after filter: {len(catchments)}")

# Index by global_id for fast lookup
catchment_by_seg: dict[int, object] = dict(
    zip(catchments["global_id"], catchments["geometry"])
)

# ---------------------------------------------------------------------------
# 4. Dissolve catchments per lake
# ---------------------------------------------------------------------------
print("Dissolving catchments per lake …")

records = []
missing_count = 0

for lake_id, seg_ids in sorted(lake_to_segs.items()):
    # Collect geometries for this lake's segments
    geoms = []
    missing = []
    for seg in seg_ids: # loop over segment fids for this lake
        geom = catchment_by_seg.get(seg) # look up geometry by segment fid
        if geom is not None: # if found, add to list
            geoms.append(geom)
        else: # if not found, track missing segment fid
            missing.append(seg)

    if missing: # if any missing segments, log a warning and count them
        missing_count += len(missing)
        # print a warning for this lake, but keep going to process the ones we do have
        print(f"  Note: lake {lake_id} — {len(missing)} segment(s) missing from catchment layer: {missing}")

    if not geoms:
        # If no geometries found for this lake, log a warning and skip to next lake
        print(f"  WARNING: lake {lake_id} — no catchment polygons found, skipping")
        continue

    # Dissolve all sub-catchments into one polygon
    merged_geom = unary_union(geoms)

    # Pull attribute row from lake graph
    graph_row = lake_graph[lake_graph["lake_id"] == lake_id]
    if graph_row.empty:
        print(f"  WARNING: lake {lake_id} not in lake graph, skipping")
        continue
    row = graph_row.iloc[0]

    records.append(
        {
            "lake_id": lake_id,
            "mst_ds_fid": str(row["most_downstream_fid"]),
            "ds_riv_fid": str(row["downstream_river_fid"]),
            "us_lake_id": str(row["upstream_lake_ids"]),
            "ds_lake_id": str(row["downstream_lake_ids"]),
            "n_segments": len(geoms),
            "geometry": merged_geom,
        }
    )

if missing_count:
    print(
        f"  Note: {missing_count} segment fid(s) had no matching catchment polygon "
        "(may be edge segments not present in the catchment layer)"
    )

print(f"Lakes with merged polygon: {len(records)}")

# ---------------------------------------------------------------------------
# 5. Build GeoDataFrame and save
# ---------------------------------------------------------------------------
result_gdf = gpd.GeoDataFrame(records, crs=catchments.crs)
result_gdf = result_gdf.sort_values("lake_id").reset_index(drop=True)

print(f"\nResult shape: {result_gdf.shape}")
print(result_gdf[["lake_id", "n_segments", "us_lake_id", "ds_lake_id"]].head(10).to_string())

result_gdf.to_file(OUTPUT_SHP)
print(f"\nSaved shapefile : {OUTPUT_SHP}")

result_gdf.to_file(OUTPUT_GPKG, driver="GPKG")
print(f"Saved GeoPackage: {OUTPUT_GPKG}")
