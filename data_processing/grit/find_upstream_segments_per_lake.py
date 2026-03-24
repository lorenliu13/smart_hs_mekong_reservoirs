"""
Find all upstream GRIT segments per lake.

BACKGROUND
----------
River reaches (from gritv06_reaches_mekong_basin_with_pld_lakes.csv) each
belong to a *segment* (the `segment_id` column), which corresponds to the
`fid` column in gritv06_segments_mekong.csv.  A segment is a longer river
unit made up of multiple consecutive reaches.

The segments CSV carries its own connectivity columns (`upstream_l`,
`downstre_1`) that encode the segment-level graph (segment fids, not reach
fids).

GOAL
----
For each lake, find every segment that drains into it — but stop traversal
at the boundaries of upstream lakes, so each segment is attributed to
exactly one lake (the first downstream lake it can reach).

Concretely, each lake owns:
  1. Its "lake segments" — segments that contain reaches belonging to that
     lake (the physical lake body).
  2. Its "upstream river segments" — non-lake segments reachable upstream
     from the lake's entry points before hitting any upstream lake's
     segments.

The result is a non-overlapping partition: every segment in the network is
assigned to at most one lake (segments above the most-upstream lake, and
isolated sub-networks, are unassigned).

ALGORITHM
---------
1. Load reaches → derive (segment_id → lake_id) mapping.
   If a segment straddles two lakes, assign it to all lakes that have
   reaches in that segment (boundary segments are shared).

2. Build the segment-level directed graph from the segments CSV.

3. For each lake:
   a. Collect its "lake segments".
   b. Find "entry segments": upstream segment neighbours of lake segments
      that are NOT themselves lake segments (of any lake).
   c. BFS upstream from entry segments through non-lake segments,
      stopping (but not entering) any segment that belongs to another lake.
   d. Collect lake segments + traversed river segments as this lake's owned
      segments.

Input:
  gritv06_reaches_mekong_basin_with_pld_lakes.csv
  gritv06_segments_mekong.csv
  gritv06_pld_lake_graph_{THRESHOLD}sqkm.csv   (for valid lake IDs)

Output:
  gritv06_lake_upstream_segments_{THRESHOLD}sqkm.csv
  Columns:
    lake_id                   - lake ID (matches lake graph)
    lake_segments             - comma-separated segment fids that are inside
                                the lake body
    upstream_river_segments   - comma-separated non-lake segment fids
                                upstream of the lake (up to upstream lakes)
    all_upstream_segments     - lake_segments + upstream_river_segments
    n_lake_segments           - count of lake_segments
    n_upstream_river_segments - count of upstream_river_segments
    n_all_segments            - total count
"""

import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
LAKE_AREA_THRESHOLD_SQKM = 0   # Must match the lake graph that was produced

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REACHES_CSV = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reaches"
    r"\gritv06_reaches_great_mekong_with_lake_id.csv"
)
SEGMENTS_CSV = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\segments"
    r"\gritv06_segments_great_mekong.csv"
)
LAKE_GRAPH_CSV = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reservoirs"
    rf"\gritv06_great_mekong_pld_lake_graph_{LAKE_AREA_THRESHOLD_SQKM}sqkm.csv"
)
PLD_PATH = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\prior_lake_database"
    r"\swot_prior_lake_database_great_mekong_overlap_with_grit.csv"
)
OUTPUT_CSV = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reservoirs"
    rf"\gritv06_great_mekong_pld_lake_upstream_segments_{LAKE_AREA_THRESHOLD_SQKM}sqkm.csv"
)

# ---------------------------------------------------------------------------
# 1. Load valid lake IDs from the pre-built lake graph
# ---------------------------------------------------------------------------
lake_graph = pd.read_csv(LAKE_GRAPH_CSV)
valid_lake_ids: set[int] = set(
    lake_graph["lake_id"].astype("int64")
)
print(f"Valid lakes from graph: {len(valid_lake_ids)}")

# ---------------------------------------------------------------------------
# Load PLD shapefile to build lake centroid lon/lat lookup
# ---------------------------------------------------------------------------
pld = pd.read_csv(PLD_PATH)
pld["lake_id"] = pld["lake_id"].astype("int64")
# Keep first occurrence per lake_id (lon/lat are the same for all sub-polygons)
lake_lonlat: dict[int, tuple[float, float]] = (
    pld.drop_duplicates("lake_id")
    .set_index("lake_id")[["lon", "lat"]]
    .apply(lambda r: (float(r["lon"]), float(r["lat"])), axis=1)
    .to_dict()
)
print(f"Lon/lat loaded for {len(lake_lonlat)} lakes")

# ---------------------------------------------------------------------------
# 2. Build segment → lake mapping from reach data
# ---------------------------------------------------------------------------
reaches = pd.read_csv(REACHES_CSV, usecols=["reach_id", "segment_id", "lake_id"])

lake_reaches = reaches[reaches["lake_id"].notna()].copy()
lake_reaches["lake_id"] = lake_reaches["lake_id"].astype("int64")
lake_reaches["segment_id"] = lake_reaches["segment_id"].astype("int64")
lake_reaches = lake_reaches[lake_reaches["lake_id"].isin(valid_lake_ids)]

print(
    f"Lake reaches (filtered): {len(lake_reaches)}, "
    f"unique segments touched: {lake_reaches['segment_id'].nunique()}"
)

# Count how many reaches each (segment, lake) pair has.
# Boundary segments shared between lakes are assigned to ALL lakes that have
# reaches in that segment.
seg_lake_counts = (
    lake_reaches.groupby(["segment_id", "lake_id"])
    .size()
    .reset_index(name="n_reaches")
)

# lake_id → set of segment fids that ARE the lake body (including shared boundary segments)
lake_to_own_segs: dict[int, set[int]] = defaultdict(set)
for _, row in seg_lake_counts.iterrows():
    lake_to_own_segs[int(row["lake_id"])].add(int(row["segment_id"]))

all_lake_seg_ids: set[int] = set(seg_lake_counts["segment_id"].astype("int64"))
print(f"Segments assigned to a lake: {len(all_lake_seg_ids)}")

# ---------------------------------------------------------------------------
# 3. Build segment-level graph
# ---------------------------------------------------------------------------
def parse_ids(value) -> list[int]:
    if pd.isna(value):
        return []
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


segments = pd.read_csv(SEGMENTS_CSV, usecols=["fid", "upstream_l", "downstre_1"])
segments["fid"] = segments["fid"].astype("int64")

seg_to_up: dict[int, list[int]] = {
    int(row["fid"]): parse_ids(row["upstream_l"])
    for _, row in segments.iterrows()
}
seg_to_dn: dict[int, list[int]] = {
    int(row["fid"]): parse_ids(row["downstre_1"])
    for _, row in segments.iterrows()
}

print(f"Segments loaded: {len(seg_to_up)}")

# ---------------------------------------------------------------------------
# 4. For each lake, BFS upstream to collect owned segments
# ---------------------------------------------------------------------------
records = []

for lake_id in sorted(valid_lake_ids):
    own_segs: set[int] = lake_to_own_segs[lake_id]

    # Entry segments: upstream segment neighbours of lake segments that are
    # not themselves lake segments (of any lake).
    entry_segs: set[int] = set()
    for seg in own_segs:
        for up_seg in seg_to_up.get(seg, []):
            if up_seg not in all_lake_seg_ids:
                entry_segs.add(up_seg)

    # BFS upstream from entry segments through non-lake segments.
    # Stop (do not enter) any segment that belongs to another lake.
    upstream_river_segs: set[int] = set()
    queue: list[int] = list(entry_segs)
    visited: set[int] = set()

    while queue:
        seg = queue.pop()
        if seg in visited:
            continue
        visited.add(seg)

        if seg in all_lake_seg_ids:
            # Belongs to another lake — stop here (do not collect).
            continue

        upstream_river_segs.add(seg)
        queue.extend(seg_to_up.get(seg, []))

    all_segs: set[int] = own_segs | upstream_river_segs

    lon, lat = lake_lonlat.get(lake_id, (None, None))
    records.append(
        {
            "lake_id": lake_id,
            "lon": lon,
            "lat": lat,
            "lake_segments": ",".join(str(s) for s in sorted(own_segs)),
            "upstream_river_segments": ",".join(
                str(s) for s in sorted(upstream_river_segs)
            ),
            "all_upstream_segments": ",".join(str(s) for s in sorted(all_segs)),
            "n_lake_segments": len(own_segs),
            "n_upstream_river_segments": len(upstream_river_segs),
            "n_all_segments": len(all_segs),
        }
    )

# ---------------------------------------------------------------------------
# 5. Save output
# ---------------------------------------------------------------------------
result_df = pd.DataFrame(records).sort_values("lake_id").reset_index(drop=True)

print(f"\nResult shape: {result_df.shape}")
print(result_df[["lake_id", "n_lake_segments", "n_upstream_river_segments", "n_all_segments"]].head(15).to_string())
print(f"\nTotal lakes processed: {len(result_df)}")
print(f"Lakes with ≥1 upstream river segment: {(result_df['n_upstream_river_segments'] > 0).sum()}")
print(f"Lakes with 0 upstream river segments (headwater lakes): {(result_df['n_upstream_river_segments'] == 0).sum()}")

result_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to: {OUTPUT_CSV}")
