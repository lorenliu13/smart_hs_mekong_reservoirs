"""
Build a lake-level connectivity graph from GRIT reach data.

BACKGROUND
----------
The raw CSV stores connectivity at the *reach* level:
  - each row is one river reach (identified by `reach_id`)
  - `upstream_l`  : reach_id(s) of the reach(es) immediately upstream  (comma-separated when a confluence)
  - `downstre_1`  : reach_id(s) of the reach(es) immediately downstream
  - `lake_id`     : which lake this reach belongs to (NaN for non-lake reaches)

A single lake can span many reaches (e.g. a large reservoir cut into segments).
This script "zooms out" to build a lake-level graph where each node is a lake
and edges represent water flowing from one lake into another.

ALGORITHM OVERVIEW
------------------
For each lake:
  1. Collect all its reach fids.
  2. Find the "exit reach" — the reach whose downstream neighbour is OUTSIDE the
     lake. That reach is the most-downstream reach of the lake.
  3. From the exit reach, follow downstream links reach-by-reach through any
     intervening plain river reaches until the NEXT lake is found → downstream lake.
  4. From each entry point of the lake (upstream neighbours outside the lake),
     follow upstream links reach-by-reach through plain river reaches until a
     lake reach is found → upstream lake(s).

Key difference from a naive approach: steps 3 & 4 traverse the full river
network rather than only checking the immediately adjacent reach. This correctly
handles cases where two lakes are separated by one or more plain river reaches.

Only lakes with poly_area > LAKE_AREA_THRESHOLD_SQKM (from the SWOT PLD) are
analysed. The threshold is embedded in the output filename.

A terminal node (lake_id = -1) representing the most-downstream river reach(es)
of the basin is appended to the graph, allowing lakes that drain directly to the
sea / dataset boundary to have an explicit downstream target.

Input:  gritv06_reaches_mekong_basin_with_pld_lakes.csv
        Columns used: reach_id, upstream_l, downstre_1, lake_id

        swot_prior_lake_database_mekong_overlap_with_grit.shp
        Columns used: lake_id, poly_area

Output: gritv06_pld_lake_graph_{threshold}sqkm.csv
        Columns:
          lake_id              - unique lake ID (-1 = terminal river node)
          most_downstream_fid  - fid of the most downstream reach for this node
          downstream_river_fid - fid(s) of the first non-lake reach(es) downstream
                                 of the lake exit (empty for the terminal node)
          upstream_lake_ids    - comma-separated upstream lake IDs (empty if none)
          downstream_lake_ids  - downstream lake ID(s); -1 means basin outlet
"""

import pandas as pd
import geopandas as gpd
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# LAKE_AREA_THRESHOLD_SQKM = 0   # Only analyse lakes with poly_area > this value (sq km)
TERMINAL_NODE_ID = -1            # Assigned to lakes with no downstream lake (basin outlets).
AREA_THRESHOLD_SQKM = 0.1  # Minimum lake area in square kilometers to retain
OBS_COUNT_THRESHOLD = 20  # Minimum number of daily observations to retain a lake
# When True, reaches shared by multiple lakes are assigned exclusively to the
# most-downstream lake (determined by BFS downstream from the contested reach).
DEDUPLICATE_SHARED_REACHES = True

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLD_PATH = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\prior_lake_database"
    r"\swot_prior_lake_database_great_mekong_overlap_with_grit.csv"
)
INPUT_CSV = (
    "E:/Project_2025_2026/Smart_hs/raw_data/grit/"
    "GRIT_mekong_mega_reservoirs/reaches/"
    "gritv06_reaches_great_mekong_with_lake_id.csv"
)
_OUTPUT_DIR = Path(
    rf"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reservoirs\lake_graph_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}"
)
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = (
    _OUTPUT_DIR
    / rf"gritv06_great_mekong_pld_lake_graph_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}.csv"
)
SWOT_DAILY_CSV = (
    r"E:\Project_2025_2026\Smart_hs\processed_data\swot\great_mekong_river_basin\lakes_daily"
    rf"\swot_lake_2023_12_2026_02_daily_wse_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}.csv"
)

# ---------------------------------------------------------------------------
# 1. Load PLD and derive valid lake IDs from SWOT daily WSE file
#    (area and observation-count filters are already applied upstream)
# ---------------------------------------------------------------------------
pld = gpd.read_file(PLD_PATH)
pld["poly_area"] = pd.to_numeric(pld["poly_area"], errors="coerce")
pld["lake_id"]   = pd.to_numeric(pld["lake_id"],   errors="coerce").astype("int64")
print(f"PLD lakes total: {len(pld)}")

# Build lake_id → (lon, lat) lookup from PLD centroid attributes
lake_lonlat: dict[int, tuple[float, float]] = (
    pld.drop_duplicates("lake_id")
    .set_index("lake_id")[["lon", "lat"]]
    .apply(lambda r: (float(r["lon"]), float(r["lat"])), axis=1)
    .to_dict()
)

# Valid lakes are those present in the SWOT daily file (already area- and
# obs-count-filtered by swot_lake_daily_wse_postprocess.py)
swot_daily = pd.read_csv(SWOT_DAILY_CSV, usecols=["lake_id"])
valid_lake_ids: set[int] = set(swot_daily["lake_id"].astype("int64").unique())
print(f"Valid lakes from SWOT daily WSE: {len(valid_lake_ids)}")

# ---------------------------------------------------------------------------
# 2. Load reach data
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"Total rows: {len(df)},  Lake rows: {df['lake_id'].notna().sum()}")

# Keep only lake reaches that belong to the area-filtered lakes
lake_df = df[df["lake_id"].notna()].copy()
lake_df["lake_id"] = lake_df["lake_id"].astype("int64")
lake_df = lake_df[lake_df["lake_id"].isin(valid_lake_ids)].copy()
print(
    f"Lake rows after filtering to valid lakes: {len(lake_df)}, "
    f"unique lakes: {lake_df['lake_id'].nunique()}"
)

# ---------------------------------------------------------------------------
# 3. Helper: parse connectivity cells
# ---------------------------------------------------------------------------
def parse_ids(value) -> list[int]:
    """
    Parse a connectivity cell into a list of integer reach fids.

    The CSV stores neighbours as:
      - NaN          → no neighbour (headwater or outlet)
      - '330130649'  → single neighbour
      - '330130230,330130106' → multiple neighbours (confluence / bifurcation)
    """
    if pd.isna(value):
        return []
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


# ---------------------------------------------------------------------------
# 4. Build helper lookups — built from ALL reaches, not just lake reaches.
# This is essential for the traversal in steps 3 & 4: when following the
# river network downstream (or upstream) from a lake's exit (or entry) point,
# we pass through plain river reaches that have no lake_id. Without lookups
# covering those reaches, traversal would stop at the lake boundary.
# ---------------------------------------------------------------------------

# Parse connectivity for every reach in the full dataframe
df["_up_ids"] = df["upstream_l"].apply(parse_ids)
df["_dn_ids"] = df["downstre_1"].apply(parse_ids)

# ALL reaches: fid → list of upstream fids
all_fid_to_up: dict[int, list[int]] = dict(zip(df["reach_id"], df["_up_ids"]))
# ALL reaches: fid → list of downstream fids
all_fid_to_dn: dict[int, list[int]] = dict(zip(df["reach_id"], df["_dn_ids"]))

# ---------------------------------------------------------------------------
# 4b. (Optional) Deduplicate reaches shared by multiple lakes.
#
# Two conflict types are resolved:
#   (a) Reach-level : a reach_id appears under >1 lake_id directly.
#   (b) Segment-level: a segment_id (column "segment_id" in the reaches CSV)
#       has reaches belonging to >1 lake.
#
# For every contested reach (union of both types), the candidate lakes are the
# set of lakes that claim the reach directly OR share its segment. BFS is run
# downstream from each contested reach through plain (non-contested) river
# reaches; the first candidate-lake exclusive reach encountered wins. Rows for
# losing lakes are dropped from lake_df so all subsequent lookups reflect the
# deduplication automatically.
# ---------------------------------------------------------------------------

# --- (a) Reach-level conflicts ---
reach_lake_pairs = lake_df[["reach_id", "lake_id"]].drop_duplicates()
reach_claim_counts = reach_lake_pairs.groupby("reach_id")["lake_id"].count()
reach_contested: set[int] = set(reach_claim_counts[reach_claim_counts > 1].index)

# reach_id → set of lakes that directly claim it
reach_to_direct_lakes: dict[int, set[int]] = (
    reach_lake_pairs.groupby("reach_id")["lake_id"].apply(set).to_dict()
)

# --- (b) Segment-level conflicts ---
# segment_id → set of lake_ids that have reaches in it
seg_to_lakes: dict[int, set[int]] = (
    lake_df.groupby("segment_id")["lake_id"].apply(set).to_dict()
)
contested_segments: set[int] = {
    seg for seg, lakes in seg_to_lakes.items() if len(lakes) > 1
}
# All reaches in a contested segment become contested
seg_contested: set[int] = set(
    lake_df.loc[lake_df["segment_id"].isin(contested_segments), "reach_id"]
)

# reach_id → segment_id (single value per reach; same reach never spans segments)
reach_to_seg: dict[int, int] = dict(zip(lake_df["reach_id"], lake_df["segment_id"]))

contested_fids: set[int] = reach_contested | seg_contested

if DEDUPLICATE_SHARED_REACHES and contested_fids:
    print(
        f"\nDeduplicating: {len(reach_contested)} reach-level conflict(s), "
        f"{len(contested_segments)} segment-level conflict(s) "
        f"({len(contested_fids)} total contested reach(es)) ..."
    )

    # Candidate lakes for each contested reach = direct claimants ∪ segment-mates
    reach_to_candidates: dict[int, frozenset[int]] = {
        fid: frozenset(
            reach_to_direct_lakes.get(fid, set())
            | seg_to_lakes.get(reach_to_seg.get(fid), set())
        )
        for fid in contested_fids
    }

    # Preliminary reach→lake using only non-contested reaches (unambiguous)
    prelim_reach_to_lake: dict[int, int] = dict(
        zip(
            lake_df.loc[~lake_df["reach_id"].isin(contested_fids), "reach_id"],
            lake_df.loc[~lake_df["reach_id"].isin(contested_fids), "lake_id"],
        )
    )

    # BFS downstream from each contested reach to find the most-downstream candidate
    contested_winner: dict[int, int] = {}
    for fid, candidates in reach_to_candidates.items():
        winner = None
        visited: set[int] = {fid}
        queue: list[int] = list(all_fid_to_dn.get(fid, []))
        while queue and winner is None:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            lake = prelim_reach_to_lake.get(cur)
            if lake in candidates:
                winner = lake  # first exclusive reach of a candidate lake wins
            else:
                queue.extend(all_fid_to_dn.get(cur, []))
        if winner is None:
            winner = min(candidates)  # fallback: smallest lake_id (deterministic)
        contested_winner[fid] = winner

    # Drop rows where the lake lost the contested reach
    drop_mask = lake_df["reach_id"].isin(contested_fids) & lake_df.apply(
        lambda r: contested_winner.get(r["reach_id"]) != r["lake_id"], axis=1
    )
    n_dropped = drop_mask.sum()
    lake_df = lake_df[~drop_mask].copy()
    print(f"  Dropped {n_dropped} reach-lake row(s) for non-downstream lakes")
    print(
        f"  Lake rows after deduplication: {len(lake_df)}, "
        f"unique lakes: {lake_df['lake_id'].nunique()}"
    )
elif contested_fids or contested_segments:
    print(
        f"\nNote: {len(contested_fids)} contested reach(es) across "
        f"{len(contested_segments)} segment(s) "
        "(DEDUPLICATE_SHARED_REACHES=False — keeping all assignments)."
    )

# Mapping: reach fid → lake_id  (FILTERED lake reaches only)
# Used to check "does a reach belong to a (filtered) lake, and which one?"
reach_to_lake: dict[int, int] = dict(
    zip(lake_df["reach_id"], lake_df["lake_id"])
)

# ---------------------------------------------------------------------------
# 5. Build the set of lake reach fids — used to identify the first river
#    reach immediately downstream of each lake exit.
# ---------------------------------------------------------------------------
lake_reach_fids_set: set[int] = set(reach_to_lake)

# ---------------------------------------------------------------------------
# 6. Build lake-level lookups (filtered lakes only; terminal excluded)
# ---------------------------------------------------------------------------

# Mapping: lake_id → set of reach fids  (inverted from reach_to_lake)
lake_to_reaches: dict[int, set[int]] = defaultdict(set)
for fid, lake in reach_to_lake.items():
    lake_to_reaches[lake].add(fid)

# Convenience lookups restricted to lake reaches (used inside the lake for
# finding exit reaches and the upstream-count tie-breaker)
lake_df["_up_ids"] = lake_df["upstream_l"].apply(parse_ids)
lake_df["_dn_ids"] = lake_df["downstre_1"].apply(parse_ids)
fid_to_up: dict[int, list[int]] = dict(zip(lake_df["reach_id"], lake_df["_up_ids"]))
fid_to_dn: dict[int, list[int]] = dict(zip(lake_df["reach_id"], lake_df["_dn_ids"]))


# ---------------------------------------------------------------------------
# 7. Traversal helpers
# ---------------------------------------------------------------------------
def find_downstream_lakes(
    start_fids: list[int],
    this_lake_id: int,
    all_fid_to_dn: dict,
    reach_to_lake: dict,
) -> set[int]:
    """
    BFS downstream from `start_fids` through plain river reaches until ALL
    reachable downstream lakes are found. Returns a SET of lake IDs.

    Collects every downstream lake across all branches so that bifurcating
    rivers (deltas) produce multiple results. Traversal stops when it enters a
    lake reach (records it, does not go further into that lake).

    Returns an empty set if no downstream lake is reachable (basin outlet).
    `this_lake_id` is passed so we never "find" the lake we started from.
    """
    downstream_lakes: set[int] = set()
    visited: set[int] = set()
    queue: list[int] = list(start_fids)
    while queue:
        fid = queue.pop(0)  # FIFO → BFS, finds nearest lakes first
        if fid in visited:
            continue
        visited.add(fid)
        lake = reach_to_lake.get(fid)
        if lake is not None and lake != this_lake_id:
            downstream_lakes.add(lake)  # found a downstream lake — stop this branch
        elif lake is None:
            # Plain river reach — keep following all downstream branches
            queue.extend(all_fid_to_dn.get(fid, []))
    return downstream_lakes  # empty = basin outlet (no downstream lake)


def find_upstream_lakes(
    start_fids: list[int],
    this_lake_reaches: set[int],
    all_fid_to_up: dict,
    reach_to_lake: dict,
) -> set[int]:
    """
    BFS upstream from `start_fids` through plain river reaches until lake
    reaches are found. Returns the set of upstream lake IDs.

    Traversal stops as soon as it enters any lake reach (we record that lake
    and do not traverse further into it — we only want the *nearest* upstream
    lake, not its upstream lakes).

    `this_lake_reaches` is excluded from traversal so we don't loop back into
    the current lake when exploring its own entry points.
    """
    upstream_lakes: set[int] = set()
    visited: set[int] = set()
    queue: list[int] = list(start_fids)
    while queue:
        fid = queue.pop()
        if fid in visited or fid in this_lake_reaches:
            continue
        visited.add(fid)
        lake = reach_to_lake.get(fid)
        if lake is not None:
            upstream_lakes.add(lake)
            # Do NOT traverse further into this upstream lake
        else:
            # Plain river reach — keep following upstream
            queue.extend(all_fid_to_up.get(fid, []))
    return upstream_lakes


# ---------------------------------------------------------------------------
# 8. Main loop: for each lake find most-downstream reach, upstream/downstream
# ---------------------------------------------------------------------------
records = []

for lake_id, reaches in lake_to_reaches.items():

    # ---- Step 1: find "exit reaches" ----------------------------------------
    # An exit reach is one whose downstream neighbour(s) are ALL outside this
    # lake's reach set. In a normal river tree there is exactly one per lake.
    exit_reaches = []
    for fid in reaches:
        dn_ids = fid_to_dn.get(fid, [])
        # Either no downstream neighbour (true outlet) or all downstream
        # neighbours lie outside this lake.
        if not dn_ids or all(d not in reaches for d in dn_ids):
            exit_reaches.append(fid)

    # ---- Step 2: pick ONE most-downstream lake reach -------------------------
    if len(exit_reaches) == 1:
        # Normal case — one unambiguous exit.
        most_downstream_fid = exit_reaches[0]

    elif len(exit_reaches) == 0:
        # Cycle — should not occur in a river DAG; fallback to any reach.
        most_downstream_fid = next(iter(reaches))

    else:
        # Multiple exits (e.g. distributary channels leaving a large lake).
        # Pick the exit with the most upstream reaches inside the lake —
        # that reach has the most water converging on it and is the most
        # representative "main" outlet.
        def _upstream_count(fid):
            """Count how many lake reaches lie upstream of `fid` (BFS)."""
            count = 0
            queue = list(fid_to_up.get(fid, []))
            visited = set()
            while queue:
                cur = queue.pop()
                if cur in visited or cur not in reaches:
                    continue
                visited.add(cur)
                count += 1
                queue.extend(fid_to_up.get(cur, []))
            return count

        most_downstream_fid = max(exit_reaches, key=_upstream_count)

    # ---- Step 3: identify immediate downstream river reach(es) ---------------
    # The first non-lake reach(es) immediately after the lake exit.
    # These are the connection reaches between this lake and the next lake.
    downstream_river_fids: list[int] = []
    for exit_fid in exit_reaches:
        for dn in all_fid_to_dn.get(exit_fid, []):
            if dn not in lake_reach_fids_set:   # river reach or terminal reach
                downstream_river_fids.append(dn)

    # ---- Step 4: identify downstream lakes ------------------------------------
    # Start from the downstream neighbours of ALL exit reaches so that
    # bifurcating channels each get explored and can lead to different lakes.
    # If no downstream lake is found the lake is a basin outlet → assign -1.
    all_exit_dn_fids = [
        d for fid in exit_reaches for d in all_fid_to_dn.get(fid, [])
    ]
    downstream_lake_ids = (
        find_downstream_lakes(all_exit_dn_fids, lake_id, all_fid_to_dn, reach_to_lake)
        or {TERMINAL_NODE_ID}
    )

    # ---- Step 5: identify upstream lakes -------------------------------------
    # Collect all upstream neighbours of lake reaches that lie OUTSIDE this
    # lake — these are the entry points where water flows into the lake from
    # the surrounding river network. Then traverse upstream from each entry
    # point until a lake reach is found.
    entry_fids: list[int] = []
    for fid in reaches:
        for u in all_fid_to_up.get(fid, []):
            if u not in reaches:
                entry_fids.append(u)

    upstream_lake_ids = find_upstream_lakes(
        entry_fids, reaches, all_fid_to_up, reach_to_lake
    )

    # ---- Collect result for this lake ----------------------------------------
    lon, lat = lake_lonlat.get(lake_id, (None, None))
    records.append(
        {
            "lake_id": lake_id,
            "lon": lon,
            "lat": lat,
            "most_downstream_fid": most_downstream_fid,
            # Comma-separated fids of the first river reach(es) downstream of exit
            "downstream_river_fid": ",".join(
                str(x) for x in sorted(set(downstream_river_fids))
            ),
            # Sort for deterministic output; empty string means no upstream lakes
            "upstream_lake_ids": ",".join(str(x) for x in sorted(upstream_lake_ids)),
            # Sort for deterministic output; -1 means basin outlet
            "downstream_lake_ids": ",".join(
                str(x) for x in sorted(downstream_lake_ids)
            ),
        }
    )

# ---------------------------------------------------------------------------
# 9. Build output dataframe
# ---------------------------------------------------------------------------
result_df = pd.DataFrame(records).sort_values("lake_id").reset_index(drop=True)

print(f"\nResult shape: {result_df.shape}")
print(result_df.head(10).to_string())
print(f"\nLakes with upstream lake(s):   {(result_df['upstream_lake_ids'] != '').sum()}")
print(f"Lakes with downstream lake(s): {(result_df['downstream_lake_ids'] != '').sum()}")
n_basin_outlets = (result_df["downstream_lake_ids"] == str(TERMINAL_NODE_ID)).sum()
print(f"Basin outlet lakes (-1):       {n_basin_outlets}")

# ---------------------------------------------------------------------------
# 10. Join HydroBasins sub-basin IDs from PLD onto result_df
#
# The hybasin_level_X columns are assigned in extract_lake_id_for_grit_reach.py
# and stored in the PLD CSV (PLD_PATH). We join them here by lake_id so the
# lake graph output also carries the sub-basin membership for each lake.
# ---------------------------------------------------------------------------
hybasin_cols = [c for c in pld.columns if c.startswith("hybasin_level_")]
if hybasin_cols:
    hybasin_df = (
        pld.drop_duplicates("lake_id")
        .set_index("lake_id")[hybasin_cols]
    )
    result_df = result_df.join(hybasin_df, on="lake_id")
    print(f"\nJoined {len(hybasin_cols)} HydroBasins level column(s) onto result_df.")
else:
    print(
        "\n[WARN] No hybasin_level_* columns found in PLD. "
        "Run extract_lake_id_for_grit_reach.py first to generate them."
    )

# ---------------------------------------------------------------------------
# 11. Save
# ---------------------------------------------------------------------------
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to: {OUTPUT_CSV}")
