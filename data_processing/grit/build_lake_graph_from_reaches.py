"""
Build a lake-level connectivity graph from GRIT reach data.

BACKGROUND
----------
The raw CSV stores connectivity at the *reach* level:
  - each row is one river reach (identified by `fid`)
  - `upstream_l`  : fid(s) of the reach(es) immediately upstream  (comma-separated when a confluence)
  - `downstre_1`  : fid(s) of the reach(es) immediately downstream
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

Input:  gritv06_reaches_mekong_basin_with_pld_lakes.csv
        Columns used: fid, upstream_l, downstre_1, lake_id

Output: lake_graph_with_upstream_downstream.csv
        Columns:
          lake_id             - unique lake ID (int64)
          most_downstream_fid - fid of the most downstream GRIT reach in this lake
          upstream_lake_ids   - comma-separated upstream lake IDs (empty if none)
          downstream_lake_id  - downstream lake ID (empty if none / outlet to sea)
"""

import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_CSV = (
    "E:/Project_2025_2026/Smart_hs/raw_data/grit/"
    "GRIT_mekong_mega_reservoirs/reaches/"
    "gritv06_reaches_mekong_basin_with_pld_lakes.csv"
)
OUTPUT_CSV = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\reservoirs\gritv06_pld_lake_graph.csv"
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"Total rows: {len(df)},  Lake rows: {df['lake_id'].notna().sum()}")

# Keep only rows that belong to a lake (drop plain river reaches)
lake_df = df[df["lake_id"].notna()].copy()

# lake_id is stored as float64 in the CSV (e.g. 4420000122.0).
# Use int64 explicitly — these values exceed the int32 range (max ~2.1 billion)
# so plain astype(int) would silently overflow on some systems.
lake_df["lake_id"] = lake_df["lake_id"].astype("int64")


# ---------------------------------------------------------------------------
# Helper: parse connectivity cells
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
# Build helper lookups — built from ALL reaches, not just lake reaches.
# This is essential for the traversal in steps 3 & 4: when following the
# river network downstream (or upstream) from a lake's exit (or entry) point,
# we pass through plain river reaches that have no lake_id. Without lookups
# covering those reaches, traversal would stop at the lake boundary.
# ---------------------------------------------------------------------------

# Parse connectivity for every reach in the full dataframe
df["_up_ids"] = df["upstream_l"].apply(parse_ids)
df["_dn_ids"] = df["downstre_1"].apply(parse_ids)

# ALL reaches: fid → list of upstream fids
all_fid_to_up: dict[int, list[int]] = dict(zip(df["fid"], df["_up_ids"]))
# ALL reaches: fid → list of downstream fids
all_fid_to_dn: dict[int, list[int]] = dict(zip(df["fid"], df["_dn_ids"]))

# Mapping: reach fid → lake_id  (lake reaches only)
# Used to check "does a reach belong to a lake, and which one?"
reach_to_lake: dict[int, int] = dict(
    zip(lake_df["fid"], lake_df["lake_id"])
)

# Mapping: lake_id → set of reach fids  (inverted from reach_to_lake)
lake_to_reaches: dict[int, set[int]] = defaultdict(set)
for fid, lake in reach_to_lake.items():
    lake_to_reaches[lake].add(fid)

# Convenience lookups restricted to lake reaches (used inside the lake for
# finding exit reaches and the upstream-count tie-breaker)
lake_df["_up_ids"] = lake_df["upstream_l"].apply(parse_ids)
lake_df["_dn_ids"] = lake_df["downstre_1"].apply(parse_ids)
fid_to_up: dict[int, list[int]] = dict(zip(lake_df["fid"], lake_df["_up_ids"]))
fid_to_dn: dict[int, list[int]] = dict(zip(lake_df["fid"], lake_df["_dn_ids"]))


# ---------------------------------------------------------------------------
# Traversal helpers
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

    Unlike a single-return version, this collects every downstream lake across
    all branches so that bifurcating rivers (deltas) produce multiple results.
    Traversal stops when it enters a lake reach (records it, does not go further
    into that lake — mirrors the upstream logic exactly).

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
    return downstream_lakes  # empty = reached sea / dataset boundary


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
# Main loop: for each lake find most-downstream reach, upstream/downstream lakes
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

    # ---- Step 2: pick ONE most-downstream reach ------------------------------
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

    # ---- Step 3: identify downstream lakes ------------------------------------
    # Start from the downstream neighbours of ALL exit reaches so that
    # bifurcating channels each get explored and can lead to different lakes.
    all_exit_dn_fids = [
        d for fid in exit_reaches for d in all_fid_to_dn.get(fid, [])
    ]
    downstream_lake_ids = find_downstream_lakes(
        all_exit_dn_fids, lake_id, all_fid_to_dn, reach_to_lake
    )

    # ---- Step 4: identify upstream lakes -------------------------------------
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
    records.append(
        {
            "lake_id": lake_id,
            "most_downstream_fid": most_downstream_fid,
            # Sort for deterministic output; empty string means no upstream lakes
            "upstream_lake_ids": ",".join(str(x) for x in sorted(upstream_lake_ids)),
            # Sort for deterministic output; empty string means no downstream lakes
            "downstream_lake_ids": ",".join(str(x) for x in sorted(downstream_lake_ids)),
        }
    )

# ---------------------------------------------------------------------------
# Build output dataframe and save
# ---------------------------------------------------------------------------
result_df = pd.DataFrame(records).sort_values("lake_id").reset_index(drop=True)

print(f"\nResult shape: {result_df.shape}")
print(result_df.head(10).to_string())
print(f"\nLakes with upstream lake(s):  {(result_df['upstream_lake_ids'] != '').sum()}")
print(f"Lakes with downstream lake(s): {(result_df['downstream_lake_ids'] != '').sum()}")

result_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to: {OUTPUT_CSV}")
