"""
Prepare static catchment attributes for PLD lake nodes.

Join chain
----------
  gritv06_pld_lake_upstream_segments_0sqkm.csv
      all_upstream_segments  →  segment IDs (comma-separated)
  gritv06_reaches_mekong_basin_with_pld_lakes.csv
      segment_id  →  fid  (one segment can contain many sub-reaches / fids)
  GRITv06_reach_predictors_shared_MEKO.csv
      reach_id == fid  →  reach-level predictor attributes

Aggregation rules
-----------------
* darea  : value of the most downstream reach (= reach with max darea),
           representing the total upstream drainage area of the lake watershed.
* all other attributes : spatial mean across all upstream reaches (fids).

Output
------
  lake_static_attrs.csv   — rows = lakes, columns = lake_id + static features

Usage
-----
    python prepare_lake_static_attrs_20260319.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─── Paths ─────────────────────────────────────────────────────────────────────

# Reach-level predictor attributes  (reach_id == fid in the reaches table)
# Use the static attributes combined by Mekong and Yangtzi
REACH_ATTRS_CSV = Path(
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\Grit_ARC\output_v06\attrs"
    "/GRITv06_reach_predictors_shared_MEKO_YANG.csv"
)

# Lake → all upstream segment IDs (comma-separated)
LAKE_UPSTREAM_SEGS_CSV = Path(
    "E:/Project_2025_2026/Smart_hs/raw_data/grit"
    "/GRIT_mekong_mega_reservoirs/reservoirs"
    "/gritv06_great_mekong_pld_lake_upstream_segments_0sqkm.csv"
)

# Reach table: fid = reach ID, segment_id = parent segment
REACHES_WITH_LAKES_CSV = Path(
    "E:/Project_2025_2026/Smart_hs/raw_data/grit"
    "/GRIT_mekong_mega_reservoirs/reaches"
    "/gritv06_reaches_great_mekong_with_lake_id.csv"
)

# Output directory (will be created if absent)
SAVE_DIR = Path(
    "E:/Project_2025_2026/Smart_hs/processed_data"
    "/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/training_data_lake_based_great_mekong_20260325"
)

# ─── Load data ─────────────────────────────────────────────────────────────────

print("Loading reach attributes …")
reach_attrs = pd.read_csv(REACH_ATTRS_CSV)
reach_attrs = reach_attrs.set_index("reach_id")   # reach_id == fid

FEATURE_COLS = reach_attrs.columns.tolist()
NUMERIC_FEATURE_COLS = reach_attrs.select_dtypes(include=[np.number]).columns.tolist()
STRING_FEATURE_COLS  = reach_attrs.select_dtypes(exclude=[np.number]).columns.tolist()

print("Loading reaches table (segment_id → fid mapping) …")
reaches = pd.read_csv(REACHES_WITH_LAKES_CSV, usecols=["reach_id", "segment_id"])

# Build segment_id → list[fid] lookup
seg_to_fids: dict[int, list[int]] = (
    reaches.groupby("segment_id")["reach_id"].apply(list).to_dict()
)
print(f"  {len(seg_to_fids)} unique segment_ids mapped to {len(reaches)} reach_id")

print("Loading lake upstream-segment table …")
lake_upstream = pd.read_csv(LAKE_UPSTREAM_SEGS_CSV)
print(f"  {len(lake_upstream)} lakes")

# ─── Build per-lake static attributes ──────────────────────────────────────────

records = []

for _, row in tqdm(lake_upstream.iterrows(), total=len(lake_upstream),
                   desc="Aggregating reach attrs per lake"):
    lake_id = row["lake_id"]
    segs_str = str(row["all_upstream_segments"])

    # Parse comma-separated segment IDs
    seg_ids = [int(s.strip()) for s in segs_str.split(",")
               if s.strip() not in ("", "nan")]

    # Expand segment IDs → fids (sub-reaches)
    fids = []
    for sid in seg_ids:
        fids.extend(seg_to_fids.get(sid, []))

    # Keep only fids present in reach_attrs
    valid_fids = [f for f in fids if f in reach_attrs.index]
    if not valid_fids:
        records.append({"lake_id": lake_id})
        continue

    subset = reach_attrs.loc[valid_fids]

    # --- Aggregate ---
    rec = {"lake_id": lake_id}

    # darea: drainage area of the most downstream reach (= max across all upstream)
    rec["darea"] = subset["darea"].max()

    # All other numeric features: spatial mean across upstream reaches
    for col in NUMERIC_FEATURE_COLS:
        if col == "darea":
            continue
        rec[col] = subset[col].mean()

    # String features: mode (most common value), e.g. 'domain'
    for col in STRING_FEATURE_COLS:
        rec[col] = subset[col].mode().iloc[0] if len(subset) > 0 else np.nan

    records.append(rec)

# Assemble DataFrame with original feature column order
static_df = pd.DataFrame(records)
ordered_cols = ["lake_id"] + FEATURE_COLS
static_df = static_df[ordered_cols]

print(f"\nResult shape: {static_df.shape}")
print(static_df.head(3).to_string())
missing = static_df.isnull().sum()
missing = missing[missing > 0]
print(f"\nMissing values:\n{missing if len(missing) else 'none'}")

# ─── Save ──────────────────────────────────────────────────────────────────────

SAVE_DIR.mkdir(parents=True, exist_ok=True)
out_path = SAVE_DIR / "lake_graph_static_attrs_0sqkm.csv"
static_df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
