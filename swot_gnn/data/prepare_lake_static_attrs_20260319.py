"""
Prepare static catchment attributes for PLD lake nodes.

For each lake, all upstream GRIT reaches are identified via
  gritv06_pld_lake_upstream_segments_0sqkm.csv  (lake → segment IDs)
  gritv06_reaches_mekong_basin_with_pld_lakes.csv  (segment_id == reach_id)
  GRITv06_reach_predictors_shared_MEKO.csv  (reach-level attributes)

Aggregation rules
-----------------
* darea  : taken from the most downstream reach (= reach with max darea),
           which equals the total upstream drainage area of the lake watershed.
* all other attributes : spatial mean across all upstream reaches.

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

# Reach-level predictor attributes
REACH_ATTRS_CSV = Path(
    "E:/Project_2025_2026/Smart_hs/raw_data/grit"
    "/GRIT_vietnam/ba_river_watershed/reach_attrs"
    "/GRITv06_reach_predictors_shared_MEKO.csv"
)

# Lake → all upstream segment IDs (comma-separated)
LAKE_UPSTREAM_SEGS_CSV = Path(
    "E:/Project_2025_2026/Smart_hs/raw_data/grit"
    "/GRIT_mekong_mega_reservoirs/reservoirs"
    "/gritv06_pld_lake_upstream_segments_0sqkm.csv"
)

# Reach table with lake_id annotation  (segment_id == reach_id in reach attrs)
REACHES_WITH_LAKES_CSV = Path(
    "E:/Project_2025_2026/Smart_hs/raw_data/grit"
    "/GRIT_mekong_mega_reservoirs/reaches"
    "/gritv06_reaches_mekong_basin_with_pld_lakes.csv"
)

# Output directory (will be created if absent)
SAVE_DIR = Path(
    "E:/Project_2025_2026/Smart_hs/processed_data"
    "/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/training_data_lake_based_20260319"
)

# ─── Load data ─────────────────────────────────────────────────────────────────

print("Loading reach attributes …")
reach_attrs = pd.read_csv(REACH_ATTRS_CSV)
# Index by reach_id for fast lookup
reach_attrs = reach_attrs.set_index("reach_id")

# Feature columns — everything except the index (reach_id)
FEATURE_COLS = reach_attrs.columns.tolist()  # includes 'darea', may include string cols

# Separate numeric from non-numeric (e.g. 'domain') for aggregation
NUMERIC_FEATURE_COLS = reach_attrs.select_dtypes(include=[np.number]).columns.tolist()
STRING_FEATURE_COLS  = reach_attrs.select_dtypes(exclude=[np.number]).columns.tolist()

print("Loading lake upstream-segment table …")
lake_upstream = pd.read_csv(LAKE_UPSTREAM_SEGS_CSV)

print(f"  {len(lake_upstream)} lakes, {len(reach_attrs)} reaches")

# ─── Build per-lake static attributes ──────────────────────────────────────────

records = []

for _, row in tqdm(lake_upstream.iterrows(), total=len(lake_upstream),
                   desc="Aggregating reach attrs per lake"):
    lake_id = row["lake_id"]
    segs_str = str(row["all_upstream_segments"])

    # Parse comma-separated segment IDs (segment_id == reach_id)
    seg_ids = [int(s.strip()) for s in segs_str.split(",")
               if s.strip() not in ("", "nan")]

    # Retrieve reach attributes for this lake's upstream reaches
    valid_ids = [sid for sid in seg_ids if sid in reach_attrs.index]
    if not valid_ids:
        # Should not happen given the data, but guard anyway
        records.append({"lake_id": lake_id})
        continue

    subset = reach_attrs.loc[valid_ids]  # may have duplicates if same reach listed twice

    # --- Aggregate ---
    rec = {"lake_id": lake_id}

    # darea: drainage area of the most downstream reach (= max across all upstream)
    rec["darea"] = subset["darea"].max()

    # All other numeric features: spatial mean across upstream reaches
    for col in NUMERIC_FEATURE_COLS:
        if col == "darea":
            continue
        rec[col] = subset[col].mean()

    # String features: take the mode (most common value) — e.g. 'domain'
    for col in STRING_FEATURE_COLS:
        rec[col] = subset[col].mode().iloc[0] if len(subset) > 0 else np.nan

    records.append(rec)

# Assemble DataFrame and reorder columns to match original feature order
static_df = pd.DataFrame(records)
ordered_cols = ["lake_id"] + FEATURE_COLS
static_df = static_df[ordered_cols]

print(f"\nResult shape: {static_df.shape}")
print(static_df.head(3).to_string())
print(f"\nMissing values:\n{static_df.isnull().sum()[static_df.isnull().sum() > 0]}")

# ─── Save ──────────────────────────────────────────────────────────────────────

SAVE_DIR.mkdir(parents=True, exist_ok=True)
out_path = SAVE_DIR / "lake_static_attrs.csv"
static_df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
