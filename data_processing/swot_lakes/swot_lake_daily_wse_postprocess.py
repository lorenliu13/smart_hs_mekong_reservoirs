"""
SWOT Lake WSE — Post-processing: Lake-graph filter + Daily aggregation
======================================================================
Steps:
  1. Load outlier-flagged SWOT lake WSE data.
  2. Keep only lakes present in the GRIT lake-graph (excl. terminal node -1).
  3. Remove confirmed outlier observations (outlier == True).
  4. Aggregate valid sub-daily / multi-pass observations to daily scale.
  5. Export final daily SWOT Lake WSE CSV.

Input:
  E:/Project_2025_2026/Smart_hs/processed_data/swot/mekong_river_basin/swot/lakes
    /swot_lake_qc_all_lakes_xtrk10_60km_dark50pct_qf01_outlier_flag.csv

Lake graph:
  E:/Project_2025_2026/Smart_hs/raw_data/grit/GRIT_mekong_mega_reservoirs
    /reservoirs/gritv06_pld_lake_graph_0sqkm.csv

Output:
  E:/Project_2025_2026/Smart_hs/processed_data/swot/mekong_river_basin/swot/lakes
    /swot_lake_daily_wse_xtrk10_60km_dark50pct_qf01_graph_lakes.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────────
OUTLIER_CSV = Path(
    r"E:\Project_2025_2026\Smart_hs\processed_data\swot\mekong_river_basin\swot\lakes"
    r"\swot_lake_qc_all_lakes_xtrk10_60km_dark50pct_qf01_outlier_flag.csv"
)
LAKE_GRAPH_PATH = Path(
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    r"\GRIT_mekong_mega_reservoirs\reservoirs"
    r"\gritv06_pld_lake_graph_0sqkm.csv"
)
OUTPUT_DIR = Path(
    r"E:\Project_2025_2026\Smart_hs\processed_data\swot\mekong_river_basin\swot\lakes_daily"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "swot_lake_daily_wse_xtrk10_60km_dark50pct_qf01_lake_graph_0sqkm.csv"

TERMINAL_NODE_ID = -1

# =============================================================================
# 1. Load lake graph — extract valid lake IDs
# =============================================================================
print("Loading lake graph ...")
lake_graph = pd.read_csv(LAKE_GRAPH_PATH)
graph_lake_ids = set(
    lake_graph.loc[lake_graph["lake_id"] != TERMINAL_NODE_ID, "lake_id"]
    .dropna()
    .astype("int64")
)
print(f"  Graph lakes (excl. terminal): {len(graph_lake_ids):,}")

# =============================================================================
# 2. Load outlier-flagged data
# =============================================================================
print("Loading outlier-flagged SWOT data ...")
df = pd.read_csv(OUTLIER_CSV, low_memory=False)
df["date"]    = pd.to_datetime(df["date"], errors="coerce")
df["lake_id"] = pd.to_numeric(df["lake_id"], errors="coerce").astype("Int64")
print(f"  Total rows     : {len(df):,}")
print(f"  Unique lakes   : {df['lake_id'].nunique():,}")

# =============================================================================
# 3. Filter to lake-graph lakes only
# =============================================================================
print("Filtering to lake-graph lakes ...")
before = len(df)
df = df[df["lake_id"].isin(graph_lake_ids)].copy()
after = len(df)
print(f"  Rows kept      : {after:,}  (dropped {before - after:,})")
print(f"  Unique lakes   : {df['lake_id'].nunique():,}")

# =============================================================================
# 4. Remove confirmed outliers
# =============================================================================
print("Removing confirmed outliers ...")
n_outlier = df["outlier"].sum()
df_clean  = df[~df["outlier"]].copy()
print(f"  Outlier obs removed : {n_outlier:,}")
print(f"  Valid obs remaining : {len(df_clean):,}")
print(f"  Unique lakes        : {df_clean['lake_id'].nunique():,}")

# =============================================================================
# 5. Daily aggregation
# =============================================================================
print("Aggregating to daily scale ...")

# WSE uncertainty combination: propagate as RSS for averaged obs, then /sqrt(n)
def wse_u_combined(u_series):
    """Combined uncertainty for the daily mean: RSS / sqrt(n)."""
    u = u_series.dropna()
    if len(u) == 0:
        return np.nan
    return np.sqrt((u ** 2).sum()) / len(u)

daily = (
    df_clean.groupby(["lake_id", "date"])
    .agg(
        wse            =("wse",        "mean"),
        wse_u          =("wse_u",      wse_u_combined),
        wse_std        =("wse_std",    "mean"),       # mean of per-pass wse_std
        area_total     =("area_total", "mean"),
        dark_frac      =("dark_frac",  "mean"),
        xtrk_dist      =("xtrk_dist",  "mean"),
        quality_f      =("quality_f",  "max"),        # conservative: take worst QF
        p_lon          =("p_lon",      "first"),
        p_lat          =("p_lat",      "first"),
        lake_name      =("lake_name",  "first"),
        n_passes       =("wse",        "count"),      # number of passes merged
    )
    .reset_index()
)

# Sort
daily = daily.sort_values(["lake_id", "date"]).reset_index(drop=True)

print(f"  Daily rows     : {len(daily):,}")
print(f"  Unique lakes   : {daily['lake_id'].nunique():,}")
print(f"  Date range     : {daily['date'].min().date()} – {daily['date'].max().date()}")
print(f"  Multi-pass days: {(daily['n_passes'] > 1).sum():,}  "
      f"(max {daily['n_passes'].max()} passes/day)")

# =============================================================================
# 6. Export
# =============================================================================
print(f"\nExporting -> {OUTPUT_CSV}")

col_order = [
    "lake_id", "date", "lake_name",
    "wse", "wse_u", "wse_std",
    "area_total", "dark_frac", "xtrk_dist",
    "quality_f", "p_lon", "p_lat",
    "n_passes",
]
col_order = [c for c in col_order if c in daily.columns]
daily[col_order].to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d")

print(f"  Rows   : {len(daily):,}")
print(f"  Columns: {list(col_order)}")
print("\nDone.")
