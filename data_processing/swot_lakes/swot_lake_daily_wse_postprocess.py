"""
SWOT Lake WSE — Post-processing: Daily aggregation
===================================================
Steps:
  1. Load outlier-flagged SWOT lake WSE data.
  2. Remove confirmed outlier observations (outlier == True).
  3. Aggregate valid sub-daily / multi-pass observations to daily scale.
  4. Discard lakes with area < AREA_THRESHOLD_SQKM.
  5. Discard lakes with fewer than OBS_COUNT_THRESHOLD daily observations.
  6. Export final daily SWOT Lake WSE CSV.

Input:
  E:/Project_2025_2026/Smart_hs/processed_data/swot/mekong_river_basin/swot/lakes
    /swot_lake_qc_all_lakes_xtrk10_60km_dark50pct_qf01_outlier_flag.csv

Output:
  E:/Project_2025_2026/Smart_hs/processed_data/swot/mekong_river_basin/swot/lakes
    /swot_lake_daily_wse_xtrk10_60km_dark50pct_qf01_daily_final.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

TIME_RANGE = '2023_12_2026_02'
AREA_THRESHOLD_SQKM = 0.1  # Minimum lake area in square kilometers to retain
OBS_COUNT_THRESHOLD = 30   # Minimum number of daily observations to retain a lake

# ── Paths ───────────────────────────────────────────────────────────────────────
OUTLIER_CSV = Path(
    r"E:\Project_2025_2026\Smart_hs\processed_data\swot\great_mekong_river_basin\lakes"
) / f"swot_lake_{TIME_RANGE}_outlier_flag.csv"
PLD_PATH = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs\prior_lake_database"
    r"\swot_prior_lake_database_great_mekong_overlap_with_grit.csv"
)

OUTPUT_DIR = Path(
    r"E:\Project_2025_2026\Smart_hs\processed_data\swot\great_mekong_river_basin\lakes_daily"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / f"swot_lake_{TIME_RANGE}_daily_wse_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}.csv"

# =============================================================================
# 1. Load outlier-flagged data
# =============================================================================
print("Loading outlier-flagged SWOT data ...")
df = pd.read_csv(OUTLIER_CSV, low_memory=False)
df["date"]    = pd.to_datetime(df["date"], errors="coerce")
df["lake_id"] = pd.to_numeric(df["lake_id"], errors="coerce").astype("Int64")
print(f"  Total rows     : {len(df):,}")
print(f"  Unique lakes   : {df['lake_id'].nunique():,}")

# =============================================================================
# 2. Remove confirmed outliers
# =============================================================================
print("Removing confirmed outliers ...")
n_outlier = df["outlier"].sum()
df_clean  = df[~df["outlier"]].copy()
print(f"  Outlier obs removed : {n_outlier:,}")
print(f"  Valid obs remaining : {len(df_clean):,}")
print(f"  Unique lakes        : {df_clean['lake_id'].nunique():,}")

# =============================================================================
# 4. Daily aggregation
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
# Discard lakes with area less than certain threshold 
# =============================================================================
print(f"Filtering lakes by minimum area (>= {AREA_THRESHOLD_SQKM} km²) ...")
pld_df = pd.read_csv(PLD_PATH, usecols=["lake_id", "poly_area"])
daily = daily.merge(pld_df[['lake_id', 'poly_area']], on='lake_id', how='left')
before_area = daily["lake_id"].nunique()
daily = daily[daily['poly_area'] >= AREA_THRESHOLD_SQKM].copy()
print(f"  Lakes before   : {before_area:,}")
print(f"  Lakes retained : {daily['lake_id'].nunique():,}  (dropped {before_area - daily['lake_id'].nunique():,})")
print(f"  Daily rows     : {len(daily):,}")
                                           
# =============================================================================
# 5. Discard lakes with fewer than 10 daily observations
# =============================================================================
print(f"Filtering lakes by minimum daily observation count (>= {OBS_COUNT_THRESHOLD} days) ...")
lake_counts = daily.groupby("lake_id")["date"].count()
valid_lakes  = lake_counts[lake_counts >= OBS_COUNT_THRESHOLD].index
before = daily["lake_id"].nunique()
daily  = daily[daily["lake_id"].isin(valid_lakes)].copy()
print(f"  Lakes before   : {before:,}")
print(f"  Lakes retained : {daily['lake_id'].nunique():,}  (dropped {before - daily['lake_id'].nunique():,})")
print(f"  Daily rows     : {len(daily):,}")

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
