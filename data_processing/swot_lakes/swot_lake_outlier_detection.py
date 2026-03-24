"""
SWOT Lake WSE — Outlier Detection (standalone script)
=====================================================
Methods: per-lake Z-score, IQR fence, seasonal MAD residual,
         rolling MAD, cross-variable consistency (wse_std/wse_u).
Consensus flag: >=2 independent methods.

Output directory:
  E:/Project_2025_2026/Smart_hs/processed_data/swot/mekong_river_basin/swot/lakes
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

TIME_RANGE = '2023_12_2026_02'

# ── Paths ──────────────────────────────────────────────────────────────────────
SWOT_QC_PATH = Path(
    r"E:\Project_2025_2026\Smart_hs\processed_data\swot\great_mekong_river_basin\lakes"
) / f"swot_lake_{TIME_RANGE}_qc_all_lakes_xtrk10_60km_dark50pct_qf01.csv"
# LAKE_GRAPH_PATH = Path(
#     r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
#     r"\GRIT_mekong_mega_reservoirs\reservoirs"
#     r"\gritv06_pld_lake_graph_0sqkm.csv"
# )
OUTPUT_DIR = Path(
    r"E:\Project_2025_2026\Smart_hs\processed_data\swot\great_mekong_river_basin\lakes"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV         = OUTPUT_DIR / f"swot_lake_{TIME_RANGE}_qc_all_lakes_xtrk10_60km_dark50pct_qf01_outlier_flag.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / f"swot_lake_{TIME_RANGE}_qc_all_lakes_xtrk10_60km_dark50pct_qf01_outlier_flag_summary.csv"

# ── Detection thresholds ───────────────────────────────────────────────────────
Z_THRESH        = 3.0   # |z| > Z_THRESH -> per-lake Z-score outlier
IQR_MULT        = 3.0   # > median +/- IQR_MULT * IQR -> IQR fence outlier
SEASONAL_MAD_THRESH = 5.0 # deviation > SEASONAL_MAD_THRESH * seasonal MAD
ROLL_MAD_THRESH = 5.0   # deviation > ROLL_MAD_THRESH * rolling MAD
ROLL_WINDOW     = 6     # rolling window (observations per lake, date-sorted)

# ── SWOT fill sentinels ────────────────────────────────────────────────────────
FILL_VALUE       = -999_999_999_999.0
FILL_INT         = -99_999_999
TERMINAL_NODE_ID = -1

# =============================================================================
# 1. Load & prepare data
# =============================================================================
print("Loading lake graph ...")
# lake_graph = pd.read_csv(LAKE_GRAPH_PATH)
# graph_lake_ids = set(
#     lake_graph.loc[lake_graph["lake_id"] != TERMINAL_NODE_ID, "lake_id"]
#     .dropna().astype("int64")
# )

print("Loading SWOT QC data ...")
df_raw = pd.read_csv(SWOT_QC_PATH, low_memory=False)
df = df_raw.copy()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

float_fill_cols = [
    "wse", "wse_u", "wse_r_u", "wse_std",
    "area_total", "area_tot_u", "area_detct", "area_det_u",
    "dark_frac", "xtrk_dist", "geoid_hght",
    "p_ref_wse", "p_ref_area", "p_storage",
    "dry_trop_c", "wet_trop_c", "iono_c", "xovr_cal_c",
]
for col in float_fill_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(df[col] > FILL_VALUE * 0.999)

for col in ["p_lon", "p_lat"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(df[col] != FILL_INT)

df["lake_id"] = pd.to_numeric(df["lake_id"], errors="coerce").astype("Int64")
if "lake_name" in df.columns:
    df["lake_name"] = df["lake_name"].replace("no_data", pd.NA)

df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter   # 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

# Keep only graph lakes
# df = df[df["lake_id"].isin(graph_lake_ids)].copy()
df = df.sort_values(["lake_id", "date"]).reset_index(drop=True)

print(f"Rows        : {len(df):,}")
print(f"Unique lakes: {df['lake_id'].nunique():,}")
print(f"Date range  : {df['date'].min().date()} - {df['date'].max().date()}")

# =============================================================================
# 2. Per-lake Z-score
# =============================================================================
print("[1/4] Per-lake Z-score ...")
lake_stats = (
    df.dropna(subset=["wse"])
    .groupby("lake_id")["wse"]
    .agg(lake_mean="mean", lake_std="std")
    .reset_index()
)
df = df.merge(lake_stats, on="lake_id", how="left")
df["z_lake"] = (df["wse"] - df["lake_mean"]) / df["lake_std"].replace(0, np.nan)
df["flag_z"] = df["z_lake"].abs() > Z_THRESH

n_z = df["flag_z"].sum()
print(f"  Flagged obs    : {n_z:,}  ({n_z/len(df)*100:.2f}%)")
print(f"  Lakes affected : {df.loc[df['flag_z'], 'lake_id'].nunique():,}")

# =============================================================================
# 3. Per-lake IQR fence
# =============================================================================
print("[2/4] Per-lake IQR fence ...")
iqr_df = (
    df.dropna(subset=["wse"])
    .groupby("lake_id")["wse"]
    .agg(
        iqr_lo=lambda x: x.quantile(0.25) - IQR_MULT * (x.quantile(0.75) - x.quantile(0.25)),
        iqr_hi=lambda x: x.quantile(0.75) + IQR_MULT * (x.quantile(0.75) - x.quantile(0.25)),
    )
    .reset_index()
)
df = df.merge(iqr_df, on="lake_id", how="left")
df["flag_iqr"] = (df["wse"] < df["iqr_lo"]) | (df["wse"] > df["iqr_hi"])

n_iqr = df["flag_iqr"].sum()
print(f"  Flagged obs    : {n_iqr:,}  ({n_iqr/len(df)*100:.2f}%)")
print(f"  Lakes affected : {df.loc[df['flag_iqr'], 'lake_id'].nunique():,}")
print(f"  Both Z & IQR   : {(df['flag_z'] & df['flag_iqr']).sum():,}"
      f"  |  Z only: {(df['flag_z'] & ~df['flag_iqr']).sum():,}"
      f"  |  IQR only: {(~df['flag_z'] & df['flag_iqr']).sum():,}")

# =============================================================================
# 4. Temporal anomaly detection
# =============================================================================
print("[3/4] Temporal anomaly detection ...")

# ── 4a. Seasonal (quarterly) residual ─────────────────────────────────────────
quarterly_med = (
    df.dropna(subset=["wse"])
    .groupby(["lake_id", "quarter"])["wse"]
    .median()
    .reset_index(name="quarterly_med_wse")
)
df = df.merge(quarterly_med, on=["lake_id", "quarter"], how="left")
df["wse_seasonal_resid"] = df["wse"] - df["quarterly_med_wse"]

seasonal_mad = (
    df.dropna(subset=["wse_seasonal_resid"])
    .groupby("lake_id")["wse_seasonal_resid"]
    .apply(lambda x: np.median(np.abs(x - np.median(x))))
    .reset_index(name="seasonal_mad")
)
df = df.merge(seasonal_mad, on="lake_id", how="left")
df["flag_seasonal"] = (
    df["wse_seasonal_resid"].abs()
    > SEASONAL_MAD_THRESH * df["seasonal_mad"].replace(0, np.nan)
)

n_seas = df["flag_seasonal"].sum()
print(f"  Seasonal MAD flags : {n_seas:,}  ({n_seas/len(df)*100:.2f}%)")

# ── 4b. Rolling MAD ───────────────────────────────────────────────────────────
def rolling_mad_flag(grp, window=ROLL_WINDOW, thresh=ROLL_MAD_THRESH):
    grp = grp.sort_values("date")
    wse_s    = grp["wse"]
    roll_med = wse_s.rolling(window=window, center=True, min_periods=3).median()
    roll_mad = (wse_s - roll_med).abs().rolling(window=window, center=True, min_periods=3).median()
    return (wse_s - roll_med).abs() > thresh * roll_mad.replace(0, np.nan)

flag_roll = (
    df.dropna(subset=["wse"])
    .groupby("lake_id", group_keys=False)
    .apply(rolling_mad_flag)
)
df["flag_rolling"] = False
df.loc[flag_roll.index, "flag_rolling"] = flag_roll.fillna(False).values

n_roll = df["flag_rolling"].sum()
print(f"  Rolling MAD flags  : {n_roll:,}  ({n_roll/len(df)*100:.2f}%)")

# =============================================================================
# 5. Consensus flag (>=2 methods)
# =============================================================================
print("\nBuilding consensus flag ...")

FLAG_COLS = ["flag_z", "flag_iqr", "flag_seasonal", "flag_rolling"]
FLAG_LABELS = {
    "flag_z":        "Z-score",
    "flag_iqr":      "IQR fence",
    "flag_seasonal": "Seasonal MAD",
    "flag_rolling":  "Rolling MAD",
}

for col in FLAG_COLS:
    if col not in df.columns:
        df[col] = False
    df[col] = df[col].fillna(False)

df["n_flags"] = df[FLAG_COLS].sum(axis=1).astype(int)
df["outlier"] = df["n_flags"] >= 2

n_confirmed  = df["outlier"].sum()
n_conf_lakes = df.loc[df["outlier"], "lake_id"].nunique()

print("=" * 55)
print("CONSENSUS OUTLIER SUMMARY")
print("=" * 55)
for col in FLAG_COLS:
    n = df[col].sum()
    print(f"  {FLAG_LABELS[col]:<20}: {n:>5,}  ({n/len(df)*100:.2f}%)")
print("-" * 55)
for k in range(len(FLAG_COLS) + 1):
    n = (df["n_flags"] == k).sum()
    print(f"  Flagged by {k} method(s) : {n:>5,}  ({n/len(df)*100:.2f}%)")
print("-" * 55)
print(f"  Confirmed outliers (>=2) : {n_confirmed:>5,}  ({n_confirmed/len(df)*100:.2f}%)")
print(f"  Lakes with >=1 outlier   : {n_conf_lakes:>5,}")
print("=" * 55)

# =============================================================================
# 7. Export results
# =============================================================================
print("\nExporting results ...")

# ── 7a. Observation-level CSV ──────────────────────────────────────────────────
export_cols = [
    "lake_id", "date", "wse", "wse_u", "wse_std",
    "area_total", "dark_frac", "xtrk_dist", "quality_f",
    "p_lon", "p_lat", "lake_name",
    "z_lake", "wse_seasonal_resid",
    "flag_z", "flag_iqr", "flag_seasonal", "flag_rolling",
    "n_flags", "outlier",
]
export_cols = [c for c in export_cols if c in df.columns]
df_export   = df[export_cols].copy()
df_export["date"] = df_export["date"].dt.date

df_export.to_csv(OUTPUT_CSV, index=False)
print(f"  Observation flags -> {OUTPUT_CSV}")
print(f"  Rows: {len(df_export):,}  |  confirmed outliers: {df_export['outlier'].sum():,}")

# ── 7b. Per-lake summary CSV ───────────────────────────────────────────────────
per_lake_summary = (
    df.groupby("lake_id")
    .agg(
        lake_name       =("lake_name",     "first"),
        n_obs           =("wse",           "count"),
        n_outlier       =("outlier",       "sum"),
        n_flag_z        =("flag_z",        "sum"),
        n_flag_iqr      =("flag_iqr",      "sum"),
        n_flag_seasonal =("flag_seasonal", "sum"),
        n_flag_rolling  =("flag_rolling",  "sum"),
        wse_mean        =("wse",           "mean"),
        wse_std         =("wse",           "std"),
        wse_median      =("wse",           "median"),
        wse_min         =("wse",           "min"),
        wse_max         =("wse",           "max"),
        lon             =("p_lon",         "first"),
        lat             =("p_lat",         "first"),
        date_start      =("date",          "min"),
        date_end        =("date",          "max"),
    )
    .reset_index()
)
per_lake_summary["outlier_pct"] = (
    per_lake_summary["n_outlier"] / per_lake_summary["n_obs"].replace(0, np.nan) * 100
).round(2)
per_lake_summary = per_lake_summary.sort_values("n_outlier", ascending=False)
per_lake_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
print(f"  Per-lake summary  -> {OUTPUT_SUMMARY_CSV}")
print(f"  Lakes: {len(per_lake_summary):,}")

print("\nDone.")
