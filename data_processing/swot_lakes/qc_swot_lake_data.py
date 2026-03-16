"""
qc_swot_lake_data.py
====================
Quality-control the raw merged SWOT lake observations for the Mekong River Basin.

QC pipeline
-----------
Step 1  Replace SWOT fill-value sentinels with NaN
Step 2  Keep quality_f ∈ {0, 1}   (good and suspect; drop bad=2, degraded=3)
Step 3  10 000 ≤ |xtrk_dist| ≤ 60 000 m   (remove near-nadir and far-swath noise)
Step 4  dark_frac ≤ 0.5   (high dark fraction → unreliable WSE)

The GRIT lake graph filter is intentionally omitted — all observed lakes are kept.

Output
------
E:\\Project_2025_2026\\Smart_hs\\processed_data\\swot\\mekong_river_basin\\swot\\lakes\\
    swot_lake_qc_all_lakes_xtrk10_60km_dark50pct_qf01.csv
    swot_lake_qc_all_lakes_xtrk10_60km_dark50pct_qf01_per_lake_summary.csv
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH = (
    r"E:\Project_2025_2026\Smart_hs\processed_data\swot"
    r"\mekong_river_basin\swot\lakes\full_swot_lake_df_2023_2025.csv"
)

OUTPUT_DIR = (
    r"E:\Project_2025_2026\Smart_hs\processed_data\swot"
    r"\mekong_river_basin\swot\lakes"
)

# QC thresholds
QUALITY_FLAGS_KEEP = [0, 1]   # 0 = good, 1 = suspect
XTRK_DIST_MIN      = 10_000   # m  minimum |cross-track distance|
XTRK_DIST_MAX      = 60_000   # m  maximum |cross-track distance|
DARK_FRAC_MAX      = 0.5      # fraction [0–1]

# SWOT fill-value sentinels
FILL_VALUE = -999_999_999_999.0
FILL_INT   = -99_999_999

# ── Helpers ────────────────────────────────────────────────────────────────────
def _log(step: str, n_before: int, n_after: int) -> None:
    removed = n_before - n_after
    pct     = removed / n_before * 100 if n_before else 0.0
    print(f"  [{step}] {n_after:>10,} rows kept   ({removed:>8,} removed, {pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load
# ══════════════════════════════════════════════════════════════════════════════
print(f"Loading  : {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

df["date"] = pd.to_datetime(df["date"])

# ══════════════════════════════════════════════════════════════════════════════
# QC Step 1 — Replace fill-value sentinels with NaN
# ══════════════════════════════════════════════════════════════════════════════
print("QC step 1 — Replace SWOT fill-value sentinels with NaN")

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

if "lake_name" in df.columns:
    df["lake_name"] = df["lake_name"].replace("no_data", pd.NA)

print(f"  After replacement : {len(df):,} rows  (no rows removed — NaN substituted)")
print(f"    WSE NaN count   : {df['wse'].isna().sum():,}")
print(f"    xtrk_dist NaN   : {df['xtrk_dist'].isna().sum():,}")
print(f"    dark_frac NaN   : {df['dark_frac'].isna().sum():,}\n")

# ══════════════════════════════════════════════════════════════════════════════
# QC Step 2 — quality_f ∈ {0, 1}
# ══════════════════════════════════════════════════════════════════════════════
print("QC step 2 — Filter by quality flag  (keep 0=good, 1=suspect)")
print("  Distribution before filter:")
n_total = len(df)
for k, v in df["quality_f"].value_counts(dropna=False).sort_index().items():
    label = {0: "good", 1: "suspect", 2: "bad", 3: "degraded"}.get(k, str(k))
    print(f"    quality_f = {k:>3}  ({label:<10}) : {v:>8,}  ({v / n_total * 100:.1f}%)")

n_before = len(df)
df = df[df["quality_f"].isin(QUALITY_FLAGS_KEEP)].copy()
_log("quality_f", n_before, len(df))
print()

# ══════════════════════════════════════════════════════════════════════════════
# QC Step 3 — cross-track distance  10 000 ≤ |xtrk_dist| ≤ 60 000 m
# ══════════════════════════════════════════════════════════════════════════════
print("QC step 3 — Filter by |xtrk_dist|  "
      f"({XTRK_DIST_MIN / 1000:.0f}–{XTRK_DIST_MAX / 1000:.0f} km)")
abs_xtrk = df["xtrk_dist"].abs()
print(f"  NaN          : {abs_xtrk.isna().sum():,}")
print(f"  < {XTRK_DIST_MIN / 1000:.0f} km      : {(abs_xtrk < XTRK_DIST_MIN).sum():,}")
print(f"  {XTRK_DIST_MIN / 1000:.0f}–{XTRK_DIST_MAX / 1000:.0f} km     : "
      f"{((abs_xtrk >= XTRK_DIST_MIN) & (abs_xtrk <= XTRK_DIST_MAX)).sum():,}")
print(f"  > {XTRK_DIST_MAX / 1000:.0f} km     : {(abs_xtrk > XTRK_DIST_MAX).sum():,}")

n_before = len(df)
mask_xtrk = (
    df["xtrk_dist"].notna()
    & (df["xtrk_dist"].abs() >= XTRK_DIST_MIN)
    & (df["xtrk_dist"].abs() <= XTRK_DIST_MAX)
)
df = df[mask_xtrk].copy()
_log("|xtrk_dist|", n_before, len(df))
print()

# ══════════════════════════════════════════════════════════════════════════════
# QC Step 4 — dark fraction ≤ 0.5
# ══════════════════════════════════════════════════════════════════════════════
print(f"QC step 4 — Filter by dark_frac  (≤ {DARK_FRAC_MAX})")
dark_valid = df["dark_frac"].dropna()
print(f"  NaN              : {df['dark_frac'].isna().sum():,}")
print(f"  dark_frac > {DARK_FRAC_MAX} : {(dark_valid > DARK_FRAC_MAX).sum():,}")
print(f"  dark_frac ≤ {DARK_FRAC_MAX} : {(dark_valid <= DARK_FRAC_MAX).sum():,}")
print(f"  Mean / Median    : {dark_valid.mean():.3f} / {dark_valid.median():.3f}")

n_before = len(df)
# Keep NaN dark_frac (unknown) and rows within threshold
mask_dark = df["dark_frac"].isna() | (df["dark_frac"] <= DARK_FRAC_MAX)
df = df[mask_dark].copy()
_log("dark_frac", n_before, len(df))
print("  Note: rows with NaN dark_frac are kept\n")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
total_raw = n_total
total_qc  = len(df)

print("=" * 60)
print("QC SUMMARY")
print("=" * 60)
print(f"  Raw observations          : {total_raw:>10,}")
print(f"  Final observations        : {total_qc:>10,}  ({total_qc / total_raw * 100:.1f}% of raw)")
print(f"  Final unique lake IDs     : {df['lake_id'].nunique():>10,}")
print(f"  Date range                : {df['date'].min().date()} → {df['date'].max().date()}")
print("=" * 60)

print("\nKey variable statistics (QC-passed data):")
stats_cols = {
    "wse"       : "WSE (m)",
    "area_total": "Area total (km²)",
    "wse_u"     : "WSE uncertainty (m)",
    "xtrk_dist" : "Cross-track dist (m)",
    "dark_frac" : "Dark fraction",
}
for col, label in stats_cols.items():
    s = df[col].dropna()
    if len(s):
        print(
            f"  {label:<24}  n={len(s):>8,}  "
            f"mean={s.mean():>10.3f}  std={s.std():>9.3f}  "
            f"min={s.min():>10.3f}  max={s.max():>10.3f}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# Per-lake summary
# ══════════════════════════════════════════════════════════════════════════════
per_lake = (
    df.groupby("lake_id")
    .agg(
        n_obs          = ("lake_id",    "count"),
        n_good         = ("quality_f",  lambda x: (x == 0).sum()),
        n_suspect      = ("quality_f",  lambda x: (x == 1).sum()),
        lake_name      = ("lake_name",  "first"),
        lon            = ("p_lon",      "first"),
        lat            = ("p_lat",      "first"),
        mean_wse       = ("wse",        "mean"),
        std_wse        = ("wse",        "std"),
        mean_area      = ("area_total", "mean"),
        mean_dark_frac = ("dark_frac",  "mean"),
        date_first     = ("date",       "min"),
        date_last      = ("date",       "max"),
    )
    .sort_values("n_obs", ascending=False)
    .reset_index()
    .round(3)
)

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)

out_obs = os.path.join(
    OUTPUT_DIR,
    "swot_lake_qc_all_lakes_xtrk10_60km_dark50pct_qf01.csv",
)
df.to_csv(out_obs, index=False)
print(f"\nSaved observations  : {out_obs}")
print(f"  Rows  : {len(df):,}")

out_summary = os.path.join(
    OUTPUT_DIR,
    "swot_lake_qc_all_lakes_xtrk10_60km_dark50pct_qf01_per_lake_summary.csv",
)
per_lake.to_csv(out_summary, index=False)
print(f"Saved lake summary  : {out_summary}")
print(f"  Lakes : {len(per_lake):,}")
