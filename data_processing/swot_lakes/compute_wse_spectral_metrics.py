"""
Compute WSE regularity metrics for SWOT lakes:
  - dominant_ls_power      : peak normalised Lomb-Scargle power
  - dominant_period_days   : period (days) at the LS peak
  - spectral_snr           : peak power / mean power
  - acf_annual_lag         : ACF of the daily-interpolated WSE series at lag-365
  - flashiness             : std(diff(wse)) / std(wse)
  - period_annual_score    : 1 - |dominant_period - 365| / 365, clipped to [0, 1]
"""

from statsmodels.tsa.stattools import acf as ts_acf
from scipy.signal import lombscargle
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AREA_THRESHOLD_SQKM = 0.1
OBS_COUNT_THRESHOLD = 30

INPUT_CSV = Path(
    r"E:\Project_2025_2026\Smart_hs\processed_data\swot\great_mekong_river_basin\lakes_daily"
    rf"\swot_lake_2023_12_2026_02_daily_wse_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}.csv"
)
LAKE_AREA_CSV = Path(
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit\GRIT_mekong_mega_reservoirs"
    r"\prior_lake_database\swot_prior_lake_database_great_mekong_overlap_with_grit.csv"
)
OUTPUT_CSV = Path(r"E:\Project_2025_2026\Smart_hs\processed_data\swot\great_mekong_river_basin\lakes_daily" 
            rf"\swot_lake_2023_12_2026_06_daily_wse_spectral_metrics_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}.csv")

FILL_VALUE = -999999999999.0
MIN_OBS_SPECTRAL = 20    # minimum observations per lake
ANNUAL_LAG_DAYS  = 365   # ACF lag for annual periodicity
PERIOD_MIN_DAYS  = 20    # shortest period tested in Lomb-Scargle
PERIOD_MAX_DAYS  = 400   # longest  period tested


# ---------------------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
area_df = pd.read_csv(LAKE_AREA_CSV)
df = df.merge(area_df[["lake_id", "poly_area"]], on="lake_id", how="left")

for col in ["wse", "area_total"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df[col] <= FILL_VALUE * 0.5, col] = np.nan

df = df.dropna(subset=["wse", "area_total", "lake_id"])
df["date"] = pd.to_datetime(df["date"])
print(f"  Rows after cleaning: {len(df):,}")


# ---------------------------------------------------------------------------
# Per-lake spectral metrics
# ---------------------------------------------------------------------------
periods   = np.linspace(PERIOD_MIN_DAYS, PERIOD_MAX_DAYS, 500)
ang_freqs = 2 * np.pi / periods

records = []

for lake_id, grp in df.groupby("lake_id"):
    grp = grp.dropna(subset=["wse"]).sort_values("date")
    n = len(grp)
    if n < MIN_OBS_SPECTRAL:
        continue

    wse    = grp["wse"].values
    t_days = (grp["date"] - grp["date"].iloc[0]).dt.days.values

    # 1. ACF at annual lag (interpolated to uniform daily grid)
    t_uniform   = np.arange(0, t_days[-1] + 1)
    wse_uniform = np.interp(t_uniform, t_days, wse)
    wse_uniform -= wse_uniform.mean()
    max_lag     = min(ANNUAL_LAG_DAYS, len(wse_uniform) - 1)
    acf_vals    = ts_acf(wse_uniform, nlags=max_lag, fft=True)
    acf_annual  = acf_vals[min(ANNUAL_LAG_DAYS, max_lag)]

    # 2. Lomb-Scargle spectral metrics on original irregular times
    wse_zm               = wse - wse.mean()
    pgram                = lombscargle(t_days.astype(float), wse_zm, ang_freqs, normalize=True)
    dominant_period_days = periods[np.argmax(pgram)]
    dominant_power       = pgram.max()
    spectral_snr         = dominant_power / (pgram.mean() + 1e-12)

    # 3. Flashiness: std of first differences relative to overall std
    flashiness = np.std(np.diff(wse)) / (grp["wse"].std() + 1e-12)

    # 4. Period annual score: proximity of dominant period to 365 days
    period_annual_score = float(np.clip(1 - abs(dominant_period_days - 365) / 365, 0, None))

    records.append({
        "lake_id":              lake_id,
        "n_obs":                n,
        "acf_annual_lag":       round(acf_annual, 4),
        "flashiness":           round(flashiness, 4),
        "dominant_period_days": round(dominant_period_days, 1),
        "dominant_ls_power":    round(dominant_power, 4),
        "spectral_snr":         round(spectral_snr, 2),
        "period_annual_score":  round(period_annual_score, 4),
    })

result_df = pd.DataFrame(records).sort_values("dominant_ls_power", ascending=False)
print(f"Lakes with spectral metrics: {len(result_df)}")
print(result_df.head(10).to_string(index=False))

result_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to {OUTPUT_CSV}")
