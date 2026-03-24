"""
Download ECMWF IFS HRES 0-15 day forecasts for the Mekong River Basin.

Variables: tp, 2t, 2d, sp, 10u, 10v, ssrd, strd, sf, sd, swvl1, swvl2
           (2r: not in HRES archive; 2q: not at 1-hourly res — both derivable from 2t/2d)
Period   : 2023-01-01 to 2025-12-31
Run      : 00 UTC only
Steps    : 0-240h (6-hourly) — 41 steps; instantaneous vars averaged to daily, accumulated vars differenced
Region   : Mekong River Basin (N=34, W=96, S=9, E=109)
Grid     : 0.1° × 0.1° (~11 km, closest to HRES native 9 km)
Format   : GRIB2 (native MARS format), one file per variable per month
           Output structure: OUTDIR/YYYY-MM/hres_mekong_YYYY-MM_<var>.grib2
"""

import argparse
import calendar
from datetime import datetime, timezone
from pathlib import Path

from ecmwfapi import ECMWFService

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Mapping: variable name → GRIB param code
VARIABLES = {
    "tp"   : "228.128",  # total precipitation
    "2t"   : "167.128",  # 2m temperature
    "2d"   : "168.128",  # 2m dewpoint
    "sp"   : "134.128",  # surface pressure
    "10u"  : "165.128",  # 10m U wind
    "10v"  : "166.128",  # 10m V wind
    "ssrd" : "169.128",  # surface solar radiation downwards
    "strd" : "175.128",  # surface thermal radiation downwards
    "sf"   : "144.128",  # snowfall
    "sd"   : "141.128",  # snow depth (m water equivalent)
    "swvl1": "39.128",   # volumetric soil water layer 1 (0–7 cm)
    "swvl2": "40.128",   # volumetric soil water layer 2 (7–28 cm)
    "swvl3": "41.128",   # volumetric soil water layer 3 (28–100 cm)
    "swvl4": "42.128",   # volumetric soil water layer 4 (100–289 cm)
}

AREA   = "34/89/7/112"                    # N/W/S/E — full Mekong River Basin
GRID   = "0.1/0.1"                        # ~11 km regular lat/lon

# 6-hourly steps 0–240h (days 0–10), 41 steps total
# Instantaneous vars (2t, 2d, sp, 10u, 10v): average 4 steps per day → daily mean
# Accumulated vars (tp, ssrd, strd): difference consecutive steps → 6-hourly totals, then sum to daily
STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

START_YEAR = 2026
END_YEAR   = 2026

OUTDIR = Path("/data/ouce-grit/cenv1160/smart_hs/raw_data/ecmwf_ifs/hres")
# OUTDIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\ecmwf_ifs\hres")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = datetime.now(timezone.utc).date()


def is_future_month(year: int, month: int) -> bool:
    """Return True if the month is in the future (not yet available in MARS)."""
    first_day = datetime(year, month, 1).date()
    return first_day > TODAY


def build_date_str(year: int, month: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01/to/{year}-{month:02d}-{last_day}"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_variable(
    server: ECMWFService,
    year: int,
    month: int,
    var_name: str,
    param_code: str,
    month_dir: Path,
    date_str: str,
    steps: str,
) -> None:
    """
    Download one variable for one month of ECMWF IFS HRES forecast data.

    Output
    ------
    File saved to: OUTDIR/YYYY-MM/hres_mekong_YYYY-MM_<var>.grib2

    GRIB2 is the native MARS format. To open in Python:
        import xarray as xr
        ds = xr.open_dataset("hres_mekong_2023-01_tp.grib2", engine="cfgrib")

    Notes on accumulated fields
    ---------------------------
    tp, ssrd, strd are accumulated from step=0 of each model run.
    To obtain per-interval rates (e.g. 3-hourly precipitation), difference
    consecutive steps:
        tp_rate = ds["tp"].diff(dim="step")
    """
    outfile = month_dir / f"hres_mekong_{year}-{month:02d}_{var_name}.grib2"

    if outfile.exists():
        print(f"[SKIP]  {outfile.name} already exists.")
        return

    print(f"[START] {year}-{month:02d}  {var_name} ({param_code})")

    server.execute({
        "class"   : "od",       # operational data
        "expver"  : "1",
        "stream"  : "oper",     # deterministic HRES
        "type"    : "fc",       # forecast
        "date"    : date_str,
        "time"    : "00",       # 00 UTC run only
        "step"    : steps,
        "levtype" : "sfc",      # surface fields
        "param"   : param_code,
        "area"    : AREA,
        "grid"    : GRID,
    }, str(outfile))

    print(f"[DONE]  Saved → {outfile}")


def download_month(server: ECMWFService, year: int, month: int, outdir: Path) -> list[str]:
    """Download all variables for one month, each into its own file.
    Returns a list of variable names that failed."""
    if is_future_month(year, month):
        print(f"[SKIP]  {year}-{month:02d} is in the future.")
        return []

    month_dir = outdir / f"{year}-{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    date_str = build_date_str(year, month)

    failed = []
    for var_name, param_code in VARIABLES.items():
        try:
            download_variable(server, year, month, var_name, param_code, month_dir, date_str, STEPS)
        except Exception as e:
            print(f"[FAIL]  {year}-{month:02d}  {var_name}: {e}")
            failed.append(f"{year}-{month:02d}/{var_name}")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ECMWF IFS HRES forecasts.")
    parser.add_argument("--start-year",  type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year",    type=int, default=END_YEAR)
    parser.add_argument("--end-month",   type=int, default=12)
    parser.add_argument("--outdir",      type=Path, default=OUTDIR,
                        help="Output root directory (default: cluster path in script)")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    server = ECMWFService("mars")

    months_requested = 0
    all_failed = []
    for year in range(args.start_year, args.end_year + 1):
        m_start = args.start_month if year == args.start_year else 1
        m_end   = args.end_month   if year == args.end_year   else 12
        for month in range(m_start, m_end + 1):
            all_failed.extend(download_month(server, year, month, outdir))
            months_requested += 1

    print(f"\nFinished. {months_requested} month(s) processed → {outdir}")
    if all_failed:
        print(f"Failed ({len(all_failed)}):")
        for f in all_failed:
            print(f"  {f}")
    else:
        print("All variables downloaded successfully.")


if __name__ == "__main__":
    main()
