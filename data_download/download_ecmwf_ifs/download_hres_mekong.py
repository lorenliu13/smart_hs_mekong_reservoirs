"""
Download ECMWF IFS HRES 0-15 day forecasts for the Mekong River Basin.

Variables: tp, 2t, 2d, sp, 10u, 10v, ssrd, strd, sf, sd, swvl1, swvl2
           (2r: not in HRES archive; 2q: not at 1-hourly res — both derivable from 2t/2d)
Period   : 2023-01-01 to 2025-12-31
Run      : 00 UTC only
Steps    : 0-240h (6-hourly) — 41 steps; instantaneous vars averaged to daily, accumulated vars differenced
Region   : Mekong River Basin (N=34, W=96, S=9, E=109)
Grid     : 0.1° × 0.1° (~11 km, closest to HRES native 9 km)
Format   : GRIB2 (native MARS format), one combined file per month (all variables)
           Output structure: OUTDIR/YYYY-MM/hres_mekong_YYYY-MM_all.grib2

Efficiency note
---------------
For HRES sfc/fc data all parameters, steps, and levels for a given date+time
are stored on the SAME MARS tape file.  Iterating over parameters in separate
requests forces MARS to mount the same tapes repeatedly.  This script therefore
issues ONE request per month with all param codes joined by "/", reducing tape
mounts from 14× to 1× per month.  See ECMWF MARS efficiency guidelines:
https://confluence.ecmwf.int/x/a4AJBQ

To open in Python (filter by variable):
    import xarray as xr
    ds = xr.open_dataset("hres_mekong_2023-01_all.grib2", engine="cfgrib",
                         filter_by_keys={"shortName": "tp"})
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

# All param codes joined for a single multi-param MARS request (MARS efficiency:
# param is an inner-loop keyword — all params for a date+time share one tape file)
ALL_PARAMS = "/".join(VARIABLES.values())

# 6-hourly steps 0–240h (days 0–10), 41 steps total
# Instantaneous vars (2t, 2d, sp, 10u, 10v): average 4 steps per day → daily mean
# Accumulated vars (tp, ssrd, strd): difference consecutive steps → 6-hourly totals, then sum to daily
STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

START_YEAR = 2026
START_MONTH = 1
END_YEAR   = 2026
END_MONTH = 2

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

def download_month(server: ECMWFService, year: int, month: int, outdir: Path) -> bool:
    """Download all variables for one month in a single MARS request.

    Returns True on success, False on failure.

    A single request with all param codes retrieves everything from the same
    tape file in one pass (MARS efficiency: param is an inner-loop keyword).
    Output: OUTDIR/YYYY-MM/hres_mekong_YYYY-MM_all.grib2

    Notes on accumulated fields
    ---------------------------
    tp, ssrd, strd are accumulated from step=0 of each model run.
    To obtain per-interval values (e.g. 6-hourly precipitation), difference
    consecutive steps:
        tp_rate = ds["tp"].diff(dim="step")
    """
    if is_future_month(year, month):
        print(f"[SKIP]  {year}-{month:02d} is in the future.")
        return True

    month_dir = outdir / f"{year}-{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    outfile = month_dir / f"hres_mekong_{year}-{month:02d}_all.grib2"
    if outfile.exists():
        print(f"[SKIP]  {outfile.name} already exists.")
        return True

    date_str = build_date_str(year, month)
    print(f"[INFO]  Output dir  : {month_dir}")
    print(f"[INFO]  Date range  : {date_str}")
    print(f"[INFO]  Params      : {len(VARIABLES)} variables in one request")
    print(f"[START] {year}-{month:02d}  all variables")

    try:
        server.execute({
            "class"   : "od",       # operational data
            "expver"  : "1",
            "stream"  : "oper",     # deterministic HRES
            "type"    : "fc",       # forecast
            "date"    : date_str,
            "time"    : "00",       # 00 UTC run only
            "step"    : STEPS,
            "levtype" : "sfc",      # surface fields
            "param"   : ALL_PARAMS, # all variables in one request (MARS efficiency)
            "area"    : AREA,
            "grid"    : GRID,
        }, str(outfile))
    except Exception as e:
        print(f"[FAIL]  {year}-{month:02d}: {e}")
        return False

    print(f"[DONE]  Saved → {outfile}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ECMWF IFS HRES forecasts.")
    parser.add_argument("--start-year",  type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=START_MONTH)
    parser.add_argument("--end-year",    type=int, default=END_YEAR)
    parser.add_argument("--end-month",   type=int, default=END_MONTH)
    parser.add_argument("--outdir",      type=Path, default=OUTDIR,
                        help="Output root directory (default: cluster path in script)")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO]  Output directory : {outdir}")
    print(f"[INFO]  Period           : {args.start_year}-{args.start_month:02d} → {args.end_year}-{args.end_month:02d}")
    print(f"[INFO]  Variables        : {', '.join(VARIABLES.keys())} (1 combined request/month)")
    print(f"[INFO]  Region (N/W/S/E) : {AREA}")
    print(f"[INFO]  Grid             : {GRID}")
    print(f"[INFO]  Steps            : {STEPS[:40]}...")
    print("-" * 60)

    server = ECMWFService("mars")
    print("[INFO]  Connected to ECMWF MARS server.")
    print("-" * 60)

    # Count total months for progress tracking
    total_months = sum(
        len(range(
            args.start_month if y == args.start_year else 1,
            (args.end_month if y == args.end_year else 12) + 1
        ))
        for y in range(args.start_year, args.end_year + 1)
    )

    months_requested = 0
    failed_months = []
    for year in range(args.start_year, args.end_year + 1):
        m_start = args.start_month if year == args.start_year else 1
        m_end   = args.end_month   if year == args.end_year   else 12
        for month in range(m_start, m_end + 1):
            months_requested += 1
            print(f"\n[MONTH] {year}-{month:02d}  ({months_requested}/{total_months})")
            ok = download_month(server, year, month, outdir)
            if not ok:
                failed_months.append(f"{year}-{month:02d}")
                print(f"[WARN]  {year}-{month:02d} failed.")
            else:
                print(f"[OK]    {year}-{month:02d} complete.")

    print("\n" + "=" * 60)
    print(f"[DONE]  {months_requested} month(s) processed → {outdir}")
    if failed_months:
        print(f"[WARN]  Failed ({len(failed_months)}):")
        for f in failed_months:
            print(f"  {f}")
    else:
        print("[OK]    All months downloaded successfully.")


if __name__ == "__main__":
    main()
