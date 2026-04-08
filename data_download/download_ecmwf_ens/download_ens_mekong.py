"""
Download ECMWF IFS ENS 0-15 day forecasts for the Mekong River Basin.

Variables: tp, 2t, 2d, sp, 10u, 10v, ssrd, strd, sf, sd, swvl1, swvl2, swvl3, swvl4
Period   : 2026-01-01 to 2026-02-28
Run      : 00 UTC only
Steps    : 0-360h (6-hourly) — 61 steps; matches ENS 15-day range
Members  : Control forecast (cf, member 0) + 50 perturbed forecasts (pf, members 1-50)
Region   : Mekong River Basin (N=34, W=89, S=7, E=112)
Grid     : 0.2° × 0.2° (~22 km, closest to ENS native ~18 km)
Format   : GRIB2 (native MARS format), one file per month per type
           Output structure:
             OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_cf_all.grib2   (control forecast)
             OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_pf_all.grib2   (perturbed forecasts 1-50)

Efficiency note
---------------
For ENS sfc/fc data all parameters and steps for a given date+time are stored
on the SAME MARS tape file.  This script issues ONE request per type (cf/pf)
per month with all param codes joined by "/", minimising tape mounts.
Control (cf) and perturbed (pf) have different MARS type values and must be
retrieved in separate requests.  See ECMWF MARS efficiency guidelines:
https://confluence.ecmwf.int/x/a4AJBQ

To open in Python (filter by variable and member):
    import xarray as xr
    # Control forecast:
    ds_cf = xr.open_dataset("ens_mekong_2026-01_cf_all.grib2", engine="cfgrib",
                             filter_by_keys={"shortName": "tp"})
    # Perturbed forecasts (one member at a time):
    ds_pf = xr.open_dataset("ens_mekong_2026-01_pf_all.grib2", engine="cfgrib",
                             filter_by_keys={"shortName": "tp", "number": 1})
"""

import argparse
import calendar
from datetime import datetime, timezone
from pathlib import Path

from ecmwfapi import ECMWFService

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Mapping: variable name → GRIB param code (same as HRES)
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

AREA = "34/89/7/112"   # N/W/S/E — full Mekong River Basin
GRID = "0.2/0.2"       # ~22 km regular lat/lon, closest to ENS native ~18 km

# All param codes joined for a single multi-param MARS request (MARS efficiency)
ALL_PARAMS = "/".join(VARIABLES.values())

# 6-hourly steps 0–360h (days 0–15), 61 steps total
# ENS range extends to 360h (vs 240h for HRES)
# Accumulated vars (tp, ssrd, strd): difference consecutive steps → 6-hourly totals
STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

# ENS perturbed member numbers (members 1–50)
PF_NUMBERS = "/".join(str(n) for n in range(1, 51))  # "1/2/.../50"

START_YEAR  = 2026
START_MONTH = 1
END_YEAR    = 2026
END_MONTH   = 2

OUTDIR = Path("/data/ouce-grit/cenv1160/smart_hs/raw_data/ecmwf_ifs/ens")
# OUTDIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\ecmwf_ifs\ens")

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
    """Download all variables for one month in two MARS requests (cf + pf).

    Returns True if both requests succeed, False if either fails.

    Two requests are needed because control (cf) and perturbed (pf) members
    have different MARS type values and cannot be combined in a single request.
    All param codes are joined in each request to minimise tape mounts.

    Output files:
      OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_cf_all.grib2   (control forecast)
      OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_pf_all.grib2   (perturbed forecasts 1-50)

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

    date_str = build_date_str(year, month)
    print(f"[INFO]  Output dir  : {month_dir}")
    print(f"[INFO]  Date range  : {date_str}")
    print(f"[INFO]  Params      : {len(VARIABLES)} variables per request")

    base_request = {
        "class"   : "od",     # operational data
        "expver"  : "1",
        "stream"  : "enfo",   # ensemble forecast
        "date"    : date_str,
        "time"    : "00",     # 00 UTC run only
        "step"    : STEPS,
        "levtype" : "sfc",    # surface fields
        "param"   : ALL_PARAMS,
        "area"    : AREA,
        "grid"    : GRID,
    }

    success = True

    # --- Control forecast (type=cf, member 0) ---
    cf_file = month_dir / f"ens_mekong_{year}-{month:02d}_cf_all.grib2"
    if cf_file.exists():
        print(f"[SKIP]  {cf_file.name} already exists.")
    else:
        print(f"[START] {year}-{month:02d}  control forecast (cf)")
        try:
            server.execute({**base_request, "type": "cf"}, str(cf_file))
            print(f"[DONE]  Saved → {cf_file}")
        except Exception as e:
            print(f"[FAIL]  {year}-{month:02d} cf: {e}")
            success = False

    # --- Perturbed forecasts (type=pf, members 1-50) ---
    pf_file = month_dir / f"ens_mekong_{year}-{month:02d}_pf_all.grib2"
    if pf_file.exists():
        print(f"[SKIP]  {pf_file.name} already exists.")
    else:
        print(f"[START] {year}-{month:02d}  perturbed forecasts (pf, members 1-50)")
        try:
            server.execute(
                {**base_request, "type": "pf", "number": PF_NUMBERS},
                str(pf_file),
            )
            print(f"[DONE]  Saved → {pf_file}")
        except Exception as e:
            print(f"[FAIL]  {year}-{month:02d} pf: {e}")
            success = False

    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ECMWF IFS ENS forecasts.")
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
    print(f"[INFO]  Variables        : {', '.join(VARIABLES.keys())} (1 combined request/type/month)")
    print(f"[INFO]  Members          : cf (control) + pf (members 1-50)")
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
                print(f"[WARN]  {year}-{month:02d} failed (one or both requests).")
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
