"""
Download ECMWF IFS ENS 0-15 day forecasts for the Mekong River Basin.

Variables: tp, 2t, 2d, sp, 10u, 10v, ssrd, strd, sf, sd, swvl1, swvl2, swvl3, swvl4
Period   : 2026-01-01 to 2026-02-28
Run      : 00 UTC only
Steps    : 0-240h (6-hourly) — 41 steps
Members  : Control forecast (cf, member 0) + perturbed forecasts (pf, configurable range)
Region   : Mekong River Basin (N=34, W=89, S=7, E=112)
Grid     : 0.1° × 0.1° (~11 km, finer than ENS native ~18 km — interpolated by MARS)
Format   : GRIB2 (native MARS format), one file per day
           Output structure:
             OUTDIR/YYYY-MM/ens_mekong_YYYY-MM-DD_cf_all.grib2
             OUTDIR/YYYY-MM/ens_mekong_YYYY-MM-DD_pf_1-50_all.grib2

Efficiency note
---------------
For ENS sfc/fc data all parameters and steps for a given date+time are stored
on the SAME MARS tape file.  This script issues ONE request per day for cf and
ONE request per day for all pf members combined (number="1/2/.../50"), with all
param codes joined by "/", minimising tape mounts.
Control (cf) and perturbed (pf) have different MARS type values and must be
retrieved in separate requests.  See ECMWF MARS efficiency guidelines:
https://confluence.ecmwf.int/x/a4AJBQ

To open in Python (filter by variable and member):
    import xarray as xr
    # Control forecast:
    ds_cf = xr.open_dataset("ens_mekong_2026-01-15_cf_all.grib2", engine="cfgrib",
                             filter_by_keys={"shortName": "tp"})
    # All perturbed members (one file contains all):
    ds_pf = xr.open_dataset("ens_mekong_2026-01-15_pf_1-50_all.grib2", engine="cfgrib",
                             filter_by_keys={"shortName": "tp"})
"""

import argparse
import calendar
from datetime import date, datetime, timezone
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
# GRID = "0.1/0.1"       # ~11 km regular lat/lon, interpolated by MARS from ENS native ~18 km

# All param codes joined for a single multi-param MARS request (MARS efficiency)
ALL_PARAMS = "/".join(VARIABLES.values())

# 6-hourly steps 0–240h (days 0–10), 41 steps total
# Accumulated vars (tp, ssrd, strd): difference consecutive steps → 6-hourly totals
STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

# Default perturbed member range (overridable via --pf-start / --pf-end)
DEFAULT_PF_START = 1
DEFAULT_PF_END   = 10

# Set to False to skip the control forecast (cf) download
DEFAULT_DOWNLOAD_CF = True

# Default output directory (overridable via --outdir)
START_YEAR  = 2023
START_MONTH = 1
END_YEAR    = 2026
END_MONTH   = 3

# Cluster path (uncomment the local path for testing on a local machine):
OUTDIR = Path("/data/ouce-grit/cenv1160/smart_hs/raw_data/ecmwf_ifs/ens")
# OUTDIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\ecmwf_ifs\ens")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = datetime.now(timezone.utc).date()


def is_future_date(d: date) -> bool:
    """Return True if the date is in the future (not yet available in MARS)."""
    return d > TODAY


def pf_member_range(pf_start: int, pf_end: int) -> range:
    """Return the range of perturbed member numbers pf_start..pf_end."""
    return range(pf_start, pf_end + 1)


def iter_days(start_year: int, start_month: int, end_year: int, end_month: int):
    """Yield (year, month, day) tuples for every calendar day in the period."""
    for year in range(start_year, end_year + 1):
        m_start = start_month if year == start_year else 1
        m_end   = end_month   if year == end_year   else 12
        for month in range(m_start, m_end + 1):
            last_day = calendar.monthrange(year, month)[1]
            for day in range(1, last_day + 1):
                yield year, month, day


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_day(server: ECMWFService, year: int, month: int, day: int,
                 outdir: Path, pf_start: int, pf_end: int,
                 download_cf: bool = True) -> bool:
    """Download all variables for one day, optionally cf and/or pf.

    Returns True if all requested downloads succeed, False if any fail.

    Two requests maximum per day: one for cf and one for all pf members combined
    (number="pf_start/.../pf_end").  All param codes are joined in each request
    to minimise tape mounts.

    Output files (inside OUTDIR/YYYY-MM/):
      ens_mekong_YYYY-MM-DD_cf_all.grib2              (if download_cf=True)
      ens_mekong_YYYY-MM-DD_pf_<start>-<end>_all.grib2

    Notes on accumulated fields
    ---------------------------
    tp, ssrd, strd are accumulated from step=0 of each model run.
    To obtain per-interval values (e.g. 6-hourly precipitation), difference
    consecutive steps:
        tp_rate = ds["tp"].diff(dim="step")
    """
    d = date(year, month, day)
    if is_future_date(d):
        print(f"[SKIP]  {d} is in the future.")
        return True

    month_dir = outdir / f"{year}-{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    date_str = f"{year}-{month:02d}-{day:02d}"
    tag      = f"{year}-{month:02d}-{day:02d}"

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
        # "grid"    : GRID,     # optional: MARS will interpolate to this grid from ENS native ~18 km
    }

    success = True

    # --- Control forecast (type=cf, member 0) ---
    if not download_cf:
        print(f"[SKIP]  {tag} control forecast (cf) — disabled by --no-cf")
    else:
        cf_file = month_dir / f"ens_mekong_{tag}_cf_all.grib2"
        if cf_file.exists():
            print(f"[SKIP]  {cf_file.name} already exists.")
        else:
            print(f"[START] {tag}  control forecast (cf)")
            try:
                server.execute({**base_request, "type": "cf"}, str(cf_file))
                print(f"[DONE]  Saved → {cf_file}")
            except Exception as e:
                print(f"[FAIL]  {tag} cf: {e}")
                success = False

    # --- Perturbed forecasts: one request for all members combined ---
    all_members = "/".join(str(m) for m in pf_member_range(pf_start, pf_end))
    pf_file = month_dir / f"ens_mekong_{tag}_pf_{pf_start}-{pf_end}_all.grib2"
    if pf_file.exists():
        print(f"[SKIP]  {pf_file.name} already exists.")
    else:
        print(f"[START] {tag}  perturbed forecasts (pf, members {pf_start}–{pf_end})")
        try:
            server.execute(
                {**base_request, "type": "pf", "number": all_members},
                str(pf_file),
            )
            print(f"[DONE]  Saved → {pf_file}")
        except Exception as e:
            print(f"[FAIL]  {tag} pf members {pf_start}–{pf_end}: {e}")
            success = False

    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ECMWF IFS ENS forecasts.")
    parser.add_argument("--start-year",  type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=START_MONTH)
    parser.add_argument("--end-year",    type=int, default=END_YEAR)
    parser.add_argument("--end-month",   type=int, default=END_MONTH)
    parser.add_argument("--pf-start",    type=int, default=DEFAULT_PF_START,
                        help=f"First perturbed member to download (default: {DEFAULT_PF_START})")
    parser.add_argument("--pf-end",      type=int, default=DEFAULT_PF_END,
                        help=f"Last perturbed member to download (default: {DEFAULT_PF_END})")
    parser.add_argument("--no-cf",       action="store_true", default=not DEFAULT_DOWNLOAD_CF,
                        help="Skip the control forecast (cf) download")
    parser.add_argument("--outdir",      type=Path, default=OUTDIR,
                        help="Output root directory (default: cluster path in script)")
    args = parser.parse_args()

    if not (1 <= args.pf_start <= args.pf_end <= 50):
        parser.error("--pf-start and --pf-end must satisfy 1 ≤ pf_start ≤ pf_end ≤ 50")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    all_days = list(iter_days(args.start_year, args.start_month, args.end_year, args.end_month))
    total_days = len(all_days)

    print(f"[INFO]  Output directory : {outdir}")
    print(f"[INFO]  Period           : {args.start_year}-{args.start_month:02d} → {args.end_year}-{args.end_month:02d}  ({total_days} days)")
    print(f"[INFO]  Variables        : {', '.join(VARIABLES.keys())} (1 combined request/member/day)")
    cf_label = "skipped (--no-cf)" if args.no_cf else "yes"
    n_pf = args.pf_end - args.pf_start + 1
    print(f"[INFO]  Members          : cf (control): {cf_label} + pf members {args.pf_start}–{args.pf_end} ({n_pf} members, 1 combined request/day)")
    print(f"[INFO]  Region (N/W/S/E) : {AREA}")
    print(f"[INFO]  Steps            : {STEPS[:40]}...")
    print("-" * 60)

    server = ECMWFService("mars")
    print("[INFO]  Connected to ECMWF MARS server.")
    print("-" * 60)

    days_requested = 0
    failed_days = []
    current_month = None
    # Iterate over all days in the period, downloading cf and pf as requested.
    for year, month, day in all_days:
        # Print month header when we reach a new month (or the first day).
        if (year, month) != current_month:
            current_month = (year, month) # Update current month
            print(f"\n[MONTH] {year}-{month:02d}") # Print month header
        days_requested += 1 # Increment requested days count
        print(f"\n[DAY]   {year}-{month:02d}-{day:02d}  ({days_requested}/{total_days})")
        # Download all variables for this day, for cf and pf members as requested.
        ok = download_day(server, year, month, day, outdir, args.pf_start, args.pf_end,
                          download_cf=not args.no_cf)
        # If any request for this day failed, add to failed_days list and print a warning.
        if not ok:
            # Record the failed day in YYYY-MM-DD format for summary reporting at the end.
            failed_days.append(f"{year}-{month:02d}-{day:02d}")
            print(f"[WARN]  {year}-{month:02d}-{day:02d} failed (one or more requests).")
        else:
            # If all requests for this day succeeded, print an OK message.
            print(f"[OK]    {year}-{month:02d}-{day:02d} complete.")

    print("\n" + "=" * 60)
    print(f"[DONE]  {days_requested} day(s) processed → {outdir}")
    if failed_days:
        print(f"[WARN]  Failed ({len(failed_days)}):")
        for f in failed_days:
            print(f"  {f}")
    else:
        print("[OK]    All days downloaded successfully.")


if __name__ == "__main__":
    main()
