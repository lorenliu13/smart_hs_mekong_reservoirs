"""
Download ECMWF IFS ENS 0-15 day forecasts for the Mekong River Basin.

Variables: tp, 2t, 2d, sp, 10u, 10v, ssrd, strd, sf, sd, swvl1, swvl2, swvl3, swvl4
Period   : configurable (default 2023-12 to 2026-03)
Run      : 00 UTC only
Steps    : 0-240h (6-hourly) — 41 steps
Members  : Control forecast (cf, member 0) + perturbed forecasts (pf, configurable range)
Region   : Mekong River Basin (N=34, W=89, S=7, E=112)
Grid     : 0.1° × 0.1° (~11 km, finer than ENS native ~18 km — interpolated by MARS)
Format   : GRIB2 (native MARS format), one file per month

Output structure:
  OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_cf_all.grib2
  OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_pf_<start>-<end>_all.grib2

Efficiency note
---------------
Each month is submitted as a SINGLE MARS request using the date-range syntax
"YYYY-MM-01/to/YYYY-MM-DD", which covers all days in the month.  This yields
N_months × 2 requests (cf + pf) instead of N_days × 2, dramatically reducing
queue overhead and tape mounts.  See ECMWF MARS efficiency guidelines:
https://confluence.ecmwf.int/x/a4AJBQ

To open in Python:
    import xarray as xr
    ds = xr.open_dataset("ens_mekong_2026-01_cf_all.grib2", engine="cfgrib",
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

VARIABLES = {
    "tp"   : "228.128",
    "2t"   : "167.128",
    "2d"   : "168.128",
    "sp"   : "134.128",
    "10u"  : "165.128",
    "10v"  : "166.128",
    "ssrd" : "169.128",
    "strd" : "175.128",
    "sf"   : "144.128",
    "sd"   : "141.128",
    "swvl1": "39.128",
    "swvl2": "40.128",
    "swvl3": "41.128",
    "swvl4": "42.128",
}

AREA = "34/89/7/112"   # N/W/S/E — full Mekong River Basin
# GRID = "0.1/0.1"

ALL_PARAMS = "/".join(VARIABLES.values())

# 6-hourly steps 0–240h
STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

DEFAULT_PF_START   = 1
DEFAULT_PF_END     = 10
DEFAULT_DOWNLOAD_CF = True

START_YEAR  = 2023
START_MONTH = 12
END_YEAR    = 2026
END_MONTH   = 3

OUTDIR = Path("/data/ouce-grit/cenv1160/smart_hs/raw_data/ecmwf_ifs/ens")
# OUTDIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\ecmwf_ifs\ens")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = datetime.now(timezone.utc).date()


def is_future_month(year: int, month: int) -> bool:
    """Return True if the entire month is still in the future."""
    return date(year, month, 1) > TODAY


def last_available_day(year: int, month: int) -> int:
    """Return the last day in the month that is not in the future."""
    last_day = calendar.monthrange(year, month)[1]
    if date(year, month, 1) > TODAY:
        return 0  # entire month is future
    # Cap at today if the month is ongoing
    if year == TODAY.year and month == TODAY.month:
        return min(TODAY.day, last_day)
    return last_day


def iter_months(start_year: int, start_month: int, end_year: int, end_month: int):
    """Yield (year, month) tuples for every month in the period."""
    for year in range(start_year, end_year + 1):
        m_start = start_month if year == start_year else 1
        m_end   = end_month   if year == end_year   else 12
        for month in range(m_start, m_end + 1):
            yield year, month


def pf_member_range(pf_start: int, pf_end: int) -> range:
    return range(pf_start, pf_end + 1)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_month(server: ECMWFService, year: int, month: int,
                   outdir: Path, pf_start: int, pf_end: int,
                   download_cf: bool = True) -> bool:
    """Download all variables for one calendar month in a single MARS request.

    Uses the MARS date-range syntax "YYYY-MM-01/to/YYYY-MM-DD" to retrieve
    all days in one submission, minimising queue overhead.

    Returns True if all requested downloads succeed, False if any fail.

    Output files (inside OUTDIR/YYYY-MM/):
      ens_mekong_YYYY-MM_cf_all.grib2
      ens_mekong_YYYY-MM_pf_<start>-<end>_all.grib2
    """
    if is_future_month(year, month):
        print(f"[SKIP]  {year}-{month:02d} is entirely in the future.")
        return True

    last_day = last_available_day(year, month)
    date_from = f"{year}-{month:02d}-01"
    date_to   = f"{year}-{month:02d}-{last_day:02d}"
    # MARS date-range: all days in the month in one request
    date_range = f"{date_from}/to/{date_to}"

    month_dir = outdir / f"{year}-{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{year}-{month:02d}"

    base_request = {
        "class"   : "od",
        "expver"  : "1",
        "stream"  : "enfo",
        "date"    : date_range,
        "time"    : "00",
        "step"    : STEPS,
        "levtype" : "sfc",
        "param"   : ALL_PARAMS,
        "area"    : AREA,
        # "grid"    : GRID,
    }

    success = True

    # --- Control forecast (type=cf) ---
    if not download_cf:
        print(f"[SKIP]  {tag} control forecast (cf) — disabled by --no-cf")
    else:
        cf_file = month_dir / f"ens_mekong_{tag}_cf_all.grib2"
        if cf_file.exists():
            print(f"[SKIP]  {cf_file.name} already exists.")
        else:
            print(f"[START] {tag}  control forecast (cf)  dates: {date_from} → {date_to}")
            try:
                server.execute({**base_request, "type": "cf"}, str(cf_file))
                print(f"[DONE]  Saved → {cf_file}")
            except Exception as e:
                print(f"[FAIL]  {tag} cf: {e}")
                success = False

    # --- Perturbed forecasts: one request for all members ---
    all_members = "/".join(str(m) for m in pf_member_range(pf_start, pf_end))
    pf_file = month_dir / f"ens_mekong_{tag}_pf_{pf_start}-{pf_end}_all.grib2"
    if pf_file.exists():
        print(f"[SKIP]  {pf_file.name} already exists.")
    else:
        print(f"[START] {tag}  perturbed forecasts (pf, members {pf_start}–{pf_end})  dates: {date_from} → {date_to}")
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
    parser = argparse.ArgumentParser(
        description="Download ECMWF IFS ENS forecasts — one MARS request per month."
    )
    parser.add_argument("--start-year",  type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=START_MONTH)
    parser.add_argument("--end-year",    type=int, default=END_YEAR)
    parser.add_argument("--end-month",   type=int, default=END_MONTH)
    parser.add_argument("--pf-start",    type=int, default=DEFAULT_PF_START,
                        help=f"First perturbed member (default: {DEFAULT_PF_START})")
    parser.add_argument("--pf-end",      type=int, default=DEFAULT_PF_END,
                        help=f"Last perturbed member (default: {DEFAULT_PF_END})")
    parser.add_argument("--no-cf",       action="store_true", default=not DEFAULT_DOWNLOAD_CF,
                        help="Skip the control forecast (cf) download")
    parser.add_argument("--outdir",      type=Path, default=OUTDIR,
                        help="Output root directory")
    args = parser.parse_args()

    if not (1 <= args.pf_start <= args.pf_end <= 50):
        parser.error("--pf-start and --pf-end must satisfy 1 ≤ pf_start ≤ pf_end ≤ 50")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    all_months   = list(iter_months(args.start_year, args.start_month, args.end_year, args.end_month))
    total_months = len(all_months)
    n_pf         = args.pf_end - args.pf_start + 1
    cf_label     = "skipped (--no-cf)" if args.no_cf else "yes"

    print(f"[INFO]  Output directory : {outdir}")
    print(f"[INFO]  Period           : {args.start_year}-{args.start_month:02d} → {args.end_year}-{args.end_month:02d}  ({total_months} months)")
    print(f"[INFO]  Variables        : {', '.join(VARIABLES.keys())} (1 combined request/member/month)")
    print(f"[INFO]  Members          : cf (control): {cf_label} + pf members {args.pf_start}–{args.pf_end} ({n_pf} members, 1 combined request/month)")
    print(f"[INFO]  Region (N/W/S/E) : {AREA}")
    print(f"[INFO]  Steps            : {STEPS[:40]}...")
    print("-" * 60)

    server = ECMWFService("mars")
    print("[INFO]  Connected to ECMWF MARS server.")
    print("-" * 60)

    months_requested = 0
    failed_months    = []

    for year, month in all_months:
        months_requested += 1
        print(f"\n[MONTH] {year}-{month:02d}  ({months_requested}/{total_months})")
        ok = download_month(server, year, month, outdir, args.pf_start, args.pf_end,
                            download_cf=not args.no_cf)
        if not ok:
            failed_months.append(f"{year}-{month:02d}")
            print(f"[WARN]  {year}-{month:02d} failed (one or more requests).")
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
