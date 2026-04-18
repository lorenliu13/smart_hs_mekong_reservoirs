"""
Download ECMWF IFS ENS 0-15 day forecasts for the Mekong River Basin.

Variables: tp, 2t, 2d, sp, 10u, 10v, ssrd, strd, sf, sd, swvl1, swvl2, swvl3, swvl4
Period   : 2026-01-01 to 2026-02-28
Run      : 00 UTC only
Steps    : 0-240h (6-hourly) — 41 steps
Members  : Control forecast (cf, member 0) + perturbed forecasts (pf, configurable range)
Region   : Mekong River Basin (N=34, W=89, S=7, E=112)
Grid     : 0.1° × 0.1° (~11 km, finer than ENS native ~18 km — interpolated by MARS)
Format   : GRIB2 (native MARS format), one file per month per member
           Output structure:
             OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_cf_all.grib2
             OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_pf_1_all.grib2
             OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_pf_2_all.grib2
             ...

Efficiency note
---------------
For ENS sfc/fc data all parameters and steps for a given date+time are stored
on the SAME MARS tape file.  This script issues ONE request per member per month
with all param codes joined by "/", minimising tape mounts.
Control (cf) and perturbed (pf) have different MARS type values and must be
retrieved in separate requests.  See ECMWF MARS efficiency guidelines:
https://confluence.ecmwf.int/x/a4AJBQ

To open in Python (filter by variable):
    import xarray as xr
    # Control forecast:
    ds_cf = xr.open_dataset("ens_mekong_2026-01_cf_all.grib2", engine="cfgrib",
                             filter_by_keys={"shortName": "tp"})
    # Perturbed forecast member 1:
    ds_pf = xr.open_dataset("ens_mekong_2026-01_pf_1_all.grib2", engine="cfgrib",
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
GRID = "0.1/0.1"       # ~11 km regular lat/lon, interpolated by MARS from ENS native ~18 km

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

START_YEAR  = 2023
START_MONTH = 1
END_YEAR    = 2023
END_MONTH   = 1

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


def pf_member_range(pf_start: int, pf_end: int) -> range:
    """Return the range of perturbed member numbers pf_start..pf_end."""
    return range(pf_start, pf_end + 1)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_month(server: ECMWFService, year: int, month: int, outdir: Path,
                   pf_start: int, pf_end: int, download_cf: bool = True) -> bool:
    """Download all variables for one month, optionally cf and/or pf.

    Returns True if all requested downloads succeed, False if any fail.

    One request is submitted per member: one for cf and one per pf member.
    All param codes are joined in each request to minimise tape mounts.

    Output files:
      OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_cf_all.grib2         (if download_cf=True)
      OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_pf_1_all.grib2
      OUTDIR/YYYY-MM/ens_mekong_YYYY-MM_pf_2_all.grib2
      ...

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
    print(f"[INFO]  CF          : {'yes' if download_cf else 'no (skipped)'}")
    print(f"[INFO]  PF members  : {pf_start}–{pf_end} (one request each)")

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
    if not download_cf:
        print(f"[SKIP]  {year}-{month:02d} control forecast (cf) — disabled by --no-cf")
    else:
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

    # --- Perturbed forecasts: one request per member ---
    for member in pf_member_range(pf_start, pf_end):
        pf_file = month_dir / f"ens_mekong_{year}-{month:02d}_pf_{member}_all.grib2"
        if pf_file.exists():
            print(f"[SKIP]  {pf_file.name} already exists.")
            continue
        print(f"[START] {year}-{month:02d}  perturbed forecast (pf, member {member})")
        try:
            server.execute(
                {**base_request, "type": "pf", "number": str(member)},
                str(pf_file),
            )
            print(f"[DONE]  Saved → {pf_file}")
        except Exception as e:
            print(f"[FAIL]  {year}-{month:02d} pf member {member}: {e}")
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

    total_months = sum(
        len(range(
            args.start_month if y == args.start_year else 1,
            (args.end_month if y == args.end_year else 12) + 1
        ))
        for y in range(args.start_year, args.end_year + 1)
    )

    print(f"[INFO]  Output directory : {outdir}")
    print(f"[INFO]  Period           : {args.start_year}-{args.start_month:02d} → {args.end_year}-{args.end_month:02d}  ({total_months} months)")
    print(f"[INFO]  Variables        : {', '.join(VARIABLES.keys())} (1 combined request/member/month)")
    cf_label = "skipped (--no-cf)" if args.no_cf else "yes"
    n_pf = args.pf_end - args.pf_start + 1
    print(f"[INFO]  Members          : cf (control): {cf_label} + pf members {args.pf_start}–{args.pf_end} ({n_pf} requests)")
    print(f"[INFO]  Region (N/W/S/E) : {AREA}")
    print(f"[INFO]  Grid             : {GRID}  (~11 km, interpolated from ENS native ~18 km)")
    print(f"[INFO]  Steps            : {STEPS[:40]}...")
    print("-" * 60)

    server = ECMWFService("mars")
    print("[INFO]  Connected to ECMWF MARS server.")
    print("-" * 60)

    months_requested = 0
    failed_months = []
    for year in range(args.start_year, args.end_year + 1):
        m_start = args.start_month if year == args.start_year else 1
        m_end   = args.end_month   if year == args.end_year   else 12
        for month in range(m_start, m_end + 1):
            months_requested += 1
            print(f"\n[MONTH] {year}-{month:02d}  ({months_requested}/{total_months})")
            ok = download_month(server, year, month, outdir, args.pf_start, args.pf_end,
                                download_cf=not args.no_cf)
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
