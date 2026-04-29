#!/usr/bin/env python
"""
Scan tape/disk availability for all 14 ENS variables across sampled dates.

Default sampling
----------------
  Dates  : 1st of each month, Dec 2023 → Feb 2026  (27 dates)
  Members: 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50  (11 PF members, one request each)
           + control forecast (CF, type=cf number=0)
  Vars   : all 14 surface variables

One MARS cost request is submitted per (date × member × variable).
PF total : 27 × 11 × 14 = 4,158 requests
CF total :  27 ×  1 × 14 =   378 requests  (--cf flag, on by default)

Results land in:
    <outdir>/<YYYYMMDD>/varcheck_<YYYYMMDD>_pf<member>_<var>.list   (PF)
    <outdir>/<YYYYMMDD>/varcheck_<YYYYMMDD>_cf0_<var>.list           (CF)

Existing files are skipped automatically so the script is safe to restart.

Usage
-----
    python scan_avail_ens_mekong.py
    python scan_avail_ens_mekong.py --start 202312 --end 202602
    python scan_avail_ens_mekong.py --members 1/5/10/25/50
    python scan_avail_ens_mekong.py --no-cf        # skip control forecast
    python scan_avail_ens_mekong.py --outdir ./my_output
    python scan_avail_ens_mekong.py --dry-run
"""

import argparse
from datetime import date
from pathlib import Path

from ecmwfapi import ECMWFService

# ---------------------------------------------------------------------------
# Configuration — kept in sync with varcheck_ens_mekong.py
# ---------------------------------------------------------------------------

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

AREA  = "34/89/7/112"  # N/W/S/E — full Mekong River Basin

STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

# 1, then every 5th up to 50
DEFAULT_MEMBERS = "1/" + "/".join(str(n) for n in range(10, 51, 10))

DEFAULT_START = "202312"
DEFAULT_END   = "202602"

OUTDIR = Path("./varsize_ens_enfo")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def monthly_firsts(start_ym: str, end_ym: str):
    """Yield date objects for the 1st of each month in [start_ym, end_ym]."""
    sy, sm = int(start_ym[:4]), int(start_ym[4:])
    ey, em = int(end_ym[:4]),   int(end_ym[4:])
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield date(y, m, 1)
        m += 1
        if m > 12:
            m, y = 1, y + 1


def parse_members(members_str: str) -> list[str]:
    """Split a slash-separated member string into a list of individual member strings."""
    return [m.strip() for m in members_str.split("/") if m.strip()]


def build_cost_request(date_str: str, param: str, member: str) -> str:
    return (
        f"list,\n"
        f"  class   = od,\n"
        f"  expver  = 1,\n"
        f"  stream  = enfo,\n"
        f"  type    = pf,\n"
        f"  number  = {member},\n"
        f"  date    = {date_str},\n"
        f"  time    = 00,\n"
        f"  step    = {STEPS},\n"
        f"  levtype = sfc,\n"
        f"  param   = {param},\n"
        f"  area    = {AREA},\n"
        f"  output  = cost\n"
    )


def build_cf_cost_request(date_str: str, param: str) -> str:
    """MARS cost request for the control forecast (type=cf, number=0)."""
    return (
        f"list,\n"
        f"  class   = od,\n"
        f"  expver  = 1,\n"
        f"  stream  = enfo,\n"
        f"  type    = cf,\n"
        f"  number  = 0,\n"
        f"  date    = {date_str},\n"
        f"  time    = 00,\n"
        f"  step    = {STEPS},\n"
        f"  levtype = sfc,\n"
        f"  param   = {param},\n"
        f"  area    = {AREA},\n"
        f"  output  = cost\n"
    )

# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def scan_date_member(server, date_obj: date, member: str, outdir: Path,
                     dry_run: bool) -> dict:
    """Submit one cost request per variable for one date + one member."""
    date_str  = date_obj.strftime("%Y-%m-%d")
    safe_date = date_obj.strftime("%Y%m%d")
    date_dir  = outdir / safe_date
    date_dir.mkdir(parents=True, exist_ok=True)

    n_vars = len(VARIABLES)
    skipped = submitted = 0

    for i, (var, param) in enumerate(VARIABLES.items(), 1):
        out_file = date_dir / f"varcheck_{safe_date}_pf{member}_{var}.list"
        print(f"    [{i:02d}/{n_vars}]  member {member:>2}  {var} ({param})")

        if dry_run:
            print("              [DRY-RUN] would submit")
            continue

        if out_file.exists():
            print("              [SKIP] already exists")
            skipped += 1
            continue

        req = build_cost_request(date_str, param, member)
        server.execute(req, str(out_file))
        print("              [DONE]")
        submitted += 1

    return {"submitted": submitted, "skipped": skipped}


def scan_date_cf(server, date_obj: date, outdir: Path, dry_run: bool) -> dict:
    """Submit one cost request per variable for the control forecast (cf, number=0)."""
    date_str  = date_obj.strftime("%Y-%m-%d")
    safe_date = date_obj.strftime("%Y%m%d")
    date_dir  = outdir / safe_date
    date_dir.mkdir(parents=True, exist_ok=True)

    n_vars = len(VARIABLES)
    skipped = submitted = 0

    for i, (var, param) in enumerate(VARIABLES.items(), 1):
        out_file = date_dir / f"varcheck_{safe_date}_cf0_{var}.list"
        print(f"    [{i:02d}/{n_vars}]  cf0  {var} ({param})")

        if dry_run:
            print("              [DRY-RUN] would submit")
            continue

        if out_file.exists():
            print("              [SKIP] already exists")
            skipped += 1
            continue

        req = build_cf_cost_request(date_str, param)
        server.execute(req, str(out_file))
        print("              [DONE]")
        submitted += 1

    return {"submitted": submitted, "skipped": skipped}


def scan_all(server, start_ym: str, end_ym: str, members: list[str],
             outdir: Path, include_cf: bool, dry_run: bool) -> None:
    dates  = list(monthly_firsts(start_ym, end_ym))
    n_pf   = len(dates) * len(members) * len(VARIABLES)
    n_cf   = len(dates) * len(VARIABLES) if include_cf else 0

    print(f"[INFO]  Dates       : {dates[0]}  →  {dates[-1]}  ({len(dates)} dates)")
    print(f"[INFO]  PF members  : {', '.join(members)}  ({len(members)} members)")
    print(f"[INFO]  Control (CF): {'yes (type=cf, number=0)' if include_cf else 'no (--no-cf)'}")
    print(f"[INFO]  Variables   : {len(VARIABLES)}")
    print(f"[INFO]  Max requests: {n_pf} PF + {n_cf} CF = {n_pf + n_cf}  (skips existing files)")
    if dry_run:
        print("[INFO]  Mode        : DRY-RUN (no requests submitted)\n")

    total_submitted = total_skipped = 0

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        print(f"\n{'='*64}")
        print(f"  {date_str}  |  {len(members)} PF members × {len(VARIABLES)} variables"
              + ("  + CF" if include_cf else ""))
        print(f"{'='*64}")

        for member in members:
            print(f"\n  -- PF member {member} --")
            counts = scan_date_member(server, d, member, outdir, dry_run)
            total_submitted += counts["submitted"]
            total_skipped   += counts["skipped"]

        if include_cf:
            print(f"\n  -- CF (control forecast) --")
            counts = scan_date_cf(server, d, outdir, dry_run)
            total_submitted += counts["submitted"]
            total_skipped   += counts["skipped"]

    print(f"\n{'='*64}")
    print(f"[DONE]  Submitted: {total_submitted}  |  Skipped: {total_skipped}")
    print(f"        Results in: {outdir.resolve()}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan ENS tape/disk availability across sampled dates and members.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start",   default=DEFAULT_START,
                        help="Start month YYYYMM (default: %(default)s)")
    parser.add_argument("--end",     default=DEFAULT_END,
                        help="End month YYYYMM (default: %(default)s)")
    parser.add_argument("--members", default=DEFAULT_MEMBERS,
                        help="Slash-separated member list — one request per member "
                             "(default: %(default)s)")
    parser.add_argument("--no-cf", action="store_true",
                        help="Skip the control forecast (CF) check")
    parser.add_argument("--outdir",  type=Path, default=OUTDIR,
                        help="Root output directory (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be submitted without actually submitting")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    members    = parse_members(args.members)
    include_cf = not args.no_cf

    if args.dry_run:
        scan_all(None, args.start, args.end, members, args.outdir,
                 include_cf=include_cf, dry_run=True)
    else:
        server = ECMWFService("mars")
        print("[INFO]  Connected to ECMWF MARS server.")
        scan_all(server, args.start, args.end, members, args.outdir,
                 include_cf=include_cf, dry_run=False)


if __name__ == "__main__":
    main()
