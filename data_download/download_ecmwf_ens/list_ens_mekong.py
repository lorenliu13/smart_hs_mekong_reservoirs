#!/usr/bin/env python
"""
Estimate the MARS request size for ECMWF IFS ENS (enfo) Mekong downloads.

Uses the MARS 'list' action with output=cost to report volume without downloading.
Variables: tp, 2t, 2d, sp, 10u, 10v, ssrd, strd, sf, sd, swvl1, swvl2, swvl3, swvl4
Spatial extent and steps match the ENS download configuration.

ENS (enfo) structure
--------------------
stream  : enfo  (real-time ensemble forecast — issued every day)
type    : pf    (perturbed forecast, 50 members)
number  : 1/2/.../50
date    : forecast initialisation date (00 UTC run)
No hdate — enfo is a real-time product available on every calendar day.

Modes
-----
    date    -- one pf request for a single date (all variables or one via --var)
    range   -- iterate over every calendar day in a month range (default)
               add --var to restrict to a single variable

Usage
-----
    python list_ens_mekong.py --mode range
    python list_ens_mekong.py --mode range --start-year 2024 --start-month 1 --end-year 2024 --end-month 3
    python list_ens_mekong.py --mode range --var tp
    python list_ens_mekong.py --mode date --date 20240101
    python list_ens_mekong.py --mode date --date 20240101 --var tp
"""

import argparse
import calendar
from datetime import date, datetime, timedelta
from pathlib import Path

from ecmwfapi import ECMWFService

# ---------------------------------------------------------------------------
# Configuration — spatial extent and steps match download_ens_mekong.py
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

ALL_PARAMS = "/".join(VARIABLES.values())

AREA  = "34/89/7/112"   # N/W/S/E — full Mekong River Basin
GRID  = "0.1/0.1"

STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

# ENS real-time has 50 perturbed members (members 1–50)
PF_NUMBERS = "/".join(str(n) for n in range(1, 51))  # "1/2/.../50"
# PF_NUMBERS = "1"

START_YEAR  = 2023
START_MONTH = 1
END_YEAR    = 2023
END_MONTH   = 1

OUTDIR = Path("./file_size_check_ens_enfo")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_days_in_range(start_year: int, start_month: int,
                      end_year: int, end_month: int):
    """Yield every calendar day from the first of start_month to the last of end_month."""
    start = date(start_year, start_month, 1)
    last_day = calendar.monthrange(end_year, end_month)[1]
    end = date(end_year, end_month, last_day)
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


# ---------------------------------------------------------------------------
# Core MARS request builder
# ---------------------------------------------------------------------------

def build_mars_list_request(date_str: str, params: str = ALL_PARAMS) -> str:
    """Return a MARS list/cost request string for IFS ENS (enfo) pf."""
    return f"""list,
  class   = od,
  expver  = 1,
  stream  = enfo,
  type    = pf,
  number  = {PF_NUMBERS},
  date    = {date_str},
  time    = 00,
  step    = {STEPS},
  levtype = sfc,
  param   = {params},
  area    = {AREA},
  grid    = {GRID},
  output  = cost
"""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def _run_list(server: ECMWFService, date_str: str, outdir: Path,
              params: str = ALL_PARAMS, var_label: str = "allvars") -> None:
    """Execute one MARS list/cost request and write the result."""
    safe_date = date_str.replace("-", "")
    list_file = outdir / f"ens_enfo_{safe_date}_pf_{var_label}_cost.list"
    req = build_mars_list_request(date_str, params)

    print(f"\n[LIST]  {safe_date} pf  →  {list_file}")
    print(f"[INFO]  Date       : {date_str}")
    print(f"[INFO]  Params     : {params}")
    print("-" * 60)
    print(req)
    print("-" * 60)

    server.execute(req, str(list_file))
    print(f"[DONE]  Written to {list_file}")


# ---------------------------------------------------------------------------
# Mode functions
# ---------------------------------------------------------------------------

def examine_range(server: ECMWFService,
                  start_year: int, start_month: int,
                  end_year: int,   end_month: int,
                  outdir: Path,
                  var_filter: str | None = None) -> None:
    """Iterate all calendar days over a month range (enfo is available every day).

    If var_filter is given, only that variable is requested; otherwise all variables.
    """
    params    = VARIABLES[var_filter] if var_filter else ALL_PARAMS
    var_label = var_filter if var_filter else "allvars"

    all_dates = list(all_days_in_range(start_year, start_month, end_year, end_month))
    total = len(all_dates)
    print(f"\n[MODE]  range  —  {start_year}-{start_month:02d} → {end_year}-{end_month:02d}")
    print(f"[INFO]  {total} day(s), variable: {var_label}")

    for i, d in enumerate(all_dates, 1):
        date_str = d.strftime("%Y-%m-%d")
        print(f"\n[DATE]  {date_str}  ({i}/{total})")
        _run_list(server, date_str, outdir, params, var_label)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate MARS request size for IFS ENS (enfo) Mekong pf downloads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["date", "range"], default="range",
                        help="Estimation mode (default: range)")
    parser.add_argument("--date", type=str, default=None,
                        help="Single date YYYYMMDD — required for date mode")
    parser.add_argument("--var", type=str, default=None,
                        help="Restrict to one variable. "
                             "Choices: " + ", ".join(VARIABLES.keys()))
    parser.add_argument("--start-year",  type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=START_MONTH)
    parser.add_argument("--end-year",    type=int, default=END_YEAR)
    parser.add_argument("--end-month",   type=int, default=END_MONTH)
    parser.add_argument("--outdir", type=Path, default=OUTDIR,
                        help="Directory to write .list cost files")
    args = parser.parse_args()

    if args.mode == "date" and not args.date:
        parser.error("--date is required for --mode date")
    if args.var and args.var not in VARIABLES:
        parser.error(f"Unknown --var '{args.var}'. Choose from: {', '.join(VARIABLES.keys())}")

    params    = VARIABLES[args.var] if args.var else ALL_PARAMS
    var_label = args.var if args.var else "allvars"

    args.outdir.mkdir(parents=True, exist_ok=True)

    server = ECMWFService("mars")
    print("[INFO]  Connected to ECMWF MARS server.")
    print(f"[INFO]  Stream     : enfo  (IFS ENS real-time, available every day)")
    print(f"[INFO]  Type       : pf  Members: 1–50")
    print(f"[INFO]  Variable(s): {var_label}")
    print(f"[INFO]  Region     : {AREA}  Grid: {GRID}")
    print(f"[INFO]  Steps      : 0–240h (6-hourly, 41 steps)")

    if args.mode == "date":
        d = datetime.strptime(args.date, "%Y%m%d").date()
        _run_list(server, d.strftime("%Y-%m-%d"), args.outdir, params, var_label)
    else:
        examine_range(server, args.start_year, args.start_month,
                      args.end_year, args.end_month, args.outdir,
                      var_filter=args.var)

    print("\n[DONE]  All cost estimates complete.")


if __name__ == "__main__":
    main()
