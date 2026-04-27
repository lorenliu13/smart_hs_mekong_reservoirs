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
# GRID  = "0.1/0.1"

STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

# ENS real-time has 50 perturbed members (members 1–50)
# PF_NUMBERS = "/".join(str(n) for n in range(1, 51))  # "1/2/.../50"
PF_NUMBERS = "1"
# PF_NUMBERS = "/".join(str(n) for n in range(1, 51))
# PF_NUMBERS = "1/2/3/4/5/6/7/8/9/10"  # for testing — adjust as needed

START_YEAR  = 2023
START_MONTH = 12
END_YEAR    = 2023
END_MONTH   = 12

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
    """Return a MARS list/cost request string for IFS ENS (enfo) pf.

    date_str can be a single date or a '/'-joined list of dates.
    """
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
  output  = cost
"""

# #   area    = {AREA},

# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def _pf_label(pf_numbers: str) -> str:
    """Return a compact pf-member label: '1-50' for a range, or the number itself."""
    parts = pf_numbers.split("/")
    return f"{parts[0]}-{parts[-1]}" if len(parts) > 1 else parts[0]


def _run_list(server: ECMWFService, date_str: str, outdir: Path,
              params: str = ALL_PARAMS, var_label: str = "allvars",
              file_label: str | None = None) -> None:
    """Execute one MARS list/cost request and write the result.

    file_label overrides the date portion in the output filename (e.g. '202312'
    for a monthly bulk request). Falls back to a sanitised date_str.
    """
    pf_lbl = _pf_label(PF_NUMBERS)
    label = file_label if file_label else date_str.replace("-", "")
    list_file = outdir / f"ens_enfo_{label}_pf{pf_lbl}_{var_label}_cost.list"
    req = build_mars_list_request(date_str, params)

    print(f"\n[LIST]  {label} pf  →  {list_file}")
    print(f"[INFO]  Date(s)    : {date_str[:60]}{'…' if len(date_str) > 60 else ''}")
    print(f"[INFO]  Params     : {params}")
    print("-" * 60)
    print(req)
    print("-" * 60)

    server.execute(req, str(list_file))
    print(f"[DONE]  Written to {list_file}")


# ---------------------------------------------------------------------------
# Mode functions
# ---------------------------------------------------------------------------

def _months_in_range(start_year: int, start_month: int,
                     end_year: int, end_month: int):
    """Yield (year, month) tuples covering the given inclusive month range."""
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def examine_range(server: ECMWFService,
                  start_year: int, start_month: int,
                  end_year: int,   end_month: int,
                  outdir: Path,
                  var_filter: str | None = None) -> None:
    """One bulk MARS list/cost request per month, covering all days in that month.

    If var_filter is given, only that variable is requested; otherwise all variables.
    """
    params    = VARIABLES[var_filter] if var_filter else ALL_PARAMS
    var_label = var_filter if var_filter else "allvars"

    months = list(_months_in_range(start_year, start_month, end_year, end_month))
    print(f"\n[MODE]  range (bulk-by-month)  —  {start_year}-{start_month:02d} → {end_year}-{end_month:02d}")
    print(f"[INFO]  {len(months)} month(s), variable: {var_label}")

    for i, (y, m) in enumerate(months, 1):
        last_day = calendar.monthrange(y, m)[1]
        days = [date(y, m, d).strftime("%Y-%m-%d") for d in range(1, last_day + 1)]
        date_str   = "/".join(days)          # all days joined for MARS
        file_label = f"{y}{m:02d}"           # e.g. 202312
        print(f"\n[MONTH] {y}-{m:02d}  ({i}/{len(months)})  —  {len(days)} days")
        _run_list(server, date_str, outdir, params, var_label, file_label=file_label)


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
    print(f"[INFO]  Type       : pf  Members: {_pf_label(PF_NUMBERS)}")
    print(f"[INFO]  Variable(s): {var_label}")
    # print(f"[INFO]  Region     : {AREA}  Grid: {GRID}")
    print(f"[INFO]  Steps      : {STEPS}")

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
