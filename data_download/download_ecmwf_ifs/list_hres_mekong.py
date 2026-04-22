#!/usr/bin/env python
"""
Estimate the MARS request size for ECMWF IFS HRES Mekong downloads.

Uses the MARS 'list' action with output=cost to report volume without
downloading data.  Mirrors the request parameters in download_hres_mekong.py.

Modes:
    date-all      -- one combined request for all variables on a single date
    date-each-var -- one request per variable on a single date (iterates all vars)
    range         -- one combined request per month over a date range (default)

Usage:
    python list_hres_mekong.py --mode range                                # default month range all variables
    python list_hres_mekong.py --mode month-each-var --start-year 2026 --start-month 1 --var tp # single var per month
    python list_hres_mekong.py --mode date-all      --date 20260101
    python list_hres_mekong.py --mode date-each-var --date 20260101
    python list_hres_mekong.py --mode date-each-var --date 20260101 --var tp  # single var only
"""

import argparse
import calendar
from pathlib import Path

from ecmwfapi import ECMWFService

# ---------------------------------------------------------------------------
# Configuration — must match download_hres_mekong.py
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

# AREA   = "34/89/7/112"   # N/W/S/E — full Mekong River Basin
# GRID   = "0.1/0.1"
STEPS  = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

ALL_PARAMS = "/".join(VARIABLES.values())

START_YEAR  = 2023
START_MONTH = 1
END_YEAR    = 2023
END_MONTH   = 1

OUTDIR = Path("./file_size_check")   # list files are small; write to cwd by default

# ---------------------------------------------------------------------------
# Core MARS request builder
# ---------------------------------------------------------------------------

def build_mars_list_request(date_str: str, params: str = ALL_PARAMS) -> str:
    """Return a MARS request string using the 'list' action with output=cost."""
    return f"""list,
  class   = od,
  expver  = 1,
  stream  = oper,
  type    = fc,
  date    = {date_str},
  time    = 00,
  step    = {STEPS},
  levtype = sfc,
  param   = {params},
  area    = {AREA},
  grid    = {GRID},
  output  = cost
"""


def _run_list(server: ECMWFService, date_str: str, label: str,
              outdir: Path, params: str, var_label: str) -> None:
    """Execute one MARS list/cost request and write the result."""
    list_file = outdir / f"hres_mekong_{label}_{var_label}_cost.list"
    req = build_mars_list_request(date_str, params)

    print(f"\n[LIST]  {label} / {var_label}  →  {list_file}")
    print(f"[INFO]  Date   : {date_str}")
    print(f"[INFO]  Params : {params}")
    print("-" * 60)
    print(req)
    print("-" * 60)

    server.execute(req, str(list_file))
    print(f"[DONE]  Written to {list_file}")


# ---------------------------------------------------------------------------
# Mode functions
# ---------------------------------------------------------------------------

def examine_date_all(server: ECMWFService, date: str, outdir: Path) -> None:
    """Mode: one combined MARS request for all variables on a single date."""
    print(f"\n[MODE]  date-all  —  {len(VARIABLES)} variables in one request")
    _run_list(server, date, date, outdir, ALL_PARAMS, "all")


def examine_date_each_var(server: ECMWFService, date: str, outdir: Path,
                          var_filter: str | None = None) -> None:
    """Mode: one MARS request per variable on a single date.

    If var_filter is given, only that variable is run; otherwise all are iterated.
    """
    vars_to_run = {var_filter: VARIABLES[var_filter]} if var_filter else VARIABLES
    print(f"\n[MODE]  date-each-var  —  {len(vars_to_run)} variable(s) on {date}")
    for vname, vparam in vars_to_run.items():
        _run_list(server, date, date, outdir, vparam, vname)


def examine_month_each_var(server: ECMWFService, year: int, month: int, outdir: Path,
                           var_filter: str | None = None) -> None:
    """Mode: one MARS request per variable for a single month.

    If var_filter is given, only that variable is run; otherwise all are iterated.
    """
    last_day = calendar.monthrange(year, month)[1]
    date_str = f"{year}-{month:02d}-01/to/{year}-{month:02d}-{last_day}"
    label    = f"{year}-{month:02d}"
    vars_to_run = {var_filter: VARIABLES[var_filter]} if var_filter else VARIABLES
    print(f"\n[MODE]  month-each-var  —  {label}, {len(vars_to_run)} variable(s)")
    for vname, vparam in vars_to_run.items():
        _run_list(server, date_str, label, outdir, vparam, vname)


def examine_range(server: ECMWFService,
                  start_year: int, start_month: int,
                  end_year: int,   end_month: int,
                  outdir: Path) -> None:
    """Mode: one combined request per month over a date range (all variables)."""
    print(f"\n[MODE]  range  —  {start_year}-{start_month:02d} → {end_year}-{end_month:02d}")
    for year in range(start_year, end_year + 1):
        m_start = start_month if year == start_year else 1
        m_end   = end_month   if year == end_year   else 12
        for month in range(m_start, m_end + 1):
            last_day = calendar.monthrange(year, month)[1]
            date_str = f"{year}-{month:02d}-01/to/{year}-{month:02d}-{last_day}"
            label    = f"{year}-{month:02d}"
            _run_list(server, date_str, label, outdir, ALL_PARAMS, "all")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate MARS request size for HRES Mekong downloads (list/cost action).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["date-all", "date-each-var", "month-each-var", "range"],
                        default="range",
                        help="Estimation mode (default: range)")
    parser.add_argument("--date", type=str, default=None,
                        help="Single date YYYYMMDD — required for date-all and date-each-var modes")
    parser.add_argument("--var", type=str, default=None,
                        help="Restrict date-each-var to one variable. "
                             "Choices: " + ", ".join(VARIABLES.keys()))
    parser.add_argument("--start-year",  type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=START_MONTH)
    parser.add_argument("--end-year",    type=int, default=END_YEAR)
    parser.add_argument("--end-month",   type=int, default=END_MONTH)
    parser.add_argument("--outdir", type=Path, default=OUTDIR,
                        help="Directory to write .list cost files (default: cwd)")
    args = parser.parse_args()

    # Validate
    if args.mode in ("date-all", "date-each-var") and not args.date:
        parser.error(f"--date is required for --mode {args.mode}")
    if args.var and args.var not in VARIABLES:
        parser.error(f"Unknown --var '{args.var}'. Choose from: {', '.join(VARIABLES.keys())}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    server = ECMWFService("mars")
    print("[INFO]  Connected to ECMWF MARS server.")
    print(f"[INFO]  Region : {AREA}  Grid: {GRID}")

    if args.mode == "date-all":
        examine_date_all(server, args.date, args.outdir)
    elif args.mode == "date-each-var":
        examine_date_each_var(server, args.date, args.outdir, var_filter=args.var)
    elif args.mode == "month-each-var":
        examine_month_each_var(server, args.start_year, args.start_month,
                               args.outdir, var_filter=args.var)
    else:
        examine_range(server, args.start_year, args.start_month,
                      args.end_year, args.end_month, args.outdir)

    print("\n[DONE]  All cost estimates complete.")


if __name__ == "__main__":
    main()
