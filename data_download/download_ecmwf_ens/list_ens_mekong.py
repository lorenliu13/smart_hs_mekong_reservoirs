#!/usr/bin/env python
"""
Estimate the MARS request size for ECMWF IFS ENS hindcast (enfh) Mekong downloads.

Uses the MARS 'list' action with output=cost to report volume without downloading.
Variable: tp (total precipitation) only.
Spatial extent and steps match the ENS download configuration.

ENS hindcast structure
----------------------
stream  : enfh  (ensemble hindcast — NOT real-time enfo)
type    : pf    (perturbed forecast, 10 members)
number  : 1/2/.../10  (hindcast uses 10 members, not 50 like real-time ENS)
date    : reference model run date (the "current" anchor date, e.g. 2024-01-01)
hdate   : the same calendar day across past years (e.g. 2023-01-01/.../2026-01-01)

ECMWF hindcast reference dates are issued twice per week (Mon + Thu).
This script generates Mon/Thu dates for the specified reference year/month range,
then builds hdate by replacing the year with each hindcast year (2023–2025).
hdate years must be strictly before the reference date year.

Modes
-----
    date    -- one pf request for a single reference date
    range   -- iterate over all Mon/Thu reference dates in a month range (default)

Usage
-----
    python list_ens_mekong.py --mode range
    python list_ens_mekong.py --mode range --start-year 2024 --start-month 1 --end-year 2024 --end-month 3
    python list_ens_mekong.py --mode date --date 20240101
"""

import argparse
import calendar
from datetime import date, datetime
from pathlib import Path

from ecmwfapi import ECMWFService

# ---------------------------------------------------------------------------
# Configuration — spatial extent and steps match download_ens_mekong.py
# ---------------------------------------------------------------------------

PARAM = "228.128"        # tp
AREA  = "34/89/7/112"   # N/W/S/E — full Mekong River Basin
GRID  = "0.1/0.1"

STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

PF_NUMBERS = "/".join(str(n) for n in range(1, 11))

HINDCAST_YEARS = list(range(2023, 2026))   # 3 years (must be strictly before the reference date year)

START_YEAR  = 2024
START_MONTH = 1
END_YEAR    = 2024
END_MONTH   = 1

OUTDIR = Path("./file_size_check_ens_hindcast")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mondays_and_thursdays(year: int, month: int):
    """Yield every Monday and Thursday in the month."""
    last_day = calendar.monthrange(year, month)[1]
    for day in range(1, last_day + 1):
        d = date(year, month, day)
        if d.weekday() in (0, 3):
            yield d


def build_hdate(ref_date: date) -> str:
    """Build the hdate string: same mm-dd across all HINDCAST_YEARS.

    If a year has no Feb-29 equivalent, Feb-28 is substituted.
    """
    hdates = []
    for yr in HINDCAST_YEARS:
        try:
            hd = date(yr, ref_date.month, ref_date.day)
        except ValueError:
            hd = date(yr, ref_date.month, 28)
        hdates.append(hd.strftime("%Y-%m-%d"))
    return "/".join(hdates)


# ---------------------------------------------------------------------------
# Core MARS request builder
# ---------------------------------------------------------------------------

def build_mars_list_request(ref_date_str: str, hdate_str: str) -> str:
    """Return a MARS list/cost request string for ENS hindcast (enfh) pf."""
    return f"""list,
  class   = od,
  expver  = 1,
  stream  = enfh,
  type    = pf,
  number  = {PF_NUMBERS},
  date    = {ref_date_str},
  hdate   = {hdate_str},
  time    = 00,
  step    = {STEPS},
  levtype = sfc,
  param   = {PARAM},
  area    = {AREA},
  grid    = {GRID},
  output  = cost
"""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def _run_list(server: ECMWFService, ref_date_str: str, hdate_str: str,
              outdir: Path) -> None:
    """Execute one MARS list/cost request and write the result."""
    safe_date = ref_date_str.replace("-", "")
    list_file = outdir / f"ens_hindcast_{safe_date}_pf_tp_cost.list"
    req = build_mars_list_request(ref_date_str, hdate_str)

    print(f"\n[LIST]  {safe_date} pf  →  {list_file}")
    print(f"[INFO]  Ref date   : {ref_date_str}")
    print(f"[INFO]  Hdate      : {hdate_str[:60]}...")
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
                  outdir: Path) -> None:
    """Iterate all Mon/Thu reference dates over a month range."""
    ref_dates = []
    for year in range(start_year, end_year + 1):
        m_start = start_month if year == start_year else 1
        m_end   = end_month   if year == end_year   else 12
        for month in range(m_start, m_end + 1):
            ref_dates.extend(mondays_and_thursdays(year, month))

    total = len(ref_dates)
    print(f"\n[MODE]  range  —  {start_year}-{start_month:02d} → {end_year}-{end_month:02d}")
    print(f"[INFO]  {total} Mon/Thu reference date(s), 1 pf request each")

    for i, ref_date in enumerate(ref_dates, 1):
        ref_str   = ref_date.strftime("%Y-%m-%d")
        hdate_str = build_hdate(ref_date)
        print(f"\n[DATE]  {ref_str}  ({i}/{total})")
        _run_list(server, ref_str, hdate_str, outdir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate MARS request size for ENS hindcast (enfh) Mekong tp pf downloads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["date", "range"], default="range",
                        help="Estimation mode (default: range)")
    parser.add_argument("--date", type=str, default=None,
                        help="Single reference date YYYYMMDD — required for date mode")
    parser.add_argument("--start-year",  type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=START_MONTH)
    parser.add_argument("--end-year",    type=int, default=END_YEAR)
    parser.add_argument("--end-month",   type=int, default=END_MONTH)
    parser.add_argument("--outdir", type=Path, default=OUTDIR,
                        help="Directory to write .list cost files")
    args = parser.parse_args()

    if args.mode == "date" and not args.date:
        parser.error("--date is required for --mode date")

    args.outdir.mkdir(parents=True, exist_ok=True)

    server = ECMWFService("mars")
    print("[INFO]  Connected to ECMWF MARS server.")
    print(f"[INFO]  Stream     : enfh  (ENS hindcast)")
    print(f"[INFO]  Type       : pf  Members: 1–10")
    print(f"[INFO]  Variable   : tp (228.128)")
    print(f"[INFO]  Region     : {AREA}  Grid: {GRID}")
    print(f"[INFO]  Steps      : 0–240h (6-hourly, 41 steps)")
    print(f"[INFO]  Hdate yrs  : {HINDCAST_YEARS[0]}–{HINDCAST_YEARS[-1]}  ({len(HINDCAST_YEARS)} years)")

    if args.mode == "date":
        ref_date  = datetime.strptime(args.date, "%Y%m%d").date()
        hdate_str = build_hdate(ref_date)
        _run_list(server, ref_date.strftime("%Y-%m-%d"), hdate_str, args.outdir)
    else:
        examine_range(server, args.start_year, args.start_month,
                      args.end_year, args.end_month, args.outdir)

    print("\n[DONE]  All cost estimates complete.")


if __name__ == "__main__":
    main()
