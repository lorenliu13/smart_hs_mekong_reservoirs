#!/usr/bin/env python
"""
Aggregate tape/disk cost results across all scanned dates, members, and CF.

Reads the .list files produced by scan_avail_ens_mekong.py and outputs:
  - a console table (grouped by date, one row per member × variable)
  - a per-date totals table
  - a CSV file  (one row per date × member × variable)

The control forecast (CF) is read from varcheck_<date>_cf0_<var>.list files
and appears as member "cf0" in the output (unless --no-cf is passed).

Usage
-----
    python multisummary_ens_mekong.py
    python multisummary_ens_mekong.py --members 1/5/10/15/20/25/30/35/40/45/50
    python multisummary_ens_mekong.py --no-cf
    python multisummary_ens_mekong.py --outdir ./varsize_ens_enfo --csv results.csv
    python multisummary_ens_mekong.py --no-detail
"""

import argparse
import csv
import re
from itertools import groupby
from pathlib import Path

# ---------------------------------------------------------------------------
# Variable list — kept in sync with scan_avail_ens_mekong.py
# ---------------------------------------------------------------------------

VARIABLES = [
    ("tp",    "Total precip"),
    ("2t",    "2m temperature"),
    ("2d",    "2m dewpoint"),
    ("sp",    "Surface pressure"),
    ("10u",   "10m U wind"),
    ("10v",   "10m V wind"),
    ("ssrd",  "Solar radiation down"),
    ("strd",  "Thermal radiation down"),
    ("sf",    "Snowfall"),
    ("sd",    "Snow depth"),
    ("swvl1", "Soil water L1"),
    ("swvl2", "Soil water L2"),
    ("swvl3", "Soil water L3"),
    ("swvl4", "Soil water L4"),
]

# 1, then every 5th up to 50
DEFAULT_MEMBERS = "1/" + "/".join(str(n) for n in range(5, 51, 5))

OUTDIR = Path("./varsize_ens_enfo")

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_members(members_str: str) -> list[str]:
    return [m.strip() for m in members_str.split("/") if m.strip()]


def parse_cost_file(path: Path) -> dict:
    """Parse a MARS cost .list file into key→int/str values."""
    data = {}
    for line in path.read_text().splitlines():
        line = line.strip().rstrip(";")
        if "=" in line:
            key, _, val = line.partition("=")
            try:
                data[key.strip()] = int(val.strip())
            except ValueError:
                data[key.strip()] = val.strip()
    return data


def status_label(online: int, offline: int) -> str:
    total = online + offline
    if total == 0:
        return "N/A"
    if offline == 0:
        return "Online"
    if online == 0:
        return "Tape"
    pct = online / total
    if pct >= 0.9:
        return "Mostly Online"
    if pct <= 0.1:
        return "Mostly Tape"
    return "Mixed"

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _make_row(date_str: str, member_label: str, var: str, desc: str,
              fpath: Path) -> dict:
    """Return one result row, reading fpath if it exists."""
    if not fpath.exists():
        return {
            "date": date_str, "member": member_label,
            "var": var, "description": desc,
            "size_MB": None, "online_fields": None,
            "offline_fields": None, "status": "MISSING",
        }
    d       = parse_cost_file(fpath)
    size_mb = d.get("size", 0) / 1e6
    n_on    = d.get("number_of_online_fields",  0)
    n_off   = d.get("number_of_offline_fields", 0)
    return {
        "date":           date_str,
        "member":         member_label,
        "var":            var,
        "description":    desc,
        "size_MB":        round(size_mb, 1),
        "online_fields":  n_on,
        "offline_fields": n_off,
        "status":         status_label(n_on, n_off),
    }


def collect_rows(outdir: Path, members: list[str],
                 include_cf: bool = True) -> list[dict]:
    """Walk every YYYYMMDD sub-directory and parse one row per date × member × variable.

    PF members are read from varcheck_<date>_pf<N>_<var>.list.
    Control forecast is read from varcheck_<date>_cf0_<var>.list (member label 'cf0').
    """
    rows = []
    date_dirs = sorted(
        d for d in outdir.iterdir()
        if d.is_dir() and re.fullmatch(r"\d{8}", d.name)
    )

    for date_dir in date_dirs:
        date_str = date_dir.name  # YYYYMMDD

        # perturbed forecast members
        for member in members:
            for var, desc in VARIABLES:
                fname = f"varcheck_{date_str}_pf{member}_{var}.list"
                rows.append(_make_row(date_str, member, var, desc,
                                      date_dir / fname))

        # control forecast
        if include_cf:
            for var, desc in VARIABLES:
                fname = f"varcheck_{date_str}_cf0_{var}.list"
                rows.append(_make_row(date_str, "cf0", var, desc,
                                      date_dir / fname))

    return rows

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_console(rows: list[dict]) -> None:
    """Print results grouped by date, then member."""
    hdr = f"{'Mbr':>4}  {'Var':<7}{'Size (MB)':>11}{'Online':>9}{'Offline':>10}  Status"
    sep = "-" * len(hdr)

    cur_date = None
    cur_mem  = None
    for r in rows:
        if r["date"] != cur_date:
            cur_date = r["date"]
            cur_mem  = None
            print(f"\n{'='*64}")
            print(f"  Date: {cur_date}")

        if r["member"] != cur_mem:
            cur_mem = r["member"]
            print(sep)
            print(f"  Member {cur_mem}")
            print(sep)
            print(hdr)
            print(sep)

        size  = f"{r['size_MB']:.1f}"    if r["size_MB"]       is not None else "N/A"
        n_on  = str(r["online_fields"])  if r["online_fields"]  is not None else "N/A"
        n_off = str(r["offline_fields"]) if r["offline_fields"] is not None else "N/A"
        print(f"{r['member']:>4}  {r['var']:<7}{size:>11}{n_on:>9}{n_off:>10}  {r['status']}")


def print_date_totals(rows: list[dict]) -> None:
    """One summary line per date (summed across all members and variables)."""
    print(f"\n{'='*64}")
    print("Per-date totals  (all members × all variables combined)")
    print(f"{'Date':<12}{'Total MB':>11}{'Online flds':>12}{'Offline flds':>14}  Status")
    print("-" * 64)

    for date_str, grp in groupby(rows, key=lambda r: r["date"]):
        grp      = list(grp)
        total_mb = sum(r["size_MB"]       or 0 for r in grp)
        tot_on   = sum(r["online_fields"] or 0 for r in grp)
        tot_off  = sum(r["offline_fields"]or 0 for r in grp)
        print(f"{date_str:<12}{total_mb:>11.1f}{tot_on:>12}{tot_off:>14}  {status_label(tot_on, tot_off)}")


def write_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = ["date", "member", "var", "description",
                  "size_MB", "online_fields", "offline_fields", "status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[CSV]  Written → {csv_path}  ({len(rows)} rows)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate tape/disk cost results across all scanned dates and members.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--members", default=DEFAULT_MEMBERS,
                        help="Slash-separated member list to look up "
                             "(default: %(default)s)")
    parser.add_argument("--no-cf", action="store_true",
                        help="Exclude control forecast (CF) from the summary")
    parser.add_argument("--outdir", type=Path, default=OUTDIR,
                        help="Root directory containing <YYYYMMDD>/ subfolders")
    parser.add_argument("--csv",    type=Path, default=None,
                        help="CSV output path (default: <outdir>/avail_summary.csv)")
    parser.add_argument("--no-detail", action="store_true",
                        help="Skip per-variable detail; show only date totals")
    args = parser.parse_args()

    members = parse_members(args.members)
    rows    = collect_rows(args.outdir, members, include_cf=not args.no_cf)

    if not rows:
        print(f"[WARN]  No .list files found in {args.outdir}")
        return

    if not args.no_detail:
        print_console(rows)

    print_date_totals(rows)

    csv_path = args.csv or args.outdir / "avail_summary.csv"
    write_csv(rows, csv_path)


if __name__ == "__main__":
    main()
