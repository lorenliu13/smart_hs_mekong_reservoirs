#!/usr/bin/env python
"""
Summarise tape/disk availability across all 14 ENS variables for a given
date and member (as produced by varcheck_ens_mekong.py).

Supports both perturbed forecast (PF) and control forecast (CF):
  --member 30    → PF member 30    (reads varcheck_<date>_pf30_<var>.list)
  --member 1-50  → PF members 1-50 (reads varcheck_<date>_pf1-50_<var>.list)
  --member cf    → control forecast (reads varcheck_<date>_cf0_<var>.list)

Usage
-----
    python varsummary_ens_mekong.py --date 20231201 --member 30
    python varsummary_ens_mekong.py --date 20231201 --member 1-50
    python varsummary_ens_mekong.py --date 20231201 --member cf
    python varsummary_ens_mekong.py --date 20231201 --member 1 --outdir ./varsize_ens_enfo
"""

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Variable metadata — kept in sync with varcheck_ens_mekong.py
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

OUTDIR = Path("./varsize_ens_enfo")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_label(member: str) -> str:
    """Return the filename label for the given member string.

    'cf'   → 'cf0'   (control forecast)
    '30'   → 'pf30'  (single PF member)
    '1-50' → 'pf1-50' (PF member range label)
    """
    return "cf0" if member == "cf" else f"pf{member}"

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_list_file(path: Path) -> dict:
    """Parse a MARS cost .list file into a dict of key->int/float values."""
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
        return "TAPE (all)"
    if online / total >= 0.5:
        return "Mostly Online"
    return "Mostly TAPE"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(date: str, member: str, outdir: Path) -> None:
    date_dir = outdir / date

    col_widths = {
        "var":    6,
        "desc":   24,
        "size":   16,
        "online": 15,
        "offline":16,
        "status": 14,
    }

    header = (
        f"{'Var':<{col_widths['var']}}"
        f"{'Description':<{col_widths['desc']}}"
        f"{'Total size (MB)':>{col_widths['size']}}"
        f"{'Online fields':>{col_widths['online']}}"
        f"{'Offline fields':>{col_widths['offline']}}"
        f"  {'Status'}"
    )
    separator = "-" * len(header)

    label    = _file_label(member)
    run_desc = "CF (control forecast)" if member == "cf" else f"PF member: {member}"
    print(f"\nDate: {date}   {run_desc}")
    print(separator)
    print(header)
    print(separator)

    missing = []
    for var, desc in VARIABLES:
        fname = f"varcheck_{date}_{label}_{var}.list"
        fpath = date_dir / fname

        if not fpath.exists():
            missing.append(var)
            row = (
                f"{var:<{col_widths['var']}}"
                f"{desc:<{col_widths['desc']}}"
                f"{'N/A':>{col_widths['size']}}"
                f"{'N/A':>{col_widths['online']}}"
                f"{'N/A':>{col_widths['offline']}}"
                f"  MISSING"
            )
            print(row)
            continue

        d = parse_list_file(fpath)
        size_mb  = d.get("size", 0) / 1e6
        n_online  = d.get("number_of_online_fields", 0)
        n_offline = d.get("number_of_offline_fields", 0)
        status   = status_label(n_online, n_offline)

        row = (
            f"{var:<{col_widths['var']}}"
            f"{desc:<{col_widths['desc']}}"
            f"{size_mb:>{col_widths['size']}.1f}"
            f"{n_online:>{col_widths['online']}}"
            f"{n_offline:>{col_widths['offline']}}"
            f"  {status}"
        )
        print(row)

    print(separator)
    if missing:
        print(f"[WARN]  Missing .list files for: {', '.join(missing)}")
        print(f"        Expected in: {date_dir}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a per-variable tape/disk summary for a given date and member.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", required=True,
                        help="Forecast date as YYYYMMDD (e.g. 20231201)")
    parser.add_argument("--member", required=True,
                        help="Member label: PF number/range (e.g. 30, 1-50) "
                             "or 'cf' for the control forecast")
    parser.add_argument("--outdir", type=Path, default=OUTDIR,
                        help="Root directory containing varsize_ens_enfo/<date>/ folders")
    args = parser.parse_args()

    summarise(args.date, args.member, args.outdir)


if __name__ == "__main__":
    main()
