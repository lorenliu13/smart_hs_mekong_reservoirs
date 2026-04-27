#!/usr/bin/env python
"""
Check tape/disk status of all 14 ENS variables for a given date.

Submits one MARS cost request per variable so each response shows the
storage tier (online/tape) and field counts.

Supports both perturbed forecast (PF) and control forecast (CF):
  --members 1          → single PF member  (type=pf, number=1)
  --members 1/2/.../50 → all 50 PF members (type=pf, number=1/2/.../50)
  --members cf         → control forecast  (type=cf, number=0)

Usage
-----
    python varcheck_ens_mekong.py --date 20231201
    python varcheck_ens_mekong.py --date 20231201 --members 1
    python varcheck_ens_mekong.py --date 20231201 --members cf
    python varcheck_ens_mekong.py --date 20231201 --outdir ./varcheck_results
"""

import argparse
from datetime import datetime
from pathlib import Path

from ecmwfapi import ECMWFService

# ---------------------------------------------------------------------------
# Configuration — kept in sync with list_ens_mekong.py
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

AREA  = "34/89/7/112"   # N/W/S/E — full Mekong River Basin

STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

DEFAULT_MEMBERS = "/".join(str(n) for n in range(1, 51))  # "1/2/.../50"

OUTDIR = Path("./varsize_ens_enfo")

# ---------------------------------------------------------------------------
# MARS request
# ---------------------------------------------------------------------------

def build_list_request(date_str: str, param: str, members: str) -> str:
    """MARS cost request for one variable.

    Pass members='cf' for the control forecast (type=cf, number=0).
    Any other value is treated as a PF member string (type=pf).
    """
    if members == "cf":
        run_type = "cf"
        number   = "0"
    else:
        run_type = "pf"
        number   = members
    return (
        f"list,\n"
        f"  class   = od,\n"
        f"  expver  = 1,\n"
        f"  stream  = enfo,\n"
        f"  type    = {run_type},\n"
        f"  number  = {number},\n"
        f"  date    = {date_str},\n"
        f"  time    = 00,\n"
        f"  step    = {STEPS},\n"
        f"  levtype = sfc,\n"
        f"  param   = {param},\n"
        f"  area    = {AREA},\n"
        f"  output  = cost\n"
    )

# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def _file_label(members: str) -> str:
    """Return the filename label for the given members string.

    'cf'       → 'cf0'
    '1'        → 'pf1'
    '1/2/.../50' → 'pf1-50'
    """
    if members == "cf":
        return "cf0"
    parts = members.split("/")
    suffix = f"{parts[0]}-{parts[-1]}" if len(parts) > 1 else parts[0]
    return f"pf{suffix}"


def check_all_variables(server: ECMWFService, date_str: str,
                        members: str, outdir: Path) -> None:
    """Submit one MARS cost request per variable and save the result."""
    safe_date = date_str.replace("-", "")
    label     = _file_label(members)
    total     = len(VARIABLES)

    date_dir = outdir / safe_date
    date_dir.mkdir(parents=True, exist_ok=True)

    run_desc = "CF (control forecast)" if members == "cf" else f"PF members: {members}"
    print(f"\n[INFO]  Date : {date_str}")
    print(f"[INFO]  Type : {run_desc}")
    print(f"[INFO]  Vars : {total}  (one request each)")
    print("=" * 60)

    for i, (var_name, param_id) in enumerate(VARIABLES.items(), 1):
        out_file = date_dir / f"varcheck_{safe_date}_{label}_{var_name}.list"
        req      = build_list_request(date_str, param_id, members)

        print(f"\n[{i:02d}/{total}]  {var_name} ({param_id})  →  {out_file.name}")
        print(req)
        print("-" * 60)

        server.execute(req, str(out_file))
        print(f"[DONE]  {var_name}  →  {out_file}")

    print("\n" + "=" * 60)
    print(f"[DONE]  All {total} variables checked. Results in: {date_dir}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check tape/disk availability of 14 ENS variables for a date.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", type=str, required=True,
                        help="Forecast date YYYYMMDD")
    parser.add_argument("--members", type=str, default=DEFAULT_MEMBERS,
                        help="Member numbers, MARS-style (default: 1/2/.../50). "
                             "Use 'cf' for the control forecast, "
                             "'1' to test with a single PF member.")
    parser.add_argument("--outdir", type=Path, default=OUTDIR,
                        help="Directory to write .list field files")
    args = parser.parse_args()

    try:
        d = datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        parser.error("--date must be YYYYMMDD, e.g. 20231201")

    date_str = d.strftime("%Y-%m-%d")
    args.outdir.mkdir(parents=True, exist_ok=True)

    server = ECMWFService("mars")
    print("[INFO]  Connected to ECMWF MARS server.")
    check_all_variables(server, date_str, args.members, args.outdir)


if __name__ == "__main__":
    main()
