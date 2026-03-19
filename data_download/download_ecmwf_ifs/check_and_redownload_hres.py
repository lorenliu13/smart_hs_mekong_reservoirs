"""
Check completeness of ECMWF IFS HRES downloads and re-download corrupt/missing files.

For each expected file (one per variable per month) the script:
  1. Checks the file exists and is non-empty.
  2. Tries to open it with xarray (engine="cfgrib").
  3. Verifies the 'time' dimension == days in month  (one 00 UTC run per day).
  4. Verifies the 'step' dimension == N_STEPS        (41 steps: 0/6/.../240 h).
  5. Re-downloads any file that fails one of the checks above.

Requires:
    pip install cfgrib ecmwf-api-client

Usage
-----
# Default paths / date range (mirrors download script):
    python check_and_redownload_hres.py

# Override:
    python check_and_redownload_hres.py --outdir /your/path \
        --start-year 2023 --start-month 1 --end-year 2025 --end-month 12
"""

import argparse
import calendar
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr
from ecmwfapi import ECMWFService

# ---------------------------------------------------------------------------
# Configuration  (kept in sync with the download script)
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

AREA  = "34/89/7/112"
GRID  = "0.1/0.1"

STEPS = (
    "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120"
    "/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240"
)

# Number of forecast steps per run (count of values in STEPS above)
N_STEPS = len(STEPS.split("/"))   # 41

START_YEAR = 2023
END_YEAR   = 2025

OUTDIR = Path("/data/ouce-grit/cenv1160/smart_hs/raw_data/ecmwf_ifs/hres")
# OUTDIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\ecmwf_ifs\hres")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = datetime.now(timezone.utc).date()


def is_future_month(year: int, month: int) -> bool:
    return datetime(year, month, 1).date() > TODAY


def build_date_str(year: int, month: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01/to/{year}-{month:02d}-{last_day}"


def days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

def check_file(path: Path, year: int, month: int) -> tuple[bool, str]:
    """
    Return (ok, reason).
    ok=True  → file passes all checks.
    ok=False → reason describes the first problem found.
    """
    if not path.exists():
        return False, "file missing"

    if path.stat().st_size == 0:
        return False, "file is empty (0 bytes)"

    try:
        ds = xr.open_dataset(path, engine="cfgrib")
    except Exception as e:
        return False, f"cannot open with cfgrib: {e}"

    # Check 'time' dimension: one 00 UTC run per calendar day
    if "time" not in ds.dims:
        ds.close()
        return False, "no 'time' dimension found"

    n_time = ds.dims["time"]
    expected_days = days_in_month(year, month)
    if n_time != expected_days:
        ds.close()
        return False, f"time steps {n_time} ≠ expected {expected_days} (days in month)"

    # Check 'step' dimension: 41 forecast steps per run
    if "step" not in ds.dims:
        ds.close()
        return False, "no 'step' dimension found"

    n_steps = ds.dims["step"]
    if n_steps != N_STEPS:
        ds.close()
        return False, f"step count {n_steps} ≠ expected {N_STEPS}"

    ds.close()
    return True, "ok"


# ---------------------------------------------------------------------------
# Re-download
# ---------------------------------------------------------------------------

def download_variable(
    server: ECMWFService,
    year: int,
    month: int,
    var_name: str,
    param_code: str,
    outfile: Path,
    date_str: str,
) -> None:
    """Download (or re-download) one variable for one month."""
    if outfile.exists():
        outfile.unlink()

    print(f"  [DOWNLOAD] {year}-{month:02d}  {var_name} ({param_code})")
    server.execute({
        "class"   : "od",
        "expver"  : "1",
        "stream"  : "oper",
        "type"    : "fc",
        "date"    : date_str,
        "time"    : "00",
        "step"    : STEPS,
        "levtype" : "sfc",
        "param"   : param_code,
        "area"    : AREA,
        "grid"    : GRID,
    }, str(outfile))
    print(f"  [SAVED]    → {outfile}")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def check_and_repair_month(
    server: ECMWFService,
    year: int,
    month: int,
    outdir: Path,
) -> dict:
    """Check all variables for one month; re-download any that are incomplete."""
    if is_future_month(year, month):
        print(f"[SKIP] {year}-{month:02d} — future month, data not yet available.")
        return {"skipped": True}

    month_dir = outdir / f"{year}-{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    date_str = build_date_str(year, month)

    results = {"ok": [], "fixed": [], "failed": []}

    for var_name, param_code in VARIABLES.items():
        outfile = month_dir / f"hres_mekong_{year}-{month:02d}_{var_name}.grib2"
        ok, reason = check_file(outfile, year, month)

        if ok:
            print(f"[OK]   {outfile.name}")
            results["ok"].append(var_name)
            continue

        print(f"[BAD]  {outfile.name}  — {reason}")

        try:
            download_variable(server, year, month, var_name, param_code, outfile, date_str)
            ok2, reason2 = check_file(outfile, year, month)
            if ok2:
                print(f"  [VERIFY OK]")
                results["fixed"].append(var_name)
            else:
                print(f"  [VERIFY FAIL] {reason2}")
                results["failed"].append(var_name)
        except Exception as e:
            print(f"  [ERROR] re-download failed: {e}")
            results["failed"].append(var_name)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check ECMWF IFS HRES download completeness and re-download corrupt files."
    )
    parser.add_argument("--start-year",  type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year",    type=int, default=END_YEAR)
    parser.add_argument("--end-month",   type=int, default=12)
    parser.add_argument("--outdir",      type=Path, default=OUTDIR,
                        help="Root directory containing YYYY-MM sub-folders")
    args = parser.parse_args()

    outdir = args.outdir
    server = ECMWFService("mars")

    total_ok = total_fixed = total_failed = 0

    for year in range(args.start_year, args.end_year + 1):
        m_start = args.start_month if year == args.start_year else 1
        m_end   = args.end_month   if year == args.end_year   else 12
        for month in range(m_start, m_end + 1):
            print(f"\n{'='*60}")
            print(f"Checking {year}-{month:02d}")
            print(f"{'='*60}")
            res = check_and_repair_month(server, year, month, outdir)
            if res.get("skipped"):
                continue
            total_ok     += len(res["ok"])
            total_fixed  += len(res["fixed"])
            total_failed += len(res["failed"])

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  OK (no action needed) : {total_ok}")
    print(f"  Fixed (re-downloaded) : {total_fixed}")
    print(f"  Failed (still broken) : {total_failed}")
    if total_failed == 0:
        print("\nAll files are complete.")
    else:
        print(f"\n{total_failed} file(s) could not be repaired — check logs above.")


if __name__ == "__main__":
    main()
