"""
Check completeness of ERA5-Land downloads and re-download corrupt/missing files.

For each expected file (one per variable per month) the script:
  1. Checks the file exists.
  2. Tries to open it with xarray and access the data variable.
  3. Verifies the time dimension has the expected number of hourly steps
     (24 × days-in-month).
  4. Re-downloads any file that fails one of the checks above.

Usage
-----
# Check & repair with default paths / date range (mirrors download script):
    python check_and_redownload_era5_land.py

# Override the directory or date range:
    python check_and_redownload_era5_land.py --outdir /data/.../era5_land \
        --start-year 2023 --start-month 1 --end-year 2025 --end-month 12
"""

import argparse
import calendar
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration  (kept in sync with the download script)
# ---------------------------------------------------------------------------

VARIABLES = {
    "tp"   : "total_precipitation",
    "2t"   : "2m_temperature",
    "2d"   : "2m_dewpoint_temperature",
    "sp"   : "surface_pressure",
    "10u"  : "10m_u_component_of_wind",
    "10v"  : "10m_v_component_of_wind",
    "ssrd" : "surface_solar_radiation_downwards",
    "strd" : "surface_thermal_radiation_downwards",
    "sf"   : "snowfall",
    "sd"   : "snow_depth",
    "swvl1": "volumetric_soil_water_layer_1",
    "swvl2": "volumetric_soil_water_layer_2",
    "swvl3": "volumetric_soil_water_layer_3",
    "swvl4": "volumetric_soil_water_layer_4",
}

AREA  = [34, 89, 7, 112]   # N/W/S/E — Mekong River Basin
GRID  = [0.1, 0.1]
HOURS = [f"{h:02d}:00" for h in range(24)]

START_YEAR = 2023
END_YEAR   = 2025

OUTDIR = Path("/data/ouce-grit/cenv1160/smart_hs/raw_data/era5_land")
# OUTDIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\era5_land")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = datetime.now(timezone.utc).date()


def is_future_month(year: int, month: int) -> bool:
    return datetime(year, month, 1).date() > TODAY


def build_days(year: int, month: int) -> list[str]:
    last_day = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, last_day + 1)]


def expected_timesteps(year: int, month: int) -> int:
    """24 hourly steps × number of days in the month."""
    return 24 * calendar.monthrange(year, month)[1]


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

def check_file(path: Path, year: int, month: int) -> tuple[bool, str]:
    """
    Return (ok, reason).
    ok=True  → file is complete.
    ok=False → reason describes the problem.
    """
    if not path.exists():
        return False, "file missing"

    if path.stat().st_size == 0:
        return False, "file is empty (0 bytes)"

    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        return False, f"cannot open with xarray: {e}"

    # Check time dimension
    if "time" not in ds.dims:
        ds.close()
        return False, "no 'time' dimension found"

    n_time = ds.dims["time"]
    expected = expected_timesteps(year, month)
    ds.close()

    if n_time != expected:
        return False, f"time steps {n_time} ≠ expected {expected}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Re-download
# ---------------------------------------------------------------------------

def download_variable(
    client: cdsapi.Client,
    year: int,
    month: int,
    var_short: str,
    var_long: str,
    outfile: Path,
    days: list[str],
) -> None:
    """Download (or re-download) one variable for one month."""
    # Remove corrupt/partial file before writing
    if outfile.exists():
        outfile.unlink()

    print(f"  [DOWNLOAD] {year}-{month:02d}  {var_short} ({var_long})")
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable" : var_long,
            "year"     : str(year),
            "month"    : f"{month:02d}",
            "day"      : days,
            "time"     : HOURS,
            "area"     : AREA,
            "grid"     : GRID,
            "format"   : "netcdf",
        },
        str(outfile),
    )
    print(f"  [SAVED]    → {outfile}")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def check_and_repair_month(
    client: cdsapi.Client,
    year: int,
    month: int,
    outdir: Path,
) -> dict:
    """
    Check all variables for one month; re-download any that are incomplete.
    Returns a summary dict with counts.
    """
    if is_future_month(year, month):
        print(f"[SKIP] {year}-{month:02d} — future month, data not yet available.")
        return {"skipped": True}

    month_dir = outdir / f"{year}-{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    days = build_days(year, month)

    results = {"ok": [], "fixed": [], "failed": []}

    for var_short, var_long in VARIABLES.items():
        outfile = month_dir / f"era5land_mekong_{year}-{month:02d}_{var_short}.nc"
        ok, reason = check_file(outfile, year, month)

        if ok:
            print(f"[OK]   {outfile.name}")
            results["ok"].append(var_short)
            continue

        print(f"[BAD]  {outfile.name}  — {reason}")

        try:
            download_variable(client, year, month, var_short, var_long, outfile, days)
            # Verify again after re-download
            ok2, reason2 = check_file(outfile, year, month)
            if ok2:
                print(f"  [VERIFY OK]")
                results["fixed"].append(var_short)
            else:
                print(f"  [VERIFY FAIL] {reason2}")
                results["failed"].append(var_short)
        except Exception as e:
            print(f"  [ERROR] re-download failed: {e}")
            results["failed"].append(var_short)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check ERA5-Land download completeness and re-download corrupt files."
    )
    parser.add_argument("--start-year",  type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year",    type=int, default=END_YEAR)
    parser.add_argument("--end-month",   type=int, default=12)
    parser.add_argument("--outdir",      type=Path, default=OUTDIR,
                        help="Root directory that contains YYYY-MM sub-folders")
    args = parser.parse_args()

    outdir = args.outdir

    client = cdsapi.Client()

    total_ok = total_fixed = total_failed = 0

    for year in range(args.start_year, args.end_year + 1):
        m_start = args.start_month if year == args.start_year else 1
        m_end   = args.end_month   if year == args.end_year   else 12
        for month in range(m_start, m_end + 1):
            print(f"\n{'='*60}")
            print(f"Checking {year}-{month:02d}")
            print(f"{'='*60}")
            res = check_and_repair_month(client, year, month, outdir)
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
