"""
Download ERA5-Land hourly reanalysis for the Mekong River Basin.

Variables : total_precipitation, 2m_temperature, 2m_dewpoint_temperature,
            surface_pressure, 10m_u_component_of_wind, 10m_v_component_of_wind,
            surface_solar_radiation_downwards, surface_thermal_radiation_downwards,
            snowfall, snow_depth,
            volumetric_soil_water_layer_1, volumetric_soil_water_layer_2
            (mirrors the 12 variables in the ECMWF IFS HRES download)
Period    : 2023-01-01 to 2025-12-31
Region    : Mekong River Basin  (N=34, W=89, S=7, E=112)
Format    : NetCDF, one file per variable per month
Output    : OUTDIR/YYYY-MM/era5land_mekong_YYYY-MM_<var>.nc

Notes
-----
ERA5-Land data typically has a ~3-month lag behind real time.
Data are downloaded as hourly; aggregate to daily in post-processing:
  - Instantaneous vars (2t, 2d, sp, 10u, 10v): daily mean of 24 hours
  - Accumulated vars   (tp, ssrd, strd)       : sum of 24 hours (units: m or J/m²)

Requires the CDS API client:
    pip install cdsapi
    ~/.cdsapirc must contain your CDS UID + API key.
    See: https://cds.climate.copernicus.eu/api-how-to
"""

import argparse
import calendar
from datetime import datetime, timezone
from pathlib import Path

import cdsapi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Mapping: short name → CDS long name
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

# N/W/S/E — full Mekong River Basin (same as HRES download)
AREA = [34, 89, 7, 112]

# All 24 hours for daily aggregation in post-processing
HOURS = [f"{h:02d}:00" for h in range(24)]

START_YEAR = 2022
START_MONTH = 1
END_YEAR   = 2022
END_MONTH = 12

OUTDIR = Path("/data/ouce-grit/cenv1160/smart_hs/raw_data/era5_land")
# OUTDIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\era5_land")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = datetime.now(timezone.utc).date()


def is_future_month(year: int, month: int) -> bool:
    """Return True if the month is in the future (not yet available in CDS)."""
    first_day = datetime(year, month, 1).date()
    return first_day > TODAY


def build_days(year: int, month: int) -> list[str]:
    last_day = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, last_day + 1)]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_variable(
    client: cdsapi.Client,
    year: int,
    month: int,
    var_short: str,
    var_long: str,
    month_dir: Path,
    days: list[str],
) -> None:
    """Download one variable for one month of ERA5-Land hourly data."""
    outfile = month_dir / f"era5land_mekong_{year}-{month:02d}_{var_short}.nc"

    if outfile.exists():
        print(f"[SKIP]  {outfile.name} already exists.")
        return

    print(f"[START] {year}-{month:02d}  {var_short} ({var_long})")

    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable"    : var_long,
            "year"        : str(year),
            "month"       : f"{month:02d}",
            "day"         : days,
            "time"        : HOURS,
            "area"        : AREA,   # [N, W, S, E]
            "format"      : "netcdf",
        },
        str(outfile),
    )

    print(f"[DONE]  Saved → {outfile}")


def download_month(client: cdsapi.Client, year: int, month: int, outdir: Path) -> list[str]:
    """Download all variables for one month. Returns list of failed variable names."""
    if is_future_month(year, month):
        print(f"[SKIP]  {year}-{month:02d} is in the future.")
        return []

    month_dir = outdir / f"{year}-{month:02d}"
    month_dir.mkdir(parents=True, exist_ok=True)

    days = build_days(year, month)

    failed = []
    for var_short, var_long in VARIABLES.items():
        try:
            download_variable(client, year, month, var_short, var_long, month_dir, days)
        except Exception as e:
            print(f"[FAIL]  {year}-{month:02d}  {var_short}: {e}")
            failed.append(f"{year}-{month:02d}/{var_short}")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5-Land reanalysis for Mekong Basin.")
    parser.add_argument("--start-year",  type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=START_MONTH)
    parser.add_argument("--end-year",    type=int, default=END_YEAR)
    parser.add_argument("--end-month",   type=int, default=END_MONTH)
    parser.add_argument("--outdir",      type=Path, default=OUTDIR,
                        help="Output root directory (default: cluster path in script)")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    months_requested = 0
    all_failed = []
    for year in range(args.start_year, args.end_year + 1):
        m_start = args.start_month if year == args.start_year else 1
        m_end   = args.end_month   if year == args.end_year   else 12
        for month in range(m_start, m_end + 1):
            all_failed.extend(download_month(client, year, month, outdir))
            months_requested += 1

    print(f"\nFinished. {months_requested} month(s) processed → {outdir}")
    if all_failed:
        print(f"Failed ({len(all_failed)}):")
        for f in all_failed:
            print(f"  {f}")
    else:
        print("All variables downloaded successfully.")


if __name__ == "__main__":
    main()
