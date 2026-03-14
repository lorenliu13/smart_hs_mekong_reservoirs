"""
Quick test: download ERA5-Land for 2024-01 to local folder.
Downloads all 8 variables for January 2024.
"""

from pathlib import Path
import cdsapi

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

AREA   = [34, 89, 7, 112]   # N, W, S, E — Mekong Basin
GRID   = [0.1, 0.1]
HOURS  = [f"{h:02d}:00" for h in range(24)]
OUTDIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\era5_land")

YEAR  = 2024
MONTH = 1
DAYS  = [f"{d:02d}" for d in range(1, 32)]  # Jan has 31 days

month_dir = OUTDIR / f"{YEAR}-{MONTH:02d}"
month_dir.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

for var_short, var_long in VARIABLES.items():
    outfile = month_dir / f"era5land_mekong_{YEAR}-{MONTH:02d}_{var_short}.nc"

    if outfile.exists():
        print(f"[SKIP]  {outfile.name} already exists.")
        continue

    print(f"[START] {var_short} ({var_long})")
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable" : var_long,
            "year"     : str(YEAR),
            "month"    : f"{MONTH:02d}",
            "day"      : DAYS,
            "time"     : HOURS,
            "area"     : AREA,
            "grid"     : GRID,
            "format"   : "netcdf",
        },
        str(outfile),
    )
    print(f"[DONE]  Saved → {outfile}")

print("\nTest download complete.")
