"""
Aggregate ERA5-Land hourly ZIP-NetCDF files to daily gridded NetCDF.

For each monthly ZIP-NetCDF file, applies the appropriate daily aggregation
and writes one NetCDF file per variable per month:

    era5land_mekong_<YYYY-MM>_<var>_daily.nc

Output dimensions:  time × latitude × longitude

Coordinates
-----------
  time      (time)      datetime64   calendar dates (midnight UTC)
  latitude  (latitude)  float64
  longitude (longitude) float64

Aggregation rules
-----------------
  Accumulated vars (tp, ssrd, strd):
      ERA5-Land stores running accumulations that reset each midnight UTC.
      valid_time 00:00 UTC on day D = 24-hour cumulative total for day D-1.
      Only the midnight (00:00 UTC) value is used for each day; it is
      attributed to the preceding calendar day (D-1).

      Note: the file boundary may cause the last day of a month to be
      missing (its midnight value is in the following month's file) or the
      first midnight to belong to the preceding month (discarded silently).

  Instantaneous vars (2t, 2d, sp, 10u, 10v, sd, swvl1, swvl2, swvl3, swvl4):
      Grouped by calendar date as-is and averaged over all hourly snapshots.

Usage
-----
  python aggregate_era5land_to_daily.py

Dependencies
------------
  numpy, pandas, xarray, h5py, zipfile
"""

import io
import warnings
import zipfile
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\era5_land")
OUTPUT_DIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\era5_land_daily")

START_YEAR  = 2024
START_MONTH = 1
END_YEAR    = 2024
END_MONTH   = 1
OVERWRITE   = True
N_WORKERS   = 8

# ---------------------------------------------------------------------------
# Variable metadata:  short_name → (col_name, agg_type, units)
# ---------------------------------------------------------------------------
VAR_META = {
    "tp"   : ("tp",   "accum", "m"),
    "2t"   : ("t2m",  "inst",  "K"),
    "2d"   : ("d2m",  "inst",  "K"),
    "sp"   : ("sp",   "inst",  "Pa"),
    "10u"  : ("u10",  "inst",  "m/s"),
    "10v"  : ("v10",  "inst",  "m/s"),
    "ssrd" : ("ssrd",  "accum", "J m-2"),
    "strd" : ("strd",  "accum", "J m-2"),
    "sf"   : ("sf",    "accum", "m"),        # snowfall (accumulated)
    "sd"   : ("sd",    "inst",  "m"),        # snow depth
    "swvl1": ("swvl1", "inst",  "m3 m-3"),  # volumetric soil water layer 1
    "swvl2": ("swvl2", "inst",  "m3 m-3"),  # volumetric soil water layer 2
    "swvl3": ("swvl3", "inst",  "m3 m-3"),  # volumetric soil water layer 3
    "swvl4": ("swvl4", "inst",  "m3 m-3"),  # volumetric soil water layer 4
}

ACCUM_VARS = {v for v, meta in VAR_META.items() if meta[1] == "accum"}

LONG_NAMES = {
    "tp"    : "Total precipitation",
    "t2m"   : "2-metre temperature",
    "d2m"   : "2-metre dewpoint temperature",
    "sp"    : "Surface pressure",
    "u10"   : "10-metre U wind component",
    "v10"   : "10-metre V wind component",
    "ssrd"  : "Surface solar radiation downwards",
    "strd"  : "Surface thermal radiation downwards",
    "sf"    : "Snowfall",
    "sd"    : "Snow depth",
    "swvl1" : "Volumetric soil water layer 1",
    "swvl2" : "Volumetric soil water layer 2",
    "swvl3" : "Volumetric soil water layer 3",
    "swvl4" : "Volumetric soil water layer 4",
}


# ===========================================================================
# I.  NetCDF loading helper
# ===========================================================================

def nc_file_path(data_dir: Path, year: int, month: int, var: str) -> Path:
    return data_dir / f"{year}-{month:02d}" / f"era5land_mekong_{year}-{month:02d}_{var}.nc"


def load_era5land_variable(
    nc_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None:
    """
    Load one variable from an ERA5-Land ZIP-wrapped NetCDF4 file.

    Returns
    -------
    (data, valid_times, lats, lons, var_name) where
        data        : float32 ndarray  (n_times, n_lat, n_lon)
        valid_times : int64 ndarray    Unix timestamps (seconds since 1970-01-01)
        lats        : float64 ndarray  decreasing
        lons        : float64 ndarray  increasing
        var_name    : str              variable key found in the file

    Returns None if the file does not exist or cannot be read.
    """
    if not nc_path.exists():
        print(f"  [MISS] {nc_path.name}")
        return None
    try:
        with zipfile.ZipFile(nc_path) as zf:
            raw = zf.open(zf.namelist()[0]).read()
        with h5py.File(io.BytesIO(raw), "r") as hf:
            coord_keys = {"latitude", "longitude", "valid_time", "expver", "number"}
            data_keys  = [k for k in hf.keys() if k not in coord_keys]
            if not data_keys:
                raise ValueError(f"No data variable found in {nc_path.name}")
            var_name    = data_keys[0]
            data        = hf[var_name][:]
            valid_times = hf["valid_time"][:]
            lats        = hf["latitude"][:]
            lons        = hf["longitude"][:]
        return data, valid_times, lats, lons, var_name
    except Exception as exc:
        print(f"  [FAIL] {nc_path.name}: {exc}")
        return None


# ===========================================================================
# II.  Daily aggregation
# ===========================================================================

def compute_daily_fields(
    data: np.ndarray,
    valid_times: np.ndarray,
    var: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate hourly ERA5-Land data to daily values.

    Parameters
    ----------
    data        : float32 ndarray  (n_hours, n_lat, n_lon)
    valid_times : int64 ndarray    Unix timestamps (1-hourly)
    var         : ERA5-Land variable short name

    Returns
    -------
    daily_data   : float32 ndarray  (n_days, n_lat, n_lon)
    date_strings : list of "YYYY-MM-DD" strings, one per day
    """
    times_dt = pd.to_datetime(valid_times, unit="s", utc=True)

    if var in ACCUM_VARS:
        # ERA5-Land accumulated vars are running totals that reset at 00:00 UTC.
        # The value at valid_time 00:00 UTC on day D is the 24-hour cumulative
        # total for the preceding day (D-1).  Use only midnight values.
        midnight_mask = np.array([t.hour == 0 for t in times_dt])
        dates         = np.array([
            (t - pd.Timedelta(days=1)).date() for t in times_dt[midnight_mask]
        ])
        data_sel      = data[midnight_mask]
    else:
        dates    = np.array([t.date() for t in times_dt])
        data_sel = data

    unique_dates = sorted(set(dates))
    daily_layers = []
    date_strings = []

    for d in unique_dates:
        mask = dates == d
        if var in ACCUM_VARS:
            # Exactly one midnight value per day; squeeze to (n_lat, n_lon)
            daily_val = data_sel[mask].squeeze(axis=0)
        else:
            daily_val = data_sel[mask].mean(axis=0)
        daily_layers.append(daily_val)
        date_strings.append(str(d))

    return np.stack(daily_layers, axis=0).astype(np.float32), date_strings


# ===========================================================================
# III.  Write daily NetCDF
# ===========================================================================

def process_variable_month(
    data_dir: Path,
    year: int,
    month: int,
    var: str,
    col_name: str,
    units: str,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Aggregate one ERA5-Land variable for one month to a daily NetCDF file.

    Output path:
        <output_dir>/<YYYY-MM>/era5land_mekong_<YYYY-MM>_<col_name>_daily.nc
    """
    month_str = f"{year}-{month:02d}"
    out_dir   = output_dir / month_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc    = out_dir / f"era5land_mekong_{month_str}_{col_name}_daily.nc"

    if out_nc.exists() and not overwrite:
        print(f"  {col_name:6s}  {month_str}: skipped (already exists)")
        return

    nc_path = nc_file_path(data_dir, year, month, var)
    result  = load_era5land_variable(nc_path)
    if result is None:
        return
    data, valid_times, lats, lons, _var_name = result

    daily_data, date_strings = compute_daily_fields(data, valid_times, var)

    times_dt64 = pd.to_datetime(date_strings).values.astype("datetime64[ns]")

    ds = xr.Dataset(
        {
            col_name: xr.DataArray(
                daily_data,
                dims=["time", "latitude", "longitude"],
                attrs={
                    "units"      : units,
                    "long_name"  : LONG_NAMES.get(col_name, col_name),
                    "aggregation": (
                        "24-hour cumulative total at 00:00 UTC, attributed to preceding calendar day"
                        if var in ACCUM_VARS else
                        "mean of hourly snapshots per calendar day"
                    ),
                },
            ),
        },
        coords={
            "time"     : ("time",      times_dt64),
            "latitude" : ("latitude",  lats),
            "longitude": ("longitude", lons),
        },
    )
    ds.attrs = {
        "source"     : f"ERA5-Land  {nc_path.name}",
        "aggregation": "1-hourly ERA5-Land → daily",
        "timestamp_convention": (
            "Accumulated vars: ERA5-Land running accumulation resets at 00:00 UTC each day. "
            "valid_time 00:00 UTC on day D = 24-hour total for day D-1. "
            "Only midnight values are used and attributed to the preceding calendar day."
        ),
        "created_by" : "aggregate_era5land_to_daily.py",
    }

    ds.to_netcdf(out_nc)
    print(
        f"  {col_name:6s}  {month_str}: "
        f"{len(date_strings)} day(s) → {out_nc.name}"
    )


def _worker(args: tuple) -> None:
    process_variable_month(*args)


# ===========================================================================
# IV.  Entry point
# ===========================================================================

def main() -> None:
    tasks = []
    for var, (col_name, _agg, units) in VAR_META.items():
        for year in range(START_YEAR, END_YEAR + 1):
            m_start = START_MONTH if year == START_YEAR else 1
            m_end   = END_MONTH   if year == END_YEAR   else 12
            for month in range(m_start, m_end + 1):
                tasks.append((
                    DATA_DIR, year, month, var, col_name, units,
                    OUTPUT_DIR, OVERWRITE,
                ))

    print(f"Dispatching {len(tasks)} task(s) across {N_WORKERS} worker(s) …")
    print(f"Output root: {OUTPUT_DIR}\n")

    with Pool(processes=N_WORKERS) as pool:
        pool.map(_worker, tasks)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
