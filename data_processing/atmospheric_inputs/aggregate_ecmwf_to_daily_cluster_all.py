"""
Aggregate ECMWF IFS HRES 6-hourly GRIB2 files to daily gridded NetCDF.
For data from 2026 onward, all variables are stored in a single combined
GRIB2 file per month:

    hres_mekong_<YYYY-MM>_all.grib2

Each variable is extracted individually, aggregated to daily values, and
written as a separate NetCDF file per variable per month:

    hres_mekong_<YYYY-MM>_<var>_daily.nc

Output dimensions:  init_time × forecast_day × latitude × longitude

Coordinates
-----------
  init_time    (init_time)               datetime64   forecast init dates
  forecast_day (forecast_day)            int          1 … N_DAYS
  valid_time   (init_time, forecast_day) datetime64   calendar date the value represents
  latitude     (latitude)                float64
  longitude    (longitude)               float64

Aggregation rules
-----------------
  Accumulated vars (tp, ssrd, strd, sf):
      day_d = field[step = d*24 h] − field[step = (d-1)*24 h]

  Instantaneous vars (2t, 2d, sp, 10u, 10v, sd, swvl1, swvl2, swvl3, swvl4):
      day_d = mean(steps [(d-1)*24+6, (d-1)*24+12, (d-1)*24+18, d*24])

Valid-date convention
---------------------
  For both aggregation types, day d covers the 24-hour window
      init_date + (d-1)*24 h  →  init_date + d*24 h
  which corresponds to calendar date  valid_date = init_date + (d-1) days.

  Example (init_date = 2026-01-14):
      d=1: step 0→24 h  →  rain on 2026-01-14  →  valid_date 2026-01-14
      d=2: step 24→48 h →  rain on 2026-01-15  →  valid_date 2026-01-15

Usage
-----
  python aggregate_ecmwf_to_daily_cluster_allinone.py

Dependencies
------------
  numpy, pandas, xarray, cfgrib
"""

import contextlib
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

# Suppress noisy warnings from cfgrib/eccodes and xarray during GRIB loading
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = Path(r"/data/ouce-grit/cenv1160/smart_hs/raw_data/ecmwf_ifs/hres")
OUTPUT_DIR = Path(r"/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/ecmwf_ifs_daily/hres")

START_YEAR  = 2026
START_MONTH = 1
END_YEAR    = 2026
END_MONTH   = 2
N_DAYS      = 10      # forecast days to retain per init date
OVERWRITE   = True
N_WORKERS   = 20      # parallel worker processes (one per variable×month task)

# ---------------------------------------------------------------------------
# Variable metadata:  short_name → (col_name, agg_type, units)
#
# col_name  : output variable name used in the NetCDF file
# agg_type  : "accum" for flux/accumulated variables (differenced between steps)
#             "inst"  for state/instantaneous variables (averaged over steps)
# units     : physical units of the variable as stored in GRIB2
# ---------------------------------------------------------------------------
VAR_META = {
    "tp"   : ("tp",    "accum", "m"),        # total precipitation (accumulated)
    "2t"   : ("t2m",   "inst",  "K"),        # 2-metre air temperature
    "2d"   : ("d2m",   "inst",  "K"),        # 2-metre dewpoint temperature
    "sp"   : ("sp",    "inst",  "Pa"),       # surface pressure
    "10u"  : ("u10",   "inst",  "m/s"),      # 10-metre U (eastward) wind
    "10v"  : ("v10",   "inst",  "m/s"),      # 10-metre V (northward) wind
    "ssrd" : ("ssrd",  "accum", "J m-2"),    # surface solar radiation downwards (accumulated)
    "strd" : ("strd",  "accum", "J m-2"),    # surface thermal radiation downwards (accumulated)
    "sf"   : ("sf",    "accum", "m"),        # snowfall (accumulated)
    "sd"   : ("sd",    "inst",  "m"),        # snow depth
    "swvl1": ("swvl1", "inst",  "m3 m-3"),  # volumetric soil water layer 1
    "swvl2": ("swvl2", "inst",  "m3 m-3"),  # volumetric soil water layer 2
    "swvl3": ("swvl3", "inst",  "m3 m-3"),  # volumetric soil water layer 3
    "swvl4": ("swvl4", "inst",  "m3 m-3"),  # volumetric soil water layer 4
}

# Derived sets for quick membership tests during aggregation
ACCUM_VARS = {v for v, meta in VAR_META.items() if meta[1] == "accum"}
INST_VARS  = {v for v, meta in VAR_META.items() if meta[1] == "inst"}

LONG_NAMES = {
    "tp"   : "Total precipitation",
    "t2m"  : "2-metre temperature",
    "d2m"  : "2-metre dewpoint temperature",
    "sp"   : "Surface pressure",
    "u10"  : "10-metre U wind component",
    "v10"  : "10-metre V wind component",
    "ssrd" : "Surface solar radiation downwards",
    "strd" : "Surface thermal radiation downwards",
    "sf"   : "Snowfall",
    "sd"   : "Snow depth",
    "swvl1": "Volumetric soil water layer 1",
    "swvl2": "Volumetric soil water layer 2",
    "swvl3": "Volumetric soil water layer 3",
    "swvl4": "Volumetric soil water layer 4",
}


# ===========================================================================
# I.  GRIB loading helpers
# ===========================================================================

@contextlib.contextmanager
def _suppress_fd2():
    """Redirect fd 2 to /dev/null to silence eccodes C-library stderr noise.

    cfgrib calls into the eccodes C library, which writes diagnostic messages
    directly to file descriptor 2 (the OS-level stderr), bypassing Python's
    warnings/logging system.  This context manager saves fd 2, points it at
    /dev/null for the duration of the block, then restores the original fd.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved   = os.dup(2)           # keep a copy of the real stderr fd
    os.dup2(devnull, 2)           # replace fd 2 with /dev/null
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)         # restore the original stderr fd
        os.close(saved)


def grib_file_path(data_dir: Path, year: int, month: int) -> Path:
    """Return the path to the combined all-variables GRIB2 file for a given month."""
    return data_dir / f"{year}-{month:02d}" / f"hres_mekong_{year}-{month:02d}_all.grib2"


def load_grib_variable(grib_path: Path, var: str) -> xr.DataArray | None:
    """
    Load a single variable from a combined all-variables GRIB2 file.

    Uses cfgrib's filter_by_keys to read only the GRIB messages matching the
    requested ECMWF short name, avoiding loading all variables into memory.

    Parameters
    ----------
    grib_path : path to the combined _all.grib2 file
    var       : ECMWF short name to extract (e.g. "tp", "2t", "swvl1")

    Returns an xr.DataArray with dims (time, step, latitude, longitude),
    or None if the file does not exist, cannot be read, or the variable is
    not found inside the file.
    """
    if not grib_path.exists():
        print(f"  [MISS] {grib_path.name}")
        return None
    try:
        with _suppress_fd2():
            ds_list = cfgrib.open_datasets(
                str(grib_path),
                filter_by_keys={"shortName": var},
            )
        if not ds_list:
            print(f"  [EMPTY] {grib_path.name} (var={var})")
            return None
        ds  = ds_list[0]
        key = list(ds.data_vars)[0]   # the filtered dataset contains only the requested variable
        da  = ds[key]
        # Files with a single init date may lack a 'time' dimension; add it so
        # downstream code can always index with da.isel(time=t_idx)
        if "time" not in da.dims:
            da = da.expand_dims("time")
        return da
    except Exception as exc:
        print(f"  [FAIL] {grib_path.name} (var={var}): {exc}")
        return None


# ===========================================================================
# II.  Daily aggregation
# ===========================================================================

def _step_hours(da: xr.DataArray) -> np.ndarray:
    """Convert the 'step' coordinate (timedelta64) to integer hours.

    GRIB2 step values are stored as timedelta64 by cfgrib (e.g. 6 h, 12 h …).
    Dividing by np.timedelta64(1, 'h') yields plain floats; casting to int
    gives clean hour values suitable for dict-based lookup.
    """
    return (da.step.values / np.timedelta64(1, "h")).astype(int)


def compute_daily_fields(
    da: xr.DataArray,
    var: str,
    t_idx: int,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate 6-hourly GRIB data to daily fields for one init date.

    Parameters
    ----------
    da     : DataArray (time, step, latitude, longitude)
    var    : ECMWF short name (key in VAR_META)
    t_idx  : index along the 'time' dimension for the desired init date
    n_days : number of forecast days to compute

    Returns
    -------
    daily_data  : np.ndarray  shape (n_days, n_lat, n_lon)
    valid_dates : np.ndarray of datetime64[D]  shape (n_days,)
                  valid_date[d] = init_date + d*day  (0-based offset)
                  i.e.  d=0 → init_date,  d=1 → init_date+1, …
    """
    da_init   = da.isel(time=t_idx)           # slice to single init date → (step, lat, lon)
    hrs       = _step_hours(da)              # array of step hours: [0, 6, 12, 18, 24, …]
    h_to_idx  = {int(h): i for i, h in enumerate(hrs)}  # hour → positional index in 'step'
    hrs_set   = set(h_to_idx)               # for fast membership checks
    init_date = pd.Timestamp(da.time.values[t_idx]).normalize()  # midnight of init date

    daily_layers = []
    for d in range(1, n_days + 1):
        # Each forecast day d covers the 24-hour window [h_start, h_end)
        h_start = (d - 1) * 24   # e.g. d=1 → 0 h, d=2 → 24 h
        h_end   = d * 24         # e.g. d=1 → 24 h, d=2 → 48 h

        if var in ACCUM_VARS:
            # Accumulated fields (e.g. precipitation) store running totals
            # from the forecast start.  The amount for day d is the increment
            # between the cumulative value at h_end and h_start.
            i0  = h_to_idx[h_start]
            i1  = h_to_idx[h_end]
            val = da_init.isel(step=i1).values - da_init.isel(step=i0).values
        else:
            # Instantaneous fields (e.g. temperature) are averaged over the
            # four 6-hourly sub-steps that fall within the day window:
            # T+6, T+12, T+18, T+24 (relative to h_start).
            # Any missing sub-steps are silently skipped.
            intra = [h for h in [h_start + 6, h_start + 12, h_start + 18, h_end]
                     if h in hrs_set]
            if not intra:
                raise ValueError(
                    f"No intra-day steps for var={var}, day={d}, h_start={h_start}"
                )
            slices = np.stack([da_init.isel(step=h_to_idx[h]).values for h in intra])
            val    = slices.mean(axis=0)   # simple arithmetic mean over sub-steps

        daily_layers.append(val)

    # Assign calendar dates to each forecast day.
    # Day d (1-based) covers the window starting at init_date + (d-1)*24 h,
    # so its valid calendar date is init_date + (d-1) days.
    #   d=1 → period T+0…T+24  → calendar date = init_date
    #   d=2 → period T+24…T+48 → calendar date = init_date + 1
    valid_dates = np.array(
        [np.datetime64(init_date + pd.Timedelta(days=d - 1), "D") for d in range(1, n_days + 1)],
        dtype="datetime64[D]",
    )
    return np.stack(daily_layers, axis=0).astype(np.float32), valid_dates


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
    n_days: int,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Extract one variable from the combined _all.grib2 file, aggregate to
    daily values, and write a per-variable NetCDF file.

    Input:
        <data_dir>/<YYYY-MM>/hres_mekong_<YYYY-MM>_all.grib2

    Output:
        <output_dir>/<YYYY-MM>/hres_mekong_<YYYY-MM>_<col_name>_daily.nc
    """
    month_str = f"{year}-{month:02d}"
    out_dir   = output_dir / month_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc    = out_dir / f"hres_mekong_{month_str}_{col_name}_daily.nc"

    if out_nc.exists() and not overwrite:
        print(f"  {col_name:6s}  {month_str}: skipped (already exists)")
        return

    grib_path = grib_file_path(data_dir, year, month)
    da = load_grib_variable(grib_path, var)
    if da is None:
        return

    # np.atleast_1d ensures a consistent array even when there is only one init date
    init_times_raw = np.atleast_1d(da.time.values)
    hrs            = _step_hours(da)
    # Cap n_days to what the file actually contains (max step ÷ 24)
    n_days_use     = min(n_days, int(hrs.max()) // 24)

    if n_days_use < n_days:
        print(f"    Warning: only {n_days_use} forecast days available in {grib_path.name}.")

    lats = da.latitude.values
    lons = da.longitude.values

    all_data        = []   # list of (n_days, n_lat, n_lon) arrays, one per init date
    all_valid_times = []   # list of (n_days,) datetime64[D] arrays, one per init date

    # Loop over every forecast initialisation date in the monthly file
    for t_idx in range(len(init_times_raw)):
        try:
            daily_data, valid_dates = compute_daily_fields(da, var, t_idx, n_days_use)
        except Exception as exc:
            init_str = str(pd.Timestamp(init_times_raw[t_idx]).date())
            print(f"    Warning: init {init_str}: {exc}")
            continue
        all_data.append(daily_data)
        all_valid_times.append(valid_dates)

    if not all_data:
        print(f"  {col_name:6s}  {month_str}: no data computed")
        return

    # Stack all init dates into a single array → (n_init, n_days, n_lat, n_lon)
    stacked_data  = np.stack(all_data,        axis=0)   # (n_init, n_days, n_lat, n_lon)
    stacked_valid = np.stack(all_valid_times, axis=0)   # (n_init, n_days)

    # Normalise init times to midnight so they compare cleanly as dates
    init_times_ns = np.array(
        [pd.Timestamp(t).normalize() for t in init_times_raw],
        dtype="datetime64[ns]",
    )
    # forecast_day coordinate: 1-based integer label for each forecast day
    forecast_days = np.arange(1, n_days_use + 1, dtype=np.int32)

    ds = xr.Dataset(
        {
            col_name: xr.DataArray(
                stacked_data,
                dims=["init_time", "forecast_day", "latitude", "longitude"],
                attrs={
                    "units"      : units,
                    "long_name"  : LONG_NAMES.get(col_name, col_name),
                    "aggregation": (
                        "step-difference (field[d*24h] - field[(d-1)*24h])"
                        if var in ACCUM_VARS else
                        "mean of 6-hourly steps [(d-1)*24+6, +12, +18, d*24]"
                    ),
                },
            ),
        },
        coords={
            "init_time"   : ("init_time",    init_times_ns),
            "forecast_day": ("forecast_day", forecast_days),
            "valid_time"  : (
                ["init_time", "forecast_day"],
                stacked_valid.astype("datetime64[ns]"),
                {"long_name": "calendar date the forecast value represents",
                 "note": "valid_time = init_time + (forecast_day - 1) days"},
            ),
            "latitude"    : ("latitude",  lats),
            "longitude"   : ("longitude", lons),
        },
    )
    ds.attrs = {
        "source"     : f"ECMWF IFS HRES  {grib_path.name}",
        "aggregation": "6-hourly GRIB2 → daily",
        "valid_date_convention": (
            "valid_time = init_time + (forecast_day - 1) days; "
            "day 1 covers T+0 … T+24 h = the init calendar date"
        ),
        "created_by" : "aggregate_ecmwf_to_daily_cluster_allinone.py",
    }

    ds.to_netcdf(out_nc)
    print(
        f"  {col_name:6s}  {month_str}: "
        f"{len(all_data)} init date(s) × {n_days_use} days → {out_nc.name}"
    )


def _worker(args: tuple) -> None:
    process_variable_month(*args)


# ===========================================================================
# IV.  Entry point
# ===========================================================================

def main() -> None:
    # Build the full list of (variable × month) tasks to process.
    # Each task reads from the same _all.grib2 file but extracts a different
    # variable and writes an independent output NetCDF.
    tasks = []
    for var, (col_name, _agg, units) in VAR_META.items():
        for year in range(START_YEAR, END_YEAR + 1):
            # Respect partial start/end months at the boundary years
            m_start = START_MONTH if year == START_YEAR else 1
            m_end   = END_MONTH   if year == END_YEAR   else 12
            for month in range(m_start, m_end + 1):
                tasks.append((
                    DATA_DIR, year, month, var, col_name, units,
                    N_DAYS, OUTPUT_DIR, OVERWRITE,
                ))

    print(f"Dispatching {len(tasks)} task(s) across {N_WORKERS} worker(s) …")
    print(f"Input format : hres_mekong_<YYYY-MM>_all.grib2  (all vapythriables combined)")
    print(f"Output root  : {OUTPUT_DIR}\n")

    # Process all tasks in parallel using a multiprocessing pool.
    # Each worker handles one (variable, month) combination independently.
    # Multiple workers may read the same _all.grib2 file concurrently, but
    # each opens it independently (read-only), so there are no conflicts.
    with Pool(processes=N_WORKERS) as pool:
        pool.map(_worker, tasks)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
