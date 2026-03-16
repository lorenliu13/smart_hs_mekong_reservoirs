"""
Extract ECMWF IFS HRES forecast data for lake catchment polygons.
Cluster version — paths point to HPC storage; 10-worker multiprocessing.

For each forecast initialisation date and each lake in the GRIT PLD database:

  * Lakes with a catchment polygon  → area-weighted average of all ECMWF
    grid cells that overlap the polygon (weights = intersection area fraction).
  * Lakes without a catchment       → value at the nearest ECMWF grid cell
    to the lake centroid.

Daily aggregation rules (same as in the exploration notebooks)
--------------------------------------------------------------
  Accumulated vars (tp, ssrd, strd, sf):
      day_d = field[step = d*24 h] − field[step = (d-1)*24 h]

  Instantaneous vars (2t, 2d, sp, 10u, 10v, sd, swvl1-4):
      day_d = mean( steps [(d-1)*24+6, (d-1)*24+12, (d-1)*24+18, d*24] )

Inputs
------
  ECMWF GRIB2 files  : <DATA_DIR>/<YYYY-MM>/hres_mekong_<YYYY-MM>_<var>.grib2
  Catchment shapefile: gritv06_pld_lake_catchments_0sqkm.shp
                       Columns: lake_id, geometry
  Lake centroid CSV  : Optional; required only for lakes absent from the
                       catchment shapefile.
                       Columns: lake_id, lon, lat

Outputs
-------
  Variables are processed one at a time.  One CSV per variable per
  initialisation date, organised into per-variable sub-directories:

      <OUTPUT_DIR>/<var>/ecmwf_per_lake_<var>_<YYYY-MM-DD>.csv

  Columns:
      lake_id, init_date, forecast_day, valid_date,
      <col>   (native ECMWF unit — see VAR_META below),
      extraction_method   (area_weighted | centroid)

Usage
-----
  Submitted via sbatch — see submit_ecmwf_extraction.sbatch.
  Can also be run interactively:
      python extract_ecmwf_per_catchment_cluster.py

Dependencies
------------
  numpy, pandas, geopandas, xarray, cfgrib, shapely
  (cartopy / matplotlib NOT required — data-only script)
"""

import contextlib
import os
import pickle
import warnings
from pathlib import Path
from multiprocessing import Pool

import cfgrib
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

warnings.filterwarnings("ignore")

LAKE_AREA_THRESHOLD_SQKM = 0

# ---------------------------------------------------------------------------
# Configuration — HPC cluster paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"/data/ouce-grit/cenv1160/smart_hs/raw_data/ecmwf_ifs/hres")
CATCHMENT_SHP = (
    r"/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    rf"/gritv06_pld_lake_catchments_{LAKE_AREA_THRESHOLD_SQKM}sqkm.shp"
)
LAKE_CENTROIDS_CSV = (
    r"/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    rf"/gritv06_pld_lake_upstream_segments_{LAKE_AREA_THRESHOLD_SQKM}sqkm.csv"
)
OUTPUT_DIR = Path(rf"/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/ecmwf_ifs/hres/ecmwf_ifs_per_pld_lake_{LAKE_AREA_THRESHOLD_SQKM}sqkm")
WEIGHTS_CACHE = ""   # path to a .pkl file to cache/load spatial weights; "" = no cache

START_YEAR  = 2023
START_MONTH = 1
END_YEAR    = 2025
END_MONTH   = 12
N_DAYS      = 10     # number of forecast days to extract per init date
OVERWRITE   = False  # set True to reprocess init dates that already have a CSV
N_WORKERS   = 20     # number of parallel worker processes (match --cpus-per-task in sbatch)

# ---------------------------------------------------------------------------
# Variable metadata
# ---------------------------------------------------------------------------
VAR_META = {
    "tp"    : ("tp",    "accum", "m"),
    "2t"    : ("t2m",   "inst",  "K"),
    "2d"    : ("d2m",   "inst",  "K"),
    "sp"    : ("sp",    "inst",  "Pa"),
    "10u"   : ("u10",   "inst",  "m/s"),
    "10v"   : ("v10",   "inst",  "m/s"),
    "ssrd"  : ("ssrd",  "accum", "J/m2"),
    "strd"  : ("strd",  "accum", "J/m2"),
    "sf"    : ("sf",    "accum", "m"),
    "sd"    : ("sd",    "inst",  "m"),
    "swvl1" : ("swvl1", "inst",  "m3/m3"),
    "swvl2" : ("swvl2", "inst",  "m3/m3"),
    "swvl3" : ("swvl3", "inst",  "m3/m3"),
    "swvl4" : ("swvl4", "inst",  "m3/m3"),
}

ACCUM_VARS = {v for v, meta in VAR_META.items() if meta[1] == "accum"}
INST_VARS  = {v for v, meta in VAR_META.items() if meta[1] == "inst"}

# ---------------------------------------------------------------------------
# ECMWF grid parameters  (must match download settings)
# ---------------------------------------------------------------------------
GRID_RES = 0.1
EQUAL_AREA_CRS = "ESRI:54034"


# ===========================================================================
# I.  GRIB loading helpers
# ===========================================================================

def grib_file_path(data_dir: Path, year: int, month: int, var: str) -> Path:
    return data_dir / f"{year}-{month:02d}" / f"hres_mekong_{year}-{month:02d}_{var}.grib2"


def step_to_hours(da: xr.DataArray) -> np.ndarray:
    return (da.step.values / np.timedelta64(1, "h")).astype(int)


@contextlib.contextmanager
def _suppress_fd2():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved   = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def load_grib_variable(grib_path: Path, var: str) -> xr.DataArray | None:
    if not grib_path.exists():
        print(f"  [MISS] {grib_path.name}")
        return None
    try:
        with _suppress_fd2():
            ds_list = cfgrib.open_datasets(str(grib_path))
        ds = ds_list[0]
        grib_key = list(ds.data_vars)[0]
        da = ds[grib_key]
        if "time" not in da.dims:
            da = da.expand_dims("time")
        return da
    except Exception as exc:
        print(f"  [FAIL] {grib_path.name}: {exc}")
        return None


# ===========================================================================
# II.  Daily aggregation
# ===========================================================================

def compute_daily_fields(
    da: xr.DataArray,
    var: str,
    t_idx: int,
    n_days: int = 10,
) -> np.ndarray:
    da_init  = da.isel(time=t_idx)
    hrs      = step_to_hours(da)
    hrs_set  = set(hrs.tolist())
    h_to_idx = {int(h): i for i, h in enumerate(hrs)}
    days     = []

    for d in range(1, n_days + 1):
        h_start = (d - 1) * 24
        h_end   = d * 24

        if var in ACCUM_VARS:
            i0 = h_to_idx[h_start]
            i1 = h_to_idx[h_end]
            daily_val = da_init.isel(step=i1).values - da_init.isel(step=i0).values
        else:
            intra_hours = [h for h in [h_start + 6, h_start + 12, h_start + 18, h_end]
                           if h in hrs_set]
            if not intra_hours:
                raise ValueError(
                    f"No intra-day steps found for var={var}, day={d}, h_start={h_start}"
                )
            indices   = [h_to_idx[h] for h in intra_hours]
            slices    = np.stack([da_init.isel(step=i).values for i in indices], axis=0)
            daily_val = slices.mean(axis=0)

        days.append(daily_val)

    return np.stack(days, axis=0)


# ===========================================================================
# III.  Spatial weights computation
# ===========================================================================

def _build_ecmwf_grid_polygons(lats, lons, res=GRID_RES):
    half = res / 2.0
    rows = []
    for lat in lats:
        for lon in lons:
            rows.append({"lat": lat, "lon": lon,
                         "geometry": box(lon - half, lat - half, lon + half, lat + half)})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf["flat_idx"] = np.arange(len(gdf), dtype=np.int32)
    return gdf


def compute_spatial_weights(catchments_gdf, lats, lons, lake_centroids=None):
    n_lons = len(lons)
    grid_gdf = _build_ecmwf_grid_polygons(lats, lons)

    catchments_proj = catchments_gdf.to_crs(EQUAL_AREA_CRS)
    grid_proj       = grid_gdf.to_crs(EQUAL_AREA_CRS)

    weights_dict = {}
    n_catch = len(catchments_proj)
    print(f"  Computing area-weighted weights for {n_catch} catchment lakes …")

    for row_i, lake_row in catchments_proj.iterrows():
        lake_id   = int(lake_row["lake_id"])
        lake_geom = lake_row.geometry

        possible_mask = grid_proj.intersects(lake_geom.envelope)
        possible      = grid_proj[possible_mask]

        flat_indices = []
        areas        = []

        for _, grid_row in possible.iterrows():
            intersection = lake_geom.intersection(grid_row.geometry)
            if not intersection.is_empty and intersection.area > 0:
                flat_indices.append(int(grid_row["flat_idx"]))
                areas.append(intersection.area)

        if not flat_indices:
            cx, cy = lake_geom.centroid.x, lake_geom.centroid.y
            cent_gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy([cx], [cy]), crs=EQUAL_AREA_CRS
            ).to_crs("EPSG:4326")
            lon_c = float(cent_gdf.geometry.x.iloc[0])
            lat_c = float(cent_gdf.geometry.y.iloc[0])
            i_lat = int(np.argmin(np.abs(lats - lat_c)))
            i_lon = int(np.argmin(np.abs(lons - lon_c)))
            flat_indices = [i_lat * n_lons + i_lon]
            areas        = [1.0]

        total = sum(areas)
        weights_dict[lake_id] = {
            "method"   : "area_weighted",
            "flat_idx" : np.array(flat_indices, dtype=np.int32),
            "weights"  : np.array([a / total for a in areas], dtype=np.float32),
        }

        if (row_i + 1) % 50 == 0:
            print(f"    … {row_i + 1}/{n_catch} done")

    catchment_lake_ids = set(catchments_gdf["lake_id"].astype(int))

    if lake_centroids:
        no_catch = {k: v for k, v in lake_centroids.items()
                    if int(k) not in catchment_lake_ids}
        if no_catch:
            print(f"  Using centroid extraction for {len(no_catch)} lakes "
                  f"without catchment polygons …")
        for lake_id, (lon_c, lat_c) in no_catch.items():
            i_lat = int(np.argmin(np.abs(lats - lat_c)))
            i_lon = int(np.argmin(np.abs(lons - lon_c)))
            weights_dict[int(lake_id)] = {
                "method"   : "centroid",
                "flat_idx" : np.array([i_lat * n_lons + i_lon], dtype=np.int32),
                "weights"  : np.array([1.0], dtype=np.float32),
            }

    print(f"  Spatial weights ready for {len(weights_dict)} lakes total.")
    return weights_dict


# ===========================================================================
# IV.  Fast per-lake extraction
# ===========================================================================

def extract_lake_values(daily_grid, weights_dict):
    n_days    = daily_grid.shape[0]
    grid_flat = daily_grid.reshape(n_days, -1)
    lake_values = {}
    for lake_id, w in weights_dict.items():
        idx  = w["flat_idx"]
        wts  = w["weights"]
        vals = (grid_flat[:, idx] * wts[np.newaxis, :]).sum(axis=1)
        lake_values[lake_id] = vals
    return lake_values


# ===========================================================================
# V.  Loading helpers
# ===========================================================================

def load_lake_centroids(csv_path: str) -> dict:
    if not csv_path or not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path)
    return {int(row.lake_id): (float(row.lon), float(row.lat))
            for row in df.itertuples()}


def load_or_compute_weights(weights_cache, catchments_gdf, lats, lons, lake_centroids):
    cache_path = Path(weights_cache) if weights_cache else None

    if cache_path and cache_path.exists():
        print(f"Loading spatial weights from cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    print("Computing spatial weights …")
    weights = compute_spatial_weights(catchments_gdf, lats, lons, lake_centroids)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(weights, fh)
        print(f"Weights cached to: {cache_path}")

    return weights


# ===========================================================================
# VI.  Main processing loop
# ===========================================================================

def process_variable_month(
    data_dir, year, month, var, col_name, weights_dict, n_days, output_dir, overwrite=False
):
    fpath = grib_file_path(data_dir, year, month, var)
    da    = load_grib_variable(fpath, var)
    if da is None:
        return

    init_times = np.atleast_1d(da.time.values)
    n_init     = len(init_times)
    hrs        = step_to_hours(da)
    n_days_use = min(n_days, int(hrs.max()) // 24)

    if n_days_use < n_days:
        print(f"    Warning: only {n_days_use} forecast days available "
              f"(requested {n_days}).")

    var_dir = output_dir / col_name
    var_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    n_skip  = 0

    for t_idx in range(n_init):
        init_date = pd.Timestamp(init_times[t_idx]).normalize()
        init_str  = init_date.strftime("%Y-%m-%d")
        out_csv   = var_dir / f"ecmwf_per_lake_{col_name}_{init_str}.csv"

        if out_csv.exists() and not overwrite:
            n_skip += 1
            continue

        valid_dates = pd.date_range(
            start=init_date + pd.Timedelta(days=1),
            periods=n_days_use, freq="1D",
        )

        try:
            daily_grid = compute_daily_fields(da, var, t_idx, n_days_use)
        except Exception as exc:
            print(f"    Warning: {init_str}: {exc}")
            continue

        lake_vals = extract_lake_values(daily_grid, weights_dict)

        records = []
        for lake_id in sorted(weights_dict.keys()):
            method = weights_dict[lake_id]["method"]
            vals   = lake_vals.get(lake_id, np.full(n_days_use, np.nan))
            for d in range(n_days_use):
                records.append({
                    "lake_id"          : lake_id,
                    "init_date"        : init_str,
                    "forecast_day"     : d + 1,
                    "valid_date"       : valid_dates[d].strftime("%Y-%m-%d"),
                    col_name           : float(vals[d]),
                    "extraction_method": method,
                })

        pd.DataFrame(records).to_csv(out_csv, index=False)
        n_saved += 1

    msg = f"  {var:6s} ({col_name})  {year}-{month:02d}:"
    msg += f"  {n_saved} init date(s) saved"
    if n_skip:
        msg += f",  {n_skip} skipped (already exist)"
    print(msg)


def _worker(args: tuple) -> None:
    process_variable_month(*args)


# ===========================================================================
# VII.  Entry point
# ===========================================================================

def main() -> None:
    print(f"Loading catchment polygons: {CATCHMENT_SHP}")
    catchments_gdf = gpd.read_file(CATCHMENT_SHP)
    catchments_gdf["lake_id"] = catchments_gdf["lake_id"].astype(int)
    print(f"  {len(catchments_gdf)} catchment polygons  (CRS: {catchments_gdf.crs})")

    lake_centroids = load_lake_centroids(LAKE_CENTROIDS_CSV)
    if lake_centroids:
        print(f"  Loaded {len(lake_centroids)} lake centroids from CSV.")

    print("\nDetecting ECMWF grid from first available GRIB2 file …")
    ref_da = None
    for year in range(START_YEAR, END_YEAR + 1):
        m_start = START_MONTH if year == START_YEAR else 1
        m_end   = END_MONTH   if year == END_YEAR   else 12
        for month in range(m_start, m_end + 1):
            for var in VAR_META:
                fpath = grib_file_path(DATA_DIR, year, month, var)
                if fpath.exists():
                    ref_da = load_grib_variable(fpath, var)
                    if ref_da is not None:
                        break
            if ref_da is not None:
                break
        if ref_da is not None:
            break

    if ref_da is None:
        raise FileNotFoundError(
            f"No GRIB2 files found under {DATA_DIR} for the requested "
            f"period {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}."
        )

    lats = ref_da.latitude.values
    lons = ref_da.longitude.values
    print(f"  Grid: lat {lats[-1]:.1f}–{lats[0]:.1f}°N ({len(lats)} pts), "
          f"lon {lons[0]:.1f}–{lons[-1]:.1f}°E ({len(lons)} pts)")

    weights_dict = load_or_compute_weights(
        WEIGHTS_CACHE, catchments_gdf, lats, lons, lake_centroids
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for var, meta in VAR_META.items():
        col_name = meta[0]
        for year in range(START_YEAR, END_YEAR + 1):
            m_start = START_MONTH if year == START_YEAR else 1
            m_end   = END_MONTH   if year == END_YEAR   else 12
            for month in range(m_start, m_end + 1):
                tasks.append((
                    DATA_DIR, year, month, var, col_name,
                    weights_dict, N_DAYS, OUTPUT_DIR, OVERWRITE,
                ))

    print(f"\nDispatching {len(tasks)} task(s) across {N_WORKERS} worker(s) …")

    with Pool(processes=N_WORKERS) as pool:
        pool.map(_worker, tasks)

    print(f"\nDone. Output CSVs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
