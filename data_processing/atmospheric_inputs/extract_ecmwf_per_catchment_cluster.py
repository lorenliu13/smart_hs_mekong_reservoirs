"""
Extract ECMWF IFS HRES forecast data for lake catchment polygons.
Cluster version — paths point to HPC storage; 20-worker multiprocessing.

Reads the daily-aggregated NetCDF files produced by aggregate_ecmwf_to_daily.py
(one file per variable per month) and applies spatial weights to compute a
per-lake time series.

For each forecast initialisation date and each lake in the GRIT PLD database:

  * Lakes with a catchment polygon  → area-weighted average of all ECMWF
    grid cells that overlap the polygon (weights = intersection area fraction).
  * Lakes without a catchment       → value at the nearest ECMWF grid cell
    to the lake centroid.

Input file format (daily NetCDF)
---------------------------------
  <DATA_DIR>/<YYYY-MM>/hres_mekong_<YYYY-MM>_<col_name>_daily.nc
  Dimensions: (init_time, forecast_day, latitude, longitude)
  Variable  : <col_name>  (e.g. "tp", "t2m", …)
  Coordinates:
    init_time    : datetime64  forecast initialisation dates
    forecast_day : int         1 … N_DAYS
    valid_time   : datetime64  (init_time, forecast_day)  calendar date per cell
  Units     : as produced by aggregate_ecmwf_to_daily.py

Outputs
-------
  Variables are processed one at a time.  One CSV per variable per
  month (all init dates combined), organised into per-variable sub-directories:

      <OUTPUT_DIR>/<col_name>/ecmwf_per_lake_<col_name>_<YYYY-MM>.csv

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
  numpy, pandas, geopandas, xarray, shapely
  (cartopy / matplotlib NOT required — data-only script)
"""

import pickle
import warnings
from pathlib import Path
from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

warnings.filterwarnings("ignore")

AREA_THRESHOLD_SQKM = 0.1  # Minimum lake area in square kilometers to retain
OBS_COUNT_THRESHOLD = 20  # Minimum number of daily observations to retain a lake

# ---------------------------------------------------------------------------
# Configuration — HPC cluster paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/ecmwf_ifs_daily/hres")
CATCHMENT_SHP = (
    rf"/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs/lake_graph_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}"
    rf"/gritv06_great_mekong_pld_lake_catchments_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}.shp"
)
LAKE_CENTROIDS_CSV = (
    rf"/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs/lake_graph_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}"
    rf"/gritv06_great_mekong_pld_lake_upstream_segments_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}.csv"
)
OUTPUT_DIR = Path(
    rf"/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs"
    rf"/ecmwf_ifs_daily_catchment_level/hres/daily_per_lake_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}sqkm"
)
WEIGHTS_CACHE = ""   # path to a .pkl file to cache/load spatial weights; "" = no cache

START_YEAR  = 2023
START_MONTH = 1
END_YEAR    = 2026
END_MONTH   = 2
N_DAYS      = 10     # number of forecast days to extract per init date
OVERWRITE   = True  # set True to reprocess init dates that already have a CSV
N_WORKERS   = 20     # number of parallel worker processes (match --cpus-per-task in sbatch)

# ---------------------------------------------------------------------------
# Variable metadata  col_name → unit
# (aggregation is already applied by aggregate_ecmwf_to_daily.py)
# ---------------------------------------------------------------------------
VAR_META = {
    "tp"    : "m",
    "t2m"   : "K",
    "d2m"   : "K",
    "sp"    : "Pa",
    "u10"   : "m/s",
    "v10"   : "m/s",
    "ssrd"  : "J/m2",
    "strd"  : "J/m2",
    "sf"    : "m",
    "sd"    : "m",
    "swvl1" : "m3/m3",
    "swvl2" : "m3/m3",
    "swvl3" : "m3/m3",
    "swvl4" : "m3/m3",
}

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
GRID_RES       = 0.1
EQUAL_AREA_CRS = "ESRI:54034"


# ===========================================================================
# I.  Daily NetCDF loading helper
# ===========================================================================

def daily_nc_file_path(data_dir: Path, year: int, month: int, col_name: str) -> Path:
    return data_dir / f"{year}-{month:02d}" / f"hres_mekong_{year}-{month:02d}_{col_name}_daily.nc"


def load_daily_ecmwf(nc_path: Path) -> xr.Dataset | None:
    if not nc_path.exists():
        print(f"  [MISS] {nc_path.name}")
        return None
    try:
        return xr.open_dataset(nc_path)
    except Exception as exc:
        print(f"  [FAIL] {nc_path.name}: {exc}")
        return None


# ===========================================================================
# II.  Spatial weights computation
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
# III.  Fast per-lake extraction
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
# IV.  Loading helpers
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
# V.  Main processing loop
# ===========================================================================

def process_variable_month(
    data_dir, year, month, col_name, weights_dict, n_days, output_dir, overwrite=False
):
    nc_path = daily_nc_file_path(data_dir, year, month, col_name)
    ds = load_daily_ecmwf(nc_path)
    if ds is None:
        return

    var_dir = output_dir / col_name
    var_dir.mkdir(parents=True, exist_ok=True)

    out_csv = var_dir / f"ecmwf_per_lake_{col_name}_{year}-{month:02d}.csv"
    if out_csv.exists() and not overwrite:
        print(f"  {col_name:6s}  {year}-{month:02d}:  skipped (already exists)")
        ds.close()
        return

    init_times     = ds["init_time"].values       # (n_init,)
    forecast_days  = ds["forecast_day"].values    # (n_days_in_file,)
    valid_time_arr = ds["valid_time"].values       # (n_init, n_days_in_file)
    data_arr       = ds[col_name].values           # (n_init, n_days_in_file, n_lat, n_lon)

    n_init     = len(init_times)
    n_days_use = min(n_days, len(forecast_days))

    if n_days_use < n_days:
        print(f"    Warning: only {n_days_use} forecast days available "
              f"(requested {n_days}).")

    all_records = []

    for t_idx in range(n_init):
        init_date = pd.Timestamp(init_times[t_idx]).normalize()
        init_str  = init_date.strftime("%Y-%m-%d")

        daily_grid  = data_arr[t_idx, :n_days_use]   # (n_days_use, n_lat, n_lon)
        valid_dates = [
            pd.Timestamp(vt).strftime("%Y-%m-%d")
            for vt in valid_time_arr[t_idx, :n_days_use]
        ]

        lake_vals = extract_lake_values(daily_grid, weights_dict)

        for lake_id in sorted(weights_dict.keys()):
            method = weights_dict[lake_id]["method"]
            vals   = lake_vals.get(lake_id, np.full(n_days_use, np.nan))
            for d in range(n_days_use):
                all_records.append({
                    "lake_id"          : lake_id,
                    "init_date"        : init_str,
                    "forecast_day"     : int(forecast_days[d]),
                    "valid_date"       : valid_dates[d],
                    col_name           : float(vals[d]),
                    "extraction_method": method,
                })

    pd.DataFrame(all_records).to_csv(out_csv, index=False)
    ds.close()

    print(f"  {col_name:6s}  {year}-{month:02d}:  {n_init} init date(s) saved → {out_csv.name}")


def _worker(args: tuple) -> None:
    process_variable_month(*args)


# ===========================================================================
# VI.  Entry point
# ===========================================================================

def main() -> None:
    print(f"Loading catchment polygons: {CATCHMENT_SHP}")
    catchments_gdf = gpd.read_file(CATCHMENT_SHP)
    catchments_gdf["lake_id"] = catchments_gdf["lake_id"].astype(int)
    print(f"  {len(catchments_gdf)} catchment polygons  (CRS: {catchments_gdf.crs})")

    lake_centroids = load_lake_centroids(LAKE_CENTROIDS_CSV)
    if lake_centroids:
        print(f"  Loaded {len(lake_centroids)} lake centroids from CSV.")

    print("\nDetecting ECMWF grid from first available daily NetCDF file …")
    ref_ds = None
    for year in range(START_YEAR, END_YEAR + 1):
        m_start = START_MONTH if year == START_YEAR else 1
        m_end   = END_MONTH   if year == END_YEAR   else 12
        for month in range(m_start, m_end + 1):
            for col_name in VAR_META:
                fpath = daily_nc_file_path(DATA_DIR, year, month, col_name)
                if fpath.exists():
                    ref_ds = load_daily_ecmwf(fpath)
                    if ref_ds is not None:
                        break
            if ref_ds is not None:
                break
        if ref_ds is not None:
            break

    if ref_ds is None:
        raise FileNotFoundError(
            f"No daily ECMWF NetCDF files found under {DATA_DIR} for the requested "
            f"period {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}."
        )

    lats = ref_ds["latitude"].values
    lons = ref_ds["longitude"].values
    ref_ds.close()
    print(f"  Grid: lat {lats[-1]:.1f}–{lats[0]:.1f}°N ({len(lats)} pts), "
          f"lon {lons[0]:.1f}–{lons[-1]:.1f}°E ({len(lons)} pts)")

    weights_dict = load_or_compute_weights(
        WEIGHTS_CACHE, catchments_gdf, lats, lons, lake_centroids
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for col_name in VAR_META:
        for year in range(START_YEAR, END_YEAR + 1):
            m_start = START_MONTH if year == START_YEAR else 1
            m_end   = END_MONTH   if year == END_YEAR   else 12
            for month in range(m_start, m_end + 1):
                tasks.append((
                    DATA_DIR, year, month, col_name,
                    weights_dict, N_DAYS, OUTPUT_DIR, OVERWRITE,
                ))

    print(f"\nDispatching {len(tasks)} task(s) across {N_WORKERS} worker(s) …")

    with Pool(processes=N_WORKERS) as pool:
        pool.map(_worker, tasks)

    print(f"\nDone. Output CSVs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
