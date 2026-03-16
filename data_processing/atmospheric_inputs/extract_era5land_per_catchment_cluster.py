"""
Extract ERA5-Land reanalysis data for lake catchment polygons.

For each calendar month and each lake in the GRIT PLD database:

  * Lakes with a catchment polygon  → area-weighted average of all ERA5-Land
    grid cells that overlap the polygon (weights = intersection area fraction).
  * Lakes without a catchment       → value at the nearest ERA5-Land grid cell
    to the lake centroid.

ERA5-Land file format
---------------------
  Each file is a ZIP archive named
      era5land_mekong_<YYYY-MM>_<var>.nc
  containing a single member ``data_0.nc`` (HDF5/NetCDF4-4 format).
  Dimensions: (valid_time, latitude, longitude)
  Time resolution: 1-hourly.

Daily aggregation rules
-----------------------
  Accumulated vars (tp, ssrd, strd, sf):
      day_d = sum of the 24 hourly values for that calendar day
              (ERA5-Land stores hourly-interval totals, not running sums)

  Instantaneous vars (2t, 2d, sp, 10u, 10v, sd, swvl1-4):
      day_d = mean of the 24 hourly snapshots for that calendar day

Inputs
------
  ERA5-Land NC (ZIP) files: <DATA_DIR>/<YYYY-MM>/era5land_mekong_<YYYY-MM>_<var>.nc
  Catchment shapefile      : gritv06_pld_lake_catchments_0sqkm.shp
                             Columns: lake_id, geometry
  Lake centroid CSV        : Optional; required only for lakes absent from the
                             catchment shapefile.
                             Columns: lake_id, lon, lat

Outputs
-------
  One CSV per variable per month, organised into per-variable sub-directories:

      <OUTPUT_DIR>/<col_name>/era5land_per_lake_<col_name>_<YYYY-MM>.csv

  Columns:
      lake_id, date, <col_name>  (native ERA5-Land unit — see VAR_META),
      extraction_method          (area_weighted | centroid)

  Example output tree:
      era5land_per_pld_lake/
        tp/    era5land_per_lake_tp_2024-01.csv  …
        t2m/   era5land_per_lake_t2m_2024-01.csv …

Usage
-----
  python extract_era5land_per_catchment.py [options]

  Run `python extract_era5land_per_catchment.py --help` for full options.

Dependencies
------------
  numpy, pandas, geopandas, xarray, h5py, shapely
  (cartopy / matplotlib NOT required — data-only script)
"""

import io
import os
import pickle
import warnings
import zipfile
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import box

warnings.filterwarnings("ignore")

LAKE_AREA_THRESHOLD_SQKM = 0

# ---------------------------------------------------------------------------
# Configuration  (edit here to change behaviour)
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\era5_land")
CATCHMENT_SHP = (
    r"/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    rf"/gritv06_pld_lake_catchments_{LAKE_AREA_THRESHOLD_SQKM}sqkm.shp"
)
LAKE_CENTROIDS_CSV = (
    r"/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    rf"/gritv06_pld_lake_upstream_segments_{LAKE_AREA_THRESHOLD_SQKM}sqkm.csv"
)
OUTPUT_DIR = Path(
    Path(rf"/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/era5_land/era5land_per_pld_lake_{LAKE_AREA_THRESHOLD_SQKM}sqkm")
)
WEIGHTS_CACHE = ""   # path to a .pkl file to cache/load spatial weights; "" = no cache

START_YEAR  = 2023
START_MONTH = 1
END_YEAR    = 2025
END_MONTH   = 12
OVERWRITE   = False  # set True to reprocess months that already have a CSV
N_WORKERS   = 20      # number of parallel worker processes

# ---------------------------------------------------------------------------
# Variable metadata
# ---------------------------------------------------------------------------
# var_short_name → (col_name, aggregation_type, unit)
#   col_name  : column name used in output CSVs and sub-directory name
#   agg_type  : "accum" | "inst"
#   unit      : native ERA5-Land unit (kept as-is in output)
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

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
GRID_RES       = 0.1            # degrees
EQUAL_AREA_CRS = "ESRI:54034"   # WGS 84 / World Cylindrical Equal Area


# ===========================================================================
# I.  NetCDF loading helpers
# ===========================================================================

def nc_file_path(data_dir: Path, year: int, month: int, var: str) -> Path:
    """Return path to a monthly ERA5-Land ZIP-wrapped NetCDF file."""
    return data_dir / f"{year}-{month:02d}" / f"era5land_mekong_{year}-{month:02d}_{var}.nc"


def load_era5land_variable(nc_path: Path, var: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load one variable from an ERA5-Land ZIP-wrapped NetCDF4 file.

    The CDS download format is: <name>.nc → ZIP containing ``data_0.nc``
    (HDF5/NetCDF4).

    Returns
    -------
    (data, valid_times, lats, lons) where
        data        : float32 ndarray  shape (n_times, n_lat, n_lon)
        valid_times : int64 ndarray   Unix timestamps (seconds since 1970-01-01)
        lats        : float64 ndarray  decreasing
        lons        : float64 ndarray  increasing

    Returns None if the file does not exist or cannot be read.
    """
    if not nc_path.exists():
        print(f"  [MISS] {nc_path.name}")
        return None
    try:
        with zipfile.ZipFile(nc_path) as zf:
            inner_name = zf.namelist()[0]   # typically "data_0.nc"
            with zf.open(inner_name) as f:
                raw = f.read()
        buf = io.BytesIO(raw)
        with h5py.File(buf, "r") as hf:
            # Identify the data variable: first non-coordinate key
            coord_keys = {"latitude", "longitude", "valid_time", "expver", "number"}
            data_keys  = [k for k in hf.keys() if k not in coord_keys]
            if not data_keys:
                raise ValueError(f"No data variable found in {nc_path.name}")
            data_key = data_keys[0]

            data        = hf[data_key][:]          # (n_times, n_lat, n_lon)
            valid_times = hf["valid_time"][:]       # Unix timestamps
            lats        = hf["latitude"][:]
            lons        = hf["longitude"][:]
        return data, valid_times, lats, lons
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
    data        : float32 ndarray  shape (n_hours, n_lat, n_lon)
    valid_times : int64 ndarray   Unix timestamps (1-hourly)
    var         : ERA5-Land variable short name

    Returns
    -------
    (daily_data, date_strings) where
        daily_data   : float32 ndarray  shape (n_days, n_lat, n_lon)
        date_strings : list of "YYYY-MM-DD" strings, one per day
    """
    # Convert Unix timestamps to pandas DatetimeIndex for easy grouping
    times_dt = pd.to_datetime(valid_times, unit="s", utc=True)
    dates     = np.array([t.date() for t in times_dt])
    unique_dates = sorted(set(dates))

    daily_layers   = []
    date_strings   = []

    for d in unique_dates:
        mask = dates == d
        hourly_slice = data[mask]   # (24, n_lat, n_lon) for a full day

        if var in ACCUM_VARS:
            # Sum hourly amounts to get daily total
            daily_val = hourly_slice.sum(axis=0)
        else:
            # Mean of 24 hourly snapshots
            daily_val = hourly_slice.mean(axis=0)

        daily_layers.append(daily_val)
        date_strings.append(str(d))

    return np.stack(daily_layers, axis=0), date_strings   # (n_days, n_lat, n_lon)


# ===========================================================================
# III.  Spatial weights computation  (identical logic to ECMWF script)
# ===========================================================================

def _build_grid_polygons(
    lats: np.ndarray,
    lons: np.ndarray,
    res: float = GRID_RES,
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame of rectangular grid-cell polygons."""
    half = res / 2.0
    rows = []
    for lat in lats:
        for lon in lons:
            geom = box(lon - half, lat - half, lon + half, lat + half)
            rows.append({"lat": lat, "lon": lon, "geometry": geom})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf["flat_idx"] = np.arange(len(gdf), dtype=np.int32)
    return gdf


def compute_spatial_weights(
    catchments_gdf: gpd.GeoDataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
    lake_centroids: dict[int, tuple[float, float]] | None = None,
) -> dict[int, dict]:
    """
    Pre-compute area-weighted spatial weights for every lake.

    Returns dict lake_id → {"method", "flat_idx", "weights"}.
    """
    n_lons   = len(lons)
    grid_gdf = _build_grid_polygons(lats, lons)

    catchments_proj = catchments_gdf.to_crs(EQUAL_AREA_CRS)
    grid_proj       = grid_gdf.to_crs(EQUAL_AREA_CRS)

    weights_dict: dict[int, dict] = {}

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
            # Tiny catchment falls inside one grid cell — use centroid fallback
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

    # --- Lakes without catchment polygons (centroid fallback) ---
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
# IV.  Fast per-lake extraction  (identical logic to ECMWF script)
# ===========================================================================

def extract_lake_values(
    daily_grid: np.ndarray,
    weights_dict: dict[int, dict],
) -> dict[int, np.ndarray]:
    """
    Apply pre-computed weights to a (n_days, n_lat, n_lon) daily grid array.

    Returns dict lake_id → np.ndarray of shape (n_days,).
    """
    n_days    = daily_grid.shape[0]
    grid_flat = daily_grid.reshape(n_days, -1)   # (n_days, n_lat*n_lon)

    lake_values: dict[int, np.ndarray] = {}
    for lake_id, w in weights_dict.items():
        idx  = w["flat_idx"]
        wts  = w["weights"]
        vals = (grid_flat[:, idx] * wts[np.newaxis, :]).sum(axis=1)
        lake_values[lake_id] = vals

    return lake_values


# ===========================================================================
# V.  Loading helpers
# ===========================================================================

def load_lake_centroids(csv_path: str) -> dict[int, tuple[float, float]]:
    """Load lake centroids from a CSV file. Returns dict lake_id → (lon, lat)."""
    if not csv_path or not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path)
    return {int(row.lake_id): (float(row.lon), float(row.lat))
            for row in df.itertuples()}


def load_or_compute_weights(
    weights_cache: str,
    catchments_gdf: gpd.GeoDataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
    lake_centroids: dict,
) -> dict:
    """Load weights from cache or compute and save them."""
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
# VI.  Main processing loop  (one variable × one month at a time)
# ===========================================================================

def process_variable_month(
    data_dir: Path,
    year: int,
    month: int,
    var: str,
    col_name: str,
    weights_dict: dict,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Process one ERA5-Land variable for one month.

    Output: one CSV saved to
        <output_dir>/<col_name>/era5land_per_lake_<col_name>_<YYYY-MM>.csv

    Columns: lake_id, date, <col_name>, extraction_method

    Skips months whose output CSV already exists (unless overwrite=True).
    """
    fpath = nc_file_path(data_dir, year, month, var)

    var_dir = output_dir / col_name
    var_dir.mkdir(parents=True, exist_ok=True)

    month_str = f"{year}-{month:02d}"
    out_csv   = var_dir / f"era5land_per_lake_{col_name}_{month_str}.csv"

    if out_csv.exists() and not overwrite:
        print(f"  {var:6s} ({col_name})  {month_str}: skipped (already exists)")
        return

    result = load_era5land_variable(fpath, var)
    if result is None:
        return
    data, valid_times, lats, lons = result

    daily_grid, date_strings = compute_daily_fields(data, valid_times, var)
    lake_vals = extract_lake_values(daily_grid, weights_dict)

    records = []
    for lake_id in sorted(weights_dict.keys()):
        method = weights_dict[lake_id]["method"]
        vals   = lake_vals.get(lake_id, np.full(len(date_strings), np.nan))
        for d, date_str in enumerate(date_strings):
            records.append({
                "lake_id"          : lake_id,
                "date"             : date_str,
                col_name           : float(vals[d]),
                "extraction_method": method,
            })

    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"  {var:6s} ({col_name})  {month_str}: {len(date_strings)} day(s) → {out_csv.name}")


def _worker(args: tuple) -> None:
    """Unpack arguments and call process_variable_month (for Pool.map)."""
    process_variable_month(*args)


# ===========================================================================
# VII.  Entry point
# ===========================================================================

def main() -> None:
    # ---- 1. Load catchment polygons ----
    print(f"Loading catchment polygons: {CATCHMENT_SHP}")
    catchments_gdf = gpd.read_file(CATCHMENT_SHP)
    catchments_gdf["lake_id"] = catchments_gdf["lake_id"].astype(int)
    print(f"  {len(catchments_gdf)} catchment polygons  (CRS: {catchments_gdf.crs})")

    # ---- 2. Load lake centroids (for lakes without catchments) ----
    lake_centroids = load_lake_centroids(LAKE_CENTROIDS_CSV)
    if lake_centroids:
        print(f"  Loaded {len(lake_centroids)} lake centroids from CSV.")

    # ---- 3. Determine ERA5-Land grid from the first available file ----
    print("\nDetecting ERA5-Land grid from first available file …")
    ref_result = None
    for year in range(START_YEAR, END_YEAR + 1):
        m_start = START_MONTH if year == START_YEAR else 1
        m_end   = END_MONTH   if year == END_YEAR   else 12
        for month in range(m_start, m_end + 1):
            for var in VAR_META:
                fpath = nc_file_path(DATA_DIR, year, month, var)
                if fpath.exists():
                    ref_result = load_era5land_variable(fpath, var)
                    if ref_result is not None:
                        break
            if ref_result is not None:
                break
        if ref_result is not None:
            break

    if ref_result is None:
        raise FileNotFoundError(
            f"No ERA5-Land files found under {DATA_DIR} for the requested "
            f"period {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}."
        )

    _, _, lats, lons = ref_result
    print(f"  Grid: lat {lats[-1]:.1f}–{lats[0]:.1f}°N ({len(lats)} pts), "
          f"lon {lons[0]:.1f}–{lons[-1]:.1f}°E ({len(lons)} pts)")

    # ---- 4. Compute (or load) spatial weights ----
    weights_dict = load_or_compute_weights(
        WEIGHTS_CACHE, catchments_gdf, lats, lons, lake_centroids
    )

    # ---- 5. Build task list (var × month) ----
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
                    weights_dict, OUTPUT_DIR, OVERWRITE,
                ))

    print(f"\nDispatching {len(tasks)} task(s) across {N_WORKERS} worker(s) …")

    # ---- 6. Parallel processing ----
    with Pool(processes=N_WORKERS) as pool:
        pool.map(_worker, tasks)

    print(f"\nDone. Output CSVs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
