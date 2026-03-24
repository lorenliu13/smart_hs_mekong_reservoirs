"""
Extract ERA5-Land reanalysis data for lake catchment polygons.

Reads the daily-aggregated NetCDF files produced by aggregate_era5land_to_daily.py
(one file per variable per month) and applies spatial weights to compute a
per-lake time series.

For each calendar month and each lake in the GRIT PLD database:

  * Lakes with a catchment polygon  → area-weighted average of all ERA5-Land
    grid cells that overlap the polygon (weights = intersection area fraction).
  * Lakes without a catchment       → value at the nearest ERA5-Land grid cell
    to the lake centroid.

Input file format (daily NetCDF)
---------------------------------
  <DAILY_DIR>/<YYYY-MM>/era5land_mekong_<YYYY-MM>_<col_name>_daily.nc
  Dimensions: (time, latitude, longitude)
  Variable  : <col_name>  (e.g. "tp", "t2m", …)
  Units     : as produced by aggregate_era5land_to_daily.py

Inputs
------
  Daily ERA5-Land NetCDF files : <DAILY_DIR>/<YYYY-MM>/era5land_mekong_<YYYY-MM>_<col_name>_daily.nc
  Catchment shapefile          : gritv06_pld_lake_catchments_0sqkm.shp
                                 Columns: lake_id, geometry
  Lake centroid CSV            : Optional; required only for lakes absent from the
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
  python extract_era5land_per_catchment.py

Dependencies
------------
  numpy, pandas, geopandas, xarray, shapely
"""

import pickle
import warnings
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

warnings.filterwarnings("ignore")

LAKE_AREA_THRESHOLD_SQKM = 0

# ---------------------------------------------------------------------------
# Configuration  (edit here to change behaviour)
# ---------------------------------------------------------------------------
DAILY_DIR = Path(r"E:\Project_2025_2026\Smart_hs\raw_data\era5_land_daily")
CATCHMENT_SHP = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    r"\GRIT_mekong_mega_reservoirs\reservoirs"
    rf"\gritv06_pld_lake_catchments_{LAKE_AREA_THRESHOLD_SQKM}sqkm.shp"
)
LAKE_CENTROIDS_CSV = (
    r"E:\Project_2025_2026\Smart_hs\raw_data\grit"
    r"\GRIT_mekong_mega_reservoirs\reservoirs"
    rf"\gritv06_pld_lake_upstream_segments_{LAKE_AREA_THRESHOLD_SQKM}sqkm.csv"
)
OUTPUT_DIR = Path(
    rf"E:\Project_2025_2026\Smart_hs\processed_data\mekong_river_basin_reservoirs"
    rf"\era5_land\era5land_daily_per_pld_lake_{LAKE_AREA_THRESHOLD_SQKM}sqkm"
)
WEIGHTS_CACHE = ""   # path to a .pkl file to cache/load spatial weights; "" = no cache

START_YEAR  = 2024
START_MONTH = 1
END_YEAR    = 2024
END_MONTH   = 1
OVERWRITE   = False  # set True to reprocess months that already have a CSV
N_WORKERS   = 8      # number of parallel worker processes

# ---------------------------------------------------------------------------
# Variable metadata  col_name → unit
# (aggregation is already applied by aggregate_era5land_to_daily.py)
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
GRID_RES       = 0.1            # degrees
EQUAL_AREA_CRS = "ESRI:54034"   # WGS 84 / World Cylindrical Equal Area


# ===========================================================================
# I.  Daily NetCDF loading helper
# ===========================================================================

def daily_nc_file_path(daily_dir: Path, year: int, month: int, col_name: str) -> Path:
    """Return path to a monthly ERA5-Land daily-aggregated NetCDF file."""
    return daily_dir / f"{year}-{month:02d}" / f"era5land_mekong_{year}-{month:02d}_{col_name}_daily.nc"


def load_daily_era5land(
    nc_path: Path,
    col_name: str,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray] | None:
    """
    Load one variable from a daily ERA5-Land NetCDF file.

    Returns
    -------
    (data, date_strings, lats, lons) where
        data         : float32 ndarray  shape (n_days, n_lat, n_lon)
        date_strings : list of "YYYY-MM-DD" strings, one per day
        lats         : float64 ndarray  decreasing
        lons         : float64 ndarray  increasing

    Returns None if the file does not exist or cannot be read.
    """
    if not nc_path.exists():
        print(f"  [MISS] {nc_path.name}")
        return None
    try:
        ds           = xr.open_dataset(nc_path)
        data         = ds[col_name].values.astype(np.float32)
        date_strings = [str(pd.Timestamp(t).date()) for t in ds["time"].values]
        lats         = ds["latitude"].values
        lons         = ds["longitude"].values
        ds.close()
        return data, date_strings, lats, lons
    except Exception as exc:
        print(f"  [FAIL] {nc_path.name}: {exc}")
        return None


# ===========================================================================
# II.  Spatial weights computation
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
        vals = (np.nan_to_num(grid_flat[:, idx], nan=0.0) * wts[np.newaxis, :]).sum(axis=1)
        lake_values[lake_id] = vals

    return lake_values


# ===========================================================================
# IV.  Loading helpers
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
# V.  Main processing loop  (one variable × one month at a time)
# ===========================================================================

def process_variable_month(
    daily_dir: Path,
    year: int,
    month: int,
    col_name: str,
    weights_dict: dict,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Extract per-lake values for one ERA5-Land variable for one month.

    Reads the pre-aggregated daily NetCDF produced by aggregate_era5land_to_daily.py
    and writes one CSV to:
        <output_dir>/<col_name>/era5land_per_lake_<col_name>_<YYYY-MM>.csv

    Columns: lake_id, date, <col_name>, extraction_method
    """
    var_dir = output_dir / col_name
    var_dir.mkdir(parents=True, exist_ok=True)

    month_str = f"{year}-{month:02d}"
    out_csv   = var_dir / f"era5land_per_lake_{col_name}_{month_str}.csv"

    if out_csv.exists() and not overwrite:
        print(f"  {col_name:6s}  {month_str}: skipped (already exists)")
        return

    nc_path = daily_nc_file_path(daily_dir, year, month, col_name)
    result  = load_daily_era5land(nc_path, col_name)
    if result is None:
        return
    daily_grid, date_strings, _, _ = result

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
    print(f"  {col_name:6s}  {month_str}: {len(date_strings)} day(s) → {out_csv.name}")


def _worker(args: tuple) -> None:
    """Unpack arguments and call process_variable_month (for Pool.map)."""
    process_variable_month(*args)


# ===========================================================================
# VI.  Entry point
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

    # ---- 3. Detect ERA5-Land grid from first available daily NetCDF ----
    print("\nDetecting ERA5-Land grid from first available daily file …")
    ref_result = None
    for year in range(START_YEAR, END_YEAR + 1):
        m_start = START_MONTH if year == START_YEAR else 1
        m_end   = END_MONTH   if year == END_YEAR   else 12
        for month in range(m_start, m_end + 1):
            for col_name in VAR_META:
                fpath = daily_nc_file_path(DAILY_DIR, year, month, col_name)
                if fpath.exists():
                    ref_result = load_daily_era5land(fpath, col_name)
                    if ref_result is not None:
                        break
            if ref_result is not None:
                break
        if ref_result is not None:
            break

    if ref_result is None:
        raise FileNotFoundError(
            f"No daily ERA5-Land files found under {DAILY_DIR} for the requested "
            f"period {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}."
        )

    _, _, lats, lons = ref_result
    print(f"  Grid: lat {lats[-1]:.1f}–{lats[0]:.1f}°N ({len(lats)} pts), "
          f"lon {lons[0]:.1f}–{lons[-1]:.1f}°E ({len(lons)} pts)")

    # ---- 4. Compute (or load) spatial weights ----
    weights_dict = load_or_compute_weights(
        WEIGHTS_CACHE, catchments_gdf, lats, lons, lake_centroids
    )

    # ---- 5. Build task list (col_name × month) ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for col_name in VAR_META:
        for year in range(START_YEAR, END_YEAR + 1):
            m_start = START_MONTH if year == START_YEAR else 1
            m_end   = END_MONTH   if year == END_YEAR   else 12
            for month in range(m_start, m_end + 1):
                tasks.append((
                    DAILY_DIR, year, month, col_name,
                    weights_dict, OUTPUT_DIR, OVERWRITE,
                ))

    print(f"\nDispatching {len(tasks)} task(s) across {N_WORKERS} worker(s) …")

    # ---- 6. Parallel processing ----
    with Pool(processes=N_WORKERS) as pool:
        pool.map(_worker, tasks)

    print(f"\nDone. Output CSVs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
