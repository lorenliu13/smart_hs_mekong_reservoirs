"""
Build the ECMWF IFS forecast datacube for the lake-based SWOT-GNN.

Output:
  swot_lake_ecmwf_forecast_datacube.nc
      dims (lake, init_time, forecast_day) — 13 derived ECMWF IFS climate variables
      vars: LWd, SWd, P, Pres, Temp, Td, Wind, sf, sd,
            swvl1, swvl2, swvl3, swvl4

ECMWF init_dates are discovered automatically by scanning the per-lake CSV directory.

Usage:
    python build_ecmwf_ifs_datacube_lake_based_20260319.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────

ECMWF_BASE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/ecmwf_ifs_daily_catchment_level/hres/ecmwf_ifs_daily_great_mekong_per_pld_lake_0sqkm"
)
LAKE_GRAPH_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    "/gritv06_great_mekong_pld_lake_graph_0sqkm.csv"
)
SAVE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/training_data_lake_based_great_mekong_20260325"
)

# Date range limits for the datacube (init_dates outside this window are excluded).
START_DATE = "2023-10-01"
END_DATE   = "2025-12-31"

# Number of lead-days in the ECMWF IFS forecast per initialisation date.
FORECAST_HORIZON = 10

# Raw variable names shared with ERA5 — unit conversions are identical.
ERA5_RAW_VARS = [
    "tp", "t2m", "d2m", "sp", "u10", "v10",
    "ssrd", "strd", "sf", "sd",
    "swvl1", "swvl2", "swvl3", "swvl4",
]

# ───────────────────────────────────────────────────────────────────────────────


def load_lake_ids_from_graph(lake_graph_csv: Path) -> np.ndarray:
    """
    Extract unique lake IDs from the GRIT PLD lake graph CSV.
    Excludes the terminal node (-1).

    Returns:
        Sorted int64 array of lake IDs.
    """
    lake_graph = pd.read_csv(lake_graph_csv)
    ids = lake_graph['lake_id'].to_numpy(dtype=np.int64)
    ids = np.unique(ids)
    ids = ids[ids != -1]
    return ids


def derive_climate_vars(raw: dict) -> dict:
    """
    Derive 13 model climate variables from raw ERA5-Land / ECMWF IFS variables.

    Accepts arrays of any shape (2-D or 3-D). Element-wise numpy operations
    broadcast correctly for both ERA5 (n_lakes, n_dates) and ECMWF
    (n_lakes, n_init_dates, forecast_horizon) arrays.

    Returns:
        Dict with keys: LWd, SWd, P, Pres, Temp, Td, Wind, sf, sd,
                        swvl1, swvl2, swvl3, swvl4
    """
    return {
        "LWd":   (raw["strd"]  / 86400.0).astype(np.float32),
        "SWd":   (raw["ssrd"]  / 86400.0).astype(np.float32),
        "P":     (raw["tp"]    * 1000.0).astype(np.float32),
        "Pres":  raw["sp"].astype(np.float32),
        "Temp":  raw["t2m"].astype(np.float32),
        "Td":    raw["d2m"].astype(np.float32),
        "Wind":  np.sqrt(raw["u10"]**2 + raw["v10"]**2).astype(np.float32),
        "sf":    (raw["sf"]    * 1000.0).astype(np.float32),
        "sd":    (raw["sd"]    * 1000.0).astype(np.float32),
        "swvl1": raw["swvl1"].astype(np.float32),
        "swvl2": raw["swvl2"].astype(np.float32),
        "swvl3": raw["swvl3"].astype(np.float32),
        "swvl4": raw["swvl4"].astype(np.float32),
    }


def _determine_ecmwf_init_dates(ecmwf_base_dir: Path, probe_var: str = "tp") -> pd.DatetimeIndex:
    """
    Scan the ECMWF per-lake directory for one variable to determine available init_dates.

    Files are named: ecmwf_per_lake_{var}_YYYY-MM.csv  (one file per month, all
    init_dates for that month combined in the init_date column).
    """
    var_dir = ecmwf_base_dir / probe_var
    if not var_dir.exists():
        raise FileNotFoundError(f"ECMWF variable directory not found: {var_dir}")

    dates: set = set()
    monthly_files = sorted(var_dir.glob(f"ecmwf_per_lake_{probe_var}_????-??.csv"))
    for fpath in monthly_files:
        try:
            df = pd.read_csv(fpath, usecols=["init_date"])
            parsed = pd.to_datetime(df["init_date"], errors="coerce").dropna()
            for d in parsed:
                dates.add(d.normalize())
        except Exception:
            pass

    if not dates:
        raise ValueError(f"No ECMWF init_date files found in {var_dir}")

    return pd.DatetimeIndex(sorted(dates))


def load_ecmwf_climate_arrays(
    ecmwf_base_dir: Path,
    raw_vars: list,
    lake_ids: np.ndarray,
    all_init_dates: pd.DatetimeIndex,
    forecast_horizon: int = 10,
) -> dict:
    """
    Load ECMWF IFS per-lake CSV files and return 13 derived climate arrays,
    each of shape (n_lakes, n_init_dates, forecast_horizon).

    CSV format: lake_id, init_date, forecast_day, valid_date, {var_name}, extraction_method
    Files:      ecmwf_base_dir/{var}/ecmwf_per_lake_{var}_YYYY-MM.csv
    """
    n_lakes     = len(lake_ids)
    n_init      = len(all_init_dates)
    lake_to_idx = {lid: i for i, lid in enumerate(lake_ids)}
    init_to_idx = {d: i for i, d in enumerate(all_init_dates)}
    year_months = sorted({(d.year, d.month) for d in all_init_dates})

    raw_dict = {}
    for var in raw_vars:
        print(f"  Loading ECMWF variable: {var} …")
        raw_cube = np.zeros((n_lakes, n_init, forecast_horizon), dtype=np.float32)
        n_missing = 0

        for (year, month) in tqdm(year_months, desc=f"    {var}", leave=False):
            fpath = ecmwf_base_dir / var / f"ecmwf_per_lake_{var}_{year}-{month:02d}.csv"
            if not fpath.exists():
                n_missing += 1
                continue

            df = pd.read_csv(fpath)
            df["lake_id"]   = pd.to_numeric(df["lake_id"],   errors="coerce").dropna().astype(np.int64)
            df["init_date"] = pd.to_datetime(df["init_date"], errors="coerce")
            df = df.dropna(subset=["init_date"])
            df["init_date"] = df["init_date"].dt.normalize()
            df = df[df["lake_id"].isin(lake_to_idx)]
            if df.empty:
                continue

            for init_date, grp in df.groupby("init_date"):
                init_date = pd.Timestamp(init_date)
                if init_date not in init_to_idx:
                    continue

                t_idx = init_to_idx[init_date]
                pivot = (
                    grp.pivot_table(index="lake_id", columns="forecast_day", values=var, aggfunc="first")
                    .reindex(index=lake_ids)
                )
                for fd in range(1, forecast_horizon + 1):
                    if fd in pivot.columns:
                        col_vals = pivot[fd].to_numpy(dtype=np.float32)
                        col_vals = np.nan_to_num(col_vals, nan=0.0)
                        raw_cube[:, t_idx, fd - 1] = col_vals

        if n_missing > 0:
            pct = 100.0 * n_missing / max(len(year_months), 1)
            print(f"    [WARN] {var}: {n_missing}/{len(year_months)} monthly files missing ({pct:.1f}%)")

        raw_dict[var] = raw_cube

    return derive_climate_vars(raw_dict)


def build_ecmwf_forecast_datacube(
    ecmwf_base_dir: Path,
    lake_ids: np.ndarray,
    all_init_dates: pd.DatetimeIndex,
    forecast_horizon: int,
    save_dir: Path,
) -> Path:
    """
    Assemble and save the ECMWF IFS forecast datacube.

    Dims: (lake, init_time, forecast_day)
    Variables: LWd, SWd, P, Pres, Temp, Td, Wind, sf, sd, swvl1, swvl2, swvl3, swvl4
    """
    print("\n=== Building ECMWF forecast datacube ===")

    climate = load_ecmwf_climate_arrays(
        ecmwf_base_dir, ERA5_RAW_VARS, lake_ids, all_init_dates, forecast_horizon
    )

    forecast_days = np.arange(1, forecast_horizon + 1, dtype=np.int32)

    data_vars = {
        var_name: (["lake", "init_time", "forecast_day"], arr)
        for var_name, arr in climate.items()
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lake":         lake_ids,
            "init_time":    all_init_dates,
            "forecast_day": forecast_days,
        },
        attrs={
            "description": "ECMWF IFS forecast climate per lake for lake-SWOT-GNN",
            "forecast_horizon": forecast_horizon,
            "created_by": "build_ecmwf_ifs_datacube_lake_based_20260319.py",
        },
    )

    out_path = save_dir / "swot_lake_ecmwf_forecast_datacube.nc"
    ds.to_netcdf(out_path)
    print(
        f"ECMWF forecast datacube saved → {out_path}  "
        f"shape: {len(lake_ids)} lakes × {len(all_init_dates)} init_dates × {forecast_horizon} days"
    )
    return out_path


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading lake IDs from: {LAKE_GRAPH_CSV}")
    lake_ids = load_lake_ids_from_graph(LAKE_GRAPH_CSV)
    print(f"  Found {len(lake_ids)} lakes in GRIT PLD lake graph.")

    print(f"\nScanning ECMWF init_dates in: {ECMWF_BASE_DIR}")
    all_init_dates = _determine_ecmwf_init_dates(ECMWF_BASE_DIR, probe_var=ERA5_RAW_VARS[0])
    print(f"  Found {len(all_init_dates)} ECMWF init_dates "
          f"({all_init_dates[0].date()} … {all_init_dates[-1].date()})")

    all_init_dates = all_init_dates[
        (all_init_dates >= START_DATE) & (all_init_dates <= END_DATE)
    ]
    print(f"  After date filter [{START_DATE} … {END_DATE}]: {len(all_init_dates)} init_dates")

    build_ecmwf_forecast_datacube(
        ecmwf_base_dir=ECMWF_BASE_DIR,
        lake_ids=lake_ids,
        all_init_dates=all_init_dates,
        forecast_horizon=FORECAST_HORIZON,
        save_dir=SAVE_DIR,
    )
