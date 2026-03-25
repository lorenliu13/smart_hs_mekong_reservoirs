"""
Build the ERA5-Land climate dynamic datacube for the lake-based SWOT-GNN.

Output:
  swot_lake_era5_climate_datacube.nc
      dims (lake, time) — 13 derived ERA5-Land climate variables
      vars: LWd, SWd, P, Pres, Temp, Td, Wind, sf, sd,
            swvl1, swvl2, swvl3, swvl4

Usage:
    python build_era5_datacube_lake_based_20260319.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List

# ─── Configuration ─────────────────────────────────────────────────────────────

ERA5_BASE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/era5_land_daily_catchment_level/era5land_daily_great_mekong_per_pld_lake_0sqkm"
)
LAKE_GRAPH_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    "/gritv06_great_mekong_pld_lake_graph_0sqkm.csv"
)
SAVE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/training_data_lake_basedtraining_data_lake_based_great_mekong_20260325_20260325"
)

START_DATE = "2023-10-01"
END_DATE   = "2025-12-31"

# Raw ERA5-Land variable names as they appear in the per-lake CSV files.
ERA5_RAW_VARS = [
    "tp",    # total precipitation            (m)
    "t2m",   # 2-m air temperature            (K)
    "d2m",   # 2-m dewpoint temperature       (K)
    "sp",    # surface pressure               (Pa)
    "u10",   # 10-m U-component of wind       (m/s)
    "v10",   # 10-m V-component of wind       (m/s)
    "ssrd",  # surface shortwave radiation↓   (J/m²/day)
    "strd",  # surface longwave  radiation↓   (J/m²/day)
    "sf",    # snowfall                        (m of water equiv.)
    "sd",    # snow depth                      (m of water equiv.)
    "swvl1", # volumetric soil water layer 1  (m³/m³, 0–7 cm)
    "swvl2", # volumetric soil water layer 2  (m³/m³, 7–28 cm)
    "swvl3", # volumetric soil water layer 3  (m³/m³, 28–100 cm)
    "swvl4", # volumetric soil water layer 4  (m³/m³, 100–289 cm)
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
    Derive 13 model climate variables from raw ERA5-Land variables.

    Returns:
        Dict with keys: LWd, SWd, P, Pres, Temp, Td, Wind, sf, sd,
                        swvl1, swvl2, swvl3, swvl4
    """
    return {
        "LWd":   (raw["strd"]  / 86400.0).astype(np.float32),   # J/m² → W/m²
        "SWd":   (raw["ssrd"]  / 86400.0).astype(np.float32),   # J/m² → W/m²
        "P":     (raw["tp"]    * 1000.0).astype(np.float32),    # m   → mm
        "Pres":  raw["sp"].astype(np.float32),                   # Pa
        "Temp":  raw["t2m"].astype(np.float32),                  # K
        "Td":    raw["d2m"].astype(np.float32),                  # K  (2-m dewpoint)
        "Wind":  np.sqrt(raw["u10"]**2 + raw["v10"]**2).astype(np.float32),  # m/s
        "sf":    (raw["sf"]    * 1000.0).astype(np.float32),    # m   → mm (snowfall)
        "sd":    (raw["sd"]    * 1000.0).astype(np.float32),    # m   → mm (snow depth)
        "swvl1": raw["swvl1"].astype(np.float32),                # m³/m³
        "swvl2": raw["swvl2"].astype(np.float32),                # m³/m³
        "swvl3": raw["swvl3"].astype(np.float32),                # m³/m³
        "swvl4": raw["swvl4"].astype(np.float32),                # m³/m³
    }


def load_era5_climate_arrays(
    era5_base_dir: Path,
    raw_vars: list,
    lake_ids: np.ndarray,
    all_dates: pd.DatetimeIndex,
) -> dict:
    """
    Load ERA5-Land per-lake CSV files and return 13 derived climate arrays,
    each of shape (n_lakes, n_dates).

    CSV format: lake_id, date, {var_name}, extraction_method
    Files:      era5_base_dir/{var}/era5land_per_lake_{var}_YYYY-MM.csv
    """
    lake_id_set = set(lake_ids.tolist())
    all_dates_set = set(all_dates.tolist())
    year_months = sorted({(d.year, d.month) for d in all_dates})

    raw_dict = {}
    for var in raw_vars:
        print(f"  Loading ERA5 variable: {var} …")
        frames = []
        for (year, month) in year_months:
            fpath = era5_base_dir / var / f"era5land_per_lake_{var}_{year}-{month:02d}.csv"
            if not fpath.exists():
                print(f"    [WARN] Missing: {fpath}")
                continue
            df = pd.read_csv(fpath)
            df["date"] = pd.to_datetime(df["date"])
            df["lake_id"] = df["lake_id"].astype(np.int64)
            frames.append(df)

        if not frames:
            print(f"  [WARN] No files found for variable {var}; filling with zeros.")
            raw_dict[var] = np.zeros((len(lake_ids), len(all_dates)), dtype=np.float32)
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined = combined[
            combined["lake_id"].isin(lake_id_set) &
            combined["date"].isin(all_dates_set)
        ]

        pivot = (
            combined
            .pivot_table(index="lake_id", columns="date", values=var, aggfunc="first")
            .reindex(index=lake_ids, columns=all_dates)
        )
        arr = pivot.to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        raw_dict[var] = arr

    return derive_climate_vars(raw_dict)


def build_era5_climate_datacube(
    era5_base_dir: Path,
    lake_ids: np.ndarray,
    all_dates: pd.DatetimeIndex,
    save_dir: Path,
) -> Path:
    """
    Assemble and save the ERA5-Land climate datacube.

    Dims: (lake, time)
    Variables: LWd, SWd, P, Pres, Temp, Td, Wind, sf, sd,
               swvl1, swvl2, swvl3, swvl4
    """
    print("\n=== Building ERA5 climate datacube ===")
    print("Loading ERA5-Land climate data …")
    climate = load_era5_climate_arrays(era5_base_dir, ERA5_RAW_VARS, lake_ids, all_dates)

    data_vars = {var: (["lake", "time"], arr) for var, arr in climate.items()}

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lake": lake_ids,
            "time": all_dates,
        },
        attrs={
            "description": "ERA5-Land climate dynamic datacube for lake-SWOT-GNN",
            "created_by": "build_era5_datacube_lake_based_20260319.py",
        },
    )

    out_path = save_dir / "swot_lake_era5_climate_datacube.nc"
    ds.to_netcdf(out_path)
    print(f"ERA5 climate datacube saved → {out_path}  shape: {len(lake_ids)} lakes × {len(all_dates)} days")
    return out_path


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading lake IDs from: {LAKE_GRAPH_CSV}")
    lake_ids = load_lake_ids_from_graph(LAKE_GRAPH_CSV)
    print(f"  Found {len(lake_ids)} lakes in GRIT PLD lake graph.")

    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")

    build_era5_climate_datacube(
        era5_base_dir=ERA5_BASE_DIR,
        lake_ids=lake_ids,
        all_dates=all_dates,
        save_dir=SAVE_DIR,
    )
