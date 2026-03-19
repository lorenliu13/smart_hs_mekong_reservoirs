"""
Prepare SWOT lake WSE training datacubes for the lake-based SWOT-GNN.

Outputs three NetCDF files:
  1. swot_lake_era5_dynamic_datacube_{wse_option}.nc
       dims (lake, time) — SWOT WSE features + ERA5-Land climate (9 vars)
  2. swot_lake_ecmwf_forecast_datacube.nc
       dims (lake, init_time, forecast_day) — ECMWF IFS climate (9 vars)
  3. swot_lake_static_datacube.nc
       dims (lake, feature) — placeholder zeros (static attrs to be added later)

Plus a side file: lake_wse_norm_stats.csv (per-lake wse_mean, wse_std).

Run on the cluster where ERA5/ECMWF per-lake CSVs are available.

Usage:
    python training_data_processing_lake_based_20260319.py
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────

# Cluster data paths
ERA5_BASE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data"
    "/mekong_river_basin_reservoirs/era5_land_daily_catchment_level"
    "/era5land_daily_per_pld_lake_0sqkm"
)
ECMWF_BASE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data"
    "/mekong_river_basin_reservoirs/ecmwf_ifs_daily_catchment_level"
    "/hres/ecmwf_ifs_daily_per_pld_lake_0sqkm"
)
SWOT_LAKE_WSE_CSV = Path(
    "E:/Project_2025_2026/Smart_hs/processed_data/swot/mekong_river_basin/swot/lakes_daily"
    "/swot_lake_daily_wse_xtrk10_60km_dark50pct_qf01_lake_graph_0sqkm.csv"
)
LAKE_GRAPH_CSV = Path(
    "E:/Project_2025_2026/Smart_hs/raw_data/grit/GRIT_mekong_mega_reservoirs"
    "/reservoirs/gritv06_pld_lake_graph_0sqkm.csv"
)
SAVE_DIR = Path(
    "E:/Project_2025_2026/Smart_hs/processed_data/swot_gnn/training_data"
    "/training_data_lake_based_20260319"
)

# Date range for ERA5 / SWOT WSE (ECMWF init_dates are determined from available files)
START_DATE = "2023-10-01"
END_DATE   = "2025-12-31"

# WSE normalization option
WSE_OPTION = "wse_norm"   # "wse_norm" | "wse_anomaly" | "wse"

# ECMWF forecast horizon (number of forecast days per init_date)
FORECAST_HORIZON = 10

# Raw ERA5-Land variable names (column names in the per-lake CSVs)
ERA5_RAW_VARS = ["tp", "t2m", "d2m", "sp", "u10", "v10", "ssrd", "strd", "sd", "swvl1"]

# ───────────────────────────────────────────────────────────────────────────────


def load_lake_ids_from_graph(lake_graph_csv: Path) -> np.ndarray:
    """
    Extract unique lake IDs from the GRIT PLD lake graph CSV.
    Excludes the terminal node (-1).

    Returns:
        Sorted int64 array of lake IDs.
    """
    lake_graph = pd.read_csv(lake_graph_csv)
    # Collect all lake_id values from every column
    all_ids = []
    for col in lake_graph.columns:
        col_lower = col.lower()
        if "lake" in col_lower or "id" in col_lower or col_lower in ("fid", "src", "dst"):
            try:
                vals = pd.to_numeric(lake_graph[col], errors="coerce").dropna().astype(np.int64)
                all_ids.append(vals)
            except Exception:
                pass
    if not all_ids:
        # Fallback: try the first two columns
        for col in lake_graph.columns[:2]:
            try:
                vals = pd.to_numeric(lake_graph[col], errors="coerce").dropna().astype(np.int64)
                all_ids.append(vals)
            except Exception:
                pass

    combined = pd.concat(all_ids).unique() if all_ids else np.array([], dtype=np.int64)
    combined = combined[combined != -1]   # exclude terminal node
    return np.sort(combined.astype(np.int64))


def derive_climate_vars(raw: dict) -> dict:
    """
    Derive 9 model climate variables from raw ERA5-Land / ECMWF IFS variables.

    Accepts arrays of any shape (2-D or 3-D). Element-wise numpy operations
    broadcast correctly for both ERA5 (n_lakes, n_dates) and ECMWF
    (n_lakes, n_init_dates, forecast_horizon) arrays.

    Returns:
        Dict with keys: LWd, SWd, P, Pres, Temp, Wind, RelHum, sd, swvl1
    """
    t2m  = raw["t2m"]   # K
    d2m  = raw["d2m"]   # K
    t_c  = t2m - 273.15
    td_c = d2m - 273.15

    # Saturation / actual vapour pressure (August-Roche-Magnus formula)
    es = np.exp(17.27 * t_c  / (t_c  + 237.3 + 1e-9))
    ea = np.exp(17.27 * td_c / (td_c + 237.3 + 1e-9))
    rel_hum = np.clip(100.0 * ea / (es + 1e-9), 0.0, 150.0).astype(np.float32)

    return {
        "LWd":    (raw["strd"]  / 86400.0).astype(np.float32),   # J/m² → W/m²
        "SWd":    (raw["ssrd"]  / 86400.0).astype(np.float32),   # J/m² → W/m²
        "P":      (raw["tp"]    * 1000.0).astype(np.float32),    # m   → mm
        "Pres":   raw["sp"].astype(np.float32),                   # Pa  (native)
        "Temp":   t2m.astype(np.float32),                         # K
        "Wind":   np.sqrt(raw["u10"]**2 + raw["v10"]**2).astype(np.float32),  # m/s
        "RelHum": rel_hum,
        "sd":     raw["sd"].astype(np.float32),                   # m  (snow depth)
        "swvl1":  raw["swvl1"].astype(np.float32),                # m³/m³ (soil moisture)
    }


def build_swot_wse_arrays(
    swot_csv: Path,
    lake_ids: np.ndarray,
    all_dates: pd.DatetimeIndex,
    wse_option: str = "wse_norm",
):
    """
    Load SWOT lake daily WSE and compute per-lake feature arrays.

    Returns:
        wse_cube:        (n_lakes, n_dates) – NaN where not observed
        mask_cube:       (n_lakes, n_dates) – 1 where observed
        latest_wse_cube: (n_lakes, n_dates) – forward-filled normalized WSE
        lag_cube:        (n_lakes, n_dates) – days since last observation
        doy_sin_cube:    (n_lakes, n_dates) – sin(2π * doy / 365.25)
        doy_cos_cube:    (n_lakes, n_dates) – cos(2π * doy / 365.25)
        norm_stats_df:   DataFrame with columns lake_id, wse_mean, wse_std
    """
    print("Loading SWOT lake WSE data …")
    swot_df = pd.read_csv(swot_csv)
    swot_df["date"] = pd.to_datetime(swot_df["date"])
    swot_df["lake_id"] = swot_df["lake_id"].astype(np.int64)

    # Per-lake mean and std for normalization
    grp = swot_df.groupby("lake_id")["wse"]
    wse_mean = grp.mean().rename("wse_mean")
    wse_std  = grp.std().rename("wse_std").fillna(1.0).clip(lower=1e-8)
    norm_stats_df = pd.DataFrame({"wse_mean": wse_mean, "wse_std": wse_std}).reset_index()

    # Add normalized columns
    swot_df = swot_df.merge(norm_stats_df, on="lake_id", how="left")
    swot_df["wse_anomaly"] = swot_df["wse"] - swot_df["wse_mean"]
    swot_df["wse_norm"]    = swot_df["wse_anomaly"] / (swot_df["wse_std"] + 1e-8)

    n_lakes = len(lake_ids)
    n_dates = len(all_dates)
    shape   = (n_lakes, n_dates)

    wse_cube        = np.full(shape, np.nan,  dtype=np.float32)
    mask_cube       = np.zeros(shape,         dtype=np.int8)
    latest_wse_cube = np.zeros(shape,         dtype=np.float32)
    lag_cube        = np.zeros(shape,         dtype=np.float32)

    # Cyclical day-of-year encoding (same for every lake)
    doy = all_dates.dayofyear.to_numpy().astype(np.float32)
    time_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    time_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    doy_sin_cube = np.tile(time_sin, (n_lakes, 1))
    doy_cos_cube = np.tile(time_cos, (n_lakes, 1))

    lake_set = set(lake_ids.tolist())
    lake_to_idx = {lid: i for i, lid in enumerate(lake_ids)}

    for lake_id, lake_df in tqdm(swot_df.groupby("lake_id"), desc="Building SWOT WSE arrays"):
        if lake_id not in lake_set:
            continue
        i = lake_to_idx[lake_id]
        lake_df = lake_df.set_index("date")
        full_series = lake_df[wse_option].reindex(all_dates)

        mask        = (~full_series.isna()).astype(np.int8).values
        latest_wse  = full_series.ffill().fillna(0.0).values

        # Days since last observation
        valid = mask.astype(float)
        last_valid_idx = np.where(valid == 1, np.arange(n_dates, dtype=float), np.nan)
        last_valid_idx = pd.Series(last_valid_idx).ffill().to_numpy()
        last_valid_idx = np.where(np.isnan(last_valid_idx), 0, last_valid_idx)
        lag = np.arange(n_dates, dtype=np.float32) - last_valid_idx

        wse_cube[i, :]        = full_series.values.astype(np.float32)
        mask_cube[i, :]       = mask
        latest_wse_cube[i, :] = latest_wse.astype(np.float32)
        lag_cube[i, :]        = lag.astype(np.float32)

    return (
        wse_cube,
        mask_cube,
        latest_wse_cube,
        lag_cube,
        doy_sin_cube,
        doy_cos_cube,
        norm_stats_df,
    )


def load_era5_climate_arrays(
    era5_base_dir: Path,
    raw_vars: list,
    lake_ids: np.ndarray,
    all_dates: pd.DatetimeIndex,
) -> dict:
    """
    Load ERA5-Land per-lake CSV files and return 9 derived climate arrays,
    each of shape (n_lakes, n_dates).

    CSV format: lake_id, date, {var_name}, extraction_method
    Files:      era5_base_dir/{var}/era5land_per_lake_{var}_YYYY-MM.csv
    """
    lake_id_set = set(lake_ids.tolist())
    all_dates_set = set(all_dates.tolist())

    # Determine year-month combinations to load
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


def build_era5_dynamic_datacube(
    swot_csv: Path,
    era5_base_dir: Path,
    lake_ids: np.ndarray,
    all_dates: pd.DatetimeIndex,
    wse_option: str,
    save_dir: Path,
) -> Path:
    """
    Assemble and save the ERA5 + SWOT WSE dynamic datacube.

    Dims: (lake, time)
    Variables: obs_mask, latest_wse, days_since_last_obs, time_doy_sin, time_doy_cos,
               wse (label), LWd, SWd, P, Pres, Temp, Wind, RelHum, sd, swvl1
    """
    print("\n=== Building ERA5 dynamic datacube ===")

    # SWOT WSE arrays
    (wse_cube, mask_cube, latest_wse_cube, lag_cube,
     doy_sin_cube, doy_cos_cube, norm_stats_df) = build_swot_wse_arrays(
        swot_csv, lake_ids, all_dates, wse_option
    )

    # ERA5 climate arrays
    print("Loading ERA5-Land climate data …")
    climate = load_era5_climate_arrays(era5_base_dir, ERA5_RAW_VARS, lake_ids, all_dates)

    # Assemble xr.Dataset
    ds = xr.Dataset(
        data_vars={
            # SWOT features
            "obs_mask":            (["lake", "time"], mask_cube.astype(np.int8)),
            "latest_wse":          (["lake", "time"], latest_wse_cube),
            "days_since_last_obs": (["lake", "time"], lag_cube),
            "time_doy_sin":        (["lake", "time"], doy_sin_cube),
            "time_doy_cos":        (["lake", "time"], doy_cos_cube),
            # Target label (NaN where not observed)
            "wse":                 (["lake", "time"], wse_cube),
            # ERA5-Land derived climate
            "LWd":    (["lake", "time"], climate["LWd"]),
            "SWd":    (["lake", "time"], climate["SWd"]),
            "P":      (["lake", "time"], climate["P"]),
            "Pres":   (["lake", "time"], climate["Pres"]),
            "Temp":   (["lake", "time"], climate["Temp"]),
            "Wind":   (["lake", "time"], climate["Wind"]),
            "RelHum": (["lake", "time"], climate["RelHum"]),
            "sd":     (["lake", "time"], climate["sd"]),
            "swvl1":  (["lake", "time"], climate["swvl1"]),
        },
        coords={
            "lake": lake_ids,
            "time": all_dates,
        },
        attrs={
            "description": "SWOT lake WSE + ERA5-Land climate dynamic datacube for lake-SWOT-GNN",
            "wse_option": wse_option,
            "created_by": "training_data_processing_lake_based_20260319.py",
        },
    )

    out_path = save_dir / f"swot_lake_era5_dynamic_datacube_{wse_option}.nc"
    ds.to_netcdf(out_path)
    print(f"ERA5 dynamic datacube saved → {out_path}  shape: {len(lake_ids)} lakes × {len(all_dates)} days")

    # Save per-lake normalization stats alongside the datacube
    stats_path = save_dir / "lake_wse_norm_stats.csv"
    norm_stats_df.to_csv(stats_path, index=False)
    print(f"WSE norm stats saved → {stats_path}")

    return out_path


def _determine_ecmwf_init_dates(ecmwf_base_dir: Path, probe_var: str = "tp") -> pd.DatetimeIndex:
    """
    Scan the ECMWF per-lake directory for one variable to determine available init_dates.

    Files are named: ecmwf_per_lake_{var}_YYYY-MM-DD.csv
    """
    var_dir = ecmwf_base_dir / probe_var
    if not var_dir.exists():
        raise FileNotFoundError(f"ECMWF variable directory not found: {var_dir}")

    dates = []
    for fpath in sorted(var_dir.glob(f"ecmwf_per_lake_{probe_var}_*.csv")):
        stem = fpath.stem  # e.g. "ecmwf_per_lake_tp_2024-01-14"
        date_str = stem.split("_")[-1]
        try:
            dates.append(pd.Timestamp(date_str))
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
    Load ECMWF IFS per-lake CSV files and return 9 derived climate arrays,
    each of shape (n_lakes, n_init_dates, forecast_horizon).

    CSV format: lake_id, init_date, forecast_day, valid_date, {var_name}, extraction_method
    Files:      ecmwf_base_dir/{var}/ecmwf_per_lake_{var}_YYYY-MM-DD.csv
    """
    n_lakes     = len(lake_ids)
    n_init      = len(all_init_dates)
    lake_to_idx = {lid: i for i, lid in enumerate(lake_ids)}
    init_to_idx = {d: i for i, d in enumerate(all_init_dates)}

    raw_dict = {}
    for var in raw_vars:
        print(f"  Loading ECMWF variable: {var} …")
        raw_cube = np.zeros((n_lakes, n_init, forecast_horizon), dtype=np.float32)
        n_missing = 0

        for init_date in tqdm(all_init_dates, desc=f"    {var}", leave=False):
            date_str = init_date.strftime("%Y-%m-%d")
            fpath = ecmwf_base_dir / var / f"ecmwf_per_lake_{var}_{date_str}.csv"
            if not fpath.exists():
                n_missing += 1
                continue

            df = pd.read_csv(fpath)
            df["lake_id"] = pd.to_numeric(df["lake_id"], errors="coerce").dropna().astype(np.int64)
            df = df[df["lake_id"].isin(lake_to_idx)]
            if df.empty:
                continue

            t_idx = init_to_idx[init_date]
            # Pivot: rows=lake_id, cols=forecast_day, values=var
            pivot = (
                df.pivot_table(index="lake_id", columns="forecast_day", values=var, aggfunc="first")
                .reindex(index=lake_ids)
            )
            # Fill available forecast days (1-indexed)
            for fd in range(1, forecast_horizon + 1):
                if fd in pivot.columns:
                    col_vals = pivot[fd].to_numpy(dtype=np.float32)
                    col_vals = np.nan_to_num(col_vals, nan=0.0)
                    raw_cube[:, t_idx, fd - 1] = col_vals

        if n_missing > 0:
            pct = 100.0 * n_missing / max(n_init, 1)
            print(f"    [WARN] {var}: {n_missing}/{n_init} init_date files missing ({pct:.1f}%)")

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
    Variables: LWd, SWd, P, Pres, Temp, Wind, RelHum, sd, swvl1
    """
    print("\n=== Building ECMWF forecast datacube ===")

    climate = load_ecmwf_climate_arrays(
        ecmwf_base_dir, ERA5_RAW_VARS, lake_ids, all_init_dates, forecast_horizon
    )

    forecast_days = np.arange(1, forecast_horizon + 1, dtype=np.int32)

    data_vars = {}
    for var_name, arr in climate.items():
        data_vars[var_name] = (["lake", "init_time", "forecast_day"], arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lake":        lake_ids,
            "init_time":   all_init_dates,
            "forecast_day": forecast_days,
        },
        attrs={
            "description": "ECMWF IFS forecast climate per lake for lake-SWOT-GNN",
            "forecast_horizon": forecast_horizon,
            "created_by": "training_data_processing_lake_based_20260319.py",
        },
    )

    out_path = save_dir / "swot_lake_ecmwf_forecast_datacube.nc"
    ds.to_netcdf(out_path)
    print(
        f"ECMWF forecast datacube saved → {out_path}  "
        f"shape: {len(lake_ids)} lakes × {len(all_init_dates)} init_dates × {forecast_horizon} days"
    )
    return out_path


def build_static_datacube(lake_ids: np.ndarray, save_dir: Path) -> Path:
    """
    Build a placeholder static datacube (zeros).
    Static lake attributes will be added in a future version.
    """
    print("\n=== Building static datacube (placeholder) ===")
    n_lakes = len(lake_ids)
    static_cube = np.zeros((n_lakes, 1), dtype=np.float32)

    ds = xr.Dataset(
        data_vars={"static_feature": (["lake", "feature"], static_cube)},
        coords={
            "lake":    lake_ids,
            "feature": ["placeholder"],
        },
        attrs={
            "description": "Placeholder static datacube for lake-SWOT-GNN (to be expanded)",
            "created_by": "training_data_processing_lake_based_20260319.py",
        },
    )

    out_path = save_dir / "swot_lake_static_datacube.nc"
    ds.to_netcdf(out_path)
    print(f"Static datacube saved → {out_path}  shape: {n_lakes} lakes × 1 feature (placeholder)")
    return out_path


def generate_lake_datacubes(
    swot_csv: Path = SWOT_LAKE_WSE_CSV,
    lake_graph_csv: Path = LAKE_GRAPH_CSV,
    era5_base_dir: Path = ERA5_BASE_DIR,
    ecmwf_base_dir: Path = ECMWF_BASE_DIR,
    save_dir: Path = SAVE_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    wse_option: str = WSE_OPTION,
    forecast_horizon: int = FORECAST_HORIZON,
) -> None:
    """
    Main entry point: build all three training datacubes for the lake-based SWOT-GNN.

    Steps:
      1. Load lake IDs from GRIT PLD lake graph
      2. Build ERA5 + SWOT WSE dynamic datacube
      3. Discover available ECMWF init_dates and build forecast datacube
      4. Build static placeholder datacube
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lake IDs
    print(f"Loading lake IDs from: {lake_graph_csv}")
    lake_ids = load_lake_ids_from_graph(lake_graph_csv)
    print(f"  Found {len(lake_ids)} lakes in GRIT PLD lake graph.")

    # 2. ERA5 dynamic datacube
    all_dates = pd.date_range(start_date, end_date, freq="D")
    build_era5_dynamic_datacube(
        swot_csv=swot_csv,
        era5_base_dir=era5_base_dir,
        lake_ids=lake_ids,
        all_dates=all_dates,
        wse_option=wse_option,
        save_dir=save_dir,
    )

    # 3. ECMWF forecast datacube
    print(f"\nScanning ECMWF init_dates in: {ecmwf_base_dir}")
    all_init_dates = _determine_ecmwf_init_dates(ecmwf_base_dir, probe_var=ERA5_RAW_VARS[0])
    print(f"  Found {len(all_init_dates)} ECMWF init_dates "
          f"({all_init_dates[0].date()} … {all_init_dates[-1].date()})")

    build_ecmwf_forecast_datacube(
        ecmwf_base_dir=ecmwf_base_dir,
        lake_ids=lake_ids,
        all_init_dates=all_init_dates,
        forecast_horizon=forecast_horizon,
        save_dir=save_dir,
    )

    # 4. Static placeholder datacube
    build_static_datacube(lake_ids=lake_ids, save_dir=save_dir)

    print("\n=== Done. Summary ===")
    print(f"  Lakes:       {len(lake_ids)}")
    print(f"  ERA5 dates:  {len(all_dates)} ({start_date} … {end_date})")
    print(f"  ECMWF inits: {len(all_init_dates)}")
    print(f"  Output dir:  {save_dir}")


if __name__ == "__main__":
    generate_lake_datacubes()
