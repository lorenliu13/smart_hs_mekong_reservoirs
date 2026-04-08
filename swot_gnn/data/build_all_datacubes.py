"""
Build all four lake-based SWOT-GNN datacubes and write them to SAVE_DIR.

Datacubes built:
  1. swot_lake_wse_datacube_{wse_option}.nc          (WSE dynamic)
  2. swot_lake_era5_climate_datacube.nc              (ERA5-Land climate dynamic)
  3. swot_lake_ecmwf_forecast_datacube.nc            (ECMWF IFS forecast dynamic)
  4. swot_lake_static_datacube.nc                    (static attributes)

Side files:
  - lake_wse_norm_stats.csv
  - {lake_graph_csv.name}  (copied to SAVE_DIR)

Usage:
    python build_all_datacubes.py
"""

import importlib.util
from pathlib import Path

import pandas as pd

from datacube_utils import load_lake_ids_from_graph


AREA_THRESHOLD_SQKM = 0.1  # Minimum lake area in square kilometers to retain
OBS_COUNT_THRESHOLD = 20  # Minimum number of daily observations to retain a lake

# ─── Configuration (edit paths and parameters here) ───────────────────────────

LAKE_GRAPH_CSV = Path(
    f"/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs/lake_graph_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}"
    f"/gritv06_great_mekong_pld_lake_graph_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}.csv"
)
SAVE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data"
    "/mekong_lakes_swotpld_era5_ifshres10d_gritv06_202312_202602_qc_area{AREA_THRESHOLD_SQKM}_obs{OBS_COUNT_THRESHOLD}"
)

# Date range — applies to WSE, ERA5, and ECMWF (static has no time axis).
# Specify as YYYY-MM; start expands to the 1st, end to the last day (inclusive).
START_MONTH = "2023-10"
END_MONTH   = "2026-02"

# ── WSE ───────────────────────────────────────────────────────────────────────
SWOT_LAKE_WSE_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/swot/great_mekong_river_basin/lakes_daily"
    f"/swot_lake_2023_12_2026_02_daily_wse_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}.csv"
)
WSE_OPTION = "wse_norm"   # "wse_norm" | "wse_anomaly" | "wse"

# ── ERA5-Land ─────────────────────────────────────────────────────────────────
ERA5_BASE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/era5_land_daily_catchment_level/daily_per_lake_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}"
)

# ── ECMWF IFS ────────────────────────────────────────────────────────────────
ECMWF_BASE_DIR = Path(
    "/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/ecmwf_ifs_daily_catchment_level/hres/daily_per_lake_area_{AREA_THRESHOLD_SQKM}_sample_{OBS_COUNT_THRESHOLD}"
)
FORECAST_HORIZON = 10

# ── Static attributes ─────────────────────────────────────────────────────────
REACH_ATTRS_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/attrs"
    "/GRITv06_reach_predictors_shared_MEKO_YANG.csv"
)
LAKE_UPSTREAM_SEGS_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reservoirs"
    "/gritv06_great_mekong_pld_lake_upstream_segments_0sqkm.csv"
)
REACHES_WITH_LAKES_CSV = Path(
    "/data/ouce-grit/cenv1160/smart_hs/raw_data/grit/mekong_river_basin/reaches"
    "/gritv06_reaches_great_mekong_with_lake_id.csv"
)
STATIC_EXCLUDE_COLS = [
    "domain",
    "mean_annual_discharge",
    "mean_sum_runoff",
    "mean_annual_discharge_da",
    "mean_sum_runoff_da",
]

# ─────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent


def _import_builder(stem: str):
    """Import a builder module from the same directory by file stem."""
    path = _HERE / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Shared setup ──────────────────────────────────────────────────────────
    print(f"Output folder : {SAVE_DIR}")

    print(f"\nLoading lake IDs from: {LAKE_GRAPH_CSV}")
    lake_ids = load_lake_ids_from_graph(LAKE_GRAPH_CSV)
    print(f"  Found {len(lake_ids)} lakes.")

    lake_graph_dst = SAVE_DIR / LAKE_GRAPH_CSV.name
    if not lake_graph_dst.exists():
        pd.read_csv(LAKE_GRAPH_CSV).to_csv(lake_graph_dst, index=False)
        print(f"Lake graph CSV copied → {lake_graph_dst}")

    start_date = pd.Timestamp(START_MONTH + "-01")
    end_date   = pd.Timestamp(END_MONTH   + "-01") + pd.offsets.MonthEnd(0)
    all_dates  = pd.date_range(start_date, end_date, freq="D")
    print(f"Date range    : {all_dates[0].date()} … {all_dates[-1].date()} ({len(all_dates)} days)")

    # ── 1. WSE ────────────────────────────────────────────────────────────────
    wse_mod = _import_builder("build_wse_datacube_lake_based")
    wse_mod.build_wse_datacube(
        swot_csv=SWOT_LAKE_WSE_CSV,
        lake_ids=lake_ids,
        all_dates=all_dates,
        wse_option=WSE_OPTION,
        save_dir=SAVE_DIR,
    )

    # ── 2. ERA5-Land climate ──────────────────────────────────────────────────
    era5_mod = _import_builder("build_era5_datacube_lake_based")
    era5_mod.build_era5_climate_datacube(
        era5_base_dir=ERA5_BASE_DIR,
        lake_ids=lake_ids,
        all_dates=all_dates,
        save_dir=SAVE_DIR,
    )

    # ── 3. ECMWF IFS forecast ─────────────────────────────────────────────────
    ecmwf_mod = _import_builder("build_ecmwf_ifs_datacube_lake_based")
    print(f"\nScanning ECMWF init_dates in: {ECMWF_BASE_DIR}")
    all_init_dates = ecmwf_mod._determine_ecmwf_init_dates(
        ECMWF_BASE_DIR, probe_var=ecmwf_mod.ERA5_RAW_VARS[0]
    )
    all_init_dates = all_init_dates[
        (all_init_dates >= all_dates[0]) & (all_init_dates <= all_dates[-1])
    ]
    print(f"  Using {len(all_init_dates)} init_dates "
          f"({all_init_dates[0].date()} … {all_init_dates[-1].date()})")
    ecmwf_mod.build_ecmwf_forecast_datacube(
        ecmwf_base_dir=ECMWF_BASE_DIR,
        lake_ids=lake_ids,
        all_init_dates=all_init_dates,
        forecast_horizon=FORECAST_HORIZON,
        save_dir=SAVE_DIR,
    )

    # ── 4. Static attributes ──────────────────────────────────────────────────
    static_mod = _import_builder("build_static_datacube_lake_based")
    static_df = static_mod.prepare_static_attrs(
        lake_upstream_segs_csv=LAKE_UPSTREAM_SEGS_CSV,
        reaches_with_lakes_csv=REACHES_WITH_LAKES_CSV,
        reach_attrs_csv=REACH_ATTRS_CSV,
    )
    static_mod.build_static_datacube(
        lake_ids=lake_ids,
        static_df=static_df,
        save_dir=SAVE_DIR,
        exclude_cols=STATIC_EXCLUDE_COLS,
    )

    print("\n=== All done ===")
