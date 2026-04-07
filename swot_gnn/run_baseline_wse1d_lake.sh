#!/bin/bash
# Local laptop run: last-observation persistence baseline for lake WSE forecasting.
# No SLURM — run directly with: bash run_baseline_wse1d_lake.sh

set -e   # abort immediately if any command fails

# ── Python interpreter (py311 conda env) ──────────────────────────────────────
PYTHON="/c/Users/xiaoye/anaconda3/envs/py311/python.exe"

# ── Data paths — update to your local copies ─────────────────────────────────
TRAINING_FOLDER="/e/Project_2025_2026/Smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data/mekong_lakes_swotpld_era5_ifshres10d_gritv06_202312_202602_qc"

WSE_DATACUBE="$TRAINING_FOLDER/swot_lake_wse_datacube_wse_norm.nc"
ERA5_DATACUBE="$TRAINING_FOLDER/swot_lake_era5_climate_datacube.nc"
ECMWF_DATACUBE="$TRAINING_FOLDER/swot_lake_ecmwf_forecast_datacube.nc"
STATIC_DATACUBE="$TRAINING_FOLDER/swot_lake_static_datacube.nc"
WSE_STATS_CSV="$TRAINING_FOLDER/lake_wse_norm_stats.csv"
LAKE_GRAPH="$TRAINING_FOLDER/gritv06_great_mekong_pld_lake_graph_0sqkm.csv"

SAVE_DIR="/e/Project_2025_2026/Smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/experiments"
RUN_NAME="exp00_mekong_wse1d_last_obs_gritv06_202312_202602"

# ── Code directory ─────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

# ── Run baseline ──────────────────────────────────────────────────────────────
echo "=== Running last-observation persistence baseline ==="
"$PYTHON" run_baseline_wse1d_lake.py \
    --wse-datacube    "$WSE_DATACUBE" \
    --era5-datacube   "$ERA5_DATACUBE" \
    --ecmwf-datacube  "$ECMWF_DATACUBE" \
    --static-datacube "$STATIC_DATACUBE" \
    --wse-stats-csv   "$WSE_STATS_CSV" \
    --lake-graph      "$LAKE_GRAPH" \
    --save-dir        "$SAVE_DIR" \
    --run-name        "$RUN_NAME"

echo "=== Done. Results saved to $SAVE_DIR/$RUN_NAME/ ==="
