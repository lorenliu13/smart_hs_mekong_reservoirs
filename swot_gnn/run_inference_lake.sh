#!/bin/bash
#SBATCH --job-name=run_inference_lake
#SBATCH --output=run_inference_lake.out
#SBATCH --error=run_inference_lake.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=short         # GPU nodes are on the htc cluster, short partition (max 12h)
#SBATCH --gres=gpu:1

set -e   # abort immediately if any command fails

module load Anaconda3
conda activate $DATA/py311_torch

# ── Shared paths ───────────────────────────────────────────────────────────────
TRAINING_FOLDER="/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data/mekong_lakes_swotpld_era5_ifshres10d_gritv06_202312_202602_qc"

WSE_DATACUBE="$TRAINING_FOLDER/swot_lake_wse_datacube_wse_norm.nc"
ERA5_DATACUBE="$TRAINING_FOLDER/swot_lake_era5_climate_datacube.nc"
ECMWF_DATACUBE="$TRAINING_FOLDER/swot_lake_ecmwf_forecast_datacube.nc"
STATIC_DATACUBE="$TRAINING_FOLDER/swot_lake_static_datacube.nc"
WSE_STATS_CSV="$TRAINING_FOLDER/lake_wse_norm_stats.csv"
LAKE_GRAPH="$TRAINING_FOLDER/gritv06_great_mekong_pld_lake_graph_0sqkm.csv"

SAVE_DIR="/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/experiments"
RUN_NAME="exp02_mekong_wse1d_era5_ifshres_gritv06_202312_202602"
SEED=42

# ── Code directory ─────────────────────────────────────────────────────────────
cd /home/cenv1160/code/smart_hs_mekong_reservoirs/swot_gnn

# ── Inference (runs only if training succeeded) ────────────────────────────────
python run_inference_lake.py \
    --wse-datacube    "$WSE_DATACUBE" \
    --era5-datacube   "$ERA5_DATACUBE" \
    --ecmwf-datacube  "$ECMWF_DATACUBE" \
    --static-datacube "$STATIC_DATACUBE" \
    --wse-stats-csv   "$WSE_STATS_CSV" \
    --lake-graph      "$LAKE_GRAPH" \
    --save-dir        "$SAVE_DIR" \
    --run-name        "$RUN_NAME" \
    --seed            $SEED \
    --device cuda
