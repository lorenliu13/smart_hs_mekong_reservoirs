#!/bin/bash
#SBATCH --job-name=full_inference_wsend
#SBATCH --output=full_inference_wsend.out
#SBATCH --error=full_inference_wsend.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1

set -e

module load Anaconda3
conda activate $DATA/py311_torch

# ── Shared paths ───────────────────────────────────────────────────────────────
AREA_THRESHOLD=0.1
SAMPLE_THRESHOLD=30
TRAINING_FOLDER="/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data/mekong_lakes_swotpld_era5_ifshres10d_gritv06_202312_202602_qc_area${AREA_THRESHOLD}_obs${SAMPLE_THRESHOLD}"

WSE_DATACUBE="$TRAINING_FOLDER/swot_lake_wse_datacube_wse_norm.nc"
ERA5_DATACUBE="$TRAINING_FOLDER/swot_lake_era5_climate_datacube.nc"
ECMWF_DATACUBE="$TRAINING_FOLDER/swot_lake_ecmwf_forecast_datacube.nc"
STATIC_DATACUBE="$TRAINING_FOLDER/swot_lake_static_datacube.nc"
WSE_STATS_CSV="$TRAINING_FOLDER/lake_wse_norm_stats.csv"
LAKE_GRAPH="$TRAINING_FOLDER/gritv06_great_mekong_pld_lake_graph_area_${AREA_THRESHOLD}_sample_${SAMPLE_THRESHOLD}.csv"

SAVE_DIR="/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/experiments/area${AREA_THRESHOLD}_obs${SAMPLE_THRESHOLD}"
CONFIG="configs/wsend/exp10_mekong_wsend_era5_ifshres_gritv06_202312_202602_temporalcv.yaml"
RUN_NAME="$(basename "$CONFIG" .yaml)"
SEED=42

# Use fold 2 (trained on most data: 2023-12-01 → 2025-09-30).
# Remove --fold-idx to auto-select the fold with the lowest val loss instead.
FOLD_IDX=2

# ── Code directory ─────────────────────────────────────────────────────────────
cd /home/cenv1160/code/smart_hs_mekong_reservoirs/swot_gnn

# ── Full inference (all init dates, no obs filter) ────────────────────────────
echo "Running full inference with fold ${FOLD_IDX} model ..."
python wsend_training/run_full_inference_wsend_temporal_cv.py \
    --config          "$CONFIG" \
    --wse-datacube    "$WSE_DATACUBE" \
    --era5-datacube   "$ERA5_DATACUBE" \
    --ecmwf-datacube  "$ECMWF_DATACUBE" \
    --static-datacube "$STATIC_DATACUBE" \
    --wse-stats-csv   "$WSE_STATS_CSV" \
    --lake-graph      "$LAKE_GRAPH" \
    --save-dir        "$SAVE_DIR" \
    --run-name        "$RUN_NAME" \
    --fold-idx        $FOLD_IDX \
    --seed            $SEED \
    --device          cuda

echo "Full inference complete."
echo "Output: $SAVE_DIR/$RUN_NAME/full_inference/full_predictions.csv"
