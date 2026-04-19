#!/bin/bash
#SBATCH --job-name=run_lake_singleld_lstm_wsend
#SBATCH --output=run_lake_singleld_lstm_wsend.out
#SBATCH --error=run_lake_singleld_lstm_wsend.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=short         # GPU nodes are on the htc cluster, short partition (max 12h)
#SBATCH --gres=gpu:1

set -e   # abort immediately if any command fails

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
CONFIG="configs/lstm/exp02_mekong_lstm_wsend_era5_ifshres_gritv06_202312_202602_temporalcv.yaml"
RUN_NAME="$(basename "$CONFIG" .yaml)_singleld"
SEED=42
LEAD_ASSIGN="random"   # "round_robin" or "random"

N_FOLDS=3

# ── Code directory ─────────────────────────────────────────────────────────────
cd /home/cenv1160/code/smart_hs_mekong_reservoirs/swot_gnn

# ── 3-fold temporal expanding-window cross-validation loop ────────────────────
# Fold → date ranges (all lakes active in every fold):
#   0: Train 2023-12-01 → 2024-11-30  |  Test 2024-12-01 → 2025-04-30
#   1: Train 2023-12-01 → 2025-04-30  |  Test 2025-05-01 → 2025-09-30
#   2: Train 2023-12-01 → 2025-09-30  |  Test 2025-10-01 → 2026-02-28
for FOLD_IDX in $(seq 0 $((N_FOLDS - 1))); do
    echo "============================================================"
    echo "  Fold ${FOLD_IDX} / $((N_FOLDS - 1))  |  lead_assign=${LEAD_ASSIGN}"
    echo "============================================================"

    # ── Training ───────────────────────────────────────────────────────────────
    python wsend_lstm_training/single_lead/run_training_lake_lstm_wsend_single_lead_cv.py \
        --config          "$CONFIG" \
        --wse-datacube    "$WSE_DATACUBE" \
        --era5-datacube   "$ERA5_DATACUBE" \
        --ecmwf-datacube  "$ECMWF_DATACUBE" \
        --static-datacube "$STATIC_DATACUBE" \
        --lake-graph      "$LAKE_GRAPH" \
        --save-dir        "$SAVE_DIR" \
        --run-name        "$RUN_NAME" \
        --fold-idx        $FOLD_IDX \
        --lead-assign     "$LEAD_ASSIGN" \
        --seed            $SEED \
        --device          cuda

    # ── Inference (runs only if training succeeded) ────────────────────────────
    python wsend_lstm_training/single_lead/run_inference_lstm_wsend_single_lead_cv.py \
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
        --lead-assign     "$LEAD_ASSIGN" \
        --seed            $SEED \
        --device          cuda

    echo "  Fold ${FOLD_IDX} complete."
    echo ""
done

echo "All ${N_FOLDS} folds complete."
