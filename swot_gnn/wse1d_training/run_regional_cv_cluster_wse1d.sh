#!/bin/bash
#SBATCH --job-name=run_lake_regionalcv
#SBATCH --output=run_lake_regionalcv.out
#SBATCH --error=run_lake_regionalcv.err
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
TRAINING_FOLDER="/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data/mekong_lakes_swotpld_era5_ifshres10d_gritv06_202312_202602_qc_area0.1_obs20"

WSE_DATACUBE="$TRAINING_FOLDER/swot_lake_wse_datacube_wse_norm.nc"
ERA5_DATACUBE="$TRAINING_FOLDER/swot_lake_era5_climate_datacube.nc"
ECMWF_DATACUBE="$TRAINING_FOLDER/swot_lake_ecmwf_forecast_datacube.nc"
STATIC_DATACUBE="$TRAINING_FOLDER/swot_lake_static_datacube.nc"
WSE_STATS_CSV="$TRAINING_FOLDER/lake_wse_norm_stats.csv"
LAKE_GRAPH="$TRAINING_FOLDER/gritv06_great_mekong_pld_lake_graph_area_0.1_sample_20.csv"

SAVE_DIR="/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/experiments"
CONFIG="configs/exp07_mekong_wse1d_era5_ifshres_gritv06_202312_202602_regionalcv.yaml"
RUN_NAME="$(basename "$CONFIG" .yaml)"
SEED=42

N_FOLDS=5
VAL_METHOD="spatial"
SPATIAL_VAL_FRAC=0.2
SPATIAL_VAL_SEED=43
HYBAS_COL="hybasin_level_4"

# ── Code directory ─────────────────────────────────────────────────────────────
cd /home/cenv1160/code/smart_hs_mekong_reservoirs/swot_gnn

# ── 5-fold regional cross-validation loop ─────────────────────────────────────
# Region → fold index:
#   0: Upper Mekong + Northern Highlands
#   1: Red River + Pearl River
#   2: Vietnam Coastal + Mekong Delta
#   3: Khorat Plateau
#   4: Tonle Sap + 3S Basin
for FOLD_IDX in $(seq 0 $((N_FOLDS - 1))); do
    echo "============================================================"
    echo "  Fold ${FOLD_IDX} / $((N_FOLDS - 1))"
    echo "============================================================"

    # ── Training ───────────────────────────────────────────────────────────────
    python run_training_lake_wse1d_regional_cv.py \
        --config           "$CONFIG" \
        --wse-datacube     "$WSE_DATACUBE" \
        --era5-datacube    "$ERA5_DATACUBE" \
        --ecmwf-datacube   "$ECMWF_DATACUBE" \
        --static-datacube  "$STATIC_DATACUBE" \
        --lake-graph       "$LAKE_GRAPH" \
        --save-dir         "$SAVE_DIR" \
        --run-name         "$RUN_NAME" \
        --fold-idx         $FOLD_IDX \
        --val-method       $VAL_METHOD \
        --spatial-val-frac $SPATIAL_VAL_FRAC \
        --spatial-val-seed $SPATIAL_VAL_SEED \
        --hybas-col        $HYBAS_COL \
        --seed             $SEED \
        --device cuda

    # ── Inference (runs only if training succeeded) ────────────────────────────
    python run_inference_regional_cv.py \
        --config           "$CONFIG" \
        --wse-datacube     "$WSE_DATACUBE" \
        --era5-datacube    "$ERA5_DATACUBE" \
        --ecmwf-datacube   "$ECMWF_DATACUBE" \
        --static-datacube  "$STATIC_DATACUBE" \
        --wse-stats-csv    "$WSE_STATS_CSV" \
        --lake-graph       "$LAKE_GRAPH" \
        --save-dir         "$SAVE_DIR" \
        --run-name         "$RUN_NAME" \
        --fold-idx         $FOLD_IDX \
        --val-method       $VAL_METHOD \
        --spatial-val-frac $SPATIAL_VAL_FRAC \
        --spatial-val-seed $SPATIAL_VAL_SEED \
        --hybas-col        $HYBAS_COL \
        --seed             $SEED \
        --device cuda

    echo "  Fold ${FOLD_IDX} complete."
    echo ""
done

echo "All ${N_FOLDS} folds complete."
