#!/bin/bash
#SBATCH --job-name=run_lake_exp01
#SBATCH --output=run_lake_exp01.out
#SBATCH --error=run_lake_exp01.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=short         # GPU nodes are on the htc cluster, short partition (max 12h)
#SBATCH --gres=gpu:l40s:1
# --constraint='gpu_cc:8.0' targets cc>=8.0 GPUs: A100, H100, L40S, RTXA6000
# This ensures BF16 and torch.compile() support; widens scheduling vs naming a specific model.
#
# Lower the constraint to relax scheduling pressure:
#   gpu_cc:7.0  → also includes V100 (cc7.0)               htc-g[035,037-038,045-049]
#   gpu_cc:6.0  → also includes P100 (cc6.0)               htc-g[032-034]  (fp32 only, no BF16/compile)
#
# Or pin to a specific model for reproducibility:
#   --gres=gpu:l40s:1   L40S  46GB cc12.9  htc-g[061-084] (24 nodes x4) — most available
#   --gres=gpu:a100:1   A100  40GB cc8.0   htc-g[015-019]  (5 nodes x4)
#   --gres=gpu:h100:1   H100  82GB cc12.6  htc-g[053-058]

set -e   # abort immediately if any command fails

module load Anaconda3
source activate $DATA/py311_torch

# ── Shared paths ───────────────────────────────────────────────────────────────
TRAINING_FOLDER="/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/training_data/training_data_lake_based_great_mekong_20260325"

WSE_DATACUBE="$TRAINING_FOLDER/swot_lake_wse_datacube_wse_norm.nc"
ERA5_DATACUBE="$TRAINING_FOLDER/swot_lake_era5_climate_datacube.nc"
ECMWF_DATACUBE="$TRAINING_FOLDER/swot_lake_ecmwf_forecast_datacube.nc"
STATIC_DATACUBE="$TRAINING_FOLDER/swot_lake_static_datacube.nc"
WSE_STATS_CSV="$TRAINING_FOLDER/lake_wse_norm_stats.csv"
LAKE_GRAPH="$TRAINING_FOLDER/gritv06_great_mekong_pld_lake_graph_0sqkm.csv"

SAVE_DIR="/data/ouce-grit/cenv1160/smart_hs/processed_data/mekong_river_basin_reservoirs/swot_gnn/experiments"
RUN_NAME="exp01_mekong_wse1d_era5_ifshres_gritv06_202312_202512_v01"

# ── Code directory ─────────────────────────────────────────────────────────────
cd /data/ouce-grit/cenv1160/smart_hs/smart_hs_mekong_mega_reservoirs/swot_gnn

# ── Training ───────────────────────────────────────────────────────────────────
python run_lake_exp01.py \
    --config configs/exp01_mekong_wse1d_era5_ifshres_gritv06_202312_202512_v01.yaml \
    --wse-datacube    "$WSE_DATACUBE" \
    --era5-datacube   "$ERA5_DATACUBE" \
    --ecmwf-datacube  "$ECMWF_DATACUBE" \
    --static-datacube "$STATIC_DATACUBE" \
    --lake-graph      "$LAKE_GRAPH" \
    --save-dir        "$SAVE_DIR" \
    --run-name        "$RUN_NAME" \
    --device cuda

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
    --device cuda


