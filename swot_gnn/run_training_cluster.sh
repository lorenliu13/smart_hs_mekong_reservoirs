#!/bin/bash
#SBATCH --job-name=swot_gnn_train
#SBATCH --output=swot_gnn_train.out
#SBATCH --error=swot_gnn_train.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=short         # GPU nodes are on the htc cluster, short partition (max 12h)
#SBATCH --gres=gpu:1 --constraint='gpu_cc:8.0'
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

module load Anaconda3
conda activate $DATA/py311_torch

# Adjust paths to your data on the cluster
DYNAMIC_DATACUBE="/path/to/dynamic_datacube.nc"
STATIC_DATACUBE="/path/to/static_datacube.nc"

cd /path/to/smart_hs_mekong_mega_reservoirs/swot_gnn

python run_training.py \
    --config configs/lake_datacube.yaml \
    --dynamic-datacube "$DYNAMIC_DATACUBE" \
    --static-datacube "$STATIC_DATACUBE" \
    --save-dir checkpoints \
    --run-name model_v1_lake \
    --device cuda
