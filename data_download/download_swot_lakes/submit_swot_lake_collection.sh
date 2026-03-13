#!/bin/bash
#SBATCH --job-name=swot_lake_collect
#SBATCH --output=logs/swot_lake_collect_%j.out
#SBATCH --error=logs/swot_lake_collect_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10        # matches process_num = 10
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=short         # adjust to your cluster's partition name

mkdir -p logs

module load python/3.10          # adjust to your cluster's module name

python swot_file_collection_passes.py
