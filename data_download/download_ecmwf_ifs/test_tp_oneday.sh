#!/bin/bash
#SBATCH --job-name=test_tp
#SBATCH --output=test_tp_oneday.out
#SBATCH --error=test_tp_oneday.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --partition=short

module load Anaconda3
conda activate $DATA/py311

python test_tp_oneday.py
