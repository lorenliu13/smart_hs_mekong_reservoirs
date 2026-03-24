#!/bin/bash
#SBATCH --job-name=download_hres_mekong_test
#SBATCH --output=hres_download_test.out
#SBATCH --error=hres_download_test.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --partition=short

module load Anaconda3
conda activate $DATA/py311

# Download only 2024-01 as a test
python download_hres_mekong.py --start-year 2026 --start-month 1 --end-year 2026 --end-month 2
