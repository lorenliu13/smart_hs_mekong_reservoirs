#!/bin/bash
#SBATCH --job-name=era5land_download_2025
#SBATCH --output=era5land_download_2025.out
#SBATCH --error=era5land_download_2025.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --partition=long

# Load any necessary modules
module load Anaconda3
conda activate $DATA/py311

# Run the download script
python download_era5_land_mekong.py
