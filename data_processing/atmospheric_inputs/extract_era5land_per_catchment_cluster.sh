#!/bin/bash
#SBATCH --job-name=extract_era5land_per_catchment_cluster          # Name of your job
#SBATCH --output=extract_era5land_per_catchment_cluster.out        # Standard output file
#SBATCH --error=extract_era5land_per_catchment_cluster.err         # Standard error file
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task
#SBATCH --cpus-per-task=20                # Request 10 CPU per task
#SBATCH --mem-per-cpu=4G                  # Request 4GB memory per CPU
#SBATCH --time=04:00:00                   # Job run time (1 hour)
#SBATCH --partition=short                 # Specify the partition/queue

# Load any necessary modules
module load Anaconda3
conda activate $DATA/py311

# Run the download script
python extract_era5land_per_catchment_cluster.py

