#!/bin/bash
#SBATCH --job-name=ens_download          # Name of your job
#SBATCH --output=ens_download_2023_2026.out        # Standard output file
#SBATCH --error=ens_download_2023_2026.err         # Standard error file
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task
#SBATCH --cpus-per-task=1                # Request 1 CPUs per task
#SBATCH --mem-per-cpu=4G                  # Request 4GB memory per CPU
#SBATCH --time=168:00:00                   # Job run time (72 hours)
#SBATCH --partition=long                 # Specify the partition/queue

# Load any necessary modules
module load Anaconda3
conda activate $DATA/py311

# Run the download script
python -u download_ens_mekong.py