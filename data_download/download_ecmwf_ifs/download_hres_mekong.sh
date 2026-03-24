#!/bin/bash
#SBATCH --job-name=hres_download          # Name of your job
#SBATCH --output=hres_download.out        # Standard output file
#SBATCH --error=hres_download.err         # Standard error file
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task
#SBATCH --cpus-per-task=1                # Request 1 CPUs per task
#SBATCH --mem-per-cpu=4G                  # Request 4GB memory per CPU
#SBATCH --time=24:00:00                   # Job run time (4 hours)
#SBATCH --partition=long                 # Specify the partition/queue

# Load any necessary modules
module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $DATA/py311

# Run the download script
python download_hres_mekong.py
