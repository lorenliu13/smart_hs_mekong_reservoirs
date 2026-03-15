#!/bin/bash
#SBATCH --job-name=check_download_status          # Name of your job
#SBATCH --output=check_download_status.out        # Standard output file
#SBATCH --error=check_download_status.err         # Standard error file
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task
#SBATCH --cpus-per-task=1                # Request 1 CPU per task
#SBATCH --mem-per-cpu=4G                  # Request 4GB memory per CPU
#SBATCH --time=01:00:00                   # Job run time (1 hour)
#SBATCH --partition=short                 # Specify the partition/queue

# Load any necessary modules
module load Anaconda3
conda activate $DATA/py311

# Run the check script
python check_download_status.py
