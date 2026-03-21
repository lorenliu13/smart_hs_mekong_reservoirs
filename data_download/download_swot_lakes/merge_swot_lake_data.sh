#!/bin/bash
#SBATCH --job-name=merge_swot_lake_data          # Name of your job
#SBATCH --output=merge_swot_lake_data.out        # Standard output file
#SBATCH --error=merge_swot_lake_data.err         # Standard error file
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task
#SBATCH --cpus-per-task=12                # Request 10 CPUs per task
#SBATCH --mem-per-cpu=4G                  # Request 4GB memory per CPU
#SBATCH --time=4:00:00                   # Job run time (4 hours)
#SBATCH --partition=short                 # Specify the partition/queue

# Load any necessary modules
module load Anaconda3
conda activate $DATA/py311

# Run the download script
python merge_swot_lake_data.py
