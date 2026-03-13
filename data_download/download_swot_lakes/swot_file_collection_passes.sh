#!/bin/bash
#SBATCH --job-name=swot_file_collection_passes          # Name of your job
#SBATCH --output=swot_file_collection_passes.out        # Standard output file
#SBATCH --error=swot_file_collection_passes.err         # Standard error file
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task
#SBATCH --cpus-per-task=10                # Request 10 CPUs per task
#SBATCH --mem-per-cpu=4G                  # Request 4GB memory per CPU
#SBATCH --time=12:00:00                   # Job run time (4 hours)
#SBATCH --partition=short                 # Specify the partition/queue

# Load any necessary modules
module load Anaconda3
conda activate $DATA/py311

# Run the download script
python swot_file_collection_passes.py
