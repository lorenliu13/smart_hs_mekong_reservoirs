#!/bin/bash
#SBATCH --job-name=unzip_swot_files          # Name of your job
#SBATCH --output=unzip_swot_files.out        # Standard output file
#SBATCH --error=unzip_swot_files.err         # Standard error file
#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task
#SBATCH --cpus-per-task=10                # Request 10 CPUs per task
#SBATCH --mem-per-cpu=2G                  # Request 4GB memory per CPU
#SBATCH --time=4:00:00                   # Job run time (4 hours)
#SBATCH --partition=short                 # Specify the partition/queue

# Load any necessary modules
module load Anaconda3
conda activate $DATA/py311

# Run the download script
python unzip_swot_files.py