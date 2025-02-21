#!/bin/bash
#SBATCH --job-name=process_videos_batches
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --array=0-99   # Adjust based on an upper-bound estimate on batches

# Activate your environment if needed, for example:
source activate py39
python landmarks.py
