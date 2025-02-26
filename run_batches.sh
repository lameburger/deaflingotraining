#!/bin/bash
#SBATCH --job-name=process_videos_batches
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-99   # Adjust based on an upper-bound estimate on batches

# Activate your environment if needed
source activate py39
python landmarks.py
