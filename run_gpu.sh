#!/bin/bash
#SBATCH --account=es3890_acc
#SBATCH --partition=pascal
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=20G
#SBATCH --time=2:00:00
#SBATCH --output=gpu-job.log

source activate py39   # Activate your environment (adjust if necessary)
python model2.py
