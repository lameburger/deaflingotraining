#!/bin/bash
#SBATCH --account=accre_gpu_acc
#SBATCH --partition=Turing
#SBATCH --gres=gpu:1           # Request 1 GPU from the Turing partition (RTX 2080 Ti)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      # Request 4 CPU cores for data loading/preprocessing
#SBATCH --mem=16G              # Request 16GB of host memory
#SBATCH --time=2:00:00
#SBATCH --output=gpu-job.log

source activate py39   # Activate your environment (adjust if necessary)
python model.py
