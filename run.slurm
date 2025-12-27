#!/bin/bash
#SBATCH --job-name=mpi_csv
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=cpu01,cpu02,gpu01,gpu02
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

export USE_CUDA=1
# === CUDA доступна ТОЛЬКО на gpu-узлах ===
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPU_WEIGHT=1
export GPU_WEIGHT=1

mpirun ./mytask
