#!/bin/bash
#SBATCH --job-name=mpi_csv
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=cpu01,cpu02,gpu01,gpu02
#SBATCH --time=00:20:00

mpirun -np 4 ./mytask