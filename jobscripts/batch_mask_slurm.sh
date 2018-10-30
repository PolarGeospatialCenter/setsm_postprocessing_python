#!/bin/bash

# number of nodes
#SBATCH -N 1

# number of cpus per task
#SBATCH -c 6

# job log path
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

# init gdal tools
source activate my_root

echo
echo $p1
echo

time $p1
