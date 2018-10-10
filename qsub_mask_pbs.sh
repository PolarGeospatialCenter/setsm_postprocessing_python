#!/bin/bash

#PBS -l walltime=40:00:00,nodes=1:ppn=6
#PBS -m n
#PBS -k oe
#PBS -j oe

cd $PBS_O_WORKDIR

echo $PBS_JOBID
echo $PBS_O_HOST
echo $PBS_NODEFILE
echo

source activate my_root

echo
echo $p1
echo

time $p1
