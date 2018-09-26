#!/bin/bash

#PBS -l walltime=40:00:00,nodes=1:ppn=6
#PBS -m n
#PBS -k oe
#PBS -j oe

cd $PBS_O_WORKDIR

echo $PBS_JOBID
echo $PBS_O_HOST
echo $PBS_NODEFILE
echo $a1

source activate my_root

echo $p1
echo $p2
echo $p3
echo $p4
echo $p5
echo $p6
echo $p7
echo $p8
echo $p9
echo $p10
echo $p11

echo "${p1} ${p2} ${p3} ${p4} ${p5} ${p6} ${p7} ${p8} ${p9} ${p10} ${p11}"

time python $p1 $p2 $p3 $p4 $p5 $p6 $p7 $p8 $p9 $p10 $p11
