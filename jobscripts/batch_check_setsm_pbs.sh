#!/bin/bash

#PBS -l walltime=40:00:00,nodes=1:ppn=2

#PBS -j oe
#PBS -o $PBS_JOBNAME.o$PBS_JOBID
#CONDOPT_PBS -k oe IF %LOGDIR is None

#CONDOPT_PBS -m ae IF %EMAIL ELSE -m n

echo ________________________________________________________
echo
echo PBS Job Log
echo Start time: `date`
echo
echo Submitted by user: $USER
echo
echo Server name: $PBS_SERVER
echo Queue name: $PBS_QUEUE
echo Node list file: $PBS_NODEFILE
echo
echo Node name: $PBS_O_HOST
echo
echo Job name: $PBS_JOBNAME
echo Job ID: $PBS_JOBID
echo
echo Task command: $p1
echo
echo Working directory: $PBS_O_WORKDIR
echo
echo Changing to working directory

cd $PBS_O_WORKDIR

echo Loading environment

# Modify this to load your own environment
source activate my_root  # load Python environment

echo Running task
echo ________________________________________________________
echo

time eval $p1
