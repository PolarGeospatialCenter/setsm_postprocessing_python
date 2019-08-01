#!/bin/bash

#SBATCH --time=40:00:00  # walltime limit (HH:MM:SS)
#SBATCH --nodes=1  # number of nodes
#SBATCH --cpus-per-task=6  # processor core(s) per task

#SBATCH -o %x.o%j  # stdout (+stderr) file name

#CONDOPT_SBATCH --mail-type=FAIL,END IF %EMAIL
#CONDOPT_SBATCH --mail-user=%EMAIL IF type(%EMAIL) is str

echo ________________________________________
echo
echo SLURM Job Log
echo Start time: `date`
echo
echo Submitted by user: $USER
echo SLURM account used: $SLURM_ACCOUNT
echo
echo Cluster name: $SLURM_CLUSTER_NAME
echo
echo Node name: $SLURMD_NODENAME
echo CPUs on node: $SLURM_CPUS_ON_NODE
echo
echo Job name: $SLURM_JOB_NAME
echo Job ID: $SLURM_JOBID
echo
echo Time limit: $SBATCH_TIMELIMIT
echo CPUs per task: $SLURM_CPUS_PER_TASK
echo
echo Task command: $p1
echo
echo Hostname of submission: $SLURM_SUBMIT_HOST
echo Working directory: $SLURM_SUBMIT_DIR
echo
echo Changing to working directory

cd $SLURM_SUBMIT_DIR

# Un-comment the following block to handle environment loading
#echo Loading environment
#
## Modify the following line to load your own environment
#source activate my_env  # load Python environment

echo Running task
echo ________________________________________
echo

time eval $p1
