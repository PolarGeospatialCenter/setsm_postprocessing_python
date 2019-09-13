#!/bin/bash

#SBATCH --time=40:00:00  # walltime limit (HH:MM:SS)
#SBATCH --nodes=1  # number of nodes
#SBATCH --cpus-per-task=2  # processor core(s) per task

#SBATCH -o %x.o%j  # stdout (+stderr) file name

#CONDOPT_SBATCH --mail-type=FAIL,END IF %EMAIL
#CONDOPT_SBATCH --mail-user=%EMAIL IF type(%EMAIL) is str

# Load environment specific to jobscript
#env_load_cmd="module load mod_name"  # example load system module
#env_load_cmd="source /path/to/env_root/bin/activate"  # example load virtual environment
#env_load_cmd="conda activate env_name"  # example load conda environment

## Expected environment variables (passed with -v argument)
# p1 :: task command
# p2 :: path to python executable (usually just "python")
# p3 :: python minimum version number (e.g. "3.7.3")
# p4 :: init file to source for default variable values

task_cmd="$p1"
py_exe="$p2"
py_ver_min="$p3"
init_file="$p4"

if [ "$task_cmd" == "" ]; then
    echo '$p1 variable (task command) not set, exiting'
    exit 1
fi
if [ "$init_file" != "" ]; then
    if [ -f $init_file ]; then
        echo Sourcing init file: $init_file
        source "$init_file"
    else
        echo Init file does not exist: $init_file
        exit 1
    fi
fi

echo ________________________________________
echo
echo SLURM Job Log
echo Start time: `date`
echo
echo Submitted by user: $USER
echo SLURM account used: $SLURM_ACCOUNT
echo Hostname of submission: $SLURM_SUBMIT_HOST
echo
echo Cluster name: $SLURM_CLUSTER_NAME
echo Node name: $SLURMD_NODENAME
echo
echo CPUs on node: $SLURM_CPUS_ON_NODE
echo CPUs per task: $SLURM_CPUS_PER_TASK
echo Time limit: $SBATCH_TIMELIMIT
echo
echo Job name: $SLURM_JOB_NAME
echo Job ID: $SLURM_JOBID
echo
echo Hostname: $HOSTNAME
echo
echo Working directory: $SLURM_SUBMIT_DIR
echo
echo Task command: $task_cmd
echo

# Environment load
if [ "$env_load_cmd" != "" ]; then
    echo Environment load command: $env_load_cmd
    echo
    echo Loading environment
    eval "$env_load_cmd"
    [ $? -eq 0 ] || exit 1
fi

# Python version check
if [ "$py_ver_min" != "" ]; then
    py_ver=$($py_exe -c 'import platform; print(platform.python_version())')
    IFS='.'
    read -ra py_ver_nums <<< "$py_ver"
    read -ra py_ver_min_nums <<< "$py_ver_min"
    for ((n=0; n < ${#py_ver_nums[*]}; n++)); do
        if [ "${py_ver_nums[n]}" == "" ]; then py_ver_nums[n]=0; fi
        if [ "${py_ver_min_nums[n]}" == "" ]; then py_ver_min_nums[n]=0; fi
        if (("${py_ver_nums[n]}" > "${py_ver_min_nums[n]}")); then
            break
        elif (("${py_ver_nums[n]}" < "${py_ver_min_nums[n]}")); then
          echo "Python version ($py_ver) is below accepted minimum ($py_ver_min), exiting"
          exit 1
        fi
    done
fi

echo Changing to working directory
cd $SLURM_SUBMIT_DIR
[ $? -eq 0 ] || exit 1

echo Running task
echo ________________________________________
echo

time eval $task_cmd
