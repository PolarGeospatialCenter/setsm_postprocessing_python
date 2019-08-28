#!/bin/bash

#PBS -l walltime=40:00:00,nodes=1:ppn=6

#PBS -j oe
#PBS -o $PBS_JOBNAME.o$PBS_JOBID
#CONDOPT_PBS -k oe IF %LOGDIR is None

#CONDOPT_PBS -m ae IF %EMAIL ELSE -m n

## Expected environment variables (passed with -v argument)
# p1 :: task command
# p2 :: path to python executable (usually just "python")
# p3 :: python minimum version number (e.g. "3.7.3")

task_cmd=$p1
python_exe=$p2
python_version_accepted_min=$p3

if [ "$task_cmd" == "" ]; then
    echo '$p1 variable (task command) not set, exiting'
    exit 1
fi

echo ________________________________________________________
echo
echo PBS Job Log
echo Start time: `date`
echo
echo Submitted by user: $USER
echo
echo Server name: $PBS_SERVER
echo Queue name: $PBS_QUEUE
echo Node name: $PBS_O_HOST
echo Node list file: $PBS_NODEFILE
echo
echo Job name: $PBS_JOBNAME
echo Job ID: $PBS_JOBID
echo
echo Task command: $p1
echo
echo Working directory: $PBS_O_WORKDIR
echo

### Un-comment this block to handle environment loading
##
## Modify the following lines to load your own environment
#env_load_cmd="module load mod_name"  # example load system module
#env_load_cmd="source /path/to/env_root/bin/activate"  # example load virtual environment
#env_load_cmd="conda activate env_name"  # example load conda environment
##
#echo Environment load command: $env_load_cmd
#echo
#echo Loading environment
#eval $env_load_cmd
#[ $? -eq 0 ] || exit 1

# Python version check
if [ "$python_version_accepted_min" != "" ]; then
    python_version=$($python_exe -c 'import platform; print(platform.python_version())')
    IFS='.'
    read -ra py_ver_nums <<< "$python_version"
    read -ra py_ver_min_nums <<< "$python_version_accepted_min"
    for ((n=0; n < ${#py_ver_nums[*]}; n++)); do
        if [ "${py_ver_nums[n]}" == "" ]; then py_ver_nums[n]=0; fi
        if [ "${py_ver_min_nums[n]}" == "" ]; then py_ver_min_nums[n]=0; fi
        if (("${py_ver_nums[n]}" > "${py_ver_min_nums[n]}")); then
            break
        elif (("${py_ver_nums[n]}" < "${py_ver_min_nums[n]}")); then
          echo "Python version ($python_version) is below accepted minimum ($python_version_accepted_min), exiting"
          exit 1
        fi
    done
fi

echo Changing to working directory
cd $PBS_O_WORKDIR
[ $? -eq 0 ] || exit 1

echo Running task
echo ________________________________________________________
echo

time eval $task_cmd
