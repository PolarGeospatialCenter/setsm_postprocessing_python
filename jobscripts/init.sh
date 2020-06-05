#!/bin/bash

set -u
job_class="$1"  # e.g. 's2s', 'Mask', 'Check'
set +u


### Handle environment loading ###
default_env_load_cmd=''
env_load_cmd=''

## Here are some example commands that can load the environment needed by a batch script
#module load mod_name  # example load system module
#source /path/to/env_root/bin/activate  # example load virtual environment
#conda activate base  # example load conda environment

## Modify the two code blocks below to set default and per-script environment load commands

# Set default environment load command for all batch scripts (fallback if no specific script setting)
#default_env_load_cmd="source /mnt/pgc/data/scratch/erik/build/miniconda3/bin/activate /mnt/pgc/data/scratch/erik/build/miniconda3/envs/s2s"
#default_env_load_cmd="source /home/husby036/build/miniconda3/bin/activate /home/husby036/build/miniconda3/envs/s2s"

# Set environment load commands specific to batch scripts
if [ "$job_class" == "s2s" ]; then
    env_load_cmd="$default_env_load_cmd"
elif [ "$job_class" == "Mask" ]; then
    env_load_cmd="$default_env_load_cmd"
elif [ "$job_class" == "Check" ]; then
    env_load_cmd="$default_env_load_cmd"
fi


# Don't modify the following two code blocks

if [ "$env_load_cmd" == '' ] && [ "$default_env_load_cmd" != '' ]; then
    echo "(Job class '${job_class}' has no specific environment load command set, so the default will be used)"
    env_load_cmd=$default_env_load_cmd
fi

if [ "$env_load_cmd" != '' ]; then
    echo "Loading environment with command: ${env_load_cmd}"
    eval "$env_load_cmd"
    env_load_status=$?
    if (( env_load_status != 0 )); then
        echo "Environment load failed with return code: ${env_load_status}"
        exit $env_load_status
    fi
fi
