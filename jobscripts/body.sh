#!/bin/bash

set -uo pipefail

### NOTE: Modify environment settings for specific batch scripts in the 'init.sh' file! ###

## Expected environment variables (passed with -v argument for PBS, --export argument for SLURM)
# p0 :: path to head jobscript file
# p1 :: job class (e.g. 's2s', 'Mask', 'Check')
# p2 :: core task command to be executed
# p3 :: python minimum version number (e.g. "3.7.3")

if [ ! -z "$PBS_JOBID" ]; then
    scheduler='PBS'
elif [ ! -z "$SLURM_JOBID" ]; then
    scheduler='SLURM'
else
    echo "Could not determine job scheduler type by checking environment variables"
    exit 1
fi

# Required environment variable arguments
jobscript_path="$p0"
job_class="$p1"
task_cmd="$p2"
set +u
# Optional environment variable arguments
py_ver_min="$p3"
set -u


jobscript_init="$(dirname "$jobscript_path")/init.sh"
if [ ! -f "$jobscript_init" ]; then
    echo "Job init script does not exist: ${jobscript_init}"
    exit 1
fi


## Print job preamble
set +u
if [ "$scheduler" == "PBS" ]; then
    echo ________________________________________________________
    echo
    echo PBS Job Log
    echo Start time: $(date)
    echo
    echo Job name: $PBS_JOBNAME
    echo Job ID: $PBS_JOBID
    echo Submitted by user: $USER
    echo User effective group ID: $(id -ng)
    echo
    echo Hostname of submission: $PBS_O_HOST
    echo Submitted to cluster: $PBS_SERVER
    echo Submitted to queue: $PBS_QUEUE
    echo Requested nodes per job: $PBS_NUM_NODES
    echo Requested cores per node: $PBS_NUM_PPN
    echo Requested cores per job: $PBS_NP
    #echo Node list file: $PBS_NODEFILE
    #echo Nodes assigned to job: $(cat $PBS_NODEFILE)
    #echo Running node index: $PBS_O_NODENUM
    echo
    echo Running on hostname: $HOSTNAME
    echo Parent PID: $PPID
    echo Process PID: $$
    echo Number of physical cores: $(grep "^core id" /proc/cpuinfo | sort -u | wc -l)
    echo Number of virtual cores: $(nproc --all)
    mem_report_cmd="free -g"
    echo "Memory report [GB] ('${mem_report_cmd}'):"
    echo -------------------------------------------------------------------------
    ${mem_report_cmd}
    echo -------------------------------------------------------------------------
    echo
    working_dir=$PBS_O_WORKDIR

elif [ "$scheduler" == "SLURM" ]; then
    echo ________________________________________
    echo
    echo SLURM Job Log
    echo Start time: $(date)
    echo
    echo Job name: $SLURM_JOB_NAME
    echo Job ID: $SLURM_JOBID
    echo Submitted by user: $USER
    echo User effective group ID: $(id -ng)
    echo
    echo SLURM account used: $SLURM_ACCOUNT
    echo Hostname of submission: $SLURM_SUBMIT_HOST
    echo Submitted to cluster: $SLURM_CLUSTER_NAME
    echo Submitted to node: $SLURMD_NODENAME
    echo Cores on node: $SLURM_CPUS_ON_NODE
    echo Requested cores per task: $SLURM_CPUS_PER_TASK
    echo Requested cores per job: $SLURM_NTASKS
    echo Requested walltime: $SBATCH_TIMELIMIT
    #echo Nodes assigned to job: $SLURM_JOB_NODELIST
    #echo Running node index: $SLURM_NODEID
    echo
    echo Running on hostname: $HOSTNAME
    echo Parent PID: $PPID
    echo Process PID: $$
    echo Number of physical cores: $(grep "^core id" /proc/cpuinfo | sort -u | wc -l)
    echo Number of virtual cores: $(nproc --all)
    mem_report_cmd="free -g"
    echo "Memory report [GB] ('${mem_report_cmd}'):"
    echo -------------------------------------------------------------------------
    ${mem_report_cmd}
    echo -------------------------------------------------------------------------
    echo
    working_dir=$SLURM_SUBMIT_DIR
fi
set -u


## Source init file / load environment
init_cmd="source ${jobscript_init} ${job_class}"
echo "Sourcing init script with command: ${init_cmd}"
set +u; eval "$init_cmd"; set -u
[ $? -eq 0 ] || exit 1


## Python version check
if [ ! -z "$py_ver_min" ]; then
    py_ver=$(python -c 'import platform; print(platform.python_version())')
    IFS='.' read -ra py_ver_nums <<< "$py_ver"
    IFS='.' read -ra py_ver_min_nums <<< "$py_ver_min"
    for ((n=0; n < ${#py_ver_nums[*]}; n++)); do
        if [ "${py_ver_nums[n]}" == "" ]; then py_ver_nums[n]=0; fi
        if [ "${py_ver_min_nums[n]}" == "" ]; then py_ver_min_nums[n]=0; fi
        if (("${py_ver_nums[n]}" > "${py_ver_min_nums[n]}")); then
            break
        elif (("${py_ver_nums[n]}" < "${py_ver_min_nums[n]}")); then
            echo "Python version (${py_ver}) is below accepted minimum (${py_ver_min}), exiting"
            exit 1
        fi
    done
fi


echo "Changing to working directory: ${working_dir}"
cd "$working_dir"
[ $? -eq 0 ] || exit 1


echo
echo "Executing task command: ${task_cmd}"
echo ________________________________________________________
echo


time eval "$task_cmd"
