#!/bin/bash

#PBS -l walltime=1:00:00,nodes=2,mem=5gb

#PBS -o $PBS_JOBNAME.o$PBS_JOBID
#PBS -j oe
#PBS -k oe

#PBS -m n


## NOTE: Look to 'body.sh' script for expected environment variable arguments

jobscript_path="$p0"
if [ -z "$jobscript_path" ]; then
    echo '$p0 variable (jobscript path) not set, exiting'
    exit 1
fi
jobscript_body="$(dirname "$jobscript_path")/body.sh"
if [ ! -f "$jobscript_body" ]; then
    echo "Job body script does not exist: ${jobscript_body}"
    exit 1
fi

$jobscript_body
