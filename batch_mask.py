#!/usr/bin/env python2

# Version 0.1; Erik Husby; Polar Geospatial Center, University of Minnesota; 2018


from __future__ import division
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

from lib.filter_scene import MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT
from lib import batch_handler
import lib.raster_array_tools as rat

JOB_ABBREV = 'Mask'

PYTHON_EXE = 'python'

SCRIPT_FILE = os.path.realpath(__file__)
SCRIPT_FNAME = os.path.basename(SCRIPT_FILE)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE)

SRC_SUFFIXES = ['dem_smooth.tif', 'dem_smooth_10m.tif', 'dem_smooth_browse.tif',
                'dem.tif', 'dem_10m.tif', 'dem_browse.tif',
                'matchtag.tif', 'ortho.tif', 'ortho_browse.tif', 'dem_coverage.tif']
BITMASK_SUFFIX = 'bitmask.tif'

ARGSTR_SRC = 'src'
ARGSTR_DSTDIR = '--dstdir'
ARGSTR_SRCDIR_SUFFIX = '--srcdir-suffix'
ARGSTR_EDGE = '--edge'
ARGSTR_WATER = '--water'
ARGSTR_CLOUD = '--cloud'
ARGSTR_RES = '--tr'
ARGSTR_SCHEDULER = '--scheduler'
ARGSTR_TASKS_PER_JOB = '--tasks-per-job'
ARGSTR_JOBSCRIPT = '--jobscript'
ARGSTR_SCRATCH = '--scratch'
ARGSTR_DRYRUN = '--dryrun'

ARGDEF_SRCDIR_SUFFIX = 'dem_smooth.tif'
ARGDEF_SCRATCH = os.path.join(os.path.expanduser('~'), 'scratch', 'task_bundles')
ARGDEF_JOBSCRIPT = os.path.join(SCRIPT_DIR, 'qsub_mask.sh')


def main():
    parser = argparse.ArgumentParser(description=(
        "Selectively apply filter components from the SETSM DEM scene/strip "
        "*_{} component raster to mask out corresponding locations ".format(BITMASK_SUFFIX)
       +"in another component raster(s), then save the resulting image(s)."))

    parser.add_argument(ARGSTR_SRC,
        help="Path to source DEM directory or raster file. "
             "Accepts a task bundle text file listing paths to *_{} ".format(BITMASK_SUFFIX)
            +"raster files along with with {} argument indicating ".format(ARGSTR_SRCDIR_SUFFIX)
            +"which components to mask.")
    parser.add_argument(ARGSTR_DSTDIR,
        help="Path to destination directory for output masked raster(s) "
             "(default is directory of `{}`).".format(ARGSTR_SRC))

    parser.add_argument(ARGSTR_SRCDIR_SUFFIX, default=ARGDEF_SRCDIR_SUFFIX,
        help="If `{}` is a directory, mask all raster images with a ".format(ARGSTR_SRC)
            +"file suffix(es) matching this string. Syntax is like "
             "'dem_smooth.tif|ortho.tif' to mask both the LSF DEM "
             "and ortho components. (default={})".format(ARGDEF_SRCDIR_SUFFIX))

    parser.add_argument('-e', ARGSTR_EDGE, action='store_true', default=False,
        help="Apply edge filter. Not necessary when masking strip DEMs, "
             "as it is already applied in the mosaicking step of scenes2strips.")
    parser.add_argument('-w', ARGSTR_WATER, action='store_true', default=False,
        help="Apply water filter.")
    parser.add_argument('-c', ARGSTR_CLOUD, action='store_true', default=False,
        help="Apply cloud filter.")

    parser.add_argument(ARGSTR_RES, type=int,
        help="Resample output raster(s) to this resolution (meters).")

    parser.add_argument(ARGSTR_SCHEDULER, choices=batch_handler.SCHED_SUPPORTED,
        help="Submit tasks to job scheduler.")
    parser.add_argument(ARGSTR_TASKS_PER_JOB, type=int,
        help="Number of tasks to bundle into a single job. (requires {} option)".format(ARGSTR_SCHEDULER))
    parser.add_argument(ARGSTR_JOBSCRIPT,
        help="Script to run in job submission to scheduler. (default={})".format(ARGDEF_JOBSCRIPT))
    parser.add_argument(ARGSTR_SCRATCH, default=ARGDEF_SCRATCH,
        help="Scratch directory to build task bundle text files. (default={})".format(ARGDEF_SCRATCH))

    parser.add_argument(ARGSTR_DRYRUN, action='store_true', default=False,
        help="Print actions without executing.")

    argstr_batch = [ARGSTR_SCHEDULER, ARGSTR_TASKS_PER_JOB]

    # Parse and validate arguments.
    args = batch_handler.ArgumentPasser(parser, PYTHON_EXE, sys.argv[0])
    src = os.path.abspath(args.get(ARGSTR_SRC))
    dstdir = args.get(ARGSTR_DSTDIR)
    src_suffixes = [s.strip().lstrip('_') for s in args.get(ARGSTR_SRCDIR_SUFFIX).split('|')]
    scheduler = args.get(ARGSTR_SCHEDULER)
    jobscript = args.get(ARGSTR_JOBSCRIPT)
    scratchdir = args.get(ARGSTR_SCRATCH)
    dryrun = args.get(ARGSTR_DRYRUN)

    if not (os.path.isdir(src) or os.path.isfile(src)):
        parser.error("`{}` must be a path to either a directory or a file".format(ARGSTR_SRC))
    if dstdir is None:
        dstdir = src if os.path.isdir(src) else os.path.dirname(src)
    elif not os.path.isdir(dstdir):
        os.makedirs(dstdir)
    if args.get(ARGSTR_TASKS_PER_JOB) is not None and not args.get(ARGSTR_SCHEDULER):
        parser.error("{} option requires {} option".format(ARGSTR_TASKS_PER_JOB, ARGSTR_SCHEDULER))
    if not os.path.isfile(jobscript):
        parser.error("{} is not a valid file path".format(ARGSTR_JOBSCRIPT))
    if not os.path.isdir(scratchdir):
        os.makedirs(scratchdir)


    # Gather list of masks to apply to a (range of) source raster suffix(es).
    if os.path.isfile(src) and not src.endswith('.txt'):
        src_raster = src
        for src_suffix in SRC_SUFFIXES:
            if src_raster.endswith(src_suffix):
                break
        src_bitmask = src_raster.replace(src_suffix, BITMASK_SUFFIX)
        if src_bitmask == src_raster:
            parser.error("Path of *_{} component for `{}` raster file could not be determined. "
                         "Is suffix of `src` path not listed in script global `SRC_SUFFIXES`?".format(
                         BITMASK_SUFFIX, ARGSTR_SRC))
        if not os.path.isfile(src_bitmask):
            parser.error("*_{}.tif component for `{}` raster file does not exist: {}".format(
                         BITMASK_SUFFIX, ARGSTR_SRC, src_bitmask))
        src_bitmasks = [src_bitmask]
        src_suffixes = [src_suffix]
    elif os.path.isfile(src) and src.endswith('.txt'):
        bundle_file = src
        src_bitmasks = batch_handler.read_task_bundle(bundle_file)
    elif os.path.isdir(src):
        srcdir = src
        src_bitmasks = glob.glob(os.path.join(srcdir, '*_{}'.format(BITMASK_SUFFIX)))
    else:
        parser.error("`{}` must be a path to either a directory or a file".format(ARGSTR_SRC))

    masking_bitstring = get_mask_bitstring(args.get(ARGSTR_EDGE),
                                           args.get(ARGSTR_WATER),
                                           args.get(ARGSTR_CLOUD))

    masks_to_apply = []
    for maskFile in src_bitmasks:
        for rasterSuffix in src_suffixes:
            if not os.path.isfile(get_dstFile(dstdir, maskFile, rasterSuffix, masking_bitstring)):
                masks_to_apply.append(maskFile)
                break

    num_tasks = len(masks_to_apply)

    print("Number of source bitmasks found: {}".format(len(masks_to_apply)))
    print("Number of incomplete masking tasks: {}".format(num_tasks))

    if num_tasks == 0:
        sys.exit(0)

    elif scheduler is not None:

        tasks_per_job = args.get(ARGSTR_TASKS_PER_JOB)
        src_files = (masks_to_apply if tasks_per_job is None else
                     batch_handler.write_task_bundles(masks_to_apply, tasks_per_job, scratchdir,
                                                      '{}_{}'.format(JOB_ABBREV, ARGSTR_SRC)))

        args.remove_args(*argstr_batch)
        for i, src_file in enumerate(src_files):
            args.set(ARGSTR_SRC, src_file)
            job_cmd = args.get_cmd()

            cmd = batch_handler.get_jobsubmit_cmd(scheduler, jobscript, JOB_ABBREV, job_cmd)
            if dryrun:
                print(cmd)
            else:
                subprocess.call(cmd, shell=True)

    else:
        for i, maskFile in masks_to_apply:
            print("Mask ({}/{}): {}".format(i+1, num_tasks, maskFile))
            if not dryrun:
                mask_rasters(dstdir, maskFile, src_suffixes, args)


def get_mask_bitstring(edge, water, cloud):
    maskcomp_state_bit = [
        (MASKCOMP_EDGE_BIT, edge),
        (MASKCOMP_WATER_BIT, water),
        (MASKCOMP_CLOUD_BIT, cloud)
    ]
    maskcomp_bits = [t[0] for t in maskcomp_state_bit]
    bit_vals = np.zeros(max(maskcomp_bits)+1, dtype=int)
    for bit_index, maskcomp_state in maskcomp_state_bit:
        bit_vals[-(bit_index+1)] = maskcomp_state
    bitstring = ''.join(bit_vals.astype(np.dtype(str)))
    return bitstring


def get_dstFile(dstDir, maskFile, rasterSuffix, masking_bitstring):
    dstFname_prefix, dstFname_ext = os.path.splitext(
        os.path.basename(maskFile).replace(BITMASK_SUFFIX, rasterSuffix))
    dstFname = '{}_{}{}'.format(dstFname_prefix, masking_bitstring, dstFname_ext)
    dstFile = os.path.join(dstDir, dstFname)
    return dstFile


def mask_rasters(dstdir, maskFile, src_suffixes, args):
    pass



if __name__ == '__main__':
    main()
