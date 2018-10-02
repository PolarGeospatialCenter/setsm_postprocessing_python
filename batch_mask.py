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

BITMASK_SUFFIX = 'bitmask.tif'
SRC_SUFFIX_NODATA = {
    'dem_smooth.tif': -9999,
    'dem_smooth_10m.tif': -9999,
    'dem_smooth_browse.tif': 0,
    'dem.tif': -9999,
    'dem_10m.tif': -9999,
    'dem_browse.tif': 0,
    'matchtag.tif': 0,
    'ortho.tif': 0,
    'ortho_browse.tif': 0,
    'dem_coverage.tif': 0
}

ARGSTR_SRC = 'src'
ARGSTR_DSTDIR = '--dstdir'
ARGSTR_SRCDIR_SUFFIX = '--srcdir-suffix'
ARGSTR_SUFFIX_NODATA = '--src-nodata'
ARGSTR_EDGE = '--edge'
ARGSTR_WATER = '--water'
ARGSTR_CLOUD = '--cloud'
ARGSTR_SCHEDULER = '--scheduler'
ARGSTR_TASKS_PER_JOB = '--tasks-per-job'
ARGSTR_JOBSCRIPT = '--jobscript'
ARGSTR_SCRATCH = '--scratch'
ARGSTR_DRYRUN = '--dryrun'

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

    parser.add_argument(ARGSTR_SRCDIR_SUFFIX,
        help="If `{}` is a directory, mask all raster images with a ".format(ARGSTR_SRC)
            +"file suffix(es) matching this string. An optional numeric string "
             "may be provided following the suffix string, delimited with a comma, "
             "to specify the value to set masked pixels in the output raster. "
             "If the numeric string component is not provided, an attempt will be "
             "made to infer its value from a lookup of the suffix string in a "
             "suffix-to-nodata-value dictionary. "
             "Syntax is like 'dem_smooth.tif,nan|ortho.tif' to mask the LSF DEM "
             "component with NaN and to mask the ortho component with the "
             "default nodata value.")
    parser.add_argument(ARGSTR_SUFFIX_NODATA,
        help="Set masked pixels in the output raster(s) to this value(s). "
             "If {} option is provided, the quantities of elements provided "
             "to this option must be either zero (an attempt will be made to "
             "infer nodata values for each provided suffix) or equal to the "
             "quantity of elements provided to that option so that the nodataand this one must be equal. Syntax is like "
             "'nan|0'")

    parser.add_argument('-e', ARGSTR_EDGE, action='store_true', default=False,
        help="Apply edge filter. Not necessary when masking strip DEMs, "
             "as it is already applied in the mosaicking step of scenes2strips.")
    parser.add_argument('-w', ARGSTR_WATER, action='store_true', default=False,
        help="Apply water filter.")
    parser.add_argument('-c', ARGSTR_CLOUD, action='store_true', default=False,
        help="Apply cloud filter.")

    parser.add_argument(ARGSTR_SCHEDULER, choices=batch_handler.SCHED_SUPPORTED,
        help="Submit tasks to job scheduler.")
    parser.add_argument(ARGSTR_TASKS_PER_JOB, type=int,
        help="Number of tasks to bundle into a single job. (requires {} option)".format(ARGSTR_SCHEDULER))
    parser.add_argument(ARGSTR_JOBSCRIPT, default=ARGDEF_JOBSCRIPT,
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
    src_suffixes = (   [s.strip().lstrip('_') for s in args.get(ARGSTR_SRCDIR_SUFFIX).split('|')]
                    if args.get(ARGSTR_SRCDIR_SUFFIX) is not None else None)
    scheduler = args.get(ARGSTR_SCHEDULER)
    jobscript = args.get(ARGSTR_JOBSCRIPT)
    scratchdir = args.get(ARGSTR_SCRATCH)
    dryrun = args.get(ARGSTR_DRYRUN)

    if not (os.path.isdir(src) or os.path.isfile(src)):
        parser.error("`{}` must be a path to either a directory or a file, "
                     "but was '{}'".format(ARGSTR_SRC, src))
    if dstdir is None:
        dstdir = src if os.path.isdir(src) else os.path.dirname(src)
        print("{} automatically set to '{}'".format(ARGSTR_DSTDIR, dstdir))
    elif not os.path.isdir(dstdir):
        os.makedirs(dstdir)
    if args.get(ARGSTR_TASKS_PER_JOB) is not None and not args.get(ARGSTR_SCHEDULER):
        parser.error("{} option requires {} option".format(ARGSTR_TASKS_PER_JOB, ARGSTR_SCHEDULER))
    if not os.path.isfile(jobscript):
        parser.error("{} must be a valid file path, but was '{}'".format(ARGSTR_JOBSCRIPT, jobscript))
    if not os.path.isdir(scratchdir):
        os.makedirs(scratchdir)

    # Gather list of masks to apply to a (range of) source raster suffix(es).
    if os.path.isfile(src) and not src.endswith('.txt'):
        src_raster = src
        src_suffix = None
        if src_suffixes is not None:
            for s in src_suffixes:
                if src_raster.endswith(s):
                    src_suffix = s
                    break
        if src_suffix is None:
            for s in SRC_SUFFIX_NODATA:
                if src_raster.endswith(s):
                    src_suffix = s
                    break
        if src_suffix is None:
            parser.error("Path of *_{} component for `{}` raster file could not be determined. ".format(BITMASK_SUFFIX, ARGSTR_SRC)
                        +"Either provide suffix string for this raster file as {} argument ".format(ARGSTR_SRCDIR_SUFFIX)
                        +"or add suffix to script global SRC_SUFFIX_NODATA dictionary.")
        src_bitmask = src_raster.replace(src_suffix, BITMASK_SUFFIX)
        if not os.path.isfile(src_bitmask):
            parser.error("*_{} component for `{}` raster file does not exist: {}".format(
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
        parser.error("`{}` must be a path to either a directory or a file, "
                     "but was '{}'".format(ARGSTR_SRC, src))

    # Determine masking value(s) for source raster suffix(es).
    src_suffix_maskval = [s.split(',') for s in src_suffixes]
    for i in range(len(src_suffix_maskval)):
        suffix = src_suffix_maskval[i][0]
        if len(src_suffix_maskval[i]) == 1:
            try:
                default_nodata = SRC_SUFFIX_NODATA[suffix]
            except KeyError:
                parser.error("Cannot determine default nodata masking value for raster suffix '{}'".format(suffix))
            src_suffix_maskval[i].append(default_nodata)
        maskval = src_suffix_maskval[i][1]
        try:
            maskval_num = float(maskval)
        except ValueError:
            parser.error("Source raster suffix '{}' masking value '{}' cannot be converted to a "
                         "number".format(suffix, maskval))
        src_suffix_maskval[i][1] = maskval_num

    # Build processing list by only adding bitmasks for which
    # an output masked raster image(s) with the specified mask settings
    # does not already exist in the destination directory.
    masking_bitstring = get_mask_bitstring(*args.get(ARGSTR_EDGE, ARGSTR_WATER, ARGSTR_CLOUD))
    masks_to_apply = []
    for maskFile in src_bitmasks:
        for rasterSuffix in src_suffixes:
            if not os.path.isfile(get_dstFile(dstdir, maskFile, rasterSuffix, masking_bitstring)):
                masks_to_apply.append(maskFile)
                break

    num_tasks = len(masks_to_apply)

    print("Number of source bitmasks found: {}".format(len(src_bitmasks)))
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
        for i, maskFile in enumerate(masks_to_apply):
            print("Mask ({}/{}): {}".format(i+1, num_tasks, maskFile))
            if not dryrun:
                mask_rasters(dstdir, maskFile, src_suffix_maskval, args)


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
    dstFname = '{}_mask{}{}'.format(dstFname_prefix, masking_bitstring, dstFname_ext)
    dstFile = os.path.join(dstDir, dstFname)
    return dstFile


def mask_rasters(dstDir, maskFile, src_suffix_maskval, args):

    masking_bitstring = get_mask_bitstring(*args.get(ARGSTR_EDGE, ARGSTR_WATER, ARGSTR_CLOUD))

    # Read in mask raster, then unset bits that will not be used to mask.
    mask_select = rat.extractRasterData(maskFile, 'array')
    mask_ones = np.ones_like(mask_select)
    if not args.get(ARGSTR_EDGE):
        np.bitwise_and(mask_select, ~np.left_shift(mask_ones, MASKCOMP_EDGE_BIT), out=mask_select)
    if not args.get(ARGSTR_WATER):
        np.bitwise_and(mask_select, ~np.left_shift(mask_ones, MASKCOMP_WATER_BIT), out=mask_select)
    if not args.get(ARGSTR_CLOUD):
        np.bitwise_and(mask_select, ~np.left_shift(mask_ones, MASKCOMP_CLOUD_BIT), out=mask_select)
    del mask_ones

    # Convert remaining component bits to a binary boolean mask.
    mask_array = mask_select.astype(np.bool)
    del mask_select

    # Apply mask to source raster images and save results.
    for src_suffix, maskval in src_suffix_maskval:

        src_rasterFile = maskFile.replace(BITMASK_SUFFIX, src_suffix)
        dst_rasterFile = get_dstFile(dstDir, maskFile, src_suffix, masking_bitstring)
        if os.path.isfile(dst_rasterFile):
            continue
        print("Masking source raster ({}) to output raster ({})".format(src_rasterFile, dst_rasterFile))

        # Read in source raster and apply mask.
        dst_array = rat.extractRasterData(src_rasterFile, 'array')
        dst_array[mask_array] = maskval
        del mask_array

        # Save output masked raster.
        rat.saveArrayAsTiff(dst_array, dst_rasterFile, like_raster=src_rasterFile)



if __name__ == '__main__':
    main()
