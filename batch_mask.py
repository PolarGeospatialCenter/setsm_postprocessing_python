#!/usr/bin/env python2

# Version 1.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2018


from __future__ import division
import argparse
import glob
import os
import subprocess
import sys
from time import sleep

import numpy as np

from lib.filter_scene import MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT
from lib import batch_handler
import lib.raster_array_tools as rat


SCRIPT_FILE = os.path.realpath(__file__)
SCRIPT_FNAME = os.path.basename(SCRIPT_FILE)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE)

# Argument option strings
ARGSTR_SRC = 'src'
ARGSTR_DSTDIR = '--dstdir'
ARGSTR_SRC_SUFFIX = '--src-suffix'
ARGSTR_DST_NODATA = '--dst-nodata'
ARGSTR_EDGE = '--edge'
ARGSTR_WATER = '--water'
ARGSTR_CLOUD = '--cloud'
ARGSTR_SCHEDULER = '--scheduler'
ARGSTR_TASKS_PER_JOB = '--tasks-per-job'
ARGSTR_JOBSCRIPT = '--jobscript'
ARGSTR_SCRATCH = '--scratch'
ARGSTR_DRYRUN = '--dryrun'

# Argument option choices
ARGCHO_DST_NODATA_SAME = 'same'
ARGCHO_DST_NODATA_ADD = 'add'
ARGCHO_DST_NODATA_SWITCH = 'switch'
ARGCHO_DST_NODATA_CONVERT = 'convert'
ARGCHO_DST_NODATA_UNSET = 'unset'
ARGCHO_DST_NODATA = [
    ARGCHO_DST_NODATA_SAME,
    ARGCHO_DST_NODATA_ADD,
    ARGCHO_DST_NODATA_SWITCH,
    ARGCHO_DST_NODATA_CONVERT,
    ARGCHO_DST_NODATA_UNSET
]

# Batch arguments
BATCH_ARGSTR = [ARGSTR_SCHEDULER, ARGSTR_TASKS_PER_JOB, ARGSTR_JOBSCRIPT]

# Argument defaults
ARGDEF_SCRATCH = os.path.join(os.path.expanduser('~'), 'scratch', 'task_bundles')

# Batch settings
JOB_ABBREV = 'Mask'
PYTHON_EXE = 'python -u'

# Per-script globals
BITMASK_SUFFIX = 'bitmask.tif'.lstrip('_')


def parse_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=' '.join([
            "Selectively apply filter components from the SETSM DEM scene/strip",
            "*_{} component raster to mask out corresponding locations".format(BITMASK_SUFFIX),
            "in another component raster(s), then save the resulting image(s)."
        ])
    )

    parser.add_argument(
        ARGSTR_SRC,
        help=' '.join([
            "Path to source DEM directory or raster file.",
            "Accepts a task bundle text file listing paths to *_{}".format(BITMASK_SUFFIX),
            "raster files along with with {} argument indicating".format(ARGSTR_SRC_SUFFIX),
            "which components to mask."
        ])
    )
    parser.add_argument(
        ARGSTR_DSTDIR,
        help=' '.join([
            "Path to destination directory for output masked raster(s)",
            "(default is directory of `{}`).".format(ARGSTR_SRC)
        ])
    )
    parser.add_argument(
        ARGSTR_SRC_SUFFIX,
        help=' '.join([
            "Mask raster images with a file suffix(es) matching this string.",
            "An optional numeric string may be provided following the suffix string,",
            "delimited with a comma, to specify the 'masking value' to set",
            "masked pixels in the output raster.",
            "If the numeric string component is not provided, the NoData value",
            "of the source raster will be taken as the masking value -- if the",
            "source raster does not have a set NoData value, masking of that",
            "raster will be skipped."
        ])
    )
    parser.add_argument(
        ARGSTR_DST_NODATA,
        choices=ARGCHO_DST_NODATA,
        default=ARGCHO_DST_NODATA_SAME,
        help=' '.join([
            "Scheme for handling NoData pixel translations from source to output raster datasets.",
            "If '{}', do not change NoData value or alter values of existing NoData pixels.".format(ARGCHO_DST_NODATA_SAME),
            "If '{}' and source raster does not already have a NoData value,".format(ARGCHO_DST_NODATA_ADD),
            "set NoData value to masking value, else function identically to '{}'.".format(ARGCHO_DST_NODATA_SAME),
            "If '{}', set NoData value to masking value".format(ARGCHO_DST_NODATA_SWITCH),
            "but do not alter values of existing NoData pixels.",
            "If '{}', set NoData value to masking value and change the value of".format(ARGCHO_DST_NODATA_CONVERT),
            "existing NoData pixels to match the masking value.",
            "If '{}', unset NoData value.".format(ARGCHO_DST_NODATA_UNSET)
        ])
    )

    parser.add_argument(
        ARGSTR_EDGE, '-e',
        action='store_true',
        default=False,
        help=' '.join([
            "Apply edge filter. Not necessary when masking strip DEMs,",
            "as it is already applied in the mosaicking step of scenes2strips."
        ])
    )
    parser.add_argument(
        ARGSTR_WATER, '-w',
        action='store_true',
        default=False,
        help="Apply water filter."
    )
    parser.add_argument(
        ARGSTR_CLOUD, '-c',
        action='store_true',
        default=False,
        help="Apply cloud filter."
    )

    parser.add_argument(
        ARGSTR_SCHEDULER,
        choices=batch_handler.SCHED_SUPPORTED,
        help="Submit tasks to job scheduler."
    )
    parser.add_argument(
        ARGSTR_TASKS_PER_JOB,
        type=int,
        help=' '.join([
            "Number of tasks to bundle into a single job.",
            "(requires {} option)".format(ARGSTR_SCHEDULER)
        ])
    )
    parser.add_argument(
        ARGSTR_JOBSCRIPT,
        help=' '.join([
            "Script to run in job submission to scheduler.",
            "(default scripts are found in {})".format(SCRIPT_DIR)
        ])
    )
    parser.add_argument(
        ARGSTR_SCRATCH,
        default=ARGDEF_SCRATCH,
        help=' '.join([
            "Scratch directory to build task bundle text files."
            "(default={})".format(ARGDEF_SCRATCH)
        ])
    )

    parser.add_argument(
        ARGSTR_DRYRUN,
        action='store_true',
        default=False,
        help="Print actions without executing."
    )

    return parser


def main():

    # Parse and validate arguments.

    arg_parser = parse_args()
    args = batch_handler.ArgumentPasser(arg_parser, PYTHON_EXE, sys.argv[0])
    src = os.path.abspath(args.get(ARGSTR_SRC))
    dstdir = args.get(ARGSTR_DSTDIR)
    src_suffixToptmaskval = (
        [[ss.strip() for ss in s.strip().lstrip('_').split(',')]
                     for s  in args.get(ARGSTR_SRC_SUFFIX).split('|')]
            if args.get(ARGSTR_SRC_SUFFIX) is not None else None)
    scheduler = args.get(ARGSTR_SCHEDULER)
    jobscript = args.get(ARGSTR_JOBSCRIPT)
    scratchdir = os.path.abspath(args.get(ARGSTR_SCRATCH))
    dryrun = args.get(ARGSTR_DRYRUN)

    if not (os.path.isdir(src) or os.path.isfile(src)):
        arg_parser.error("`{}` must be a path to either a directory or a file, "
                         "but was '{}'".format(ARGSTR_SRC, src))

    if dstdir is None:
        dstdir = src if os.path.isdir(src) else os.path.dirname(src)
        print("{} automatically set to '{}'".format(ARGSTR_DSTDIR, dstdir))
    elif not os.path.isdir(dstdir):
        os.makedirs(dstdir)
    dstdir = os.path.abspath(dstdir)

    if args.get(ARGSTR_TASKS_PER_JOB) is not None and not args.get(ARGSTR_SCHEDULER):
        arg_parser.error("{} option requires {} option".format(ARGSTR_TASKS_PER_JOB, ARGSTR_SCHEDULER))

    if scheduler is not None:
        if jobscript is None:
            jobscript = os.path.join(SCRIPT_DIR, 'qsub_mask_{}.sh'.format(scheduler))
        jobscript = os.path.abspath(jobscript)
        if not os.path.isfile(jobscript):
            arg_parser.error("{} must be a valid file path, but was '{}'".format(ARGSTR_JOBSCRIPT, jobscript))

    if not os.path.isdir(scratchdir):
        os.makedirs(scratchdir)

    # Further parse source suffix/maskval argument.
    suffix_maskval_dict = None
    if src_suffixToptmaskval is not None:
        suffix_maskval_dict = {}
        for suffixToptmaskval in src_suffixToptmaskval:
            suffix = suffixToptmaskval[0]
            maskval = suffixToptmaskval[1] if len(suffixToptmaskval) == 2 else None
            if maskval is not None:
                try:
                    maskval_num = float(maskval)
                    maskval = maskval_num
                except ValueError:
                    arg_parser.error("{} argument masking value '{}' "
                                     "is invalid".format(ARGSTR_SRC_SUFFIX, maskval))
            suffix_maskval_dict[suffix] = maskval


    # Gather list of masks to apply to a (range of) source raster suffix(es).
    src_bitmasks = None

    if os.path.isfile(src) and src.endswith(BITMASK_SUFFIX):
        src_bitmask = src
        src_bitmasks = [src_bitmask]
        if suffix_maskval_dict is None:
            suffix_maskval_dict = {}
            src_prefix = src.replace(BITMASK_SUFFIX, '')
            for suffix in [f.replace(src_prefix, '') for f in glob.glob(src_prefix+'*.tif')]:
                if suffix != BITMASK_SUFFIX:
                    suffix_maskval_dict[suffix] = None

    elif os.path.isfile(src) and not src.endswith('.txt'):
        src_raster = src
        src_suffix = None
        if suffix_maskval_dict is not None:
            for suffix in suffix_maskval_dict.keys():
                if src_raster.endswith(suffix):
                    src_suffix = suffix
                    break
        if src_suffix is None:
            beg, end = 0, len(src_raster)
            end = float('inf')
            while end != -1:
                end = src_raster.rfind('_', beg, end)
                if os.path.isfile('{}_{}'.format(src_raster[beg:end], BITMASK_SUFFIX)):
                    src_suffix = src_raster[end:].lstrip('_')
                    break
        if src_suffix is None:
            arg_parser.error("Path of *_{} component for `{}` raster file "
                             "could not be determined".format(BITMASK_SUFFIX, ARGSTR_SRC))
        if suffix_maskval_dict is None:
            suffix_maskval_dict = {src_suffix: None}

        src_bitmask = src_raster.replace(src_suffix, BITMASK_SUFFIX)
        if not os.path.isfile(src_bitmask):
            arg_parser.error("*_{} mask component for `{}` raster file does not exist: {}".format(
                             BITMASK_SUFFIX, ARGSTR_SRC, src_bitmask))
        src_bitmasks = [src_bitmask]

    elif os.path.isfile(src) and src.endswith('.txt'):
        bundle_file = src
        if suffix_maskval_dict is None:
            arg_parser.error("{} option must be provided when `{}` is a task bundle text file".format(
                             ARGSTR_SRC_SUFFIX, ARGSTR_SRC))
        src_bitmasks = batch_handler.read_task_bundle(bundle_file)

    elif os.path.isdir(src):
        srcdir = src
        if suffix_maskval_dict is None:
            arg_parser.error("{} option must be provided when `{}` is a directory".format(
                             ARGSTR_SRC_SUFFIX, ARGSTR_SRC))
        src_bitmasks = glob.glob(os.path.join(srcdir, '*_{}'.format(BITMASK_SUFFIX)))
        src_bitmasks.sort()

    if src_bitmasks is None:
        arg_parser.error("`{}` must be a path to either a directory or a file, "
                         "but was '{}'".format(ARGSTR_SRC, src))

    print("-----")
    print("[Raster Suffix, Masking Value]")
    for suffix, maskval in suffix_maskval_dict.items():
        print("{}, {}".format(suffix, maskval if maskval is not None else 'source NoDataVal'))
    print("-----")

    # Build processing list by only adding bitmasks for which
    # an output masked raster image(s) with the specified mask settings
    # does not already exist in the destination directory.
    src_suffixes = suffix_maskval_dict.keys()
    masking_bitstring = get_mask_bitstring(*args.get(ARGSTR_EDGE, ARGSTR_WATER, ARGSTR_CLOUD))
    masks_to_apply = []
    for maskFile in src_bitmasks:
        for rasterSuffix in src_suffixes:
            src_rasterFile = maskFile.replace(BITMASK_SUFFIX, rasterSuffix)
            dst_rasterFile = get_dstFile(dstdir, maskFile, rasterSuffix, masking_bitstring)
            if os.path.isfile(src_rasterFile) and not os.path.isfile(dst_rasterFile):
                masks_to_apply.append(maskFile)
                break

    num_tasks = len(masks_to_apply)

    print("Number of source bitmasks found: {}".format(len(src_bitmasks)))
    print("Number of incomplete masking tasks: {}".format(num_tasks))

    if num_tasks == 0:
        sys.exit(0)

    print("-----")
    wait_seconds = 5
    print("Sleeping {} seconds before job submission".format(wait_seconds))
    sleep(wait_seconds)
    print("-----")

    if scheduler is not None:

        tasks_per_job = args.get(ARGSTR_TASKS_PER_JOB)
        src_files = (masks_to_apply if tasks_per_job is None else
                     batch_handler.write_task_bundles(masks_to_apply, tasks_per_job, scratchdir,
                                                      '{}_{}'.format(JOB_ABBREV, ARGSTR_SRC)))

        jobnum_fmt = '{:0>'+str(len(str(len(src_files))))+'}'

        args.remove_args(*BATCH_ARGSTR)
        for i, src_file in enumerate(src_files):
            args.set(ARGSTR_SRC, src_file)

            job_cmd = args.get_cmd()
            job_name = JOB_ABBREV+jobnum_fmt.format(i+1)

            cmd = batch_handler.get_jobsubmit_cmd(scheduler, jobscript, job_name, job_cmd)
            if dryrun:
                print(cmd)
            else:
                subprocess.call(cmd, shell=True)

    else:
        for i, maskFile in enumerate(masks_to_apply):
            print("Mask ({}/{}): {}".format(i+1, num_tasks, maskFile))
            if not dryrun:
                mask_rasters(dstdir, maskFile, suffix_maskval_dict, args)


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


def mask_rasters(dstDir, maskFile, suffix_maskval_dict, args):

    masking_bitstring = get_mask_bitstring(*args.get(ARGSTR_EDGE, ARGSTR_WATER, ARGSTR_CLOUD))
    nodata_opt = args.get(ARGSTR_DST_NODATA)

    if int(masking_bitstring) > 0:
        # Read in mask raster, then unset bits that will not be used to mask.
        mask_select, mask_x, mask_y = rat.extractRasterData(maskFile, 'z', 'x', 'y')
        mask_ones = np.ones_like(mask_select)
        if not args.get(ARGSTR_EDGE):
            np.bitwise_and(mask_select, ~np.left_shift(mask_ones, MASKCOMP_EDGE_BIT), out=mask_select)
        if not args.get(ARGSTR_WATER):
            np.bitwise_and(mask_select, ~np.left_shift(mask_ones, MASKCOMP_WATER_BIT), out=mask_select)
        if not args.get(ARGSTR_CLOUD):
            np.bitwise_and(mask_select, ~np.left_shift(mask_ones, MASKCOMP_CLOUD_BIT), out=mask_select)
        del mask_ones

        # Convert remaining component bits to a binary boolean mask.
        mask_select = mask_select.astype(np.bool)
    else:
        mask_select = None

    # Make list for downsampled mask array "pyramids"
    # that may be built on-the-fly.
    mask_pyramids = [mask_select]

    # Apply mask to source raster images and save results.
    for src_suffix, maskval in suffix_maskval_dict.items():

        src_rasterFile = maskFile.replace(BITMASK_SUFFIX, src_suffix)
        dst_rasterFile = get_dstFile(dstDir, maskFile, src_suffix, masking_bitstring)

        if not os.path.isfile(src_rasterFile):
            print("Source raster does not exist: {}".format(src_rasterFile))
            continue
        if os.path.isfile(dst_rasterFile):
            print("Output raster already exists: {}".format(dst_rasterFile))
            continue

        print("Masking source raster ({}) to output raster ({})".format(src_rasterFile, dst_rasterFile))

        # Read in source raster.
        dst_array, dst_x, dst_y, src_nodataval = rat.extractRasterData(
            src_rasterFile, 'z', 'x', 'y', 'nodata_val')

        # Set masking value to source NoDataVal if necessary.
        if maskval is None:
            if src_nodataval is None:
                print("Source raster does not have a set NoData value, "
                      "so masking value cannot be automatically determined; skipping")
                continue
            else:
                maskval = src_nodataval

        if mask_select is not None:
            # Apply mask.
            if mask_select.shape != dst_array.shape:
                # See if the required mask pyramid level has already been built.
                mask_select = None
                for arr in mask_pyramids:
                    if arr.shape == dst_array.shape:
                        mask_select = arr
                        break
                if mask_select is None:
                    # Build the required mask pyramid level and add to pyramids.
                    mask_select = rat.interp2_gdal(
                        mask_x, mask_y, mask_select, dst_x, dst_y,
                        'nearest', extrapolate=True).astype(np.bool)
                    mask_pyramids.append(mask_select)
            dst_array[mask_select] = maskval

        # Handle nodata options.
        if nodata_opt == ARGCHO_DST_NODATA_SAME:
            dst_nodataval = src_nodataval
        elif nodata_opt == ARGCHO_DST_NODATA_ADD:
            dst_nodataval = maskval if src_nodataval is None else src_nodataval
        elif nodata_opt == ARGCHO_DST_NODATA_SWITCH:
            dst_nodataval = maskval
        elif nodata_opt == ARGCHO_DST_NODATA_CONVERT:
            if src_nodataval is not None:
                dst_nodata = ((dst_array == src_nodataval)
                              if not np.isnan(src_nodataval) else np.isnan(dst_array))
                dst_array[dst_nodata] = maskval
            dst_nodataval = maskval
        elif nodata_opt == ARGCHO_DST_NODATA_UNSET:
            dst_nodataval = None

        # Save output masked raster.
        rat.saveArrayAsTiff(dst_array, dst_rasterFile,
                            nodata_val=dst_nodataval,
                            like_raster=src_rasterFile)



if __name__ == '__main__':
    main()
