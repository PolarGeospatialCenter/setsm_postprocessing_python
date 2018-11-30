
# Erik Husby; Polar Geospatial Center, University of Minnesota; 2018


from __future__ import division
import argparse
import copy
import functools
import glob
import os
import subprocess
import sys
from time import sleep

import numpy as np

from lib.filter_scene import MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT
from lib import batch_handler
import lib.raster_array_tools as rat


##############################

## Core globals

SCRIPT_VERSION_NUM = 1.0

# Paths
SCRIPT_FILE = os.path.realpath(__file__)
SCRIPT_FNAME = os.path.basename(SCRIPT_FILE)
SCRIPT_NAME, SCRIPT_EXT = os.path.splitext(SCRIPT_FNAME)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE)

##############################

## Argument globals

# Argument strings
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
ARGSTR_LOGDIR = '--logdir'
ARGSTR_DRYRUN = '--dryrun'

# Argument choices
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

# Argument defaults
ARGDEF_SCRATCH = os.path.join(os.path.expanduser('~'), 'scratch', 'task_bundles')

# Argument groups
ARGGRP_OUTDIR = [ARGSTR_DSTDIR, ARGSTR_LOGDIR, ARGSTR_SCRATCH]
ARGGRP_BATCH = [ARGSTR_SCHEDULER, ARGSTR_JOBSCRIPT, ARGSTR_TASKS_PER_JOB]

##############################

## Batch settings

JOBSCRIPT_DIR = os.path.join(SCRIPT_DIR, 'jobscripts')
PYTHON_EXE = 'python -u'
JOB_ABBREV = 'Mask'

##############################

## Custom globals

BITMASK_SUFFIX = 'bitmask.tif'.lstrip('_')

##############################


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


class RawTextArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter): pass

def argtype_path_handler(path, argstr,
                         abspath_fn=os.path.abspath,
                         existcheck_fn=None, existcheck_reqval=None):
    if existcheck_fn is not None and existcheck_fn(path) != existcheck_reqval:
        if existcheck_fn is os.path.isfile:
            existtype_str = 'file'
        elif existcheck_fn is os.path.isdir:
            existtype_str = 'directory'
        elif existcheck_fn is os.path.exists:
            existtype_str = 'file/directory'
        existresult_str = 'does not exist' if existcheck_reqval is True else 'already exists'
        raise InvalidArgumentError("argument {}: {} {}".format(argstr, existtype_str, existresult_str))
    return abspath_fn(path) if abspath_fn is not None else path

ARGTYPE_PATH = functools.partial(functools.partial, argtype_path_handler)

def argparser_init():

    parser = argparse.ArgumentParser(
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        description=' '.join([
            "Selectively apply filter components from the SETSM DEM scene/strip",
            "*_{} component raster to mask out corresponding locations".format(BITMASK_SUFFIX),
            "in another component raster(s), then save the resulting image(s)."
        ])
    )

    # Positional arguments

    parser.add_argument(
        ARGSTR_SRC,
        type=ARGTYPE_PATH(
            argstr=ARGSTR_SRC,
            existcheck_fn=os.path.exists,
            existcheck_reqval=True),
        help=' '.join([
            "Path to source DEM directory or raster file.",
            "Accepts a task bundle text file listing paths to *_{}".format(BITMASK_SUFFIX),
            "raster files along with with {} argument indicating".format(ARGSTR_SRC_SUFFIX),
            "which components to mask."
        ])
    )

    # Optional arguments

    parser.add_argument(
        ARGSTR_DSTDIR,
        type=ARGTYPE_PATH(
            argstr=ARGSTR_DSTDIR,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=False),
        help=' '.join([
            "Path to destination directory for output masked raster(s)",
            "(default is directory of `{}`).".format(ARGSTR_SRC)
        ])
    )
    parser.add_argument(
        ARGSTR_SRC_SUFFIX,
        type=str,
        default=None,
        help=' '.join([
            "Mask raster images with a file suffix(es) matching this string.",
            "An optional numeric string may be provided following the suffix string,",
            "delimited with a comma, to specify the 'masking value' to set",
            "masked pixels in the output raster.",
            "\nIf the numeric string component is not provided, the NoData value",
            "of the source raster will be taken as the masking value.",
            "\nIf the source raster does not have a set NoData value, masking of that",
            "raster will be skipped.",
            "\n"
        ])
    )
    parser.add_argument(
        ARGSTR_DST_NODATA,
        type=str,
        choices=ARGCHO_DST_NODATA,
        default=ARGCHO_DST_NODATA_SAME,
        help=' '.join([
            "Scheme for handling NoData pixel translations from source to output raster datasets.",
            "\nIf '{}', do not change NoData value or alter values of existing NoData pixels.".format(ARGCHO_DST_NODATA_SAME),
            "\nIf '{}' and source raster does not already have a NoData value,".format(ARGCHO_DST_NODATA_ADD),
            "set NoData value to masking value, else function identically to '{}'.".format(ARGCHO_DST_NODATA_SAME),
            "\nIf '{}', set NoData value to masking value".format(ARGCHO_DST_NODATA_SWITCH),
            "but do not alter values of existing NoData pixels.",
            "\nIf '{}', set NoData value to masking value and change the value of".format(ARGCHO_DST_NODATA_CONVERT),
            "existing NoData pixels to match the masking value.",
            "\nIf '{}', unset NoData value.".format(ARGCHO_DST_NODATA_UNSET),
            "\n"
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
        type=str,
        choices=batch_handler.SCHED_SUPPORTED,
        default=None,
        help="Submit tasks to job scheduler."
    )
    parser.add_argument(
        ARGSTR_TASKS_PER_JOB,
        type=int,
        choices=None,
        default=None,
        help=' '.join([
            "Number of tasks to bundle into a single job.",
            "(requires {} option)".format(ARGSTR_SCHEDULER)
        ])
    )
    parser.add_argument(
        ARGSTR_JOBSCRIPT,
        type=ARGTYPE_PATH(
            argstr=ARGSTR_JOBSCRIPT,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=True),
        default=None,
        help=' '.join([
            "Script to run in job submission to scheduler.",
            "(default scripts are found in {})".format(JOBSCRIPT_DIR)
        ])
    )
    parser.add_argument(
        ARGSTR_SCRATCH,
        type=ARGTYPE_PATH(
            argstr=ARGSTR_SCRATCH,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=False),
        default=ARGDEF_SCRATCH,
        help="Scratch directory to build task bundle text files."
    )
    parser.add_argument(
        ARGSTR_LOGDIR,
        type=ARGTYPE_PATH(
            argstr=ARGSTR_LOGDIR,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=False),
        default=None,
        help=' '.join([
            "Directory to which standard output/error log files will be written for batch job runs.",
            "\nIf not provided, default scheduler (or jobscript #CONDOPT_) options will be used.",
            "\n**Note that due to implementation difficulties, this directory will also become the",
            "working directory for the job process. Since relative path inputs are always changed",
            "to absolute paths in this script, this should not be an issue."
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

    # Invoke argparse argument parsing.
    arg_parser = argparser_init()
    try:
        args = batch_handler.ArgumentPasser(arg_parser, PYTHON_EXE, SCRIPT_FILE)
    except InvalidArgumentError as e:
        arg_parser.error(e)


    ## Further parse/adjust argument values.

    src = args.get(ARGSTR_SRC)

    if args.get(ARGSTR_SRC_SUFFIX) is not None:
        src_suffixToptmaskval = [[ss.strip() for ss in s.strip().lstrip('_').split(',')]
                                             for s  in args.get(ARGSTR_SRC_SUFFIX).split('|')]
        suffix_maskval_dict = {}
        for suffixToptmaskval in src_suffixToptmaskval:
            suffix = suffixToptmaskval[0]
            maskval = suffixToptmaskval[1] if len(suffixToptmaskval) == 2 else None
            if maskval is not None:
                try:
                    maskval_num = float(maskval)
                    maskval = maskval_num
                except ValueError:
                    arg_parser.error("argument {} masking value '{}' is invalid".format(ARGSTR_SRC_SUFFIX, maskval))
            suffix_maskval_dict[suffix] = maskval
    else:
        suffix_maskval_dict = None

    if args.get(ARGSTR_DSTDIR) is None:
        args.set(ARGSTR_DSTDIR, src if os.path.isdir(src) else os.path.dirname(src))
        print("argument {} set automatically to: {}".format(ARGSTR_DSTDIR, args.get(ARGSTR_DSTDIR)))

    if args.get(ARGSTR_SCHEDULER) is not None:
        if args.get(ARGSTR_JOBSCRIPT) is None:
            jobscript_default = os.path.join(JOBSCRIPT_DIR,
                                             '{}_{}.sh'.format(SCRIPT_NAME, args.get(ARGSTR_SCHEDULER)))
            if not os.path.isfile(jobscript_default):
                arg_parser.error(
                    "Default jobscript ({}) does not exist, ".format(jobscript_default)
                    + "please specify one with {} argument".format(ARGSTR_JOBSCRIPT))
            else:
                args.set(ARGSTR_JOBSCRIPT, jobscript_default)


    ## Validate argument values.

    if args.get(ARGSTR_TASKS_PER_JOB) is not None and not args.get(ARGSTR_SCHEDULER):
        arg_parser.error("{} option requires {} option".format(ARGSTR_TASKS_PER_JOB, ARGSTR_SCHEDULER))


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
            src_raster_dir = os.path.dirname(src_raster)
            src_raster_fname = os.path.basename(src_raster)
            beg, end = 0, len(src_raster_fname)
            end = float('inf')
            while end != -1:
                end = src_raster_fname.rfind('_', beg, end)
                if os.path.isfile(os.path.join(src_raster_dir, '{}_{}'.format(src_raster_fname[beg:end], BITMASK_SUFFIX))):
                    src_suffix = src_raster_fname[end:].lstrip('_')
                    break
        if src_suffix is None:
            arg_parser.error("Path of {} component for argument {} raster file "
                             "could not be determined".format(BITMASK_SUFFIX, ARGSTR_SRC))
        if suffix_maskval_dict is None:
            suffix_maskval_dict = {src_suffix: None}

        src_bitmask = src_raster.replace(src_suffix, BITMASK_SUFFIX)
        if not os.path.isfile(src_bitmask):
            arg_parser.error("{} mask component for argument {} raster file does not exist: {}".format(
                             BITMASK_SUFFIX, ARGSTR_SRC, src_bitmask))
        src_bitmasks = [src_bitmask]

    elif os.path.isfile(src) and src.endswith('.txt'):
        bundle_file = src
        if suffix_maskval_dict is None:
            arg_parser.error("{} option must be provided when argument {} is a task bundle text file".format(
                             ARGSTR_SRC_SUFFIX, ARGSTR_SRC))
        src_bitmasks = batch_handler.read_task_bundle(bundle_file)

    elif os.path.isdir(src):
        srcdir = src
        if suffix_maskval_dict is None:
            arg_parser.error("{} option must be provided when argument {} is a directory".format(
                             ARGSTR_SRC_SUFFIX, ARGSTR_SRC))
        src_bitmasks = glob.glob(os.path.join(srcdir, '*_{}'.format(BITMASK_SUFFIX)))
        src_bitmasks.sort()

    if src_bitmasks is None:
        arg_parser.error("argument {} must be a path to either a directory or a file, "
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
            dst_rasterFile = get_dstFile(args.get(ARGSTR_DSTDIR), maskFile, rasterSuffix, masking_bitstring)
            if os.path.isfile(src_rasterFile) and not os.path.isfile(dst_rasterFile):
                masks_to_apply.append(maskFile)
                break

    num_tasks = len(masks_to_apply)

    print("Number of source bitmasks found: {}".format(len(src_bitmasks)))
    print("Number of incomplete masking tasks: {}".format(num_tasks))

    if num_tasks == 0:
        sys.exit(0)

    # Create output directories if they don't already exist.
    if not args.get(ARGSTR_DRYRUN):
        for dir_argstr, dir_path in list(zip(ARGGRP_OUTDIR, list(args.get(*ARGGRP_OUTDIR)))):
            if dir_path is not None and not os.path.isdir(dir_path):
                print("Creating argument {} directory: {}".format(dir_argstr, dir_path))
                os.makedirs(dir_path)

    # Pause for user review.
    print("-----")
    wait_seconds = 5
    print("Sleeping {} seconds before task submission".format(wait_seconds))
    sleep(wait_seconds)
    print("-----")

    ## Process masks.

    if args.get(ARGSTR_SCHEDULER) is not None:
        # Process masks in batch.

        tasks_per_job = args.get(ARGSTR_TASKS_PER_JOB)
        src_files = (masks_to_apply if tasks_per_job is None else
                     batch_handler.write_task_bundles(masks_to_apply, tasks_per_job,
                                                      args.get(ARGSTR_SCRATCH),
                                                      '{}_{}'.format(JOB_ABBREV, ARGSTR_SRC)))

        jobnum_fmt = batch_handler.get_jobnum_fmtstr(src_files)

        args_batch = args
        args_single = copy.deepcopy(args)
        args_single.remove_args(*ARGGRP_BATCH)

        for i, src_file in enumerate(src_files):
            args_single.set(ARGSTR_SRC, src_file)

            job_name = JOB_ABBREV+jobnum_fmt.format(i+1)
            cmd_single = args_single.get_cmd()

            cmd = args_single.get_jobsubmit_cmd(args_batch.get(ARGSTR_SCHEDULER),
                                                args_batch.get(ARGSTR_JOBSCRIPT),
                                                job_name, cmd_single)
            if args_batch.get(ARGSTR_DRYRUN):
                print(cmd)
            else:
                subprocess.call(cmd, shell=True, cwd=args_batch.get(ARGSTR_LOGDIR))

    else:
        # Process masks in serial.

        for i, maskFile in enumerate(masks_to_apply):
            print("Mask ({}/{}): {}".format(i+1, num_tasks, maskFile))
            if not args.get(ARGSTR_DRYRUN):
                mask_rasters(args.get(ARGSTR_DSTDIR), maskFile, suffix_maskval_dict, args)


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
