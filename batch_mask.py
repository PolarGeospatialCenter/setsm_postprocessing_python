
# Erik Husby; Polar Geospatial Center, University of Minnesota; 2019


from __future__ import division
from lib import script_utils

PYTHON_VERSION_ACCEPTED_MIN = "2.7"  # supports multiple dot notation
if script_utils.PYTHON_VERSION < script_utils.VersionString(PYTHON_VERSION_ACCEPTED_MIN):
    raise script_utils.VersionError("Python version ({}) is below accepted minimum ({})".format(
        script_utils.PYTHON_VERSION, PYTHON_VERSION_ACCEPTED_MIN))


import argparse
import copy
import glob
import os
import subprocess
import sys
import traceback
import warnings
from time import sleep

from lib import script_utils
from lib.script_utils import ScriptArgumentError
# from lib.filter_scene import MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT
MASKCOMP_EDGE_BIT = 0
MASKCOMP_WATER_BIT = 1
MASKCOMP_CLOUD_BIT = 2


##############################

## Core globals

SCRIPT_VERSION_NUM = 1.0

# Paths
SCRIPT_FILE = os.path.realpath(__file__)
SCRIPT_FNAME = os.path.basename(SCRIPT_FILE)
SCRIPT_NAME, SCRIPT_EXT = os.path.splitext(SCRIPT_FNAME)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE)
SCRIPT_RUNCMD = ' '.join(sys.argv)+'\n'
PYTHON_EXE = 'python -u'

##############################

## Custom globals

SUFFIX_BITMASK = '_bitmask.tif'

##############################

## Argument globals

# Argument strings
ARGSTR_SRC = 'src'
ARGSTR_DSTDIR = '--dstdir'
ARGSTR_MASK_SUFFIX = '--mask-suffix'
ARGSTR_MASK_VALUE = '--mask-value'
ARGSTR_SRC_SUFFIX = '--src-suffix'
ARGSTR_DST_SUFFIX = '--dst-suffix'
ARGSTR_DST_NODATA = '--dst-nodata'
ARGSTR_EDGE = '--edge'
ARGSTR_WATER = '--water'
ARGSTR_CLOUD = '--cloud'
ARGSTR_FILTER_OFF = '--filter-off'
ARGSTR_OVERWRITE = '--overwrite'
ARGSTR_SCHEDULER = '--scheduler'
ARGSTR_JOBSCRIPT = '--jobscript'
ARGSTR_TASKS_PER_JOB = '--tasks-per-job'
ARGSTR_SCRATCH = '--scratch'
ARGSTR_LOGDIR = '--logdir'
ARGSTR_EMAIL = '--email'
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
ARGDEF_SRC_SUFFIX = '_dem.tif'
ARGDEF_MASK_SUFFIX = SUFFIX_BITMASK
ARGDEF_SCRATCH = os.path.join(os.path.expanduser('~'), 'scratch', 'task_bundles')

# Argument groups
ARGGRP_OUTDIR = [ARGSTR_DSTDIR, ARGSTR_LOGDIR, ARGSTR_SCRATCH]
ARGGRP_BATCH = [ARGSTR_SCHEDULER, ARGSTR_JOBSCRIPT, ARGSTR_TASKS_PER_JOB, ARGSTR_EMAIL]
ARGGRP_FILTER_COMP = [ARGSTR_EDGE, ARGSTR_WATER, ARGSTR_CLOUD]

##############################

## Batch settings

JOBSCRIPT_DIR = os.path.join(SCRIPT_DIR, 'jobscripts')
JOBSCRIPT_INIT = os.path.join(JOBSCRIPT_DIR, 'init.sh')
JOB_ABBREV = 'Mask'
JOB_WALLTIME_HR = 30
JOB_MEMORY_GB = 20

##############################

## Custom globals

MASKED_SUFFIX_DEFAULT = 'masked'

SRC_SUFFIX_CATCH_ALL = '*.tif'

STRIP_LSF_PREFIX = '_lsf'
SUFFIX_STRIP_BITMASK = SUFFIX_BITMASK
SUFFIX_STRIP_BITMASK_LSF = STRIP_LSF_PREFIX+SUFFIX_BITMASK
SUFFIX_SCENE_BITMASK = '_dem'+SUFFIX_BITMASK
SUFFIX_SCENE_BITMASK_LSF = '_dem_smooth'+SUFFIX_BITMASK

global SUFFIX_PRIORITY_BITMASK
SUFFIX_PRIORITY_BITMASK = [
    SUFFIX_STRIP_BITMASK_LSF,
    SUFFIX_SCENE_BITMASK_LSF,
    SUFFIX_SCENE_BITMASK,
    SUFFIX_STRIP_BITMASK
]

##############################


def argparser_init():

    parser = argparse.ArgumentParser(
        formatter_class=script_utils.RawTextArgumentDefaultsHelpFormatter,
        description=' '.join([
            "Selectively apply filter components from the SETSM DEM scene/strip",
            "*{} component raster to mask out corresponding locations".format(ARGDEF_MASK_SUFFIX),
            "in another component raster(s), then save the resulting image(s)."
        ])
    )

    # Positional arguments

    parser.add_argument(
        ARGSTR_SRC,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_SRC,
            existcheck_fn=os.path.exists,
            existcheck_reqval=True),
        help=' '.join([
            "Path to source DEM directory or raster file.",
            "Accepts a task bundle text file listing paths to *{} files.".format(ARGDEF_MASK_SUFFIX)
        ])
    )

    # Optional arguments

    parser.add_argument(
        ARGSTR_DSTDIR,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_DSTDIR,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=False),
        help=' '.join([
            "Path to destination directory for output masked raster(s)",
            "(default is alongside source raster file(s))."
        ])
    )
    parser.add_argument(
        ARGSTR_MASK_SUFFIX,
        type=str,
        default=ARGDEF_MASK_SUFFIX,
        help=' '.join([
            "Filename suffix to search for applicable masking rasters.",
            "Currently only the bitmask mask version is supported."
        ])
    )
    parser.add_argument(
        ARGSTR_MASK_VALUE,
        type=str,
        default=None,
        help=' '.join([
            "The 'masking value' to set masked pixels in the output raster."
            "This value can be overridden on a per-raster-suffix basis using the {} option.".format(ARGSTR_SRC_SUFFIX),
            "\nIf not provided, the NoData value of the source raster will be taken as the masking value.",
            "If the source raster does not have a set NoData value, the raster will not be masked."
        ])
    )
    parser.add_argument(
        ARGSTR_SRC_SUFFIX,
        type=str,
        default=ARGDEF_SRC_SUFFIX,
        help=' '.join([
            "Mask raster images with a file suffix(es) matching this string.",
            "An optional numeric string may be provided following the suffix string,",
            "delimited with an equal sign (=), to specify the 'masking value' to set",
            "masked pixels in the output raster.",
            "\nIf the numeric string component is not provided, the NoData value",
            "of the source raster will be taken as the masking value. If the source raster",
            "does not have a set NoData value, masking of that raster will be skipped.",
            "\nSpecify multiple source file suffixes (with or without added masking value)",
            "by delimiting string with the pipe character (/), noting that you must then",
            "wrap the whole argument string with quotes like 'dem.tif=0/ortho.tif=0'."
            "\n"
        ])
    )
    parser.add_argument(
        ARGSTR_DST_SUFFIX,
        type=str,
        default=None,
        help=' '.join([
            "Suffix appended to filename of output masked rasters."
            "\nWorks like 'src-raster-fname.tif' -> 'src-raster-fname_[DST_SUFFIX].tif'.",
            "\nIf not provided, the default output suffix is '{}XXX', where [XXX] is the".format(MASKED_SUFFIX_DEFAULT),
            "bit-code corresponding to the filter components ([cloud, water, edge], respectively)",
            "applied in the masking for this run with the (-c, -w, -e) mask filter options.",
            "\nIf the --filter-off option is instead provided, by default all output filenames",
            "will be the same as input filenames.",
            "\nIf none of the (-c, -w, -e, --filter-off) filter options are provided, all filter",
            "components are applied and the default output suffix is simply '{}'.".format(MASKED_SUFFIX_DEFAULT),
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
        help=' '.join([
            "Selectively apply edge filter. Not really necessary when masking strip DEMs,",
            "as it is already applied in the mosaicking step of scenes2strips."
        ])
    )
    parser.add_argument(
        ARGSTR_WATER, '-w',
        action='store_true',
        help="Selectively apply water filter."
    )
    parser.add_argument(
        ARGSTR_CLOUD, '-c',
        action='store_true',
        help="Selectively apply cloud filter."
    )
    parser.add_argument(
        ARGSTR_FILTER_OFF,
        action='store_true',
        help="Turn off default (edge & water & cloud) so no filters are applied."
    )

    parser.add_argument(
        ARGSTR_OVERWRITE,
        action='store_true',
        help="Overwrite existing output rasters."
    )

    parser.add_argument(
        ARGSTR_SCHEDULER,
        type=str,
        choices=script_utils.SCHED_SUPPORTED,
        default=None,
        help="Submit tasks to job scheduler."
    )
    parser.add_argument(
        ARGSTR_JOBSCRIPT,
        type=script_utils.ARGTYPE_PATH(
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
        ARGSTR_SCRATCH,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_SCRATCH,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=False),
        default=ARGDEF_SCRATCH,
        help="Scratch directory to build task bundle text files."
    )
    parser.add_argument(
        ARGSTR_LOGDIR,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_LOGDIR,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=False),
        default=None,
        help=' '.join([
            "Directory to which standard output/error log files will be written for batch job runs.",
            "\nIf not provided, default scheduler (or jobscript #CONDOPT_) options will be used.",
            "\n**Note:** Due to implementation difficulties, this directory will also become the",
            "working directory for the job process. Since relative path inputs are always changed",
            "to absolute paths in this script, this should not be an issue."
        ])
    )
    parser.add_argument(
        ARGSTR_EMAIL,
        type=script_utils.ARGTYPE_BOOL_PLUS(
            parse_fn=str),
        nargs='?',
        help="Send email to user upon end or abort of the LAST SUBMITTED task."
    )

    parser.add_argument(
        ARGSTR_DRYRUN,
        action='store_true',
        help="Print actions without executing."
    )

    return parser


def main():

    # Invoke argparse argument parsing.
    arg_parser = argparser_init()
    try:
        args = script_utils.ArgumentPasser(PYTHON_EXE, SCRIPT_FILE, arg_parser, sys.argv)
    except ScriptArgumentError as e:
        arg_parser.error(e)


    ## Further parse/adjust argument values.

    src = args.get(ARGSTR_SRC)

    global SUFFIX_PRIORITY_BITMASK
    mask_suffix_raw = args.get(ARGSTR_MASK_SUFFIX)
    mask_suffix_fixed = '_'+mask_suffix_raw.lstrip('_')
    use_raw_suffixes = False
    if args.provided(ARGSTR_MASK_SUFFIX):
        if mask_suffix_fixed in SUFFIX_PRIORITY_BITMASK:
            SUFFIX_PRIORITY_BITMASK.remove(mask_suffix_fixed)
            SUFFIX_PRIORITY_BITMASK.insert(0, mask_suffix_fixed)
        else:
            use_raw_suffixes = True
            SUFFIX_PRIORITY_BITMASK = [args.get(ARGSTR_MASK_SUFFIX)]
    if not use_raw_suffixes and mask_suffix_fixed != mask_suffix_raw:
        args.set(ARGSTR_MASK_SUFFIX, mask_suffix_fixed)

    if args.get(ARGSTR_SRC_SUFFIX) is None or args.get(ARGSTR_SRC_SUFFIX).strip() == '':
        args.set(ARGSTR_SRC_SUFFIX, SRC_SUFFIX_CATCH_ALL)
        print("argument {} set automatically from null value to catch-all default: '{}'".format(
            ARGSTR_SRC_SUFFIX, ARGSTR_SRC_SUFFIX, args.get(ARGSTR_SRC_SUFFIX)
        ))

    if args.get(ARGSTR_SRC_SUFFIX) is None:
        suffix_maskval_dict = {SRC_SUFFIX_CATCH_ALL: None}
    else:
        src_suffixToptmaskval = [[ss.strip() for ss in s.strip().split('=')]
                                             for s  in args.get(ARGSTR_SRC_SUFFIX).split('/')]
        suffix_maskval_dict = {}
        for suffixToptmaskval in src_suffixToptmaskval:
            suffix = suffixToptmaskval[0]
            if not use_raw_suffixes and not suffix.startswith('*'):
                suffix = '_'+suffix.lstrip('_')
            maskval = suffixToptmaskval[1] if len(suffixToptmaskval) == 2 else args.get(ARGSTR_MASK_VALUE)
            if maskval is not None:
                if startswith_one_of_coll(maskval, ['nodata', 'no-data'], case_sensitive=False):
                    maskval = None
                else:
                    try:
                        maskval_num = float(maskval)
                        maskval = maskval_num
                    except ValueError:
                        arg_parser.error("argument {} masking value '{}' is invalid".format(ARGSTR_SRC_SUFFIX, maskval))
            suffix_maskval_dict[suffix] = maskval

    if args.get(ARGGRP_FILTER_COMP).count(True) > 0 and args.get(ARGSTR_FILTER_OFF):
        arg_parser.error("argument {} is incompatible with filter options {}".format(ARGSTR_FILTER_OFF, ARGGRP_FILTER_COMP))

    if args.get(ARGSTR_DST_SUFFIX) is None:
        if args.get(ARGGRP_FILTER_COMP).count(True) > 0:
            args.set(ARGSTR_DST_SUFFIX, MASKED_SUFFIX_DEFAULT+get_mask_bitstring(*args.get(ARGGRP_FILTER_COMP)))
        else:
            args.set(ARGSTR_DST_SUFFIX, MASKED_SUFFIX_DEFAULT*(not args.get(ARGSTR_FILTER_OFF)))
        print("argument {} set automatically to: '{}'".format(ARGSTR_DST_SUFFIX, args.get(ARGSTR_DST_SUFFIX)))
    dst_suffix_raw = args.get(ARGSTR_DST_SUFFIX)
    dst_suffix_fixed = '_'+dst_suffix_raw.lstrip('_') if dst_suffix_raw != '' else ''
    if dst_suffix_fixed != dst_suffix_raw:
        args.set(ARGSTR_DST_SUFFIX, dst_suffix_fixed)

    if args.get(ARGGRP_FILTER_COMP).count(True) == 0 and not args.get(ARGSTR_FILTER_OFF):
        args.set(ARGGRP_FILTER_COMP)


    if args.get(ARGSTR_SCHEDULER) is not None:
        if args.get(ARGSTR_JOBSCRIPT) is None:
            jobscript_default = os.path.join(JOBSCRIPT_DIR, 'head_{}.sh'.format(args.get(ARGSTR_SCHEDULER)))
            if not os.path.isfile(jobscript_default):
                arg_parser.error(
                    "Default jobscript ({}) does not exist, ".format(jobscript_default)
                    + "please specify one with {} argument".format(ARGSTR_JOBSCRIPT))
            else:
                args.set(ARGSTR_JOBSCRIPT, jobscript_default)
                print("argument {} set automatically to: {}".format(ARGSTR_JOBSCRIPT, args.get(ARGSTR_JOBSCRIPT)))


    ## Validate argument values.

    if args.get(ARGSTR_TASKS_PER_JOB) is not None and not args.get(ARGSTR_SCHEDULER):
        arg_parser.error("{} option requires {} option".format(ARGSTR_TASKS_PER_JOB, ARGSTR_SCHEDULER))


    # Gather list of masks to apply to a (range of) source raster suffix(es).
    src_bitmasks = None

    if os.path.isfile(src) and src.endswith(args.get(ARGSTR_MASK_SUFFIX)):
        src_bitmask = src
        src_bitmasks = [src_bitmask]
        if suffix_maskval_dict is None:
            # maskFile_base = src_bitmask.replace(args.get(ARGSTR_MASK_SUFFIX), '')
            # suffix_maskval_dict = {src_rasterFile.replace(maskFile_base, ''): None
            #                        for src_rasterFile in glob.glob(maskFile_base+'*.tif')
            #                        if not src_rasterFile.endswith(args.get(ARGSTR_MASK_SUFFIX))}
            suffix_maskval_dict = {SRC_SUFFIX_CATCH_ALL: None}

    elif os.path.isfile(src) and not src.endswith('.txt'):
        if args.provided(ARGSTR_SRC_SUFFIX):
            raise ScriptArgumentError("{} option cannot be used when argument {} is a path to "
                                      "a source raster file".format(ARGSTR_SRC_SUFFIX, ARGSTR_SRC))
        args.set(ARGSTR_SRC_SUFFIX, None)
        suffix_maskval_dict = None
        src_raster = src
        src_suffix = None

        src_raster_dir = os.path.dirname(src_raster)
        src_raster_fname = os.path.basename(src_raster)
        beg, end = 0, len(src_raster_fname)
        # end = None
        while end >= 0:
            # end = max(src_raster_fname.rfind('_', beg, end), src_raster_fname.rfind('.', beg, end))
            # if os.path.isfile(os.path.join(src_raster_dir, src_raster_fname[beg:end].rstrip('_')+args.get(ARGSTR_MASK_SUFFIX))):
            if os.path.isfile(os.path.join(src_raster_dir, src_raster_fname[beg:end]+args.get(ARGSTR_MASK_SUFFIX))):
                src_suffix = src_raster_fname[end:]
                break
            end -= 1
        if src_suffix is None:
            arg_parser.error("Path of {} component for argument {} raster file "
                             "could not be determined".format(args.get(ARGSTR_MASK_SUFFIX), ARGSTR_SRC))

        src_bitmask = src_raster.replace(src_suffix, args.get(ARGSTR_MASK_SUFFIX))
        bitmaskSuffix = getBitmaskSuffix(src_bitmask)
        maskFile_base = src_bitmask.replace(bitmaskSuffix, '')
        src_suffix = src_raster.replace(maskFile_base, '')

        args.set(ARGSTR_MASK_SUFFIX, bitmaskSuffix)
        args.set(ARGSTR_SRC_SUFFIX, src_suffix)
        suffix_maskval_dict = {src_suffix: None}

        if not os.path.isfile(src_bitmask):
            arg_parser.error("{} mask component for argument {} raster file does not exist: {}".format(
                             args.get(ARGSTR_MASK_SUFFIX), ARGSTR_SRC, src_bitmask))
        src_bitmasks = [src_bitmask]

    elif os.path.isfile(src) and src.endswith('.txt'):
        bundle_file = src
        src_bitmasks = script_utils.read_task_bundle(bundle_file)

    elif os.path.isdir(src):
        srcdir = src
        src_bitmasks = []
        for root, dnames, fnames in os.walk(srcdir):
            for fn in fnames:
                if fn.endswith(args.get(ARGSTR_MASK_SUFFIX)):
                    src_bitmasks.append(os.path.join(root, fn))
        src_bitmasks.sort()

    if src_bitmasks is None:
        arg_parser.error("argument {} must be a path to either a directory or a file, "
                         "but was '{}'".format(ARGSTR_SRC, src))


    print("-----")
    print("Mask file suffix: {}".format(args.get(ARGSTR_MASK_SUFFIX)))
    print(
"""Selected bitmask components to mask:
[{}] EDGE
[{}] WATER
[{}] CLOUD""".format(
    *['X' if opt is True else ' ' for opt in args.get(ARGSTR_EDGE, ARGSTR_WATER, ARGSTR_CLOUD)]
    ))
    print("Output file suffix: '{}[.EXT]'".format(args.get(ARGSTR_DST_SUFFIX)))

    print("-----")
    print("Any warnings would appear here:")


    ## Create processing list.
    if suffix_maskval_dict is None:
        masks_to_apply = src_bitmasks
        num_src_rasters = None
    else:
        # Build processing list by only adding bitmasks for which
        # an output masked raster image(s) with the specified mask settings
        # does not already exist in the destination directory.
        src_suffixes = list(suffix_maskval_dict.keys())
        num_src_rasters = 0

        masks_to_apply = []
        for maskFile in src_bitmasks:
            bitmaskSuffix = getBitmaskSuffix(maskFile)
            bitmask_is_strip_lsf = (bitmaskSuffix == SUFFIX_STRIP_BITMASK_LSF)
            rasterFiles_checklist = set()
            switched_to_glob = False

            for rasterSuffix in src_suffixes:
                bitmaskSuffix_temp = bitmaskSuffix
                if rasterSuffix.startswith(STRIP_LSF_PREFIX):
                    if not bitmask_is_strip_lsf:
                        continue
                elif bitmask_is_strip_lsf:
                    bitmaskSuffix_temp = SUFFIX_STRIP_BITMASK
                src_rasterFile = maskFile.replace(bitmaskSuffix_temp, rasterSuffix)

                if '*' in rasterSuffix:
                    src_rasterffile_glob = set(glob.glob(src_rasterFile))
                    num_src_rasters += len(src_rasterffile_glob)
                    double_dipped = src_rasterffile_glob.intersection(rasterFiles_checklist)
                    if double_dipped:
                        raise ScriptArgumentError("{} option '{}' matches source raster files that "
                                                  "are already caught by another provided suffix!".format(
                            ARGSTR_SRC_SUFFIX, rasterSuffix
                        ))
                    rasterFiles_checklist = rasterFiles_checklist.union(src_rasterffile_glob)
                    mask_added = False
                    for src_rasterFile in src_rasterffile_glob:
                        if src_rasterFile.endswith(args.get(ARGSTR_MASK_SUFFIX)):
                            continue
                        if args.get(ARGSTR_OVERWRITE):
                            pass
                        else:
                            dst_rasterFile = get_dstFile(src_rasterFile, args)
                            if not os.path.isfile(dst_rasterFile):
                                pass
                            else:
                                continue
                        masks_to_apply.append(maskFile)
                        mask_added = True
                        break
                    if len(src_rasterffile_glob) == 0:
                        warnings.warn(
                            "{} option '{}' did not match any source raster files "
                            "corresponding to at least one mask file".format(
                            ARGSTR_SRC_SUFFIX, rasterSuffix
                        ))
                    if mask_added:
                        break

                elif os.path.isfile(src_rasterFile):
                    num_src_rasters += 1
                    double_dipped = (src_rasterFile in rasterFiles_checklist)
                    if double_dipped:
                        raise ScriptArgumentError("{} option '{}' matches source raster files that "
                                                  "are already caught by another provided suffix!".format(
                            ARGSTR_SRC_SUFFIX, rasterSuffix
                        ))
                    rasterFiles_checklist.add(src_rasterFile)
                    if args.get(ARGSTR_OVERWRITE):
                        pass
                    else:
                        dst_rasterFile = get_dstFile(src_rasterFile, args)
                        if not os.path.isfile(dst_rasterFile):
                            pass
                        else:
                            continue
                    masks_to_apply.append(maskFile)
                    break

                else:
                    rasterSuffix_glob = '*'+rasterSuffix.lstrip('_')
                    src_rasterFile = maskFile.replace(bitmaskSuffix_temp, rasterSuffix_glob)
                    src_rasterffile_glob = set(glob.glob(src_rasterFile))
                    num_src_rasters += len(src_rasterffile_glob)
                    double_dipped = src_rasterffile_glob.intersection(rasterFiles_checklist)
                    if double_dipped:
                        raise ScriptArgumentError(
                            "{} option '{}' (auto-globbed) matches source raster files that "
                            "are already caught by another provided suffix!".format(
                            ARGSTR_SRC_SUFFIX, rasterSuffix_glob
                        ))
                    rasterFiles_checklist = rasterFiles_checklist.union(src_rasterffile_glob)
                    mask_added = False
                    for src_rasterFile in src_rasterffile_glob:
                        if src_rasterFile.endswith(args.get(ARGSTR_MASK_SUFFIX)):
                            continue
                        if args.get(ARGSTR_OVERWRITE):
                            pass
                        else:
                            dst_rasterFile = get_dstFile(src_rasterFile, args)
                            if not os.path.isfile(dst_rasterFile):
                                pass
                            else:
                                continue
                        masks_to_apply.append(maskFile)
                        mask_added = True
                        break
                    if len(src_rasterffile_glob) == 0:
                        warnings.warn(
                            "{} option '{}' (auto-globbed) did not match any source raster files "
                            "corresponding to at least one mask file".format(
                            ARGSTR_SRC_SUFFIX, rasterSuffix_glob
                        ))
                    else:
                        suffix_maskval_dict[rasterSuffix_glob] = suffix_maskval_dict[rasterSuffix]
                        del suffix_maskval_dict[rasterSuffix]
                        switched_to_glob = True
                    if mask_added:
                        break
            if switched_to_glob:
                src_suffixes = list(suffix_maskval_dict.keys())

    print("-----")

    if suffix_maskval_dict is None:
        print("Masking all *_[RASTER-SUFFIX(.tif)] raster components corresponding to "
              "source *{} file(s), using source NoData values".format(args.get(ARGSTR_MASK_SUFFIX)))
    else:
        print("[Raster Suffix, Masking Value]")
        for suffix, maskval in suffix_maskval_dict.items():
            print("{}, {}".format(suffix, maskval if maskval is not None else '(source NoDataVal)'))
        print("-----")

    num_tasks = len(masks_to_apply)

    print("Number of source mask files found: {}".format(len(src_bitmasks)))
    if num_src_rasters is not None:
        print("Number of source raster files found: {}".format(num_src_rasters))
    print("Number of incomplete masking tasks: {}".format(num_tasks))

    if num_tasks == 0:
        sys.exit(0)

    # Pause for user review.
    print("-----")
    wait_seconds = 10
    print("Sleeping {} seconds before task submission".format(wait_seconds))
    sleep(wait_seconds)
    print("-----")


    ## Create output directories if they don't already exist.
    if not args.get(ARGSTR_DRYRUN):
        for dir_argstr, dir_path in list(zip(ARGGRP_OUTDIR, args.get_as_list(ARGGRP_OUTDIR))):
            if dir_path is not None and not os.path.isdir(dir_path):
                print("Creating argument {} directory: {}".format(dir_argstr, dir_path))
                os.makedirs(dir_path)


    ## Process masks.

    if args.get(ARGSTR_SCHEDULER) is not None:
        # Process masks in batch.

        tasks_per_job = args.get(ARGSTR_TASKS_PER_JOB)
        src_files = (masks_to_apply if tasks_per_job is None else
                     script_utils.write_task_bundles(masks_to_apply, tasks_per_job,
                                                     args.get(ARGSTR_SCRATCH),
                                                      '{}_{}'.format(JOB_ABBREV, ARGSTR_SRC)))

        jobnum_fmt = script_utils.get_jobnum_fmtstr(src_files)
        last_job_email = args.get(ARGSTR_EMAIL)

        args_batch = args
        args_single = copy.deepcopy(args)
        args_single.unset(*ARGGRP_BATCH)

        job_num = 0
        num_jobs = len(src_files)
        for srcfp in src_files:
            job_num += 1

            args_single.set(ARGSTR_SRC, srcfp)
            if last_job_email and job_num == num_jobs:
                args_single.set(ARGSTR_EMAIL, last_job_email)
            cmd_single = args_single.get_cmd()

            job_name = JOB_ABBREV+jobnum_fmt.format(job_num)
            cmd = args_single.get_jobsubmit_cmd(
                args_batch.get(ARGSTR_SCHEDULER),
                jobscript=args_batch.get(ARGSTR_JOBSCRIPT),
                jobname=job_name, time_hr=JOB_WALLTIME_HR, memory_gb=JOB_MEMORY_GB, email=args.get(ARGSTR_EMAIL),
                envvars=[args_batch.get(ARGSTR_JOBSCRIPT), JOB_ABBREV, cmd_single, PYTHON_VERSION_ACCEPTED_MIN]
            )

            print(cmd)
            if not args_batch.get(ARGSTR_DRYRUN):
                subprocess.call(cmd, shell=True, cwd=args_batch.get(ARGSTR_LOGDIR))

    else:
        error_trace = None
        try:
            # Process masks in serial.

            for i, maskFile in enumerate(masks_to_apply):
                print("Mask ({}/{}): {}".format(i+1, num_tasks, maskFile))
                if not args.get(ARGSTR_DRYRUN):
                    mask_rasters(maskFile, suffix_maskval_dict, args)

        except KeyboardInterrupt:
            raise

        except Exception as e:
            with script_utils.capture_stdout_stderr() as out:
                traceback.print_exc()
            caught_out, caught_err = out
            error_trace = caught_err
            print(error_trace)
            if e.__class__ is ImportError:
                print("\nFailed to import necessary module(s)")
                print("If running on a Linux system where the jobscripts/init.sh file has been properly"
                      " set up, try running the following command to activate a working environment"
                      " in your current shell session:\n{}".format("source {} {}".format(JOBSCRIPT_INIT, JOB_ABBREV)))
                print('')

        if type(args.get(ARGSTR_EMAIL)) is str:
            # Send email notification of script completion.
            email_body = SCRIPT_RUNCMD
            if error_trace is not None:
                email_status = "ERROR"
                email_body += "\n{}\n".format(error_trace)
            else:
                email_status = "COMPLETE"
            email_subj = "{} - {}".format(email_status, SCRIPT_FNAME)
            script_utils.send_email(args.get(ARGSTR_EMAIL), email_subj, email_body)

        if error_trace is not None:
            sys.exit(1)


def startswith_one_of_coll(check_string, string_starting_coll, case_sensitive=True, return_match=False):
    for s_start in string_starting_coll:
        if check_string.startswith(s_start) or (not case_sensitive and check_string.lower().startswith(s_start.lower())):
            return s_start if return_match else True
    return None if return_match else False


def getBitmaskSuffix(bitmaskFile):
    global SUFFIX_PRIORITY_BITMASK
    for bitmaskSuffix in SUFFIX_PRIORITY_BITMASK:
        if bitmaskFile.endswith(bitmaskSuffix):
            return bitmaskSuffix
    return None


def get_mask_bitstring(edge, water, cloud):
    import numpy as np
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


def get_dstFile(rasterFile, args):
    dstFname_prefix, dstFname_ext = os.path.splitext(os.path.basename(rasterFile))
    dstFname = '{}{}{}'.format(
        dstFname_prefix, args.get(ARGSTR_DST_SUFFIX) if args.get(ARGSTR_DST_SUFFIX) != '' else '', dstFname_ext)
    dstFile = (os.path.join(args.get(ARGSTR_DSTDIR), dstFname) if args.get(ARGSTR_DSTDIR) is not None else
               os.path.join(os.path.dirname(rasterFile), dstFname))
    return dstFile


def mask_rasters(maskFile, suffix_maskval_dict, args):
    import numpy as np
    import lib.raster_array_tools as rat

    global SRC_SUFFIX_CATCH_ALL
    nodata_opt = args.get(ARGSTR_DST_NODATA)
    bitmaskSuffix = getBitmaskSuffix(maskFile)
    bitmask_is_strip_lsf = (bitmaskSuffix == SUFFIX_STRIP_BITMASK_LSF)

    if suffix_maskval_dict is None:
        # maskFile_base = maskFile.replace(bitmaskSuffix, '')
        # suffix_maskval_dict = {src_rasterFile.replace(maskFile_base, ''): None
        #                        for src_rasterFile in glob.glob(maskFile_base+SRC_SUFFIX_CATCH_ALL)
        #                        if not src_rasterFile.endswith(bitmaskSuffix)}
        suffix_maskval_dict = {SRC_SUFFIX_CATCH_ALL: None}

    if True in args.get(ARGSTR_EDGE, ARGSTR_WATER, ARGSTR_CLOUD):
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
    mask_pyramid_current = mask_pyramids[0]

    # Apply mask to source raster images and save results.
    for src_suffix, maskval in suffix_maskval_dict.items():
        bitmaskSuffix_temp = bitmaskSuffix
        if src_suffix.startswith(STRIP_LSF_PREFIX):
            if not bitmask_is_strip_lsf:
                continue
        elif bitmask_is_strip_lsf:
            bitmaskSuffix_temp = SUFFIX_STRIP_BITMASK

        src_rasterFile = maskFile.replace(bitmaskSuffix_temp, src_suffix)
        if '*' in src_rasterFile:
            src_rasterFile_list = glob.glob(src_rasterFile)
            if len(src_rasterFile_list) == 0:
                print("No source rasters found matching filename pattern: {}".format(src_rasterFile))
                continue
            maskFile_globbed = None
            for src_rasterFile in src_rasterFile_list:
                if src_rasterFile.endswith(bitmaskSuffix_temp):
                    maskFile_globbed = src_rasterFile
                    break
            if maskFile_globbed is not None:
                src_rasterFile_list.remove(maskFile_globbed)
        else:
            src_rasterFile_list = [src_rasterFile]

        for src_rasterFile in src_rasterFile_list:
            if not os.path.isfile(src_rasterFile):
                print("Source raster does not exist: {}".format(src_rasterFile))
                continue
            dst_rasterFile = get_dstFile(src_rasterFile, args)
            if os.path.isfile(dst_rasterFile):
                print("Output raster already exists: {}".format(dst_rasterFile))
                if not args.get(ARGSTR_OVERWRITE):
                    continue

            print("Masking source raster ({}) to output raster ({})".format(src_rasterFile, dst_rasterFile))

            # Read in source raster.
            dst_array, src_nodataval = rat.extractRasterData(src_rasterFile, 'array', 'nodata_val')

            print("Source NoData value: {}".format(src_nodataval))

            # Set masking value to source NoDataVal if necessary.
            if maskval is None:
                if src_nodataval is None:
                    print("Source raster does not have a set NoData value, "
                          "so masking value cannot be automatically determined; skipping")
                    continue
                else:
                    maskval = src_nodataval

            print("Masking value: {}".format(maskval))

            if mask_pyramid_current is not None:
                # Apply mask.
                if mask_pyramid_current.shape != dst_array.shape:
                    # See if the required mask pyramid level has already been built.
                    mask_pyramid_current = None
                    for arr in mask_pyramids:
                        if arr.shape == dst_array.shape:
                            mask_pyramid_current = arr
                            break
                    if mask_pyramid_current is None:
                        # Build the required mask pyramid level and add to pyramids.
                        mask_pyramid_current = rat.imresize(mask_select, dst_array.shape, interp='nearest')
                        mask_pyramids.append(mask_pyramid_current)
                dst_array[mask_pyramid_current] = maskval

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

            print("Output NoData value: {}".format(dst_nodataval))

            # Save output masked raster.
            rat.saveArrayAsTiff(dst_array, dst_rasterFile,
                                nodata_val=dst_nodataval,
                                like_raster=src_rasterFile)



if __name__ == '__main__':
    main()
