#!/usr/bin/env python

# Erik Husby, Claire Porter; Polar Geospatial Center, University of Minnesota; 2019


from __future__ import division
from lib import script_utils

PYTHON_VERSION_ACCEPTED_MIN = "2.7"  # supports multiple dot notation
if script_utils.PYTHON_VERSION < script_utils.VersionString(PYTHON_VERSION_ACCEPTED_MIN):
    raise script_utils.VersionError("Python version ({}) is below accepted minimum ({})".format(
        script_utils.PYTHON_VERSION, PYTHON_VERSION_ACCEPTED_MIN))


import argparse
import copy
import filecmp
import gc
import glob
import os
import re
import shutil
import subprocess
import sys
import traceback
import warnings
from collections import OrderedDict
from time import sleep
from datetime import datetime
from distutils.version import StrictVersion

from lib import script_utils
from lib.script_utils import ScriptArgumentError, ExternalError, InvalidArgumentError


##############################

## Core globals

SCRIPT_VERSION_NUM = script_utils.VersionString('4.1')

# Script paths and execution
SCRIPT_FILE = os.path.abspath(os.path.realpath(__file__))
SCRIPT_FNAME = os.path.basename(SCRIPT_FILE)
SCRIPT_NAME, SCRIPT_EXT = os.path.splitext(SCRIPT_FNAME)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE)
SCRIPT_RUNCMD = ' '.join(sys.argv)+'\n'
PYTHON_EXE = 'python -u'

##############################

## Argument globals

# Argument strings
ARGSTR_SRC = 'src'
ARGSTR_RES = 'res'
ARGSTR_DST = '--dst'
ARGSTR_META_TRANS_DIR = '--meta-trans-dir'
ARGSTR_NO_BROWSE = '--no-browse'
ARGSTR_BUILD_AUX = '--build-aux'
ARGSTR_REBUILD_AUX = '--rebuild-aux'
ARGSTR_DEM_TYPE = '--dem-type'
ARGSTR_MASK_VER = '--mask-ver'
ARGSTR_NOENTROPY = '--noentropy'
ARGSTR_NOWATER = '--nowater'
ARGSTR_NOCLOUD = '--nocloud'
ARGSTR_UNFILTERED = '--unf'
ARGSTR_NOFILTER_COREG = '--nofilter-coreg'
ARGSTR_SAVE_COREG_STEP = '--save-coreg-step'
ARGSTR_RMSE_CUTOFF = '--rmse-cutoff'
ARGSTR_REMERGE_STRIPS = '--remerge-strips'
ARGSTR_SCHEDULER = '--scheduler'
ARGSTR_JOBSCRIPT = '--jobscript'
ARGSTR_LOGDIR = '--logdir'
ARGSTR_EMAIL = '--email'
ARGSTR_RESTART = '--restart'
ARGSTR_REMOVE_INCOMPLETE = '--remove-incomplete'
ARGSTR_USE_OLD_MASKS = '--use-old-masks'
ARGSTR_CLEANUP_ON_FAILURE = '--cleanup-on-failure'
ARGSTR_OLD_ORG = '--old-org'
ARGSTR_DRYRUN = '--dryrun'
ARGSTR_STRIPID = '--stripid'
ARGSTR_SCENEDIRNAME = '--scenedirname'
ARGSTR_SKIP_ORTHO2_ERROR = '--skip-xtrack-missing-ortho2-error'
ARGSTR_SCENE_MASKS_ONLY = '--build-scene-masks-only'
ARGSTR_USE_PIL_IMRESIZE = '--use-pil-imresize'

# Argument choices
ARGCHO_DEM_TYPE_LSF = 'lsf'
ARGCHO_DEM_TYPE_NON_LSF = 'non-lsf'
ARGCHO_DEM_TYPE = [
    ARGCHO_DEM_TYPE_LSF,
    ARGCHO_DEM_TYPE_NON_LSF
]
ARGCHO_MASK_VER_MASKV1 = 'maskv1'
ARGCHO_MASK_VER_MASKV2 = 'maskv2'
ARGCHO_MASK_VER_REMA2A = 'rema2a'
ARGCHO_MASK_VER_MASK8M = 'mask8m'
ARGCHO_MASK_VER_BITMASK = 'bitmask'
ARGCHO_MASK_VER = [
    ARGCHO_MASK_VER_MASKV1,
    ARGCHO_MASK_VER_MASKV2,
    ARGCHO_MASK_VER_REMA2A,
    ARGCHO_MASK_VER_MASK8M,
    ARGCHO_MASK_VER_BITMASK
]
ARGCHO_SAVE_COREG_STEP_OFF = 'off'
ARGCHO_SAVE_COREG_STEP_META = 'meta'
ARGCHO_SAVE_COREG_STEP_ALL = 'all'
ARGCHO_SAVE_COREG_STEP = [
    ARGCHO_SAVE_COREG_STEP_OFF,
    ARGCHO_SAVE_COREG_STEP_META,
    ARGCHO_SAVE_COREG_STEP_ALL
]
ARGCHO_CLEANUP_ON_FAILURE_MASKS = 'masks'
ARGCHO_CLEANUP_ON_FAILURE_STRIP = 'strip'
ARGCHO_CLEANUP_ON_FAILURE_OUTPUT = 'output'
ARGCHO_CLEANUP_ON_FAILURE_NONE = 'none'
ARGCHO_CLEANUP_ON_FAILURE = [
    ARGCHO_CLEANUP_ON_FAILURE_MASKS,
    ARGCHO_CLEANUP_ON_FAILURE_STRIP,
    ARGCHO_CLEANUP_ON_FAILURE_OUTPUT,
    ARGCHO_CLEANUP_ON_FAILURE_NONE
]

# Segregation of argument choices
MASK_VER_8M = [
    ARGCHO_MASK_VER_MASKV1,
    ARGCHO_MASK_VER_REMA2A,
    ARGCHO_MASK_VER_MASK8M,
    ARGCHO_MASK_VER_BITMASK
]
MASK_VER_2M = [
    ARGCHO_MASK_VER_MASKV1,
    ARGCHO_MASK_VER_MASKV2,
    ARGCHO_MASK_VER_BITMASK
]
MASK_VER_XM = [
    ARGCHO_MASK_VER_BITMASK
]

# Argument groups
ARGGRP_OUTDIR = [ARGSTR_DST, ARGSTR_LOGDIR]
ARGGRP_BATCH = [ARGSTR_SCHEDULER, ARGSTR_JOBSCRIPT, ARGSTR_LOGDIR, ARGSTR_EMAIL]
ARGGRP_UNFILTERED = [ARGSTR_NOWATER, ARGSTR_NOCLOUD]

##############################

## Batch settings

JOBSCRIPT_DIR = os.path.join(SCRIPT_DIR, 'jobscripts')
JOBSCRIPT_INIT = os.path.join(JOBSCRIPT_DIR, 'init.sh')
JOB_ABBREV = 's2s'
JOB_WALLTIME_HR = 3
JOB_MEMORY_GB = 40
JOB_NCORES = 4
JOB_NODE = None

##############################

## Custom globals

SUFFIX_PRIORITY_DEM = ['dem_smooth.tif', 'dem.tif']
SUFFIX_PRIORITY_MATCHTAG = ['matchtag_mt.tif', 'matchtag.tif']
SUFFIX_PRIORITY_ORTHO1 = ['ortho_image1.tif', 'ortho_image_1.tif', 'ortho1.tif', 'ortho_1.tif', 'ortho.tif']
SUFFIX_PRIORITY_ORTHO2 = ['ortho_image2.tif', 'ortho_image_2.tif', 'ortho2.tif', 'ortho_2.tif']

DEM_TYPE_SUFFIX_DICT = {
    ARGCHO_DEM_TYPE_LSF: 'dem_smooth.tif',
    ARGCHO_DEM_TYPE_NON_LSF: 'dem.tif'
}

RE_STRIPID_STR = "^(?:SETSM_s2s[0-9]{3}_)?([A-Z0-9]{4}_.*?_?[0-9A-F]{16}_.*?_?[0-9A-F]{16}).*$"
RE_STRIPID = re.compile(RE_STRIPID_STR)
RE_STRIPFNAME_SEGNUM = re.compile("_seg(\d+)_", re.I)
RE_SCENEMETA_SETSM_VERSION_STR = "^setsm[ _]version=.*$"
RE_SCENEMETA_SETSM_VERSION = re.compile(RE_SCENEMETA_SETSM_VERSION_STR, re.I|re.MULTILINE)
RE_SCENEMETA_GROUP_VERSION = re.compile("^group[ _]version=.*$", re.I|re.MULTILINE)
RE_STRIPMETA_SCENE_NAME_KEY = re.compile("^scene \d+ name=", re.I)

##############################


class MetaReadError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


def argparser_init():

    parser = argparse.ArgumentParser(
        formatter_class=script_utils.RawTextArgumentDefaultsHelpFormatter,
        description=' '.join([
            "Filters scene DEMs in a source directory,",
            "then mosaics them into strips and saves the results.",
            "\nBatch work is done in units of strip-pair IDs, as parsed from scene dem filenames",
            "(see {} argument for how this is parsed).".format(ARGSTR_STRIPID)
        ])
    )

    # Positional arguments

    parser.add_argument(
        ARGSTR_SRC,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_SRC,
            existcheck_fn=os.path.isdir,
            existcheck_reqval=True),
        help=' '.join([
            "Path to source directory containing scene DEMs to process.",
            "Scene DEMs should be organized in folders named like '<stripid>_<resolution>' (e.g. WV02_20201204_10300100B061A200_10300100B19D1900_2m)",
            "placed directly within the source directory, unless {} option is provided, in which case".format(ARGSTR_OLD_ORG),
            "all scene DEM result files must be stored flat within the source directory.",
            "If {} is not specified, this path should contain the folder 'tif_results'".format(ARGSTR_DST),
            "so that the destination directory can be automatically derived.".format(ARGSTR_DST),
            "The range of source scenes worked on may be limited with the {} argument.".format(ARGSTR_STRIPID)
        ])
    )
    parser.add_argument(
        ARGSTR_RES,
        type=float,
        help="Resolution of target DEMs in meters."
    )

    # Optional arguments

    parser.add_argument(
        ARGSTR_DST,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_DST,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=False),
        help=' '.join([
            "Path to destination directory for output mosaicked strip data.",
            "(default is {}.(reverse)replace('tif_results', 'strips'))".format(ARGSTR_SRC)
        ])
    )

    parser.add_argument(
        ARGSTR_OLD_ORG,
        action='store_true',
        help=' '.join([
            "For source and destination directories, use old scene and strip results organization",
            "(*flat directory structure*, used prior to reorganization into strip pairname folders)."
        ])
    )

    parser.add_argument(
        ARGSTR_DEM_TYPE,
        type=str,
        choices=ARGCHO_DEM_TYPE,
        default=ARGCHO_DEM_TYPE_LSF,
        help=' '.join([
            "Which version of all scene DEMs to work with.",
            "\n'{}': Use the LSF DEM with '{}' file suffix.".format(ARGCHO_DEM_TYPE_LSF, DEM_TYPE_SUFFIX_DICT[ARGCHO_DEM_TYPE_LSF]),
            "\n'{}': Use the non-LSF DEM with '{}' file suffix.".format(ARGCHO_DEM_TYPE_NON_LSF, DEM_TYPE_SUFFIX_DICT[ARGCHO_DEM_TYPE_NON_LSF]),
            "\n"
        ])
    )

    parser.add_argument(
        ARGSTR_UNFILTERED,
        action='store_true',
        help=' '.join([
            "Shortcut for setting {} options to create \"unfiltered\" strips.".format(ARGGRP_UNFILTERED),
            "\nDefault for {} argument becomes "
            "({}.(reverse)replace('tif_results', 'strips_unf')).".format(ARGSTR_DST, ARGSTR_SRC),
            "\nCan only be used when {}={}.".format(ARGSTR_MASK_VER, ARGCHO_MASK_VER_BITMASK)
        ])
    )

    parser.add_argument(
        ARGSTR_USE_OLD_MASKS,
        action='store_true',
        help="Use existing scene masks instead of deleting and re-filtering."
    )

    parser.add_argument(
        ARGSTR_SCENE_MASKS_ONLY,
        action='store_true',
        help="Build scene masks and then exit before proceeding to strip-building steps."
    )

    parser.add_argument(
        ARGSTR_REMERGE_STRIPS,
        action='store_true',
        help=' '.join([
            "Source are strip segment results to be treated as input scenes for",
            "rerunning through the coregistration and mosaicking steps to produce",
            "a new set of 're-merged' strip results."
        ])
    )

    parser.add_argument(
        ARGSTR_RMSE_CUTOFF,
        type=float,
        choices=None,
        default=3.5,
        help=' '.join([
            "Maximum RMSE from coregistration step tolerated for scene merging.",
            "A value greater than this causes a new strip segment to be created."
        ])
    )

    parser.add_argument(
        ARGSTR_MASK_VER,
        type=str,
        choices=ARGCHO_MASK_VER,
        default=ARGCHO_MASK_VER_BITMASK,
        help=' '.join([
            "Filtering scheme to use when generating mask raster images,",
            "to classify bad data in scene DEMs.",
            "\n'{}': Two-component (edge, data density) filter to create".format(ARGCHO_MASK_VER_MASKV1),
                    "separate edgemask and datamask files for each scene.",
            "\n'{}': Three-component (edge, water, cloud) filter to create".format(ARGCHO_MASK_VER_MASKV2),
                    "classical 'flat' binary masks for 2m DEMs.",
            "\n'{}': Same filter as '{}', but distinguish between".format(ARGCHO_MASK_VER_BITMASK, ARGCHO_MASK_VER_MASKV2),
                    "the different filter components by creating a bitmask.",
            "\n'{}': Filter designed specifically for 8m Antarctic DEMs.".format(ARGCHO_MASK_VER_REMA2A),
            "\n'{}': General-purpose filter for 8m DEMs.".format(ARGCHO_MASK_VER_MASK8M),
            "\n"
        ])
    )

    parser.add_argument(
        ARGSTR_NOENTROPY,
        action='store_true',
        help=' '.join([
            "Use filter without entropy protection.",
            "Can only be used when {}={}.".format(ARGSTR_MASK_VER, ARGCHO_MASK_VER_MASKV1)
        ])
    )
    parser.add_argument(
        ARGSTR_NOWATER,
        action='store_true',
        help=' '.join([
            "Use filter without water masking.",
            "Can only be used when {}={}.".format(ARGSTR_MASK_VER, ARGCHO_MASK_VER_BITMASK)
        ])
    )
    parser.add_argument(
        ARGSTR_NOCLOUD,
        action='store_true',
        help=' '.join([
            "Use filter without cloud masking.",
            "Can only be used when {}={}.".format(ARGSTR_MASK_VER, ARGCHO_MASK_VER_BITMASK)
        ])
    )
    parser.add_argument(
        ARGSTR_NOFILTER_COREG,
        action='store_true',
        help=' '.join([
            "If {}/{}, turn off the respective filter(s) during".format(ARGSTR_NOWATER, ARGSTR_NOCLOUD),
            "coregistration step in addition to mosaicking step.",
            "Can only be used when {}={}.".format(ARGSTR_MASK_VER, ARGCHO_MASK_VER_BITMASK)
        ])
    )
    parser.add_argument(
        ARGSTR_SAVE_COREG_STEP,
        type=str,
        choices=ARGCHO_SAVE_COREG_STEP,
        default=ARGCHO_SAVE_COREG_STEP_OFF,
        help=' '.join([
            "If {}/{}, save output from coregistration step in directory".format(ARGSTR_NOWATER, ARGSTR_NOCLOUD),
            "'`dstdir`_coreg_filtXXX' where [XXX] is the bit-code corresponding to filter components",
            "([cloud, water, edge], respectively) applied during the coregistration step.",
            "By default, all three filter components are applied so this code is 111.",
            "\nIf '{}', do not save output from coregistration step.".format(ARGCHO_SAVE_COREG_STEP_OFF),
            "\nIf '{}', save only the *_meta.txt component of output strip segments.".format(ARGCHO_SAVE_COREG_STEP_META),
            "(useful for subsequent runs with {} argument)".format(ARGSTR_META_TRANS_DIR),
            "\nIf '{}', save all output from coregistration step, including both".format(ARGCHO_SAVE_COREG_STEP_ALL),
            "metadata and raster components.",
            "\nCan only be used when {}={}, and has no affect if neither".format(ARGSTR_MASK_VER, ARGCHO_MASK_VER_BITMASK),
            "{} nor {} arguments are provided, or either".format(ARGSTR_NOWATER, ARGSTR_NOCLOUD),
            "{} or {} arguments are provided since then the".format(ARGSTR_META_TRANS_DIR, ARGSTR_NOFILTER_COREG),
            "coregistration and mosaicking steps are effectively rolled into one step.",
            "\n"
        ])
    )

    parser.add_argument(
        ARGSTR_META_TRANS_DIR,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_META_TRANS_DIR,
            existcheck_fn=os.path.isdir,
            existcheck_reqval=True),
        help=' '.join([
            "Path to directory of old strip metadata from which translation values",
            "will be parsed to skip scene coregistration step."
        ])
    )

    parser.add_argument(
        ARGSTR_NO_BROWSE,
        action='store_true',
        help=' '.join([
            "Do not build 10m hillshade '*_dem_10m_shade.tif' images alongside output DEM strip segments.",
        ])
    )

    parser.add_argument(
        ARGSTR_BUILD_AUX,
        action='store_true',
        help=' '.join([
            "Build a suite of downsampled browse images alongside output DEM strip segments.",
            "These images are primarily useful to PGC in tile mosaicking efforts."
        ])
    )

    parser.add_argument(
        ARGSTR_REBUILD_AUX,
        action='store_true',
        help=' '.join([
            "Rebuild browse images along existing output DEM strip segments."
        ])
    )

    parser.add_argument(
        ARGSTR_CLEANUP_ON_FAILURE,
        type=str,
        choices=ARGCHO_CLEANUP_ON_FAILURE,
        default=ARGCHO_CLEANUP_ON_FAILURE_OUTPUT,
        help=' '.join([
            "Which type of output files should be automatically removed upon encountering an error.",
            "\nIf '{}', remove all scene masks for the strip-pair ID if any scene masks are created".format(ARGCHO_CLEANUP_ON_FAILURE_MASKS),
            "during this run (meaning {} option was not used, or if it was used but additional".format(ARGSTR_USE_OLD_MASKS),
            "scene masks were created).",
            "\nIf '{}', remove all output strip results for the strip-pair ID from the destination.".format(ARGCHO_CLEANUP_ON_FAILURE_STRIP),
            "\nIf '{}', remove both scene masks and output strip results for the strip-pair ID.".format(ARGCHO_CLEANUP_ON_FAILURE_OUTPUT),
            "\nIf '{}', remove nothing on error.".format(ARGCHO_CLEANUP_ON_FAILURE_NONE),
            "\n"
        ])
    )
    parser.add_argument(
        ARGSTR_REMOVE_INCOMPLETE,
        action='store_true',
        help="Only remove unfinished (no .fin file) output, do not build strips."
    )
    parser.add_argument(
        ARGSTR_RESTART,
        action='store_true',
        help="Remove any unfinished (no .fin file) output before submitting all unfinished strips."
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

    parser.add_argument(
        ARGSTR_STRIPID,
        help=' '.join([
            "Run filtering and mosaicking for a single strip with strip-pair ID",
            "as parsed from scene DEM filenames using the following regex: '{}'".format(RE_STRIPID_STR),
            "\nA text file containing a list of strip-pair IDs, each on a separate line,"
            "may instead be provided for batch processing of select strips."
        ])
    )

    parser.add_argument(
        ARGSTR_SCENEDIRNAME,
        help=' '.join([
            "Name of folder containing the scene DEM files for a single strip-pair ID",
            "designated by the {} argument".format(ARGSTR_STRIPID)
        ])
    )

    parser.add_argument(
        ARGSTR_SKIP_ORTHO2_ERROR,
        action='store_true',
        help=' '.join([
            "If at least one scene in a cross-track strip is missing the second ortho component raster,",
            "do not throw an error but instead build the strip as if it were in-track with only one ortho."
        ])
    )

    parser.add_argument(
        ARGSTR_USE_PIL_IMRESIZE,
        action='store_true',
        help=' '.join([
            "Use PIL imresize method over usual fast OpenCV resize method for final resize from",
            "8m processing resolution back to native scene raster resolution when generating",
            "output scene mask rasters. This is to avoid an OpenCV error with unknown cause that",
            "can occur when using OpenCV resize on some 50cm scenes at Blue Waters."
        ])
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

    res_m = args.get(ARGSTR_RES)
    if int(res_m) == res_m:
        res_m = int(res_m)
        res_str = '{}m'.format(res_m)
        args.set(ARGSTR_RES, int(res_m))
    else:
        res_cm = res_m * 100
        if int(res_cm) == res_cm:
            res_cm = int(res_cm)
            res_str = '{}cm'.format(res_cm)
        else:
            raise ScriptArgumentError("Resolution is not in whole meters or whole centimeters")

    if args.get(ARGSTR_DST) is not None:
        if (   args.get(ARGSTR_SRC) == args.get(ARGSTR_DST)
            or (    os.path.isdir(args.get(ARGSTR_DST))
                and filecmp.cmp(args.get(ARGSTR_SRC), args.get(ARGSTR_DST)))):
            arg_parser.error("argument {} directory is the same as "
                             "argument {} directory".format(ARGSTR_SRC, ARGSTR_DST))
    elif args.get(ARGSTR_SCENE_MASKS_ONLY):
        args.set(ARGSTR_DST, ARGSTR_SRC)
    else:
        # Set default dst dir.
        split_ind = args.get(ARGSTR_SRC).rfind('tif_results')
        if split_ind == -1:
            arg_parser.error("argument {} path does not contain 'tif_results', "
                             "so default argument {} cannot be set".format(ARGSTR_SRC, ARGSTR_DST))
        args.set(ARGSTR_DST, (  args.get(ARGSTR_SRC)[:split_ind]
                              + args.get(ARGSTR_SRC)[split_ind:].replace(
                    'tif_results', 'strips' if not args.get(ARGSTR_UNFILTERED) else 'strips_unf')))
        print("argument {} set automatically to: {}".format(ARGSTR_DST, args.get(ARGSTR_DST)))

    argcho_dem_type_opp = ARGCHO_DEM_TYPE_NON_LSF if args.get(ARGSTR_DEM_TYPE) == ARGCHO_DEM_TYPE_LSF else ARGCHO_DEM_TYPE_LSF

    if args.get(ARGSTR_UNFILTERED):
        args.set(ARGGRP_UNFILTERED)
        print("via provided argument {}, arguments {} set automatically".format(ARGSTR_UNFILTERED, ARGGRP_UNFILTERED))

    if args.get(ARGSTR_REBUILD_AUX):
        args.set(ARGSTR_USE_OLD_MASKS)
        print("via provided argument {}, argument {} set automatically".format(ARGSTR_REBUILD_AUX, ARGSTR_USE_OLD_MASKS))

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

    if args.get(ARGSTR_REMERGE_STRIPS):
        demSuffix = 'dem.tif'
    else:
        demSuffix = DEM_TYPE_SUFFIX_DICT[args.get(ARGSTR_DEM_TYPE)]

    if args.get(ARGSTR_REMERGE_STRIPS) and not args.provided(ARGSTR_CLEANUP_ON_FAILURE):
        args.set(ARGSTR_CLEANUP_ON_FAILURE, ARGCHO_CLEANUP_ON_FAILURE_STRIP)


    ## Validate argument values.

    if args.get(ARGSTR_RES) == 8:
        res_req_mask_ver = MASK_VER_8M
    elif args.get(ARGSTR_RES) == 2:
        res_req_mask_ver = MASK_VER_2M
    else:
        res_req_mask_ver = MASK_VER_XM
    if args.get(ARGSTR_MASK_VER) not in res_req_mask_ver:
        arg_parser.error("argument {} must be one of {} for {}-meter argument {}".format(
            ARGSTR_MASK_VER, res_req_mask_ver, args.get(ARGSTR_RES), ARGSTR_RES
        ))

    if args.get(ARGSTR_NOENTROPY) and args.get(ARGSTR_MASK_VER) != ARGCHO_MASK_VER_MASKV1:
        arg_parser.error("{} option is compatible only with {} option".format(
            ARGSTR_NOENTROPY, ARGCHO_MASK_VER_MASKV1
        ))
    if (    (args.get(ARGSTR_NOWATER) or args.get(ARGSTR_NOCLOUD))
        and args.get(ARGSTR_MASK_VER) != ARGCHO_MASK_VER_BITMASK):
        arg_parser.error("{}/{} option(s) can only be used when {}='{}'".format(
            ARGSTR_NOWATER, ARGSTR_NOCLOUD, ARGSTR_MASK_VER, ARGCHO_MASK_VER_BITMASK
        ))
    if args.get(ARGSTR_NOFILTER_COREG) and [args.get(ARGSTR_NOWATER), args.get(ARGSTR_NOCLOUD)].count(True) == 0:
        arg_parser.error("{} option must be used in conjunction with {}/{} option(s)".format(
            ARGSTR_NOFILTER_COREG, ARGSTR_NOWATER, ARGSTR_NOCLOUD
        ))
    if args.get(ARGSTR_NOFILTER_COREG) and args.get(ARGSTR_META_TRANS_DIR) is not None:
        arg_parser.error("{} option cannot be used in conjunction with {} argument".format(
            ARGSTR_NOFILTER_COREG, ARGSTR_META_TRANS_DIR
        ))

    if (    args.get(ARGSTR_SAVE_COREG_STEP) != ARGCHO_SAVE_COREG_STEP_OFF
        and (   (not (args.get(ARGSTR_NOWATER) or args.get(ARGSTR_NOCLOUD)))
             or (args.get(ARGSTR_META_TRANS_DIR) is not None or args.get(ARGSTR_NOFILTER_COREG)))):
        arg_parser.error("Non-'{}' {} option must be used in conjunction with ({}/{}) arguments "
                         "and cannot be used in conjunction with ({}/{}) arguments".format(
            ARGCHO_SAVE_COREG_STEP_OFF, ARGSTR_SAVE_COREG_STEP,
            ARGSTR_NOWATER, ARGSTR_NOCLOUD,
            ARGSTR_META_TRANS_DIR, ARGSTR_NOFILTER_COREG
        ))

    if args.get(ARGSTR_RMSE_CUTOFF) <= 0:
        arg_parser.error("argument {} must be greater than zero".format(ARGSTR_RMSE_CUTOFF))


    ## Create output directories if they don't already exist.
    if not args.get(ARGSTR_DRYRUN):
        for dir_argstr, dir_path in list(zip(ARGGRP_OUTDIR, args.get_as_list(ARGGRP_OUTDIR))):
            if dir_path is not None and not os.path.isdir(dir_path):
                print("Creating argument {} directory: {}".format(dir_argstr, dir_path))
                os.makedirs(dir_path)


    if args.get(ARGSTR_STRIPID) is None or os.path.isfile(args.get(ARGSTR_STRIPID)) or args.get(ARGSTR_SCHEDULER) is not None:
        ## Batch processing

        # Gather strip-pair IDs to process.

        if args.get(ARGSTR_STRIPID) is None:

            # Find all scene DEMs to be merged into strips.
            src_scenedem_ffile_pattern = os.path.join(
                args.get(ARGSTR_SRC),
                '*'*(not args.get(ARGSTR_OLD_ORG)),
                '*_{}'.format(demSuffix)
            )
            src_scenedem_ffile_glob = glob.glob(src_scenedem_ffile_pattern)
            if not src_scenedem_ffile_glob:
                print("No scene DEMs found to process with pattern: '{}'".format(src_scenedem_ffile_pattern))
                print("Check --help to see if {} or {} options should be provided".format(
                    ARGSTR_OLD_ORG, ARGSTR_DEM_TYPE
                ))
                sys.exit(0)

            # Find all unique strip IDs.
            stripids = set()
            stripid_cannot_be_parsed_flag = False
            old_org = args.get(ARGSTR_OLD_ORG)
            for scenedem_ffile in src_scenedem_ffile_glob:
                sID_match = re.match(RE_STRIPID, os.path.basename(scenedem_ffile))
                if sID_match is None:
                    stripid_cannot_be_parsed_flag = True
                    print("Could not parse strip ID from the following scene DEM file name: {}".format(scenedem_ffile))
                    warnings.warn("There are source scene DEMs for which a strip ID cannot be parsed. "
                                  "Please fix source raster filenames so that a strip ID can be parsed "
                                  "using the following regular expression: '{}'".format(RE_STRIPID_STR))
                    continue
                sID = sID_match.group(1)
                scenedirname = None
                if not old_org:
                    scenedirname_test = os.path.basename(os.path.dirname(scenedem_ffile))
                    scenedirname_match = re.match(RE_STRIPID, scenedirname_test)
                    if scenedirname_match is not None:
                        scenedirname = scenedirname_test
                stripids.add((sID, scenedirname))

            if stripid_cannot_be_parsed_flag:
                print("One or more scene DEMs could not have their strip ID parsed, exiting")
                sys.exit(1)

            del src_scenedem_ffile_glob

        elif os.path.isfile(args.get(ARGSTR_STRIPID)):

            # Assume file is a list of strip-pair IDs, one per line.
            stripids = set(script_utils.read_task_bundle(args.get(ARGSTR_STRIPID)))

            # Check that source scenes exist for each strip-pair ID, else exclude and notify user.
            stripids_to_process = [
                sID for sID in stripids if glob.glob(os.path.join(
                    args.get(ARGSTR_SRC),
                    '{}_{}*'.format(sID, res_str)*(not args.get(ARGSTR_OLD_ORG)),
                    '{}_*_{}'.format(sID, demSuffix)
                ))
            ]

            stripids_missing = stripids.difference(set(stripids_to_process))
            if stripids_missing:
                print('')
                print("Missing scene data for {} of the listed strip-pair IDs:".format(len(stripids_missing)))
                for stripid in sorted(list(stripids_missing)):
                    print(stripid)
                print('')

            stripids = stripids_to_process
            demSuffix = None

        else:
            stripids = {args.get(ARGSTR_STRIPID)}

        stripids = sorted(list(stripids))

        if len(stripids) == 0:
            print("No strip-pair IDs found")
            sys.exit(0)


        ## Create processing list.
        ## Existence check. Filter out strips with existing .fin output file.
        stripids_to_process = list()
        dstdir = args.get(ARGSTR_DST)
        stripdirname_s2sidentifier = '{}{}'.format(
            res_str, '_lsf' if args.get(ARGSTR_DEM_TYPE) == ARGCHO_DEM_TYPE_LSF else ''
        )
        use_scenedirname = (len(stripids[0]) == 2)
        scenedirname = None
        stripdirname = None
        for stripid_tuple in stripids:
            if use_scenedirname:
                sID, scenedirname = stripid_tuple
            else:
                sID = stripid_tuple
            if scenedirname is not None:
                if stripdirname_s2sidentifier in scenedirname:
                    stripdirname = scenedirname
                else:
                    stripdirname = scenedirname.replace(res_str, stripdirname_s2sidentifier)
            else:
                stripdirname = '{}_{}*'.format(sID, stripdirname_s2sidentifier)
            dst_sID_ffile_glob = glob.glob(os.path.join(dstdir, stripdirname, '{}_{}*.fin'.format(sID, res_str)))
            if len(dst_sID_ffile_glob) == 0:
                stripids_to_process.append((sID, scenedirname, stripdirname))

        print("Found {}{} strip-pair IDs, {} unfinished".format(
            len(stripids), ' *'+demSuffix if demSuffix is not None else '', len(stripids_to_process)))
        if len(stripids) == 0:
            print("(Did you mean to pass `{} {}` or `{}` arguments?)".format(
                ARGSTR_DEM_TYPE, argcho_dem_type_opp, ARGSTR_OLD_ORG
            ))

        if args.get(ARGSTR_REBUILD_AUX):
            stripids_to_process = stripids
        stripids_to_process.sort()

        if len(stripids_to_process) == 0:
            print("No unfinished strip DEMs found to process, exiting")
            sys.exit(0)
        elif args.get(ARGSTR_DRYRUN) and args.get(ARGSTR_SCHEDULER) is not None:
            print("Exiting dryrun")
            sys.exit(0)


        # Pause for user review.
        if not args.get(ARGSTR_REMOVE_INCOMPLETE):
            wait_seconds = 5
            print("Sleeping {} seconds before task submission".format(wait_seconds))
            sleep(wait_seconds)


        ## Batch process each strip-pair ID.

        jobnum_fmt = script_utils.get_jobnum_fmtstr(stripids)
        last_job_email = args.get(ARGSTR_EMAIL)

        args_batch = args
        args_single = copy.deepcopy(args)
        args_single.unset(*ARGGRP_BATCH)

        gen_job_node = script_utils.loop_items(JOB_NODE) if type(JOB_NODE) is list else None

        job_num = 0
        num_jobs = len(stripids_to_process)
        for sID, scenedirname, stripdirname in stripids_to_process:
            job_num += 1

            if args.get(ARGSTR_OLD_ORG):
                strip_dname_pattern = args_batch.get(ARGSTR_DST)
            else:
                if stripdirname is None:
                    stripdirname = '{}_{}*/'.format(sID, stripdirname_s2sidentifier)
                strip_dname_pattern = os.path.join(args_batch.get(ARGSTR_DST), stripdirname)

            strip_dfull_glob = glob.glob(strip_dname_pattern)
            if len(strip_dfull_glob) > 1:
                raise InvalidArgumentError("Found more than one match for output strip folder in"
                                           " destination directory with pattern: {}".format(strip_dname_pattern))
            elif len(strip_dfull_glob) == 1:
                strip_dfull = strip_dfull_glob[0]

                # If output does not already exist, add to task list.
                stripid_fin_ffile = glob.glob(os.path.join(
                    strip_dfull,
                    '{}_{}*.fin'.format(sID, res_str)
                ))
                dst_sID_ffile_glob = glob.glob(os.path.join(
                    strip_dfull,
                    '*{}_{}{}_*'.format(
                        sID,
                        res_str,
                        '_lsf' if args.get(ARGSTR_DEM_TYPE) == ARGCHO_DEM_TYPE_LSF else ''
                    )
                ))

                if len(stripid_fin_ffile) > 0 and not args.get(ARGSTR_REBUILD_AUX):
                    print("{}, {} {} :: ({}) .fin file exists, skipping".format(
                        job_num, ARGSTR_STRIPID, sID, res_str))
                    continue
                elif len(dst_sID_ffile_glob) > 0:
                    if args.get(ARGSTR_REMOVE_INCOMPLETE) or args.get(ARGSTR_RESTART):
                        print("{}, {} {} :: {} ({}) output files exist ".format(
                            job_num, ARGSTR_STRIPID, sID, len(dst_sID_ffile_glob), res_str)
                              + "(potentially unfinished since no *.fin file), REMOVING"+" (dryrun)"*args.get(ARGSTR_DRYRUN))
                        for f in dst_sID_ffile_glob:
                            cmd = "rm {}".format(f)
                            print(cmd)
                            if not args.get(ARGSTR_DRYRUN):
                                os.remove(f)
                        if not args.get(ARGSTR_OLD_ORG):
                            if not args.get(ARGSTR_DRYRUN):
                                os.rmdir(strip_dfull)

                    elif not args.get(ARGSTR_REBUILD_AUX):
                        print("{}, {} {} :: {} ({}) output files exist ".format(
                            job_num, ARGSTR_STRIPID, sID, len(dst_sID_ffile_glob), res_str)
                              + "(potentially unfinished since no *.fin file), skipping")
                        continue

            if args.get(ARGSTR_REMOVE_INCOMPLETE):
                continue

            args_single.set(ARGSTR_STRIPID, sID)
            args_single.set(ARGSTR_SCENEDIRNAME, scenedirname)
            if last_job_email and job_num == num_jobs:
                args_single.set(ARGSTR_EMAIL, last_job_email)
            cmd_single = args_single.get_cmd()

            if args_batch.get(ARGSTR_SCHEDULER) is not None:
                job_name = JOB_ABBREV+jobnum_fmt.format(job_num)
                job_node_single = next(gen_job_node) if gen_job_node is not None else JOB_NODE

                cmd = args_single.get_jobsubmit_cmd(
                    args_batch.get(ARGSTR_SCHEDULER),
                    jobscript=args_batch.get(ARGSTR_JOBSCRIPT), jobname=job_name,
                    time_hr=JOB_WALLTIME_HR, memory_gb=JOB_MEMORY_GB,
                    node=job_node_single, ncores=JOB_NCORES,
                    email=args.get(ARGSTR_EMAIL),
                    envvars=[args_batch.get(ARGSTR_JOBSCRIPT), JOB_ABBREV, cmd_single, PYTHON_VERSION_ACCEPTED_MIN],
                )
            else:
                cmd = cmd_single

            print("{}, {}".format(job_num, cmd))
            if not args_batch.get(ARGSTR_DRYRUN):
                # For most cases, set `shell=True`.
                # For attaching process to PyCharm debugger,
                # set `shell=False`.
                subprocess.call(cmd, shell=True, cwd=args_batch.get(ARGSTR_LOGDIR))


    else:
        run_s2s(args, res_str, argcho_dem_type_opp, demSuffix)


def run_s2s(args, res_str, argcho_dem_type_opp, demSuffix):

    error_trace = None
    sceneMaskSuffix = None
    dstdir_coreg = None
    scene_dname = None
    strip_dname = None
    scene_dfull = None
    strip_dfull = None
    strip_dfull_coreg = None

    cleanup_on_failure_backup = args.get(ARGSTR_CLEANUP_ON_FAILURE)
    args.set(ARGSTR_CLEANUP_ON_FAILURE, ARGCHO_CLEANUP_ON_FAILURE_NONE)

    try:
        import numpy as np  # necessary check for later requirement
        from lib.filter_scene import generateMasks
        from lib.filter_scene import MASK_FLAT, MASK_SEPARATE, MASK_BIT
        from lib.filter_scene import DEBUG_NONE, DEBUG_ALL, DEBUG_MASKS, DEBUG_ITHRESH
        from lib.scenes2strips import scenes2strips
        from lib.scenes2strips import HOLD_GUESS_OFF, HOLD_GUESS_ALL, HOLD_GUESS_UPDATE_RMSE
        from batch_mask import get_mask_bitstring

        ## Process a single strip.
        print('')

        # Parse arguments in context of strip.

        stripid_is_xtrack = args.get(ARGSTR_STRIPID)[1].isdigit()
        bypass_ortho2 = False

        use_old_trans = True if args.get(ARGSTR_META_TRANS_DIR) is not None else False

        mask_name = 'mask' if args.get(ARGSTR_MASK_VER) == ARGCHO_MASK_VER_MASKV2 else args.get(ARGSTR_MASK_VER)
        if args.get(ARGSTR_REMERGE_STRIPS):
            scene_mask_name = mask_name
        else:
            scene_mask_name = demSuffix.replace('.tif', '_'+mask_name)
        strip_mask_name = mask_name

        sceneMaskSuffix = scene_mask_name+'.tif'
        stripMaskSuffix = strip_mask_name+'.tif'
        stripDemSuffix = 'dem.tif'

        filter_options_mask = ()
        if args.get(ARGSTR_NOWATER):
            filter_options_mask += ('nowater',)
        if args.get(ARGSTR_NOCLOUD):
            filter_options_mask += ('nocloud',)

        filter_options_coreg = filter_options_mask if args.get(ARGSTR_NOFILTER_COREG) else ()

        if args.get(ARGSTR_SAVE_COREG_STEP) != ARGCHO_SAVE_COREG_STEP_OFF:
            dstdir_coreg = os.path.join(
                os.path.dirname(args.get(ARGSTR_DST)),
                '{}_coreg_filt{}'.format(
                    os.path.basename(args.get(ARGSTR_DST)),
                    get_mask_bitstring(True,
                                       'nowater' not in filter_options_coreg,
                                       'nocloud' not in filter_options_coreg)))
        else:
            dstdir_coreg = None

        if args.get(ARGSTR_OLD_ORG):
            scene_dfull = args.get(ARGSTR_SRC)
            scene_dname_root = ''
            scene_dname = ''
        else:
            scene_dname_root = '{}_{}'.format(args.get(ARGSTR_STRIPID), res_str)
            if args.get(ARGSTR_SCENEDIRNAME) is not None:
                scene_dname = args.get(ARGSTR_SCENEDIRNAME)
                scene_dfull = os.path.join(args.get(ARGSTR_SRC), scene_dname)
                if not os.path.isdir(scene_dfull):
                    raise InvalidArgumentError(
                        "Source scene directory specified by '{}' and {} arguments does not exist: {}".format(
                            ARGSTR_SRC, ARGSTR_SCENEDIRNAME, scene_dfull
                        )
                    )
            else:
                scene_dfull_pattern = os.path.join(args.get(ARGSTR_SRC), scene_dname_root+'*/')
                scene_dfull_glob = glob.glob(scene_dfull_pattern)
                if len(scene_dfull_glob) != 1:
                    raise InvalidArgumentError("Cannot find only one match for input strip folder in"
                                               " source directory with pattern: {}".format(scene_dfull_pattern))
                scene_dfull = scene_dfull_glob[0]
                scene_dname = os.path.basename(os.path.normpath(scene_dfull))

        strip_dname = '{}_{}{}{}'.format(
            args.get(ARGSTR_STRIPID),
            res_str,
            ('_lsf' if args.get(ARGSTR_DEM_TYPE) == ARGCHO_DEM_TYPE_LSF else '')*(not args.get(ARGSTR_REMERGE_STRIPS)),
            scene_dname.replace(scene_dname_root, '')
        )

        if args.get(ARGSTR_OLD_ORG):
            strip_dfull = args.get(ARGSTR_DST)
            strip_dfull_coreg = dstdir_coreg
        else:
            strip_dfull = os.path.join(args.get(ARGSTR_DST), strip_dname)
            strip_dfull_coreg = os.path.join(dstdir_coreg, strip_dname) if dstdir_coreg is not None else None

        # Print arguments for this run.
        print("stripid: {}".format(args.get(ARGSTR_STRIPID)))
        print("scenedirname: {}".format(args.get(ARGSTR_SCENEDIRNAME)))
        print("res: {}".format(res_str))
        print("src dir: {}".format(args.get(ARGSTR_SRC)))
        print("dst dir: {}".format(args.get(ARGSTR_DST)))
        print("scene dir: {}".format(scene_dfull))
        print("strip dir: {}".format(strip_dfull))
        print("dst dir for coreg step: {}".format(dstdir_coreg))
        print("strip dir for coreg step: {}".format(strip_dfull_coreg))
        print("metadir: {}".format(args.get(ARGSTR_META_TRANS_DIR)))
        print("dem type: {}".format(args.get(ARGSTR_DEM_TYPE)))
        print("mask version: {}".format(args.get(ARGSTR_MASK_VER)))
        print("scene mask name: {}".format(scene_mask_name))
        print("strip mask name: {}".format(strip_mask_name))
        print("coreg filter options: {}".format(filter_options_coreg))
        print("mask filter options: {}".format(filter_options_mask))
        print("rmse cutoff: {}".format(args.get(ARGSTR_RMSE_CUTOFF)))
        print("remerge strips: {}".format(args.get(ARGSTR_REMERGE_STRIPS)))
        print("dryrun: {}".format(args.get(ARGSTR_DRYRUN)))
        print('')

        # Find scene DEMs for this stripid to be merged into strips.
        src_scenedem_ffile_glob = glob.glob(os.path.join(
            scene_dfull,
            '{}_*_{}'.format(args.get(ARGSTR_STRIPID), demSuffix)))
        print("Processing strip-pair ID: {}, {} scenes".format(args.get(ARGSTR_STRIPID), len(src_scenedem_ffile_glob)))
        if not src_scenedem_ffile_glob:
            print("No scene DEMs found to process, skipping")
            sys.exit(0)
        src_scenedem_ffile_glob.sort()
        print('')

        # Verify source strip finfile exists if doing strip remerge
        stripid_fin_ffile_src = None
        if args.get(ARGSTR_REMERGE_STRIPS):
            stripid_fin_ffile_src = os.path.join(scene_dfull, strip_dname+'.fin')
            if not os.path.isfile(stripid_fin_ffile_src):
                print("Source strip directory for remerge does not contain expected finfile: {}".format(stripid_fin_ffile_src))
                sys.exit(1)


        # Derive stripdemid from SETSM versions in scene metadata.
        setsm_version_list = list()
        group_version_list = list()

        for src_demFile in src_scenedem_ffile_glob:
            src_metaFile = src_demFile.replace(demSuffix, 'meta.txt')
            if not os.path.isfile(src_metaFile):
                raise FileNotFoundError("{} not found".format(src_metaFile))
            with open(src_metaFile, 'r') as src_metaFile_fp:
                src_metaFile_text = src_metaFile_fp.read()
            setsm_version_list.extend(re.findall(RE_SCENEMETA_SETSM_VERSION, src_metaFile_text))
            group_version_list.extend(re.findall(RE_SCENEMETA_GROUP_VERSION, src_metaFile_text))

        if len(setsm_version_list) == 0:
            raise MetaReadError("No matches for regex '{}' among all scene DEM meta.txt files".format(RE_SCENEMETA_SETSM_VERSION_STR))

        setsm_version_list = list(set([item.split('=')[1].strip() for item in setsm_version_list]))
        group_version_list = list(set([item.split('=')[1].strip() for item in group_version_list]))

        setsm_version_list_fixed = list()
        for ver in setsm_version_list:
            if ver.count('.') == 1:
                ver = "{}.0.0".format(ver.split('.')[0])
            elif ver.count('.') != 2:
                raise MetaReadError("Unexpected SETSM version format: '{}'".format(ver))
            setsm_version_list_fixed.append(ver)

        setsm_version_list_fixed.sort(key=StrictVersion)
        group_version_list.sort(key=StrictVersion)

        derived_group_version = setsm_version_list_fixed[0]
        parsed_group_version = None
        if len(group_version_list) > 0:
            if len(group_version_list) == 1:
                parsed_group_version = group_version_list[0]
            else:
                raise MetaReadError("Found more than one 'Group_version' among source "
                                    "scene DEM meta.txt files: {}".format(group_version_list))

        if parsed_group_version is not None and parsed_group_version != derived_group_version:
            raise MetaReadError(
                "Parsed 'Group_version' ({}) from source scene DEM meta.txt files "
                "does not match group version ({}) derived from 'SETSM Version' meta entries: {}".format(
                    parsed_group_version, derived_group_version, setsm_version_list
                )
            )
        if parsed_group_version is not None:
            derived_group_version = parsed_group_version

        setsm_verkey = "v{:02}{:02}{:02}".format(*[int(n) for n in derived_group_version.split('.')])
        stripdemid = "{}_{}_{}".format(
            args.get(ARGSTR_STRIPID), res_str, setsm_verkey
        )

        print("SETSM 'group version' derived from scene DEM meta.txt files: {}".format(derived_group_version))
        print("Strip DEM ID: {}".format(stripdemid))


        # TODO: Reconsider naming finfile using Strip DEM ID
        stripid_fin_fname = strip_dname+'.fin'
        stripid_remergeinfo_fname = strip_dname+'_remerge.info'
        stripid_fin_ffile = os.path.join(strip_dfull, stripid_fin_fname)
        stripid_remergeinfo_ffile = os.path.join(strip_dfull, stripid_remergeinfo_fname)
        if dstdir_coreg is not None:
            stripid_fin_ffile_coreg = os.path.join(dstdir_coreg, stripid_fin_fname)
            stripid_remergeinfo_ffile_coreg = os.path.join(dstdir_coreg, stripid_remergeinfo_fname)
        else:
            stripid_fin_ffile_coreg = None
            stripid_remergeinfo_ffile_coreg = None

        # Strip output existence check.
        if os.path.isfile(stripid_fin_ffile) and not args.get(ARGSTR_REBUILD_AUX):
            print("{} .fin file exists, strip output finished, skipping".format(stripid_fin_ffile))
            sys.exit(0)
        dstdir_stripFiles = glob.glob(os.path.join(strip_dfull, '*'+args.get(ARGSTR_STRIPID)+'*'))
        if len(dstdir_stripFiles) > 0:
            if args.get(ARGSTR_REMOVE_INCOMPLETE) or args.get(ARGSTR_RESTART):
                print("Strip output exists (potentially unfinished), REMOVING"+" (dryrun)"*args.get(ARGSTR_DRYRUN))
                for f in dstdir_stripFiles:
                    cmd = "rm {}".format(f)
                    print(cmd)
                    if not args.get(ARGSTR_DRYRUN):
                        os.remove(f)
                if not args.get(ARGSTR_OLD_ORG):
                    if not args.get(ARGSTR_DRYRUN):
                        os.rmdir(strip_dfull)

                if not args.get(ARGSTR_RESTART):
                    sys.exit(0)
            elif not args.get(ARGSTR_REBUILD_AUX):
                print("Strip output exists (potentially unfinished), skipping")
                sys.exit(0)

        # Make sure all DEM component files exist. If missing, skip.
        src_scenefile_missing_flag = False
        for scenedem_ffile in src_scenedem_ffile_glob:
            if selectBestMatchtag(scenedem_ffile) is None:
                print("matchtag file for {} missing, skipping".format(scenedem_ffile))
                src_scenefile_missing_flag = True
            if selectBestOrtho(scenedem_ffile) is None:
                print("ortho file for {} missing, skipping".format(scenedem_ffile))
                src_scenefile_missing_flag = True
            if stripid_is_xtrack and selectBestOrtho2(scenedem_ffile) is None:
                print("ortho2 file for {} missing (stripid is xtrack), {}".format(
                    scenedem_ffile, "bypassing for whole strip" if args.get(ARGSTR_SKIP_ORTHO2_ERROR) else "skipping"))
                if args.get(ARGSTR_SKIP_ORTHO2_ERROR):
                    bypass_ortho2 = True
                else:
                    src_scenefile_missing_flag = True
                    print("If you want to bypass this error and filter the strip with only one of the two orthos,"
                          " pass the {} argument".format(ARGSTR_SKIP_ORTHO2_ERROR))
            if not os.path.isfile(scenedem_ffile.replace(demSuffix, 'meta.txt')):
                print("meta file for {} missing, skipping".format(scenedem_ffile))
                src_scenefile_missing_flag = True

        # If working on LSF DEMs, make sure that all DEM scenes for the strip exist in LSF.
        demSuffix_opp = DEM_TYPE_SUFFIX_DICT[argcho_dem_type_opp]
        src_scenedem_opp_ffile_glob = glob.glob(os.path.join(
            scene_dfull,
            '{}_*_{}'.format(args.get(ARGSTR_STRIPID), demSuffix_opp)
        ))
        missing_dem_type_ffile = list(
            set([f.replace(demSuffix_opp, demSuffix) for f in src_scenedem_opp_ffile_glob]).difference(
                set(src_scenedem_ffile_glob)))
        if len(missing_dem_type_ffile) > 0:
            print("{} DEM version of the following {} DEM scenes does not exist:".format(
                args.get(ARGSTR_DEM_TYPE), argcho_dem_type_opp))
            missing_dem_type_ffile.sort()
            for f in missing_dem_type_ffile:
                print(f.replace(demSuffix, demSuffix_opp))
            print("Skipping this strip")
            src_scenefile_missing_flag = True

        if src_scenefile_missing_flag:
            raise FileNotFoundError("Source scene file(s) missing")

        # Clean up old strip results in the coreg folder, if they exist.
        if strip_dfull_coreg is not None and os.path.isdir(strip_dfull_coreg):
            dstdir_coreg_stripFiles = glob.glob(os.path.join(strip_dfull_coreg, args.get(ARGSTR_STRIPID)+'*'))
            if len(dstdir_coreg_stripFiles) > 0:
                print("Deleting old strip output in dstdir for coreg step"+" (dryrun)"*args.get(ARGSTR_DRYRUN))
                for f in dstdir_coreg_stripFiles:
                    cmd = "rm {}".format(f)
                    print(cmd)
                    if not args.get(ARGSTR_DRYRUN):
                        os.remove(f)
            if not args.get(ARGSTR_OLD_ORG):
                if not args.get(ARGSTR_DRYRUN):
                    os.rmdir(strip_dfull_coreg)


        print('')

        # Filter all scenes in this strip.
        if args.get(ARGSTR_REMERGE_STRIPS):
            scenedems_to_filter = []
        elif args.get(ARGSTR_USE_OLD_MASKS):
            scenedems_to_filter = [f for f in src_scenedem_ffile_glob if shouldDoMasking(selectBestMatchtag(f), scene_mask_name)]
        else:
            src_scenemask_ffile_glob = glob.glob(os.path.join(
                scene_dfull,
                '{}_*_{}'.format(args.get(ARGSTR_STRIPID), sceneMaskSuffix)))
            if len(src_scenemask_ffile_glob) > 0:
                print( "Deleting {} existing *_{}.tif scene masks".format(len(src_scenemask_ffile_glob), scene_mask_name)
                      +" (dryrun)"*args.get(ARGSTR_DRYRUN))
                for f in src_scenemask_ffile_glob:
                    cmd = "rm {}".format(f)
                    print(cmd)
                    if not args.get(ARGSTR_DRYRUN):
                        os.remove(f)
            scenedems_to_filter = src_scenedem_ffile_glob.copy()

        filter_total = len(scenedems_to_filter)
        if filter_total == 0:
            if cleanup_on_failure_backup == ARGCHO_CLEANUP_ON_FAILURE_MASKS:
                cleanup_on_failure_backup = ARGCHO_CLEANUP_ON_FAILURE_NONE
            elif cleanup_on_failure_backup == ARGCHO_CLEANUP_ON_FAILURE_OUTPUT:
                cleanup_on_failure_backup = ARGCHO_CLEANUP_ON_FAILURE_STRIP
        args.set(ARGSTR_CLEANUP_ON_FAILURE, cleanup_on_failure_backup)

        i = 0
        for scenedem_ffile in scenedems_to_filter:
            i += 1
            print("Filtering {} of {}: {}".format(i, filter_total, scenedem_ffile))
            if not args.get(ARGSTR_DRYRUN):
                generateMasks(scenedem_ffile, scene_mask_name, noentropy=args.get(ARGSTR_NOENTROPY),
                              save_component_masks=MASK_BIT, use_second_ortho=(stripid_is_xtrack and not bypass_ortho2),
                              debug_component_masks=DEBUG_NONE, nbit_masks=False,
                              use_pil_imresize=args.get(ARGSTR_USE_PIL_IMRESIZE))

        if not args.get(ARGSTR_DRYRUN):
            print('')
            print("All *_{}.tif scene masks have been created in source scene directory".format(scene_mask_name))

        if args.get(ARGSTR_SCENE_MASKS_ONLY):
            sys.exit(0)

        print('')
        print("Running scenes2strips")
        print('')

        if args.get(ARGSTR_DRYRUN):
            print("Exiting dryrun")
            sys.exit(0)

        ## Mosaic scenes in this strip together.
        # Output separate segments if there are breaks in overlap.
        scenedem_fname_remaining = [f for f in src_scenedem_ffile_glob]
        segnum = 1
        try:
            while len(scenedem_fname_remaining) > 0:

                print("Building segment {}".format(segnum))

                coreg_step_attempted = False
                all_data_masked = False

                # Determine output strip segment DEM file paths.
                stripdem_fname = "SETSM_s2s{:0>3}_{}_{}{}_seg{}_{}".format(
                    str(SCRIPT_VERSION_NUM).replace('.', ''),
                    args.get(ARGSTR_STRIPID),
                    res_str,
                    '_lsf' if args.get(ARGSTR_DEM_TYPE) == ARGCHO_DEM_TYPE_LSF else '',
                    segnum,
                    stripDemSuffix
                )
                stripdem_ffile = os.path.join(strip_dfull, stripdem_fname)
                if not args.get(ARGSTR_OLD_ORG):
                    if not os.path.isdir(strip_dfull):
                        os.makedirs(strip_dfull)
                if dstdir_coreg is not None:
                    stripdem_ffile_coreg = os.path.join(strip_dfull_coreg, stripdem_fname)
                    if not args.get(ARGSTR_OLD_ORG):
                        if not os.path.isdir(strip_dfull_coreg):
                            os.makedirs(strip_dfull_coreg)

                if args.get(ARGSTR_REBUILD_AUX):
                    if os.path.isfile(stripdem_ffile):
                        saveStripBrowse(args, stripdem_ffile, stripDemSuffix, stripMaskSuffix)
                        segnum += 1
                        continue
                    else:
                        break


                ## COREGISTER/mosaic scenes in this strip segment.
                if use_old_trans:
                    # Attempt to parse RMSE and translation vector from
                    # strip metadata text file from prior strip creation.
                    stripmeta_ffile_old = os.path.join(args.get(ARGSTR_META_TRANS_DIR),
                                                       stripdem_fname.replace(stripDemSuffix, 'meta.txt'))
                    scenedem_fname_coregistered, rmse, trans, trans_err = readStripMeta_stats(stripmeta_ffile_old)
                    if not set(scenedem_fname_coregistered).issubset(set(scenedem_fname_remaining)):
                        print("Current source DEMs do not include source DEMs referenced in old strip meta file")
                        use_old_trans = False
                else:
                    print("Running s2s with coregistration filter options: {}".format(
                        ', '.join(filter_options_coreg) if filter_options_coreg else None))
                    X, Y, Z, M, O, O2, MD, trans, trans_err, rmse, scenedem_fname_coregistered, spat_ref = scenes2strips(
                        scenedem_fname_remaining,
                        sceneMaskSuffix, filter_options_coreg, args.get(ARGSTR_RMSE_CUTOFF),
                        use_second_ortho=(stripid_is_xtrack and not bypass_ortho2),
                        remerge_strips=args.get(ARGSTR_REMERGE_STRIPS)
                    )
                    if X is None:
                        all_data_masked = True
                    coreg_step_attempted = True


                if (   (filter_options_mask == filter_options_coreg and not use_old_trans)
                    or all_data_masked):
                    # No need to run second pass of scenes2strips.
                    scenedem_fname_mosaicked = scenedem_fname_coregistered

                else:
                    ## Apply translation vector values to MOSAIC scenes in this strip segment.

                    print("Running s2s with masking filter options: {}".format(
                        ', '.join(filter_options_mask) if filter_options_mask else None))

                    if coreg_step_attempted:
                        if dstdir_coreg is not None:
                            if not os.path.isdir(dstdir_coreg):
                                print("Creating dstdir for coreg step directory: {}".format(dstdir_coreg))
                                os.makedirs(dstdir_coreg)
                            print("Saving output from coregistration step")
                            if args.get(ARGSTR_SAVE_COREG_STEP) in (ARGCHO_SAVE_COREG_STEP_META, ARGCHO_SAVE_COREG_STEP_ALL):
                                saveStripMeta(stripdem_ffile_coreg, stripid_remergeinfo_ffile_coreg,
                                              stripDemSuffix, stripdemid,
                                              X, Y, Z, M, MD, trans, trans_err, rmse, spat_ref,
                                              scene_dfull, scenedem_fname_coregistered, args,
                                              filter_options_applied=filter_options_coreg)
                            if args.get(ARGSTR_SAVE_COREG_STEP) == ARGCHO_SAVE_COREG_STEP_ALL:
                                saveStripRasters(stripdem_ffile_coreg, stripDemSuffix, stripMaskSuffix,
                                                 X, Y, Z, M, O, O2, MD, spat_ref)
                                saveStripBrowse(args, stripdem_ffile_coreg, stripDemSuffix, stripMaskSuffix)
                        del X, Y, Z, M, O, O2, MD
                        gc.collect()

                    X, Y, Z, M, O, O2, MD, trans, trans_err, rmse, scenedem_fname_mosaicked, spat_ref = scenes2strips(
                        scenedem_fname_coregistered.copy(),
                        sceneMaskSuffix, filter_options_mask, args.get(ARGSTR_RMSE_CUTOFF),
                        trans_guess=trans, trans_err_guess=trans_err, rmse_guess=rmse,
                        hold_guess=HOLD_GUESS_ALL, check_guess=use_old_trans,
                        use_second_ortho=(stripid_is_xtrack and not bypass_ortho2),
                        remerge_strips=args.get(ARGSTR_REMERGE_STRIPS)
                    )
                    if X is None:
                        all_data_masked = True
                    if use_old_trans and scenedem_fname_mosaicked != scenedem_fname_coregistered:
                        print("Current strip segmentation does not match that found in old strip meta file")
                        print("Rerunning s2s to get new coregistration translation values")
                        use_old_trans = False
                        continue


                scenedem_fname_remaining = list(set(scenedem_fname_remaining).difference(set(scenedem_fname_mosaicked)))
                if all_data_masked:
                    if args.get(ARGSTR_REMERGE_STRIPS):
                        remergeinfo_ffile_list = [stripid_remergeinfo_ffile]
                        if stripid_remergeinfo_ffile_coreg is not None:
                            remergeinfo_ffile_list.append(stripid_remergeinfo_ffile_coreg)
                        newseg_remerge_info = ""
                        for stripseg_fname in scenedem_fname_mosaicked:
                            input_stripseg_fname = os.path.basename(stripseg_fname)
                            input_segnum = int(re.search(RE_STRIPFNAME_SEGNUM, input_stripseg_fname).groups()[0])
                            newseg_remerge_info += "seg{}>none (masked out)\n".format(input_segnum)
                        for remergeinfo_ffile in remergeinfo_ffile_list:
                            with open(remergeinfo_ffile, 'a') as remergeinfo_fp:
                                remergeinfo_fp.write(newseg_remerge_info)
                    continue


                print("Writing output strip segment with DEM: {}".format(stripdem_ffile))

                saveStripMeta(stripdem_ffile, stripid_remergeinfo_ffile,
                              stripDemSuffix, stripdemid,
                              X, Y, Z, M, MD, trans, trans_err, rmse, spat_ref,
                              scene_dfull, scenedem_fname_mosaicked, args,
                              filter_options_applied=filter_options_mask)
                saveStripRasters(stripdem_ffile, stripDemSuffix, stripMaskSuffix,
                                 X, Y, Z, M, O, O2, MD, spat_ref)
                saveStripBrowse(args, stripdem_ffile, stripDemSuffix, stripMaskSuffix)
                del X, Y, Z, M, O, O2, MD

                segnum += 1

        except:
            if (    scene_dfull is not None and sceneMaskSuffix is not None
                and (    args.get(ARGSTR_CLEANUP_ON_FAILURE) == ARGCHO_CLEANUP_ON_FAILURE_MASKS
                     or (args.get(ARGSTR_CLEANUP_ON_FAILURE) == ARGCHO_CLEANUP_ON_FAILURE_OUTPUT and not args.get(ARGSTR_USE_OLD_MASKS)))):
                src_mask_ffile_glob = sorted(glob.glob(os.path.join(scene_dfull, args.get(ARGSTR_STRIPID))+'*'+sceneMaskSuffix))
                if len(src_mask_ffile_glob) > 0:
                    print("Detected error; deleting output scene masks"+" (dryrun)"*args.get(ARGSTR_DRYRUN))
                    for f in src_mask_ffile_glob:
                        cmd = "rm {}".format(f)
                        print(cmd)
                        if not args.get(ARGSTR_DRYRUN):
                            os.remove(f)
            if strip_dfull is not None and args.get(ARGSTR_CLEANUP_ON_FAILURE) in (ARGCHO_CLEANUP_ON_FAILURE_STRIP, ARGCHO_CLEANUP_ON_FAILURE_OUTPUT):
                print("Detected error; deleting incomplete strip output"+" (dryrun)"*args.get(ARGSTR_DRYRUN))
                for strip_dir in [strip_dfull, strip_dfull_coreg]:
                    if strip_dir is None or not os.path.isdir(strip_dir):
                        continue
                    dst_strip_ffile_glob = sorted(glob.glob(os.path.join(strip_dir, args.get(ARGSTR_STRIPID))+'*'))
                    for f in dst_strip_ffile_glob:
                        if strip_dir == strip_dfull_coreg and f.endswith('meta.txt'):
                            continue
                        cmd = "rm {}".format(f)
                        print(cmd)
                        if not args.get(ARGSTR_DRYRUN):
                            os.remove(f)
                    if not args.get(ARGSTR_OLD_ORG):
                        if not args.get(ARGSTR_DRYRUN):
                            print("Removing strip output directory: {}".format(strip_dir))
                            os.rmdir(strip_dir)
            raise

        print('')
        print("Completed processing for this strip-pair ID")

        if not args.get(ARGSTR_REBUILD_AUX):
            if args.get(ARGSTR_REMERGE_STRIPS):
                shutil.copyfile(stripid_fin_ffile_src, stripid_fin_ffile)
                if args.get(ARGSTR_SAVE_COREG_STEP) == ARGCHO_SAVE_COREG_STEP_ALL and os.path.isdir(dstdir_coreg):
                    shutil.copyfile(stripid_fin_ffile_src, stripid_fin_ffile_coreg)
            else:
                with open(stripid_fin_ffile, 'w') as stripid_fin_fp:
                    for scenedem_ffile in src_scenedem_ffile_glob:
                        stripid_fin_fp.write(os.path.basename(scenedem_ffile)+'\n')

                if args.get(ARGSTR_SAVE_COREG_STEP) == ARGCHO_SAVE_COREG_STEP_ALL and os.path.isdir(dstdir_coreg):
                    with open(stripid_fin_ffile_coreg, 'w') as stripid_fin_coreg_fp:
                        for scenedem_ffile in src_scenedem_ffile_glob:
                            stripid_fin_coreg_fp.write(os.path.basename(scenedem_ffile)+'\n')

            print(".fin finished indicator file created: {}".format(stripid_fin_ffile))
        print('')


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


def saveStripMeta(strip_demFile, strip_remergeInfoFile,
                  demSuffix, stripdemid,
                  X, Y, Z, M, MD, trans, trans_err, rmse, spat_ref,
                  scene_dir, scene_demFiles, args,
                  filter_options_applied):
    import numpy as np
    from osgeo.ogr import CreateGeometryFromWkt
    from lib.raster_array_tools import getFPvertices, coordsToWkt
    from lib.filter_scene import MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT

    strip_metaFile = strip_demFile.replace(demSuffix, 'meta.txt')
    scene_demFnames = [os.path.basename(f) for f in scene_demFiles]

    bitmask_compbit_pixelcount_dict = None
    bitmask_nonedge_pixelcount = None
    bitmask_nonedge_masked_pixelcount = None
    if args.get(ARGSTR_MASK_VER) == ARGCHO_MASK_VER_BITMASK:
        bitmask_compbit_pixelcount_dict = dict()
        mask_select = np.empty_like(MD, dtype=np.uint8)
        for maskcomp_bit in [MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT]:
            # We NEED to conserve memory at this point (strip rasters are big).
            # Pixels that are both water/cloud AND edge are outside the AOI.
            # The following two lines select all non-edgemask pixels,
            # which always have an EVEN base 10 value in the bitmask.
            mask_select = np.mod(MD, 2, out=mask_select)
            mask_select = np.logical_not(mask_select, out=mask_select)
            if bitmask_nonedge_masked_pixelcount is None:
                bitmask_nonedge_pixelcount = np.count_nonzero(mask_select)
                bitmask_nomask_pixelcount = MD.size - np.count_nonzero(MD)
                bitmask_nonedge_masked_pixelcount = bitmask_nonedge_pixelcount - bitmask_nomask_pixelcount
            np.left_shift(mask_select, maskcomp_bit, out=mask_select)
            np.bitwise_and(MD, mask_select, out=mask_select)
            maskcomp_pixelcount = np.count_nonzero(mask_select)
            bitmask_compbit_pixelcount_dict[maskcomp_bit] = maskcomp_pixelcount
        del mask_select

    fp_vertices = getFPvertices(Z, X, Y, label=-9999, label_type='nodata',
                                replicate_matlab=True, dtype_out_int64_if_equal=True)

    strip_geom = CreateGeometryFromWkt(coordsToWkt(fp_vertices.T))
    strip_area = strip_geom.Area()
    data_pixel_count = np.count_nonzero(M)
    pixel_area = abs(X[1] - X[0]) * abs(Y[1] - Y[0])
    data_density = data_pixel_count * pixel_area / strip_area

    dem_valid_pixel_count = np.count_nonzero(Z != -9999)
    if bitmask_nonedge_pixelcount is not None:
        # More accurate
        dem_coverage = dem_valid_pixel_count / bitmask_nonedge_pixelcount
    else:
        # Less accurate because strip footprint geometry is somewhat simplified
        dem_coverage = dem_valid_pixel_count * pixel_area / strip_area

    data_density_info = OrderedDict([
        ("Output DEM Coverage", dem_coverage),
        ("Output Data Density", data_density),
        ("Unmasked Data Density", 'NA'),
        ("Fully Masked Data Density", 'NA'),
        ("Water Mask Coverage", 'NA'),
        ("Cloud Mask Coverage", 'NA'),
        ("Combined Mask Coverage", 'NA')
    ])
    if bitmask_compbit_pixelcount_dict is not None:
        if 'nowater' in filter_options_applied and 'nocloud' in filter_options_applied:
            data_density_info["Unmasked Data Density"] = data_density
        if len(filter_options_applied) == 0:
            data_density_info["Fully Masked Data Density"] = data_density
        else:
            masked_data_array = (MD == 0)
            np.logical_and(M, masked_data_array, out=masked_data_array)
            masked_data_pixel_count = np.count_nonzero(masked_data_array)
            masked_data_area = masked_data_pixel_count * pixel_area
            masked_data_density = masked_data_area / strip_area
            data_density_info["Fully Masked Data Density"] = masked_data_density
            del masked_data_array
        data_density_info["Combined Mask Coverage"] = (
            bitmask_nonedge_masked_pixelcount * pixel_area / strip_area
        )
        data_density_info["Water Mask Coverage"] = (
            bitmask_compbit_pixelcount_dict[MASKCOMP_WATER_BIT] * pixel_area / strip_area
        )
        data_density_info["Cloud Mask Coverage"] = (
            bitmask_compbit_pixelcount_dict[MASKCOMP_CLOUD_BIT] * pixel_area / strip_area
        )

    dem_nodata = (Z == -9999)
    Z[dem_nodata] = np.nan
    elevation_stats = OrderedDict([
        ("Minimum elevation value", np.nanmin(Z)),
        ("Maximum elevation value", np.nanmax(Z))
    ])
    Z[dem_nodata] = -9999
    del dem_nodata

    proj4 = spat_ref.ExportToProj4()
    time = datetime.today().strftime("%d-%b-%Y %H:%M:%S")

    writeStripMeta(strip_metaFile, strip_remergeInfoFile,
                   scene_dir, scene_demFnames, stripdemid,
                   trans, trans_err, rmse,
                   proj4, fp_vertices,
                   data_density_info, elevation_stats,
                   time, args)


def saveStripRasters(strip_demFile, demSuffix, maskSuffix,
                     X, Y, Z, M, O, O2, MD, spat_ref):
    from lib.raster_array_tools import saveArrayAsTiff

    strip_matchFile  = strip_demFile.replace(demSuffix, 'matchtag.tif')
    strip_orthoFile  = strip_demFile.replace(demSuffix, 'ortho.tif')
    strip_ortho2File = strip_demFile.replace(demSuffix, 'ortho2.tif')
    strip_maskFile   = strip_demFile.replace(demSuffix, maskSuffix)

    saveArrayAsTiff(Z, strip_demFile,   X, Y, spat_ref, nodata_val=-9999,   dtype_out='float32')
    del Z
    saveArrayAsTiff(M, strip_matchFile, X, Y, spat_ref, nodata_val=0,       dtype_out='uint8')
    del M
    saveArrayAsTiff(O, strip_orthoFile, X, Y, spat_ref, nodata_val=0,       dtype_out='int16')
    del O
    if O2 is not None:
        saveArrayAsTiff(O2, strip_ortho2File, X, Y, spat_ref, nodata_val=0, dtype_out='int16')
        del O2
    saveArrayAsTiff(MD, strip_maskFile, X, Y, spat_ref, nodata_val=None,    dtype_out='uint8')
    del MD


def saveStripBrowse(args, demFile, demSuffix, maskSuffix):

    maskFile           = demFile.replace(demSuffix, maskSuffix)
    orthoFile          = demFile.replace(demSuffix, 'ortho.tif')
    ortho2File         = demFile.replace(demSuffix, 'ortho2.tif')
    matchFile          = demFile.replace(demSuffix, 'matchtag.tif')
    dem_browse         = demFile.replace(demSuffix, 'dem_browse.tif')
    maskFile_10m       = demFile.replace(demSuffix, 'bitmask_10m.tif')
    orthoFile_10m      = demFile.replace(demSuffix, 'ortho_10m.tif')
    ortho2File_10m     = demFile.replace(demSuffix, 'ortho2_10m.tif')
    matchFile_10m      = demFile.replace(demSuffix, 'matchtag_10m.tif')
    demFile_10m        = demFile.replace(demSuffix, 'dem_10m.tif')
    demFile_10m_temp   = demFile.replace(demSuffix, 'dem_10m_temp.tif')
    demFile_10m_masked = demFile.replace(demSuffix, 'dem_10m_masked.tif')
    demFile_10m_shade  = demFile.replace(demSuffix, 'dem_10m_shade.tif')
    demFile_shade_mask = demFile.replace(demSuffix, 'dem_10m_shade_masked.tif')
    demFile_40m_masked = demFile.replace(demSuffix, 'dem_40m_masked.tif')
    demFile_coverage   = demFile.replace(demSuffix, 'dem_40m_coverage.tif')

    output_files = set()
    keep_files = set()

    if not args.get(ARGSTR_NO_BROWSE):
        output_files.update([
            maskFile_10m,
            demFile_10m,
            demFile_10m_temp,
            demFile_10m_shade,
            demFile_shade_mask
        ])
        keep_files.update([
            demFile_10m,
            demFile_10m_shade,
            demFile_shade_mask
        ])

    if args.get(ARGSTR_BUILD_AUX):
        output_files.update([
            maskFile_10m,
            orthoFile_10m,
            matchFile_10m,
            demFile_10m,
            demFile_10m_temp,
            demFile_10m_shade,
            demFile_10m_masked,
            demFile_shade_mask,
            demFile_40m_masked,
            demFile_coverage
        ])
        keep_files.update([
            maskFile_10m,
            orthoFile_10m,
            matchFile_10m,
            demFile_10m,
            demFile_10m_shade,
            demFile_10m_masked,
            demFile_shade_mask,
            demFile_40m_masked,
            demFile_coverage
        ])
        if os.path.isfile(ortho2File):
            output_files.add(ortho2File_10m)
            keep_files.add(ortho2File_10m)

    for ofile in output_files:
        if os.path.isfile(ofile):
            os.remove(ofile)

    commands = []
    if maskFile_10m in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -r near '
             ' -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW'.format(maskFile, maskFile_10m, 10))
        )
    if orthoFile_10m in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -r cubic -dstnodata 0'
             ' -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW -co PREDICTOR=1'.format(orthoFile, orthoFile_10m, 10))
        )
    if ortho2File_10m in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -r cubic -dstnodata 0'
             ' -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW -co PREDICTOR=1'.format(ortho2File, ortho2File_10m, 10))
        )
    if matchFile_10m in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -r near -dstnodata 0'
             ' -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW'.format(matchFile, matchFile_10m, 10))
        )
    if demFile_10m in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -r bilinear -dstnodata -9999'
             ' -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW'.format(demFile, demFile_10m_temp, 10))
        )
        commands.append(
            ('gdal_calc.py --quiet --overwrite -A "{0}" --outfile="{1}" --calc="round_(A*128.0)/128.0" --NoDataValue=-9999'
             ' --co TILED=YES --co BIGTIFF=YES --co COMPRESS=LZW --co PREDICTOR=3'.format(demFile_10m_temp, demFile_10m))
        )
    if demFile_10m_shade in output_files:
        commands.append(
            ('gdaldem hillshade "{0}" "{1}" -q -z 3 -compute_edges -of GTiff'
             ' -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW'.format(demFile_10m, demFile_10m_shade))
        )
    if dem_browse in output_files:
        commands.append(
            ('gdaldem hillshade "{0}" "{1}" -q -z 3 -compute_edges -of GTiff'
             ' -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW'.format(demFile_10m, dem_browse))
        )
    if demFile_10m_masked in output_files:
        commands.append(
            ('gdal_calc.py --quiet --overwrite -A "{0}" -B "{1}" --outfile="{2}" --calc="A*(B==0)+(-9999)*(B!=0)" --NoDataValue=-9999'
             ' --co TILED=YES --co BIGTIFF=YES --co COMPRESS=LZW'.format(demFile_10m, maskFile_10m, demFile_10m_masked))
        )
    if demFile_shade_mask in output_files:
        commands.append(
            ('gdal_calc.py --quiet --overwrite -A "{0}" -B "{1}" --outfile="{2}" --calc="A*(B==0)" --NoDataValue=0'
             ' --co TILED=YES --co BIGTIFF=YES --co COMPRESS=LZW'.format(demFile_10m_shade, maskFile_10m, demFile_shade_mask))
        )
    if demFile_40m_masked in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -tap -r bilinear -dstnodata -9999'
             ' -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW'.format(demFile_10m_masked, demFile_40m_masked, 40))
        )
    if demFile_coverage in output_files:
        commands.append(
            ('gdal_calc.py --quiet --overwrite -A "{0}" --outfile="{1}" --type Byte --calc="A!=-9999" --NoDataValue=0'
             ' --co TILED=YES --co BIGTIFF=YES --co COMPRESS=LZW'.format(demFile_40m_masked, demFile_coverage))
        )

    for cmd in commands:
        print(cmd)
        script_utils.exec_cmd(cmd)

    for outfile in output_files:
        if not os.path.isfile(outfile):
            raise ExternalError("External program call did not create output file: {}".format(outfile))
        if outfile not in keep_files:
            os.remove(outfile)


def getDemSuffix(demFile):
    for demSuffix in SUFFIX_PRIORITY_DEM:
        if demFile.endswith(demSuffix):
            return demSuffix
    return None


def getMatchtagSuffix(matchFile):
    for matchSuffix in SUFFIX_PRIORITY_MATCHTAG:
        if matchFile.endswith(matchSuffix):
            return matchSuffix
    return None


def getOrthoSuffix(orthoFile):
    for orthoSuffix in SUFFIX_PRIORITY_ORTHO1:
        if orthoFile.endswith(orthoSuffix):
            return orthoSuffix
    return None

def getOrtho2Suffix(orthoFile):
    for orthoSuffix in SUFFIX_PRIORITY_ORTHO2:
        if orthoFile.endswith(orthoSuffix):
            return orthoSuffix
    return None


def selectBestMatchtag(demFile):
    demSuffix = getDemSuffix(demFile)
    for matchSuffix in SUFFIX_PRIORITY_MATCHTAG:
        matchFile = demFile.replace(demSuffix, matchSuffix)
        if os.path.isfile(matchFile):
            return matchFile
    return None


def selectBestOrtho(demFile):
    demSuffix = getDemSuffix(demFile)
    for orthoSuffix in SUFFIX_PRIORITY_ORTHO1:
        orthoFile = demFile.replace(demSuffix, orthoSuffix)
        if os.path.isfile(orthoFile):
            return orthoFile
    return None

def selectBestOrtho2(demFile):
    demSuffix = getDemSuffix(demFile)
    for orthoSuffix in SUFFIX_PRIORITY_ORTHO2:
        orthoFile = demFile.replace(demSuffix, orthoSuffix)
        if os.path.isfile(orthoFile):
            return orthoFile
    return None


def shouldDoMasking(matchFile, mask_name):
    matchFile_date = os.path.getmtime(matchFile)
    demFile_base = matchFile.replace(getMatchtagSuffix(matchFile), '')
    maskFiles = (     [demFile_base+s for s in ('edgemask.tif', 'datamask.tif')] if mask_name == ARGCHO_MASK_VER_MASKV1
                 else ['{}{}.tif'.format(demFile_base, mask_name)])
    for m in maskFiles:
        if os.path.isfile(m):
            # Update Mode - will only reprocess masks older than the matchtag file.
            maskFile_date = os.path.getmtime(m)
            if (matchFile_date - maskFile_date) > 6.9444e-04:
                return True
        else:
            return True
    return False


def writeStripMeta(o_metaFile, strip_remergeInfoFile,
                   scene_dir, scenedem_fname_list, stripdemid,
                   trans, trans_err, rmse,
                   proj4, fp_vertices,
                   data_density_info, elevation_stats,
                   strip_time, args):
    import numpy as np
    from lib.filter_scene import MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT
    from lib.filter_scene import BITMASK_VERSION_NUM

    MASK_VER_VERNUM_DICT = {
        ARGCHO_MASK_VER_MASKV1:  script_utils.VersionString('1'),
        ARGCHO_MASK_VER_MASKV2:  script_utils.VersionString('1'),
        ARGCHO_MASK_VER_REMA2A:  script_utils.VersionString('1'),
        ARGCHO_MASK_VER_MASK8M:  script_utils.VersionString('1'),
        ARGCHO_MASK_VER_BITMASK: BITMASK_VERSION_NUM
    }

    demSuffix = getDemSuffix(scenedem_fname_list[0])

    nowater, nocloud = args.get(ARGSTR_NOWATER, ARGSTR_NOCLOUD)
    nofilter_coreg = args.get(ARGSTR_NOFILTER_COREG)

    strip_info = (
"""Strip Metadata (v{})
Creation Date: {}
Strip creation date: {}
Strip projection (proj4): '{}'
Strip DEM ID: {}

Strip Footprint Vertices
X: {}
Y: {}

Mosaicking Alignment Statistics (meters, rmse-cutoff={})
scene, rmse, dz, dx, dy, dz_err, dx_err, dy_err
""".format(
    SCRIPT_VERSION_NUM,
    datetime.today().strftime("%d-%b-%Y %H:%M:%S"),
    strip_time,
    proj4,
    stripdemid,
    ' '.join(np.array_str(fp_vertices[0], max_line_width=float('inf')).strip()[1:-1].split()),
    ' '.join(np.array_str(fp_vertices[1], max_line_width=float('inf')).strip()[1:-1].split()),
    args.get(ARGSTR_RMSE_CUTOFF)
)
    )

    scene_align_format = "{} {}".format('{}', ' '.join(['{:.4f}']*7))

    for i, scenedem_fname in enumerate(scenedem_fname_list):

        scene_align_nums = np.array([
            rmse[0, i],
            trans[0, i], trans[1, i], trans[2, i],
            trans_err[0, i], trans_err[1, i], trans_err[2, i]
        ], dtype=np.float64)

        if not args.get(ARGSTR_REMERGE_STRIPS):
            scene_align_str = scene_align_format.format(scenedem_fname_list[i], *scene_align_nums)
            strip_info += scene_align_str+'\n'
        else:
            stripseg_align_nums = scene_align_nums

            if np.isnan(rmse[0, i]):
                scene_align_nums_addition = np.full(7, np.nan, dtype=np.float64)
            else:
                scene_align_nums_addition = np.array(
                    [0, trans[0, i], trans[1, i], trans[2, i], 0, 0, 0],
                    dtype=np.float64
                )

            stripseg_metaFile = os.path.join(scene_dir, scenedem_fname.replace(demSuffix, 'meta.txt'))
            in_strip_align_stats = False
            on_first_scene_in_strip_align = False
            with open(stripseg_metaFile, 'r') as stripseg_metaFile_fp:
                for line in stripseg_metaFile_fp:
                    if in_strip_align_stats:
                        line = line.strip()
                        if line == '':
                            break
                        else:
                            scene_alignment = line

                        scene_align_parts = scene_alignment.split(' ')
                        scene_align_demFname = scene_align_parts[0]

                        if on_first_scene_in_strip_align:
                            scene_align_str = scene_align_format.format(scene_align_demFname, *stripseg_align_nums)
                            on_first_scene_in_strip_align = False
                        else:
                            scene_align_nums = np.array(scene_align_parts[1:]).astype(np.float64)
                            if scene_align_nums.size == 4:
                                scene_align_nums = np.concatenate((scene_align_nums, np.full(3, np.nan)))
                            scene_align_nums += scene_align_nums_addition
                            scene_align_str = scene_align_format.format(scene_align_demFname, *scene_align_nums)

                        strip_info += scene_align_str+'\n'
                    elif line.startswith("scene, rmse, dz, dx, dy"):
                        in_strip_align_stats = True
                        on_first_scene_in_strip_align = True

    filter_info = "\nFiltering Applied: {} (v{})\n".format(
        args.get(ARGSTR_MASK_VER), MASK_VER_VERNUM_DICT[args.get(ARGSTR_MASK_VER)])

    if args.get(ARGSTR_MASK_VER) == ARGCHO_MASK_VER_BITMASK:
        filter_info += "bit, class, coreg, mosaic\n"
        filter_info_components = (
"""
{} edge 1 1
{} water {} {}
{} cloud {} {}
""".format(
        MASKCOMP_EDGE_BIT,
        MASKCOMP_WATER_BIT, int(not nofilter_coreg*nowater), int(not nowater),
        MASKCOMP_CLOUD_BIT, int(not nofilter_coreg*nocloud), int(not nocloud),
    )
        )
        filter_info += '\n'.join(sorted(filter_info_components.strip().splitlines())) + '\n'
        strip_info += filter_info

    strip_info += "\nData Coverage Statistics\n{}\n".format(
        '\n'.join(["{}: {}".format(info_key, info_val)
                   for info_key, info_val in data_density_info.items()])
    )

    strip_info += "\nElevation Statistics\n{}\n".format(
        '\n'.join(["{}: {}".format(info_key, info_val)
                   for info_key, info_val in elevation_stats.items()])
    )

    strip_info += "\nScene Metadata \n\n"

    scene_info = ""
    if not args.get(ARGSTR_REMERGE_STRIPS):
        for i, scenedem_fname in enumerate(scenedem_fname_list):
            scene_info += "scene {} name={}\n".format(i+1, scenedem_fname)

            scene_metaFile = os.path.join(scene_dir, scenedem_fname.replace(demSuffix, 'meta.txt'))
            if os.path.isfile(scene_metaFile):
                scene_metaFile_fp = open(scene_metaFile, 'r')
                scene_info += scene_metaFile_fp.read()
                scene_metaFile_fp.close()
            else:
                # scene_info += "{} not found".format(scene_metaFile)
                raise FileNotFoundError("{} not found".format(scene_metaFile))
            scene_info += " \n"
    else:
        stripmeta_curr_scene_num = 1
        for stripseg_fname in scenedem_fname_list:
            stripseg_metaFile = os.path.join(scene_dir, stripseg_fname.replace(demSuffix, 'meta.txt'))
            if os.path.isfile(stripseg_metaFile):
                in_scene_metadata = False
                skip_empty_line = False
                with open(stripseg_metaFile, 'r') as stripseg_metaFile_fp:
                    for line in stripseg_metaFile_fp:
                        if in_scene_metadata:
                            if skip_empty_line:
                                skip_empty_line = False
                                continue
                            scene_name_key_match = re.match(RE_STRIPMETA_SCENE_NAME_KEY, line)
                            if scene_name_key_match is not None:
                                scene_name_val = line.strip().split('=')[1]
                                line = "scene {} name={}\n".format(stripmeta_curr_scene_num, scene_name_val)
                                stripmeta_curr_scene_num += 1
                            scene_info += line
                        elif line.startswith("Scene Metadata"):
                            in_scene_metadata = True
                            skip_empty_line = True
            else:
                raise FileNotFoundError("{} not found".format(stripseg_metaFile))

    with open(o_metaFile, 'w') as strip_metaFile_fp:
        strip_metaFile_fp.write(strip_info)
        strip_metaFile_fp.write(scene_info)

    if args.get(ARGSTR_REMERGE_STRIPS):
        output_segnum = int(re.search(RE_STRIPFNAME_SEGNUM, os.path.basename(o_metaFile)).groups()[0])
        input_segnum_list = list()
        for i, input_stripseg_fname in enumerate(scenedem_fname_list):
            input_segnum = int(re.search(RE_STRIPFNAME_SEGNUM, input_stripseg_fname).groups()[0])
            if np.isnan(rmse[0, i]):
                input_segnum *= -1
            input_segnum_list.append(input_segnum)

        # newseg_remerge_info = "seg{}={}\n".format(
        #     output_segnum, ','.join([str(n) for n in input_segnum_list])
        # )
        # with open(strip_remergeInfoFile, 'a') as strip_remergeInfoFile_fp:
        #     strip_remergeInfoFile_fp.write(newseg_remerge_info)

        newseg_remerge_info = ""
        for input_segnum in input_segnum_list:
            newseg_remerge_info += "seg{}>seg{}{}\n".format(
                abs(input_segnum), output_segnum, ' (redundant)' if input_segnum < 0 else ''
            )
        with open(strip_remergeInfoFile, 'a') as strip_remergeInfoFile_fp:
            strip_remergeInfoFile_fp.write(newseg_remerge_info)


def readStripMeta_stats(metaFile):
    import numpy as np

    metaFile_fp = open(metaFile, 'r')
    try:
        line = metaFile_fp.readline()
        while not line.startswith('Mosaicking Alignment Statistics') and line != '':
            line = metaFile_fp.readline()
        while not line.startswith('scene, rmse, dz, dx, dy, dz_err, dx_err, dy_err') and line != '':
            line = metaFile_fp.readline()
        if line == '':
            raise MetaReadError("{}: Could not parse 'Mosaicking Alignment Statistics'".format(metaFile))

        line = metaFile_fp.readline().strip()
        line_items = line.split(' ')
        sceneDemFnames = [line_items[0]]
        rmse = [line_items[1]]
        trans = np.array([[float(s) for s in line_items[2:5]]])
        trans_err = np.array([[float(s) for s in line_items[5:8]]])

        while True:
            line = metaFile_fp.readline().strip()
            if line == '':
                break
            line_items = line.split(' ')
            sceneDemFnames.append(line_items[0])
            rmse.append(line_items[1])
            trans = np.vstack((trans, np.array([[float(s) for s in line_items[2:5]]])))
            trans_err = np.vstack((trans_err, np.array([[float(s) for s in line_items[5:8]]])))

        rmse = np.array([[float(s) for s in rmse]])
        trans = trans.T
        trans_err = trans_err.T

    finally:
        metaFile_fp.close()

    return sceneDemFnames, rmse, trans, trans_err



if __name__ == '__main__':
    main()
