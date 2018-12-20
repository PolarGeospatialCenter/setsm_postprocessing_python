
# Erik Husby, Claire Porter; Polar Geospatial Center, University of Minnesota; 2018


import argparse
import contextlib
import copy
import filecmp
import functools
import gc
import glob
import os
import platform
import re
import smtplib
import subprocess
import sys
import traceback
from email.mime.text import MIMEText
from time import sleep
from datetime import datetime
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import numpy as np

from lib import batch_handler


##############################

## Core globals

SCRIPT_VERSION_NUM = 3.1

# Script paths and execution
SCRIPT_FILE = os.path.abspath(os.path.realpath(__file__))
SCRIPT_FNAME = os.path.basename(SCRIPT_FILE)
SCRIPT_NAME, SCRIPT_EXT = os.path.splitext(SCRIPT_FNAME)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE)
SCRIPT_RUNCMD = ' '.join(sys.argv)+'\n'

##############################

## Argument globals

# Argument strings
ARGSTR_SRC = 'src'
ARGSTR_RES = 'res'
ARGSTR_DST = '--dst'
ARGSTR_META_TRANS_DIR = '--meta-trans-dir'
ARGSTR_HILLSHADE_OFF = '--hillshade-off'
ARGSTR_MASK_VER = '--mask-ver'
ARGSTR_NOENTROPY = '--noentropy'
ARGSTR_NOWATER = '--nowater'
ARGSTR_NOCLOUD = '--nocloud'
ARGSTR_NOFILTER_COREG = '--nofilter-coreg'
ARGSTR_SAVE_COREG_STEP = '--save-coreg-step'
ARGSTR_RMSE_CUTOFF = '--rmse-cutoff'
ARGSTR_SCHEDULER = '--scheduler'
ARGSTR_JOBSCRIPT = '--jobscript'
ARGSTR_LOGDIR = '--logdir'
ARGSTR_EMAIL = '--email'
ARGSTR_DRYRUN = '--dryrun'
ARGSTR_STRIPID = '--stripid'

# Argument choices
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

##############################

## Batch settings

JOBSCRIPT_DIR = os.path.join(SCRIPT_DIR, 'jobscripts')
PYTHON_EXE = 'python -u'
JOB_ABBREV = 's2s'

##############################

## Custom globals

SUFFIX_PRIORITY_DEM = ['dem_smooth.tif', 'dem.tif']
SUFFIX_PRIORITY_MATCHTAG = ['matchtag_mt.tif', 'matchtag.tif']

RE_STRIPID_STR = "(^[A-Z0-9]{4}_.*?_?[0-9A-F]{16}_.*?_?[0-9A-F]{16}).*$"
RE_STRIPID = re.compile(RE_STRIPID_STR)

##############################


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class MetaReadError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class ExternalError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


@contextlib.contextmanager
def capture_stdout_stderr():
    oldout, olderr = sys.stdout, sys.stderr
    out = [StringIO(), StringIO()]
    try:
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()


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
        raise InvalidArgumentError("argument {}: {} {}: {}".format(argstr, existtype_str, existresult_str, path))
    return abspath_fn(path) if abspath_fn is not None else path

ARGTYPE_PATH = functools.partial(functools.partial, argtype_path_handler)
ARGTYPE_BOOL_PLUS = functools.partial(functools.partial, batch_handler.argtype_bool_plus)

def argparser_init():

    parser = argparse.ArgumentParser(
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
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
        type=ARGTYPE_PATH(
            argstr=ARGSTR_SRC,
            existcheck_fn=os.path.isdir,
            existcheck_reqval=True),
        help=' '.join([
            "Path to source directory containing scene DEMs to process.",
            "If {} is not specified, this path should contain the folder 'tif_results'.".format(ARGSTR_DST),
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
        type=ARGTYPE_PATH(
            argstr=ARGSTR_DST,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=False),
        help=' '.join([
            "Path to destination directory for output mosaicked strip data.",
            "(default is src.(reverse)replace('tif_results', 'strips'))"
        ])
    )

    parser.add_argument(
        ARGSTR_META_TRANS_DIR,
        type=ARGTYPE_PATH(
            argstr=ARGSTR_META_TRANS_DIR,
            existcheck_fn=os.path.isdir,
            existcheck_reqval=True),
        help=' '.join([
            "Path to directory of old strip metadata from which translation values",
            "will be parsed to skip scene coregistration step."
        ])
    )

    parser.add_argument(
        ARGSTR_HILLSHADE_OFF,
        action='store_true',
        help=' '.join([
            "TURN OFF building of 10m hillshade *_dem_browse.tif browse images of all output",
            "DEM strip segments after they are created inside {} directory.".format(ARGSTR_DST)
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
        ARGSTR_RMSE_CUTOFF,
        type=float,
        choices=None,
        default=1.0,
        help=' '.join([
            "Maximum RMSE from coregistration step tolerated for scene merging.",
            "A value greater than this causes a new strip segment to be created."
        ])
    )

    parser.add_argument(
        ARGSTR_SCHEDULER,
        type=str,
        choices=batch_handler.SCHED_SUPPORTED,
        default=None,
        help="Submit tasks to job scheduler."
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
        ARGSTR_LOGDIR,
        type=ARGTYPE_PATH(
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
        type=ARGTYPE_BOOL_PLUS(
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

    return parser


def main():
    from lib.filter_scene import generateMasks
    from lib.filter_scene import MASK_FLAT, MASK_SEPARATE, MASK_BIT
    from lib.filter_scene import DEBUG_NONE, DEBUG_ALL, DEBUG_MASKS, DEBUG_ITHRESH
    from lib.scenes2strips import scenes2strips
    from lib.scenes2strips import HOLD_GUESS_OFF, HOLD_GUESS_ALL, HOLD_GUESS_UPDATE_RMSE
    from batch_mask import get_mask_bitstring

    # Invoke argparse argument parsing.
    arg_parser = argparser_init()
    try:
        args = batch_handler.ArgumentPasser(arg_parser, PYTHON_EXE, SCRIPT_FILE, sys.argv)
    except InvalidArgumentError as e:
        arg_parser.error(e)


    ## Further parse/adjust argument values.

    res = args.get(ARGSTR_RES)
    if int(res) == res:
        args.set(ARGSTR_RES, int(res))

    if args.get(ARGSTR_DST) is not None:
        if (   args.get(ARGSTR_SRC) == args.get(ARGSTR_DST)
            or (    os.path.isdir(args.get(ARGSTR_DST))
                and filecmp.cmp(args.get(ARGSTR_SRC), args.get(ARGSTR_DST)))):
            arg_parser.error("argument {} directory is the same as "
                             "argument {} directory".format(ARGSTR_SRC, ARGSTR_DST))
    else:
        # Set default dst dir.
        split_ind = args.get(ARGSTR_SRC).rfind('tif_results')
        if split_ind == -1:
            arg_parser.error("argument {} path does not contain 'tif_results', "
                             "so default argument {} cannot be set".format(ARGSTR_SRC, ARGSTR_DST))
        args.set(ARGSTR_DST, (  args.get(ARGSTR_SRC)[:split_ind]
                              + args.get(ARGSTR_SRC)[split_ind:].replace('tif_results', 'strips')))
        print("argument {} set automatically to: {}".format(ARGSTR_DST, args.get(ARGSTR_DST)))

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


    if args.get(ARGSTR_STRIPID) is None or os.path.isfile(args.get(ARGSTR_STRIPID)):
        ## Batch processing

        # Gather strip-pair IDs to process.

        if args.get(ARGSTR_STRIPID) is None:

            # Find all scene DEMs to be merged into strips.
            for demSuffix in SUFFIX_PRIORITY_DEM:
                scene_dems = glob.glob(
                    os.path.join(args.get(ARGSTR_SRC),
                    '*_{}_{}'.format(str(args.get(ARGSTR_RES))[0], demSuffix)))
                if scene_dems:
                    break
            if not scene_dems:
                print("No scene DEMs found to process, exiting")
                sys.exit(1)

            # Find all unique strip IDs.
            try:
                stripids = {re.match(RE_STRIPID, os.path.basename(s)).group(1) for s in scene_dems}
            except AttributeError:
                print("There are source scene DEMs for which a strip ID cannot be parsed. "
                      "Please fix source raster filenames so that a strip ID can be parsed "
                      "using the following regular expression: '{}'".format(RE_STRIPID_STR))
                raise
            del scene_dems

        else:

            # Assume file is a list of strip-pair IDs, one per line.
            stripids = set(batch_handler.read_task_bundle(args.get(ARGSTR_STRIPID)))

            # Check that source scenes exist for each strip-pair ID, else exclude and notify user.
            stripids_to_process = [
                sID for sID in stripids if glob.glob(os.path.join(
                    args.get(ARGSTR_SRC), '{}*_{}_*'.format(sID, str(args.get(ARGSTR_RES))[0])))]

            stripids_missing = stripids.difference(set(stripids_to_process))
            if stripids_missing:
                print('')
                print("Missing scene data for {} of the listed strip-pair IDs:".format(len(stripids_missing)))
                for stripid in sorted(list(stripids_missing)):
                    print(stripid)
                print('')

            stripids = stripids_to_process
            demSuffix = None

        stripids = sorted(list(stripids))


        ## Create processing list.
        ## Existence check. Filter out strips with existing .fin output file.
        dstdir, res = args.get(ARGSTR_DST, ARGSTR_RES)
        stripids_to_process = [
            sID for sID in stripids if not os.path.isfile(
                os.path.join(dstdir, '{}_{}m.fin'.format(sID, res)))]
        print("Found {}{} strip-pair IDs, {} unfinished".format(
            len(stripids), ' *'+demSuffix if demSuffix is not None else '', len(stripids_to_process)))
        if len(stripids_to_process) == 0:
            print("No unfinished strip DEMs found to process, exiting")
            sys.exit(0)
        stripids_to_process.sort()


        # Pause for user review.
        wait_seconds = 5
        print("Sleeping {} seconds before task submission".format(wait_seconds))
        sleep(wait_seconds)


        ## Batch process each strip-pair ID.

        jobnum_fmt = batch_handler.get_jobnum_fmtstr(stripids)
        last_job_email = args.get(ARGSTR_EMAIL)

        args_batch = args
        args_single = copy.deepcopy(args)
        args_single.unset_args(*ARGGRP_BATCH)

        job_num = 0
        num_jobs = len(stripids)
        for sID in stripids:
            job_num += 1

            # If output does not already exist, add to task list.
            stripid_finFile = os.path.join(
                args_batch.get(ARGSTR_DST), '{}_{}m.fin'.format(sID, args_batch.get(ARGSTR_RES)))
            dst_files = glob.glob(os.path.join(
                args_batch.get(ARGSTR_DST), '{}_seg*_{}m_{}'.format(sID, args_batch.get(ARGSTR_RES), demSuffix)))

            if os.path.isfile(stripid_finFile):
                print("{}, {} {} :: ({}m) .fin file exists, skipping".format(
                    job_num, ARGSTR_STRIPID, sID, args_batch.get(ARGSTR_RES)))
                continue
            elif dst_files:
                print("{}, {} {} :: {} ({}m) output files exist ".format(
                    job_num, ARGSTR_STRIPID, sID, len(dst_files), args_batch.get(ARGSTR_RES))
                      + "(potentially unfinished since no *.fin file), skipping")
                continue

            args_single.set(ARGSTR_STRIPID, sID)
            if last_job_email and job_num == num_jobs:
                args_single.set(ARGSTR_EMAIL, last_job_email)
            cmd_single = args_single.get_cmd()

            if args_batch.get(ARGSTR_SCHEDULER) is not None:
                job_name = JOB_ABBREV+jobnum_fmt.format(job_num)
                cmd = args_single.get_jobsubmit_cmd(args_batch.get(ARGSTR_SCHEDULER),
                                                    args_batch.get(ARGSTR_JOBSCRIPT),
                                                    job_name, cmd_single)
            else:
                cmd = cmd_single

            print("{}, {}".format(job_num, cmd))
            if not args_batch.get(ARGSTR_DRYRUN):
                # For most cases, set `shell=True`.
                # For attaching process to PyCharm debugger,
                # set `shell=False`.
                subprocess.call(cmd, shell=True, cwd=args_batch.get(ARGSTR_LOGDIR))


    else:
        error_trace = None
        try:
            ## Process a single strip.
            print('')

            # Parse arguments in context of strip.

            use_old_trans = True if args.get(ARGSTR_META_TRANS_DIR) is not None else False

            mask_name = 'mask' if args.get(ARGSTR_MASK_VER) == ARGCHO_MASK_VER_MASKV2 else args.get(ARGSTR_MASK_VER)

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
                                           not 'nowater' in filter_options_coreg,
                                           not 'nocloud' in filter_options_coreg)))
            else:
                dstdir_coreg = None

            # Print arguments for this run.
            print("stripid: {}".format(args.get(ARGSTR_STRIPID)))
            print("res: {}m".format(args.get(ARGSTR_RES)))
            print("srcdir: {}".format(args.get(ARGSTR_SRC)))
            print("dstdir: {}".format(args.get(ARGSTR_DST)))
            print("dstdir for coreg step: {}".format(dstdir_coreg))
            print("metadir: {}".format(args.get(ARGSTR_META_TRANS_DIR)))
            print("mask version: {}".format(args.get(ARGSTR_MASK_VER)))
            print("mask name: {}".format(mask_name))
            print("coreg filter options: {}".format(filter_options_coreg))
            print("mask filter options: {}".format(filter_options_mask))
            print("rmse cutoff: {}".format(args.get(ARGSTR_RMSE_CUTOFF)))
            print("dryrun: {}".format(args.get(ARGSTR_DRYRUN)))
            print('')

            # Find scene DEMs for this stripid to be merged into strips.
            for demSuffix in SUFFIX_PRIORITY_DEM:
                scene_demFiles = glob.glob(os.path.join(
                    args.get(ARGSTR_SRC), '{}*_{}_{}'.format(args.get(ARGSTR_STRIPID), str(args.get(ARGSTR_RES))[0], demSuffix)))
                if scene_demFiles:
                    break
            print("Processing strip-pair ID: {}, {} scenes".format(args.get(ARGSTR_STRIPID), len(scene_demFiles)))
            if not scene_demFiles:
                print("No scene DEMs found to process, skipping")
                sys.exit(1)
            scene_demFiles.sort()

            stripid_finFname = '{}_{}m.fin'.format(args.get(ARGSTR_STRIPID), args.get(ARGSTR_RES))
            stripid_finFile = os.path.join(args.get(ARGSTR_DST), stripid_finFname)
            if dstdir_coreg is not None:
                stripid_finFile_coreg = os.path.join(dstdir_coreg, stripid_finFname)
            else:
                stripid_finFile_coreg = None

            # Existence check. If output already exists, skip.
            if os.path.isfile(stripid_finFile):
                print("{} file exists, strip output finished, skipping".format(stripid_finFile))
                sys.exit(0)
            if glob.glob(os.path.join(args.get(ARGSTR_DST), args.get(ARGSTR_STRIPID)+'*')):
                print("strip output exists (potentially unfinished), skipping")
                sys.exit(1)

            # Make sure all DEM component files exist. If missing, skip.
            missingflag = False
            for scene_demFile in scene_demFiles:
                if selectBestMatchtag(scene_demFile) is None:
                    print("matchtag file for {} missing, skipping".format(scene_demFile))
                    missingflag = True
                if not os.path.isfile(scene_demFile.replace(demSuffix, 'ortho.tif')):
                    print("ortho file for {} missing, skipping".format(scene_demFile))
                    missingflag = True
                if not os.path.isfile(scene_demFile.replace(demSuffix, 'meta.txt')):
                    print("meta file for {} missing, skipping".format(scene_demFile))
                    missingflag = True
            if missingflag:
                sys.exit(1)

            # Clean up old strip results in the coreg folder, if they exist.
            if dstdir_coreg is not None and os.path.isdir(dstdir_coreg):
                dstdir_coreg_stripFiles = glob.glob(os.path.join(dstdir_coreg, args.get(ARGSTR_STRIPID)+'*'))
                if len(dstdir_coreg_stripFiles) > 0:
                    print("Deleting old strip output in dstdir for coreg step")
                    if not args.get(ARGSTR_DRYRUN):
                        for f in dstdir_coreg_stripFiles:
                            os.remove(f)

            print('')

            # Filter all scenes in this strip.
            filter_list = [f for f in scene_demFiles if shouldDoMasking(selectBestMatchtag(f), mask_name)]
            filter_total = len(filter_list)
            i = 0
            for demFile in filter_list:
                i += 1
                print("Filtering {} of {}: {}".format(i, filter_total, demFile))
                if not args.get(ARGSTR_DRYRUN):
                    generateMasks(demFile, mask_name, noentropy=args.get(ARGSTR_NOENTROPY),
                                  save_component_masks=MASK_BIT, debug_component_masks=DEBUG_NONE,
                                  nbit_masks=False)

            print('')
            print("All *_{}.tif scene masks have been created in source scene directory".format(mask_name))
            print('')

            print("Running scenes2strips")
            if args.get(ARGSTR_DRYRUN):
                sys.exit(0)
            print('')

            # Mosaic scenes in this strip together.
            # Output separate segments if there are breaks in overlap.
            maskSuffix = mask_name+'.tif'
            remaining_sceneDemFnames = [os.path.basename(f) for f in scene_demFiles]
            segnum = 1
            while len(remaining_sceneDemFnames) > 0:

                print("Building segment {}".format(segnum))

                strip_demFname = "{}_seg{}_{}m_{}".format(args.get(ARGSTR_STRIPID), segnum, args.get(ARGSTR_RES), demSuffix)
                strip_demFile = os.path.join(args.get(ARGSTR_DST), strip_demFname)
                if dstdir_coreg is not None:
                    strip_demFile_coreg = os.path.join(dstdir_coreg, strip_demFname)

                if use_old_trans:
                    old_strip_metaFile = os.path.join(args.get(ARGSTR_META_TRANS_DIR),
                                                      strip_demFname.replace(demSuffix, 'meta.txt'))
                    mosaicked_sceneDemFnames, rmse, trans = readStripMeta_stats(old_strip_metaFile)
                    if not set(mosaicked_sceneDemFnames).issubset(set(remaining_sceneDemFnames)):
                        print("Current source DEMs do not include source DEMs referenced in old strip meta file")
                        use_old_trans = False

                all_data_masked = False
                if not use_old_trans:
                    print("Running s2s with coregistration filter options: {}".format(
                        ', '.join(filter_options_coreg) if filter_options_coreg else None))
                    X, Y, Z, M, O, MD, trans, rmse, mosaicked_sceneDemFnames, spat_ref = scenes2strips(
                        args.get(ARGSTR_SRC), remaining_sceneDemFnames,
                        maskSuffix, filter_options_coreg, args.get(ARGSTR_RMSE_CUTOFF))
                    if X is None:
                        all_data_masked = True

                if not all_data_masked and (filter_options_mask != filter_options_coreg or use_old_trans):
                    print("Running s2s with masking filter options: {}".format(
                        ', '.join(filter_options_mask) if filter_options_mask else None))

                    if 'X' in vars():
                        if dstdir_coreg is not None:
                            if not os.path.isdir(dstdir_coreg):
                                print("Creating dstdir for coreg step directory: {}".format(dstdir_coreg))
                                os.makedirs(dstdir_coreg)
                            print("Saving output from coregistration step")
                            if args.get(ARGSTR_SAVE_COREG_STEP) in (ARGCHO_SAVE_COREG_STEP_META, ARGCHO_SAVE_COREG_STEP_ALL):
                                saveStripMeta(strip_demFile_coreg, demSuffix,
                                              X, Y, Z, trans, rmse, spat_ref,
                                              args.get(ARGSTR_SRC), mosaicked_sceneDemFnames, args)
                            if args.get(ARGSTR_SAVE_COREG_STEP) == ARGCHO_SAVE_COREG_STEP_ALL:
                                saveStripRasters(strip_demFile_coreg, demSuffix, maskSuffix,
                                                 X, Y, Z, M, O, MD, spat_ref)
                                if not args.get(ARGSTR_HILLSHADE_OFF):
                                    saveStripBrowse(strip_demFile_coreg, demSuffix)
                        del X, Y, Z, M, O, MD
                        gc.collect()

                    input_sceneDemFnames = mosaicked_sceneDemFnames
                    X, Y, Z, M, O, MD, trans, rmse, mosaicked_sceneDemFnames, spat_ref = scenes2strips(
                        args.get(ARGSTR_SRC), input_sceneDemFnames,
                        maskSuffix, filter_options_mask, args.get(ARGSTR_RMSE_CUTOFF),
                        trans_guess=trans, rmse_guess=rmse,
                        hold_guess=HOLD_GUESS_ALL, check_guess=True)
                    if X is None:
                        all_data_masked = True
                    if mosaicked_sceneDemFnames != input_sceneDemFnames and use_old_trans:
                        print("Current strip segmentation does not match that found in old strip meta file")
                        print("Rerunning s2s to get new coregistration translation values")
                        use_old_trans = False
                        continue

                remaining_sceneDemFnames = list(set(remaining_sceneDemFnames).difference(set(mosaicked_sceneDemFnames)))
                if all_data_masked:
                    continue

                print("Writing output strip segment with DEM: {}".format(strip_demFile))

                saveStripMeta(strip_demFile, demSuffix,
                              X, Y, Z, trans, rmse, spat_ref,
                              args.get(ARGSTR_SRC), mosaicked_sceneDemFnames, args)
                saveStripRasters(strip_demFile, demSuffix, maskSuffix,
                                 X, Y, Z, M, O, MD, spat_ref)
                if not args.get(ARGSTR_HILLSHADE_OFF):
                    saveStripBrowse(strip_demFile, demSuffix)
                del X, Y, Z, M, O, MD

                segnum += 1

            print('')
            print("Completed processing for this strip-pair ID")

            with open(stripid_finFile, 'w'):
                pass
            if args.get(ARGSTR_SAVE_COREG_STEP) == ARGCHO_SAVE_COREG_STEP_ALL and os.path.isdir(dstdir_coreg):
                with open(stripid_finFile_coreg, 'w'):
                    pass

            print(".fin finished indicator file created: {}".format(stripid_finFile))
            print('')

        except KeyboardInterrupt:
            raise

        except:
            with capture_stdout_stderr() as out:
                traceback.print_exc()
            caught_out, caught_err = out
            error_trace = caught_err
            print(error_trace)

        if type(args.get(ARGSTR_EMAIL)) is str:
            # Send email notification of script completion.

            email_body = SCRIPT_RUNCMD

            if error_trace is not None:
                email_status = "ERROR"
                email_body += "\n{}\n".format(error_trace)
            else:
                email_status = "COMPLETE"

            email_subj = "{} - {}".format(email_status, SCRIPT_FNAME)
            platform_node = platform.node()

            # subprocess.call('echo "{}" | mail -s "{}" {}'.format(email_body, email_subj, email_addr), shell=True)
            msg = MIMEText(email_body)
            msg['Subject'] = email_subj
            msg['From'] = platform_node if platform_node is not None else 'your-computer'
            msg['To'] = args.get(ARGSTR_EMAIL)
            s = smtplib.SMTP('localhost')
            s.sendmail(args.get(ARGSTR_EMAIL), [args.get(ARGSTR_EMAIL)], msg.as_string())
            s.quit()

        if error_trace is not None:
            sys.exit(1)


def saveStripMeta(strip_demFile, demSuffix,
                  X, Y, Z, trans, rmse, spat_ref,
                  scenedir, scene_demFnames, args):
    from lib.raster_array_tools import getFPvertices

    strip_metaFile = strip_demFile.replace(demSuffix, 'meta.txt')

    fp_vertices = getFPvertices(Z, Y, X, label=-9999, label_type='nodata', replicate_matlab=True)
    del Z, X, Y
    proj4 = spat_ref.ExportToProj4()
    time = datetime.today().strftime("%d-%b-%Y %H:%M:%S")

    writeStripMeta(strip_metaFile, scenedir, scene_demFnames,
                   trans, rmse, proj4, fp_vertices, time, args)


def saveStripRasters(strip_demFile, demSuffix, maskSuffix,
                     X, Y, Z, M, O, MD, spat_ref):
    from lib.raster_array_tools import saveArrayAsTiff

    strip_matchFile = strip_demFile.replace(demSuffix, 'matchtag.tif')
    strip_maskFile  = strip_demFile.replace(demSuffix, maskSuffix)
    strip_orthoFile = strip_demFile.replace(demSuffix, 'ortho.tif')

    saveArrayAsTiff(Z, strip_demFile,   X, Y, spat_ref, nodata_val=-9999, dtype_out='float32')
    del Z
    saveArrayAsTiff(M, strip_matchFile, X, Y, spat_ref, nodata_val=0,     dtype_out='uint8')
    del M
    saveArrayAsTiff(O, strip_orthoFile, X, Y, spat_ref, nodata_val=0,     dtype_out='int16')
    del O
    saveArrayAsTiff(MD, strip_maskFile, X, Y, spat_ref, nodata_val=0,     dtype_out='uint8')
    del MD


def saveStripBrowse(strip_demFile, demSuffix):

    strip_demFile_10m    = strip_demFile.replace(demSuffix, 'dem_10m.tif')
    strip_demFile_browse = strip_demFile.replace(demSuffix, 'dem_browse.tif')

    commands = []
    commands.append(
        ('gdal_translate "{0}" "{1}" -q -tr {2} {2} -r bilinear -a_nodata -9999 '
         '-co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(strip_demFile, strip_demFile_10m, 10))
    )
    commands.append(
        ('gdaldem hillshade "{0}" "{1}" -q -z 3 -compute_edges -of GTiff '
         '-co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(strip_demFile_10m, strip_demFile_browse))
    )

    for cmd in commands:
        print(cmd)
        batch_handler.exec_cmd(cmd)

    if not os.path.isfile(strip_demFile_10m):
        raise ExternalError("`gdal_translate` program did not create "
                            "output 10m strip DEM file: {}".format(strip_demFile_10m))
    if not os.path.isfile(strip_demFile_browse):
        raise ExternalError("`gdaldem hillshade` program did not create "
                            "output 10m DEM hillshade file: {}".format(strip_demFile_browse))

    if os.path.isfile(strip_demFile_10m):
        os.remove(strip_demFile_10m)


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


def selectBestMatchtag(demFile):
    demSuffix = getDemSuffix(demFile)
    for matchSuffix in SUFFIX_PRIORITY_MATCHTAG:
        matchFile = demFile.replace(demSuffix, matchSuffix)
        if os.path.isfile(matchFile):
            return matchFile
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


def writeStripMeta(o_metaFile, scenedir, scene_demFnames,
                   trans, rmse, proj4, fp_vertices, strip_time, args):
    from lib.filter_scene import MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT
    from lib.filter_scene import BITMASK_VERSION_NUM

    demSuffix = getDemSuffix(scene_demFnames[0])
    if fp_vertices.dtype != np.int64 and np.array_equal(fp_vertices, fp_vertices.astype(np.int64)):
        fp_vertices = fp_vertices.astype(np.int64)

    mask_version = args.get(ARGSTR_MASK_VER)
    nowater, nocloud = args.get(ARGSTR_NOWATER, ARGSTR_NOCLOUD)
    nofilter_coreg = args.get(ARGSTR_NOFILTER_COREG)

    strip_info = (
"""Strip Metadata (v{})
Creation Date: {}
Strip creation date: {}
Strip projection (proj4): '{}'

Strip Footprint Vertices
X: {}
Y: {}

Mosaicking Alignment Statistics (meters, rmse-cutoff={})
scene, rmse, dz, dx, dy
""".format(
    SCRIPT_VERSION_NUM,
    datetime.today().strftime("%d-%b-%Y %H:%M:%S"),
    strip_time,
    proj4,
    ' '.join(np.array_str(fp_vertices[1], max_line_width=float('inf')).strip()[1:-1].split()),
    ' '.join(np.array_str(fp_vertices[0], max_line_width=float('inf')).strip()[1:-1].split()),
    args.get(ARGSTR_RMSE_CUTOFF)
)
    )

    for i in range(len(scene_demFnames)):
        line = "{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
            scene_demFnames[i], rmse[0, i], trans[0, i], trans[1, i], trans[2, i])
        strip_info += line

        filter_info = "\nFiltering Applied: {} (v{})\n".format(mask_version, BITMASK_VERSION_NUM)

    if mask_version == 'bitmask':
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

    strip_info += "\nScene Metadata \n\n"

    scene_info = ""
    for i in range(len(scene_demFnames)):
        scene_info += "scene {} name={}\n".format(i+1, scene_demFnames[i])

        scene_metaFile = os.path.join(scenedir, scene_demFnames[i].replace(demSuffix, 'meta.txt'))
        if os.path.isfile(scene_metaFile):
            scene_metaFile_fp = open(scene_metaFile, 'r')
            scene_info += scene_metaFile_fp.read()
            scene_metaFile_fp.close()
        else:
            scene_info += "{} not found".format(scene_metaFile)
        scene_info += " \n"

    with open(o_metaFile, 'w') as strip_metaFile_fp:
        strip_metaFile_fp.write(strip_info)
        strip_metaFile_fp.write(scene_info)


def readStripMeta_stats(metaFile):
    metaFile_fp = open(metaFile, 'r')
    try:
        line = metaFile_fp.readline()
        while not line.startswith('Mosaicking Alignment Statistics') and line != '':
            line = metaFile_fp.readline()
        while not line.startswith('scene, rmse, dz, dx, dy') and line != '':
            line = metaFile_fp.readline()
        if line == '':
            raise MetaReadError("{}: Could not parse 'Mosaicking Alignment Statistics'".format(metaFile))

        line = metaFile_fp.readline().strip()
        line_items = line.split(' ')
        sceneDemFnames = [line_items[0]]
        rmse = [line_items[1]]
        trans = np.array([[float(s) for s in line_items[2:5]]])

        while True:
            line = metaFile_fp.readline().strip()
            if line == '':
                break
            line_items = line.split(' ')
            sceneDemFnames.append(line_items[0])
            rmse.append(line_items[1])
            trans = np.vstack((trans, np.array([[float(s) for s in line_items[2:5]]])))

        rmse = np.array([[float(s) for s in rmse]])
        trans = trans.T

    finally:
        metaFile_fp.close()

    return sceneDemFnames, rmse, trans



if __name__ == '__main__':
    main()
