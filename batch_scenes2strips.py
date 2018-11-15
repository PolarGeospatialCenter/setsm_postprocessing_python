
# Erik Husby, Claire Porter; Polar Geospatial Center, University of Minnesota; 2018


import argparse
import filecmp
import gc
import glob
import os
import subprocess
import re
import sys
from time import sleep
from datetime import datetime

import numpy as np

from lib import batch_handler


SCRIPT_VERSION_NUM = 3.1


# Script argument defaults
SCRIPT_FILE = os.path.realpath(__file__)
SCRIPT_FNAME = os.path.basename(SCRIPT_FILE)
SCRIPT_NAME, SCRIPT_EXT = os.path.splitext(SCRIPT_FNAME)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE)
JOBSCRIPT_DIR = os.path.join(SCRIPT_DIR, 'jobscripts')

# Script argument option strings
ARGSTR_SRC = 'src'
ARGSTR_RES = 'res'
ARGSTR_DST = '--dst'
ARGSTR_META_TRANS_DIR = '--meta-trans-dir'
ARGSTR_MASK_VER = '--mask-ver'
ARGSTR_NOENTROPY = '--noentropy'
ARGSTR_NOWATER = '--nowater'
ARGSTR_NOCLOUD = '--nocloud'
ARGSTR_NOFILTER_COREG = '--nofilter-coreg'
ARGSTR_SAVE_COREG_STEP = '--save-coreg-step'
ARGSTR_RMSE_CUTOFF = '--rmse-cutoff'
ARGSTR_SCHEDULER = '--scheduler'
ARGSTR_JOBSCRIPT = '--jobscript'
ARGSTR_DRYRUN = '--dryrun'
ARGSTR_STRIPID = '--stripid'

# Script argument option choices
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

# Segregation of argument option choices
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

# Script batch arguments
BATCH_ARGSTR = [ARGSTR_SCHEDULER, ARGSTR_JOBSCRIPT]

# Batch settings
JOB_ABBREV = 's2s'
PYTHON_EXE = 'python -u'


# Per-script globals

SUFFIX_PRIORITY_DEM = ['dem_smooth.tif', 'dem.tif']
SUFFIX_PRIORITY_MATCHTAG = ['matchtag_mt.tif', 'matchtag.tif']

RE_STRIPID_STR = "(^[A-Z0-9]{4}_.*?_?[0-9A-F]{16}_.*?_?[0-9A-F]{16}).*$"
RE_STRIPID = re.compile(RE_STRIPID_STR)


class MetaReadError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


class RawTextArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter): pass
def parse_args():

    parser = argparse.ArgumentParser(
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
        description=' '.join([
            "Filters scene dems in a source directory,",
            "then mosaics them into strips and saves the results.",
            "\nBatch work is done in units of strip-pair IDs, as parsed from scene dem filenames",
            "(see {} argument for how this is parsed).".format(ARGSTR_STRIPID)
        ])
    )

    # Positional arguments

    parser.add_argument(
        ARGSTR_SRC,
        help=' '.join([
            "Path to source directory containing scene DEMs to process.",
            "If {} is not specified, this path should contain the folder 'tif_results'.".format(ARGSTR_DST)
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
        help=' '.join([
            "Path to destination directory for output mosaicked strip data.",
            "(default is src.(reverse)replace('tif_results', 'strips'))"
        ])
    )

    parser.add_argument(
        ARGSTR_META_TRANS_DIR,
        help=' '.join([
            "Path to directory of old strip metadata from which translation values",
            "will be parsed to skip scene coregistration step."
        ])
    )

    parser.add_argument(
        ARGSTR_MASK_VER,
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
        default=False,
        help=' '.join([
            "Use filter without entropy protection.",
            "Can only be used when {}={}.".format(ARGSTR_MASK_VER, ARGCHO_MASK_VER_MASKV1)
        ])
    )
    parser.add_argument(
        ARGSTR_NOWATER,
        action='store_true',
        default=False,
        help=' '.join([
            "Use filter without water masking.",
            "Can only be used when {}={}.".format(ARGSTR_MASK_VER, ARGCHO_MASK_VER_BITMASK)
        ])
    )
    parser.add_argument(
        ARGSTR_NOCLOUD,
        action='store_true',
        default=False,
        help=' '.join([
            "Use filter without cloud masking.",
            "Can only be used when {}={}.".format(ARGSTR_MASK_VER, ARGCHO_MASK_VER_BITMASK)
        ])
    )
    parser.add_argument(
        ARGSTR_NOFILTER_COREG,
        action='store_true',
        default=False,
        help=' '.join([
            "If {}/{}, turn off the respective filter(s) during".format(ARGSTR_NOWATER, ARGSTR_NOCLOUD),
            "coregistration step in addition to mosaicking step.",
            "Can only be used when {}={}.".format(ARGSTR_MASK_VER, ARGCHO_MASK_VER_BITMASK)
        ])
    )
    parser.add_argument(
        ARGSTR_SAVE_COREG_STEP,
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
        default=1.0,
        help=' '.join([
            "Maximum RMSE from coregistration step tolerated for scene merging.",
            "A value greater than this causes a new strip segment to be created."
        ])
    )

    parser.add_argument(
        ARGSTR_SCHEDULER,
        choices=batch_handler.SCHED_SUPPORTED,
        help="Submit tasks to job scheduler."
    )
    parser.add_argument(
        ARGSTR_JOBSCRIPT,
        help=' '.join([
            "Script to run in job submission to scheduler.",
            "(default scripts are found in {})".format(JOBSCRIPT_DIR)
        ])
    )
    parser.add_argument(
        ARGSTR_DRYRUN,
        action='store_true',
        default=False,
        help="Print actions without executing."
    )

    parser.add_argument(
        ARGSTR_STRIPID,
        help=' '.join([
            "Run filtering and mosaicking for a single strip with strip-pair ID",
            "as parsed from scene DEM filenames using the following regex: '{}'".format(RE_STRIPID_STR)
        ])
    )

    return parser


def main():
    from lib.filter_scene import generateMasks
    from lib.filter_scene import MASK_FLAT, MASK_SEPARATE, MASK_BIT
    from lib.filter_scene import DEBUG_NONE, DEBUG_ALL, DEBUG_MASKS, DEBUG_ITHRESH
    from lib.scenes2strips import scenes2strips
    from batch_mask import get_mask_bitstring

    # Parse and validate arguments.

    arg_parser = parse_args()
    args = batch_handler.ArgumentPasser(arg_parser, PYTHON_EXE, SCRIPT_FILE)
    srcdir = os.path.abspath(args.get(ARGSTR_SRC))
    res = args.get(ARGSTR_RES)
    if int(res) == res:
        res = int(res)
    dstdir = args.get(ARGSTR_DST)
    metadir = args.get(ARGSTR_META_TRANS_DIR)
    mask_version = args.get(ARGSTR_MASK_VER)
    noentropy = args.get(ARGSTR_NOENTROPY)
    nowater = args.get(ARGSTR_NOWATER)
    nocloud = args.get(ARGSTR_NOCLOUD)
    nofilter_coreg = args.get(ARGSTR_NOFILTER_COREG)
    save_coreg_step = args.get(ARGSTR_SAVE_COREG_STEP)
    rmse_cutoff = args.get(ARGSTR_RMSE_CUTOFF)
    scheduler = args.get(ARGSTR_SCHEDULER)
    jobscript = args.get(ARGSTR_JOBSCRIPT)
    jobscript_default = os.path.join(JOBSCRIPT_DIR, '{}_{}.sh'.format(SCRIPT_NAME, scheduler))
    dryrun = args.get(ARGSTR_DRYRUN)
    stripid = args.get(ARGSTR_STRIPID)

    if not os.path.isdir(srcdir):
        arg_parser.error("`{}` must be a directory".format(ARGSTR_SRC))

    if dstdir is not None:
        dstdir = os.path.abspath(dstdir)
        if os.path.isdir(dstdir) and filecmp.cmp(srcdir, dstdir):
            arg_parser.error("`{}` dir is the same as {} dir".format(ARGSTR_SRC, ARGSTR_DST))
    else:
        # Set default dst dir.
        split_ind = srcdir.rfind('tif_results')
        if split_ind == -1:
            arg_parser.error("`{}` path does not contain 'tif_results', "
                             "so default {} cannot be set".format(ARGSTR_SRC, ARGSTR_DST))
        dstdir = srcdir[:split_ind] + srcdir[split_ind:].replace('tif_results', 'strips')
        dstdir = os.path.abspath(dstdir)
        print("{} dir set to: {}".format(ARGSTR_DST, dstdir))
        args.set(ARGSTR_DST, dstdir)

    if metadir is not None:
        if os.path.isdir(metadir):
            metadir = os.path.abspath(metadir)
        else:
            arg_parser.error("{} must be an existing directory".format(ARGSTR_META_TRANS_DIR))

    if res == 8:
        res_req_mask_ver = MASK_VER_8M
    elif res == 2:
        res_req_mask_ver = MASK_VER_2M
    else:
        res_req_mask_ver = MASK_VER_XM
    if mask_version not in res_req_mask_ver:
        arg_parser.error("{} must be one of {} for {}-meter `{}`".format(
            ARGSTR_MASK_VER, res_req_mask_ver, res, ARGSTR_RES
        ))

    if args.get(ARGSTR_NOENTROPY) and mask_version != ARGCHO_MASK_VER_MASKV1:
        arg_parser.error("{} option is compatible only with {} option".format(
            ARGSTR_NOENTROPY, ARGCHO_MASK_VER_MASKV1
        ))
    if (nowater or nocloud) and mask_version != ARGCHO_MASK_VER_BITMASK:
        arg_parser.error("{}/{} option(s) can only be used when {}='{}'".format(
            ARGSTR_NOWATER, ARGSTR_NOCLOUD, ARGSTR_MASK_VER, ARGCHO_MASK_VER_BITMASK
        ))
    if nofilter_coreg and [nowater, nocloud].count(True) == 0:
        arg_parser.error("{} option must be used in conjunction with {}/{} option(s)".format(
            ARGSTR_NOFILTER_COREG, ARGSTR_NOWATER, ARGSTR_NOCLOUD
        ))
    if nofilter_coreg and metadir is not None:
        arg_parser.error("{} option cannot be used in conjunction with {} argument".format(
            ARGSTR_NOFILTER_COREG, ARGSTR_META_TRANS_DIR
        ))

    if (    save_coreg_step != ARGCHO_SAVE_COREG_STEP_OFF
        and ((not (nowater or nocloud))
             or (metadir is not None or nofilter_coreg))):
        arg_parser.error("Non-'{}' {} option must be used in conjunction with ({}/{}) arguments "
                         "and cannot be used in conjunction with ({}/{}) arguments".format(
            ARGCHO_SAVE_COREG_STEP_OFF, ARGSTR_SAVE_COREG_STEP,
            ARGSTR_NOWATER, ARGSTR_NOCLOUD,
            ARGSTR_META_TRANS_DIR, ARGSTR_NOFILTER_COREG
        ))

    if rmse_cutoff <= 0:
        arg_parser.error("{} must be greater than zero".format(ARGSTR_RMSE_CUTOFF))

    if scheduler is not None:
        if jobscript is None:
            jobscript = jobscript_default
        jobscript = os.path.abspath(jobscript)
        if not os.path.isfile(jobscript):
            arg_parser.error("{} must be a valid file path, but was '{}'".format(
                ARGSTR_JOBSCRIPT, jobscript
            ))

    # Create strip output directory if it doesn't already exist.
    if not os.path.isdir(dstdir):
        if not dryrun:
            os.makedirs(dstdir)


    if stripid is None:
        # Do batch processing.


        # Find all scene DEMs to be merged into strips.
        for demSuffix in SUFFIX_PRIORITY_DEM:
            scene_dems = glob.glob(os.path.join(srcdir, '*_{}_{}'.format(str(res)[0], demSuffix)))
            if scene_dems:
                break
        if not scene_dems:
            print("No scene DEMs found to process, exiting")
            sys.exit(1)
        scene_dems.sort()

        # Find all unique strip IDs.
        try:
            stripids = list(set([re.match(RE_STRIPID, os.path.basename(s)).group(1) for s in scene_dems]))
        except AttributeError:
            print("There are source scene DEMs for which a strip ID cannot be parsed. "
                  "Please fix source raster filenames so that a strip ID can be parsed "
                  "using the following regular expression: '{}'".format(RE_STRIPID_STR))
            raise
        stripids.sort()

        # Check for existing strip output.
        stripids_to_process = [sID for sID in stripids
                               if not os.path.isfile(os.path.join(dstdir, '{}_{}m.fin'.format(sID, res)))]
        print("Found {} {} strip-pair IDs, {} unfinished".format(
            len(stripids), '*'+demSuffix, len(stripids_to_process)))
        del scene_dems
        if len(stripids_to_process) == 0:
            print("No unfinished strip DEMs found to process, exiting")
            sys.exit(0)
        stripids_to_process.sort()


        wait_seconds = 5
        print("Sleeping {} seconds before task submission".format(wait_seconds))
        sleep(wait_seconds)


        # Process each strip-pair ID.

        jobnum_fmt = batch_handler.get_jobnum_fmtstr(stripids)
        args.remove_args(*BATCH_ARGSTR)
        for i, stripid in enumerate(stripids):

            # If output does not already exist, add to task list.
            stripid_finFile = os.path.join(dstdir, '{}_{}m.fin'.format(stripid, res))
            dst_dems = glob.glob(os.path.join(dstdir, '*{}_seg*_{}m_{}'.format(stripid, res, demSuffix)))
            if os.path.isfile(stripid_finFile):
                print("{} {}_{}m.fin file exists, skipping".format(ARGSTR_STRIPID, stripid, res))
                continue
            elif dst_dems:
                print("{} {} output files exist (potentially unfinished), skipping".format(ARGSTR_STRIPID, stripid))
                continue

            args.set(ARGSTR_STRIPID, stripid)
            s2s_command = args.get_cmd()

            if scheduler is not None:
                job_name = JOB_ABBREV+jobnum_fmt.format(i+1)
                cmd = batch_handler.get_jobsubmit_cmd(scheduler, jobscript, job_name, s2s_command)
            else:
                cmd = s2s_command

            print("{}, {}".format(i+1, cmd))
            if not dryrun:
                # For most cases, set `shell=True`.
                # For attaching process to PyCharm debugger,
                # set `shell=False`.
                subprocess.call(cmd, shell=True)


    else:
        # Process a single strip.
        print('')

        # Handle arguments.

        use_old_trans = True if metadir is not None else False

        mask_name = 'mask' if mask_version == ARGCHO_MASK_VER_MASKV2 else mask_version

        filter_options_mask = ()
        if nowater:
            filter_options_mask += ('nowater',)
        if nocloud:
            filter_options_mask += ('nocloud',)

        filter_options_coreg = filter_options_mask if nofilter_coreg else ()

        dstdir_coreg = os.path.join(
            os.path.dirname(dstdir),
            '{}_coreg_filt{}'.format(
                os.path.basename(dstdir),
                get_mask_bitstring(True,
                                   not 'nowater' in filter_options_coreg,
                                   not 'nocloud' in filter_options_coreg)))

        # Print arguments for this run.
        print("stripid: {}".format(stripid))
        print("res: {}m".format(res))
        print("srcdir: {}".format(srcdir))
        print("dstdir: {}".format(dstdir))
        print("dstdir for coreg step: {}".format(dstdir_coreg))
        print("metadir: {}".format(metadir))
        print("mask version: {}".format(mask_version))
        print("mask name: {}".format(mask_name))
        print("coreg filter options: {}".format(filter_options_coreg))
        print("mask filter options: {}".format(filter_options_mask))
        print("rmse cutoff: {}".format(rmse_cutoff))
        print("dryrun: {}".format(dryrun))
        print('')

        if os.path.isdir(dstdir_coreg) and save_coreg_step != ARGCHO_SAVE_COREG_STEP_OFF:
            dstdir_coreg_stripFiles = glob.glob(os.path.join(dstdir_coreg, stripid+'*'))
            if len(dstdir_coreg_stripFiles) > 0:
                print("Deleting old strip output in dstdir for coreg step")
                if not dryrun:
                    for f in dstdir_coreg_stripFiles:
                        os.remove(f)

        stripid_finFile = os.path.join(dstdir, '{}_{}m.fin'.format(stripid, res))
        stripid_finFile_coreg = os.path.join(dstdir_coreg, '{}_{}m.fin'.format(stripid, res))

        # Find scene DEMs for this stripid to be merged into strips.
        for demSuffix in SUFFIX_PRIORITY_DEM:
            scene_demFiles = glob.glob(os.path.join(srcdir, '{}*_{}_{}'.format(stripid, str(res)[0], demSuffix)))
            if scene_demFiles:
                break
        print("Processing strip-pair ID: {}, {} scenes".format(stripid, len(scene_demFiles)))
        if not scene_demFiles:
            print("No scene DEMs found to process, skipping")
            sys.exit(1)
        scene_demFiles.sort()

        # Existence check. If output already exists, skip.
        if os.path.isfile(stripid_finFile):
            print("{} file exists, strip output finished, skipping".format(stripid_finFile))
            sys.exit(0)
        if glob.glob(os.path.join(dstdir, stripid+'*')):
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

        print('')

        # Filter all scenes in this strip.
        filter_list = [f for f in scene_demFiles if shouldDoMasking(selectBestMatchtag(f), mask_name)]
        filter_total = len(filter_list)
        i = 0
        for demFile in filter_list:
            i += 1
            print("Filtering {} of {}: {}".format(i, filter_total, demFile))
            if not dryrun:
                generateMasks(demFile, mask_name, noentropy=noentropy,
                              save_component_masks=MASK_BIT, debug_component_masks=DEBUG_NONE,
                              nbit_masks=False)

        print('')
        print("All *_{}.tif scene masks have been created in source scene directory".format(mask_name))
        print('')

        print("Running scenes2strips")
        if dryrun:
            sys.exit(0)
        print('')

        # Mosaic scenes in this strip together.
        # Output separate segments if there are breaks in overlap.
        maskSuffix = mask_name+'.tif'
        remaining_sceneDemFnames = [os.path.basename(f) for f in scene_demFiles]
        segnum = 1
        while len(remaining_sceneDemFnames) > 0:

            print("Building segment {}".format(segnum))

            strip_demFname = "{}_seg{}_{}m_{}".format(stripid, segnum, res, demSuffix)
            strip_demFile = os.path.join(dstdir, strip_demFname)
            strip_demFile_coreg = os.path.join(dstdir_coreg, strip_demFname)

            if use_old_trans:
                strip_metaFile = os.path.join(metadir, strip_demFname.replace(demSuffix, 'meta.txt'))
                mosaicked_sceneDemFnames, rmse, trans = readStripMeta_stats(strip_metaFile)
                if not set(mosaicked_sceneDemFnames).issubset(set(remaining_sceneDemFnames)):
                    print("Current source DEMs do not include source DEMs referenced in old strip meta file")
                    use_old_trans = False

            all_data_masked = False
            if not use_old_trans:
                print("Running s2s with coregistration filter options: {}".format(
                    ', '.join(filter_options_coreg) if filter_options_coreg else None))
                X, Y, Z, M, O, MD, trans, rmse, mosaicked_sceneDemFnames, spat_ref = scenes2strips(
                    srcdir, remaining_sceneDemFnames, maskSuffix, filter_options_coreg, rmse_cutoff)
                if X is None:
                    all_data_masked = True

            if not all_data_masked and (filter_options_mask != filter_options_coreg or use_old_trans):
                print("Running s2s with masking filter options: {}".format(
                    ', '.join(filter_options_mask) if filter_options_mask else None))

                if 'X' in vars():
                    if save_coreg_step != ARGCHO_SAVE_COREG_STEP_OFF:
                        if not os.path.isdir(dstdir_coreg):
                            os.makedirs(dstdir_coreg)
                        if save_coreg_step in (ARGCHO_SAVE_COREG_STEP_META, ARGCHO_SAVE_COREG_STEP_ALL):
                            saveStripMeta(strip_demFile_coreg, demSuffix,
                                          X, Y, Z, trans, rmse, spat_ref,
                                          srcdir, mosaicked_sceneDemFnames, args)
                        if save_coreg_step == ARGCHO_SAVE_COREG_STEP_ALL:
                            saveStripRasters(strip_demFile_coreg, demSuffix, maskSuffix,
                                             X, Y, Z, M, O, MD, spat_ref)
                    del X, Y, Z, M, O, MD
                    gc.collect()

                input_sceneDemFnames = mosaicked_sceneDemFnames
                X, Y, Z, M, O, MD, trans, rmse, mosaicked_sceneDemFnames, spat_ref = scenes2strips(
                    srcdir, input_sceneDemFnames, maskSuffix, filter_options_mask, rmse_cutoff,
                    trans_guess=trans, rmse_guess=(rmse if use_old_trans else None), hold_guess=True)
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

            print("DEM: {}".format(strip_demFile))

            saveStripMeta(strip_demFile, demSuffix,
                          X, Y, Z, trans, rmse, spat_ref,
                          srcdir, mosaicked_sceneDemFnames, args)
            saveStripRasters(strip_demFile, demSuffix, maskSuffix,
                             X, Y, Z, M, O, MD, spat_ref)
            del X, Y, Z, M, O, MD

            segnum += 1

        with open(stripid_finFile, 'w'):
            pass
        if save_coreg_step == ARGCHO_SAVE_COREG_STEP_ALL and os.path.isdir(dstdir_coreg):
            with open(stripid_finFile_coreg, 'w'):
                pass

        print('')
        print("Fin!")


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
        line = "{} {:.2f} {:.4f} {:.4f} {:.4f}\n".format(
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
        while not line.startswith('Mosaicking Alignment Statistics (meters)') and line != '':
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
