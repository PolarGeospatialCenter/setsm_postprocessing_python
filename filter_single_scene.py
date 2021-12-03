#!/usr/bin/env python

# Erik Husby, Claire Porter; Polar Geospatial Center, University of Minnesota; 2019


from __future__ import division
from lib import script_utils

PYTHON_VERSION_ACCEPTED_MIN = "2.7"  # supports multiple dot notation
if script_utils.PYTHON_VERSION < script_utils.VersionString(PYTHON_VERSION_ACCEPTED_MIN):
    raise script_utils.VersionError("Python version ({}) is below accepted minimum ({})".format(
        script_utils.PYTHON_VERSION, PYTHON_VERSION_ACCEPTED_MIN))


import argparse
import os
import re
import sys

from lib import script_utils
from lib.script_utils import ScriptArgumentError, ExternalError, InvalidArgumentError

from lib.filter_scene import generateMasks
from lib.filter_scene import MASK_FLAT, MASK_SEPARATE, MASK_BIT
from lib.filter_scene import DEBUG_NONE, DEBUG_ALL, DEBUG_MASKS, DEBUG_ITHRESH


##############################

## Core globals

SCRIPT_VERSION_NUM = script_utils.VersionString('4')

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
ARGSTR_SRCDEM = 'srcdem'
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

RE_STRIPID_STR = "(^[A-Z0-9]{4}_.*?_?[0-9A-F]{16}_.*?_?[0-9A-F]{16}).*$"
RE_STRIPID = re.compile(RE_STRIPID_STR)

##############################


class MetaReadError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


def argparser_init():

    parser = argparse.ArgumentParser(
        formatter_class=script_utils.RawTextArgumentDefaultsHelpFormatter,
        description=' '.join([
            "Generate a scene mask raster file for a single scene DEM."
        ])
    )

    # Positional arguments

    parser.add_argument(
        ARGSTR_SRCDEM,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_SRCDEM,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=True),
        help=' '.join([
            "Path to source scene DEM file to be masked.",
        ])
    )
    parser.add_argument(
        ARGSTR_RES,
        type=float,
        help="Resolution of target DEMs in meters."
    )

    # Optional arguments

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
        ARGSTR_USE_PIL_IMRESIZE,
        action='store_true',
        help=' '.join([
            "Use PIL imresize method over usual fast OpenCV resize method for final resize from ",
            "8m processing resolution back to native scene raster resolution when generating ",
            "output bitmask raster. This is to avoid an OpenCV error with unknown cause that ",
            "can occur when using OpenCV resize on some 50cm scenes at Blue Waters."
        ])
    )

    parser.add_argument(
        ARGSTR_DRYRUN,
        action='store_true',
        help="Print actions without executing."
    )

    parser.add_argument(
        ARGSTR_SKIP_ORTHO2_ERROR,
        action='store_true',
        help=' '.join([
            "If at least one scene in a cross-track strip is missing the second ortho component raster,",
            "do not throw an error but instead build the strip as if it were in-track with only one ortho."
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


    scenedem_ffile = args.get(ARGSTR_SRCDEM)
    demSuffix = getDemSuffix(scenedem_ffile)
    if demSuffix is None:
        arg_parser.error("argument {} source scene DEM must end with one of the following suffixes: {}".format(
            ARGSTR_SRCDEM, SUFFIX_PRIORITY_DEM
        ))
    
    mask_name = 'mask' if args.get(ARGSTR_MASK_VER) == ARGCHO_MASK_VER_MASKV2 else args.get(ARGSTR_MASK_VER)
    scene_mask_name = demSuffix.replace('.tif', '_'+mask_name)

    scenedem_fname = os.path.basename(scenedem_ffile)
    stripid_is_xtrack = scenedem_fname[1].isdigit()


    # Make sure all DEM component files exist. If missing, skip.
    bypass_ortho2 = False
    src_scenefile_missing_flag = False
    src_scenedem_ffile_glob = [scenedem_ffile]
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

    if src_scenefile_missing_flag:
        raise FileNotFoundError("Source scene file(s) missing")


    print("Generating '{}' mask file for scene DEM: {}".format(scene_mask_name, scenedem_ffile))
    if not args.get(ARGSTR_DRYRUN):
        generateMasks(scenedem_ffile, scene_mask_name, noentropy=args.get(ARGSTR_NOENTROPY),
                      save_component_masks=MASK_BIT, use_second_ortho=(stripid_is_xtrack and not bypass_ortho2),
                      debug_component_masks=DEBUG_NONE, nbit_masks=False,
                      use_pil_imresize=args.get(ARGSTR_USE_PIL_IMRESIZE))

    print("Done!")


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
    
    
    
if __name__ == '__main__':
    main()
