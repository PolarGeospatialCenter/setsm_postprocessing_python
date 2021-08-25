#!/usr/bin/env python

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
import logging
import os
import re
import shutil
import subprocess
import sys
import traceback
from time import sleep

import numpy as np
from osgeo import gdal, ogr, osr

from lib import script_utils
from lib.script_utils import ScriptArgumentError
from lib import raster_io
from lib import demregex

gdal.UseExceptions()


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

## Argument globals

# Argument strings
ARGSTR_SRC = 'src'
ARGSTR_DSTDIR = '--dstdir'
ARGSTR_TARGET_EPSG = '--target-epsg'
ARGSTR_TARGET_RESOLUTION = '--target-res'
ARGSTR_OVERWRITE = '--overwrite'
ARGSTR_ADD_RES_SUFFIX = '--add-res-suffix'
ARGSTR_SIMPLE_META_FP_VERTS = '--simple-meta-fp-verts'
ARGSTR_STRIPS_NO_BROWSE = '--strips-no-browse'
ARGSTR_STRIPS_BUILD_AUX = '--strips-build-aux'
ARGSTR_SCHEDULER = '--scheduler'
ARGSTR_JOBSCRIPT = '--jobscript'
ARGSTR_TASKS_PER_JOB = '--tasks-per-job'
ARGSTR_SCRATCH = '--scratch'
ARGSTR_LOGDIR = '--logdir'
ARGSTR_EMAIL = '--email'
ARGSTR_DRYRUN = '--dryrun'

# Argument choices

# Argument defaults
ARGDEF_SCRATCH = os.path.join(os.path.expanduser('~'), 'scratch', 'task_bundles')

# Argument groups
ARGGRP_OUTDIR = [ARGSTR_DSTDIR, ARGSTR_LOGDIR, ARGSTR_SCRATCH]
ARGGRP_BATCH = [ARGSTR_SCHEDULER, ARGSTR_JOBSCRIPT, ARGSTR_TASKS_PER_JOB, ARGSTR_EMAIL]

##############################

## Batch settings

JOBSCRIPT_DIR = os.path.join(SCRIPT_DIR, 'jobscripts')
JOBSCRIPT_INIT = os.path.join(JOBSCRIPT_DIR, 'init.sh')
JOB_ABBREV = 'Reproj'
JOB_WALLTIME_HR = 30
JOB_MEMORY_GB = 20

##############################

## Custom globals

SETSM_META_SUFFIX = 'meta.txt'
STRIP_DEM_SUFFIX = 'dem.tif'

SETSM_SUFFIX_KEY_TO_RESAMPLE_METHOD_LOOKUP = [
    ('shade', 'cubic'),
    ('dem', 'bilinear'),
    ('ortho', 'cubic'),
    ('matchtag', 'near'),
    ('bitmask', 'near'),
    ('mask', 'near'),
]

BUILD_STRIP_AUX_CORE_STRIP_SUFFIXES = [
    'dem.tif',
    'ortho.tif',
    'matchtag.tif',
    'bitmask.tif',
]

RE_UTM_PROJNAME = re.compile("\Autm\d+[ns]\Z")
RE_SHELVED_STRIP = re.compile("\A.*/strips_v4/2m/")

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
lso = logging.StreamHandler(sys.stdout)
lso.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s- %(message)s','%m-%d-%Y %H:%M:%S')
lso.setFormatter(formatter)
logger.addHandler(lso)

##############################


def osr_srs_preserve_axis_order(osr_srs):
    try:
        # revert to GDAL 2.x axis conventions to maintain consistent results if GDAL 3+ used
        osr_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except AttributeError:
        pass
    return osr_srs


def argparser_init():

    parser = argparse.ArgumentParser(
        formatter_class=script_utils.RawTextArgumentDefaultsHelpFormatter,
        description=' '.join([
            "Reproject SETSM scene/strip DEM result files."
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
            "Path to source SETSM scene/strip *{} file or directory containing these files.".format(SETSM_META_SUFFIX),
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
            "Path to destination directory for reprojected SETSM data.",
        ])
    )

    parser.add_argument(
        ARGSTR_TARGET_EPSG,
        type=int,
        default=None,
        help=' '.join([
            "Target EPSG code of reprojected data.",
        ])
    )

    parser.add_argument(
        ARGSTR_TARGET_RESOLUTION,
        type=int,
        default=None,
        help=' '.join([
            "Target resolution of reprojected data in meters.",
        ])
    )

    parser.add_argument(
        ARGSTR_OVERWRITE,
        action='store_true',
        help="Overwrite existing data in destination directory."
    )

    parser.add_argument(
        ARGSTR_ADD_RES_SUFFIX,
        action='store_true',
        help=' '.join([
            "Add '_<res>m' suffix to filenames of reprojected rasters.",
            "where <res> is the supplied {} argument in meters.".format(ARGSTR_TARGET_RESOLUTION),
        ])
    )

    parser.add_argument(
        ARGSTR_SIMPLE_META_FP_VERTS,
        action='store_true',
        help=' '.join([
            "When reprojecting strip DEM data, simply reproject strip footprint vertices "
            "instead of recalculating correct vertices from reprojected strip *dem.tif raster."
        ])
    )

    parser.add_argument(
        ARGSTR_STRIPS_NO_BROWSE,
        action='store_true',
        help=' '.join([
            "When reprojecting 2-meter strip DEMs, do not build 10-meter hillshade raster."
        ])
    )

    parser.add_argument(
        ARGSTR_STRIPS_BUILD_AUX,
        action='store_true',
        default=False,
        help=' '.join([
            "When reprojecting 2-meter strip DEMs, build the same set of auxiliary rasters that would",
            "normally be built at the end of the scenes2strips routine (useful for mosaicking)."
        ])
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

    if args.get(ARGSTR_ADD_RES_SUFFIX):
        global STRIP_DEM_SUFFIX
        if args.get(ARGSTR_TARGET_RESOLUTION) is None:
            arg_parser.error("{} argument must be provided to use {} option".format(
                ARGSTR_TARGET_RESOLUTION, ARGSTR_ADD_RES_SUFFIX
            ))
        dem_suffix_base, dem_suffix_ext = os.path.splitext(STRIP_DEM_SUFFIX)
        STRIP_DEM_SUFFIX = '{}_{}m{}'.format(
            dem_suffix_base, args.get(ARGSTR_TARGET_RESOLUTION), dem_suffix_ext
        )

    if args.get(ARGSTR_STRIPS_NO_BROWSE) or args.get(ARGSTR_STRIPS_BUILD_AUX):
        if args.get(ARGSTR_ADD_RES_SUFFIX):
            arg_parser.error("{}/{} and {} options are incompatible".format(
                ARGSTR_STRIPS_NO_BROWSE, ARGSTR_STRIPS_BUILD_AUX, ARGSTR_ADD_RES_SUFFIX
            ))
        if args.get(ARGSTR_TARGET_RESOLUTION) not in (None, 2):
            arg_parser.error("{}/{} options are only compatible with a target resolution of 2 meters".format(
                ARGSTR_STRIPS_NO_BROWSE, ARGSTR_STRIPS_BUILD_AUX
            ))

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


    # Gather list of tasks.
    src_tasks_all = None

    if os.path.isfile(src) and src.endswith(SETSM_META_SUFFIX):
        src_tasks_all = [src]

    elif os.path.isfile(src) and src.endswith('.txt'):
        bundle_file = src
        src_tasks_all = script_utils.read_task_bundle(bundle_file, args_delim=' ')
        
    elif os.path.isdir(src):
        srcdir = src
        src_tasks_all = []
        for root, dnames, fnames in os.walk(srcdir):
            for fn in fnames:
                if fn.endswith(SETSM_META_SUFFIX):
                    src_tasks_all.append(os.path.join(root, fn))
        src_tasks_all.sort()

    if src_tasks_all is None:
        arg_parser.error("argument {} must be a path to either a directory or a file ending with '{}', "
                         "but was '{}'".format(ARGSTR_SRC, SETSM_META_SUFFIX, src))
        
    src_tasks_incomplete = []
        
    test_srs = osr.SpatialReference()
    for task in src_tasks_all:
        task_metafile_src = ''
        task_dstdir = args.get(ARGSTR_DSTDIR)
        task_target_epsg = args.get(ARGSTR_TARGET_EPSG)
        
        if type(task) is list:
            if len(task) == 2:
                task_metafile_src, task_dstdir = task
            elif len(task) == 3:
                task_metafile_src, task_dstdir, task_target_epsg = task
                task_target_epsg = int(task_target_epsg)
                if task_dstdir in ('psn', 'pss') or re.match(RE_UTM_PROJNAME, task_dstdir) is not None:
                    shelved_strip_match = re.search(RE_SHELVED_STRIP, task_metafile_src)
                    if shelved_strip_match is not None:
                        strips_res_dir = shelved_strip_match.group(0)
                        task_dstdir = "{}_{}".format(strips_res_dir.rstrip('/'), task_dstdir)
                        task[1] = task_dstdir
            else:
                arg_parser.error("Source task list can only have up to three columns: "
                                 "src_metafile, dstdir, target_epsg")
        else:
            task_metafile_src = task

        task = [task_metafile_src, task_dstdir, task_target_epsg]
            
        if not os.path.isfile(task_metafile_src):
            arg_parser.error("Source metafile does not exist: {}".format(task_metafile_src))
        if task_dstdir is None:
            arg_parser.error("Destination directory must be specified")
        if task_target_epsg is None:
            arg_parser.error("Target EPSG code must be specified")
        status = test_srs.ImportFromEPSG(task_target_epsg)
        if status != 0:
            arg_parser.error("Target EPSG code is invalid: {}".format(task_target_epsg))

        task_dstdir = get_task_dstdir(task_metafile_src, task_dstdir)
        task_metafile_dst = os.path.join(task_dstdir, os.path.basename(task_metafile_src))
        if not os.path.isfile(task_metafile_dst) or args.get(ARGSTR_OVERWRITE):
            src_tasks_incomplete.append(task)
            
    num_tasks = len(src_tasks_incomplete)

    print("Number of source SETSM metafiles found: {}".format(len(src_tasks_all)))
    print("Number of incomplete reprojection tasks: {}".format(num_tasks))

    if num_tasks == 0:
        sys.exit(0)
    elif args.get(ARGSTR_DRYRUN) and args.get(ARGSTR_SCHEDULER) is not None:
        print("Exiting dryrun")
        sys.exit(0)

    # Pause for user review.
    print("-----")
    wait_seconds = 8
    print("Sleeping {} seconds before task submission".format(wait_seconds))
    sleep(wait_seconds)
    print("-----")


    ## Create output directories if they don't already exist.
    if not args.get(ARGSTR_DRYRUN):
        for dir_argstr, dir_path in list(zip(ARGGRP_OUTDIR, args.get_as_list(ARGGRP_OUTDIR))):
            if dir_path is not None and not os.path.isdir(dir_path):
                print("Creating argument {} directory: {}".format(dir_argstr, dir_path))
                os.makedirs(dir_path, exist_ok=True)


    ## Process tasks.

    if args.get(ARGSTR_SCHEDULER) is not None:
        # Process tasks in batch.

        tasks_per_job = args.get(ARGSTR_TASKS_PER_JOB)
        src_tasks = (src_tasks_incomplete if tasks_per_job is None else
                     script_utils.write_task_bundles(src_tasks_incomplete, tasks_per_job,
                                                     args.get(ARGSTR_SCRATCH),
                                                     '{}_{}'.format(JOB_ABBREV, ARGSTR_SRC),
                                                     task_delim=' '))

        jobnum_fmt = script_utils.get_jobnum_fmtstr(src_tasks)
        last_job_email = args.get(ARGSTR_EMAIL)

        args_batch = args
        args_single = copy.deepcopy(args)
        args_single.unset(*ARGGRP_BATCH)

        job_num = 0
        num_jobs = len(src_tasks)
        for task in src_tasks:
            job_num += 1

            if type(task) is list:
                task_metafile_src, task_dstdir, task_target_epsg = task
                args_single.set(ARGSTR_SRC, task_metafile_src)
                args_single.set(ARGSTR_DSTDIR, task_dstdir if task_dstdir is not None else args.get(ARGSTR_DSTDIR))
                args_single.set(ARGSTR_TARGET_EPSG, task_target_epsg if task_target_epsg is not None else args.get(ARGSTR_TARGET_EPSG))
            else:
                task_bundlefile_src = task
                args_single.set(ARGSTR_SRC, task_bundlefile_src)

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
            # Process tasks in serial.

            for i, task in enumerate(src_tasks_incomplete):

                task_metafile_src, task_dstdir, task_target_epsg = task

                print("Reproject ({}/{}): {}".format(i+1, num_tasks, task_metafile_src))
                # if not args.get(ARGSTR_DRYRUN):
                reproject_setsm(task_metafile_src, task_dstdir, task_target_epsg, args=args)

            print("\nCompleted task processing loop")

        except KeyboardInterrupt:
            print("\nTask processing loop interrupted with KeyboardInterrupt")
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

            print("\nTask processing loop was not completed due to exception")

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


def get_task_dstdir(src_metafile, dstdir):

    src_metafile_dname = os.path.basename(os.path.dirname(src_metafile))
    dstdname = os.path.basename(dstdir)

    src_dname_pname_match = demregex.Pairname(src_metafile_dname).match()
    dst_dname_pname_match = demregex.Pairname(dstdname).match()

    if src_dname_pname_match.matched and not dst_dname_pname_match.matched:
        dstdir = os.path.join(dstdir, src_metafile_dname)
    elif src_dname_pname_match.matched and dst_dname_pname_match.matched:
        if src_dname_pname_match.pairname != dst_dname_pname_match.pairname:
            raise script_utils.ScriptArgumentError(
                "SETSM 'pairname' in folder containing source metafile ({}) is not the same as "
                "that found in destination directory ({}): {}, {}".format(
                    src_dname_pname_match.pairname, dst_dname_pname_match.pairname,
                    src_metafile, dstdir
                )
            )

    return dstdir


def reproject_setsm(src_metafile, dstdir=None, target_epsg=None, target_resolution=None, args=None):

    src_is_strip_metafile = demregex.StripDemFile(os.path.basename(src_metafile)).match().matched
    normal_strip_output = (target_resolution in (None, 2) and not args.get(ARGSTR_ADD_RES_SUFFIX))

    if args is None and any([a is None for a in [dstdir, target_epsg]]):
        raise script_utils.InvalidArgumentError("`args` must be provided if `dstdir` or `target_epsg` is None")
    if dstdir is None:
        dstdir = args.get(ARGSTR_DSTDIR)
    if target_epsg is None:
        target_epsg = args.get(ARGSTR_TARGET_EPSG)
    if target_resolution is None:
        target_resolution = args.get(ARGSTR_TARGET_RESOLUTION)

    if not os.path.isfile(src_metafile):
        raise script_utils.InvalidArgumentError("Source metafile does not exist: {}".format(src_metafile))

    dstdir = get_task_dstdir(src_metafile, dstdir)

    if not os.path.isdir(dstdir):
        if not args.get(ARGSTR_DRYRUN):
            os.makedirs(dstdir, exist_ok=True)

    if src_is_strip_metafile and normal_strip_output:
        src_rasters_to_reproject = [
            src_metafile.replace(SETSM_META_SUFFIX, suffix)
            for suffix in BUILD_STRIP_AUX_CORE_STRIP_SUFFIXES
        ]
        src_ortho2file = src_metafile.replace(SETSM_META_SUFFIX, 'ortho2.tif')
        if os.path.isfile(src_ortho2file):
            src_rasters_to_reproject.append(src_ortho2file)
    else:
        src_rasters_to_reproject = glob.glob(src_metafile.replace(SETSM_META_SUFFIX, '*.tif'))

    commands = []

    for rasterfile_src in src_rasters_to_reproject:
        rasterfile_dst = os.path.join(dstdir, os.path.basename(rasterfile_src))
        if args.get(ARGSTR_ADD_RES_SUFFIX):
            rasterfile_dst_base, rasterfile_dst_ext = os.path.splitext(rasterfile_dst)
            rasterfile_dst = "{}_{}m{}".format(
                rasterfile_dst_base, args.get(ARGSTR_TARGET_RESOLUTION), rasterfile_dst_ext
            )
        rasterfile_dst_res = (target_resolution if target_resolution is not None else
                              raster_io.extractRasterData(rasterfile_src, 'res'))
        resample_method = None
        for suffix_key, suffix_method in SETSM_SUFFIX_KEY_TO_RESAMPLE_METHOD_LOOKUP:
            if suffix_key in rasterfile_src:
                resample_method = suffix_method
                break
        if resample_method is None:
            raise script_utils.DeveloperError(
                "No suffix key in `SETSM_SUFFIX_KEY_TO_RESAMPLE_METHOD_LOOKUP` matches "
                "source raster file: {}".format(rasterfile_src)
            )
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -tap -t_srs EPSG:{2} -tr {3} {3} -r {4} {5} '
             '-overwrite -co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(
                rasterfile_src, rasterfile_dst,
                target_epsg, rasterfile_dst_res, resample_method,
                '-dstnodata 1' if 'bitmask' in rasterfile_src else ''
            ))
        )
        if 'bitmask' in rasterfile_src:
            commands.append('gdal_edit.py -unsetnodata {}'.format(rasterfile_dst))

    for cmd in commands:
        print(cmd)
        if not args.get(ARGSTR_DRYRUN):
            script_utils.exec_cmd(cmd)

    metafile_src = src_metafile
    metafile_dst = os.path.join(dstdir, os.path.basename(metafile_src))
    
    if src_is_strip_metafile and normal_strip_output:
        demfile_dst = metafile_dst.replace(SETSM_META_SUFFIX, STRIP_DEM_SUFFIX)
        saveStripBrowse(args, demfile_dst, STRIP_DEM_SUFFIX, 'bitmask.tif')

    if src_is_strip_metafile:
        print("Reprojecting strip footprint vertices in metafile to EPSG:{}: {} -> {}".format(
            target_epsg, metafile_src, metafile_dst
        ))
        if not args.get(ARGSTR_DRYRUN):
            reproject_stripmeta(metafile_src, metafile_dst, target_epsg, args)
    else:
        print("Simply copying scene metafile: {} -> {}".format(
            metafile_src, metafile_dst
        ))
        if not args.get(ARGSTR_DRYRUN):
            shutil.copyfile(metafile_src, metafile_dst)


def reproject_stripmeta(metafile_src, metafile_dst, target_epsg, args):
    outmeta_txt = ""
    inmeta_fp = open(metafile_src, 'r')

    target_res = args.get(ARGSTR_TARGET_RESOLUTION)

    # Create output spatial reference and get
    # output projection in proj4 format.
    spatref_out = osr_srs_preserve_axis_order(osr.SpatialReference())
    spatref_out.ImportFromEPSG(target_epsg)
    proj4_out = spatref_out.ExportToProj4()

    # Get strip projection.
    line = inmeta_fp.readline()
    while not line.startswith('Strip projection (proj4):') and line != "":
        outmeta_txt += line
        line = inmeta_fp.readline()
    if line == "":
        logger.error("Projection string cannot be parsed from meta file: {}".format(metafile_src))
        inmeta_fp.close()
        return
    proj4_in = line.split("'")[1]
    outmeta_txt += "Strip projection (proj4): '{}'\n".format(proj4_out)

    # Get strip footprint geometry.
    line = inmeta_fp.readline()
    while not line.startswith('Strip Footprint Vertices') and line != "":
        outmeta_txt += line
        line = inmeta_fp.readline()
    if line == "":
        logger.error("Footprint vertices cannot be parsed from meta file: {}".format(metafile_src))
        inmeta_fp.close()
        return
    outmeta_txt += line

    if not args.get(ARGSTR_SIMPLE_META_FP_VERTS):
        ## Recalculate strip footprint vertices from reprojected strip DEM raster.
        from lib import raster_array_tools as rat
        for i in range(2):
            inmeta_fp.readline()
        demfile_dst = metafile_dst.replace(SETSM_META_SUFFIX, STRIP_DEM_SUFFIX)
        Z, Y, X = raster_io.extractRasterData(demfile_dst, 'z', 'y', 'x')
        fp_vertices = rat.getFPvertices(Z, X, Y, label=-9999, label_type='nodata',
                                        replicate_matlab=True, dtype_out_int64_if_equal=True)
        del Z, Y, X
        x_out, y_out = fp_vertices

    else:
        ## Reproject existing strip footprint vertices.
        line = inmeta_fp.readline()
        x_in = np.fromstring(line.replace('X:', '').strip(), dtype=np.float64, sep=' ')
        line = inmeta_fp.readline()
        y_in = np.fromstring(line.replace('Y:', '').strip(), dtype=np.float64, sep=' ')
        wkt_in = raster_io.coordsToWkt(np.array([x_in, y_in]).T)

        # Create input footprint geometry with spatial reference.
        geom = ogr.Geometry(wkt=wkt_in)
        spatref_in = osr_srs_preserve_axis_order(osr.SpatialReference())
        spatref_in.ImportFromProj4(proj4_in)
        geom.AssignSpatialReference(spatref_in)

        # Transform geometry to new spatial reference
        # and extract transformed coordinates.
        geom.TransformTo(spatref_out)
        wkt_out = geom.ExportToWkt()
        x_out, y_out = raster_io.wktToCoords(wkt_out).T
        # x_out = (np.around(x_out / target_res) * target_res).astype(np.int64)
        # y_out = (np.around(y_out / target_res) * target_res).astype(np.int64)

    outmeta_txt += "X: {}\n".format(' '.join(np.array_str(x_out, max_line_width=np.inf).strip()[1:-1].split()))
    outmeta_txt += "Y: {}\n".format(' '.join(np.array_str(y_out, max_line_width=np.inf).strip()[1:-1].split()))

    outmeta_txt += inmeta_fp.read()
    inmeta_fp.close()

    # Write output metadata file.
    outmeta_fp = open(metafile_dst, 'w')
    outmeta_fp.write(outmeta_txt)
    outmeta_fp.close()


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
    demFile_10m_masked = demFile.replace(demSuffix, 'dem_10m_masked.tif')
    demFile_10m_shade  = demFile.replace(demSuffix, 'dem_10m_shade.tif')
    demFile_shade_mask = demFile.replace(demSuffix, 'dem_10m_shade_masked.tif')
    demFile_40m_masked = demFile.replace(demSuffix, 'dem_40m_masked.tif')
    demFile_coverage   = demFile.replace(demSuffix, 'dem_40m_coverage.tif')

    output_files = set()
    keep_files = set()

    if not args.get(ARGSTR_STRIPS_NO_BROWSE):
        output_files.update([
            maskFile_10m,
            demFile_10m,
            demFile_10m_shade,
            demFile_shade_mask
        ])
        keep_files.update([
            demFile_10m,
            demFile_10m_shade,
            demFile_shade_mask
        ])

    if args.get(ARGSTR_STRIPS_BUILD_AUX):
        output_files.update([
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
             ' -co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(maskFile, maskFile_10m, 10))
        )
    if orthoFile_10m in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -r cubic -dstnodata 0'
             ' -co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(orthoFile, orthoFile_10m, 10))
        )
    if ortho2File_10m in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -r cubic -dstnodata 0'
             ' -co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(ortho2File, ortho2File_10m, 10))
        )
    if matchFile_10m in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -r near -dstnodata 0'
             ' -co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(matchFile, matchFile_10m, 10))
        )
    if demFile_10m in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -r bilinear -dstnodata -9999'
             ' -co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(demFile, demFile_10m, 10))
        )
    if demFile_10m_shade in output_files:
        commands.append(
            ('gdaldem hillshade "{0}" "{1}" -q -z 3 -compute_edges -of GTiff'
             ' -co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(demFile_10m, demFile_10m_shade))
        )
    if dem_browse in output_files:
        commands.append(
            ('gdaldem hillshade "{0}" "{1}" -q -z 3 -compute_edges -of GTiff'
             ' -co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(demFile_10m, dem_browse))
        )
    if demFile_10m_masked in output_files:
        commands.append(
            ('gdal_calc.py --quiet --overwrite -A "{0}" -B "{1}" --outfile="{2}" --calc="A*(B==0)+(-9999)*(B!=0)" --NoDataValue=-9999'
             ' --co TILED=YES --co BIGTIFF=IF_SAFER --co COMPRESS=LZW'.format(demFile_10m, maskFile_10m, demFile_10m_masked))
        )
    if demFile_shade_mask in output_files:
        commands.append(
            ('gdal_calc.py --quiet --overwrite -A "{0}" -B "{1}" --outfile="{2}" --calc="A*(B==0)" --NoDataValue=0'
             ' --co TILED=YES --co BIGTIFF=IF_SAFER --co COMPRESS=LZW'.format(demFile_10m_shade, maskFile_10m, demFile_shade_mask))
        )
    if demFile_40m_masked in output_files:
        commands.append(
            ('gdalwarp "{0}" "{1}" -q -overwrite -tr {2} {2} -tap -r bilinear -dstnodata -9999'
             ' -co TILED=YES -co BIGTIFF=IF_SAFER -co COMPRESS=LZW'.format(demFile_10m_masked, demFile_40m_masked, 40))
        )
    if demFile_coverage in output_files:
        commands.append(
            ('gdal_calc.py --quiet --overwrite -A "{0}" --outfile="{1}" --type Byte --calc="A!=-9999" --NoDataValue=0'
             ' --co TILED=YES --co BIGTIFF=IF_SAFER --co COMPRESS=LZW'.format(demFile_40m_masked, demFile_coverage))
        )

    for cmd in commands:
        print(cmd)
        if not args.get(ARGSTR_DRYRUN):
            script_utils.exec_cmd(cmd)

    if not args.get(ARGSTR_DRYRUN):
        for outfile in output_files:
            if not os.path.isfile(outfile):
                raise script_utils.ExternalError("External program call did not create output file: {}".format(outfile))
            if outfile not in keep_files:
                os.remove(outfile)



if __name__ == '__main__':
    main()
