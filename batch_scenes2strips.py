#!/usr/bin/env python2

# Version 3.0; Erik Husby, Claire Porter; Polar Geospatial Center, University of Minnesota; 2018


import argparse
import filecmp
import glob
import os
import subprocess
import sys
from time import sleep
from datetime import datetime

import numpy as np


# Script argument defaults
SCRIPT_FILE = os.path.realpath(__file__)
SCRIPT_FNAME = os.path.basename(SCRIPT_FILE)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE)
ARGDEF_QSUBSCRIPT = os.path.join(SCRIPT_DIR, 'qsub_scenes2strips.sh')

SUFFIX_PRIORITY_DEM = ['dem_smooth.tif', 'dem.tif']
SUFFIX_PRIORITY_MATCHTAG = ['matchtag_mt.tif', 'matchtag.tif']


class MetaReadError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


def main():
    from lib.filter_scene import generateMasks
    from lib.filter_scene import MASK_FLAT, MASK_SEPARATE, MASK_BIT
    from lib.scenes2strips import scenes2strips
    from lib.raster_array_tools import saveArrayAsTiff, getFPvertices

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=(
        "Filters scene dems in a source directory, "
        "then mosaics them into strips and saves the results.\n"
        "Batch work is done in units of strip ID (<catid1_catid2>), as parsed from scene dem "
        "filenames."))

    parser.add_argument('src',
        help="Path to source directory containing scene DEMs to process. "
             "If --dst is not specified, this path should contain the folder 'tif_results'.")
    parser.add_argument('res', type=int, choices=[2, 8],
        help="Resolution of target DEMs (2 or 8 meters).")

    parser.add_argument('--dst',
        help="Path to destination directory for output mosaicked strip data."
             " (default is src.(reverse)replace('tif_results', 'strips'))")

    parser.add_argument('--meta-trans-dir',
        help="Path to directory of old strip metadata from which translation values "
             "will be parsed to skip scene coregistration step.")

    parser.add_argument('--mask-ver', choices=['maskv1','maskv2','rema2a','mask8m','bitmask'],
        default='bitmask',
        help="Filtering scheme to use when generating mask raster images "
             "to classify bad data in scene DEMs. "
             "\n'maskv1': Two-component (edge, data density) filter to create "
                         "separate edgemask and datamask files for each scene. "
             "\n'maskv2': Three-component (edge, water, cloud) filter to create "
                         "classical 'flat' binary masks for 2m DEMs."
             "\n'bitmask': Same filter as 'maskv2', but distinguish between "
                          "the different filter components by creating a bitmask."
             "\n'rema2a': Filter designed specifically for 8m Antarctic DEMs. "
             "\n'mask8m': General-purpose filter for 8m DEMs.")

    parser.add_argument('--noentropy', action='store_true', default=False,
        help="Use filter without entropy protection. "
             "Can only be used when --mask-ver='maskv1'.")
    parser.add_argument('--nowater', action='store_true', default=False,
        help="Use filter without water masking. "
             "Can only be used when --mask-ver='bitmask'.")
    parser.add_argument('--nocloud', action='store_true', default=False,
        help="Use filter without cloud masking. "
             "Can only be used when --mask-ver='bitmask'.")
    parser.add_argument('--nofilter-coreg', action='store_true', default=False,
        help="If --nowater/--nocloud, turn off filter(s) during "
             "scene coregistration step in addition to mosaicking step. "
             "Can only be used when --mask-ver='bitmask'.")

    parser.add_argument('--pbs', action='store_true', default=False,
        help="Submit tasks to PBS.")
    parser.add_argument('--qsubscript', default=ARGDEF_QSUBSCRIPT,
        help="Path to qsub script to use in PBS submission."
             " (default={})".format(ARGDEF_QSUBSCRIPT))
    parser.add_argument('--dryrun', action='store_true', default=False,
        help="Print actions without executing.")

    parser.add_argument('--stripid',
        help="Run filtering and mosaicking for a single strip with id "
             "<catid1_catid2> (as parsed from scene DEM filenames).")

    # Parse and validate arguments.
    args = parser.parse_args()
    scriptpath = os.path.abspath(sys.argv[0])
    srcdir = os.path.abspath(args.src)
    dstdir = args.dst
    metadir = args.meta_trans_dir
    qsubpath = os.path.abspath(args.qsubscript)

    if args.res == 2 and args.mask_ver not in ('maskv1', 'maskv2', 'bitmask'):
        parser.error("--mask-ver must be one of ('maskv1', 'maskv2', or 'bitmask') for 2-meter `res`")
    if args.res == 8 and args.mask_ver not in ('maskv1', 'rema2a', 'mask8m'):
        parser.error("--mask-ver must be one of ('maskv1', 'rema2a', or 'mask8m') for 8-meter `res`")
    if args.noentropy and args.mask_ver != 'maskv1':
        parser.error("--noentropy option is compatible only with --maskv1 option")
    if (args.nowater or args.nocloud) and args.mask_ver != 'bitmask':
        parser.error("--nowater/--nocloud can only be used when --mask-ver='bitmask'")
    if args.nofilter_coreg and [args.nowater, args.nocloud].count(True) == 0:
        parser.error("--nofilter-coreg option must be used in conjunction with --nowater/--nocloud option(s)")
    if args.nofilter_coreg and args.meta_trans_dir is not None:
        parser.error("--nofilter-coreg option cannot be used in conjunction with --meta-trans-dir argument")

    if not os.path.isdir(srcdir):
        parser.error("`src` must be a directory")

    if dstdir is not None:
        dstdir = os.path.abspath(dstdir)
        if os.path.isdir(dstdir) and filecmp.cmp(srcdir, dstdir):
            parser.error("`src` dir is the same as --dst dir: {}".format(srcdir))
    else:
        # Set default dst dir.
        split_ind = srcdir.rfind('tif_results')
        if split_ind == -1:
            parser.error("`src` path does not contain 'tif_results', so default --dst cannot be set")
        dstdir = srcdir[:split_ind] + srcdir[split_ind:].replace('tif_results', 'strips')
        print("--dst dir set to: {}".format(dstdir))

    if metadir is not None:
        if os.path.isdir(metadir):
            metadir = os.path.abspath(metadir)
        else:
            parser.error("--meta-trans-dir must be a directory")

    if not os.path.isfile(qsubpath):
        parser.error("--qsubscript path is not a valid file path: {}".format(qsubpath))

    # Create strip output directory if it doesn't already exist.
    if not os.path.isdir(dstdir):
        if not args.dryrun:
            os.makedirs(dstdir)


    if args.stripid is None:
        # Do batch processing.


        # Find all scene DEMs to be merged into strips.
        for demSuffix in SUFFIX_PRIORITY_DEM:
            scene_dems = glob.glob(os.path.join(srcdir, '*_{}_{}'.format(args.res, demSuffix)))
            if scene_dems:
                break
        if not scene_dems:
            print("No scene DEMs found to merge, exiting")
            sys.exit(1)

        # Find unique strip IDs (<catid1_catid2>).
        stripids = list(set([os.path.basename(s)[14:47] for s in scene_dems]))
        stripids.sort()
        print("{} {} pair ids".format(len(stripids), '*'+demSuffix))
        del scene_dems

        wait_seconds = 5
        print("Sleeping {} seconds before job submission".format(wait_seconds))
        sleep(wait_seconds)

        # Process each strip ID.
        i = 0
        for stripid in stripids:
            i += 1

            # If output does not already exist, add to task list.
            dst_dems = glob.glob(os.path.join(dstdir, '*{}_seg*_{}m_{}'.format(stripid, args.res, demSuffix)))
            if dst_dems:
                print("--stripid {} output files exist, skipping".format(stripid))
                continue

            # If PBS, submit to scheduler.
            if args.pbs:
                job_name = 's2s{:04g}'.format(i)
                cmd = r'qsub -N {0} -v p1={1},p2={2},p3={3},p4={4},p5={5},p6={6}{7}{8}{9}{10}{11} {12}'.format(
                    job_name,
                    scriptpath,
                    srcdir,
                    args.res,
                    '"--dst {}"'.format(dstdir),
                    '"--stripid {}"'.format(stripid),
                    '"--mask-ver {}"'.format(args.mask_ver),
                    ',p7="--meta-trans-dir {}"'.format(metadir) if metadir is not None else '',
                    ',p8=--noentropy' * args.noentropy,
                    ',p9=--nowater' * args.nowater,
                    ',p10=--nocloud' * args.nocloud,
                    ',p11=--nofilter-coreg' * args.nofilter_coreg,
                    qsubpath
                )
                print(cmd)

            # ...else run Python.
            else:
                cmd = r'{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}'.format(
                    'python',
                    scriptpath,
                    srcdir,
                    args.res,
                    '--dst {}'.format(dstdir),
                    '--stripid {}'.format(stripid),
                    '--mask-ver {}'.format(args.mask_ver),
                    '--meta-trans-dir {}'.format(metadir) if metadir is not None else '',
                    '--noentropy' * args.noentropy,
                    '--nowater' * args.nowater,
                    '--nocloud' * args.nocloud,
                    '--nofilter-coreg' * args.nofilter_coreg,
                )
                print('{}, {}'.format(i, cmd))

            if not args.dryrun:
                # For most cases, set `shell=True`.
                # For attaching process to PyCharm debugger,
                # set `shell=False`.
                subprocess.call(cmd, shell=True)


    else:
        # Process a single strip.


        # Handle arguments.

        metadir = args.meta_trans_dir
        use_old_trans = True if metadir is not None else False

        filter_options_mask = ()
        if args.nowater:
            filter_options_mask += ('nowater',)
        if args.nocloud:
            filter_options_mask += ('nocloud',)
        filter_options_coreg = filter_options_mask if args.nofilter_coreg else ()

        if args.mask_ver == 'maskv2':
            mask_version = 'mask'
        else:
            mask_version = args.mask_ver

        # Print arguments for this run.
        print("res: {}m".format(args.res))
        print("srcdir: {}".format(srcdir))
        print("dstdir: {}".format(dstdir))
        print("metadir: {}".format(metadir))
        print("mask version: {}".format(mask_version))
        print("coreg filter options: {}".format(filter_options_coreg))
        print("mask filter options: {}".format(filter_options_mask))

        # Find scene DEMs for this stripid to be merged into strips.
        for demSuffix in SUFFIX_PRIORITY_DEM:
            scene_demFiles = glob.glob(os.path.join(srcdir, '*{}*_{}_{}'.format(args.stripid, args.res, demSuffix)))
            if scene_demFiles:
                break
        print("Merging pair id: {}, {} scenes".format(args.stripid, len(scene_demFiles)))
        if not scene_demFiles:
            print("No scene DEMs found to merge, skipping")
            sys.exit(1)

        # Existence check. If output already exists, skip.
        strip_demFiles = glob.glob(os.path.join(dstdir, '*{}_seg*_{}m_{}'.format(args.stripid, args.res, demSuffix)))
        if strip_demFiles:
            print("Output files exist, skipping")
            sys.exit(0)

        # Make sure all matchtag and ortho files exist. If missing, skip.
        missingflag = False
        for f in scene_demFiles:
            if selectBestMatchtag(f) is None:
                print("matchtag file for {} missing, skipping".format(f))
                missingflag = True
            if not os.path.isfile(f.replace(demSuffix, 'ortho.tif')):
                print("ortho file for {} missing, skipping".format(f))
                missingflag = True
            if not os.path.isfile(f.replace(demSuffix, 'meta.txt')):
                print("meta file for {} missing, skipping".format(f))
                missingflag = True
        if missingflag:
            sys.exit(1)

        # Filter all scenes in this strip.
        filter_list = [f for f in scene_demFiles if shouldDoMasking(selectBestMatchtag(f), mask_version)]
        filter_total = len(filter_list)
        i = 0
        for demFile in filter_list:
            i += 1
            sys.stdout.write("Filtering {} of {}: ".format(i, filter_total))
            generateMasks(demFile, mask_version, noentropy=args.noentropy)

        # Mosaic scenes in this strip together.
        # Output separate segments if there are breaks in overlap.
        maskSuffix = mask_version+'.tif'
        remaining_sceneDemFnames = [os.path.basename(f) for f in scene_demFiles]
        segnum = 1
        while len(remaining_sceneDemFnames) > 0:

            print("Building segment {}".format(segnum))

            stripid_full = remaining_sceneDemFnames[0][0:47]
            strip_demFname = "{}_seg{}_{}m_{}".format(stripid_full, segnum, args.res, demSuffix)
            strip_demFile = os.path.join(dstdir, strip_demFname)
            if use_old_trans:
                strip_metaFile = os.path.join(metadir, strip_demFname.replace(demSuffix, 'meta.txt'))
                mosaicked_sceneDemFnames, rmse, trans = readStripMeta_stats(strip_metaFile)
                if not set(mosaicked_sceneDemFnames).issubset(set(remaining_sceneDemFnames)):
                    print("Current source DEMs do not include source DEMs referenced in old strip meta file")
                    use_old_trans = False

            all_data_masked = False
            if not use_old_trans:
                print("Running s2s with coregistration filter options: {}".format(', '.join(filter_options_coreg)))
                X, Y, Z, M, O, MD, trans, rmse, mosaicked_sceneDemFnames, spat_ref = scenes2strips(
                    srcdir, remaining_sceneDemFnames, maskSuffix, filter_options_coreg)
                if X is None:
                    all_data_masked = True

            if not all_data_masked and filter_options_mask != filter_options_coreg or use_old_trans:
                print("Running s2s with masking filter options: {}".format(', '.join(filter_options_mask)))
                input_sceneDemFnames = mosaicked_sceneDemFnames
                # Set `rmse_guess=rmse` in the following call of `scenes2strips` to get stats on the
                # difference between RMSE values in masked versus unmasked coregistration output to
                # `os.path.join(testing.test.TESTDIR, 's2s_stats.log')`.
                X, Y, Z, M, O, MD, trans, rmse, mosaicked_sceneDemFnames, spat_ref = scenes2strips(
                    srcdir, input_sceneDemFnames, maskSuffix, filter_options_mask,
                    trans_guess=trans, hold_guess=True)
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

            strip_matchFile = strip_demFile.replace(demSuffix, 'matchtag.tif')
            strip_maskFile  = strip_demFile.replace(demSuffix, maskSuffix)
            strip_orthoFile = strip_demFile.replace(demSuffix, 'ortho.tif')
            strip_metaFile  = strip_demFile.replace(demSuffix, 'meta.txt')

            saveArrayAsTiff(Z, strip_demFile,   X, Y, spat_ref, nodata_val=-9999, dtype_out='float32')
            saveArrayAsTiff(M, strip_matchFile, X, Y, spat_ref, nodata_val=0,     dtype_out='uint8')
            del M
            saveArrayAsTiff(O, strip_orthoFile, X, Y, spat_ref, nodata_val=0,     dtype_out='int16')
            del O
            saveArrayAsTiff(MD, strip_maskFile, X, Y, spat_ref, nodata_val=0,     dtype_out='uint8')
            del MD

            fp_vertices = getFPvertices(Z, Y, X, label=-9999, label_type='nodata', replicate_matlab=True)
            del Z, X, Y

            proj4 = spat_ref.ExportToProj4()
            time = datetime.today().strftime("%d-%b-%Y %H:%M:%S")
            writeStripMeta(strip_metaFile, srcdir, mosaicked_sceneDemFnames,
                           trans, rmse, proj4, fp_vertices, time, args)

            segnum += 1


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


def shouldDoMasking(matchFile, mask_version='mask'):
    matchFile_date = os.path.getmtime(matchFile)
    demFile_base = matchFile.replace(getMatchtagSuffix(matchFile), '')
    maskFiles = (     [demFile_base+s for s in ('edgemask.tif', 'datamask.tif')] if mask_version == 'maskv1'
                 else ['{}{}.tif'.format(demFile_base, mask_version)])
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

    demSuffix = getDemSuffix(scene_demFnames[0])
    if fp_vertices.dtype != np.int64 and np.array_equal(fp_vertices, fp_vertices.astype(np.int64)):
        fp_vertices = fp_vertices.astype(np.int64)

    strip_info = (
"""Strip Metadata 
Creation Date: {}
Strip creation date: {}
Strip projection (proj4): '{}'

Strip Footprint Vertices
X: {} 
Y: {} 

Mosaicking Alignment Statistics (meters) 
scene, rmse, dz, dx, dy
""".format(
    datetime.today().strftime("%d-%b-%Y %H:%M:%S"),
    strip_time,
    proj4,
    ' '.join(np.array_str(fp_vertices[1], max_line_width=float('inf')).strip()[1:-1].split()),
    ' '.join(np.array_str(fp_vertices[0], max_line_width=float('inf')).strip()[1:-1].split()),
)
    )

    for i in range(len(scene_demFnames)):
        line = "{} {:.2f} {:.4f} {:.4f} {:.4f}\n".format(
            scene_demFnames[i], rmse[0, i], trans[0, i], trans[1, i], trans[2, i])
        strip_info += line

        filter_info = "\nFiltering Applied: {} (v3.1)\n".format(args.mask_ver)

    if args.mask_ver == 'bitmask':
        filter_info += "bit, class, coreg, mosaic\n"
        filter_info_components = (
"""
{} edge 1 1
{} water {} {}
{} cloud {} {}
""".format(
        MASKCOMP_EDGE_BIT,
        MASKCOMP_WATER_BIT, int(not args.nofilter_coreg*args.nowater), int(not args.nowater),
        MASKCOMP_CLOUD_BIT, int(not args.nofilter_coreg*args.nocloud), int(not args.nocloud),
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

    strip_metaFile_fp = open(o_metaFile, 'w')
    strip_metaFile_fp.write(strip_info)
    strip_metaFile_fp.write(scene_info)
    strip_metaFile_fp.close()


def readStripMeta_stats(metaFile):
    metaFile_fp = open(metaFile, 'r')
    try:
        line = metaFile_fp.readline()
        while not line.startswith('Mosaicking Alignment Statistics (meters)') and line != '':
            line = metaFile_fp.readline()
        if line == '':
            raise MetaReadError("{}: Could not parse 'Mosaicking Alignment Statistics'".format(metaFile))
        line = metaFile_fp.readline().strip()

        line_items = line.split(' ')
        sceneDemFnames = [line_items[0]]
        rmse = [line_items[1]]
        trans = np.array([[float(s) for s in line_items[2:5]]])

        line = metaFile_fp.readline().strip()
        while line != '':
            line_items = line.split(' ')
            sceneDemFnames.append(line_items[0])
            rmse.append(line_items[1])
            trans = np.vstack((trans, np.array([[float(s) for s in line_items[2:5]]])))
            line = metaFile_fp.readline().strip()

        rmse = np.array([[float(s) for s in rmse]])
        trans = trans.T

    finally:
        metaFile_fp.close()

    return sceneDemFnames, rmse, trans



if __name__ == '__main__':
    main()
