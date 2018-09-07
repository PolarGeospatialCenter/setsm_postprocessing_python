#!/usr/bin/env python2

# Version 3.0; Erik Husby, Claire Porter; Polar Geospatial Center, University of Minnesota; 2018


import argparse
import filecmp
import glob
import os
import subprocess
import sys
from datetime import datetime

import numpy as np


class MetaReadError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


def main():
    from lib.filter_scene import generateMasks
    from lib.filter_scene import MASK_FLAT, MASK_SEPARATE, MASK_BIT
    from lib.scenes2strips import scenes2strips
    from lib.raster_array_tools import saveArrayAsTiff, getFPvertices

    parser = argparse.ArgumentParser(description=(
        "Filters scene dems in a source directory, "
        "then mosaics them into strips and saves the results.\n"
        "Batch work is done in units of strip ID (<catid1_catid2>), as parsed from scene dem "
        "filenames."))

    parser.add_argument('src',
        help=("Path to source directory containing scene dems to process."
              " If --dst is not specified, this path should contain the substring 'tif_results'."))
    parser.add_argument('res', choices=['2', '8'],
        help="Resolution of target dems (2 or 8 meters).")

    parser.add_argument('--dst',
        help="Path to destination directory for output mosaicked strip data "
             "(default is src.(reverse)replace('tif_results', 'strips')).")

    parser.add_argument('--meta-trans-dir',
        help="Path to directory of old strip metadata from which translation values "
             "will be parsed to skip scene coregistration step.")

    parser.add_argument('--nowater', action='store_true', default=False,
        help="Use filter without water masking.")
    parser.add_argument('--nocloud', action='store_true', default=False,
        help="Use filter without cloud masking.")
    parser.add_argument('--nofilter-coreg', action='store_true', default=False,
        help=("If --nowater/--nocloud, turn off filter(s) during "
              "scene coregistration step in addition to mosaicking step."))

    parser.add_argument('--maskv1', action='store_true', default=False,
        help="Use two-mask filter with edgemask and datamask.")
    parser.add_argument('--noentropy', action='store_true', default=False,
        help="Use filter without entropy protection.")

    parser.add_argument('--rema2a', action='store_true', default=False,
        help="Use rema2a filter.")

    parser.add_argument('--mask8m', action='store_true', default=False,
        help="Use mask8m filter.")

    parser.add_argument('--pbs', action='store_true', default=False,
        help="Submit tasks to PBS.")
    parser.add_argument('--qsubscript',
        help="Path to qsub script to use in PBS submission "
             "(default is qsub_scenes2strips.sh in script root folder).")
    parser.add_argument('--dryrun', action='store_true', default=False,
        help="Print actions without executing.")

    parser.add_argument('--stripid',
        help="Run filtering and mosaicking for a single strip with id "
             "<catid1_catid2> (as parsed from scene dem filenames).")

    # Parse and validate arguments.
    args = parser.parse_args()
    scriptpath = os.path.abspath(sys.argv[0])
    scriptdir = os.path.dirname(scriptpath)
    srcdir = os.path.abspath(args.src)
    dstdir = args.dst
    metadir = args.meta_trans_dir
    qsubpath = args.qsubscript

    if (args.nowater or args.nocloud) and [args.maskv1, args.rema2a, args.mask8m].count(True) > 0:
        parser.error("--nowater/--nocloud filter options are not applicable with special mask options")
    if args.nofilter_coreg and [args.nowater, args.nocloud].count(True) == 0:
        parser.error("--nofilter-coreg option must be used in conjunction with --nowater/--nocloud option(s)")
    if args.nofilter_coreg and args.meta_trans_dir is not None:
        parser.error("--nofilter-coreg option cannot be used in conjunction with --meta-trans-dir argument")
    if [args.maskv1, args.rema2a, args.mask8m].count(True) > 1:
        parser.error("Only one of the following masking options is allowed: (--maskv1, --rema2a, --mask8m)")
    if args.noentropy and not args.maskv1:
        parser.error("--noentropy option is compatible only with --maskv1 option")

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

    if qsubpath is not None:
        if os.path.isfile(qsubpath):
            qsubpath = os.path.abspath(qsubpath)
        else:
            parser.error("--qsubscript path is not a valid file path: {}".format(qsubpath))
    else:
        # Set default qsubpath.
        qsubpath = os.path.abspath(os.path.join(scriptdir, 'qsub_scenes2strips.sh'))

    # Create strip output directory if it doesn't already exist.
    if not os.path.isdir(dstdir):
        if not args.dryrun:
            os.makedirs(dstdir)


    if args.stripid is None:
        # Do batch processing.


        # Find all scene DEMs to be merged into strips.
        scene_dems = glob.glob(os.path.join(srcdir, '*dem.tif'))
        if not scene_dems:
            print("No scene dems found to merge, exiting")
            sys.exit(1)

        # Find unique strip IDs (<catid1_catid2>).
        stripids = list(set([os.path.basename(s)[14:47] for s in scene_dems]))
        stripids.sort()
        print("{} pair ids".format(len(stripids)))

        del scene_dems

        # Process each strip ID.
        i = 0
        for stripid in stripids:
            i += 1

            # If output does not already exist, add to task list.
            dst_dems = glob.glob(os.path.join(dstdir, '*'+stripid+'_seg*_dem.tif'))
            if dst_dems:
                print("--stripid {} output files exist, skipping".format(stripid))
                continue

            # If PBS, submit to scheduler.
            if args.pbs:
                job_name = 's2s{:04g}'.format(i)
                cmd = r'qsub -N {0} -v p1={1},p2={2},p3={3},p4={4},p5={5}{6}{7}{8}{9}{10}{11}{12}{13} {14}'.format(
                    job_name,
                    scriptpath,
                    srcdir,
                    args.res,
                    '"--dst {}"'.format(dstdir),
                    '"--stripid {}"'.format(stripid),
                    ',p6="--meta-trans-dir {}"'.format(metadir) if metadir is not None else '',
                    ',p7=--nowater' * args.nowater,
                    ',p8=--nocloud' * args.nocloud,
                    ',p9=--nofilter-coreg' * args.nofilter_coreg,
                    ',p10=--maskv1' * args.maskv1,
                    ',p11=--noentropy' * args.noentropy,
                    ',p12=--rema2a' * args.rema2a,
                    ',p13=--mask8m' * args.mask8m,
                    qsubpath
                )
                print(cmd)

            # ...else run Python.
            else:
                cmd = r'{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}'.format(
                    'python',
                    scriptpath,
                    srcdir,
                    args.res,
                    '--dst {}'.format(dstdir),
                    '--stripid {}'.format(stripid),
                    '--meta-trans-dir {}'.format(metadir) if metadir is not None else '',
                    '--nowater' * args.nowater,
                    '--nocloud' * args.nocloud,
                    '--nofilter-coreg' * args.nofilter_coreg,
                    '--maskv1' * args.maskv1,
                    '--noentropy' * args.noentropy,
                    '--rema2a' * args.rema2a,
                    '--mask8m' * args.mask8m,
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

        if args.maskv1:
            maskFileSuffix = 'edgemask/datamask'
        elif args.rema2a:
            maskFileSuffix = 'mask2a'
        elif args.mask8m:
            maskFileSuffix = 'mask8m'
        else:
            maskFileSuffix = 'mask'

        # Print arguments for this run.
        print("res: {}m".format(args.res))
        print("srcdir: {}".format(srcdir))
        print("dstdir: {}".format(dstdir))
        print("metadir: {}".format(metadir))
        print("maskFileSuffix: {}".format(maskFileSuffix))
        print("coreg filter options: {}".format(filter_options_coreg))
        print("mask filter options: {}".format(filter_options_mask))

        # Find scene DEMs for this stripid to be merged into strips.
        scene_dems = glob.glob(os.path.join(srcdir, '*'+args.stripid+'*_dem.tif'))
        print("Merging pair id: {}, {} scenes".format(args.stripid, len(scene_dems)))
        if not scene_dems:
            print("No scene DEMs found to merge, skipping")
            sys.exit(1)

        # Existence check. If output already exists, skip.
        dst_dems = glob.glob(os.path.join(dstdir, '*'+args.stripid+'_seg*_dem.tif'))
        if dst_dems:
            print("Output files exist, skipping")
            sys.exit(0)

        # Make sure all matchtag and ortho files exist. If missing, skip.
        missingflag = False
        for f in scene_dems:
            if selectBestMatchtag(f) is None:
                print("matchtag file for {} missing, skipping".format(f))
                missingflag = True
            if not os.path.isfile(f.replace('dem.tif', 'ortho.tif')):
                print("ortho file for {} missing, skipping".format(f))
                missingflag = True
            if not os.path.isfile(f.replace('dem.tif', 'meta.txt')):
                print("meta file for {} missing, skipping".format(f))
                missingflag = True
        if missingflag:
            sys.exit(1)

        # Filter all scenes in this strip.
        filter_list = [f for f in scene_dems if shouldDoMasking(selectBestMatchtag(f), maskFileSuffix)]
        filter_total = len(filter_list)
        i = 0
        for demFile in filter_list:
            i += 1
            sys.stdout.write("Filtering {} of {}: ".format(i, filter_total))
            generateMasks(demFile, maskFileSuffix, noentropy=args.noentropy,
                          save_component_masks=MASK_BIT,
                          nbit_masks=False)

        # Mosaic scenes in this strip together.
        # Output separate segments if there are breaks in overlap.
        remaining_sceneDemFnames = [os.path.basename(f) for f in scene_dems]
        segnum = 1
        while len(remaining_sceneDemFnames) > 0:

            print("Building segment {}".format(segnum))

            stripid_full = remaining_sceneDemFnames[0][0:47]
            strip_demFname = "{}_seg{}_{}m_dem.tif".format(stripid_full, segnum, args.res)
            strip_demFile = os.path.join(dstdir, strip_demFname)
            if use_old_trans:
                strip_metaFile = os.path.join(metadir, strip_demFname.replace('dem.tif', 'meta.txt'))
                mosaicked_sceneDemFnames, rmse, trans = readStripMeta_stats(strip_metaFile)
                if not set(mosaicked_sceneDemFnames).issubset(set(remaining_sceneDemFnames)):
                    print("Current source DEMs do not include source DEMs referenced in old strip meta file")
                    use_old_trans = False

            if not use_old_trans:
                print("Running s2s with coregistration filter options: {}".format(', '.join(filter_options_coreg)))
                X, Y, Z, M, O, MD, trans, rmse, mosaicked_sceneDemFnames, spat_ref = scenes2strips(
                    srcdir, remaining_sceneDemFnames, maskFileSuffix, filter_options_coreg)

            if filter_options_mask != filter_options_coreg or use_old_trans:
                print("Running s2s with masking filter options: {}".format(', '.join(filter_options_mask)))
                input_sceneDemFnames = mosaicked_sceneDemFnames
                # TODO: Remove `rmse` argument once testing is complete.
                X, Y, Z, M, O, MD, trans, rmse, mosaicked_sceneDemFnames, spat_ref = scenes2strips(
                    srcdir, input_sceneDemFnames, maskFileSuffix, filter_options_mask, trans, rmse)
                if mosaicked_sceneDemFnames != input_sceneDemFnames and use_old_trans:
                    print("Current strip segmentation does not match that found in old strip meta file")
                    print("Rerunning s2s to get new coregistration translation values")
                    use_old_trans = False
                    continue

            remaining_sceneDemFnames = list(set(remaining_sceneDemFnames).difference(set(mosaicked_sceneDemFnames)))
            if X is None:
                continue

            print("DEM: {}".format(strip_demFile))

            strip_matchFile = strip_demFile.replace('dem.tif', 'matchtag.tif')
            strip_maskFile  = strip_demFile.replace('dem.tif', 'mask.tif')
            strip_orthoFile = strip_demFile.replace('dem.tif', 'ortho.tif')
            strip_metaFile  = strip_demFile.replace('dem.tif', 'meta.txt')

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


def selectBestMatchtag(demFile):
    matchFile = demFile.replace('dem.tif', 'matchtag_mt.tif')
    if not os.path.isfile(matchFile):
        matchFile = demFile.replace('dem.tif', 'matchtag.tif')
        if not os.path.isfile(matchFile):
            matchFile = None
    return matchFile


def shouldDoMasking(matchFile, maskFileSuffix='mask'):
    matchFile_date = os.path.getmtime(matchFile)
    demFile_base = matchFile.replace('matchtag_mt.tif', '').replace('matchtag.tif', '')
    maskFiles = (     [demFile_base+s for s in ('edgemask.tif', 'datamask.tif')] if maskFileSuffix == 'edgemask/datamask'
                 else ['{}{}.tif'.format(demFile_base, maskFileSuffix)])
    for m in maskFiles:
        if os.path.isfile(m):
            # Update Mode - will only reprocess masks older than the matchtag file.
            maskFile_date = os.path.getmtime(m)
            if (matchFile_date - maskFile_date) > 6.9444e-04:
                return True
        else:
            return True
    return False


def writeStripMeta(o_metaFile, scenedir, dem_list,
                   trans, rmse, proj4, fp_vertices, strip_time, args):
    from lib.filter_scene import MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT

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

    for i in range(len(dem_list)):
        line = "{} {:.2f} {:.4f} {:.4f} {:.4f}\n".format(
            dem_list[i], rmse[0, i], trans[0, i], trans[1, i], trans[2, i])
        strip_info += line

#     strip_info += (
# """
# Filtering Applied
# bit, class, coreg, mosaic
# {} edge 1 1
# {} water {} {}
# {} cloud {} {}
# """.format(
#     MASKCOMP_EDGE_BIT,
#     MASKCOMP_WATER_BIT, int(args.nofilter_coreg*args.nowater), int(args.nowater),
#     MASKCOMP_CLOUD_BIT, int(args.nofilter_coreg*args.nocloud), int(args.nocloud),
# )
#     )

    strip_info += "\nScene Metadata \n\n"

    scene_info = ""
    for i in range(len(dem_list)):
        scene_info += "scene {} name={}\n".format(i+1, dem_list[i])

        scene_metaFile = os.path.join(scenedir, dem_list[i].replace('dem.tif', 'meta.txt'))
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
        demFnames = [line_items[0]]
        rmse = [line_items[1]]
        trans = np.array([[float(s) for s in line_items[2:5]]])

        line = metaFile_fp.readline().strip()
        while line != '':
            line_items = line.split(' ')
            demFnames.append(line_items[0])
            rmse.append(line_items[1])
            trans = np.vstack((trans, np.array([[float(s) for s in line_items[2:5]]])))
            line = metaFile_fp.readline().strip()

        rmse = np.array([[float(s) for s in rmse]])
        trans = trans.T

    finally:
        metaFile_fp.close()

    return demFnames, rmse, trans



if __name__ == '__main__':
    main()
