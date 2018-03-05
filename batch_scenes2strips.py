#!/usr/bin/env python2

# Version 3.0; Erik Husby, Claire Porter; Polar Geospatial Center, University of Minnesota; 2018


import argparse
import filecmp
import glob
import os
import subprocess
import sys
from datetime import datetime

from numpy import array_equal, array_str, int64

from lib.filter_scene import generateMasks
from lib.raster_array_tools import saveArrayAsTiff, getFPvertices
from lib.scenes2strips import scenes2strips


def main():
    parser = argparse.ArgumentParser(description=(
        "Filters scene dems in a source directory, "
        "then mosaics them into strips and saves the results."
        "\nBatch work is done in units of strip ID (<catid1_catid2>), as parsed from scene dem "
        "filenames."))

    parser.add_argument('src',
        help=("Path to source directory containing scene dems to process."
              " If --dst is not specified, this path should contain the substring 'tif_results'."))
    parser.add_argument('res', choices=['2', '8'],
        help="Resolution of target dems (2 or 8 meters).")

    parser.add_argument('--dst',
        help="Path to destination directory for output mosaicked strip data "
             "(default is src.(reverse)replace('tif_results', 'strips')).")

    parser.add_argument('--edgemask', action='store_true', default=False,
        help="Use two-mask filter with edgemask and datamask.")
    parser.add_argument('--noentropy', action='store_true', default=False,
        help="Use filter without entropy protection.")
    parser.add_argument('--rema2a', action='store_true', default=False,
        help="Use filter rema2a.")

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
    qsubpath = args.qsubscript

    if not os.path.isdir(srcdir):
        parser.error("src must be a directory")

    if dstdir is not None:
        if os.path.isdir(srcdir):
            dstdir = os.path.abspath(dstdir)
            if filecmp.cmp(srcdir, dstdir):
                parser.error("src dir is the same as dst dir: {}".format(srcdir))
        else:
            parser.error("dst must be a directory")
    else:
        # Set default dst dir.
        split_ind = srcdir.rfind('tif_results')
        if split_ind == -1:
            parser.error("src path does not contain 'tif_results', so default dst cannot be set")
        dstdir = srcdir[:split_ind] + srcdir[split_ind:].replace('tif_results', 'strips')
        print "dst dir set to: {}".format(dstdir)

    if qsubpath is not None:
        if os.path.isfile(qsubpath):
            qsubpath = os.path.abspath(qsubpath)
        else:
            parser.error("qsubscript path is not a valid file path: {}".format(qsubpath))
    else:
        # Set default qsubpath.
        qsubpath = os.path.abspath(os.path.join(scriptdir, 'qsub_scenes2strips.sh'))

    if args.rema2a and (args.edgemask or args.noentropy):
        parser.error("rema2a and (edgemask or noentropy) filters are incompatible")

    # Create strip output directory if it doesn't already exist.
    if not os.path.isdir(dstdir):
        if not args.dryrun:
            os.makedirs(dstdir)

    if args.stripid is None:
        # Do batch processing.

        # Find all scene dems to be merged into strips.
        scene_dems = glob.glob(os.path.join(srcdir, '*dem.tif'))
        if not scene_dems:
            print "No scene dems found to merge. Exiting."
            sys.exit(1)

        # Find unique strip IDs (<catid1_catid2>).
        stripids = list(set([os.path.basename(s)[14:47] for s in scene_dems]))
        stripids.sort()
        print "{} pair ids".format(len(stripids))

        del scene_dems

        # Process each strip ID.
        i = 0
        for stripid in stripids:
            i += 1

            # If output does not already exist, add to task list.
            dst_dems = glob.glob(os.path.join(dstdir, '*'+stripid+'_seg*_dem.tif'))
            if dst_dems:
                print "{} output files exist, skipping".format(stripid)
                continue

            # If PBS, submit to scheduler.
            if args.pbs:
                job_name = 's2s{:04g}'.format(i)
                cmd = r'qsub -N {0} -v p1={1},p2={2},p3={3},p4={4},p5={5}{6}{7}{8} {9}'.format(
                    job_name,
                    scriptpath,
                    srcdir,
                    args.res,
                    '"--dst {}"'.format(dstdir),
                    '"--stripid {}"'.format(stripid),
                    ',p6=--edgemask' * args.edgemask,
                    ',p7=--noentropy' * args.noentropy,
                    ',p8=--rema2a' * args.rema2a,
                    qsubpath
                )
                print cmd

            # ...else run Python.
            else:
                cmd = r'{0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(
                    'python',
                    scriptpath,
                    srcdir,
                    args.res,
                    '--dst {}'.format(dstdir),
                    '--stripid {}'.format(stripid),
                    '--edgemask' * args.edgemask,
                    '--noentropy' * args.noentropy,
                    '--rema2a' * args.rema2a,
                )
                print '{}, {}'.format(i, cmd)

            if not args.dryrun:
                # For most cases, set `shell=True`.
                # For attaching process to PyCharm debugger,
                # set `shell=False`.
                subprocess.call(cmd, shell=True)

    else:
        # Process a single strip.
        maskFileSuffix = None
        if args.edgemask:
            maskFileSuffix = 'edgemask/datamask'
        elif args.rema2a:
            maskFileSuffix = 'mask2a'
        else:
            maskFileSuffix = 'mask'

        # TODO: "change this" (?)
        print "source: {}".format(srcdir)
        print "res: {}m".format(args.res)
        print "maskFileSuffix: {}".format(maskFileSuffix)

        # Find scene dems for this stripid to be merged into strips.
        scene_dems = glob.glob(os.path.join(srcdir, '*'+args.stripid+'*_dem.tif'))
        print "merging pair id: {}, {} scenes".format(args.stripid, len(scene_dems))
        if not scene_dems:
            print "no scene dems found to merge, skipping"
            sys.exit(1)

        # Existence check. If output already exists, skip.
        dst_dems = glob.glob(os.path.join(dstdir, '*'+args.stripid+'_seg*_dem.tif'))
        if dst_dems:
            print "output files exist, skipping"

        # Make sure all matchtag and ortho files exist. If missing, skip.
        missingflag = False
        for f in scene_dems:
            if not os.path.isfile(f.replace('dem.tif', 'matchtag.tif')):
                print "matchtag file for {} missing, skipping".format(f)
                missingflag = True
            if not os.path.isfile(f.replace('dem.tif', 'ortho.tif')):
                print "ortho file for {} missing, skipping".format(f)
                missingflag = True
            if not os.path.isfile(f.replace('dem.tif', 'meta.txt')):
                print "meta file for {} missing, skipping".format(f)
                missingflag = True
        if missingflag:
            sys.exit(1)

        # Filter all scenes in this strip.
        # TODO: Change the following once testing is complete
        # -t    and it is decided whether re-maskng based on timestamps should be done,
        # -t    or if masking should just be skipped if the mask already exists.
        filter_list = [f for f in scene_dems if shouldDoMasking(f.replace('dem.tif', 'matchtag.tif'), maskFileSuffix)]
        filter_total = len(filter_list)
        i = 0
        for demFile in filter_list:
            i += 1
            sys.stdout.write("filtering {} of {}: ".format(i, filter_total))
            generateMasks(demFile, maskFileSuffix, noentropy=args.noentropy)

        # Mosaic scenes in this strip together.
        # Output separate segments if there are breaks in overlap.
        input_sceneDemFnames = [os.path.basename(f) for f in scene_dems]
        segnum = 1
        while len(input_sceneDemFnames) > 0:

            print "building segment {}".format(segnum)
            X, Y, Z, M, O, trans, rmse, mosaicked_sceneDemFnames, spat_ref = scenes2strips(
                srcdir, input_sceneDemFnames, maskFileSuffix=maskFileSuffix
            )
            input_sceneDemFnames = list(set(input_sceneDemFnames).difference(set(mosaicked_sceneDemFnames)))
            if X is None:
                continue

            stripid_full = mosaicked_sceneDemFnames[0][0:47]
            strip_demFile = os.path.join(
                dstdir, "{}_seg{}_{}m_dem.tif".format(stripid_full, segnum, args.res)
            )
            print "DEM: {}".format(strip_demFile)

            strip_matchFile = strip_demFile.replace('dem.tif', 'matchtag.tif')
            strip_orthoFile = strip_demFile.replace('dem.tif', 'ortho.tif')
            strip_metaFile  = strip_demFile.replace('dem.tif', 'meta.txt')

            saveArrayAsTiff(M, strip_matchFile, X, Y, spat_ref, nodata_val=0,     dtype_out='uint8')
            del M
            saveArrayAsTiff(O, strip_orthoFile, X, Y, spat_ref, nodata_val=0,     dtype_out='int16')
            del O
            saveArrayAsTiff(Z, strip_demFile,   X, Y, spat_ref, nodata_val=-9999, dtype_out='float32')

            fp_vertices = getFPvertices(Z, Y, X, label=-9999, label_type='nodata', replicate_matlab=True)
            del Z, X, Y

            proj4 = spat_ref.ExportToProj4()
            time = datetime.today().strftime("%d-%b-%Y %H:%M:%S")
            writeStripMeta(strip_metaFile, srcdir, mosaicked_sceneDemFnames, trans, rmse, proj4, fp_vertices, time)

            segnum += 1


def shouldDoMasking(matchFile, maskFileSuffix='mask'):
    matchFile_date = os.path.getmtime(matchFile)
    maskFiles = ([matchFile.replace('matchtag.tif', 'edgemask.tif'),
                  matchFile.replace('matchtag.tif', 'datamask.tif')] if maskFileSuffix == 'edgemask/datamask'
            else [matchFile.replace('matchtag.tif', maskFileSuffix+'.tif')])
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
                   trans, rmse, proj4, fp_vertices, strip_time):

    if fp_vertices.dtype != int64 and array_equal(fp_vertices, fp_vertices.astype(int64)):
        fp_vertices = fp_vertices.astype(int64)

    # FIXME: Four lines in the following meta template have trailing space to replicate MATLAB.
    # -f     Remove these?
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
        ' '.join(array_str(fp_vertices[1], max_line_width=float('inf')).strip()[1:-1].split()),
        ' '.join(array_str(fp_vertices[0], max_line_width=float('inf')).strip()[1:-1].split()),
        )
    )

    for i in range(len(dem_list)):
        line = "{} {:.2f} {:.4f} {:.4f} {:.4f}\n".format(
            dem_list[i], rmse[0, i], trans[0, i], trans[1, i], trans[2, i])
        strip_info += line

    strip_info += "\nScene Metadata \n\n"

    scene_info = ""
    for i in range(len(dem_list)):
        scene_info += "scene {} name={}\n".format(i+1, dem_list[i])

        scene_metaFile = os.path.join(scenedir, dem_list[i].replace("dem.tif", "meta.txt"))
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



if __name__ == '__main__':
    main()
