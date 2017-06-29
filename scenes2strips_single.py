# Version 1.0; Erik Husby, Claire Porter; Polar Geospatial Center, University of Minnesota; 2017

import argparse
import glob
import os
import sys
from datetime import datetime

from mask_scene import generateMasks
from raster_array_tools import saveArrayAsTiff
from scenes2strips import scenes2strips, MissingFileError


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filters scene dems in a source directory,"
            " then mosaics them into strips and saves the results."
        )
    )
    parser.add_argument(
        'srcdir',
        help=(
            "Path to source directory containing scene dems to process."
            " If --dstdir is not specified, this path should contain the substring 'tif_results'."
        )
    )
    parser.add_argument(
        'stripid',
        help="<catid1_catid2> (as parsed from scene dem filenames)"
    )
    parser.add_argument(
        'res',
        choices=['2', '8'],
        help="Resolution of target dems (2 or 8 meters)."
    )
    parser.add_argument(
        '--dstdir',
        help=(
            "Path to directory for output mosaicked strip data"
            " (default is srcdir.replace('tif_results', 'strips'))."
        )
    )
    parser.add_argument(
        '--no-entropy',
        action='store_true',
        default=False,
        help="Use filter without entropy protection."
    )
    parser.add_argument(
        '--remask',
        action='store_true',
        default=False,
        help="Recreate and overwrite all existing edgemasks and datamasks (filtering output)."
    )
    parser.add_argument(
        '--restrip',
        action='store_true',
        default=False,
        help="Recreate and overwrite all existing output strips (mosaicking output)."
    )
    parser.add_argument(
        '--first-run',
        action='store_true',
        default=False,
        help="Skip strip output existence checks, but do filtering checks."
    )
    args = parser.parse_args()

    # Verify source directory.
    if not os.path.isdir(args.srcdir):
        parser.error("srcdir must be a directory.")
    abs_srcdir = os.path.abspath(args.srcdir)

    # Verify output directory.
    if args.dstdir is None:
        # TODO: Use the following regex method of determining strip output directory?
        # dirtail_pattern = re.compile("[/\\\]tif_results[/\\\]\d+m[/\\\]?$")
        # result = re.search(dirtail_pattern, abs_srcdir)
        # if result:
        #     dirtail = result.group(0)
        #     abs_dstdir = re.sub(
        #         dirtail_pattern,
        #         dirtail.replace('tif_results', 'strips').replace('\\', '\\\\'),
        #         abs_srcdir
        #     )
        # else:
        #     parser.error(
        #         "srcdir does not meet naming convention requirements for setting a default dstdir."
        #     )
        abs_dstdir = abs_srcdir.replace('tif_results', 'strips')
        print "Strip output directory set to: {}".format(abs_dstdir)
    else:
        abs_dstdir = os.path.abspath(args.dstdir)

    # TODO: "change this" (?)
    print "source: {}".format(abs_srcdir)
    print "res: {}m".format(args.res)

    # Find scene dems to be merged into strips.
    src_dems = glob.glob(os.path.join(abs_srcdir, '*'+args.stripid+'*_dem.tif'))
    print "merging pair id: {}, {} scenes".format(args.stripid, len(src_dems))
    if not src_dems:
        print "No scene dems found to merge. Exiting."
        sys.exit(1)

    # Create strip output directory if it doesn't already exist.
    isFirstRun = args.first_run
    if not os.path.isdir(abs_dstdir):
        isFirstRun = True
        os.makedirs(abs_dstdir)

    # Make sure all matchtag and ortho files exist. If missing, skip.
    skip = False
    for f in src_dems:
        if not os.path.isfile(f.replace('dem.tif', 'matchtag.tif')):
            print "matchtag file for {} missing, skipping this strip".format(f)
            skip = True
            break
        if not os.path.isfile(f.replace('dem.tif', 'ortho.tif')):
            print "ortho file for {} missing, skipping this strip".format(f)
            skip = True
            break
    if skip:
        sys.exit(1)

    skip = False
    reprocess = False
    if args.restrip or isFirstRun:
        reprocess = True
    else:
        # Perform existence check.
        dst_dems = glob.glob(os.path.join(abs_dstdir, '*'+args.stripid+'_seg*_dem.tif'))
        if len(dst_dems) > 0:

            # If new source dems have been added since the last run of s2s,
            # remove existing strip output and reprocess.
            a = min([os.path.getmtime(f) for f in dst_dems])
            b = max([os.path.getmtime(f) for f in src_dems])
            if not a > b:  # "If NOT all strip dems have been created (modified) after all source dems:"
                reprocess = True

            # NOTE: The following check prevents an interrupted run of this strip
            # from being properly restarted.
            # TODO: Come up with something better?
            # If new source masks have been created since the last run of s2s,
            # remove existing strip output and reprocess, else skip this strip.
            src_masks = glob.glob(os.path.join(abs_srcdir, '*'+args.stripid+'*mask.tif'))
            if src_masks:
                b = max([os.path.getmtime(f) for f in src_masks])
                if a > b:  # "If all strip dems have been created (modified) after all source masks:"
                    skip = True
                else:
                    reprocess = True

    if reprocess:
        dst_output = glob.glob(os.path.join(abs_dstdir, '*'+args.stripid+'_seg*'))
        if dst_output:
            if isFirstRun:
                print (
                    "ERROR: --first-run argument was given, but strip output exists for {}".
                    format(args.stripid)
                )
                print "Exiting."
                sys.exit(1)
            print "{} old strip output exists, deleting and reprocessing".format(args.stripid)
            for f in dst_output:
                # print "removed {}".format(f)
                if not args.dryrun:
                    os.remove(f)
    elif skip:
        print "{} younger strip than scene masks exists, skipping".format(args.stripid)
        sys.exit(1)

    # Filter all scenes in this strip.
    src_matches = [f.replace('dem.tif', 'matchtag.tif') for f in src_dems]
    filter_list = src_matches if args.remask else [f for f in src_matches if not shouldSkipMasking(f)]
    filter_total = len(filter_list)
    i = 0
    for matchFile in filter_list:
        i += 1
        print "processing {} of {}: {} ".format(i, filter_total, matchFile)
        generateMasks(matchFile)

    input_list = [os.path.basename(f) for f in src_dems]

    # Mosaic scenes in this strip together.
    # Output separate segments if there are breaks in overlap.
    segnum = 0
    while len(input_list) > 0:
        segnum += 1
        print "building segment {}".format(segnum)
        try:
            X, Y, Z, M, O, trans, rmse, proj_ref, proj4, fp_vertices, output_list = scenes2strips(
                args.src, input_list
            )
        except MissingFileError as err:
            print >>sys.stderr, err.msg
            sys.exit(1)

        stripid_full = output_list[0][0:47]
        output_demFile = os.path.join(
            abs_dstdir, "{}_seg{}_{}m_dem.tif".format(stripid_full, segnum, args.res)
        )
        print "DEM: {}".format(output_demFile)

        output_matchFile = output_demFile.replace('dem.tif', 'matchtag.tif')
        output_orthoFile = output_demFile.replace('dem.tif', 'ortho.tif')
        output_metaFile  = output_demFile.replace('dem.tif', 'meta.txt')

        saveArrayAsTiff(Z, output_demFile,   X, Y, proj_ref, nodataVal=-9999, dtype_out='float32')
        saveArrayAsTiff(M, output_matchFile, X, Y, proj_ref, nodataVal=0,     dtype_out='uint8')
        saveArrayAsTiff(O, output_orthoFile, X, Y, proj_ref, nodataVal=0,     dtype_out='int16')

        time = datetime.today().strftime("%d-%b-%Y %H:%M:%S")
        writeStripMeta(output_metaFile, args.src, output_list, trans, rmse, proj4, fp_vertices, time)

        input_list = list(set(input_list).difference(set(output_list)))


def shouldSkipMasking(matchFile):
    matchFile_date = os.path.getmtime(matchFile)
    skip_masking = True
    for maskFile in (matchFile.replace('matchtag.tif', 'edgemask.tif'),
                     matchFile.replace('matchtag.tif', 'datamask.tif')):
        if not os.path.isfile(maskFile):  # Masking must be done.
            skip_masking = False
            break
        # Update Mode - will only reprocess masks older than the matchtag file.
        maskFile_date = os.path.getmtime(maskFile)
        if (maskFile_date - matchFile_date) < -6.9444e-04:
            skip_masking = False
            break

    return skip_masking


def writeStripMeta(o_metaFile, scenedir, dem_list,
                   trans, rmse, proj4, fp_vertices, strip_time):
    strip_info = \
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
        str(fp_vertices[0]).replace(',', '')[1:-1],
        str(fp_vertices[1]).replace(',', '')[1:-1],
        )

    for i in range(len(dem_list)):
        line = "{} {:.2f} {:.4f} {:.4f} {:.4f}\n".format(
            dem_list[i], rmse[i], trans[0, i], trans[1, i], trans[2, i])
        strip_info += line

    strip_info += "\nScene Metadata \n\n"

    scene_info = ""
    for i in range(len(dem_list)):
        scene_info += "scene {} name={}\n".format(i+1, dem_list[i])

        scene_metaFile = os.path.join(scenedir, dem_list[i].replace("dem.tif", "meta.txt"))
        if os.path.isfile(scene_metaFile):
            scene_meta = open(scene_metaFile, 'r')
            scene_info += scene_meta.read()
            scene_meta.close()
        else:
            scene_info += "{} not found".format(scene_metaFile)
        scene_info += " \n"

    strip_meta = open(o_metaFile, 'w')
    strip_meta.write(strip_info)
    strip_meta.write(scene_info)
    strip_meta.close()



if __name__ == '__main__':
    main()
