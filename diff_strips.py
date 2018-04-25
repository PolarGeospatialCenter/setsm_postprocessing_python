#!/usr/bin/env python2

# Erik Husby; Polar Geospatial Center, University of Minnesota; 2018


from __future__ import division
import argparse
import filecmp
import os
from datetime import datetime

import numpy as np

import lib.raster_array_tools as rat
from lib.scenes2strips import coregisterdems

# import pdb


class MetadataError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class SpatialRefError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class NoOverlapError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


def main():
    parser = argparse.ArgumentParser(description=(
        "Difference two strip DEMs."))

    parser.add_argument('dem1',
        help="Path to reference DEM.")
    parser.add_argument('dem2',
        help="Path to comparison DEM.")

    parser.add_argument('-o', '--out', default=os.path.join(os.getcwd(), 'diff.tif'),
        help="File path of output difference image (default is './diff.tif').")

    parser.add_argument('-m', '--match', action='store_true', default=False,
        help="Save matchtag corresponding to difference image.")

    # Parse and validate arguments.
    args = parser.parse_args()
    demFile1 = os.path.abspath(args.dem1)
    demFile2 = os.path.abspath(args.dem2)
    diff_demFile = os.path.abspath(args.out)
    save_match = args.match
    outdir = os.path.dirname(diff_demFile)

    if not os.path.isfile(demFile1):
        parser.error("dem1 is not a valid file path")
    if not os.path.isfile(demFile2):
        parser.error("dem2 is not a valid file path")
    if filecmp.cmp(demFile1, demFile2):
        parser.error("dem1 is the same as dem2")
    if os.path.isfile(diff_demFile):
        parser.error("out difference image already exists")
    if save_match:
        matchFile1 = demFile1.replace('dem.tif', 'matchtag.tif')
        matchFile2 = demFile2.replace('dem.tif', 'matchtag.tif')
        if not os.path.isfile(matchFile1):
            parser.error("matchtag corresponding to dem1 does not exist: '{}'".format(matchFile1))
        if not os.path.isfile(matchFile2):
            parser.error("matchtag corresponding to dem2 does not exist: '{}'".format(matchFile2))
    if not os.path.isdir(os.path.dirname(outdir)):
        print "Creating directory for output results file: {}".format(outdir)
        os.makedirs(outdir)

    diff_strips(demFile1, demFile2, diff_demFile, save_match)


def get_trans_vector(regFile):
    # TODO: Write docstring.

    reg_fp = open(regFile, 'r')
    try:
        line = reg_fp.readline()
        while not line.startswith('Translation Vector (dz,dx,dy)(m)=') and line != "":
            line = reg_fp.readline()
        if line == "":
            reg_fp.close()
            raise MetadataError("Translation vector cannot be parsed "
                                "from registration file: {}".format(regFile))
    finally:
        reg_fp.close()

    vector_txt = line.replace('Translation Vector (dz,dx,dy)(m)=', '').strip()
    vector = np.fromstring(vector_txt, dtype=np.float32, sep=', ')

    return vector


def diff_strips(demFile1, demFile2, diff_demFile, save_match):
    # TODO: Write docstring.

    # Construct filenames.
    matchFile1 = demFile1.replace('dem.tif', 'matchtag.tif')
    matchFile2 = demFile2.replace('dem.tif', 'matchtag.tif')
    metaFile1  = demFile1.replace('dem.tif', 'mdf.txt')
    metaFile2  = demFile2.replace('dem.tif', 'mdf.txt')
    regFile1   = demFile1.replace('dem.tif', 'reg.txt')
    regFile2   = demFile2.replace('dem.tif', 'reg.txt')
    diff_matchFile = diff_demFile.replace('.tif', '_matchtag.tif')
    diff_metaFile  = diff_demFile.replace('.tif', '_meta.txt')

    # Read georeferenced strip geometries.
    x1, y1, spatref1 = rat.extractRasterData(demFile1, 'x', 'y', 'spat_ref')
    x2, y2, spatref2 = rat.extractRasterData(demFile2, 'x', 'y', 'spat_ref')

    # Make sure strips have same projection.
    if spatref2.IsSame(spatref1) != 1:
        raise SpatialRefError("Base strip '{}' spatial reference ({}) mismatch with "
                              "compare strip spatial reference ({})".format(
                              demFile1, spatref1.ExportToWkt(), spatref2.ExportToWkt()))
    spat_ref = spatref1

    # Find area of overlap.
    z1_c0, z1_r0 = 0, 0
    z1_c1, z1_r1 = None, None
    z2_c0, z2_r0 = 0, 0
    z2_c1, z2_r1 = None, None
    try:
        if x1[0] < x2[0]:
            overlap_ind = np.where(x1 == x2[0])[0]
            if overlap_ind.size == 0:
                raise NoOverlapError("")
            z1_c0 = overlap_ind[0]
        else:
            overlap_ind = np.where(x1[0] == x2)[0]
            if overlap_ind.size == 0:
                raise NoOverlapError("")
            z2_c0 = overlap_ind[0]
        if x1[-1] > x2[-1]:
            overlap_ind = np.where(x1 == x2[-1])[0]
            if overlap_ind.size == 0:
                raise NoOverlapError("")
            z1_c1 = overlap_ind[0] + 1
        else:
            overlap_ind = np.where(x1[-1] == x2)[0]
            if overlap_ind.size == 0:
                raise NoOverlapError("")
            z2_c1 = overlap_ind[0] + 1
        if y1[0] > y2[0]:
            overlap_ind = np.where(y1 == y2[0])[0]
            if overlap_ind.size == 0:
                raise NoOverlapError("")
            z1_r0 = overlap_ind[0]
        else:
            overlap_ind = np.where(y1[0] == y2)[0]
            if overlap_ind.size == 0:
                raise NoOverlapError("")
            z2_r0 = overlap_ind[0]
        if y1[-1] < y2[-1]:
            overlap_ind = np.where(y1 == y2[-1])[0]
            if overlap_ind.size == 0:
                raise NoOverlapError("")
            z1_r1 = overlap_ind[0] + 1
        else:
            overlap_ind = np.where(y1[-1] == y2)[0]
            if overlap_ind.size == 0:
                raise NoOverlapError("")
            z2_r1 = overlap_ind[0] + 1
    except NoOverlapError:
        raise NoOverlapError("Strip geometries do not overlap")

    if save_match:
        # Load matchtag data into arrays.
        print "Loading matchtag data"
        m1 = rat.extractRasterData(matchFile1, 'array')
        m2 = rat.extractRasterData(matchFile2, 'array')
        m1 = m1[z1_r0:z1_r1, z1_c0:z1_c1]
        m2 = m2[z2_r0:z2_r1, z2_c0:z2_c1]
        r0, r1, c0, c1 = crop_strip(m1, m2, method='data_density')
        # del m1, m2

    # Load DEM data into arrays.
    print "Loading raster data"
    z1 = rat.extractRasterData(demFile1, 'z')
    z2 = rat.extractRasterData(demFile2, 'z')
    z1[z1 == -9999] = np.nan
    z2[z2 == -9999] = np.nan

    # Crop arrays to area of overlap.
    x1 = x1[z1_c0:z1_c1]
    y1 = y1[z1_r0:z1_r1]
    x2 = x2[z2_c0:z2_c1]
    y2 = y2[z2_r0:z2_r1]
    z1 = z1[z1_r0:z1_r1, z1_c0:z1_c1]
    z2 = z2[z2_r0:z2_r1, z2_c0:z2_c1]

    # Crop arrays further to decrease memory use.
    if 'r0' not in vars():
        # r0, r1, c0, c1 = crop_strip(z1, method='center')
        r0, r1, c0, c1 = crop_strip(rat.getDataArray(z1, np.nan),
                                    rat.getDataArray(z2, np.nan),
                                    method='data_density')
    x1_crop = x1[c0:c1]
    y1_crop = y1[r0:r1]
    x2_crop = x2[c0:c1]
    y2_crop = y2[r0:r1]
    z1_crop = z1[r0:r1, c0:c1]
    z2_crop = z2[r0:r1, c0:c1]

    # # Get initial guess of translation vector to
    # hopefully speed up coregistration..
    # trans1 = get_trans_vector(regFile1)
    # trans2 = get_trans_vector(regFile2)
    # trans_guess = trans2 - trans1
    # trans_guess = np.reshape(trans_guess, (3, 1))

    # Coregister the two DEMs.
    print "Beginning coregistration"
    _, trans, rmse = coregisterdems(x1_crop, y1_crop, z1_crop, x2_crop, y2_crop, z2_crop)
    dz, dx, dy = trans

    # Interpolate comparison DEM to reference DEM.
    print "Interpolating dem2 to dem1"
    z2i = rat.interp2_gdal(x2-dx, y2-dy, z2-dz, x1, y1, 'linear')
    del z2

    # Difference DEMs and save result.
    print "Saving difference DEM"
    z_diff = z2i - z1
    z_diff[np.isnan(z_diff)] = -9999
    del z1, z2i
    rat.saveArrayAsTiff(z_diff, diff_demFile, x1, y1, spat_ref, nodata_val=-9999, dtype_out='float32')

    print "Extracting footprint vertices for metadata"
    fp_vertices = rat.getFPvertices(z_diff, y1, x1, label=-9999, label_type='nodata', replicate_matlab=True)
    del z_diff

    if save_match:
        if 'm2' not in vars():
            print "Loading match2"
            m2 = rat.extractRasterData(matchFile2, 'array').astype(np.float32)
            m2 = m2[z2_r0:z2_r1, z2_c0:z2_c1]
        elif m2.dtype != np.float32:
            m2 = m2.astype(np.float32)

        print "Interpolating match2 to match1"
        m2i = rat.interp2_gdal(x2-dx, y2-dy, m2, x1, y1, 'nearest')
        del m2
        m2i[np.isnan(m2i)] = 0  # convert back to uint8
        m2i = m2i.astype(np.bool)

        if 'm1' not in vars():
            print "Loading match1"
            m1 = rat.extractRasterData(matchFile1, 'array').astype(np.bool)
            m1 = m1[z1_r0:z1_r1, z1_c0:z1_c1]
        elif m1.dtype != np.bool:
            m1 = m1.astype(np.bool)

        print "Saving difference matchtag"
        m_diff = m2i
        np.logical_and(m1, m2i, m_diff)
        del m1
        rat.saveArrayAsTiff(m_diff, diff_matchFile, x1, y1, spat_ref, nodata_val=0, dtype_out='uint8')
        del m_diff

    # Write metadata for difference image.
    proj4 = spat_ref.ExportToProj4()
    time = datetime.today().strftime("%d-%b-%Y %H:%M:%S")
    writeDiffMeta(diff_metaFile, demFile1, demFile2, trans, rmse, proj4, fp_vertices, time)


def crop_strip(a1, a2=None, size=1.0, sampling=0.5, method='center'):
    # TODO: Write docstring.

    c0, r0 = 0, 0
    c1, r1 = None, None

    nrows, ncols = a1.shape
    crop_sz = int(np.ceil(size * min(nrows, ncols)))

    if method == 'center':
        border_rows = int(np.ceil((nrows - crop_sz) / 2))
        border_cols = int(np.ceil((ncols - crop_sz) / 2))
        if border_rows > 0:
            r0 = border_rows
            r1 = border_rows + crop_sz
        if border_cols > 0:
            c0 = border_cols
            c1 = border_cols + crop_sz

    elif method == 'data_density':
        crop_i = min(nrows, crop_sz)
        crop_j = min(ncols, crop_sz)
        crop_shape = (crop_i, crop_j)
        check_i = np.floor(np.arange(crop_i/2, nrows-crop_i/2+0.001, crop_i*sampling)).astype(np.int64)
        check_j = np.floor(np.arange(crop_j/2, ncols-crop_j/2+0.001, crop_j*sampling)).astype(np.int64)
        check_res = np.zeros((len(check_i), len(check_j)), dtype=np.int64)
        for m in range(len(check_i)):
            i = check_i[m]
            for n in range(len(check_j)):
                j = check_j[n]
                check_res[m, n] = np.count_nonzero(
                    np.logical_and(rat.getWindow(a1, i, j, crop_shape),
                                   rat.getWindow(a2, i, j, crop_shape))
                )
        best_m, best_n = np.unravel_index(np.argmax(check_res), (len(check_i), len(check_j)))
        best_i, best_j = check_i[best_m], check_j[best_n]
        r0, r1, c0, c1 = rat.getWindow(a1, best_i, best_j, crop_shape, output='indices')

    return r0, r1, c0, c1


def writeDiffMeta(o_metaFile, demFile1, demFile2,
                  trans, rmse, proj4, fp_vertices, creation_time):

    if fp_vertices.dtype != np.int64 and np.array_equal(fp_vertices, fp_vertices.astype(np.int64)):
        fp_vertices = fp_vertices.astype(np.int64)

    # FIXME: Four lines in the following meta template have trailing space to replicate MATLAB.
    # -f     Remove these?
    diff_info = (
"""DoD Metadata 
Creation Date: {}
DoD creation date: {}
DoD projection (proj4): '{}'

DoD Footprint Vertices
X: {} 
Y: {} 

Mosaicking Alignment Statistics (meters) 
scene, rmse, dz, dx, dy
""".format(
        creation_time,
        creation_time,
        proj4,
        ' '.join(np.array_str(fp_vertices[1], max_line_width=float('inf')).strip()[1:-1].split()),
        ' '.join(np.array_str(fp_vertices[0], max_line_width=float('inf')).strip()[1:-1].split()),
        )
    )

    diff_info += "{} {:.2f} {:.4f} {:.4f} {:.4f}\n".format(os.path.basename(demFile1),
                                                           0, 0, 0, 0)
    diff_info += "{} {:.2f} {:.4f} {:.4f} {:.4f}\n".format(os.path.basename(demFile2),
                                                           rmse, trans[0], trans[1], trans[2])

    diff_info += "\nStrip Registration \n\n"

    dem_list = [demFile1, demFile2]
    strip_info = ""
    for i in range(len(dem_list)):
        strip_info += "strip {} name={}\n".format(i+1, dem_list[i])

        strip_metaFile = dem_list[i].replace('dem.tif', 'reg.txt')
        if os.path.isfile(strip_metaFile):
            strip_metaFile_fp = open(strip_metaFile, 'r')
            strip_info += strip_metaFile_fp.read()
            strip_metaFile_fp.close()
        else:
            strip_info += "{} not found".format(strip_metaFile)
        strip_info += " \n"

    diff_metaFile_fp = open(o_metaFile, 'w')
    diff_metaFile_fp.write(diff_info)
    diff_metaFile_fp.write(strip_info)
    diff_metaFile_fp.close()



if __name__ == '__main__':
    main()
