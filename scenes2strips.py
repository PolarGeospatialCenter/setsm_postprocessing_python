# Version 3.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2016
# Translated from MATLAB code written by Ian Howat, Ohio State University, 2017

from __future__ import division
import os.path
import sys
import warnings

import gdal, ogr, osr
import numpy as np
from scipy import ndimage, interpolate, misc
from skimage import morphology

import raster_array_tools as rat
import test


class DimensionError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)


class CornerCoords:
    """
    This class is designed to make easily accessible the coordinates of the four
    corner points of a raster dataset ds, allowing for nonzero rotation factors.
    """
    def __init__(self, ds, spatialRef=False):
        self.coords = self.get_corner_coords(self, ds)
        self.x = self.coords[:, 0]
        self.y = self.coords[:, 1]
        self.sr = osr.SpatialReference(ds.GetProjectionRef()) if spatialRef else None

    @staticmethod
    def get_corner_coords(self, ds):
        """
        Returns a 5x2 matrix of corner coordinate pairs for the raster dataset ds.
        Ordered with the top left corner first, goes clockwise, ends back at top left.
        """
        gt = ds.GetGeoTransform()

        top_left_x = np.full((5, 1), gt[0])
        top_left_y = np.full((5, 1), gt[3])
        top_left_mat = np.concatenate((top_left_x, top_left_y), axis=1)

        raster_XY_size_mat = np.array([
            [0, 0],
            [ds.RasterXSize, 0],
            [ds.RasterXSize, ds.RasterYSize],
            [0, ds.RasterYSize],
            [0, 0]
        ])

        gt_mat = np.array([
            [gt[1], gt[4]],
            [gt[2], gt[5]]
        ])

        return top_left_mat + np.dot(raster_XY_size_mat, gt_mat)

    def wkt(self):
        c = []
        for i in range(5):
            c.append([])
            for j in range(2):
                c[i].append(str(self.coords[i][j]))
            c[i] = " ".join(c[i])
        return 'POLYGON ((' + ','.join(c) + '))'

    def geometry(self):
        return ogr.Geometry(wkt=self.wkt())

    def geometrySR(self):
        geom = self.geometry()
        geom.AssignSpatialReference(self.sr)
        return geom


def rectFootprint(*geoms):
    """
    Returns the smallest rectangular footprint that contains all input polygons,
    all as OGRGeometry objects.
    """
    minx = float('inf')
    miny = float('inf')
    maxx = float('-inf')
    maxy = float('-inf')

    for geom in geoms:
        coords = geom.ExportToWkt().replace('POLYGON ((', '').replace('))', '').split(',')
        for coord in coords:
            c = coord.split(' ')
            x, y = float(c[0]), float(c[1])
            minx = min(x, minx)
            miny = min(y, miny)
            maxx = max(x, maxx)
            maxy = max(y, maxy)

    fp_wkt = 'POLYGON (({} {},{} {},{} {},{} {},{} {}))'.format(
        minx, maxy,
        maxx, maxy,
        maxx, miny,
        minx, miny,
        minx, maxy
    )

    return ogr.Geometry(wkt=fp_wkt)


def orderPairs(demdir, files):
    """
    Scene order is determined in relation to *grid north* of the common projection
    by comparing total x-extent and y-extent of the scenes as a whole (aspect ratio).
    The larger of these extents determines the coordinate by which to do ordering.
    """
    R0 = np.zeros((len(files), 4))  # matrix to place rectangular parameters of the images
    geoms = []                      # list to place OGR geometries of the images (as rectangles)

    # Get rectangular parameters and geometries.
    for i in range(len(files)):
        ds = gdal.Open(os.path.join(demdir, files[i]), gdal.GA_ReadOnly)
        ds_CC = CornerCoords(ds)
        R0[i,:] = [min(ds_CC.x), min(ds_CC.y), max(ds_CC.x)-min(ds_CC.x), max(ds_CC.y)-min(ds_CC.y)]
        geoms.append((i, ds_CC.geometry()))
        del ds, ds_CC

    # Calculate aspect ratio, ar = x-extent/y-extent
    ar = (  (max(R0[:,0] + R0[:,2]) - min(R0[:,0]))
          / (max(R0[:,1] + R0[:,3]) - min(R0[:,1])))

    if ar >= 1:
        # Scenes are in east-west direction; start with scene with minimum x.
        ordered_file_indices = [np.argmin(R0[:,0])]
    else:
        # Scenes are in north-south direction; start with scene with minimum y.
        ordered_file_indices = [np.argmin(R0[:,1])]

    # Start with the footprint of this scene and let the strip grow from there.
    footprint_geom = geoms[ordered_file_indices[0]][1]
    del geoms[ordered_file_indices[0]]

    # Loop through scene pair geometries and sequentially add the
    # next pair with the most overlap to the ordered indices list.
    for i in range(len(geoms)):
        overlap_area = [footprint_geom.Intersection(index_geom_tup[1]).GetArea() for index_geom_tup in geoms]
        if max(overlap_area) > 0:
            selected_tup = geoms[np.argmax(overlap_area)]
            scene_index, scene_geom = selected_tup
            # Extend the rectangular footprint geometry to include the new scene.
            footprint_geom = rectFootprint(footprint_geom, scene_geom)
            # Add the new scene to the file order at the last position.
            ordered_file_indices.append(scene_index)
            # Remove the geometry of this scene (and its index) from the input list.
            del geoms[np.argmax(overlap_area)]
        else:
            print "Break in overlap detected, returning this segment only"
            break

    demFiles_ordered = [files[i] for i in ordered_file_indices]

    return demFiles_ordered


def loaddata(demFile, matchFile, orthoFile, maskFile):
    """
    Load data files and perform basic conversions.
    """
    z, x_dem, y_dem = rat.oneBandImageToArrayZXY(demFile)

    m = rat.oneBandImageToArray(matchFile)
    if m.shape != z.shape:
        print "WARNING: matchFile '{}' has wrong dimensions".format(matchFile)
        print "Interpolating to match associated dem's dimensions"
        x, y = rat.getXYarrays(matchFile)
        m = rat.interp2_gdal(x, y, m.astype(np.float32), x_dem, y_dem, 'nearest')
        m[np.where(np.isnan(m))] = 0  # Convert back to bool.
    m = m.astype(np.bool)

    if os.path.isfile(orthoFile):
        o = rat.oneBandImageToArray(orthoFile)
        if o.shape != z.shape:
            print "WARNING: orthoFile '{}' has wrong dimensions".format(orthoFile)
            print "Interpolating to match associated dem's dimensions"
            x, y = rat.getXYarrays(orthoFile)
            o[np.where(np.isnan(o))] = np.nan  # Set border to NaN so it won't be interpolated.
            o = rat.interp2_gdal(x, y, o.astype(np.float32), x_dem, y_dem, 'cubic')
            o[np.where(np.isnan(o))] = 0  # Convert back to uint16.
    else:
        o = np.zeros(z.shape)
    o = o.astype(np.uint16)

    if maskFile is not None:
        md = rat.oneBandImageToArray(maskFile)
        if md.shape != z.shape:
            raise DimensionError("maskFile '{}' has wrong dimensions".format(maskFile))
    else:
        md = np.ones(z.shape, dtype=bool)
    md = md.astype(np.bool)

    # A pixel with a value of -9999 is a nodata pixel; interpret it as NaN.
    z[np.where((z < -100) | (z == 0))] = np.nan

    return x_dem, y_dem, z, m, o, md


def cropnans(matrix, buff=0):
    """
    Crop matrix of bordering NaNs.
    """
    data = ~np.isnan(matrix)
    if ~np.any(data):
        return None, None  # This shouldn't happen on call from applyMasks.

    # Get indices.
    row_data = np.sum(data, axis=1).nonzero()[0]
    rowcrop_i = row_data[0]  - buff
    rowcrop_j = row_data[-1] + buff

    col_data = np.sum(data, axis=0).nonzero()[0]
    colcrop_i = col_data[0]  - buff
    colcrop_j = col_data[-1] + buff

    if rowcrop_i < 0:
        rowcrop_i = 0
    if rowcrop_j >= data.shape[0]:
        rowcrop_j = data.shape[0] - 1

    if colcrop_i < 0:
        colcrop_i = 0
    if colcrop_j >= data.shape[1]:
        colcrop_j = data.shape[1] - 1

    return (rowcrop_i, rowcrop_j+1), (colcrop_i, colcrop_j+1)


def applyMasks(x, y, z, match, ortho, mask):
    # TODO: Write docstring.

    z[~mask] = np.nan
    match[~mask] = 0

    # If there is any good data, crop the matrices of bordering NaNs.
    if np.any(~np.isnan(z)):
        rowcrop, colcrop = cropnans(z)

        x = x[colcrop[0]:colcrop[1]]
        y = y[rowcrop[0]:rowcrop[1]]
        z =         z[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]
        match = match[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]
        ortho = ortho[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]

    return x, y, z, match, ortho


def regrid(x, y, z, match, ortho):
    # TODO: Write docstring.

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    xi = np.arange(x[0] + dx - ((x[0]/dx) % 1)*dx, x[-1], dx)
    yi = np.arange(y[0] + dy - ((y[0]/dy) % 1)*dy, y[-1], dy)

    zi = rat.interp2_gdal(x, y, z, xi, yi, 'linear').astype(np.float32)

    match = rat.interp2_gdal(x, y, match.astype(np.float32), xi, yi, 'nearest')
    match[np.where(np.isnan(match))] = 0  # Convert back to uint8.
    match = match.astype(np.bool)

    # Interpolate ortho to same grid.
    ortho = ortho.astype(np.float32)
    ortho[np.where(np.isnan(z))] = np.nan  # Set border to NaN so it won't be interpolated.
    ortho = rat.interp2_gdal(x, y, ortho, xi, yi, 'cubic')
    ortho[np.where(np.isnan(ortho))] = 0  # Convert back to uint16.
    ortho = ortho.astype(np.uint16)

    return xi, yi, zi, match, ortho


def batchConcatenate(pairs, direction, axis_num):
    # TODO: Write docstring.

    if direction in ('left', 'up'):
        for i in range(len(pairs)):
            pairs[i] = np.concatenate(pairs[i], axis=axis_num)
    else:
        for i in range(len(pairs)):
            pairs[i].reverse()
            pairs[i] = np.concatenate(pairs[i], axis=axis_num)

    return pairs


def expandCoverage(Z, M, O, R1, direction):
    """
    Expands strip coverage for Z, M, and O
    based upon the direction of expansion.
    When (X1/Y1) is passed in for R1,
    (('left' or 'right') / ('up' or 'down'))
    must be passed in for direction.
    """

    # NumPy FutureWarning stating that
    # numpy.full(shapeTup, False/0) will return an array of dtype('bool'/'int64')
    # can be ignored since returned arrays are explicitly cast to wanted types.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if direction in ('left', 'right'):
            # R1 is X1
            Z1 = np.full((Z.shape[0], R1.size), np.nan).astype(np.float32)
            M1 = np.full((M.shape[0], R1.size), False).astype(np.bool)
            O1 = np.full((O.shape[0], R1.size), 0).astype(np.uint16)
            axis_num = 1
        else:
            # R1 is Y1
            Z1 = np.full((R1.size, Z.shape[1]), np.nan).astype(np.float32)
            M1 = np.full((R1.size, M.shape[1]), False).astype(np.bool)
            O1 = np.full((R1.size, O.shape[1]), 0).astype(np.uint16)
            axis_num = 0

    Z, M, O = batchConcatenate([[Z1,Z], [M1,M], [O1,O]],
                               direction, axis_num)
    return Z, M, O


def coregisterdems(x1, y1, z1, x2, y2, z2, *varargin):
    """
    % COREGISTERDEM registers a floating to a reference DEM

    % [z2r,trans,rms] = coregisterdems(x1,y1,z1,x2,y2,z2) registers the
    % floating DEM in 2D array z2 with coordinate vectors x2 and y2 to the
    % reference DEM in z1 using the iterative procedure in Nuth and Kaab,
    % 2010. z2r is the regiestered DEM, p is the z,x,y transformation
    % parameters and rms is the rms of the transformation in the vertical.
    """

    # Maximum offset allowed
    maxp = 15

    if len(x1) < 3 or len(y1) < 3 or len(x2) < 3 or len(y2) < 3:
        raise DimensionError("minimum array dimension is 3")

    interpflag = True
    if (x1.size == x2.size) and (y1.size == y2.size):
        if ~np.any(x2 - x1) and ~np.any(y2 - y1):
            interpflag = False

    if len(varargin) == 2:
        m1 = varargin[0]
        m2 = varargin[1]

    rx = x1[1] - x1[0]  # coordinate spacing
    p  = np.zeros(3)    # initial trans variable
    pn = p.copy()       # iteration variable
    d0 = np.inf         # initial rmse
    it = 1              # iteration step

    while it:

        if interpflag:
            # Interpolate the floating data to the reference grid.
            z2n = rat.interp2_gdal(x2 - pn[1], y2 - pn[2], z2 - pn[0],
                                   x1, y1, 'linear')
            if 'm2' in vars():
                m2n = rat.interp2_gdal(x2 - pn[1], y2 - pn[2], m2.astype(np.float32),
                                       x1, y1, 'nearest')
                m2n[np.where(np.isnan(m2n))] = 0  # convert back to uint8
                m2n = m2n.astype(np.bool)
        else:
            z2n = z2 - pn[0]
            if 'm2' in vars():
                m2n = m2

        interpflag = True

        # Slopes
        sy, sx = np.gradient(z2n, rx)
        sx = -sx

        sys.stdout.write("Planimetric Correction Iteration {} ".format(it))

        # Difference grids.
        dz = z2n - z1

        if 'm1' in vars() and 'm2' in vars():
            dz[~m2n | ~m1] = np.nan

        if ~np.any(~np.isnan(dz)):
            print "No overlap"
            z2out = z2
            p = np.full(3, np.nan)
            d0 = np.nan
            break

        # Filter NaNs and outliers.
        # FIXME: The following throws "RuntimeWarning: invalid value encountered in less_equal".
        n = ~np.isnan(sx) & ~np.isnan(sy) & \
            (abs(dz - np.nanmedian(dz)) <= np.nanstd(dz))

        if ~np.any(n):
            sys.stdout.write("regression failure, all overlap filtered\n")
            p = np.full(3, np.nan)  # initial trans variable
            d0 = np.nan
            z2out = z2
            break

        # Get RMSE and break if below threshold.
        RMSE_THRESH = 0.001
        d1 = np.sqrt(np.mean(np.power(dz[n], 2)))

        # Keep median dz if first iteration.
        if it == 1:
            meddz = np.median(dz[n])
            d00 = np.sqrt(np.mean(np.power(dz[n] - meddz, 2)))

        sys.stdout.write("rmse= {:.3f} ".format(d1))

        if ((d0 - d1) < RMSE_THRESH) or np.isnan(d0):
            sys.stdout.write("stopping \n")
            # If fails after first registration attempt,
            # set dx and dy to zero and subtract the median offset.
            if it == 2:
                sys.stdout.write(
                    "regression failure, returning median vertical offset: {:.3f}\n".format(meddz)
                )
                p[0] = meddz
                d0 = d00
                z2out = z2 - meddz
            break

        # Keep this adjustment.
        p = pn.copy()
        d0 = d1
        z2out = z2n.copy()

        # Build design matrix.
        X = np.column_stack((np.ones(dz[n].size), sx[n], sy[n]))

        # Solve for new adjustment.
        px = np.linalg.lstsq(X, dz[n])[0]
        pn = p + px

        # Display offsets.
        sys.stdout.write("offset(z,x,y): {:.3f}, {:.3f}, {:.3f}\n".format(pn[0], pn[1], pn[2]))

        if np.any(abs(pn[1:]) > maxp):
            sys.stdout.write(
                "maximum horizontal offset reached,"
                " returning median vertical offset: {:.3f}\n".format(meddz)
            )
            p = np.array([meddz, 0, 0])
            d0 = d00
            z2out = z2 - meddz
            break

        # Update iteration vars.
        it += 1

    return np.array([z2out, p, d0])


def scenes2strips(demdir, demFiles, noMask=False, max_coreg_rmse=1):
    """
    function [X,Y,Z,M,O,trans,rmse,f]=scenes2strips(demdir,f)
    %SCENES2STRIPS merge scenes into strips
    %
    %   [x,y,z,m,o,trans,rmse,f]=scenes2strips(demdir,f) merges the
    %   scene geotiffs listed in cellstr f within directory demdir after
    %   ordering them by position. If a break in coverage is detected between
    %   scene n and n+1 only the first 1:n scenes will be merged. The data are
    %   coregistered at overlaps using iterative least squares, starting with
    %   scene n=1.
    %   Outputs are the strip grid coorinates x,y and strip elevation, z,
    %   matchtag, m and orthoimage, o. The 3D translations are given in 3xn
    %   vector trans, along with root-mean-squared of residuals, rmse. The
    %   output f gives the list of filenames in the mosaic. If a break is
    %   detected, the list of output files will be less than the input.
    %
    % Version 3.0, Ian Howat, Ohio State University, 2015

    If 'noMask' is True, a mask will not be applied.
    'max_coreg_rmse' is the maximum coregistration error limit in meters.
    --Errors above this limit will result in a segment break.
    """

    # Order scenes in north-south or east-west direction by aspect ratio.
    print "ordering {} scenes".format(len(demFiles))
    demFiles_ordered = orderPairs(demdir, demFiles)

    # FIXME: Create a systematic way of determining the projection in which
    # -t     output strips are saved.
    proj_ref = rat.getProjRef(os.path.join(demdir, demFiles_ordered[0]))

    # Initialize output stats.
    trans = np.zeros((3, len(demFiles_ordered)))
    rmse = np.zeros(len(demFiles_ordered))

    # File loop.
    for i in range(len(demFiles_ordered)):

        # Construct filenames.
        demFile = os.path.join(demdir, demFiles_ordered[i])
        matchFile = demFile.replace('dem.tif', 'matchtag.tif')
        orthoFile = demFile.replace('dem.tif', 'ortho.tif')
        # %shadeFile= demFile.replace('dem.tif','dem_shade.tif')
        # %maskFile=  demFile.replace('dem.tif','mask.tif')

        maskFile = None if noMask else demFile.replace('dem.tif','mask.tif')

        print "scene {} of {}: {}".format(i+1, len(demFiles_ordered), demFile)

        try:
            x, y, z, m, o, md = loaddata(demFile, matchFile, orthoFile, maskFile)
        except Exception as e:
            print "Data read error:"
            print >>sys.stderr, e.msg
            print "...skipping"
            continue

        # Check for no data.
        if ~md.any():
            print "no data, skipping"
            continue

        # Apply masks.
        x, y, z, m, o = applyMasks(x, y, z, m, o, md)

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Fix grid so that x, y coordinates of
        # pixels in overlapping scenes will match up.
        # TODO: The following function still needs testing.
        if ((x[1] / dx) % 1 != 0) or ((y[1] / dy) % 1 != 0):
            x, y, z, m, o = regrid(x, y, z, m, o)

        # If this is the first scene in strip,
        # set as strip and continue to next scene.
        if 'X' not in vars():
            X, Y, Z, M, O = x, y, z, m, o
            del x, y, z, m, o
            continue

        # Pad new arrays to stabilize interpolation.
        buff = int(10*dx + 1)
        z = np.pad(z, buff, 'constant', constant_values=np.nan)
        m = np.pad(m, buff, 'constant', constant_values=0)
        o = np.pad(o, buff, 'constant', constant_values=0)
        x = np.concatenate((x[0]  - dx*np.arange(buff, 0, -1), x,
                            x[-1] + dx*np.arange(1, buff+1)))
        y = np.concatenate((y[0]  + dx*np.arange(buff, 0, -1), y,
                            y[-1] - dx*np.arange(1, buff+1)))

        # Expand strip coverage to encompass new scene.
        if x[0] < X[0]:
            X1 = np.arange(x[0], X[0], dx)
            X = np.concatenate((X1, X))
            Z, M, O = expandCoverage(Z, M, O, X1, direction='left')
            del X1
        if x[-1] > X[-1]:
            X1 = np.arange(X[-1]+dx, x[-1]+dx, dx)
            X = np.concatenate((X, X1))
            Z, M, O = expandCoverage(Z, M, O, X1, direction='right')
            del X1
        if y[0] > Y[0]:
            Y1 = np.arange(y[0], Y[0], -dx)
            Y = np.concatenate((Y1, Y))
            Z, M, O = expandCoverage(Z, M, O, Y1, direction='up')
            del Y1
        if y[-1] < Y[-1]:
            Y1 = np.arange(Y[-1]-dx, y[-1]-dx, -dx)
            Y = np.concatenate((Y, Y1))
            Z, M, O = expandCoverage(Z, M, O, Y1, direction='down')
            del Y1

        # Map new dem pixels to swath. These must return integers. If not,
        # interpolation will be required, which is currently not supported.
        c0 = np.where(X == x[0])[0][0]
        c1 = np.where(X == x[-1])[0][0] + 1
        r0 = np.where(Y == y[0])[0][0]
        r1 = np.where(Y == y[-1])[0][0] + 1

        # Crop to overlap.
        Xsub = X[c0:c1]
        Ysub = Y[r0:r1]
        Zsub = Z[r0:r1, c0:c1]
        Msub = M[r0:r1, c0:c1]
        Osub = O[r0:r1, c0:c1]

        # NEW MOSAICKING CODE

        cmin = 1000  # Minimum data cluster area for 2m.

        # Crop to just region of overlap.
        A = (~np.isnan(Zsub) & ~np.isnan(z)).astype(np.float32)

        # Check for segment break.
        if np.sum(A) <= cmin:
            demFiles_ordered = demFiles_ordered[:i]
            trans = trans[:, :i]
            rmse = rmse[:i]
            break

        A[np.where(A == 0)] = np.nan
        r, c = cropnans(A, buff)

        # Make overlap mask removing isolated pixels.
        Z_cropped_nodata =  np.isnan(Zsub[r[0]:r[1], c[0]:c[1]])
        z_cropped_data   = ~np.isnan(   z[r[0]:r[1], c[0]:c[1]])
        # Nodata in strip and data in scene is a one.
        A = morphology.remove_small_objects(
            Z_cropped_nodata & z_cropped_data, min_size=cmin, connectivity=2).astype(np.float32)

        # Check for redundant scene.
        if np.sum(A) <= cmin:
            print "redundant scene, skipping "
            continue

        # Data in strip and nodata in scene is a two.
        A[morphology.remove_small_objects(
            ~Z_cropped_nodata & ~z_cropped_data, min_size=cmin, connectivity=2)] = 2

        # Check for segment break.
        if np.sum(A) <= cmin:
            demFiles_ordered = demFiles_ordered[:i]
            trans = trans[:, :i]
            rmse = rmse[:i]
            break

        # FIXME: What does the following commented out code
        # -f     (taken from Ian's scenes2strips.m) do?
        # %     tic
        # %     % USING REGIONFILL - Requires matlab 2015a or newer
        # %     A = regionfill(A,A==0) -1;
        # %     toc

        Ar = rat.my_imresize(A, 0.1, 'nearest', PILmode='F')

        # Locate pixels on outside of boundary of overlap region.
        Ar_nonzero = (Ar != 0)
        B = (Ar_nonzero != ndimage.binary_erosion(Ar_nonzero))
        B = np.where(B)

        # TODO: The following method of extrapolation uses a cheap trick
        # -t    in an attempt to get as close as possible to replicating
        # -t    the results of MATLAB's "scatteredInterpolant" function.
        # -t    Decide if this is really okay to do.
        # Pixels outside of the convex hull of input points for interpolate.griddata
        # currently can't be extrapolated (by default they are filled with NaN),
        # so I do a trick here where I use interpolate.SmoothBivariateSpline to assign
        # a value of 1 or 2 to every corner of Ar before doing the main interpolation.
        fn = interpolate.SmoothBivariateSpline(B[0], B[1], Ar[B].astype(np.float64), kx=1, ky=1)
                                                                # kx, ky = 1 since first order
                                                                # spline interpolation is linear.
        corner_coords = (np.array([0,             0, Ar.shape[0]-1, Ar.shape[0]-1]),
                         np.array([0, Ar.shape[1]-1, Ar.shape[1]-1,             0]))
        Ar_interp = fn.ev(corner_coords[0], corner_coords[1])
        Ar_interp = np.round(Ar_interp).astype(int)
        Ar_interp[np.where(Ar_interp < 1)] = 1
        Ar_interp[np.where(Ar_interp > 2)] = 2
        Ar[corner_coords] = Ar_interp

        # Add the corner coordinates to the list of boundary coordinates,
        # which will be used for interpolation.
        By = np.concatenate((B[0], corner_coords[0]))
        Bx = np.concatenate((B[1], corner_coords[1]))
        B = (By, Bx)

        # Use the coordinates and values of boundary pixels
        # to interpolate values for pixels with value zero.
        Ar_zero_coords = np.where(~Ar_nonzero)
        Ar_interp = interpolate.griddata(B, Ar[B].astype(np.float64), Ar_zero_coords, 'linear')
        Ar[Ar_zero_coords] = Ar_interp

        Ar = misc.imresize(Ar, A.shape, 'bilinear', mode='F')
        Ar[np.where((A == 1) & (Ar != 1))] = 1
        Ar[np.where((A == 2) & (Ar != 2))] = 2
        A = Ar - 1
        A[np.where(A < 0)] = 0
        A[np.where(A > 1)] = 1

        W = (~np.isnan(Zsub)).astype(np.float32)
        W[r[0]:r[1], c[0]:c[1]] = A
        del A
        W[np.where(np.isnan(Zsub) & np.isnan(z))] = np.nan

        # Shift weights so that more of the reference layer is kept.
        f0 = 0.25  # overlap fraction where ref z weight goes to zero
        f1 = 0.55  # overlap fraction where ref z weight goes to one

        W = (1/(f1-f0))*W - f0/(f1-f0)
        W[np.where(W > 1)] = 1
        W[np.where(W < 0)] = 0

        # Remove <25% edge of coverage from each in pair.
        # FIXME: Each throws "RuntimeWarning: invalid value encountered in less_equal".
        Zsub[np.where(W == 0)] = np.nan
        Msub[np.where(W == 0)] = 0
        Osub[np.where(W == 0)] = 0

        z[np.where(W >= 1)] = np.nan
        m[np.where(W >= 1)] = 0
        o[np.where(W >= 1)] = 0

        # Coregistration

        P0 = rat.getDataDensityMap(Msub[r[0]:r[1], c[0]:c[1]]) > 0.9
        P1 = rat.getDataDensityMap(m[r[0]:r[1], c[0]:c[1]])    > 0.9

        # Coregister this scene to the strip mosaic.
        trans[:, i], rmse[i] = coregisterdems(
            Xsub[c[0]:c[1]], Ysub[r[0]:r[1]], Zsub[r[0]:r[1], c[0]:c[1]],
               x[c[0]:c[1]],    y[r[0]:r[1]],    z[r[0]:r[1], c[0]:c[1]],
            Msub[r[0]:r[1], c[0]:c[1]],
               m[r[0]:r[1], c[0]:c[1]]
        )[[1, 2]]

        # Check for segment break.
        if np.isnan(rmse[i]) or rmse[i] > max_coreg_rmse:
            print "Unable to coregister, breaking segment"
            demFiles_ordered = demFiles_ordered[:i]
            trans = trans[:, :i]
            rmse = rmse[:i]
            break

        # Interpolation grid
        xi = x - trans[1, i]
        yi = y - trans[2, i]

        # Check that uniform spacing is maintained (sometimes rounding errors).
        if len(np.unique(np.diff(xi))) > 1:
            xi = np.round(xi, 4)
        if len(np.unique(np.diff(yi))) > 1:
            yi = np.round(yi, 4)

        # Interpolate the floating data to the reference grid.
        zi = rat.interp2_gdal(xi, yi, z-trans[0,i], Xsub, Ysub, 'linear')
        del z

        # Interpolate the mask to the same grid.
        mi = rat.interp2_gdal(xi, yi, m.astype(np.float32), Xsub, Ysub, 'nearest')
        mi[np.where(np.isnan(mi))] = 0  # convert back to uint8
        mi = mi.astype(np.bool)
        del m

        # Interpolate ortho to same grid.
        oi = o.astype(np.float32)
        oi[np.where(oi == 0)] = np.nan  # Set border to NaN so it won't be interpolated.
        oi = rat.interp2_gdal(xi, yi, oi, Xsub, Ysub, 'cubic')
        del o

        del Xsub, Ysub

        # Remove border 0's introduced by nn interpolation.
        M3 = ~np.isnan(zi)
        M3 = ndimage.binary_erosion(M3, structure=np.ones((6, 6)))  # border cutline

        zi[~M3] = np.nan
        mi[~M3] = 0
        del M3

        # Remove border on orthos separately.
        M4 = (oi != 0)
        M4 = ndimage.binary_erosion(M4, structure=np.ones((6, 6)))
        oi[~M4] = 0
        del M4

        # Make weighted elevation grid.
        A = Zsub*W + zi*(1-W)
        A[np.where( np.isnan(Zsub) & ~np.isnan(zi))] =   zi[np.where( np.isnan(Zsub) & ~np.isnan(zi))]
        A[np.where(~np.isnan(Zsub) &  np.isnan(zi))] = Zsub[np.where(~np.isnan(Zsub) &  np.isnan(zi))]
        del zi, Zsub

        # Put strip subset back into full array.
        Z[r0:r1, c0:c1] = A
        del A

        # For the matchtag, just straight combination.
        M[r0:r1, c0:c1] = (Msub | mi)
        del Msub, mi

        # Make weighted ortho grid.
        Osub = Osub.astype(np.float32)
        Osub[np.where(Osub == 0)] = np.nan
        A = Osub*W + oi*(1-W)

        del W

        A[np.where( np.isnan(Osub) & ~np.isnan(oi))] =   oi[np.where( np.isnan(Osub) & ~np.isnan(oi))]
        A[np.where(~np.isnan(Osub) &  np.isnan(oi))] = Osub[np.where(~np.isnan(Osub) &  np.isnan(oi))]
        del Osub, oi

        A[np.where(np.isnan(A))] = 0  # convert back to uint16
        A = A.astype(np.uint16)

        O[r0:r1, c0:c1] = A
        del A

    # print "strip data broken"

    # Crop to data.
    if 'Z' in vars() and np.any(~np.isnan(Z)):
        rcrop, ccrop = cropnans(Z)
        if rcrop is not None:
            X = X[ccrop[0]:ccrop[1]]
            Y = Y[rcrop[0]:rcrop[1]]
            Z = Z[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            M = M[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            O = O[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            Z[np.where(np.isnan(Z))] = -9999

        fp_vertices = rat.getFPvertices(Z, X, Y, nodataVal=-9999)
        proj_ref = rat.getProjRef(os.path.join(demdir, demFiles_ordered[0]))
        proj4 = osr.SpatialReference(proj_ref).ExportToProj4()
    else:
        X = np.array([])
        Y = np.array([])
        Z = np.array([])
        M = np.array([])
        O = np.array([])
        proj_ref = None
        proj4 = None
        fp_vertices = (None, None)

    return X, Y, Z, M, O, trans, rmse, proj_ref, proj4, fp_vertices, demFiles_ordered
