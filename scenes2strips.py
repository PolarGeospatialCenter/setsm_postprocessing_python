# Version 3.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2016
# Translated from MATLAB code written by Ian Howat, Ohio State University, 2017

from __future__ import division
import os.path
import sys
import warnings
from traceback import print_exc

import ogr
import numpy as np
from scipy import interpolate, misc
from skimage import morphology

import raster_array_tools as rat
import test


# The spatial reference of the strip, set at the beginning of scenes2strips()
# to the spatial reference of the first scene dem in order and used for
# comparison to the spatial references of all other source raster files.
__STRIP_SPAT_REF__ = None


class RasterInputError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


def scenes2strips(demdir, demFiles, maskFileSuffix=None, max_coreg_rmse=1):
    """
    From MATLAB version in Github repo 'setsm_postprocessing', 3.0 branch:

    function [X,Y,Z,M,O,trans,rmse,f]=scenes2strips(varargin)
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
    %   [...]=scenes2strips(...,'maskFileSuffix',value) will apply the mask
    %   identified as the dem filename with the _dem.tif replaced by
    %   _maskFileSuffix.tif
    %   [...]=scenes2strips(...,'max_coreg_rmse',value) will set a new maximum
    %   coregistration error limit in meters (default=1). Errors above this
    %   limit will result in a segment break.
    %
    % Version 3.1, Ian Howat, Ohio State University, 2015.

    If maskFileSuffix='legacy', edge and data masks identified as the dem
    filename with the _dem.tif replaced by _edgemask.tif and _datamask.tif,
    respectively, will be applied.
    """

    # Order scenes in north-south or east-west direction by aspect ratio.
    print "ordering {} scenes".format(len(demFiles))
    demFiles_ordered = orderPairs(demdir, demFiles)

    # Initialize output stats.
    trans = np.zeros((3, len(demFiles_ordered)))
    rmse = np.zeros(len(demFiles_ordered))

    # Get projection reference of the first scene to be used in equality checks
    # with the projection reference of all scenes that follow.
    global __STRIP_SPAT_REF__
    __STRIP_SPAT_REF__ = rat.extractRasterParams(os.path.join(demdir, demFiles_ordered[0]), 'spat_ref')

    # File loop.
    for i in range(len(demFiles_ordered)):

        # Construct filenames.
        demFile = os.path.join(demdir, demFiles_ordered[i])
        matchFile = demFile.replace('dem.tif', 'matchtag.tif')
        orthoFile = demFile.replace('dem.tif', 'ortho.tif')
        maskFile = None
        if maskFileSuffix is not None:
            if maskFileSuffix == 'legacy':
                dataMaskFile = demFile.replace('dem.tif', 'datamask.tif')
                edgeMaskFile = demFile.replace('dem.tif', 'edgemask.tif')
            else:
                maskFile = demFile.replace('dem.tif', maskFileSuffix+'.tif')
        else:
            print "no mask applied"

        print "scene {} of {}: {}".format(i+1, len(demFiles_ordered), demFile)

        try:
            if maskFileSuffix == 'legacy':
                x, y, z, m, o, md, me = loaddata(demFile, matchFile, orthoFile, dataMaskFile, edgeMaskFile)
            else:
                x, y, z, m, o, md = loaddata(demFile, matchFile, orthoFile, maskFile)
        except:
            print "Data read error:"
            print_exc()
            print "...skipping"
            continue

        # Check for no data.
        if ~md.any():
            print "no data, skipping"
            continue

        # Apply masks.
        if maskFileSuffix == 'legacy':
            x, y, z, m, o = applyMasks(x, y, z, m, o, md, me)
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
        c0 = np.where(x[0]  == X)[0][0]
        c1 = np.where(x[-1] == X)[0][0] + 1
        r0 = np.where(y[0]  == Y)[0][0]
        r1 = np.where(y[-1] == Y)[0][0] + 1

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

        A[A == 0] = np.nan
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

        del z_cropped_data, Z_cropped_nodata

        # FIXME: What does the following commented out code
        # -f     (taken from Ian's scenes2strips.m) do?
        # %     tic
        # %     % USING REGIONFILL - Requires matlab 2015a or newer
        # %     A = regionfill(A,A==0) -1;
        # %     toc

        Ar = misc.imresize(A, 0.1, 'nearest', mode='F')

        # Locate pixels on outside of boundary of overlap region.
        Ar_nonzero = (Ar != 0)
        B = rat.bwboundaries_array(Ar_nonzero, noholes=True)
        B = np.where(B)

        cz_rows, cz_cols = [], []
        for cc in [
            [0,             0            ],
            [0,             Ar.shape[1]-1],
            [Ar.shape[0]-1, Ar.shape[1]-1],
            [Ar.shape[0]-1, 0            ]]:
            if Ar[tuple(cc)] == 0:
                cz_rows.append(cc[0])
                cz_cols.append(cc[1])

        if len(cz_rows) > 0:
            # Pixels outside of the convex hull of input points for interpolate.griddata
            # currently can't be extrapolated (by default they are filled with NaN),
            # so I do a trick here where I use interpolate.SmoothBivariateSpline to interpolate
            # from the boundary coordinates a value between 1 and 2 for every zero corner in Ar
            # (there are usually two) before doing the main interpolation.
            corner_zeros = (np.array(cz_rows), np.array(cz_cols))
            fn = interpolate.SmoothBivariateSpline(B[0], B[1], Ar[B].astype(np.float64), kx=2, ky=2)
            Ar_interp = fn.ev(corner_zeros[0], corner_zeros[1])
            Ar_interp[Ar_interp < 1] = 1
            Ar_interp[Ar_interp > 2] = 2
            Ar[corner_zeros] = Ar_interp

            # Add the corner coordinates to the list of boundary coordinates,
            # which will be used for interpolation.
            By = np.concatenate((B[0], corner_zeros[0]))
            Bx = np.concatenate((B[1], corner_zeros[1]))
            B = (By, Bx)

            del corner_zeros, fn, By, Bx
        del cz_rows, cz_cols

        # Use the coordinates and values of boundary pixels
        # to interpolate values for pixels with value zero.
        Ar_zero_coords = np.where(~Ar_nonzero)
        Ar_interp = interpolate.griddata(B, Ar[B].astype(np.float64), Ar_zero_coords, 'linear')
        Ar[Ar_zero_coords] = Ar_interp

        del Ar_interp, Ar_nonzero, Ar_zero_coords

        Ar = misc.imresize(Ar, A.shape, 'bilinear', mode='F')
        Ar[(A == 1) & (Ar != 1)] = 1
        Ar[(A == 2) & (Ar != 2)] = 2
        A = Ar - 1
        A[A < 0] = 0
        A[A > 1] = 1
        del Ar

        W = (~np.isnan(Zsub)).astype(np.float32)
        W[r[0]:r[1], c[0]:c[1]] = A
        del A
        W[np.isnan(Zsub) & np.isnan(z)] = np.nan

        # Shift weights so that more of the reference layer is kept.
        f0 = 0.25  # overlap fraction where ref z weight goes to zero
        f1 = 0.55  # overlap fraction where ref z weight goes to one

        W = (1/(f1-f0))*W - f0/(f1-f0)
        W[W > 1] = 1
        W[W < 0] = 0

        # Remove <25% edge of coverage from each in pair.
        # FIXME: Each throws "RuntimeWarning: invalid value encountered in less_equal".
        Zsub[W == 0] = np.nan
        Msub[W == 0] = 0
        Osub[W == 0] = 0

        z[W >= 1] = np.nan
        m[W >= 1] = 0
        o[W >= 1] = 0

        # Coregistration

        P0 = rat.getDataDensityMap(Msub[r[0]:r[1], c[0]:c[1]]) > 0.9
        P1 = rat.getDataDensityMap(   m[r[0]:r[1], c[0]:c[1]]) > 0.9

        # Coregister this scene to the strip mosaic.
        trans[:, i], rmse[i] = coregisterdems(
            Xsub[c[0]:c[1]], Ysub[r[0]:r[1]], Zsub[r[0]:r[1], c[0]:c[1]],
               x[c[0]:c[1]],    y[r[0]:r[1]],    z[r[0]:r[1], c[0]:c[1]],
            P0, P1
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
        mi[np.isnan(mi)] = 0  # convert back to uint8
        mi = mi.astype(np.bool)
        del m

        # TODO: Investigate ortho interpolation.
        # Interpolate ortho to same grid.
        oi = o.astype(np.float32)
        oi[oi == 0] = np.nan  # Set border to NaN so it won't be interpolated.
        oi = rat.interp2_gdal(xi, yi, oi, Xsub, Ysub, 'cubic')
        del o

        del Xsub, Ysub

        # Remove border 0's introduced by NaN interpolation.
        M3 = ~np.isnan(zi)
        M3 = rat.imerode_binary(M3, structure=np.ones((6, 6)))  # border cutline

        zi[~M3] = np.nan
        mi[~M3] = 0
        del M3

        # Remove border on orthos separately.
        M4 = (oi != 0)
        M4 = rat.imerode_binary(M4, structure=np.ones((6, 6)))
        oi[~M4] = 0
        del M4

        # Make weighted elevation grid.
        A = Zsub*W + zi*(1-W)
        A[ np.isnan(Zsub) & ~np.isnan(zi)] =   zi[ np.isnan(Zsub) & ~np.isnan(zi)]
        A[~np.isnan(Zsub) &  np.isnan(zi)] = Zsub[~np.isnan(Zsub) &  np.isnan(zi)]
        del zi, Zsub

        # Put strip subset back into full array.
        Z[r0:r1, c0:c1] = A
        del A

        # For the matchtag, just straight combination.
        M[r0:r1, c0:c1] = (Msub | mi)
        del Msub, mi

        # Make weighted ortho grid.
        Osub = Osub.astype(np.float32)
        Osub[Osub == 0] = np.nan
        A = Osub*W + oi*(1-W)

        del W

        A[ np.isnan(Osub) & ~np.isnan(oi)] =   oi[ np.isnan(Osub) & ~np.isnan(oi)]
        A[~np.isnan(Osub) &  np.isnan(oi)] = Osub[~np.isnan(Osub) &  np.isnan(oi)]
        del Osub, oi

        A[np.isnan(A)] = 0  # convert back to uint16
        A = A.astype(np.uint16)

        O[r0:r1, c0:c1] = A
        del A

    # Crop to data.
    if 'Z' in vars() and np.any(~np.isnan(Z)):
        rcrop, ccrop = cropnans(Z)
        if rcrop is not None:
            X = X[ccrop[0]:ccrop[1]]
            Y = Y[rcrop[0]:rcrop[1]]
            Z = Z[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            M = M[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            O = O[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            Z[np.isnan(Z)] = -9999
    else:
        X, Y, Z, M, O = None, None, None, None, None

    return X, Y, Z, M, O, trans, rmse, demFiles_ordered, __STRIP_SPAT_REF__


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
        raise RasterInputError("minimum array dimension is 3")

    interpflag = True
    if (len(x1) == len(x2)) and (len(y1) == len(y2)):
        if not np.any(x2 - x1) and not np.any(y2 - y1):
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
                m2n[np.isnan(m2n)] = 0  # convert back to uint8
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

        if not np.any(~np.isnan(dz)):
            print "No overlap"
            z2out = z2
            p = np.full(3, np.nan)
            d0 = np.nan
            break

        # Filter NaNs and outliers.
        # FIXME: The following throws "RuntimeWarning: invalid value encountered in less_equal".
        n = ~np.isnan(sx) & ~np.isnan(sy) & \
            (abs(dz - np.nanmedian(dz)) <= np.nanstd(dz))

        if not np.any(n):
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


def rectFootprint(*geoms):
    """
    Returns the smallest rectangular footprint that contains all input polygons,
    all as OGRGeometry objects.
    """
    minx = np.inf
    miny = np.inf
    maxx = -np.inf
    maxy = -np.inf

    for geom in geoms:
        cc = rat.wktToCoords(geom.ExportToWkt())
        cc_x = cc[:, 0]
        cc_y = cc[:, 1]
        minx = min(minx, np.min(cc_x))
        miny = min(miny, np.min(cc_y))
        maxx = max(maxx, np.max(cc_x))
        maxy = max(maxy, np.max(cc_y))

    fp_wkt = 'POLYGON (({} {},{} {},{} {},{} {},{} {}))'.format(
        minx, maxy,
        maxx, maxy,
        maxx, miny,
        minx, miny,
        minx, maxy
    )

    return ogr.Geometry(wkt=fp_wkt)


def orderPairs(demdir, fnames):
    """
    Scene order is determined in relation to *grid north* of the common projection
    by comparing total x-extent and y-extent of the scenes as a whole (aspect ratio).
    The larger of these extents determines the coordinate by which to do ordering.
    """
    R0 = np.zeros((len(fnames), 4))  # Matrix to place rectangular parameters of the raster images.
    indexed_geoms = []               # List to place rectangular (footprint) geometries of the rasters
                                     # as OGR geometries, each tupled with the index corresponding to the
                                     # filename of the raster it is extracted from.

    # Get rectangular parameters and geometries.
    for i in range(len(fnames)):
        cc, geom = rat.extractRasterParams(os.path.join(demdir, fnames[i]), 'corner_coords', 'geom')
        cc_x = cc[:, 0]
        cc_y = cc[:, 1]
        R0[i, :] = [min(cc_x), min(cc_y), max(cc_x)-min(cc_x), max(cc_y)-min(cc_y)]
        indexed_geoms.append((i, geom))

    # Calculate aspect ratio, ar = x-extent/y-extent
    ar =  (max(R0[:, 0] + R0[:, 2]) - min(R0[:, 0])) \
        / (max(R0[:, 1] + R0[:, 3]) - min(R0[:, 1]))

    first_fname_index = None
    if ar >= 1:
        # Scenes are in east-west direction; start with scene with minimum x.
        first_fname_index = np.argmin(R0[:, 0])
    else:
        # Scenes are in north-south direction; start with scene with minimum y.
        first_fname_index = np.argmin(R0[:, 1])

    # Start with the footprint of this scene and let the strip grow from there.
    footprint_geom = indexed_geoms[first_fname_index][1]
    ordered_fname_indices = [first_fname_index]
    del indexed_geoms[first_fname_index]

    # Loop through scene pair geometries and sequentially add the
    # next pair with the most overlap to the ordered indices list.
    for i in range(len(indexed_geoms)):
        overlap_area = [footprint_geom.Intersection(ind_geom[1]).GetArea() for ind_geom in indexed_geoms]
        if max(overlap_area) > 0:
            selected_tup = indexed_geoms[np.argmax(overlap_area)]
            scene_index, scene_geom = selected_tup
            # Extend the rectangular footprint geometry to include the new scene.
            footprint_geom = rectFootprint(footprint_geom, scene_geom)
            # Add the new scene to the file order at the last position.
            ordered_fname_indices.append(scene_index)
            # Remove the geometry of this scene (and its index) from the input list.
            indexed_geoms.remove(selected_tup)
        else:
            print "Break in overlap detected, returning this segment only"
            break

    return [fnames[i] for i in ordered_fname_indices]


def loaddata(demFile, matchFile, orthoFile, maskFile, edgemaskFile=None):
    """
    Load data files and perform basic conversions.
    """
    global __STRIP_SPAT_REF__

    z, x_dem, y_dem, spat_ref = rat.extractRasterParams(demFile, 'z', 'x', 'y', 'spat_ref')
    if spat_ref.IsSame(__STRIP_SPAT_REF__) != 1:
        raise RasterInputError("demFile '{}' spatial reference ({}) mismatch with strip spatial reference ({})".format(
                               demFile, spat_ref.ExportToWkt(), __STRIP_SPAT_REF__.ExportToWkt()))

    m = rat.extractRasterParams(matchFile, 'z').astype(np.bool)
    if m.shape != z.shape:
        print "WARNING: matchFile '{}' dimensions differ from dem dimensions".format(matchFile)
        print "Interpolating to dem dimensions"
        x, y = rat.extractRasterParams(matchFile, 'x', 'y')
        m = rat.interp2_gdal(x, y, m.astype(np.float32), x_dem, y_dem, 'nearest')
        m[np.isnan(m)] = 0  # Convert back to bool.
        m = m.astype(np.bool)

    if os.path.isfile(orthoFile):
        o = rat.extractRasterParams(orthoFile, 'z').astype(np.uint16)
        if o.shape != z.shape:
            print "WARNING: orthoFile '{}' dimensions differ from dem dimensions".format(orthoFile)
            print "Interpolating to dem dimensions"
            x, y = rat.extractRasterParams(orthoFile, 'x', 'y')
            # TODO: Is the following line necessary?
            o[np.isnan(o)] = np.nan  # Set border to NaN so it won't be interpolated.
            o = rat.interp2_gdal(x, y, o.astype(np.float32), x_dem, y_dem, 'cubic')
            o[np.isnan(o)] = 0  # Convert back to uint16.
            o = o.astype(np.uint16)
    else:
        o = np.zeros(z.shape, dtype=np.uint16)

    if maskFile is not None:
        md = rat.extractRasterParams(maskFile, 'z').astype(np.bool)
        if md.shape != z.shape:
            raise RasterInputError("maskFile '{}' has wrong dimensions".format(maskFile))
    else:
        md = np.ones(z.shape, dtype=np.bool)

    if edgemaskFile is not None:
        me = rat.extractRasterParams(edgemaskFile, 'z').astype(np.bool)
        if me.shape != z.shape:
            raise RasterInputError("edgemaskFile '{}' has wrong dimensions".format(maskFile))

    # A pixel with a value of -9999 is a nodata pixel; interpret it as NaN.
    z[(z < -100) | (z == 0) | (z == -np.inf) | (z == np.inf)] = np.nan

    if edgemaskFile is not None:
        return x_dem, y_dem, z, m, o, md, me
    else:
        return x_dem, y_dem, z, m, o, md


def applyMasks(x, y, z, m, o, md, me=None):
    # TODO: Write docstring.

    z[~md] = np.nan
    m[~md] = 0
    if me is not None:
        o[~me] = 0

    # If there is any good data, crop the matrices of bordering NaNs.
    if np.any(~np.isnan(z)):
        rowcrop, colcrop = cropnans(z)

        x = x[colcrop[0]:colcrop[1]]
        y = y[rowcrop[0]:rowcrop[1]]
        z = z[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]
        m = m[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]
        o = o[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]

    return x, y, z, m, o


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


def regrid(x, y, z, m, o):
    # TODO: Write docstring.

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    xi = np.arange(x[0] + dx - ((x[0]/dx) % 1)*dx, x[-1], dx)
    yi = np.arange(y[0] + dy - ((y[0]/dy) % 1)*dy, y[-1], dy)

    zi = rat.interp2_gdal(x, y, z, xi, yi, 'linear').astype(np.float32)

    m = rat.interp2_gdal(x, y, m.astype(np.float32), xi, yi, 'nearest')
    m[np.isnan(m)] = 0  # Convert back to uint8.
    m = m.astype(np.bool)

    # Interpolate ortho to same grid.
    o = o.astype(np.float32)
    o[np.isnan(z)] = np.nan  # Set border to NaN so it won't be interpolated.
    o = rat.interp2_gdal(x, y, o, xi, yi, 'cubic')
    o[np.isnan(o)] = 0  # Convert back to uint16.
    o = o.astype(np.uint16)

    return xi, yi, zi, m, o


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
