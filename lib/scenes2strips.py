
# Version 3.1; Erik Husby; Polar Geospatial Center, University of Minnesota; 2018
# Translated from MATLAB code written by Ian Howat, Ohio State University, 2018


from __future__ import division
import os.path
import sys
import warnings
from traceback import print_exc

import ogr
import numpy as np
from scipy import interpolate

if sys.version_info[0] < 3:
    import raster_array_tools as rat
    from filter_scene import getDataDensityMap
else:
    from lib import raster_array_tools as rat
    from lib.filter_scene import getDataDensityMap
from testing.test import validateTestFileSave


# The spatial reference of the strip, set at the beginning of scenes2strips()
# to the spatial reference of the first scene DEM in order and used for
# comparison to the spatial references of all other source raster files.
__STRIP_SPAT_REF__ = None


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class SpatialRefError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class RasterDimensionError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


def scenes2strips(demdir, demFiles,
                  maskSuffix=None, filter_options=(),
                  trans_guess=None, rmse_guess=None, hold_guess=False,
                  max_coreg_rmse=1):
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
    %   _maskSuffix
    %   [...]=scenes2strips(...,'max_coreg_rmse',value) will set a new maximum
    %   coregistration error limit in meters (default=1). Errors above this
    %   limit will result in a segment break.
    %
    % Version 3.1, Ian Howat, Ohio State University, 2015.

    If maskFileSuffix='edgemask', edge and data masks identified as the DEM
    filename with the _dem.tif replaced by _edgemask.tif and _datamask.tif,
    respectively, will be applied.

    """
    from batch_scenes2strips import getDemSuffix, selectBestMatchtag
    demSuffix = getDemSuffix(demFiles[0])

    # Order scenes in north-south or east-west direction by aspect ratio.
    if trans_guess is None and rmse_guess is None:
        print("Ordering {} scenes".format(len(demFiles)))
        demFiles_ordered = orderPairs(demdir, demFiles)
    elif trans_guess is not None and trans_guess.shape[1] != len(demFiles):
        raise InvalidArgumentError("`trans_guess` array must be of shape (3, N) where N=len(demFiles), "
                                   "but was {}".format(trans_guess.shape))
    elif rmse_guess is not None and rmse_guess.shape[1] != len(demFiles):
        raise InvalidArgumentError("`rmse_guess` array must be of shape (1, N) where N=len(demFiles), "
                                   "but was {}".format(rmse_guess.shape))
    else:
        # Files should already be properly ordered if a guess is provided.
        # Running `orderPairs` on them could detrimentally change their order.
        demFiles_ordered = list(demFiles)

    # Initialize output stats.
    trans = np.zeros((3, len(demFiles_ordered))) if trans_guess is None else trans_guess.copy()
    rmse = np.zeros((1, len(demFiles_ordered))) if rmse_guess is None else rmse_guess.copy()

    # Get projection reference of the first scene to be used in equality checks
    # with the projection reference of all scenes that follow.
    global __STRIP_SPAT_REF__
    __STRIP_SPAT_REF__ = rat.extractRasterData(os.path.join(demdir, demFiles_ordered[0]), 'spat_ref')

    # File loop.
    segment_break = False
    for i in range(len(demFiles_ordered)):

        # Construct filenames.
        demFile = os.path.join(demdir, demFiles_ordered[i])
        matchFile = selectBestMatchtag(demFile)
        orthoFile = demFile.replace(demSuffix, 'ortho.tif')
        if maskSuffix is None:
            print("No mask applied")
            maskFile = None
        else:
            maskFile = demFile.replace(demSuffix, maskSuffix)

        print("scene {} of {}: {}".format(i+1, len(demFiles_ordered), demFile))

        try:
            x, y, z, m, o, md = loadData(demFile, matchFile, orthoFile, maskFile)
        except:
            print("Data read error:")
            print_exc()
            print("...skipping")
            continue

        # Apply masks.
        x, y, z, m, o, md = applyMasks(x, y, z, m, o, md, filter_options, maskSuffix)

        # Check for no data.
        if np.all(np.isnan(z)):
            print("All data is masked, skipping")
            continue

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Fix grid so that x, y coordinates of
        # pixels in overlapping scenes will match up.
        if ((x[1] / dx) % 1 != 0) or ((y[1] / dy) % 1 != 0):
            x, y, z, m, o, md = regrid(x, y, z, m, o, md)

        # If this is the first scene in strip,
        # set as strip and continue to next scene.
        if 'X' not in vars():
            X, Y, Z, M, O, MD = x, y, z, m, o, md
            del x, y, z, m, o, md
            continue

        # Pad new arrays to stabilize interpolation.
        buff = int(10*dx + 1)
        z = np.pad(z, buff, 'constant', constant_values=np.nan)
        m = np.pad(m, buff, 'constant', constant_values=0)
        o = np.pad(o, buff, 'constant', constant_values=0)
        md = np.pad(md, buff, 'constant', constant_values=1)
        x = np.concatenate((x[0]  - dx*np.arange(buff, 0, -1), x,
                            x[-1] + dx*np.arange(1, buff+1)))
        y = np.concatenate((y[0]  + dx*np.arange(buff, 0, -1), y,
                            y[-1] - dx*np.arange(1, buff+1)))

        # Expand strip coverage to encompass new scene.
        if x[0] < X[0]:
            X1 = np.arange(x[0], X[0], dx)
            X = np.concatenate((X1, X))
            Z, M, O, MD = expandCoverage(Z, M, O, MD, X1, direction='left')
            del X1
        if x[-1] > X[-1]:
            X1 = np.arange(X[-1]+dx, x[-1]+dx, dx)
            X = np.concatenate((X, X1))
            Z, M, O, MD = expandCoverage(Z, M, O, MD, X1, direction='right')
            del X1
        if y[0] > Y[0]:
            Y1 = np.arange(y[0], Y[0], -dx)
            Y = np.concatenate((Y1, Y))
            Z, M, O, MD = expandCoverage(Z, M, O, MD, Y1, direction='up')
            del Y1
        if y[-1] < Y[-1]:
            Y1 = np.arange(Y[-1]-dx, y[-1]-dx, -dx)
            Y = np.concatenate((Y, Y1))
            Z, M, O, MD = expandCoverage(Z, M, O, MD, Y1, direction='down')
            del Y1

        # Map new DEM pixels to swath. These must return integers. If not,
        # interpolation will be required, which is currently not supported.
        c0 = np.where(x[0]  == X)[0][0]
        c1 = np.where(x[-1] == X)[0][0] + 1
        r0 = np.where(y[0]  == Y)[0][0]
        r1 = np.where(y[-1] == Y)[0][0] + 1

        # Crop to overlap.
        Xsub = np.copy(X[c0:c1])
        Ysub = np.copy(Y[r0:r1])
        Zsub = np.copy(Z[r0:r1, c0:c1])
        Msub = np.copy(M[r0:r1, c0:c1])
        Osub = np.copy(O[r0:r1, c0:c1])
        MDsub = np.copy(MD[r0:r1, c0:c1])

        # NEW MOSAICKING CODE

        cmin = 1000  # Minimum data cluster area for 2m.

        # Crop to just region of overlap.
        A = (~np.isnan(Zsub) & ~np.isnan(z))

        # Check for segment break.
        if np.count_nonzero(A) <= cmin:
            print("Not enough overlap, segment break")
            segment_break = True
            break

        r, c = cropBorder(A, 0, buff)

        # Make overlap mask removing isolated pixels.
        strip_nodata =  np.isnan(Zsub[r[0]:r[1], c[0]:c[1]])
        scene_data   = ~np.isnan(   z[r[0]:r[1], c[0]:c[1]])

        # Nodata in strip and data in scene is a one.
        A = rat.bwareaopen(strip_nodata & scene_data, cmin, in_place=True).astype(np.float32)

        # TODO: Remove redundant scenes from strip metadata?
        # Check for redundant scene.
        if np.count_nonzero(A) <= cmin:
            print("Redundant scene, skipping")
            continue

        # Data in strip and nodata in scene is a two.
        A[rat.bwareaopen(~strip_nodata & ~scene_data, cmin, in_place=True)] = 2

        del strip_nodata, scene_data

        Ar = rat.imresize(A, 0.1, 'nearest')

        # Locate pixels on outside of boundary of overlap region.
        Ar_nonzero = (Ar != 0)
        B = np.where(rat.bwboundaries_array(Ar_nonzero, noholes=True))

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
            # currently can't be extrapolated linear-ly (by default they are filled with NaN).
            # Let this region be filled with NaN, but we get a better edge on the convex hull
            # by changing corner pixels that are zero to NaN.
            corner_zeros = (np.array(cz_rows), np.array(cz_cols))
            Ar[corner_zeros] = np.nan

            # Add the corner coordinates to the list of boundary coordinates,
            # which will be used for interpolation.
            By = np.concatenate((B[0], corner_zeros[0]))
            Bx = np.concatenate((B[1], corner_zeros[1]))
            B = (By, Bx)

            del corner_zeros, By, Bx
        del cz_rows, cz_cols

        # Use the coordinates and values of boundary pixels
        # to interpolate values for pixels with zero value.
        Ar_zero_coords = np.where(~Ar_nonzero)
        Ar_interp = interpolate.griddata(B, Ar[B], Ar_zero_coords, 'linear')
        Ar[Ar_zero_coords] = Ar_interp

        del Ar_nonzero, Ar_zero_coords, Ar_interp

        # Fill in the regions outside the convex hull of the boundary points
        # using a nearest extrapolation of all points on the boundary of the
        # overlap region (including the gaps that were just interpolated).
        Ar_outer = np.isnan(Ar)
        Ar_outer_coords = np.where(Ar_outer)
        B = np.where(rat.bwboundaries_array(~Ar_outer))
        Ar_extrap = interpolate.griddata(B, Ar[B], Ar_outer_coords, 'nearest')
        Ar[Ar_outer_coords] = Ar_extrap
        # Nearest extrapolation is granular, so it is smoothed.
        Ar_smooth = rat.moving_average(Ar, 5, zero_border=False)
        Ar[Ar_outer] = Ar_smooth[Ar_outer]

        del Ar_outer, Ar_outer_coords, Ar_extrap

        Ar = rat.imresize(Ar, A.shape, 'bilinear')
        Ar[(A == 1) & (Ar != 1)] = 1
        Ar[(A == 2) & (Ar != 2)] = 2
        A = np.clip(Ar - 1, 0, 1)
        del Ar

        W = (~np.isnan(Zsub)).astype(np.float32)
        W[r[0]:r[1], c[0]:c[1]] = A
        del A
        W[np.isnan(Zsub) & np.isnan(z)] = np.nan

        # Shift weights so that more of the reference layer is kept.
        f0 = 0.25  # overlap fraction where ref z weight goes to zero
        f1 = 0.55  # overlap fraction where ref z weight goes to one

        W = np.clip((1/(f1-f0))*W - f0/(f1-f0), 0, 1)

        # Remove <25% edge of coverage from each in pair.
        strip_nodata = (W == 0)
        Zsub[strip_nodata] = np.nan
        Msub[strip_nodata] = 0
        Osub[strip_nodata] = 0
        MDsub[strip_nodata] = 0

        scene_nodata = (W == 1)
        z[scene_nodata] = np.nan
        m[scene_nodata] = 0
        o[scene_nodata] = 0
        md[scene_nodata] = 0

        del strip_nodata, scene_nodata

        # Coregistration

        P0 = getDataDensityMap(Msub[r[0]:r[1], c[0]:c[1]]) > 0.9

        # Check for segment break.
        if not np.any(P0):
            print("Not enough data overlap, segment break")
            segment_break = True
            break

        P1 = getDataDensityMap(m[r[0]:r[1], c[0]:c[1]]) > 0.9

        # TODO: Remove redundant scenes from strip metadata?
        # Check for redundant scene.
        if not np.any(P1):
            print("Redundant scene, skipping")
            continue

        # Coregister this scene to the strip mosaic.
        if trans_guess is not None and hold_guess and rmse_guess is None:
            pass
        else:
            trans[:, i], rmse[0, i] = coregisterdems(
                Xsub[c[0]:c[1]], Ysub[r[0]:r[1]], Zsub[r[0]:r[1], c[0]:c[1]],
                   x[c[0]:c[1]],    y[r[0]:r[1]],    z[r[0]:r[1], c[0]:c[1]],
                P0, P1, (trans_guess[:, i] if trans_guess is not None else None)
            )[[1, 2]]

            if rmse_guess is not None:
                rmse_change = rmse[0, i] - rmse_guess[0, i]
                stats_str = "First-run rmse={:.3f}, second-run rmse={:.3f}, change={:.3f}".format(
                    rmse_guess[0, i], rmse[0, i], rmse_change)
                print(stats_str)
                statsFile = validateTestFileSave('s2s_stats.log', overwrite=True)
                print("Writing to {} ...".format(statsFile))
                statsFile_fp = open(statsFile, 'a')
                statsFile_fp.write('{}: {}\n'.format(demFiles_ordered[i], stats_str))
                statsFile_fp.close()

            if hold_guess:
                if trans_guess is not None:
                    trans = trans_guess.copy()
                if rmse_guess is not None:
                    rmse = rmse_guess.copy()

        # Check for segment break.
        if np.isnan(rmse[0, i]) or rmse[0, i] > max_coreg_rmse:
            print("Unable to coregister, segment break")
            segment_break = True
            break

        # Interpolation grid
        xi = x - trans[1, i]
        yi = y - trans[2, i]

        # Check that uniform spacing is maintained (sometimes rounding errors).
        if len(np.unique(np.diff(xi))) > 1:
            xi = np.round(xi, 4)
        if len(np.unique(np.diff(yi))) > 1:
            yi = np.round(yi, 4)

        # Interpolate floating data to the reference grid.
        zi = rat.interp2_gdal(xi, yi, z-trans[0, i], Xsub, Ysub, 'linear')
        del z

        # Interpolate matchtag to the same grid.
        mi = rat.interp2_gdal(xi, yi, m.astype(np.float32), Xsub, Ysub, 'nearest')
        mi[np.isnan(mi)] = 0  # convert back to uint8
        mi = mi.astype(np.bool)
        del m

        # Interpolate ortho to same grid.
        oi = o.astype(np.float32)
        oi[oi == 0] = np.nan  # Set border to NaN so it won't be interpolated.
        oi = rat.interp2_gdal(xi, yi, oi, Xsub, Ysub, 'cubic')
        del o

        # Interpolate mask to the same grid.
        mdi = rat.interp2_gdal(xi, yi, md.astype(np.float32), Xsub, Ysub, 'nearest')
        mdi[np.isnan(mdi)] = 0  # convert back to uint8
        mdi = mdi.astype(np.uint8)
        del md

        del Xsub, Ysub

        # Remove border 0's introduced by NaN interpolation.
        M3 = ~np.isnan(zi)
        M3 = rat.imerode(M3, 6)  # border cutline
        zi[~M3] = np.nan
        mi[~M3] = 0  # also apply to matchtag
        del M3

        # Remove border on orthos separately.
        M4 = ~np.isnan(oi)
        M4 = rat.imerode(M4, 6)
        oi[~M4] = np.nan
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

        # For the mask, bitwise combination.
        MD[r0:r1, c0:c1] = np.bitwise_or(MDsub, mdi)
        del MDsub, mdi

    if segment_break:
        demFiles_ordered = demFiles_ordered[:i]
        trans = trans[:, :i]
        rmse = rmse[:, :i]

    # Crop to data.
    if 'Z' in vars() and np.any(~np.isnan(Z)):
        rcrop, ccrop = cropBorder(Z, np.nan)
        if rcrop is not None:
            X = X[ccrop[0]:ccrop[1]]
            Y = Y[rcrop[0]:rcrop[1]]
            Z = Z[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            M = M[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            O = O[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            MD = MD[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            Z[np.isnan(Z)] = -9999
    else:
        X, Y, Z, M, O, MD = None, None, None, None, None, None

    return X, Y, Z, M, O, MD, trans, rmse, demFiles_ordered, __STRIP_SPAT_REF__


def coregisterdems(x1, y1, z1, x2, y2, z2, m1=None, m2=None, trans_guess=None):
    """
    % COREGISTERDEM registers a floating to a reference DEM

    % [z2r,trans,rms] = coregisterdems(x1,y1,z1,x2,y2,z2) registers the
    % floating DEM in 2D array z2 with coordinate vectors x2 and y2 to the
    % reference DEM in z1 using the iterative procedure in Nuth and Kaab,
    % 2010. z2r is the regiestered DEM, p is the z,x,y transformation
    % parameters and rms is the rms of the transformation in the vertical.

    """
    if (m1 is None) ^ (m2 is None):
        raise InvalidArgumentError("Either none of both of arguments 'm1' and 'm2' must be provided")

    # Maximum offset allowed
    maxp = 15

    if len(x1) < 3 or len(y1) < 3 or len(x2) < 3 or len(y2) < 3:
        raise RasterDimensionError("Minimum array dimension is 3")

    interpflag = True
    if (len(x1) == len(x2)) and (len(y1) == len(y2)):
        if not np.any(x2 - x1) and not np.any(y2 - y1):
            interpflag = False

    # initial trans variable
    if trans_guess is not None:
        p = np.reshape(trans_guess, (3, 1))
    else:
        p = np.zeros((3, 1))
    d0 = np.inf            # initial rmse

    rx = x1[1] - x1[0]     # coordinate spacing
    pn = p.copy()          # iteration variable
    it = 1                 # iteration step

    while it:

        if interpflag:
            # Interpolate the floating data to the reference grid.
            z2n = rat.interp2_gdal(x2 - pn[1], y2 - pn[2], z2 - pn[0],
                                   x1, y1, 'linear')
            if m2 is not None:
                m2n = rat.interp2_gdal(x2 - pn[1], y2 - pn[2], m2.astype(np.float32),
                                       x1, y1, 'nearest')
                m2n[np.isnan(m2n)] = 0  # convert back to uint8
                m2n = m2n.astype(np.bool)
        else:
            z2n = z2 - pn[0]
            if m2 is not None:
                m2n = m2

        interpflag = True

        # Slopes
        sy, sx = np.gradient(z2n, rx)
        sx = -sx

        sys.stdout.write("Planimetric Correction Iteration {} ".format(it))

        # Difference grids.
        dz = z2n - z1

        if m1 is not None and m2 is not None:
            dz[~m2n | ~m1] = np.nan

        if not np.any(~np.isnan(dz)):
            print("No overlap")
            z2out = z2
            p = np.full((3, 1), np.nan)
            d0 = np.nan
            break

        # Filter NaNs and outliers.
        n = (~np.isnan(sx) & ~np.isnan(sy)
             & (np.abs(dz - np.nanmedian(dz)) <= np.nanstd(dz)))

        if not np.any(n):
            sys.stdout.write("regression failure, all overlap filtered\n")
            p = np.full((3, 1), np.nan)  # initial trans variable
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
        px = np.array([np.linalg.lstsq(X, dz[n])[0]]).T

        pn = p + px

        # Display offsets.
        sys.stdout.write("offset(z,x,y): {:.3f}, {:.3f}, {:.3f}\n".format(
            pn[0, 0], pn[1, 0], pn[2, 0]))

        if np.any(np.abs(pn[1:]) > maxp):
            sys.stdout.write(
                "maximum horizontal offset reached, "
                "returning median vertical offset: {:.3f}\n".format(meddz)
            )
            p = np.array([[meddz, 0, 0]]).T
            d0 = d00
            z2out = z2 - meddz
            break

        # Update iteration vars.
        it += 1

    return np.array([z2out, p.T[0], d0])


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
        cc, geom = rat.extractRasterData(os.path.join(demdir, fnames[i]), 'corner_coords', 'geom')
        cc_x = cc[:, 0]
        cc_y = cc[:, 1]
        R0[i, :] = [min(cc_x), min(cc_y), max(cc_x)-min(cc_x), max(cc_y)-min(cc_y)]
        indexed_geoms.append((i, geom))

    # Calculate aspect ratio, ar = x-extent/y-extent
    ar = (  (max(R0[:, 0] + R0[:, 2]) - min(R0[:, 0]))
          / (max(R0[:, 1] + R0[:, 3]) - min(R0[:, 1])))

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
            print("Break in overlap detected, returning this segment only")
            break

    return [fnames[i] for i in ordered_fname_indices]


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


def loadData(demFile, matchFile, orthoFile, maskFile):
    """
    Load data files and perform basic conversions.
    """
    global __STRIP_SPAT_REF__

    z, x_dem, y_dem, spat_ref = rat.extractRasterData(demFile, 'array', 'x', 'y', 'spat_ref')
    if spat_ref.IsSame(__STRIP_SPAT_REF__) != 1:
        raise SpatialRefError("demFile '{}' spatial reference ({}) mismatch with strip spatial reference ({})".format(
                              demFile, spat_ref.ExportToWkt(), __STRIP_SPAT_REF__.ExportToWkt()))

    # A DEM pixel with a value of -9999 is a nodata pixel; interpret it as NaN.
    # TODO: Ask Ian about the following interpretation of nodata values.
    z[(z < -100) | (z == 0) | (z == -np.inf) | (z == np.inf)] = np.nan

    m = rat.extractRasterData(matchFile, 'array').astype(np.bool)
    if m.shape != z.shape:
        warnings.warn("matchFile '{}' dimensions differ from dem dimensions".format(matchFile)
                     +"\nInterpolating to dem dimensions")
        x, y = rat.extractRasterData(matchFile, 'x', 'y')
        m = rat.interp2_gdal(x, y, m.astype(np.float32), x_dem, y_dem, 'nearest')
        m[np.isnan(m)] = 0  # Convert back to bool/uint8.
        m = m.astype(np.bool)

    if os.path.isfile(orthoFile):
        o = rat.extractRasterData(orthoFile, 'array')
        if o.shape != z.shape:
            warnings.warn("orthoFile '{}' dimensions differ from dem dimensions".format(orthoFile)
                         +"\nInterpolating to dem dimensions")
            x, y = rat.extractRasterData(orthoFile, 'x', 'y')
            o[o == 0] = np.nan  # Set border to NaN so it won't be interpolated.
            o = rat.interp2_gdal(x, y, o.astype(np.float32), x_dem, y_dem, 'cubic')
            o[np.isnan(o)] = 0  # Convert back to uint16.
            o = rat.astype_round_and_crop(o, np.uint16, allow_modify_array=True)
    else:
        o = np.zeros(z.shape, dtype=np.uint16)

    if maskFile is None:
        md = np.zeros_like(z, dtype=np.uint8)
    else:
        md = rat.extractRasterData(maskFile, 'array').astype(np.uint8)
        if md.shape != z.shape:
            raise RasterDimensionError("maskFile '{}' dimensions {} do not match dem dimensions {}".format(
                                       maskFile, md.shape, z.shape))

    return x_dem, y_dem, z, m, o, md


def applyMasks(x, y, z, m, o, md, filter_options=(), maskSuffix=None):
    """
    Apply masks to the scene DEM, matchtag, and ortho matrices.
    """
    if sys.version_info[0] < 3:
        from filter_scene import mask_v2, MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT
    else:
        from lib.filter_scene import mask_v2, MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT

    # if len(filter_options) > 0:
    #     mask_select = np.bitwise_and(md, np.full_like(md, 2**MASKCOMP_EDGE_BIT)).astype(np.bool)
    #     if 'nowater' not in filter_options:
    #         mask_select[np.bitwise_and(md, np.full_like(md, 2**MASKCOMP_WATER_BIT)).astype(np.bool)] = 1
    #     if 'nocloud' not in filter_options:
    #         mask_select[np.bitwise_and(md, np.full_like(md, 2**MASKCOMP_CLOUD_BIT)).astype(np.bool)] = 1
    # else:
    #     mask_select = md
    mask_select = np.copy(md)
    if len(filter_options) > 0:
        mask_ones = np.ones_like(mask_select)
        if 'nowater' in filter_options:
            np.bitwise_and(mask_select, ~np.left_shift(mask_ones, MASKCOMP_WATER_BIT), out=mask_select)
        if 'nocloud' in filter_options:
            np.bitwise_and(mask_select, ~np.left_shift(mask_ones, MASKCOMP_CLOUD_BIT), out=mask_select)

    mask = (mask_select > 0)
    if maskSuffix in ('mask.tif', 'bitmask.tif'):
        mask = mask_v2(postprocess_mask=mask, postprocess_res=abs(x[1]-x[0]))

    z[mask] = np.nan
    m[mask] = 0

    # If there is any good data, crop the matrices of bordering NaNs.
    if np.any(~np.isnan(z)):
        rowcrop, colcrop = cropBorder(z, np.nan)

        x = x[colcrop[0]:colcrop[1]]
        y = y[rowcrop[0]:rowcrop[1]]
        z = z[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]
        m = m[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]
        o = o[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]
        md = md[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]

    return x, y, z, m, o, md


def cropBorder(matrix, border_val, buff=0):
    """
    Crop matrix of a bordering value.
    """
    data = None
    if np.isnan(border_val):
        data = ~np.isnan(matrix)
    elif border_val == 0 and matrix.dtype == np.bool:
        data = matrix
    else:
        data = (matrix != border_val)

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


def regrid(x, y, z, m, o, md):
    """
    Interpolate scene DEM, matchtag, and ortho matrices
    to a new set of x-y grid coordinates.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    xi = np.arange(x[0] + dx - np.fmod(x[0]/dx, 1)*dx, x[-1]+0.001*dx, dx)
    yi = np.arange(y[0]      - np.fmod(y[0]/dy, 1)*dy, y[-1]+0.001*dy, dy)

    zi = rat.interp2_gdal(x, y, z, xi, yi, 'linear')

    m = rat.interp2_gdal(x, y, m.astype(np.float32), xi, yi, 'nearest')
    m[np.isnan(m)] = 0  # Convert back to uint8.
    m = m.astype(np.bool)

    # Interpolate ortho to same grid.
    o = o.astype(np.float32)
    o[np.isnan(z)] = np.nan  # Set border to NaN so it won't be interpolated.
    o = rat.interp2_gdal(x, y, o, xi, yi, 'cubic')
    o[np.isnan(o)] = 0  # Convert back to uint16.
    o = rat.astype_round_and_crop(o, np.uint16, allow_modify_array=True)

    md = rat.interp2_gdal(x, y, md.astype(np.float32), xi, yi, 'nearest')
    md[np.isnan(md)] = 0  # Convert back to uint8.
    md = md.astype(np.uint8)

    return xi, yi, zi, m, o, md


def expandCoverage(Z, M, O, MD, R1, direction):
    """
    Expand strip coverage for DEM, matchtag, and ortho matrices
    based upon the direction of expansion.
    When (X1/Y1) is passed in for R1,
    (('left' or 'right') / ('up' or 'down'))
    must be passed in for direction.
    """
    if direction in ('up', 'down'):
        # R1 is Y1.
        Z1 = np.full((R1.size, Z.shape[1]), np.nan, dtype=np.float32)
        M1 = np.full((R1.size, M.shape[1]), False,  dtype=np.bool)
        O1 = np.full((R1.size, O.shape[1]), 0,      dtype=np.uint16)
        MD1 = np.full((R1.size, MD.shape[1]), 1,    dtype=np.uint8)
        axis_num = 0
    else:
        # R1 is X1.
        Z1 = np.full((Z.shape[0], R1.size), np.nan, dtype=np.float32)
        M1 = np.full((M.shape[0], R1.size), False,  dtype=np.bool)
        O1 = np.full((O.shape[0], R1.size), 0,      dtype=np.uint16)
        MD1 = np.full((MD.shape[0], R1.size), 1,    dtype=np.uint8)
        axis_num = 1

    if direction in ('left', 'up'):
        pairs = [[Z1, Z], [M1, M], [O1, O], [MD1, MD]]
    else:
        pairs = [[Z, Z1], [M, M1], [O, O1], [MD, MD1]]

    Z, M, O, MD = [np.concatenate(p, axis=axis_num) for p in pairs]

    return Z, M, O, MD
