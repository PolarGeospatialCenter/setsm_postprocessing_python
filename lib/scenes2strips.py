
# Version 3.1; Erik Husby; Polar Geospatial Center, University of Minnesota; 2019
# Translated from MATLAB code written by Ian Howat, Ohio State University, 2018


from __future__ import division
import os
import sys
import traceback
import warnings

from osgeo import ogr
import numpy as np
import scipy.stats
from scipy import interpolate

if sys.version_info[0] < 3:
    import raster_array_tools as rat
    import setsm_srs
    from filter_scene import getDataDensityMap, readSceneMeta, rescaleDN
else:
    from lib import raster_array_tools as rat
    from lib import setsm_srs
    from lib.filter_scene import getDataDensityMap, readSceneMeta, rescaleDN


# The spatial reference of the strip, set at the beginning of scenes2strips()
# to the spatial reference of the first scene DEM in order and used for
# comparison to the spatial references of all other source raster files.
__STRIP_SPAT_REF__ = 'NULL'

# The Catalog ID of "Image 1" as parsed from the output scene metadata files for
# an intrack stereo SETSM DEM strip. It is expected that all ortho scenes in the
# intrack strip correspond to the same Catalog ID.
__INTRACK_ORTHO_CATID__ = None

HOLD_GUESS_OFF = 0
HOLD_GUESS_ALL = 1
HOLD_GUESS_UPDATE_RMSE = 2


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class SpatialRefError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class RasterDimensionError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class MetadataError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


def scenes2strips(demFiles,
                  maskSuffix=None, filter_options=(), max_coreg_rmse=1,
                  trans_guess=None, trans_err_guess=None, rmse_guess=None,
                  hold_guess=HOLD_GUESS_OFF, check_guess=True,
                  target_srs=None, use_second_ortho=False, remerge_strips=False,
                  force_single_scene_strips=False,
                  preceding_scene_geom_union=None):
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
    from batch_scenes2strips import getDemSuffix, selectBestMatchtag, selectBestOrtho, selectBestOrtho2
    demSuffix = getDemSuffix(demFiles[0])

    cluster_min_px = 1000  # Minimum data cluster area for 2m.
    add_min_px = 50000  # Minimum number of unmasked pixels scene must add to existing segment to not be skipped.
    add_min_area = add_min_px * (2**2)

    # Order scenes in north-south or east-west direction by aspect ratio.
    num_scenes = len(demFiles)
    if trans_guess is None and trans_err_guess is None and rmse_guess is None:
        print("Ordering {} scenes".format(num_scenes))
        demFiles_ordered, demFiles_ordered_by_direction = orderPairs(demFiles)
    elif trans_err_guess is not None and trans_guess is None:
        raise InvalidArgumentError("`trans_guess_err` argument can only be used in conjunction "
                                   "with `trans_guess` argument")
    elif trans_guess is not None and trans_guess.shape[1] != num_scenes:
        raise InvalidArgumentError("`trans_guess` array must be of shape (3, N) where N is the number "
                                   "of scenes in `demFiles`, but was {}".format(trans_guess.shape))
    elif rmse_guess is not None and rmse_guess.shape[1] != num_scenes:
        raise InvalidArgumentError("`rmse_guess` array must be of shape (1, N) where N is the number "
                                   "of scenes in `demFiles`, but was {}".format(rmse_guess.shape))
    else:
        # Files should already be properly ordered if a guess is provided.
        # Running `orderPairs` on them could detrimentally change their order.
        demFiles_ordered = list(demFiles)
        demFiles_ordered_by_direction = None
    num_scenes = len(demFiles_ordered)

    # Initialize output stats.
    trans = np.zeros((3, num_scenes))
    trans_err = trans.copy()
    rmse = np.zeros((1, num_scenes))
    if check_guess:
        trans_check = np.copy(trans)
        trans_err_check = np.copy(trans_err)
        rmse_check = np.copy(rmse)

    if target_srs is None:
        # Get projection reference of the first scene to be used in equality checks
        # with the projection reference of all scenes that follow.
        dem_srs = rat.extractRasterData(demFiles_ordered[0], 'spat_ref')
        if dem_srs is None or dem_srs.ExportToProj4() == '':
            raise SpatialRefError(
                "DEM '{}' spatial reference ({}) has no PROJ4 representation "
                "and is likely erroneous".format(
                    demFiles_ordered[0], dem_srs.ExportToWkt()
                )
            )
        target_srs = dem_srs
    target_srs_proper = setsm_srs.get_matching_srs(target_srs)
    if target_srs_proper is None:
        raise SpatialRefError(
            "DEM '{}' spatial reference (PROJ4='{}') is not the same as any of "
            "the expected SETSM output projections (WGS84 Polar Stereo North/South "
            "and UTM North/South)".format(
                demFiles_ordered[0], target_srs.ExportToProj4()
            )
        )

    # File loop.
    skipped_scene = False
    segment_break = False
    picking_up_small_scenes_in_existing_strip_segment = False
    i = 0
    while i <= len(demFiles_ordered):

        if i >= len(demFiles_ordered):
            break

        if (   (trans_guess is not None and np.any(np.isnan(trans_guess[:, i])))
            or (trans_err_guess is not None and np.any(np.isnan(trans_err_guess[:, i])))
            or (rmse_guess is not None and np.isnan(rmse_guess[0, i]))):
            # State of scene is somewhere between naturally redundant
            # or redundant by masking, as classified by prior s2s run.
            print("Scene {} of {}: {}".format(i+1, len(demFiles_ordered), demFiles_ordered[i]))
            print("Scene was considered redundant in coregistration step and will be skipped")
            skipped_scene = True

        if skipped_scene:
            skipped_scene = False
            trans[:, i] = np.nan
            trans_err[:, i] = np.nan
            rmse[0, i] = np.nan
            i += 1
            continue

        elif segment_break:
            segment_break = False
            if not picking_up_small_scenes_in_existing_strip_segment:
                demFiles_ordered = demFiles_ordered[:i]
                trans = trans[:, :i]
                trans_err = trans_err[:, :i]
                rmse = rmse[:, :i]

                if force_single_scene_strips:
                    break

                if demFiles_ordered_by_direction is None:
                    # Unexpected segment break when coregistration value guesses were provided.
                    # In the usual two-step workflow, this has been seen to happen when the
                    # masked coregistration step somehow succeeds, but then the unmasked
                    # mosaicking step fails. It appears this happens due to deficiencies in
                    # the core merging/feathering logic when s2s analyzes the overlap area
                    # between the existing swath and the new scene to be added, AND the data
                    # quality is bad (have seen this happen over water or spare data clusters).
                    # Regardless of the true source of the issue, it should be okay to break
                    # to a new segment at this time.
                    break

                demFile_last_added = demFiles_ordered[-1]
                last_added_index = demFiles_ordered_by_direction.index(demFile_last_added)
                demFiles_within_existing_strip = demFiles_ordered_by_direction[:(last_added_index+1)]
                demFiles_to_try_to_pick_up = set.difference(
                    set(demFiles_within_existing_strip),
                    set(demFiles_ordered)
                )

                if len(demFiles_to_try_to_pick_up) == 0:
                    break
                else:
                    picking_up_small_scenes_in_existing_strip_segment = True

                    num_scenes_pickup = len(demFiles_to_try_to_pick_up)
                    print("Will attempt to add {} small scenes that were skipped over "
                          "in initial scene ordering".format(num_scenes_pickup))

                    demFiles_to_try_to_pick_up = [
                        df for df in demFiles_ordered_by_direction if df in demFiles_to_try_to_pick_up
                    ]
                    demFiles_to_try_to_pick_up.reverse()
                    demFiles_ordered.extend(demFiles_to_try_to_pick_up)

                    trans = np.concatenate((trans, np.zeros((3, num_scenes_pickup))), axis=1)
                    trans_err = np.concatenate((trans_err, np.zeros((3, num_scenes_pickup))), axis=1)
                    rmse = np.concatenate((rmse, np.zeros((1, num_scenes_pickup))), axis=1)
            else:
                del demFiles_ordered[i]
                trans = np.delete(trans, i, axis=1)
                trans_err = np.delete(trans_err, i, axis=1)
                rmse = np.delete(rmse, i, axis=1)

                if i >= len(demFiles_ordered):
                    break

        # Construct filenames.
        demFile = demFiles_ordered[i]
        matchFile  = selectBestMatchtag(demFile)
        orthoFile  = selectBestOrtho(demFile)
        ortho2File = selectBestOrtho2(demFile) if use_second_ortho else None
        metaFile   = demFile.replace(demSuffix, 'meta.txt')
        if maskSuffix is None:
            print("No mask applied")
            maskFile = None
        else:
            maskFile = demFile.replace(demSuffix, maskSuffix)

        if use_second_ortho and ortho2File is None:
            raise InvalidArgumentError("`use_second_ortho=True`, but second ortho could not be found")

        print("Scene {} of {}: {}".format(i+1, len(demFiles_ordered), demFile))

        # try:
        x, y, z, m, o, o2, md = loadData(demFile, matchFile, orthoFile, ortho2File, maskFile, metaFile, target_srs, remerge_strips)
        # except:
        #     print("Data read error:")
        #     traceback.print_exc()
        #     print("...skipping")
        #     continue

        # Apply masks.
        x, y, z, m, o, o2, md = applyMasks(x, y, z, m, o, o2, md, filter_options, maskSuffix)

        # Check for redundant scene.
        if np.count_nonzero(~np.isnan(z)) <= add_min_px:
            print("Not enough (unmasked) data, skipping")
            skipped_scene = True
            continue

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Fix grid so that x, y coordinates of
        # pixels in overlapping scenes will match up.
        if ((x[1] / dx) % 1 != 0) or ((y[1] / dy) % 1 != 0):
            x, y, z, m, o, o2, md = regrid(x, y, z, m, o, o2, md)

        # If this is the first scene in strip,
        # set as strip and continue to next scene.
        if 'X' not in vars():
            X, Y, Z, M, O, O2, MD = x, y, z, m, o, o2, md
            del x, y, z, m, o, o2, md

            if not force_single_scene_strips:
                i += 1
            else:
                fp_vertices = rat.getFPvertices(Z, X, Y, label=np.nan, label_type='nodata',
                                                replicate_matlab=True, dtype_out_int64_if_equal=True)
                scene_geom = ogr.CreateGeometryFromWkt(rat.coordsToWkt(fp_vertices.T))

                if preceding_scene_geom_union is None:
                    preceding_scene_geom_union = scene_geom
                else:
                    geom_to_add = scene_geom.Difference(preceding_scene_geom_union)
                    geom_to_add_area = geom_to_add.Area()
                    print("Scene adds an estimated {} square meters to strip geometry".format(geom_to_add_area))
                    if geom_to_add_area <= add_min_area:
                        print("Redundant scene, skipping")
                        skipped_scene = True
                        del X
                    else:
                        preceding_scene_geom_union = preceding_scene_geom_union.Union(scene_geom)

                if not skipped_scene:
                    segment_break = True
                    i += 1

            continue

        # Pad new arrays to stabilize interpolation.
        buff = int(10*dx + 1)
        z  = np.pad(z,  buff, 'constant', constant_values=np.nan)
        m  = np.pad(m,  buff, 'constant', constant_values=0)
        o  = np.pad(o,  buff, 'constant', constant_values=0)
        o2 = np.pad(o2, buff, 'constant', constant_values=0) if o2 is not None else None
        md = np.pad(md, buff, 'constant', constant_values=1)
        x = np.concatenate((x[0]  - dx*np.arange(buff, 0, -1), x,
                            x[-1] + dx*np.arange(1, buff+1)))
        y = np.concatenate((y[0]  + dx*np.arange(buff, 0, -1), y,
                            y[-1] - dx*np.arange(1, buff+1)))

        # Expand strip coverage to encompass new scene.
        if x[0] < X[0]:
            X1 = np.arange(x[0], X[0], dx)
            X = np.concatenate((X1, X))
            Z, M, O, O2, MD = expandCoverage(Z, M, O, O2, MD, X1, direction='left')
            del X1
        if x[-1] > X[-1]:
            X1 = np.arange(X[-1]+dx, x[-1]+dx, dx)
            X = np.concatenate((X, X1))
            Z, M, O, O2, MD = expandCoverage(Z, M, O, O2, MD, X1, direction='right')
            del X1
        if y[0] > Y[0]:
            Y1 = np.arange(y[0], Y[0], -dx)
            Y = np.concatenate((Y1, Y))
            Z, M, O, O2, MD = expandCoverage(Z, M, O, O2, MD, Y1, direction='up')
            del Y1
        if y[-1] < Y[-1]:
            Y1 = np.arange(Y[-1]-dx, y[-1]-dx, -dx)
            Y = np.concatenate((Y, Y1))
            Z, M, O, O2, MD = expandCoverage(Z, M, O, O2, MD, Y1, direction='down')
            del Y1

        # Map new DEM pixels to swath. These must return integers. If not,
        # interpolation will be required, which is currently not supported.
        c0 = np.where(x[0]  == X)[0][0]
        c1 = np.where(x[-1] == X)[0][0] + 1
        r0 = np.where(y[0]  == Y)[0][0]
        r1 = np.where(y[-1] == Y)[0][0] + 1

        # Crop to overlap.
        Xsub  = np.copy( X[c0:c1])
        Ysub  = np.copy( Y[r0:r1])
        Zsub  = np.copy( Z[r0:r1, c0:c1])
        Msub  = np.copy( M[r0:r1, c0:c1])
        Osub  = np.copy( O[r0:r1, c0:c1])
        O2sub = np.copy(O2[r0:r1, c0:c1]) if O2 is not None else None
        MDsub = np.copy(MD[r0:r1, c0:c1])

        # NEW MOSAICKING CODE

        # Crop to just region of overlap.
        A = (~np.isnan(Zsub) & ~np.isnan(z))

        # Check for segment break.
        if np.count_nonzero(A) <= cluster_min_px:
            print("Not enough overlap, segment break")
            segment_break = True
            continue

        r, c = cropBorder(A, 0, buff)

        # Make overlap mask removing isolated pixels.
        strip_nodata =  np.isnan(Zsub[r[0]:r[1], c[0]:c[1]])
        scene_data   = ~np.isnan(   z[r[0]:r[1], c[0]:c[1]])
        strip_mask_water_and_cloud = (MDsub[r[0]:r[1], c[0]:c[1]] > 1)

        # Nodata in strip and data in scene is a one.
        A = rat.bwareaopen(strip_nodata & scene_data, cluster_min_px, in_place=True).astype(np.float32)

        # Check for redundant scene.
        num_px_to_add = np.count_nonzero((A == 1) & ~strip_mask_water_and_cloud)
        print("Number of unmasked pixels to add to strip: {}".format(num_px_to_add))
        if num_px_to_add <= add_min_px:
            print("Redundant scene, skipping")
            skipped_scene = True
            continue

        # Data in strip and nodata in scene is a two.
        A[rat.bwareaopen(~strip_nodata & ~scene_data, cluster_min_px, in_place=True)] = 2

        del strip_nodata, scene_data

        Ar = rat.imresize(A, 0.1, 'nearest')

        # Check for redundant scene.
        if not np.any(Ar):
            print("Redundant scene, skipping")
            skipped_scene = True
            continue

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
        if O2sub is not None:
            O2sub[strip_nodata] = 0
        MDsub[strip_nodata] = 0

        scene_nodata = (W == 1)
        z[scene_nodata] = np.nan
        m[scene_nodata] = 0
        o[scene_nodata] = 0
        if o2 is not None:
            o2[scene_nodata] = 0
        md[scene_nodata] = 0

        del strip_nodata, scene_nodata

        # Coregistration

        P0 = getDataDensityMap(Msub[r[0]:r[1], c[0]:c[1]]) > 0.9

        # Check for segment break.
        if not np.any(P0):
            print("Not enough data overlap, segment break")
            segment_break = True
            continue

        P1 = getDataDensityMap(m[r[0]:r[1], c[0]:c[1]]) > 0.9

        # Check for redundant scene.
        if not np.any(P1):
            print("Redundant scene, skipping")
            skipped_scene = True
            continue

        # Coregister this scene to the strip mosaic.
        if (    hold_guess == HOLD_GUESS_ALL and not check_guess
            and (trans_guess is not None and trans_err_guess is not None and rmse_guess is not None)):
            trans[:, i] = trans_guess[:, i]
            trans_err[:, i] = trans_err_guess[:, i]
            rmse[0, i] = rmse_guess[0, i]
        else:
            trans[:, i], trans_err[:, i], rmse[0, i] = coregisterdems(
                Xsub[c[0]:c[1]], Ysub[r[0]:r[1]], Zsub[r[0]:r[1], c[0]:c[1]],
                   x[c[0]:c[1]],    y[r[0]:r[1]],    z[r[0]:r[1], c[0]:c[1]],
                P0, P1,
                (trans_guess[:, i] if trans_guess is not None else trans_guess),
                hold_guess != HOLD_GUESS_OFF
            )[[1, 2, 3]]

            if check_guess:
                error_tol = 10**-2
                if trans_guess is not None:
                    trans_check[:, i] = trans[:, i]
                    if not np.allclose(trans_check[:, i], trans_guess[:, i], rtol=0, atol=error_tol, equal_nan=True):
                        print("`trans_check` vector out of `coregisterdems` does not match `trans_guess` within error tol ({})".format(error_tol))
                        print("`trans_guess`:")
                        print(np.array2string(trans_guess, precision=4, max_line_width=np.inf))
                        print("`trans_check`:")
                        print(np.array2string(trans_check, precision=4, max_line_width=np.inf))
                if rmse_guess is not None:
                    rmse_check[0, i] = rmse[0, i]
                    if not np.allclose(rmse_check[0, i], rmse_guess[0, i], rtol=0, atol=error_tol, equal_nan=True):
                        print("`rmse_check` out of `coregisterdems` does not match `rmse_guess` within error tol ({})".format(error_tol))
                        print("`rmse_guess`:")
                        print(np.array2string(rmse_guess, precision=4, max_line_width=np.inf))
                        print("`rmse_check`:")
                        print(np.array2string(rmse_check, precision=4, max_line_width=np.inf))

            if hold_guess != HOLD_GUESS_OFF:
                if trans_guess is not None:
                    trans[:, i] = trans_guess[:, i]
                if trans_err_guess is not None:
                    trans_err[:, i] = trans_err_guess[:, i]
                if rmse_guess is not None and hold_guess == HOLD_GUESS_ALL:
                    rmse[0, i] = rmse_guess[0, i]

        # Check for segment break.
        if np.isnan(rmse[0, i]):
            print("Unable to coregister, segment break")
            segment_break = True
        elif rmse[0, i] > max_coreg_rmse:
            print("Final RMSE is greater than cutoff value ({} > {}), segment break".format(
                rmse[0, i], max_coreg_rmse))
            segment_break = True
        else:
            pass
        if segment_break:
            continue

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

        if o2 is not None:
            # Interpolate ortho2 to same grid.
            o2i = o2.astype(np.float32)
            o2i[o2i == 0] = np.nan  # Set border to NaN so it won't be interpolated.
            o2i = rat.interp2_gdal(xi, yi, o2i, Xsub, Ysub, 'cubic')
            del o2
        else:
            o2i = None

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

        # Remove border on ortho separately.
        M4 = ~np.isnan(oi)
        M4 = rat.imerode(M4, 6)
        oi[~M4] = np.nan
        del M4

        if o2i is not None:
            # Remove border on ortho2 separately.
            M5 = ~np.isnan(o2i)
            M5 = rat.imerode(M5, 6)
            o2i[~M5] = np.nan
            del M5

        # Make weighted elevation grid.
        A = Zsub*W + zi*(1-W)
        Zsub_only = ~np.isnan(Zsub) &  np.isnan(zi)
        zi_only   =  np.isnan(Zsub) & ~np.isnan(zi)
        A[Zsub_only] = Zsub[Zsub_only]
        A[zi_only]   =   zi[zi_only]
        del Zsub, zi, Zsub_only, zi_only

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

        # del W

        Osub_only = ~np.isnan(Osub) &  np.isnan(oi)
        oi_only   =  np.isnan(Osub) & ~np.isnan(oi)
        A[Osub_only] = Osub[Osub_only]
        A[oi_only]   =   oi[oi_only]
        del Osub, oi, Osub_only, oi_only

        A[np.isnan(A)] = 0  # convert back to uint16
        A = A.astype(np.uint16)

        O[r0:r1, c0:c1] = A
        del A

        if O2sub is not None:
            # Make weighted ortho2 grid.
            O2sub = O2sub.astype(np.float32)
            O2sub[O2sub == 0] = np.nan
            A = O2sub*W + o2i*(1-W)

            # del W

            O2sub_only = ~np.isnan(O2sub) &  np.isnan(o2i)
            o2i_only   =  np.isnan(O2sub) & ~np.isnan(o2i)
            A[O2sub_only] = O2sub[O2sub_only]
            A[o2i_only]   =   o2i[o2i_only]
            del O2sub, o2i, O2sub_only, o2i_only

            A[np.isnan(A)] = 0  # convert back to uint16
            A = A.astype(np.uint16)

            O2[r0:r1, c0:c1] = A
            del A

        del W

        # For the mask, bitwise combination.
        MD[r0:r1, c0:c1] = np.bitwise_or(MDsub, mdi)
        del MDsub, mdi

        i += 1

    # Crop to data.
    if 'X' in vars() and ~np.all(np.isnan(Z)):
        rcrop, ccrop = cropBorder(Z, np.nan)
        if rcrop is not None:
            X  =  X[ccrop[0]:ccrop[1]]
            Y  =  Y[rcrop[0]:rcrop[1]]
            Z  =  Z[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            M  =  M[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            O  =  O[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]
            O2 = O2[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]] if O2 is not None else None
            MD = MD[rcrop[0]:rcrop[1], ccrop[0]:ccrop[1]]

            # Round DEM values to 1/128 to greatly improve compression effectiveness
            np.multiply(Z, 128.0, out=Z)
            np.round_(Z, decimals=0, out=Z)
            np.divide(Z, 128.0, out=Z)

            Z[np.isnan(Z)] = -9999
    else:
        X, Y, Z, M, O, O2, MD = None, None, None, None, None, None, None

    return X, Y, Z, M, O, O2, MD, trans, trans_err, rmse, demFiles_ordered, target_srs_proper, preceding_scene_geom_union


def coregisterdems(x1, y1, z1,
                   x2, y2, z2,
                   m1=None, m2=None,
                   trans_guess=None, hold_guess=False,
                   max_horiz_offset=15, rmse_step_tresh=-0.001, max_iterations=5):
    """
% COREGISTERDEM registers a floating to a reference DEM
%
% [z2r,trans,trans_err,rms] = coregisterdems(x1,y1,z1,x2,y2,z2) registers the
% floating DEM in 2D array z2 with coordinate vectors x2 and y2 to the
% reference DEM in z1 using the iterative procedure in Nuth and Kaab,
% 2011. z2r is the regiestered DEM, p is the [dz,dx,dy] transformation
% parameters, with their 1-sigma errors in trans_err, and rms is the rms of the
% transformation in the vertical from the residuals. If the registration fails
% due to lack of overlap, NaNs are returned in p and perr. If the registration
% fails to converge or exceeds the maximum shift, the median vertical offset is
% applied.
%
% [...]= coregisterdems(x1,y1,z1,x2,y2,z2,m1,m2) allows a data mask to be applied
% where 0 values will be ignored in the solution.

    """
    if len(x1) < 3 or len(y1) < 3 or len(x2) < 3 or len(y2) < 3:
        raise RasterDimensionError("Minimum array dimension is 3")

    interpflag = True
    if (len(x1) == len(x2)) and (len(y1) == len(y2)):
        if not np.any(x2 - x1) and not np.any(y2 - y1):
            interpflag = False

    if (m1 is None) ^ (m2 is None):
        raise InvalidArgumentError("Either none of both of arguments `m1` and `m2` must be provided")

    rx = x1[1] - x1[0]  # coordinate spacing

    # Initial trans and RMSE settings
    if trans_guess is not None:
        p = np.reshape(np.copy(trans_guess), (3, 1))
        interpflag = True
    else:
        p = np.zeros((3, 1))  # p  is prior iteration trans var
    pn = p.copy()             # pn is current iteration trans var
    perr = np.zeros((3, 1))   # perr is prior iteration regression errors
    pnerr = perr.copy()       # pnerr is current iteration regression errors
    d0 = np.inf               # initial RMSE

    # Edge case markers
    meddz = None
    return_meddz = False
    critical_failure = False

    it = 0
    while True:
        it += 1

        print("Planimetric Correction Iteration {}".format(it))

        print("Offset (z,x,y): {:.3f}, {:.3f}, {:.3f}".format(pn[0, 0], pn[1, 0], pn[2, 0]))

        if np.any(np.abs(pn[1:]) > max_horiz_offset):
            print("Maximum horizontal offset reached")
            return_meddz = True
            break

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
            z2n = z2 - pn[0].astype(np.float32)
            if m2 is not None:
                m2n = m2

        interpflag = True

        # Calculate slopes.
        sy, sx = np.gradient(z2n, rx)
        sx = -sx

        # Difference grids.
        dz = z2n - z1

        if m1 is not None:
            dz[~m1] = np.nan
        if m2 is not None:
            dz[~m2n] = np.nan

        if np.all(np.isnan(dz)):
            print("No overlap")
            critical_failure = True
            break

        # Filter NaNs and outliers.
        n = (  ~np.isnan(sx)
             & ~np.isnan(sy)
             & (np.abs(dz - np.nanmedian(dz)) <= 3*np.nanstd(dz)))
        n_count = np.count_nonzero(n)

        if n_count < 10:
            print("Too few ({}) registration points".format(n_count))
            critical_failure = True
            break

        # Get RMSE and break if above threshold.
        d1 = np.sqrt(np.mean(np.power(dz[n], 2)))

        print("RMSE = {}".format(d1))

        # Keep median dz if first iteration.
        if it == 1 and trans_guess is None:
            meddz = np.median(dz[n])
            meddz_err = np.std(dz[n] / np.sqrt(n_count))
            d00 = np.sqrt(np.mean(np.power(dz[n] - meddz, 2)))

        rmse_step = d1 - d0

        if rmse_step > rmse_step_tresh or np.isnan(d0):
            print("RMSE step in this iteration ({:.5f}) is above threshold ({}), "
                  "stopping and returning values of prior iteration".format(rmse_step, rmse_step_tresh))
            # If fails after first registration attempt,
            # set dx and dy to zero and subtract the median offset.
            if it == 2 and trans_guess is None:
                print("Second iteration regression failure")
                return_meddz = True
            break
        elif it == max_iterations:
            print("Maximum number of iterations ({}) reached".format(max_iterations))
            break

        # Keep this adjustment.
        z2out = z2n.copy()
        p = pn.copy()
        perr = pnerr.copy()
        d0 = d1

        if trans_guess is not None and hold_guess:
            print("Holding trans guess, stopping")
            break

        # Build design matrix.
        X = np.column_stack((np.ones(n_count, dtype=np.float32), sx[n], sy[n]))

        # Solve for new adjustment.
        p1 = np.reshape(np.linalg.lstsq(X, dz[n])[0], (-1, 1))

        # Calculate p errors.
        _, R = np.linalg.qr(X)
        RI = np.linalg.lstsq(R, np.identity(3, dtype=np.float32))[0]
        nu = X.shape[0] - X.shape[1]  # residual degrees of freedom
        yhat = np.matmul(X, p1)       # predicted responses at each data point
        r = dz[n] - yhat.T[0]         # residuals
        normr = np.linalg.norm(r)

        rmse = normr / np.sqrt(nu)
        tval = scipy.stats.t.ppf((1-0.32/2), nu)

        se = rmse * np.sqrt(np.sum(np.square(np.abs(RI)), axis=1, keepdims=True))
        p1err = tval * se

        # Update shifts.
        pn = p + p1
        pnerr = np.sqrt(np.square(perr) + np.square(p1err))

    if return_meddz:
        if meddz is None:
            assert trans_guess is None, "`trans_guess` is not None"
            if hold_guess:
                print("Regression failure under held trans guess")
                critical_failure = True
            else:
                print("Rerunning coregistration without `trans_guess`")
                return coregisterdems(x1, y1, z1,
                                      x2, y2, z2,
                                      m1=m1, m2=m2,
                                      max_horiz_offset=max_horiz_offset,
                                      rmse_step_tresh=rmse_step_tresh)
        else:
            print("Returning median vertical offset: {:.3f}".format(meddz))
            z2out = z2 - meddz
            p = np.array([[meddz, 0, 0]]).T
            perr = np.array([[meddz_err, 0, 0]]).T
            d0 = d00

    if critical_failure:
        print("Regression critical failure, returning NaN trans and RMSE")
        z2out = z2
        p = np.full((3, 1), np.nan)
        perr = np.full((3, 1), np.nan)
        d0 = np.nan

    print("Final offset (z,x,y): {:.3f}, {:.3f}, {:.3f}".format(p[0, 0], p[1, 0], p[2, 0]))
    print("Final RMSE = {}".format(d0))

    return np.array([z2out, p.T[0], perr.T[0], d0])


def orderPairs(fnames):
    """
    Scene order is determined in relation to *grid north* of the common projection
    by comparing total x-extent and y-extent of the scenes as a whole (aspect ratio).
    The larger of these extents determines the coordinate by which to do ordering.
    """
    R0 = np.zeros((len(fnames), 4))  # Matrix to place rectangular parameters of the raster images.
    indexed_geoms = []               # List to place rectangular (footprint) geometries of the rasters
                                     # as OGR geometries, each tupled with the index corresponding to the
                                     # filename of the raster it is extracted from.

    sample_spat_ref = rat.extractRasterData(fnames[0], 'spat_ref')

    # Get rectangular parameters and geometries.
    for fname_index, fn in enumerate(fnames):
        cc, scene_geom = rat.extractRasterData(rat.openRaster(fn, sample_spat_ref), 'corner_coords', 'geom')
        cc_x = cc[:, 0]
        cc_y = cc[:, 1]
        R0[fname_index, :] = [min(cc_x), min(cc_y), max(cc_x)-min(cc_x), max(cc_y)-min(cc_y)]
        indexed_geoms.append((fname_index, scene_geom))

    # Calculate aspect ratio, ar = x-extent/y-extent
    ar = (  (max(R0[:, 0] + R0[:, 2]) - min(R0[:, 0]))
          / (max(R0[:, 1] + R0[:, 3]) - min(R0[:, 1])))

    if ar >= 1:
        # Scenes are in east-west direction; start with scene with minimum x.
        direction_ordered_indices = np.argsort(R0[:, 0]).tolist()
    else:
        # Scenes are in north-south direction; start with scene with minimum y.
        direction_ordered_indices = np.argsort(R0[:, 1]).tolist()
    direction_start_index = 0

    # The first scene by direction could be almost fully overlapped by a larger
    # subsequent scene as determined by the main ordering algorithm.
    # That ordering can be detrimental to the quality of the first segment in the merged strip,
    # since the feather/merge procedure is ill-defined when working on this case.
    # The following will check if this is the case by natural ordering,
    # and if it is, select a different first scene to avoid this case.
    if len(fnames) > 1:
        for direction_index, fname_index in enumerate(direction_ordered_indices):
            scene_geom = indexed_geoms[fname_index][1]
            scene_area = scene_geom.GetArea()
            overlap_area = max([scene_geom.Intersection(ind_geom[1]).GetArea() for ind_geom in indexed_geoms if ind_geom[0] != fname_index])
            if overlap_area < (0.9*scene_area):
                direction_start_index = direction_index
                break
    print("orderPairs direction_start_index={}".format(direction_start_index))

    # Start with the footprint of this scene and let the strip grow from there.
    first_fname_index = direction_ordered_indices[direction_start_index]
    first_geom = indexed_geoms[first_fname_index][1]
    footprint_geom = first_geom
    ordered_fname_indices = [first_fname_index]
    del indexed_geoms[first_fname_index]

    # Loop through scene pair geometries and sequentially add the
    # next pair with the most overlap to the ordered indices list.
    while len(indexed_geoms) > 0:
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
            print("Break in scene pair coverage detected, segment break")
            break

    overlap_ordered_fnames = [fnames[i] for i in ordered_fname_indices]
    direction_ordered_fnames = [fnames[i] for i in direction_ordered_indices]

    return overlap_ordered_fnames, direction_ordered_fnames


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


def loadData(demFile, matchFile, orthoFile, ortho2File, maskFile, metaFile, target_srs, remerge_strips=False):
    """
    Load data files and perform basic conversions.
    """
    global __INTRACK_ORTHO_CATID__

    demFile_srs = rat.extractRasterData(rat.openRaster(demFile), 'spat_ref')

    z, x_dem, y_dem, spat_ref = rat.extractRasterData(rat.openRaster(demFile, target_srs, 'bilinear'), 'array', 'x', 'y', 'spat_ref')
    if spat_ref.IsSame(target_srs) != 1:
        raise SpatialRefError("DEM '{}' spatial reference ({}) mismatch with strip spatial reference ({})".format(
                              demFile, spat_ref.ExportToProj4(), target_srs.ExportToProj4()))

    # A DEM pixel with a value of -9999 is a nodata pixel; interpret it as NaN.
    # TODO: Ask Ian about the following interpretation of nodata values.
    z[(z < -100) | (z == 0) | (z == -np.inf) | (z == np.inf)] = np.nan

    check_srs = rat.extractRasterData(rat.openRaster(matchFile), 'spat_ref')
    if check_srs.IsSame(demFile_srs) != 1:
        raise SpatialRefError("Matchtag '{}' spatial reference ({}) mismatch with DEM spatial reference ({})".format(
                              matchFile, check_srs.ExportToProj4(), demFile_srs.ExportToProj4()))

    m = rat.extractRasterData(rat.openRaster(matchFile, target_srs, 'nearest'), 'array').astype(np.bool)
    if m.shape != z.shape:
        warnings.warn("Matchtag '{}' dimensions differ from DEM dimensions".format(matchFile)
                     +"\nInterpolating to DEM dimensions")
        x, y = rat.extractRasterData(rat.openRaster(matchFile, target_srs, 'nearest'), 'x', 'y')
        m = rat.interp2_gdal(x, y, m.astype(np.float32), x_dem, y_dem, 'nearest')
        m[np.isnan(m)] = 0  # Convert back to bool/uint8.
        m = m.astype(np.bool)

    ortho_arrays = []
    for i, ortho_file in enumerate([orthoFile, ortho2File]):
        ortho_num = i+1
        if ortho_file is None:
            o = None
        else:
            check_srs = rat.extractRasterData(rat.openRaster(ortho_file), 'spat_ref')
            if check_srs.IsSame(demFile_srs) != 1:
                raise SpatialRefError("Ortho{} '{}' spatial reference ({}) mismatch with DEM spatial reference ({})".format(
                                      ortho_num if ortho_num > 1 else '', ortho_file, check_srs.ExportToProj4(), demFile_srs.ExportToProj4()))

            o = rat.extractRasterData(rat.openRaster(ortho_file, target_srs, 'bicubic'), 'array')
            if o.shape != z.shape:
                warnings.warn("Ortho{} '{}' dimensions differ from DEM dimensions".format(ortho_num if ortho_num > 1 else '', ortho_file)
                             +"\nInterpolating to DEM dimensions")
                x, y = rat.extractRasterData(rat.openRaster(ortho_file, target_srs, 'bicubic'), 'x', 'y')
                o = o.astype(np.float32)
                o[o == 0] = np.nan  # Set border to NaN so it won't be interpolated.
                o = rat.interp2_gdal(x, y, o, x_dem, y_dem, 'cubic')
                o[np.isnan(o)] = 0  # Convert back to uint16.
                o = rat.astype_round_and_crop(o, np.uint16, allow_modify_array=True)
        ortho_arrays.append(o)
    o1, o2 = ortho_arrays

    if maskFile is None:
        md = np.zeros_like(z, dtype=np.uint8)
    else:
        check_srs = rat.extractRasterData(rat.openRaster(maskFile), 'spat_ref')
        if check_srs.IsSame(demFile_srs) != 1:
            raise SpatialRefError("Mask '{}' spatial reference ({}) mismatch with DEM spatial reference ({})".format(
                                  maskFile, check_srs.ExportToProj4(), demFile_srs.ExportToProj4()))

        md = rat.extractRasterData(rat.openRaster(maskFile, target_srs, 'nearest'), 'array').astype(np.uint8)
        if md.shape != z.shape:
            raise RasterDimensionError("Mask '{}' dimensions {} do not match DEM dimensions {}".format(
                                       maskFile, md.shape, z.shape))

    if remerge_strips:
        o, o2 = o1, o2
    else:
        # Re-scale ortho data if WorldView correction is detected in the meta file.
        # Allow for inconsistent Image 1/2 paths in SETSM metadata, making sure that
        # first ortho 'o' corresponds to first catalogid in strip pairname and that
        # second ortho 'o2' (optional, xtrack only) corresponds to second catalogid.
        meta = readSceneMeta(metaFile)

        ortho_arrays = []
        ortho_catids = []
        for i, o in enumerate([o1, o2]):
            catid = None
            if o is not None:
                ortho_num = i+1
                wv_correct_flag = meta['image_{}_wv_correct'.format(ortho_num)]
                maxDN = meta['image_{}_max'.format(ortho_num)] if wv_correct_flag else None
                if maxDN is not None:
                    print("Ortho{} had wv_correct applied, rescaling values to range [0, {}]".format(
                        ortho_num if ortho_num > 1 else '', maxDN))
                    o = rescaleDN(o, maxDN)
                    o = rat.astype_round_and_crop(o, np.uint16, allow_modify_array=True)
                ortho_image_file = meta['image_{}'.format(ortho_num)]
                catid = os.path.basename(ortho_image_file).split('_')[2]
            ortho_arrays.append(o)
            ortho_catids.append(catid)

        pairname_catids = os.path.basename(orthoFile).split('_')[2:4]
        if ortho_catids[1] is None:
            # Strip is intrack.
            intrack_ortho_catid = ortho_catids[0]
            if __INTRACK_ORTHO_CATID__ is None:
                __INTRACK_ORTHO_CATID__ = intrack_ortho_catid
            elif intrack_ortho_catid is None or ortho_catids[0] != __INTRACK_ORTHO_CATID__:
                warnings.warn("Catalog ID of Image 1 (assumed ortho) is not consistent across"
                              " scene metadata files for intrack strip")
        elif ortho_catids[0] != pairname_catids[0]:
            # if ortho_catids[1] is None:
            #     raise MetadataError("Single intrack ortho from Image 1 in '{}' has catalogid ({})"
            #                         " that does not match first catalogid of pairname".format(metaFile, ortho_catids[0]))
            if ortho_catids[0] != pairname_catids[1] or ortho_catids[1] != pairname_catids[0]:
                raise MetadataError("xtrack orthos from Image 1/2 in '{}' have Catalog IDs ({})"
                                    " that do not match Catalog IDs of pairname".format(metaFile, ortho_catids))
            # Assume strip pairname is xtrack at this point.
            assert ortho2File is not None, "`ortho2File` is None"
            ortho_catids.reverse()
            ortho_arrays.reverse()

        o, o2 = ortho_arrays

    return x_dem, y_dem, z, m, o, o2, md


def applyMasks(x, y, z, m, o, o2, md, filter_options=(), maskSuffix=None):
    """
    Apply masks to the scene DEM, matchtag, and ortho matrices.
    """
    if sys.version_info[0] < 3:
        from filter_scene import mask_v2, MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT
    else:
        from lib.filter_scene import mask_v2, MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT

    mask_select = md

    if len(filter_options) > 0:
        mask_select = np.copy(md)
        mask_ones = np.ones_like(mask_select)
        for opt in filter_options:
            unmask_bit = None
            if opt == 'nowater':
                unmask_bit = MASKCOMP_WATER_BIT
            elif opt == 'nocloud':
                unmask_bit = MASKCOMP_CLOUD_BIT
            if unmask_bit is not None:
                np.bitwise_and(mask_select, ~np.left_shift(mask_ones, unmask_bit), out=mask_select)

    mask_select = (mask_select > 0)

    if maskSuffix.endswith(('mask.tif', 'bitmask.tif')):
        mask_select = mask_v2(postprocess_mask=mask_select, postprocess_res=abs(x[1]-x[0]))

    z[mask_select] = np.nan
    m[mask_select] = 0

    # If there is any good data, crop the matrices of bordering NaNs.
    if np.any(~np.isnan(z)):
        rowcrop, colcrop = cropBorder(z, np.nan)

        x  =  x[colcrop[0]:colcrop[1]]
        y  =  y[rowcrop[0]:rowcrop[1]]
        z  =  z[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]
        m  =  m[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]
        o  =  o[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]
        o2 = o2[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]] if o2 is not None else None
        md = md[rowcrop[0]:rowcrop[1], colcrop[0]:colcrop[1]]

    return x, y, z, m, o, o2, md


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


def regrid(x, y, z, m, o, o2, md):
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

    # Interpolate ortho to same grid.
    if o2 is not None:
        o2 = o2.astype(np.float32)
        o2[np.isnan(z)] = np.nan  # Set border to NaN so it won't be interpolated.
        o2 = rat.interp2_gdal(x, y, o2, xi, yi, 'cubic')
        o2[np.isnan(o2)] = 0  # Convert back to uint16.
        o2 = rat.astype_round_and_crop(o2, np.uint16, allow_modify_array=True)

    md = rat.interp2_gdal(x, y, md.astype(np.float32), xi, yi, 'nearest')
    md[np.isnan(md)] = 0  # Convert back to uint8.
    md = md.astype(np.uint8)

    return xi, yi, zi, m, o, o2, md


def expandCoverage(Z, M, O, O2, MD, R1, direction):
    """
    Expand strip coverage for DEM, matchtag, and ortho matrices
    based upon the direction of expansion.
    When (X1/Y1) is passed in for R1,
    (('left' or 'right') / ('up' or 'down'))
    must be passed in for direction.
    """
    if direction in ('up', 'down'):
        # R1 is Y1.
        Z1  = np.full((R1.size,  Z.shape[1]), np.nan, dtype=np.float32)
        M1  = np.full((R1.size,  M.shape[1]), False,  dtype=np.bool)
        O1  = np.full((R1.size,  O.shape[1]), 0,      dtype=np.uint16)
        O21 = np.full((R1.size, O2.shape[1]), 0,      dtype=np.uint16) if O2 is not None else None
        MD1 = np.full((R1.size, MD.shape[1]), 1,      dtype=np.uint8)
        axis_num = 0
    else:
        # R1 is X1.
        Z1  = np.full((Z.shape[0],  R1.size), np.nan, dtype=np.float32)
        M1  = np.full((M.shape[0],  R1.size), False,  dtype=np.bool)
        O1  = np.full((O.shape[0],  R1.size), 0,      dtype=np.uint16)
        O21 = np.full((O2.shape[0], R1.size), 0,      dtype=np.uint16) if O2 is not None else None
        MD1 = np.full((MD.shape[0], R1.size), 1,      dtype=np.uint8)
        axis_num = 1

    if direction in ('left', 'up'):
        pairs = [[Z1, Z], [M1, M], [O1, O], [MD1, MD]]
        if O2 is not None:
            pairs.append([O21, O2])
    else:
        pairs = [[Z, Z1], [M, M1], [O, O1], [MD, MD1]]
        if O2 is not None:
            pairs.append([O2, O21])

    if O2 is not None:
        Z, M, O, MD, O2 = [np.concatenate(p, axis=axis_num) for p in pairs]
    else:
        Z, M, O, MD = [np.concatenate(p, axis=axis_num) for p in pairs]

    return Z, M, O, O2, MD
