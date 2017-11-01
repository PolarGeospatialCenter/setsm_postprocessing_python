#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

# Version 3.0; Ryan Shellberg, Erik Husby; Polar Geospatial Center, University of Minnesota; 2017
# Translated from MATLAB code written by Ian Howat, Ohio State University, 2017


from __future__ import division
import math
import os
import re
from sys import stdout

import numpy as np
import shapely.geometry as geometry
from scipy import ndimage, spatial
from shapely.ops import polygonize, unary_union
from skimage import draw
from skimage.filters.rank import entropy

import raster_array_tools as rat
import test


class InvalidArgumentError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)

class RasterDimensionError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


def generateMasks(demFile, maskFileSuffix, noentropy=False):
    # TODO: Write docstring.

    if maskFileSuffix == 'edgemask/datamask':
        matchFile = demFile.replace('dem.tif', 'matchtag.tif')
        stdout.write(matchFile+"\n")
        mask_v1(matchFile, noentropy)
    else:
        maskFile = demFile.replace('dem.tif', maskFileSuffix+'.tif')
        stdout.write(demFile+"\n")
        mask = None
        if maskFileSuffix == 'rema2a':
            mask = mask_v2a(demFile, maskFile)
        else:
            mask = mask_v2(demFile, maskFile)
        # TODO: Check that this save function works properly.
        rat.saveArrayAsTiff(mask.astype(np.uint8), maskFile, like_rasterFile=demFile)


def mask_v1(matchFile, noentropy=False):
    """Creates edgemask and datamask of the matchtag array, with or without entropy protection.

    Source file: batch_mask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 10/17/17
    """
    # Find SETSM version.
    metaFile = matchFile.replace('matchtag.tif', 'meta.txt')
    if not os.path.isfile(metaFile):
        print "No meta file, assuming SETSM version > 2.0"
        setsmVersion = 3
    else:
        setsm_pattern = re.compile("SETSM Version=(.*)")
        metaFile_fp = open(metaFile, 'r')
        line = metaFile_fp.readline()
        setsmVersion = None
        while line != '':
            match = re.search(setsm_pattern, line)
            if match:
                try:
                    setsmVersion = float(match.group(1).strip())
                except ValueError:
                    setsmVersion = float('inf')
                break
            line = metaFile_fp.readline()
        metaFile_fp.close()
        if setsmVersion is None:
            setsmVersion = 2.03082016
            print "WARNING: Missing SETSM Version number in '{}'".format(metaFile)
            print "--> Using settings for default 'SETSM Version={}'".format(setsmVersion)

    match_array, res = rat.extractRasterParams(matchFile, 'array', 'res')
    match_array = match_array.astype(np.bool)

    if setsmVersion < 2.01292016:
        n = int(math.floor(21*2/res))   # data density kernel window size
        Pmin = 0.8                      # data density threshold for masking
        Amin = int(2000/res)            # minimum data cluster area
        cf = 0.5                        # boundary curvature factor (0 = convex hull, 1 = point boundary)
        crop = n
    else:
        n = int(math.floor(101*2/res))
        Pmin = 0.99
        Amin = int(2000/res)
        cf = 0.5
        crop = n

    edgemask = getDataDensityMask(match_array, kernel_size=n, density_thresh=Pmin)
    if not noentropy:
        entropy_mask = getEntropyMask(matchFile.replace('matchtag.tif', 'ortho.tif'))
        np.logical_or(edgemask, entropy_mask, out=edgemask)
    edgemask = maskEdges(edgemask, min_data_cluster=Amin, hull_concavity=cf, crop=crop)
    rat.saveArrayAsTiff(edgemask, matchFile.replace('matchtag.tif', 'edgemask.tif'),
                        like_rasterFile=matchFile)

    match_array[~edgemask] = 0
    del edgemask

    # Set datamask filtering parameters based on SETSM version and image resolution.
    if setsmVersion <= 2.0:
        n = int(math.floor(21*2/res))   # data density kernel window size
        Pmin = 0.3                      # data density threshold for masking
        Amin = 1000                     # minimum data cluster area
        Amax = 10000                    # maximum data gap area to leave filled
    else:
        n = int(math.floor(101*2/res))
        Pmin = 0.90
        Amin = 1000
        Amax = 1000

    datamask = getDataDensityMask(match_array, kernel_size=n, density_thresh=Pmin)
    del match_array
    datamask = maskDataClustersAndHoles(datamask, min_data_cluster=Amin, hole_fill_max_size=Amax)
    # TODO: Check that this save function works properly.
    rat.saveArrayAsTiff(datamask.astype(np.uint8), matchFile.replace('matchtag.tif', 'datamask.tif'),
                        like_rasterFile=matchFile)


def mask_v2(demFile, maskFile, avg_kernel_size=21, processing_res=8, min_data_cluster=500):
    # TODO: Write my own docstring.
    """
    % MASK ArcticDEM masking algorithm
    %
    % m = mask(demFile,satID,effectiveBandwidth,abScaleFactor,meanSunElevation)
    % returns the mask stucture (m.x,m.y,m.z) for the demFile using the
    % given image parameters.
    %
    % m = mask(...,maxDigitalNumber,previewPlot) maxDigitalNumber is optional
    % for rescaling the orthoimage to the original source image range.
    % If it's mot included or is empty, no rescaling will be applied. If
    % previewPlot == 'true', a *_maskPreview.tif image will be saved to the
    % same directory as the demFile that shows the DEM hillshade with and
    % without the mask applied.
    %
    % m = mask(demFile,meta) returns the mask stucture (m.x,m.y,m.z) for the
    % demFile and meta structure, where meta is the output of readSceneMeta.
    % Required fields in the meta structure are:
    % 'image_1_satID'
    % 'image_1_wv_correct'
    % 'image_1_effbw'
    % 'image_1_abscalfact'
    % 'image_1_mean_sun_elevation'
    % additionally, if image_1_wv_correct==1, the image_1_max field is also
    % required.
    %
    % REQUIRED FUNCTIONS: readGeotiff, DataDensityMap, rescaleDN, edgeSlopeMask
    % cloudMask, DG_DN2RAD, waterMask
    %
    % Ian Howat, ihowat@gmail.com
    % 25-Jul-2017 12:49:25
    """
    metaFile  = demFile.replace('dem.tif', 'meta.txt')
    matchFile = demFile.replace('dem.tif', 'matchtag.tif')
    orthoFile = demFile.replace('dem.tif', 'ortho.tif')

    meta = readSceneMeta(metaFile)
    satID              = meta['image_1_satID']
    wv_correct_flag    = meta['image_1_wv_correct']
    effbw              = meta['image_1_effbw']
    abscalfact         = meta['image_1_abscalfact']
    mean_sun_elevation = meta['image_1_mean_sun_elevation']
    maxDN = meta['image_1_max'] if wv_correct_flag else None

    dem_array, mask_shape, image_res = rat.extractRasterParams(demFile, 'array', 'shape', 'res')
    match_array = rat.extractRasterParams(matchFile, 'array')
    ortho_array = rat.extractRasterParams(orthoFile, 'array')

    if match_array.shape != mask_shape:
        raise RasterDimensionError("matchFile '{}' dimensions {} do not match dem dimensions {}".format(
                                   matchFile, match_array.shape, mask_shape))

    # FIXME: Mirror functionality from MATLAB code to allow correcting the following dimension error?
    if ortho_array.shape != mask_shape:
        raise RasterDimensionError("orthoFile '{}' dimensions {} do not match dem dimensions {}".format(
                                   orthoFile, ortho_array.shape, mask_shape))
        # print "WARNING: orthoFile '{}' dimensions {} do not match dem dimensions {}".format(
        #     orthoFile, ortho_array.shape, mask_shape)

    dem_nodata = np.isnan(dem_array)  # original background for rescaling
    dem_array[dem_array == -9999] = np.nan
    data_density_map = rat.getDataDensityMap(match_array, avg_kernel_size)
    del match_array

    # Re-scale ortho data if WorldView correction is detected in the meta file.
    if maxDN is not None:
        print "rescaled to: 0 to {}".format(maxDN)
        # TODO: Combine the following two functions once testing is complete?
        ortho_array = rescaleDN(ortho_array, maxDN)
        ortho_array = DG_DN2RAD(ortho_array, satID=satID, effectiveBandwith=effbw, abscalFactor=abscalfact)
        print "radiance value range: {.2f} to {.2f}".format(np.min(ortho_array), np.max(ortho_array))

    # Resize arrays to processing resolution.
    if image_res != processing_res:
        resize_factor = image_res / processing_res
        dem_array        = rat.imresize(dem_array,        resize_factor, 'bicubic')
        ortho_array      = rat.imresize(ortho_array,      resize_factor, 'bicubic')
        data_density_map = rat.imresize(data_density_map, resize_factor, 'bicubic')

    # Mask edges using dem slope.
    mask = maskEdges(getSlopeMask(dem_array, res=processing_res))
    dem_array[~mask] = np.nan
    if not np.any(~np.isnan(dem_array)):
        return mask
    del mask

    # Mask water.
    ortho_array[np.isnan(dem_array)] = 0
    data_density_map[np.isnan(dem_array)] = 0
    mask = getWaterMask(ortho_array, mean_sun_elevation, data_density_map)
    dem_array[~mask] = np.nan
    data_density_map[~mask] = 0
    if not np.any(~np.isnan(dem_array)):
        return mask
    del mask

    # Filter clouds.
    mask = getCloudMask(dem_array, ortho_array, data_density_map)
    dem_array[mask] = np.nan

    mask = ~np.isnan(dem_array)
    if not np.any(mask):
        return mask

    mask = rat.bwareaopen(mask, min_data_cluster, in_place=True)
    mask = rat.imresize(mask, mask_shape, 'nearest')
    mask[dem_nodata] = False

    return mask


def getDataDensityMask(data_array, kernel_size=21, density_thresh=0.3):
    """Mask areas of poor data coverage in a data array.

    Data is defined as pixels with non-zero value.
    The returned mask sets to 1 all pixels for which the surrounding [kernel_size x kernel_size]
    neighborhood has a fraction of pixels containing data that is >= density_thresh.

    :param data_array: The 2D array of data values to mask.
    :type data_array: ndarray
    :param kernel_size: The side length of the neighborhood to use for calculating data density fraction.
    :type kernel_size: int
    :param density_thresh: Minimum data density fraction for a pixel to be set to 1 in the mask.
    :type density_thresh: float
    :return: The data density mask of the input data array.
    :rtype: ndarray of type bool, same shape as data_array
    """
    return rat.getDataDensityMap(data_array, kernel_size) >= density_thresh


def getEntropyMask(orthoFile,
                   entropy_thresh=0.2, min_data_cluster=1000,
                   processing_res=8, kernel_size=None):
    """Classify areas of low entropy in an image such as water.

    :param orthoFile: Path to ortho image to process.
    :type orthoFile: str
    :param entropy_thresh: Minimum entropy threshold. 0.2 seems to be good for water.
    :type entropy_thresh: float
    :param min_data_cluster: Minimum number of contiguous data pixels in a kept data cluster.
    :type min_data_cluster: int
    :param processing_res: Resample to this resolution (in meters) for processing for speed and smooth.
    :type processing_res: float
    :param kernel_size: Side length of square neighborhood (of ones) for entropy filter.
    :type kernel_size: int
    :return: The low entropy classification mask from the geotif image in orthoFile.
    :rtype: ndarray of type bool, same shape as raster data in orthoFile

    Source file: entropyMask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 10/17/17

    Source docstring:
    % entropyMask classify areas of low entropy in an image such as water
    %
    % M = entropyMask(orthoFile) returns the low entropy classification mask
    % from the geotif image in orthoFile. Also checks whether wvc was applied
    % from the metafile.
    %
    % Ian Howat,ihowat@gmail.com, Ohio State
    % 13-Apr-2017 10:41:41
    """
    if kernel_size is None:
        kernel_size = int(math.floor(21*2/processing_res))

    ortho_array, mask_shape, image_res = rat.extractRasterParams(orthoFile, 'array', 'shape', 'res')

    background_mask = (ortho_array == 0)  # image background mask

    # Resize ortho to pres.
    if image_res != processing_res:
        ortho_array = rat.imresize(ortho_array, image_res/processing_res, 'bicubic')

    # Subtraction image
    ortho_subtraction = (  ndimage.maximum_filter1d(ortho_array, kernel_size, axis=0)
                         - ndimage.minimum_filter1d(ortho_array, kernel_size, axis=0))

    # Entropy image
    entropy_array = rat.entropyfilt(ortho_subtraction, np.ones((kernel_size, kernel_size)))
    mask = (entropy_array < entropy_thresh)
    del entropy_array

    mask = maskDataClustersAndHoles(mask,
                                    min_data_cluster=min_data_cluster,
                                    hole_fill_max_size=min_data_cluster)

    # Resize ortho to 8m.
    if image_res != processing_res:
        mask = rat.imresize(mask, mask_shape, 'nearest').astype(np.bool)

    mask[background_mask] = False

    return mask


def getSlopeMask(dem_array,
                 x_dem=None, y_dem=None,
                 res=None,
                 dilate_bad=13, avg_kernel_size=None):
    # TODO: Clean up docstring.
    """Mask artifacts with high slope values in a DEM array.

    The returned mask sets to 1 all pixels for which the surrounding [kernel_size x kernel_size]
    neighborhood has an average slope greater than 1, then erode it by a kernel of ones with side
    length dilate_bad.
    Provide either (x and y coordinate vectors) or image resolution "res".

    :param dem_array: DEM data 2D array
    :type dem_array: ndarray, 2D
    :param x_dem: coordinate vector x
    :type x_dem: ndarray, 1D
    :param y_dem: coordinate vector y
    :type y_dem: ndarray, 2D
    :param res: square resolution of pixels in z (meters)
    :type res: float
    :param dilate_bad: dilates masked pixels by this number of surrounding pixels
    :type dilate_bad: int
    :param avg_kernel_size: size of kernel for calculating mean slope, use 21 for 2m, 5 for 8m source data
    :type avg_kernel_size: int
    :return: The slope mask of the input DEM data array.
    :rtype: ndarray of type bool, same shape as z

    *Source file: edgeSlopeMask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 10/17/17

    Source docstring:
    % ESDGESLOPEMASK mask artificats on edges of DEM using high slope values
    %
    % M = edgeSlopeMask(x,y,z) masks bad edge values in the DEM with coordinate
    % vectors x any y.
    %
    % [M,dx,dy] = edgeSlopeMask(x,y,z) returns the gradient arrays
    %
    % Ian Howat, ihowat@gmail.com
    % 24-Jul-2017 15:39:07
    % 04-Oct-2017 15:19:47: Added option/value argument support

    *Functionality has been modified in translation:
        - Removal of edge masking.
        To replicate functionality of edgeSlopeMask.m, pass the result of this function to maskEdges().
    """
    if (   (res is None and (x_dem is None or y_dem is None))
        or (res is not None and (x_dem is not None or y_dem is not None))):
        raise InvalidArgumentError("One type of pixel spacing inputs (coordinate vectors [x, y], res) must be provided")
    if res is not None:
        res = abs(x_dem[1] - x_dem[0])
    if avg_kernel_size is None:
        avg_kernel_size = int(math.floor(21*2/res))

    # Get elevation grade at each pixel.
    dy, dx = (np.gradient(dem_array, y_dem, x_dem) if (x_dem is not None and y_dem is not None)
         else np.gradient(dem_array, res))
    grade = np.sqrt(np.square(dx) + np.square(dy))

    # Mean slope
    mean_slope_array = rat.moving_average(grade, avg_kernel_size, mode='same')

    # Mask mean slopes greater than 1.
    mask = mean_slope_array < 1

    # TODO: Check if the following can be accomplished with a binary erosion instead.
    # -t    Change name of variable dilate_bad (also in docstring) if so.
    # Dilate high mean slope pixels and set to false.
    mask[rat.imdilate_binary((mean_slope_array > 1), np.ones((dilate_bad, dilate_bad)))] = False

    return mask


def getWaterMask(ortho_array, meanSunElevation, data_density_map,
                 sunElevation_split=30, ortho_thresh_low=5, ortho_thresh_high=20,
                 entropy_thresh=0.2, data_density_thresh=0.98, min_data_cluster=500,
                 kernel_size=5, dilate=7):
    # TODO: Write my own docstring.
    """
    Source file: waterMask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 10/19/17
    """
    ortho_thresh = ortho_thresh_low if meanSunElevation < sunElevation_split else ortho_thresh_high

    # Subtraction image
    # FIXME: Should the following really be using the 1d versions of max and min filter?
    ortho_subtraction = (  ndimage.maximum_filter1d(ortho_array, kernel_size, axis=0)
                         - ndimage.minimum_filter1d(ortho_array, kernel_size, axis=0))

    # Entropy image
    entropy_array = entropy(ortho_subtraction.astype(np.uint8), np.ones((kernel_size, kernel_size)))

    # Set edge-effected values to zero.
    entropy_array[ortho_array == 0] = 0

    # Mask data with entropy less than threshold.
    entropy_mask = ((ortho_array != 0) & (entropy_array < entropy_thresh))

    # Remove isolated clusters of masked pixels.
    entropy_mask = rat.bwareaopen(entropy_mask, min_data_cluster, in_place=True)

    # Dilate masked pixels.
    entropy_mask = rat.imdilate_binary(entropy_mask, np.ones((dilate, dilate)))

    # Mask data with low radiance and matchpoint density.
    radiance_mask = ((ortho_array != 0) & (ortho_array < ortho_thresh) & (data_density_map < data_density_thresh))

    # Remove isolated clusters of masked pixels.
    radiance_mask = rat.bwareaopen(radiance_mask, min_data_cluster, in_place=True)

    # Assemble water mask.
    mask = (~entropy_mask & ~radiance_mask & (ortho_array != 0))

    # Remove isolated clusters of data.
    mask = maskDataClustersAndHoles(mask,
                                    min_data_cluster=min_data_cluster,
                                    hole_fill_max_size=min_data_cluster)

    return mask


def getCloudMask(dem_array, ortho_array, data_density_map,
                 elevation_percentile_split=80, ortho_thresh_cloud=70,
                 data_density_thresh_cloud=0.9, data_density_thresh_nocloud=0.6,
                 min_data_cluster=10000, min_nodata_cluster=1000,
                 avg_kernel_size=21, dilate_bad=21,
                 erode_border=31, dilate_border=61):
    # TODO: Write my own docstring.
    """
    Source file: cloudMask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 10/19/17

    Source docstring:
    % cloudMask mask bad surfaces on DEM based on slope and radiance

    % M = cloudMask(z,or)masks bad edge values in the DEM with coordinate
    % vectors x any y.
    %
    % Ian Howat, ihowat@gmail.com
    % 24-Jul-2017 15:39:07
    """

    # Make sure sufficient non NaN pixels exist, otherwise cut to the chase.
    if np.sum(~np.isnan(dem_array)) < 2*min_nodata_cluster:
        mask = np.ones(dem_array.shape, dtype=np.bool)
        return mask

    # Calculate standard deviation of elevation.
    mean_elevation_array = rat.moving_average(dem_array, avg_kernel_size, mode='same')
    stdev_elevation_array = np.sqrt(
        rat.moving_average(np.square(dem_array), avg_kernel_size, mode='same') - np.square(mean_elevation_array)
    )
    stdev_elevation_array = np.real(stdev_elevation_array)

    # Calculate elevation percentile difference.
    percentile_diff = (  np.percentile(dem_array, elevation_percentile_split)
                       - np.percentile(dem_array, 100 - elevation_percentile_split))

    # Set standard deviation difference based on percentile difference.
    stdev_thresh = None
    if percentile_diff <= 40:
        stdev_thresh = 10.5
    elif 40 < percentile_diff <= 50:
        stdev_thresh = 15
    elif 50 < percentile_diff <= 75:
        stdev_thresh = 19
    elif 75 < percentile_diff <= 100:
        stdev_thresh = 27
    elif percentile_diff > 100:
        stdev_thresh = 50

    print "{}/{} percentile elevation difference: {:.1f}, sigma-z threshold: {:.1f}".format(
        100 - elevation_percentile_split, elevation_percentile_split, percentile_diff, stdev_thresh
    )

    # Apply mask conditions.
    mask = (~np.isnan(dem_array)
            & (((ortho_array > ortho_thresh_cloud) & (data_density_map < data_density_thresh_cloud))
                | (data_density_map < data_density_thresh_nocloud)
                | (stdev_elevation_array > stdev_thresh)))

    # Fill holes in masked clusters.
    mask = ndimage.morphology.binary_fill_holes(mask)

    # Remove small masked clusters.
    mask = rat.bwareaopen(mask, min_nodata_cluster, in_place=True)

    # Remove thin borders caused by cliffs/ridges.
    mask_edge = rat.imerode_binary(mask, np.ones((erode_border, erode_border)))
    mask_edge = rat.imdilate_binary(mask_edge, np.ones((dilate_border, dilate_border)))

    mask = (mask & mask_edge)

    # Dilate nodata.
    mask = rat.imdilate_binary(mask, np.ones((dilate_bad, dilate_bad)))

    # Remove small clusters of unfiltered data.
    mask = ~rat.bwareaopen(~mask, min_data_cluster, in_place=True)

    return mask


def maskEdges(data_array, hull_concavity=0.5, crop=None,
              res=None, min_data_cluster=1000):
    """Mask bad edges in a data (matchtag) array.

    The input array is presumed to contain a large "mass" of data (non-zero) values near its center,
    which may or may not have holes.
    The returned mask discards all area outside of the (convex) hull of the region containing both
    the data mass and all data clusters of more pixels than min_data_cluster.

    :param data_array: The 2D array of data values to mask.
    :type data_array: ndarray, 2D
    :param hull_concavity: Boundary curvature factor (0 = convex hull, 1 = point boundary)
    :type hull_concavity: float
    :param crop: Erode the mask by a square neighborhood (ones) of this side length before return.
    :type crop: int
    :param res: Image resolution corresponding to data_array, for setting parameter default values.
    :type res: float
    :param min_data_cluster: Minimum number of contiguous data pixels in a kept data cluster.
    :type min_data_cluster: int
    :return: The edge mask of the input data array.
    :rtype: ndarray of type bool, same shape as data_array

    *Source file: edgeMask.m
    Source author: Ian Howat, ihowat@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 10/17/17

    Source docstring:
    % edgeMask returns mask for bad edges using the matchag field
    %
    % m1 = edgeMask(m0) where m0 is the matchtag array returns a binary mask of
    % size(m0) designed to filter data bad edges using match point density

    *Functionality has been modified in translation:
        - Removal of data density masking.
        - Removal of entropy masking.
        To replicate functionality of edgeMask.m, do masking of data_array with getDataDensityMask()
        and getEntropyMask() before passing the result to this function.
    """
    if res is None and min_data_cluster is None:
        raise InvalidArgumentError("Resolution 'res' argument must be provided to set default values"
                                   " of other parameters")
    if not np.any(data_array):
        return data_array.astype(np.bool)
    if min_data_cluster is None:
        min_data_cluster = int(math.floor(1000*2/res))

    # Fill interior holes since we're just looking for edges here.
    mask = ndimage.morphology.binary_fill_holes(data_array)
    # Get rid of isolated little clusters of data.
    mask = rat.bwareaopen(mask, min_data_cluster, in_place=True)

    if not np.any(mask):
        # No clusters exceed minimum cluster area.
        return mask

    # Find data coverage boundaries.
    data_boundary = rat.bwboundaries_array(mask, connectivity=8, noholes=True)
    boundary_points = np.argwhere(data_boundary)

    del mask, data_boundary

    # ##################################################################
    # # Concave hull method, using Ken Clarkson's Hull (ANSI C program)
    # print "Computing concave hull"
    #
    # # Downsample boundary to < 10000 points, since Hull cannot handle any more than that.
    # downsample_factor = len(boundary_points) // 10000 + 1
    # bpoints_sample = boundary_points[::downsample_factor]
    #
    # # Write boundary points into temporary file.
    # bpoints_txt = open(EM_TEMPFILE_BPOINTS, 'w')
    # for p in bpoints_sample:
    #     bpoints_txt.write((str(p)[1:-1]).lstrip() + '\n')
    # bpoints_txt.close()
    #
    # # -A argument to first call of clarkson-hull is supposed to:
    # # "compute the alpha shape of the input, finding the smallest alpha
    # #  so that the sites are all contained in the alpha-shape."
    # # This value of alpha is printed on the command line after being computed,
    # # so our objective is to capture it.
    # command = "clarkson-hull -A -i{} -oF{}".format(EM_TEMPFILE_BPOINTS, EM_TEMPFILE_CCHULL)
    # alpha_pattern = re.compile("alpha=(.*)")
    # alpha = -1
    # for line in run_command(command):
    #     print line
    #     match = re.search(alpha_pattern, line)
    #     if match:
    #         alpha = float(match.group(1).strip())
    #
    # if alpha == -1:
    #     raise MaskingError("Initial call to clarkson-hull failed to create alpha shape for edgemask")
    #
    # # -aa argument specifies the alpha value to use in the second call to clarkson-hull.
    # # After testing, a value 100 times the alpha value reported from the -A run creates
    # # a similar concave hull to MATLAB's boundary function with curvature factor of 0.5.
    # command = "clarkson-hull -aa{} -i{} -oF{}".format(alpha*100, EM_TEMPFILE_BPOINTS, EM_TEMPFILE_CCHULL)
    # subprocess.check_call(command.split())
    #
    # hull_edges = np.loadtxt(EM_TEMPFILE_CCHULL+'-alf', dtype=int, skiprows=1)
    # hull_vertices = rat.connectEdges(hull_edges)
    # hull_edgepoints = bpoints_sample[hull_vertices]
    # row_coords, column_coords = draw.polygon(bpoints_sample[hull_vertices, 0],
    #                                          bpoints_sample[hull_vertices, 1])
    # ##################################################################

    # #############################
    # # Concave hull method (old)
    # print "Computing concave hull"
    # hull = alpha_shape(boundary_points, alpha=.007)
    # edge_coords_r, edge_coords_c = hull.exterior.coords.xy
    # row_coords, column_coords = draw.polygon(edge_coords_r,
    #                                          edge_coords_c)
    # #############################

    #############################
    # Convex hull method
    print "Computing convex hull"
    hull = spatial.ConvexHull(boundary_points)
    row_coords, column_coords = draw.polygon(hull.points[hull.vertices, 0],
                                             hull.points[hull.vertices, 1])
    # hull_points = boundary_points[hull.vertices]  # For testing purposes.
    # print hull_points
    del hull
    #############################

    mask = np.zeros(data_array.shape).astype(np.bool)
    mask[row_coords, column_coords] = 1
    if crop is not None:
        mask = rat.imerode_binary(mask, structure=np.ones((crop, crop)))

    return mask


def maskDataClustersAndHoles(data_array, min_data_cluster=1000, hole_fill_max_size=10000):
    """Mask a data array to remove small data clusters and fill small voids of no data.

    Data is defined as pixels with non-zero value.
    The returned mask discards all data clusters of fewer pixels than min_data_cluster, while
    filling all no-data voids of fewer pixels than hole_fill_max_size.

    :param data_array: The 2D array of data values to mask.
    :type data_array: ndarray, 2D
    :param min_data_cluster: Minimum number of contiguous data pixels in a kept data cluster.
    :type min_data_cluster: int
    :param hole_fill_max_size: Maximum number of contiguous no-data pixels in a filled data void.
    :type min_data_cluster: int
    :return: The cluster and hole mask of the input data array.
    :rtype: ndarray of type bool, same shape as data_array
    """
    if not np.any(data_array):
        return data_array.astype(np.bool)

    # Remove small data clusters.
    mask =  rat.bwareaopen(data_array, min_data_cluster)
    # Fill small data voids.
    return ~rat.bwareaopen(~mask, hole_fill_max_size, in_place=True)


def readSceneMeta(metaFile):
    # TODO: Write my docstring.
    """
    Source file: readSceneMeta.m
    Source author: Ian Howat, ihowat@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 10/20/17

    Source docstring:
    %READMETA read SETSM output metafile for an individual DEM
    %
    % meta=readSceneMeta(metaFile) reads each line of the metaFile into the
    % structure meta. The field neames are the same as in the metaFile but with
    % underscores for spaces and all lowercase.
    %
    % Ian Howat, ihowat@gmail.com
    % 24-Jul-2017 14:57:49
    """
    meta = {}
    metaFile_fp = open(metaFile, 'r')
    line = metaFile_fp.readline()
    while line != '':
        equal_index = line.find('=')
        if equal_index != -1:
            field_name = line[:equal_index].strip().replace(' ', '_').lower()
            if field_name in meta:
                meta['image_1_'+field_name] = meta.pop(field_name)
                field_name = 'image_2_'+field_name
            field_value = line[(equal_index+1):].strip()
            try:
                field_value = float(field_value)
            except ValueError:
                pass
            meta[field_name] = field_value
        line = metaFile_fp.readline()
    metaFile_fp.close()

    # Get satID and check for cross track naming convention.
    satID = os.path.basename(meta['image_1'])[0:1].upper()
    if   satID == 'W1':
        satID = 'WV01'
    elif satID == 'W2':
        satID = 'WV02'
    elif satID == 'W3':
        satID = 'WV03'
    elif satID == 'G1':
        satID = 'GE01'
    elif satID == 'Q1':
        satID = 'QB01'
    elif satID == 'Q2':
        satID = 'QB02'
    elif satID == 'I1':
        satID = 'IK01'
    meta['image_1_satID'] = satID

    return meta


def rescaleDN(ortho_array, dnmax):
    # TODO: Write my docstring.
    """
    Source file: rescaleDN.m
    Source author: Ian Howat, ihowat@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 10/23/17

    Source docstring:
    % RESCALEDN rescale digitial numbers to new maximum
    %
    % dn=rescaleDN(dn,dnmax)rescales the digital number image dn to the new
    % maximum in dnmax.
    %
    % Ian Howat, ihowat@gmail.com
    % 24-Jul-2017 15:50:25
    """
    # Set the minimum and maximum values of this scale.
    # We use a fixed scale because this is what all data is scaled to after application of
    # wv_correct regardless of actual min or max.
    ormin = 0
    ormax = 32767

    # Set the new minimum and maximum.
    # dnmin is zero because nodata is apparently used in the scaling.
    dnmin = 0
    dnmax = float(dnmax)

    # Rescale back to original dn.
    return dnmin + (dnmax-dnmin)*(ortho_array.astype(np.float32) - ormin)/(ormax-ormin)


def DG_DN2RAD(DN,
              xmlFile=None,
              satID=None, effectiveBandwith=None, abscalFactor=None):
    # TODO: Write my docstring.
    """
    Source file: DG_DN2RAD.m
    Source author: Ian Howat, ihowat@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 10/23/17

    Source docstring:
    % DG_DN2RAD converts DG DN images to top-of-atmosphere radiance
    %
    % L =  = DG_DN2RAD(DN, satID, effectiveBandwith, abscalFactor) applies the
    % conversion using the supplied factors with a table look-up for the satID.
    % The output L is top-of-atmosphere radiance in units of Wµm^-1 m^-2 sr^-1.
    %
    % L =  = DG_DN2RAD(DN,xmlFile) reads the factors from the supplied xml file
    %
    % [L, effectiveBandwith, abscalFactor, gain, offset] = DG_DN2RAD(...)
    % returns the scaling parameters used.
    """
    xml_params = [
        [satID, 'SATID'],
        [effectiveBandwith, 'EFFECTIVEBANDWIDTH'],
        [abscalFactor, 'ABSCALFACTOR']
    ]
    if None in [p[0] for p in xml_params]:
        if xmlFile is None:
            raise InvalidArgumentError("'xmlFile' argument must be given to automatically set xml params")
        fillMissingXmlParams(xmlFile, xml_params)
        satID, effectiveBandwith, abscalFactor = [p[0] for p in xml_params]
        effectiveBandwith = float(effectiveBandwith)
        abscalFactor = float(abscalFactor)

    # Values from:
    # https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/209/DGConstellationAbsRadCalAdjustmentFactors_2015v2.pdf
    sensor = ('WV03',   'WV02', 'GE01', 'QB2',  'IKO',  'WV01')
    gain   = (0.923,    0.96,   0.978,  0.876,  0.907,  1.016)
    offset = [-1.7,     -2.957, -1.948, -2.157, -4.461, -3.932]

    sensor_index = sensor.index(satID)
    gain = gain[sensor_index]
    offset = offset[sensor_index]

    DN = DN.astype(np.float32)
    DN[DN == 0] = np.nan
    return gain*DN*(abscalFactor/effectiveBandwith) + offset


def fillMissingXmlParams(xmlFile, xml_params):
    # TODO: Write docstring.
    xml_paramstrs = [p[1] for p in xml_params]
    xml_paramstrs_to_read = [p[1] for p in xml_params if p[0] is None]
    for paramstr, paramval in zip(xml_paramstrs_to_read, readFromXml(xmlFile, xml_paramstrs_to_read)):
        xml_params[xml_paramstrs.index(paramstr)][0] = paramval


def readFromXml(xmlFile, xml_paramstrs):
    # TODO: Write docstring.
    xml_paramstrs_left = list(xml_paramstrs)
    values = [None]*len(xml_paramstrs)
    xmlFile_fp = open(xmlFile, 'r')
    line = xmlFile_fp.readline()
    while line != '' and None in values:
        for ps in xml_paramstrs_left:
            if ps in line:
                values[xml_paramstrs.index(ps)] = line.replace("<{}>".format(ps), '').replace("</{}>".format(ps), '')
                xml_paramstrs_left.remove(ps)
                break
    xmlFile_fp.close()
    return values


# The following functions (alpha_shape and add_edge)
# are based on KEVIN DWYER's code found at the following URL:
# http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
def alpha_shape(coords, alpha):
    """
    Computes the alpha shape (concave hull) of a set of points.
    @param coords: array of coords
    @param alpha: alpha value to influence the
        gooey-ness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    tri = spatial.Delaunay(coords)
    edges = set()
    edge_points = []
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semi-perimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    union = unary_union(triangles)

    return union


def add_edge(edges, edge_points, coords, i, j):
    """
    Add a line between the i-th and j-th points,
    if not in the list already.
    """
    if (i, j) in edges or (j, i) in edges:
        return
    edges.add((i, j))
    edge_points.append(coords[[i, j]])
