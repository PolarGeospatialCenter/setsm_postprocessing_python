# -*- coding: UTF-8 -*-

# Version 3.1; Erik Husby, Ryan Shellberg; Polar Geospatial Center, University of Minnesota; 2019
# Translated from MATLAB code written by Ian Howat, Ohio State University, 2018


from __future__ import division
import math
import os
import re
import sys
import traceback
from warnings import warn
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import cv2
import numpy as np
import skimage.morphology as sk_morphology
from scipy import ndimage as sp_ndimage

from batch_scenes2strips import getDemSuffix, getMatchtagSuffix, selectBestMatchtag, selectBestOrtho, selectBestOrtho2
from testing import TESTDIR
if sys.version_info[0] < 3:
    import raster_array_tools as rat
    from script_utils import VersionString
else:
    from lib import raster_array_tools as rat
    from lib.script_utils import VersionString


BITMASK_VERSION_NUM = VersionString('1.2')

MASK_FLAT = 0
MASK_SEPARATE = 1
MASK_BIT = 2
MASKCOMP_EDGE_NAME = 'edgemask'
MASKCOMP_WATER_NAME = 'watermask'
MASKCOMP_CLOUD_NAME = 'cloudmask'
MASKCOMP_EDGE_BIT = 0
MASKCOMP_WATER_BIT = 1
MASKCOMP_CLOUD_BIT = 2
MASKCOMP_NAME_BIT_ZIP = list(zip(
    [MASKCOMP_EDGE_NAME, MASKCOMP_WATER_NAME, MASKCOMP_CLOUD_NAME],
    [MASKCOMP_EDGE_BIT, MASKCOMP_WATER_BIT, MASKCOMP_CLOUD_BIT]
))


DEBUG_DIR = TESTDIR
DEBUG_FNAME_PREFIX = ''
DEBUG_NONE = 0
DEBUG_ALL = 1
DEBUG_MASKS = 2
DEBUG_ITHRESH = 3


ITHRESH_START_TAG = 'exec(ITHRESH_START)'
ITHRESH_END_TAG = '# ITHRESH_END'
ITHRESH_START = """
if debug_component_masks != DEBUG_NONE:
    ithresh_num = 1 if 'ithresh_num' not in vars() else ithresh_num + 1
    if debug_component_masks in (DEBUG_ALL, DEBUG_ITHRESH):
        ithresh_save(ithresh_num, vars())
    elif debug_component_masks == DEBUG_MASKS:
        ithresh_data = ithresh_load(ithresh_num)
        for thresh in ithresh_data:
            exec('{} = ithresh_data[thresh]'.format(thresh))
"""
ITHRESH_QUICK_SAVED = False


HOSTNAME = os.getenv('HOSTNAME')
# if HOSTNAME is not None:
#     HOSTNAME = HOSTNAME.lower()
#     RUNNING_AT_PGC = True if True in [s in HOSTNAME for s in ['rookery', 'nunatak']] else False
# else:
#     RUNNING_AT_PGC = False
RUNNING_AT_PGC = False
RE_SCENE_DEM_FNAME_PARTS_STR = "^([A-Z0-9]{4})_([0-9]{4})([0-9]{2})([0-9]{2})_([0-9A-F]{16})_([0-9A-F]{16})_(.+?\-)?([A-Z0-9]+_[0-9]+)_(P[0-9]{3})_(.+?\-)?([A-Z0-9]+_[0-9]+)_(P[0-9]{3})_.*$"
RE_SCENE_DEM_FNAME_PARTS = re.compile(RE_SCENE_DEM_FNAME_PARTS_STR)
RE_SOURCE_IMAGE_FNAME_PARTS_STR = "^([A-Z0-9]{4})_([0-9]{4})([0-9]{2})([0-9]{2})([0-9]{6})_([0-9A-F]{16})_[0-9]{2}([A-Z]{3})[0-9]{2}[0-9]{6}\-(.+?)\-([A-Z0-9]+_[0-9]+)_(P[0-9]{3})\..*$"
RE_SOURCE_IMAGE_FNAME_PARTS = re.compile(RE_SOURCE_IMAGE_FNAME_PARTS_STR)


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class RasterDimensionError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class MaskComponentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class MetadataError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class DebugError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


def check_arggroups(arggroup_list, check='exist'):
    """
    Given a list of items, some of which may also be lists,
    report whether or not each item is not None or contains
    elements which are not None.

    This function is meant to be called in the verification step
    of functions that have multiple sets of optional arguments
    which need to be filtered for proper usage.

    Parameters
    ----------
    arggroup_list : tuple or list
        Collection of items to be checked.
    check : str; 'exist' or 'full'
        Which metric to use when reporting the state of an item
        in `arggroup_list` that is a collection of elements.
        If 'exist', a collection that contains at least one
        non-None element passes the check.
        If 'full', a collection must contain only non-None
        elements to pass the check.

    Returns
    -------
    arggroup_report : list of bool, same length as `arggroup_list`
        An ordered list of boolean elements corresponding to items
        in `arggroup_list` reporting the results of the check on
        each item.

    """
    check_choices = ('exist', 'full')
    if check not in check_choices:
        raise InvalidArgumentError("`check` must be one of {}, but was '{}'".format(check_choices, check))

    arggroup_report = []
    for arggroup in arggroup_list:
        ag_check = None
        if type(arggroup) in (list, tuple):
            set_count = [arg is not None for arg in arggroup].count(True)
            if (   (check == 'full'  and set_count == len(arggroup))
                or (check == 'exist' and set_count > 0)):
                ag_check = True
            else:
                ag_check = False
        elif arggroup is not None:
            ag_check = True
        else:
            ag_check = False
        arggroup_report.append(ag_check)

    return arggroup_report


def isValidArggroups(arggroup_list):
    """
    Given a list of items, some of which may also be lists,
    report whether or not there is only one item that both
    (1) is not None or contains some elements which are not
    None, and (2) if that item is a list, it contains only
    non-None elements.

    This function is meant to be called in the verification step
    of functions that have multiple sets of optional arguments
    of which one is required to be provided, while the sets of
    arguments are mutually exclusive.

    Parameters
    ----------
    arggroup_list : tuple or list
        Collection of items to be verified.

    Returns
    -------
    isValidArggroups : bool
        Flag indicating whether or not the argument groups
        passes the described verification.

    """
    if (   (check_arggroups(arggroup_list, check='exist').count(True) != 1)
        or (check_arggroups(arggroup_list, check='full').count(True)  != 1)):
        return False
    else:
        return True


def generateMasks(demFile, mask_version, dstdir=None, noentropy=False, nbit_masks=False,
                  save_component_masks=None, debug_component_masks=DEBUG_NONE,
                  use_second_ortho=False, use_pil_imresize=False):
    """
    Create and save scene masks that mask ON regions of bad data.

    Parameters
    ----------
    demFile : str (file path)
        File path of the scene DEM image that is to be masked.
    mask_version : str; 'maskv1', 'mask', 'mask2a',
                        'mask8m', 'bitmask', or 'maskv2_debug'
        Type of mask(s) to create, and what filename suffix(es) to use
        when saving the mask(s) to disk.
    dstdir : None or str (directory path)
        Path of the output directory for saved masks.
        If None, `dstdir` is set to the directory of `demFile`.
    noentropy : bool
        (Option only applies when mask_version='edgemask/datamask'.)
        If True, entropy filter is not applied.
        If False, entropy filter is applied.
    nbit_masks : bool
        If True, save mask raster images in n-bit format.
        If False, save in uint8 format.
    save_component_masks : bool
        (Option only applies when mask_version='maskv2_debug'.)
        If MASK_FLAT, save one binary mask with all filter
          components merged.
        If MASK_SEPARATE, save additional binary masks for
          edge, water, and cloud filter components as separate rasters.
        If MASK_BIT, save one n-bit mask with each bit
          representing one of the n different edge, water, and cloud
          filter components.
    debug_component_masks : int
        (Option only applies when mask_version='mask'.)
        If DEBUG_NONE, has no effect.
        If DEBUG_ALL, perform all of the following.
        If DEBUG_MASKS, save additional
          edge/water/cloud mask components as separate rasters.
        If DEBUG_ITHRESH, save data for interactive testing
          of threshold values for component mask generation.

    Returns
    -------
    None

    Notes
    -----
    Mask is saved at `demFile.replace('dem.tif', mask_version+'.tif')`.
    For mask_version='maskv1', separate mask rasters '*_edgemask.tif'
    and '*_datamask.tif' are saved.

    """
    suffix_choices = ('maskv1', 'mask', 'maskv2_debug', 'mask2a', 'mask8m', 'bitmask')
    mask_dtype = 'n-bit' if nbit_masks else 'uint8'
    demFname = os.path.basename(demFile)
    demSuffix = getDemSuffix(demFile)
    if dstdir is None:
        dstdir = os.path.dirname(demFile)

    mask_sets_to_combine = []
    combine_image_num = [1]
    if use_second_ortho:
        combine_image_num.append(2)

    for image_num in combine_image_num:
        masks = dict()

        if mask_version.endswith('maskv1'):
            masks = mask_v1(demFile, noentropy, image_num=image_num)
        else:
            if mask_version.endswith(('bitmask', 'mask', 'maskv2_debug')):
                if save_component_masks is None:
                    if mask_version.endswith('bitmask'):
                        save_component_masks = MASK_BIT
                    elif mask_version.endswith('mask'):
                        save_component_masks = MASK_FLAT
                    elif mask_version.endswith('maskv2_debug'):
                        save_component_masks = MASK_SEPARATE
                mask = mask_v2(demFile, mask_version,
                               save_component_masks=(save_component_masks != MASK_FLAT),
                               debug_component_masks=debug_component_masks,
                               image_num=image_num)
            elif mask_version.endswith('mask2a'):
                mask = mask_v2a(demFile)
            elif mask_version.endswith('mask8m'):
                mask = mask8m(demFile)
            else:
                raise InvalidArgumentError("`mask_version` must end with one of {}, "
                                           "but was {}".format(suffix_choices, mask_version))

            if type(mask) == dict:
                component_masks = mask

                if save_component_masks == MASK_BIT:
                    mask_bin = component_masks[mask_version]
                    mask_comp = np.zeros_like(mask_bin, dtype=np.uint8)
                    for mask_name, mask_bit in MASKCOMP_NAME_BIT_ZIP:
                        np.bitwise_or(mask_comp, np.left_shift(component_masks[mask_name].astype(np.uint8), mask_bit), out=mask_comp)
                    try:
                        if not np.array_equal(mask_comp.astype(np.bool), mask_bin):
                            raise MaskComponentError("Coverage of edge/water/cloud component mask arrays "
                                                     "does not match coverage of official binary mask array")
                    except MaskComponentError:
                        print("Saving mask coverage arrays in question for inspection")
                        from testing.test import saveImage
                        saveImage(mask_bin, demFname.replace(demSuffix, mask_version+'_binary.tif'), overwrite=True)
                        saveImage(mask_comp.astype(np.bool), demFname.replace(demSuffix, mask_version+'_components.tif'), overwrite=True)
                        raise
                    if debug_component_masks in (DEBUG_MASKS, DEBUG_ALL):
                        component_masks['binary'] = mask_bin
                        if len(combine_image_num) > 1:
                            component_masks['{}_image{}'.format(mask_version, image_num)] = mask_comp
                    else:
                        for mask_name in (MASKCOMP_EDGE_NAME, MASKCOMP_WATER_NAME, MASKCOMP_CLOUD_NAME):
                            del component_masks[mask_name]
                    component_masks[mask_version] = mask_comp

                mask = component_masks[mask_version]
                masks = {'{}_{}{}'.format(mask_version, 'image{}_'.format(image_num)*(len(combine_image_num) > 1), mask_name): mask_array
                         for mask_name, mask_array in component_masks.items()
                         if (mask_name != mask_version)}

            masks[mask_version] = mask
        mask_sets_to_combine.append(masks)

    if len(mask_sets_to_combine) == 1:
        masks = mask_sets_to_combine[0]
    else:
        masks_merged = dict()
        for mask_set in mask_sets_to_combine:
            for mask_name, mask in mask_set.items():
                if mask_name not in masks_merged:
                    masks_merged[mask_name] = mask
                else:
                    masks_merged[mask_name] = np.bitwise_or(masks_merged[mask_name], mask)
        masks = masks_merged

    image_shape = rat.extractRasterData(demFile, 'shape')

    nbits = None
    for mask_name, mask in masks.items():
        maskFile = os.path.join(dstdir, demFname.replace(demSuffix, mask_name+'.tif'))
        if mask_dtype == 'n-bit':
            nbits = 3 if (save_component_masks == MASK_BIT and mask_name == mask_version) else 1

        if mask.shape != image_shape:
            print("Resizing mask array {} to native image size {}".format(mask.shape, image_shape))
            mask = rat.imresize(mask, image_shape, 'nearest', method=('pil' if use_pil_imresize else 'cv2'))

        rat.saveArrayAsTiff(mask, maskFile, like_raster=demFile, nodata_val=1, dtype_out=mask_dtype, nbits=nbits, co_predictor=1)

        del mask


def mask_v1(demFile, noentropy=False,
            image_num=1):
    """
    Creates an edgemask and datamask masking ON regions of bad data in
    a scene from a matchtag image and saves the two mask files to disk.

    Optionally, the corresponding ortho-ed spectral image may be
    considered so that image regions with low data density in the
    matchtag but low entropy values in the ortho image are masked.

    Parameters
    ----------
    demFile : str (file path)
        File path of the DEM raster image.
    noentropy : bool
        If True, entropy filter is not applied.
        If False, entropy filter is applied.

    Returns
    -------
    component_masks : dict of ndarray of bool, 2D
        Scene mask masking ON regions of good data.

    Notes
    -----
    Edgemask is saved at matchFile.replace('matchtag.tif', 'edgemask.tif'),
    and likewise with datamask.

    Source file: batch_mask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 12/05/17

    """
    component_masks = {}

    matchFile = selectBestMatchtag(demFile)
    if image_num == 1:
        orthoFile = selectBestOrtho(demFile)
    elif image_num == 2:
        orthoFile = selectBestOrtho2(demFile)
    metaFile = matchFile.replace(getMatchtagSuffix(matchFile), 'meta.txt')

    # Find SETSM version.
    setsmVersion = None
    if not os.path.isfile(metaFile):
        print("No meta file, assuming SETSM version > 2.0")
        setsmVersion = 3
    else:
        setsm_pattern = re.compile("SETSM Version=(.*)")
        metaFile_fp = open(metaFile, 'r')
        line = metaFile_fp.readline()
        while line != '':
            match = re.search(setsm_pattern, line)
            if match:
                try:
                    setsmVersion = float('.'.join(match.group(1).strip().split('.')[0:2]))
                except ValueError:
                    setsmVersion = float('inf')
                break
            line = metaFile_fp.readline()
        metaFile_fp.close()
        if setsmVersion is None:
            warn("Missing SETSM Version number in '{}'".format(metaFile))
            # Use settings for default SETSM version.
            setsmVersion = 2.03082016
    print("Using settings for SETSM Version = {}".format(setsmVersion))

    match_array, res = rat.extractRasterData(matchFile, 'array', 'res')

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

    high_datadensity_mask = getHighDataDensityMask(match_array, kernel_size=n, density_thresh=Pmin)
    if not noentropy:
        low_entropy_mask = getLowEntropyMask(orthoFile)
        mask = (high_datadensity_mask | low_entropy_mask)
    edgemask = getEdgeMask(mask, min_data_cluster=Amin, hull_concavity=cf, crop=crop)
    component_masks[MASKCOMP_EDGE_NAME] = edgemask

    match_array[edgemask] = 0
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

    datamask = getHighDataDensityMask(match_array, kernel_size=n, density_thresh=Pmin)
    del match_array
    datamask = clean_mask(datamask, remove_pix=Amin, fill_pix=Amax, in_place=True)
    component_masks['datamask'] = ~datamask

    return component_masks


def mask_v2(demFile=None, mask_version='mask',
            ddm_kernel_size=21, processing_res=8, min_data_cluster=500,
            save_component_masks=False, debug_component_masks=DEBUG_NONE,
            postprocess_mask=None, postprocess_res=None,
            image_num=1):
    # TODO: Write documentation for post-processing option.
    """
    Create a single mask masking ON regions of bad data in a scene,
    utilizing information from the DEM, matchtag, and panchromatic
    spectral images corresponding to a single scene.

    Crops out bad data on the edges of the scene where high average
    slope values are found in the DEM. Cuts out areas classified as
    water and clouds from the panchromatic image.

    Parameters
    ----------
    demFile : str (file path)
        File path of the DEM raster image.
    mask_version : str
        Suffix for the filename of the mask raster file to be created.
    ddm_kernel_size : positive int
        Side length of the neighborhood to use for calculating the
        data density map, at native image resolution.
        If None, is set to `int(math.floor(21*2/image_res))` where
        `image_res` is the resolution of the raster images.
    processing_res : positive int
        Downsample images to this resolution for processing for
        speed and smooth.
    min_data_cluster : positive int
        Minimum number of contiguous data pixels in a kept good data
        cluster in the returned mask, at `processing_res` resolution.
    save_component_masks : bool
        If True, return additional edge, water, and cloud mask arrays.
    debug_component_masks : int
        If DEBUG_NONE, has no effect.
        If DEBUG_ALL, perform all of the following.
        If DEBUG_MASKS, return additional
          edge/water/cloud mask component arrays.
        If DEBUG_ITHRESH, save data for interactive testing
          of threshold values for component mask generation.

    Returns
    -------
    mask_out, component_masks : (ndarray of bool, 2D; dict of the former)
        Scene mask masking ON regions of bad data.

    Notes
    -----
    This method is currently designed for masking images with 2-meter
    square pixel resolution.

    Source file: mask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 2/28/18

    Source docstring:
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
    if postprocess_mask is not None:
        # Handle post-processing option.
        mask = ~rat.bwareaopen(~postprocess_mask,
                               min_data_cluster*(processing_res/postprocess_res)**2)
        return mask

    demSuffix = getDemSuffix(demFile)
    matchFile = selectBestMatchtag(demFile)
    if image_num == 1:
        orthoFile = selectBestOrtho(demFile)
    elif image_num == 2:
        orthoFile = selectBestOrtho2(demFile)
    metaFile = demFile.replace(demSuffix, 'meta.txt')

    meta = readSceneMeta(metaFile)
    try:
        satID              = meta['image_{}_sensor'.format(image_num)]
        wv_correct_flag    = meta['image_{}_wv_correct'.format(image_num)]
        effbw              = meta['image_{}_effbw'.format(image_num)]
        abscalfact         = meta['image_{}_abscalfact'.format(image_num)]
        mean_sun_elevation = meta['image_{}_mean_sun_elevation'.format(image_num)]
        maxDN = meta['image_{}_max'.format(image_num)] if wv_correct_flag else None
    except KeyError as e:
        print(e)
        raise MetadataError("Scene metadata file ({}) is missing critical information for filtering step; "
                            "cannot proceed".format(metaFile))

    # Load DEM.
    dem_array, image_shape, image_gt = rat.extractRasterData(demFile, 'array', 'shape', 'geo_trans')
    image_dx = image_gt[1]
    image_dy = image_gt[5]
    image_res = abs(image_dx)
    resize_factor = (image_res / processing_res) if (image_res != processing_res) else None

    dem_array[dem_array == -9999] = np.nan
    if resize_factor is not None:
        dem_array = rat.imresize(dem_array, resize_factor)
    dem_nodata = np.isnan(dem_array)

    if image_res != processing_res:
        processing_dy, processing_dx = image_res * np.array(image_shape) / np.array(dem_array.shape)
    else:
        processing_dx = processing_res
        processing_dy = processing_res
    # Coordinate ascending/descending directionality affects gradient used in getSlopeMask.
    if image_dx < 0:
        processing_dx = -processing_dx
    if image_dy < 0:
        processing_dy = -processing_dy

    # Load matchtag and get data density map.
    match_array = rat.extractRasterData(matchFile, 'array')
    if match_array.shape != image_shape:
        raise RasterDimensionError("matchFile '{}' dimensions {} do not match dem dimensions {}".format(
                                   matchFile, match_array.shape, image_shape))
    if ddm_kernel_size is None:
        ddm_kernel_size = int(math.floor(21*2/image_res))
    data_density_map = getDataDensityMap(match_array, ddm_kernel_size)
    del match_array
    if resize_factor is not None:
        data_density_map = rat.imresize(data_density_map, resize_factor)
    data_density_map[dem_nodata] = 0

    # Load ortho.
    ortho_array = rat.extractRasterData(orthoFile, 'array')
    if ortho_array.shape != image_shape:
        raise RasterDimensionError("orthoFile '{}' dimensions {} do not match dem dimensions {}".format(
                                   orthoFile, ortho_array.shape, image_shape))
    # Re-scale ortho data if WorldView correction is detected in the meta file.
    if maxDN is not None:
        print("rescaled to: 0 to {}".format(maxDN))
        ortho_array = rescaleDN(ortho_array, maxDN)
    # Convert ortho data to radiance.
    ortho_array = DG_DN2RAD(ortho_array, satID=satID, effectiveBandwith=effbw, abscalFactor=abscalfact)
    print("radiance value range: {:.2f} to {:.2f}".format(np.nanmin(ortho_array), np.nanmax(ortho_array)))
    if resize_factor is not None:
        ortho_array = rat.imresize(ortho_array, resize_factor)
    ortho_array[dem_nodata] = 0

    # Initialize output.
    component_masks = {}
    component_masks_out = {}
    mask_components = [mask_version]
    if save_component_masks:
        mask_components.extend([MASKCOMP_EDGE_NAME, MASKCOMP_WATER_NAME, MASKCOMP_CLOUD_NAME])
    for mask_name in mask_components:
        component_masks_out[mask_name] = np.ones_like(dem_array, np.bool)

    # Mask edges using DEM slope.
    mask = getSlopeMask(dem_array, dx=processing_dx, dy=processing_dy, source_res=image_res)
    mask = handle_component_masks(MASKCOMP_EDGE_NAME+'_slopemask', mask, component_masks,
                                  (debug_component_masks in (DEBUG_ALL, DEBUG_MASKS)))
    mask = getEdgeMask(~mask)
    edge_mask = mask
    edge_mask = mask_envelope_nodata(edge_mask, dem_nodata)
    mask = edge_mask
    mask = handle_component_masks(MASKCOMP_EDGE_NAME, mask, component_masks,
                                  save_component_masks or (debug_component_masks in (DEBUG_ALL, DEBUG_MASKS)))
    mask_out = mask

    # Mask water.
    mask = getWaterMask(ortho_array, data_density_map, mean_sun_elevation,
                        debug_component_masks=debug_component_masks)
    water_mask, mask_comp = mask
    water_mask = mask_envelope_nodata(water_mask, dem_nodata, edge_mask=edge_mask)
    mask = water_mask, mask_comp
    # water_mask[edge_mask] = False
    mask = handle_component_masks(MASKCOMP_WATER_NAME, mask, component_masks,
                                  save_component_masks or (debug_component_masks in (DEBUG_ALL, DEBUG_MASKS)))
    mask_out = (mask_out | mask)

    # Filter clouds.
    mask = getCloudMask(dem_array, ortho_array, data_density_map,
                        edge_mask=edge_mask, water_mask=water_mask,
                        debug_component_masks=debug_component_masks)
    cloud_mask, mask_comp = mask
    cloud_mask = mask_envelope_nodata(cloud_mask, dem_nodata, edge_mask=edge_mask)
    mask = cloud_mask, mask_comp
    mask = handle_component_masks(MASKCOMP_CLOUD_NAME, mask, component_masks,
                                  save_component_masks or (debug_component_masks in (DEBUG_ALL, DEBUG_MASKS)))
    mask_out = (mask_out | mask)

    component_masks[mask_version] = mask_out

    # for mask_name in component_masks:
    #     mask = component_masks[mask_name]
    #     mask = rat.imresize(mask, image_shape, 'nearest')
    #     component_masks[mask_name] = mask

    return component_masks


def mask_envelope_nodata(mask, nodata, edge_mask=None):
    # TODO: Write docstring.

    mask_addition = (rat.imdilate(mask, sk_morphology.diamond(1)) & nodata)

    if edge_mask is not None:
        mask_addition[edge_mask] = False
        mask_addition = sp_ndimage.morphology.binary_fill_holes(mask_addition)
        mask_addition[~nodata] = False

    return (mask | mask_addition)


def handle_component_masks(mask_name, mask_tuple, component_masks, save_component_masks):
    # TODO: Write docstring.

    submask_components = {}

    if type(mask_tuple) == tuple:
        mask, submask_components = mask_tuple
        submask_components = {'{}_{}'.format(mask_name, submask_name): submask_array
                              for submask_name, submask_array in submask_components.items()}
    else:
        mask = mask_tuple

    if save_component_masks:
        component_masks[mask_name] = mask
        for submask_name in submask_components:
            component_masks[submask_name] = submask_components[submask_name]

    return mask


def mask_v2a(demFile, avg_kernel_size=5,
             min_nocloud_cluster=10, min_data_cluster=5000,
             cloud_stdev_thresh=0.75, cloud_density_thresh=1,
             iter_stdev_thresh=0.1, iter_dilate=11,
             dilate_nodata=5):
    """
    Create a single mask masking ON regions of bad data in a scene,
    utilizing information from the DEM and matchtag corresponding to
    a single scene.

    Crops out bad data on the edges of the scene where high average
    slope values are found in the DEM. Cuts out areas of probable
    cloud cover using a local standard deviation filter on the DEM
    along with a match point data density map derived from the matchtag.

    Parameters
    ----------
    demFile : str (file path)
        File path of the DEM raster image.
    avg_kernel_size : positive int
        Side length of the neighborhood to use for both elevation
        standard deviation filter and calculating data density map.
        If None, is set to `int(math.floor(21*2/image_res))` where
        `image_res` is the resolution of the raster images.
    min_nocloud_cluster : positive int
        Minimum number of contiguous cloud-classified pixels
        in a kept cluster during masking.
    min_data_cluster : positive int
        Minimum number of contiguous good data pixels in a kept good
        data cluster of the returned mask.
    cloud_stdev_thresh : positive float
        Minimum local elevation standard deviation for a region to
        be classified as clouds.
    cloud_density_thresh : 0 < float <= 1
        Maximum match point density for a region to be classified
        as clouds.
    iter_stdev_thresh : positive float
        Minimum local elevation standard deviation for a region to be
        considered bad data during iterative expansion of cloud mask.
    iter_dilate : positive int
        Side length of square neighborhood (of ones) used as structure
        for dilation in iterative expansion of cloud mask.
    dilate_nodata : positive int
        Side length of square neighborhood (of ones) used as structure
        for dilation of nodata pixels in the DEM, to be set OFF in the
        returned mask.

    Returns
    -------
    mask : ndarray of bool, 2D
        Scene mask masking ON regions of bad data.

    Notes
    -----
    This method is currently designed for masking images with 8-meter
    square pixel resolution.

    Source file: remaMask2a.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 2/28/18

    Source docstring:
    % remaMask2a mask dem using point density and slopes deviations
    %
    % m = remaMask2a(demFile) masks the dem file by first applying an edgemask
    % and then using an iterative bad pixel search starting from points of low
    % point density and high standard deviation in slope.

    """
    matchFile = selectBestMatchtag(demFile)

    # Read DEM data and extract information for slope/cloud masking.

    dem_array, image_gt = rat.extractRasterData(demFile, 'array', 'geo_trans')
    image_dx = image_gt[1]
    image_dy = image_gt[5]
    image_res = abs(image_dx)

    # Initialize output.
    mask_out = np.zeros_like(dem_array, np.bool)

    if avg_kernel_size is None:
        avg_kernel_size = int(math.floor(21*2/image_res))

    dem_array[dem_array == -9999] = np.nan
    dy, dx = np.gradient(dem_array, image_dy, image_dx)

    # Mask edges using dem slope.
    mask = getEdgeMask(~getSlopeMask(dem_array,
                                     grad_dx=dx, grad_dy=dy,
                                     avg_kernel_size=avg_kernel_size))
    # No data check
    dem_array[mask] = np.nan
    if not np.any(mask):
        return mask_out
    del mask

    # Iterative expanding matchtag density / slope mask

    dem_nodata = np.isnan(dem_array)
    dx[dem_nodata] = np.nan
    dy[dem_nodata] = np.nan

    avg_kernel = np.ones((avg_kernel_size, avg_kernel_size), dtype=np.float32)

    dk_list = [dx, dy]
    dk_nodata_list = []
    stdev_dk_list = []
    for dk in dk_list:
        dk_nodata = np.isnan(dk)
        dk[dk_nodata] = 0
        mean_dk = rat.moving_average(dk, avg_kernel)
        stdev_dk = rat.moving_average(np.square(dk), avg_kernel) - np.square(mean_dk)
        stdev_dk[stdev_dk < 0] = 0
        stdev_dk = np.sqrt(stdev_dk)
        dk_nodata_list.append(dk_nodata)
        stdev_dk_list.append(stdev_dk)
    del dk_list, dx, dy, dk, dk_nodata, mean_dk, stdev_dk

    stdev_elev_array = np.sqrt(
        np.square(stdev_dk_list[0]) + np.square(stdev_dk_list[1])
    )
    stdev_elev_nodata = rat.imdilate(dk_nodata_list[0] | dk_nodata_list[1],
                                     avg_kernel.astype(np.uint8))
    stdev_elev_array[stdev_elev_nodata] = np.nan
    del stdev_dk_list, dk_nodata_list

    # Read matchtag and make data density map.
    match_array = rat.extractRasterData(matchFile, 'array')
    data_density_map = getDataDensityMap(match_array, avg_kernel_size)
    data_density_map[dem_nodata] = np.nan

    # Locate probable cloud pixels.
    mask = (  (stdev_elev_array > cloud_stdev_thresh)
            & (data_density_map < cloud_density_thresh))

    # Remove small data clusters.
    mask = rat.bwareaopen(mask, min_nocloud_cluster, in_place=True)

    # Initialize masked pixel counters.
    N0 = np.count_nonzero(mask)
    N1 = np.inf

    # Background mask
    mask_bkg = dem_nodata | stdev_elev_nodata | (stdev_elev_array < iter_stdev_thresh)

    # Expand mask to surrounding bad pixels,
    # stop when mask stops growing.
    dilate_structure = np.ones((iter_dilate, iter_dilate), dtype=np.uint8)
    while N0 != N1:
        N0 = N1  # Set new to old.
        mask = rat.imdilate(mask, dilate_structure)  # Dilate the mask.
        mask[mask_bkg] = False  # Unmask low standard deviation pixels.
        N1 = np.count_nonzero(mask)  # Count number of new masked pixels.

    # Remove small data gaps.
    mask = ~rat.bwareaopen(~mask, min_data_cluster, in_place=True)

    # Remove border effect.
    mask = (mask | rat.imdilate(dem_nodata, dilate_nodata))

    # remove small data gaps.
    mask = ~rat.bwareaopen(~mask, min_data_cluster, in_place=True)

    return mask


def mask8m(demFile, avg_kernel_size=21,
           data_density_thresh=0.9,
           min_data_cluster=1000, min_data_gap=1000,
           min_data_cluster_final=500):
    # TODO: Complete docstring.
    """

    Parameters
    ----------
    demFile :
    avg_kernel_size :
    data_density_thresh :
    min_data_cluster :
    min_data_gap :
    min_data_cluster_final :

    Returns
    -------

    Notes
    -----
    Source docstring:
    % MASK masking algorithm for 8m resolution data
    %
    % m = mask(demFile)
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
    %
    % REQUIRED FUNCTIONS: readGeotiff, DataDensityMap,  edgeSlopeMask,
    % DataDensityMask
    %
    % Ian Howat, ihowat@gmail.com
    % 25-Jul-2017 12:49:25

    """
    matchFile = selectBestMatchtag(demFile)

    # Read raster data.
    dem_array, x, y = rat.extractRasterData(demFile, 'z', 'x', 'y')
    dem_nodata = (dem_array == -9999)
    dem_array[dem_nodata] = np.nan
    match_array = rat.extractRasterData(matchFile, 'array')

    # Raster size consistency checks
    if dem_array.shape != match_array.shape:
        raise RasterDimensionError("matchFile '{}' dimensions {} do not match dem dimensions {}".format(
                                   matchFile, match_array.shape, dem_array.shape))

    # Initialize output.
    mask_out = np.zeros_like(dem_array, np.bool)

    # Get data density map
    data_density_map = getDataDensityMap(match_array, avg_kernel_size)
    data_density_map[dem_nodata] = np.nan
    del match_array

    # Edge crop
    mask = ~getEdgeMask(~getSlopeMask(dem_array, X=x, Y=y, avg_kernel_size=avg_kernel_size))
    mask[dem_nodata] = False
    del dem_array

    # Data existence check
    if not np.any(mask):
        return mask_out

    # Data density filter
    mask = clean_mask(getHighDataDensityMask(mask, avg_kernel_size, data_density_thresh),
                      remove_pix=min_data_cluster, fill_pix=min_data_gap)

    # Data existence check
    if not np.any(mask):
        return mask_out

    mask = rat.bwareaopen(mask, min_data_cluster_final, in_place=True)

    return ~mask


def getDataDensityMap(array, kernel_size=11,
                      label=0, label_type='nodata',
                      conv_depth='single'):
    """
    Calculate the density of data points in an array.

    Parameters
    ----------
    array : ndarray
        Array for which the density of "data" pixels
        is to be calculated.
    kernel_size : positive int
        Side length of the neighborhood to use for
        calculating data density fraction.
    label : bool/int/float
        Value of nodes in `array` that are classified
        as "data" (if label_type='data')
        or non-"data" (if label_type='nodata').
    label_type : str; 'data' or 'nodata'
        Whether `label` is a classification for "data"
        or non-"data" nodes.
    conv_depth : str; 'single' or 'double'
        The floating data type of the returned array.

    Returns
    -------
    data_density_map : ndarray of float (bit depth set
                       by `conv_depth`), same shape as `array`
        Data density map with each node of the array
        carrying the fraction of "data" nodes in the
        surrounding `kernel_size` x `kernel_size`
        neighborhood of `array`.

    """
    data_array = rat.getDataArray(array, label, label_type)
    return rat.moving_average(data_array, kernel_size,
                              shape='same', conv_depth=conv_depth)


def getHighDataDensityMask(match_array, kernel_size=21,
                           density_thresh=0.3,
                           conv_depth='single'):
    """
    Return an array masking OFF areas of poor data
    coverage in a match point array.

    Parameters
    ----------
    match_array : ndarray, 2D
        Binary array to mask containing locations of
        good data values.
    kernel_size : positive int
        Side length of the neighborhood to use for
        calculating data density fraction.
    density_thresh : positive float
        Minimum data density fraction for a pixel to
        be set to 1 in the mask.
    conv_depth : str; 'single' or 'double'
        The floating data type of the data density map
        array that will be compared against `density_thresh`.

    Returns
    -------
    mask : ndarray of bool, same shape as data_array
        The data density mask of the input matchtag array,
        masking OFF areas of good data coverage.

    Notes
    -----
    *Source file: DataDensityMask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 12/05/17

    *Functionality has been modified in translation:
        - Removal of small data clusters and small data gaps.
        To replicate functionality of DataDensityMask.m,
        pass the result of this function to clean_mask().

    """
    return getDataDensityMap(match_array, kernel_size, conv_depth=conv_depth) >= density_thresh


def getLowEntropyMask(orthoFile, entropy_thresh=0.2,
                      processing_res=8, kernel_size=None, min_data_cluster=1000):
    """
    Return an array masking ON areas of low entropy, such as water,
    in an orthorectified panchromatic spectral image.

    Parameters
    ----------
    orthoFile : str (file path)
        File path of the image to process.
    entropy_thresh : 0 < float < -log2(`kernel_size`^(-2))
        Minimum entropy threshold.
        0.2 seems to be good for water.
    processing_res : positive float (meters)
        Downsample ortho image to this resolution for
        processing for speed and smooth.
    kernel_size : None or positive int
        Side length of square neighborhood (of ones)
        to be used as kernel for entropy filter, at
        `processing_res` resolution.
        If None, is set automatically by `processing_res`.
    min_data_cluster : positive int
        Minimum number of contiguous data pixels in a
        kept data cluster, at `processing_res` resolution.

    Returns
    -------
    mask : ndarray of bool, same shape as image array
        Entropy mask masking ON areas of low entropy in the image.

    Notes
    -----
    Source file: entropyMask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 12/05/17

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

    metaFile = orthoFile.replace('ortho.tif', 'meta.txt')
    wvcFlag = False
    if not os.path.isfile(metaFile):
        warn("no meta file found, assuming no wv_correct applied")
    else:
        meta = readSceneMeta(metaFile)
        try:
            wvcFlag = meta['image_1_wv_correct']
        except KeyError:
            raise MetadataError("Scene metadata file ({}) is missing key information; cannot proceed".format(metaFile))
        if wvcFlag == 1:
            print("wv_correct applied")
        else:
            print("wv_correct not applied")

    # Read ortho.
    ortho_array, image_shape, image_res = rat.extractRasterData(orthoFile, 'array', 'shape', 'res')

    background_mask = (ortho_array == 0)  # image background mask

    # Resize ortho to pres.
    if image_res != processing_res:
        ortho_array = rat.imresize(ortho_array, image_res/processing_res)

    # Subtraction image
    ortho_subtraction = (  sp_ndimage.maximum_filter1d(ortho_array, kernel_size, axis=0)
                         - sp_ndimage.minimum_filter1d(ortho_array, kernel_size, axis=0))
    if not wvcFlag:
        ortho_subtraction = rat.astype_round_and_crop(ortho_subtraction, np.uint8, allow_modify_array=True)

    # Entropy image
    entropy_array = rat.entropyfilt(ortho_subtraction, kernel_size)
    mask = (entropy_array < entropy_thresh)
    del entropy_array

    mask = clean_mask(mask, remove_pix=min_data_cluster, fill_pix=min_data_cluster, in_place=True)

    # Resize ortho to 8m.
    if image_res != processing_res:
        mask = rat.imresize(mask, image_shape, 'nearest')

    mask[background_mask] = False

    return mask


def getSlopeMask(dem_array,
                 res=None,
                 dx=None, dy=None,
                 X=None, Y=None,
                 grad_dx=None, grad_dy=None,
                 source_res=None, avg_kernel_size=None,
                 dilate_bad=13):
    """
    Return an array masking ON artifacts with
    high slope values in a DEM array.

    Parameters
    ----------
    dem_array : ndarray, 2D
        Array containing floating point DEM data.
    res : None or positive float (meters)
        Square resolution of pixels in `dem_array`.
    dx : None or positive int
        Horizontal length of pixels in `dem_array`.
    dy : None or positive int
        Vertical length of pixels in `dem_array`.
    X : None or (ndarray, 1D)
        x-axis coordinate vector for `dem_array`.
    Y : None or (darray, 1D)
        y-axis coordinate vector for `dem_array`.
    grad_dx : None or (ndarray, 2D, shape like `dem_array`)
        x-axis gradient of `dem_array`.
    grad_dy : None or (ndarray, 2D, shape like `dem_array`)
        y-axis gradient of `dem_array`.
    source_res : positive float (meters)
        Square resolution of pixels in the source image.
    avg_kernel_size : None or positive int
        Side length of square neighborhood (of ones)
        to be used as kernel for calculating mean slope.
        If None, is set automatically by `source_res`.
    dilate_bad : None or positive int
        Side length of square neighborhood (of ones)
        to be used as kernel for dilating masked pixels.

    Returns
    -------
    mask : ndarray of bool, same shape as dem_array
        The slope mask masking ON artifacts with
        high slope values in `dem_array`.

    Notes
    -----
    Provide one of (x and y coordinate vectors x_dem and y_dem),
    input image resolution input_res,
    (pre-calculated x and y gradient arrays dx and dy).
    Note that y_dem and x_dem must both have the same uniform
    coordinate spacing AS WELL AS being both in increasing or
    both in decreasing order for the results of
    np.gradient(dem_array, y_dem, x_dem) to be equal to the results
    of np.gradient(dem_array, input_res), with input_res being a
    positive number for increasing order or a negative number for
    decreasing order.

    The returned mask sets to 1 all pixels for which the
    surrounding [kernel_size x kernel_size] neighborhood has an
    average slope greater than 1, then erode it by a kernel of ones
    with side length dilate_bad.

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
        To replicate functionality of edgeSlopeMask.m, pass the result of this function to getEdgeMask().

    """
    if not isValidArggroups((res, (dx, dy), (X, Y), (grad_dx, grad_dy))):
        raise InvalidArgumentError(
            "One type of pixel spacing input ([full regular `res`], [regular x `dx`, regular y `dy`]  "
            "[x and y coordinate arrays `X` and `Y`], or [x and y gradient 2D arrays `grad_dx` and `grad_dy`]) "
            "must be provided"
        )
    if avg_kernel_size is None:
        # avg_kernel_size = int(math.floor(21*2/source_res))
        avg_kernel_size = min(21, int(math.floor(21*2/source_res)))

    # Get elevation grade at each pixel.
    if grad_dx is None:
        if res is not None:
            grad_dy, grad_dx = np.gradient(dem_array, res)
        elif dx is not None:
            grad_dy, grad_dx = np.gradient(dem_array, dy, dx)
        elif X is not None:
            grad_dy, grad_dx = np.gradient(dem_array, Y, X)
    grade = np.sqrt(np.square(grad_dx) + np.square(grad_dy))

    # Mean grade over n-pixel kernel
    mean_slope_array = rat.moving_average(grade, avg_kernel_size, conv_depth='single')

    # Mask mean slopes greater than 1.
    mask = (mean_slope_array > 1)
    if dilate_bad is not None:
        mask = rat.imdilate(mask, dilate_bad)
    mask = (mask | np.isnan(mean_slope_array))

    return mask


def getWaterMask(ortho_array, data_density_map,
                 meanSunElevation, sunElevation_split=30,
                 ortho_thresh_lowsunelev=5, ortho_thresh_highsunelev=20,
                 entropy_thresh=0.2, data_density_thresh=0.98, min_data_cluster=500,
                 ent_kernel_size=5, dilate=7,
                 debug_component_masks=DEBUG_NONE):
    """
    Classify areas of water coverage in a panchromatic
    satellite image, masking ON water.

    The water mask is derived from the combination of an
    entropy filter applied to the smoothed panchromatic image
    and a radiance filter applied to the unsmoothed image,
    along with the provided data density map.

    Classified as water are regions all with low entropy,
    low radiance, and low match point density.

    Parameters
    ----------
    ortho_array : ndarray, 2D, same shape as `data_density_map`
        The orthorectified panchromatic image to be masked.
    data_density_map : ndarray of float, 2D, same shape as
                       `ortho_array`
        Data density map corresponding to the input image,
        where the value of each node describes the fraction
        of surrounding pixels in the input image that were
        match points in the photogrammetric process.
    meanSunElevation : 0 <= float <= 90
        Mean sun elevation angle where the image was taken,
        as parsed from the scene metadata file corresponding
        to the input image.
        Value decides the threshold for the radiance filter.
    sunElevation_split : 0 <= float <= 90
        A mean sun elevation angle (less / greater) than
        this value sets the radiance filter threshold to
        (`ortho_thresh_lowsunelev` / `ortho_thresh_highsunelev`).
    ortho_thresh_lowsunelev : positive int, units of `ortho_array`
        Radiance filter threshold used when mean sun elevation
        angle is classified as "low" by `sunElevation_split`.
    ortho_thresh_highsunelev : positive int, units of `ortho_array`
        Radiance filter threshold used when mean sun elevation
        angle is classified as "high" by `sunElevation_split`.
    entropy_thresh : 0 < float < -log2(`ent_kernel_size`^(-2))
        Entropy filter threshold.
    data_density_thresh : 0 < float <= 1
        Match point data density threshold.
    min_data_cluster : positive int
        Minimum number of contiguous data pixels in a kept
        data cluster during masking.
    ent_kernel_size : positive int
        Side length of square neighborhood (of ones)
        used as kernel for entropy filtering.
    dilate : positive int
        Side length of square neighborhood (of ones)
        used as structure for dilation of low-pass
        entropy filter result.
        A larger value results in areas classified as
        water having a thicker border.
    debug_component_masks : int
        If DEBUG_NONE, has no effect.
        If DEBUG_ALL, perform all of the following.
        If DEBUG_MASKS, return additional
          entropy/radiance mask components.
        If DEBUG_ITHRESH, save data for interactive testing
          of threshold values for component mask generation.

    Returns
    -------
    mask, component_masks : (ndarray of bool, same shape as `ortho_array`; dict of former)
        The water mask masking ON areas classified as water
        in the input panchromatic image.

    Notes
    -----
    Source file: waterMask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 2/28/18

    """
    component_masks = {}

    ortho_data = (ortho_array != 0)
    ortho_thresh = ortho_thresh_lowsunelev if meanSunElevation < sunElevation_split else ortho_thresh_highsunelev

    # Subtraction image
    ortho_subtraction = (  sp_ndimage.maximum_filter1d(ortho_array, ent_kernel_size, axis=0)
                         - sp_ndimage.minimum_filter1d(ortho_array, ent_kernel_size, axis=0))

    # Entropy image
    entropy_array = rat.entropyfilt(
        rat.astype_round_and_crop(ortho_subtraction, np.uint8, allow_modify_array=True),
        ent_kernel_size)

    # Set edge-effected values to zero.
    entropy_array[~ortho_data] = 0

    exec(ITHRESH_START)

    # Mask data with entropy less than threshold.
    mask_entropy = (ortho_data & (entropy_array < entropy_thresh))

    # Remove isolated clusters of masked pixels.
    mask_entropy = rat.bwareaopen(mask_entropy, min_data_cluster, in_place=True)

    # Dilate masked pixels.
    mask_entropy = rat.imdilate(mask_entropy, dilate)

    # Mask data with low radiance and low matchpoint density.
    mask_radiance_ortho = (ortho_data & (ortho_array < ortho_thresh))
    mask_radiance_datadensity = (ortho_data & (data_density_map < data_density_thresh))
    mask_radiance = (ortho_data & mask_radiance_ortho & mask_radiance_datadensity)

    # Remove isolated clusters of masked pixels.
    mask_radiance = rat.bwareaopen(mask_radiance, min_data_cluster, in_place=True)

    # Assemble water mask.
    mask = (ortho_data & (mask_entropy | mask_radiance))

    # Remove isolated clusters of data.
    mask_pp = ~clean_mask(~mask, remove_pix=min_data_cluster, fill_pix=min_data_cluster, in_place=True)

    # ITHRESH_END

    if debug_component_masks in (DEBUG_ALL, DEBUG_MASKS):
        component_masks['entropy'] = mask_entropy
        component_masks['radiance'] = mask_radiance
        component_masks['radiance_ortho'] = mask_radiance_ortho
        component_masks['radiance_datadensity'] = mask_radiance_datadensity

    return mask_pp, component_masks


def getCloudMask(dem_array, ortho_array, data_density_map,
                 water_mask=None, edge_mask=None,
                 ortho_thresh=70,
                 data_density_thresh_hirad=0.9, data_density_thresh_lorad=0.6,
                 min_nocloud_cluster=10000, min_cloud_cluster=1000,
                 stdev_kernel_size=21,
                 erode_border=31, dilate_border=61,
                 dilate_cloud=21,
                 debug_component_masks=DEBUG_NONE):
    """
    Classify areas of cloud coverage in a panchromatic
    satellite image with derived DEM, masking ON clouds.

    The cloud mask is derived from the combination of a
    radiance filter on the panchromatic image and a local
    standard deviation filter on the DEM, along with the
    provided data density map.

    Classified as clouds are regions that have one or more
    of the following attributes:
    1) High radiance and low match point density
    2) Very low match point density
    3) High local standard deviation in elevation

    Parameters
    ----------
    dem_array : ndarray, 2D, same shape as `ortho_array`
                and `data_density_map`
        The DEM of the scene.
    ortho_array : ndarray, 2D, same shape as `dem_array`
        The orthorectified panchromatic image of the scene.
    data_density_map : ndarray of float, 2D, same shape as `dem_array`
        Data density map for the scene, where the value of
        each node describes the fraction of surrounding
        pixels in the input image that were match points
        in the photogrammetric process.
    water_mask : None or ndarray, 2D, same shape as `dem_array`
        Boolean mask of water for the scene.
        If provided, cloud mask is prevent from excessive overlap
        with the water mask by subtracting the water mask from the
        cloud mask pre-dilation step.
    edge_mask : None or ndarray, 2D, same shape as `dem_array`
        Boolean mask of bad edges for the scene.
        If provided, cloud mask is clipped to the inside extent
        of the edge mask between erosion and dilation steps.
    ortho_thresh : positive int, units of `ortho_array`
        Radiance filter threshold.
    data_density_thresh_hirad : `data_density_thresh_lorad` <= float <= 1
        Maximum match point data density for high radiance
        regions to be classified as clouds.
    data_density_thresh_lorad : 0 < float <= `data_density_thresh_hirad`
        Maximum match point data density for low radiance
        to be classified as clouds.
    min_nocloud_cluster : positive int
        Minimum number of contiguous cloud-classified pixels
        in a kept cluster during masking.
    min_cloud_cluster : positive int
        Minimum number of contiguous non-cloud-classified pixels
        in a kept cluster during masking.
    stdev_kernel_size : positive int
        Side length of square neighborhood (of ones)
        used as kernel for elevation standard deviation filter.
    erode_border : positive int
        Side length of square neighborhood (of ones) used as
        structure for erosion in smoothing the edges of
        cloud-classified regions.
    dilate_border : positive int
        Side length of square neighborhood (of ones) used as
        structure for dilation in smoothing the edges of
        cloud-classified regions.
    dilate_cloud : positive int
        Side length of square neighborhood (of ones)
        used as structure for dilation of cloud-classified
        regions before returning mask.
    debug_component_masks : int
        If DEBUG_NONE, has no effect.
        If DEBUG_ALL, perform all of the following.
        If DEBUG_MASKS, return additional
          radiance/datadensity/stdev mask components.
        If DEBUG_ITHRESH, save data for interactive testing
          of threshold values for component mask generation.

    Returns
    -------
    mask, component_masks : (ndarray of bool, same shape as input arrays; dict of former)
        The cloud mask masking ON areas classified as clouds
        in the input scene information.

    Notes
    -----
    Source file: cloudMask.m
    Source author: Ian Howat, ihowa@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 2/28/18

    Source docstring:
    % cloudMask mask bad surfaces on DEM based on slope and radiance

    % M = cloudMask(z,or)masks bad edge values in the DEM with coordinate
    % vectors x any y.
    %
    % Ian Howat, ihowat@gmail.com
    % 24-Jul-2017 15:39:07

    """
    component_masks = {}

    dem_data = ~np.isnan(dem_array)

    # Make sure sufficient non NaN pixels exist, otherwise cut to the chase.
    if np.count_nonzero(dem_data) < 2*min_cloud_cluster:
        mask = np.ones(dem_array.shape, dtype=np.bool)
        return mask, component_masks

    # Calculate standard deviation of elevation.
    mean_elev_array = rat.moving_average(dem_array, stdev_kernel_size,
                                         shape='same', conv_depth='single')
    stdev_elev_array = (rat.moving_average(np.square(dem_array), stdev_kernel_size,
                                           shape='same', conv_depth='single')
                        - np.square(mean_elev_array))
    stdev_elev_array[stdev_elev_array < 0] = 0
    stdev_elev_array = np.sqrt(stdev_elev_array)

    # Calculate elevation percentile difference.
    percentile_diff = (  np.nanpercentile(dem_array, 80)
                       - np.nanpercentile(dem_array, 20))

    # Set standard deviation difference based on percentile difference.
    stdev_thresh = None
    if percentile_diff <= 40:
        stdev_thresh = 10.5
    elif percentile_diff <= 50:
        stdev_thresh = 15
    elif percentile_diff <= 75:
        stdev_thresh = 19
    elif percentile_diff <= 100:
        stdev_thresh = 27
    elif percentile_diff > 100:
        stdev_thresh = 50

    print("20/80 percentile elevation difference: {:.1f}, sigma-z threshold: {:.1f}".format(
        percentile_diff, stdev_thresh))

    exec(ITHRESH_START)

    # Apply mask conditions.
    mask_radiance_ortho = (dem_data & (ortho_array > ortho_thresh))
    mask_radiance_datadensity = (dem_data & (data_density_map < data_density_thresh_hirad))
    mask_radiance = (dem_data & mask_radiance_ortho & mask_radiance_datadensity)
    mask_datadensity = (dem_data & (data_density_map < data_density_thresh_lorad))
    mask_stdev = (dem_data & (stdev_elev_array > stdev_thresh))

    # Assemble cloud mask.
    mask = (dem_data & (mask_radiance | mask_datadensity | mask_stdev))

    # Subtract water mask, if provided.
    if water_mask is not None:
        mask[water_mask] = False

    # The following call is commented out to avoid the cloud mask growing
    # to have an egrigous extent when the current mask makes a large ring
    # around the scene.
    # # Fill holes in masked clusters.
    # mask_pp = sp_ndimage.morphology.binary_fill_holes(mask)
    mask_pp = mask

    # Remove small masked clusters.
    mask_pp = rat.bwareaopen(mask_pp, min_cloud_cluster, in_place=True)

    # Remove thin borders caused by cliffs/ridges.
    mask_smooth = rat.imerode(mask_pp, erode_border)
    mask_smooth = rat.imdilate(mask_smooth, dilate_border)

    mask_pp = (mask_pp & mask_smooth)

    # Dilate nodata.
    if edge_mask is not None:
        mask_pp[edge_mask] = False
    mask_pp = rat.imdilate(mask_pp, dilate_cloud)

    # Remove small clusters of unfiltered data.
    mask_pp = ~rat.bwareaopen(~mask_pp, min_nocloud_cluster, in_place=True)

    # ITHRESH_END

    if debug_component_masks in (DEBUG_ALL, DEBUG_MASKS):
        component_masks['radiance'] = mask_radiance
        component_masks['radiance_ortho'] = mask_radiance_ortho
        component_masks['radiance_datadensity'] = mask_radiance_datadensity
        component_masks['datadensity'] = mask_datadensity
        component_masks['stdev'] = mask_stdev
        component_masks['smooth'] = mask_smooth

    return mask_pp, component_masks


def getEdgeMask(match_array, hull_concavity=0.5, crop=None,
                res=None, min_data_cluster=1000):
    """
    Return an array masking ON bad edges on a mass
    of good data (see Notes) in a matchtag array.

    Parameters
    ----------
    match_array : ndarray, 2D
        Binary array to mask containing locations of
        good data values.
    hull_concavity : 0 <= float <= 1
        Boundary curvature factor argument be passed
        to concave_hull_image().
        (0 = convex hull, 1 = point boundary)
    crop : None or positive int
        Erode the mask by a square neighborhood (ones)
        of this side length before returning.
    res : positive int
        Image resolution corresponding to data_array,
        for setting parameter default values.
    min_data_cluster : None or positive int
        Minimum number of contiguous data pixels in a
        kept data cluster.
        If None, is set automatically by `res`.

    Returns
    -------
    mask : ndarray of bool, same shape as data_array
        The edge mask masking ON bad data hull edges
        in input match_array.

    See also
    --------
    concave_hull_image

    Notes
    -----
    The input array is presumed to contain a large "mass"
    of data (nonzero) values near its center, surrounded by
    a border of nodata (zero) values. The mass of data may
    or may not have holes (clusters of zeros).
    The returned mask discards all area outside of the
    (convex) hull of the region containing both the data mass
    and all data clusters of more pixels than `min_data_cluster`.
    Either `res` or `min_data_cluster` must be provided.

    *Source file: edgeMask.m
    Source author: Ian Howat, ihowat@gmail.com, Ohio State University
    Source repo: setsm_postprocessing, branch "3.0" (GitHub)
    Translation date: 10/17/17

    Source docstring:
    % edgeMask returns mask for bad edges using the matchtag field
    %
    % m1 = edgeMask(m0) where m0 is the matchtag array returns a binary mask of
    % size(m0) designed to filter data bad edges using match point density

    *Functionality has been modified in translation:
        - Removal of data density masking.
        - Removal of entropy masking.
        To replicate functionality of edgeMask.m, do masking of
        data_array with getHighDataDensityMask() and getLowEntropyMask()
        before passing the result to this function.

    """
    if res is None and min_data_cluster is None:
        raise InvalidArgumentError("Resolution `res` argument must be provided "
                                   "to set default values of min_data_cluster")
    if not np.any(match_array):
        return match_array.astype(np.bool)
    if min_data_cluster is None:
        min_data_cluster = int(math.floor(1000*2/res))

    # Fill interior holes since we're just looking for edges here.
    mask = sp_ndimage.morphology.binary_fill_holes(match_array)
    # Get rid of isolated little clusters of data.
    mask = rat.bwareaopen(mask, min_data_cluster, in_place=True)

    if not np.any(mask):
        # No clusters exceed minimum cluster area.
        print("Boundary filter removed all data")
        return mask

    mask = rat.concave_hull_image(mask, hull_concavity)

    if crop is not None:
        mask = rat.imerode(mask, crop)

    return ~mask


def clean_mask(mask, remove_pix=1000, fill_pix=10000, in_place=False):
    """
    Remove small clusters of data (ones) and fill
    small holes of nodata (zeros) in a binary mask array.

    Parameters
    ----------
    mask : ndarray, 2D
        Binary array to mask.
    remove_pix : positive int
        Minimum number of contiguous one pixels in a
        kept data cluster.
    fill_pix : positive int
        Maximum number of contiguous zero pixels in a
        filled data void.
    in_place : bool
        If True, clean the mask in the input array itself.
        Otherwise, make a copy.

    Returns
    -------
    mask_out : ndarray of bool, same shape as data_array
        The cluster and hole mask of the input mask array.

    """
    if not np.any(mask):
        return mask.astype(np.bool)

    # Remove small data clusters.
    cleaned_mask = rat.bwareaopen(mask, remove_pix, in_place=in_place)
    # Fill small data voids.
    return ~rat.bwareaopen(~cleaned_mask, fill_pix, in_place=True)


def readSceneMeta(metaFile):
    # TODO: Write my own docstring.
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

    # Detect scenes produced with flawed SETSM version 3.4.2 and quit building strip.
    if meta['setsm_version'] == '3.4.2':
        raise MetadataError("SETSM version is 3.4.2; strip must be re-run to newest SETSM version")

    # Get satID and check for cross track naming convention.
    try:
        for image_num in (1, 2):
            image_path = meta['image_{}'.format(image_num)]
            meta['image_{}_sensor'.format(image_num)] = os.path.basename(image_path)[0:4].upper()
    except KeyError as e:
        traceback.print_exc()
        raise MetadataError("Could not parse sensor from Image 1 tag: {}".format(metaFile))

    return meta


def rescaleDN(ortho_array, dnmax):
    # TODO: Write my own docstring.
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
    # We use a fixed scale because this is what all data
    # is scaled to after application of wv_correct
    # regardless of actual min or max.
    ormin = 0
    ormax = 32767

    # Set the new minimum and maximum.
    # dnmin is zero because nodata is apparently used
    # in the scaling.
    dnmin = 0
    dnmax = float(dnmax)

    # Rescale back to original dn.
    return dnmin + (dnmax-dnmin)*(ortho_array.astype(np.float32) - ormin)/(ormax-ormin)


def DG_DN2RAD(DN,
              xmlFile=None,
              satID=None, effectiveBandwith=None, abscalFactor=None):
    # TODO: Write my own docstring.
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
            raise InvalidArgumentError("`xmlFile` argument must be given to automatically set xml params")
        fillMissingXmlParams(xmlFile, xml_params)
        satID, effectiveBandwith, abscalFactor = [p[0] for p in xml_params]
        if satID == 'QB2':
            satID = 'QB02'
        elif satID == 'IKO':
            satID = 'IK01'
        effectiveBandwith = float(effectiveBandwith)
        abscalFactor = float(abscalFactor)

    # Values from:
    # https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/209/DGConstellationAbsRadCalAdjustmentFactors_2015v2.pdf
    sensor = ('WV03',   'WV02', 'GE01', 'QB02',  'IK01',  'WV01')
    gain   = (0.923,    0.96,   0.978,  0.876,  0.907,  1.016)
    offset = [-1.7,     -2.957, -1.948, -2.157, -4.461, -3.932]

    sensor_index = sensor.index(satID)
    gain = gain[sensor_index]
    offset = offset[sensor_index]

    DN = DN.astype(np.float32)
    DN[DN == 0] = np.nan

    # Calculate radiance.
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


def ithresh_save(block_num, vars_dict, funcname=None):
    # TODO: Write docstring.
    global ITHRESH_QUICK_SAVED

    from inspect import stack

    caller_funcName = funcname if funcname is not None else stack()[2][3]
    caller_funcDef = 'def {}('.format(caller_funcName)
    # this_funcName = 'ithresh_save'
    this_funcName = stack()[0][3]
    this_funcCall = '{}('.format(this_funcName)

    this_file_fp = open(__file__.replace('.pyc', '.py'), 'r')
    this_file_txt = this_file_fp.read()
    this_file_fp.close()
    this_file_fp = StringIO(this_file_txt)
    line = this_file_fp.readline()

    # Locate caller function definition in this file.
    found = False
    while not found and line != '':
        if line.startswith(caller_funcDef):
            found = True
        line = this_file_fp.readline()
    if not found:
        raise DebugError("Could not find function definition matching '{}'".format(caller_funcDef))

    # Locate start of ithresh region within the caller function.
    found_block_num = 0
    while found_block_num != block_num and line != '':
        if line.lstrip().startswith(ITHRESH_START_TAG):
            found_block_num += 1
        line = this_file_fp.readline()
    if not found:
        raise DebugError("{} block_num={} missing start tag '{}'".format(
                         caller_funcName, block_num, ITHRESH_START_TAG))

    # Capture code in this interactive threshold region.
    ithresh_code_raw = ''
    ithresh_code_exec = ''
    indent = None
    done = False
    while not done and line != '':
        ithresh_code_raw += line

        if line.strip() != '' and not line.lstrip().startswith('#'):
            if indent is None:
                # The first line of code after `ithresh_save` call
                # sets the indentation for the whole code region.
                indent = line[:line.find(line.lstrip()[0])]
            ithresh_code_exec += line.replace(indent, '', 1)

        if line.lstrip().startswith(ITHRESH_END_TAG):
            done = True
        elif line.lstrip().startswith(this_funcName):
            break
        elif line.startswith('def '):
            break

        line = this_file_fp.readline()
    if not done:
        raise DebugError("{} block_num={} missing end tag '{}'".format(
                         caller_funcName, block_num, ITHRESH_END_TAG))

    this_file_fp.close()

    # Save data necessary for performing interactive thresholding.
    vars_dict_save = {name: data for name, data in vars_dict.items() if name in ithresh_code_exec}
    vars_dict_save['ITHRESH_FUNCTION_NAME'] = caller_funcName
    vars_dict_save['ITHRESH_BLOCK_NUM'] = block_num
    vars_dict_save['ITHRESH_CODE_RAW'] = ithresh_code_raw
    vars_dict_save['ITHRESH_CODE_EXEC'] = ithresh_code_exec
    vars_dict_file = os.path.join(DEBUG_DIR, DEBUG_FNAME_PREFIX+"ithresh_{}_block{}.npy".format(caller_funcName, block_num))
    sys.stdout.write("Dumping vars to {} ...".format(vars_dict_file))
    np.save(vars_dict_file, vars_dict_save)
    sys.stdout.write(" done\n")


def ithresh_load(block_num, funcname=None):
    # TODO: Write docstring.

    from inspect import stack
    caller_funcName = funcname if funcname is not None else stack()[1][3]

    vars_dict_file = os.path.join(DEBUG_DIR, DEBUG_FNAME_PREFIX+"ithresh_{}_block{}.npy".format(caller_funcName, block_num))
    if not os.path.isfile(vars_dict_file):
        return {}

    print("Loading thresh vars from {}".format(vars_dict_file))
    vars_dict = np.load(vars_dict_file).item()
    vars_dict = {name: data for name, data in vars_dict.items() if 'thresh' in name}

    return vars_dict
