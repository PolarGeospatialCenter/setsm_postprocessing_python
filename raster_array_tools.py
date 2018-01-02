#!/usr/bin/env python2

# Version 3.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2017


from __future__ import division
import os
import copy
import math
import warnings
from collections import deque
from itertools import product
from subprocess import check_call
from traceback import print_exc
from warnings import warn

import gdal, ogr, osr
import numpy as np
import scipy
import shapely.geometry
import shapely.ops
from osgeo import gdal_array
from scipy import ndimage as sp_ndimage
from scipy.spatial import ConvexHull
from skimage.draw import polygon_perimeter
from skimage import morphology as sk_morphology
from skimage.filters.rank import entropy
from skimage.util import unique_rows

import test

_outline = open("outline.c", "r").read()
_outline_every1 = open("outline_every1.c", "r").read()


RASTER_PARAMS = ['ds', 'shape', 'z', 'array', 'x', 'y', 'dx', 'dy', 'res', 'geo_trans', 'corner_coords', 'proj_ref', 'spat_ref', 'geom', 'geom_sr']


warnings.simplefilter('always', UserWarning)
gdal.UseExceptions()

class RasterIOError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)

class UnsupportedDataTypeError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)

class InvalidArgumentError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)

class UnsupportedMethodError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)



#############
# Raster IO #
#############


# Legacy; Retained for quick instruction of useful GDAL raster information extraction methods.
def oneBandImageToArrayZXY_projRef(rasterFile):
    """
    Opens a single-band raster image as a NumPy 2D array [Z] and returns it along
    with [X, Y] coordinate ranges of pixels in the raster grid as NumPy 1D arrays
    and the projection definition string for the raster dataset in OpenGIS WKT format.
    """
    if not os.path.isfile(rasterFile):
        raise RasterIOError("No such rasterFile: '{}'".format(rasterFile))

    ds = gdal.Open(rasterFile, gdal.GA_ReadOnly)
    proj_ref = ds.GetProjectionRef()
    gt = ds.GetGeoTransform()

    xmin, ymax = gt[0], gt[3]
    dx, dy     = gt[1], gt[5]

    X = xmin + np.arange(ds.RasterXSize) * dx
    Y = ymax + np.arange(ds.RasterYSize) * dy

    Z = ds.GetRasterBand(1).ReadAsArray()

    return Z, X, Y, proj_ref


def openRaster(rasterFile_or_ds):
    ds = None
    if type(rasterFile_or_ds) == gdal.Dataset:
        ds = rasterFile_or_ds
    elif type(rasterFile_or_ds) == str:
        if not os.path.isfile(rasterFile_or_ds):
            raise RasterIOError("No such rasterFile: '{}'".format(rasterFile_or_ds))
        ds = gdal.Open(rasterFile_or_ds, gdal.GA_ReadOnly)
    else:
        raise InvalidArgumentError("Invalid input type for 'rasterFile_or_ds': {}".format(
                                   type(rasterFile_or_ds)))
    return ds


def getCornerCoords(gt, shape):
    top_left_x = np.full((5, 1), gt[0])
    top_left_y = np.full((5, 1), gt[3])
    top_left_mat = np.concatenate((top_left_x, top_left_y), axis=1)

    ysize, xsize = shape
    raster_XY_size_mat = np.array([
        [0, 0],
        [xsize, 0],
        [xsize, ysize],
        [0, ysize],
        [0, 0]
    ])

    gt_mat = np.array([
        [gt[1], gt[4]],
        [gt[2], gt[5]]
    ])

    return top_left_mat + np.dot(raster_XY_size_mat, gt_mat)


def coordsToWkt(corner_coords):
    return 'POLYGON (({}))'.format(
        ','.join([" ".join([str(c) for c in cc]) for cc in corner_coords])
    )


def wktToCoords(wkt):
    eval_str = 'np.array({})'.format(
        wkt.replace('POLYGON ','').replace('(','[').replace(')',']').replace(',','],[').replace(' ',',')
    )
    return eval(eval_str)


def extractRasterParams(rasterFile_or_ds, *params):
    ds = openRaster(rasterFile_or_ds)
    pset = set(params)
    invalid_pnames = pset.difference(set(RASTER_PARAMS))
    if invalid_pnames:
        raise InvalidArgumentError("Invalid parameter(s) for extraction: {}".format(invalid_pnames))

    if pset.intersection({'z', 'array'}):
        array_data = ds.GetRasterBand(1).ReadAsArray()
    if pset.intersection({'shape', 'x', 'y', 'corner_coords', 'geom', 'geom_sr'}):
        shape = (ds.RasterYSize, ds.RasterXSize) if 'array_data' not in vars() else array_data.shape
    if pset.intersection({'x', 'y', 'dx', 'dy', 'res', 'geo_trans', 'corner_coords', 'geom', 'geom_sr'}):
        geo_trans = ds.GetGeoTransform()
    if pset.intersection({'proj_ref', 'spat_ref', 'geom_sr'}):
        proj_ref = ds.GetProjectionRef()
    if pset.intersection({'corner_coords', 'geom', 'geom_sr'}):
        corner_coords = getCornerCoords(geo_trans, shape)
    if pset.intersection({'spat_ref', 'geom_sr'}):
        spat_ref = osr.SpatialReference(proj_ref) if proj_ref is not None else None
    if pset.intersection({'geom', 'geom_sr'}):
        geom = ogr.Geometry(wkt=coordsToWkt(corner_coords))

    value_list = []
    for pname in params:
        pname = pname.lower()
        value = None
        if pname == 'ds':
            value = ds
        elif pname == 'shape':
            value = shape
        elif pname in ('z', 'array'):
            value = array_data
        elif pname == 'x':
            value = geo_trans[0] + np.arange(shape[1]) * geo_trans[1]
        elif pname == 'y':
            value = geo_trans[3] + np.arange(shape[0]) * geo_trans[5]
        elif pname == 'dx':
            value = geo_trans[2]
        elif pname == 'dy':
            value = -geo_trans[5]
        elif pname == 'res':
            value = geo_trans[1] if geo_trans[1] == -geo_trans[5] else np.nan
        elif pname == 'geo_trans':
            value = geo_trans
        elif pname == 'corner_coords':
            value = corner_coords
        elif pname == 'proj_ref':
            value = proj_ref
        elif pname == 'spat_ref':
            value = spat_ref
        elif pname == 'geom':
            value = geom
        elif pname == 'geom_sr':
            value = geom.Clone() if 'geom' in params else geom
            if spat_ref is not None:
                value.AssignSpatialReference(spat_ref)
            else:
                warn("Spatial reference could not be extracted from raster dataset,"
                     " so extracted geometry has not been assigned a spatial reference.")
        value_list.append(value)

    if len(value_list) == 1:
        value_list = value_list[0]
    return value_list


# Legacy; Retained for a visual aid of equivalences between NumPy and GDAL data types.
def dtype_np2gdal(dtype_in, form_out='gdal', force_conversion=False):
    """
    Converts between input NumPy data type (dtype_in may be either
    NumPy 'dtype' object or already a string) and output GDAL data type.
    If form_out='numpy', the corresponding NumPy 'dtype' object will be
    returned instead, allowing for quick lookup by string name.
    If the third element of a dtype_dict conversion tuple is zero,
    that conversion of NumPy to GDAL data type is not recommended. However,
    the conversion may be forced with the argument force_conversion=True.
    """
    dtype_dict = {                                            # ---GDAL LIMITATIONS---
        'bool'      : (np.bool,       gdal.GDT_Byte,     0),  # GDAL no bool/logical/1-bit
        'int8'      : (np.int8,       gdal.GDT_Byte,     1),  # GDAL byte is unsigned
        'int16'     : (np.int16,      gdal.GDT_Int16,    1),
        'int32'     : (np.int32,      gdal.GDT_Int32,    1),
        'intc'      : (np.intc,       gdal.GDT_Int32,    1),  # np.intc ~= np.int32
        'int64'     : (np.int64,      gdal.GDT_Int32,    0),  # GDAL no int64
        'intp'      : (np.intp,       gdal.GDT_Int32,    0),  # intp ~= np.int64
        'uint8'     : (np.uint8,      gdal.GDT_Byte,     1),
        'uint16'    : (np.uint16,     gdal.GDT_UInt16,   1),
        'uint32'    : (np.uint32,     gdal.GDT_UInt32,   1),
        'uint64'    : (np.uint64,     gdal.GDT_UInt32,   0),  # GDAL no uint64
        'float16'   : (np.float16,    gdal.GDT_Float32,  1),  # GDAL no float16
        'float32'   : (np.float32,    gdal.GDT_Float32,  1),
        'float64'   : (np.float64,    gdal.GDT_Float64,  1),
        'complex64' : (np.complex64,  gdal.GDT_CFloat32, 1),
        'complex128': (np.complex128, gdal.GDT_CFloat64, 1),
    }
    errmsg_unsupported_dtype = "Conversion of NumPy data type '{}' to GDAL is not supported".format(dtype_in)

    try:
        dtype_tup = dtype_dict[str(dtype_in).lower()]
    except KeyError:
        raise UnsupportedDataTypeError("No such NumPy data type in lookup table: '{}'".format(dtype_in))

    if form_out.lower() == 'gdal':
        if dtype_tup[2] == 0:
            if force_conversion:
                print errmsg_unsupported_dtype
            else:
                raise UnsupportedDataTypeError(errmsg_unsupported_dtype)
        dtype_out = dtype_tup[1]
    elif form_out.lower() == 'numpy':
        dtype_out = dtype_tup[0]
    else:
        raise UnsupportedDataTypeError("The following output data type format is not supported: '{}'".format(form_out))

    return dtype_out


def saveArrayAsTiff(array, dest,
                    X=None, Y=None, proj_ref=None, geotrans_rot_tup=(0, 0),
                    like_rasterFile=None,
                    nodataVal=None, dtype_out=None):
    # FIXME: Rewrite docstring in new standard.
    """
    Saves a NumPy 2D array as a single-band raster image in GeoTiff format.
    Takes as input [X, Y] coordinate ranges of pixels in the raster grid as
    NumPy 1D arrays and geotrans_rot_tup specifying rotation coefficients
    in the output raster's geotransform tuple normally accessed via
    {GDALDataset}.GetGeoTransform()[[2, 4]] in respective index order.
    If like_rasterFile is provided, its geotransform and projection reference
    may be used for the output dataset and [X, Y, geotrans_rot_tup, proj_ref]
    should not be given.
    """
    dtype_gdal = None
    if dtype_out is not None:
        if type(dtype_out) == str:
            dtype_out = eval('np.{}'.format(dtype_out.lower()))
        dtype_gdal = gdal_array.NumericTypeCodeToGDALTypeCode(dtype_out)
        if dtype_gdal is None:
            raise InvalidArgumentError("Output array data type ({}) does not have equivalent "
                                       "GDAL data type and is not supported".format(dtype_out))

    dest_temp = dest.replace('.tif', '_temp.tif')

    dtype_in = array.dtype
    promote_dtype = None
    if dtype_in == np.bool:
        promote_dtype = np.uint8
    elif dtype_in == np.int8:
        promote_dtype = np.int16
    elif dtype_in == np.float16:
        promote_dtype = np.float32
    if promote_dtype is not None:
        warn("Input array data type ({}) does not have equivalent GDAL data type and is not "
             "supported, but will be safely promoted to {}".format(dtype_in, promote_dtype))
        array = array.astype(promote_dtype)
        dtype_in = promote_dtype

    if dtype_out is not None:
        if dtype_in != dtype_out:
            raise InvalidArgumentError("Input array data type ({}) differs from desired "
                                       "output data type ({})".format(dtype_in, dtype_out))
    else:
        dtype_gdal = gdal_array.NumericTypeCodeToGDALTypeCode(dtype_in)
        if dtype_gdal is None:
            raise InvalidArgumentError("Input array data type ({}) does not have equivalent "
                                       "GDAL data type and is not supported".format(dtype_in))

    if proj_ref is not None and type(proj_ref) == osr.SpatialReference:
        proj_ref = proj_ref.ExportToWkt()

    shape = array.shape
    geo_trans = None
    if like_rasterFile is not None:
        ds_like = gdal.Open(like_rasterFile, gdal.GA_ReadOnly)
        if shape[0] != ds_like.RasterYSize or shape[1] != ds_like.RasterXSize:
            raise InvalidArgumentError("Shape of like_rasterFile '{}' ({}, {}) does not match "
                                       "the shape of 'array' to be saved ({})".format(
                like_rasterFile, ds_like.RasterYSize, ds_like.RasterXSize, shape)
            )
        geo_trans = ds_like.GetGeoTransform()
        if proj_ref is None:
            proj_ref = ds_like.GetProjectionRef()
    else:
        if shape[0] != Y.size or shape[1] != X.size:
            raise InvalidArgumentError("Lengths of [Y, X] grid coordinates ({}, {}) do not match "
                                       "the shape of 'array' to be saved ({})".format(Y.size, X.size, shape))
        geo_trans = (X[0], X[1]-X[0], geotrans_rot_tup[0],
                     Y[0], geotrans_rot_tup[1], Y[1]-Y[0])

    # Create and write the output dataset to a temporary file.
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(dest_temp, shape[1], shape[0], 1, dtype_gdal)
    ds_out.SetGeoTransform(geo_trans)
    if proj_ref is not None:
        ds_out.SetProjection(proj_ref)
    else:
        warn("Missing projection reference for saved raster '{}'".format(dest))
    ds_out.GetRasterBand(1).WriteArray(array)
    ds_out = None  # Dereference dataset to initiate write to disk of intermediate image.

    ###################################################
    # Run gdal_translate with the following arguments #
    ###################################################
    args = [r'C:\OSGeo4W64\bin\gdal_translate', dest_temp, dest]

    if nodataVal is not None:
        args.extend(['-a_nodata', str(nodataVal)])  # Create internal nodata mask.

    args.extend(['-co', 'BIGTIFF=IF_SAFER'])        # Will create BigTIFF
                                                    # :: if the resulting file *might* exceed 4GB.
    args.extend(['-co', 'COMPRESS=LZW'])            # Do LZW compression on output image.
    args.extend(['-co', 'TILED=YES'])               # Force creation of tiled TIFF files.

    print "Running: {}".format(' '.join(args))
    check_call(args)
    os.remove(dest_temp)  # Delete the intermediate image.



######################
# Array Calculations #
######################


def rotate_array_if_kernel_has_even_sidelength(array, kernel):
    # TODO: Write docstring.
    for s in kernel.shape:
        if s % 2 == 0:
            return np.rot90(array, 2), np.rot90(kernel, 2), True
    return array, kernel, False


def fix_array_if_rotation_was_applied(array, rotation_flag):
    # TODO: Write docstring.
    return np.rot90(array, 2) if rotation_flag else array


def round_half_up(array):
    return np.floor(array + 0.5)


def astype_matlab(array, np_dtype):
    if 'float' in str(array.dtype):
        return round_half_up(array).astype(np_dtype)
    else:
        return array.astype(np_dtype)


def interp2_gdal(X, Y, Z, Xi, Yi, method, borderNaNs=True):
    """
    Performs a resampling of the input NumPy 2D array [Z],
    from initial grid coordinates [X, Y] to final grid coordinates [Xi, Yi]
    (all four ranges as NumPy 1D arrays) using the desired interpolation method.
    To best match output with MATLAB's interp2 function, extrapolation of
    row and column data outside the [X, Y] domain of the input 2D array [Z]
    is manually wiped away and set to NaN by default when borderNaNs=True.
    """
    method_dict = {
        'nearest'   : gdal.GRA_NearestNeighbour,
        'linear'    : gdal.GRA_Bilinear,
        'bilinear'  : gdal.GRA_Bilinear,
        'cubic'     : gdal.GRA_Cubic,
        'bicubic'   : gdal.GRA_Cubic,
        'spline'    : gdal.GRA_CubicSpline,
        'lanczos'   : gdal.GRA_Lanczos,
        'average'   : gdal.GRA_Average,
        'mode'      : gdal.GRA_Mode,
    }
    try:
        method_gdal = method_dict[method]
    except KeyError:
        raise UnsupportedMethodError("Cannot find GDAL equivalent of method='{}'".format(method))

    dtype_in = Z.dtype
    promote_dtype = None
    if dtype_in == np.bool:
        promote_dtype = np.uint8
    elif dtype_in == np.int8:
        promote_dtype = np.int16
    elif dtype_in == np.float16:
        promote_dtype = np.float32
    if promote_dtype is not None:
        warn("Input array data type ({}) does not have equivalent GDAL data type and is not "
             "supported, but will be safely promoted to {}".format(dtype_in, promote_dtype))
        Z = Z.astype(promote_dtype)
        dtype_in = promote_dtype

    dtype_gdal = gdal_array.NumericTypeCodeToGDALTypeCode(dtype_in)
    if dtype_gdal is None:
        raise InvalidArgumentError("Input array data type ({}) does not have equivalent "
                                   "GDAL data type and is not supported".format(dtype_in))

    mem_drv = gdal.GetDriverByName('MEM')

    ds_in = mem_drv.Create('', X.size, Y.size, 1, dtype_gdal)
    ds_in.SetGeoTransform((X[0], X[1]-X[0], 0,
                           Y[0], 0, Y[1]-Y[0]))
    ds_in.GetRasterBand(1).WriteArray(Z)

    ds_out = mem_drv.Create('', Xi.size, Yi.size, 1, dtype_gdal)
    ds_out.SetGeoTransform((Xi[0], Xi[1]-Xi[0], 0,
                            Yi[0], 0, Yi[1]-Yi[0]))

    gdal.ReprojectImage(ds_in, ds_out, '', '', method_gdal)

    Zi = ds_out.GetRasterBand(1).ReadAsArray()

    if borderNaNs:
        # Rows and columns of Zi outside the domain of Z are made NaN.
        i = 0
        while Xi[i] < X[0]:
            Zi[:, i] = np.full(Zi.shape[0], np.nan)
            i += 1
        i = -1
        while Xi[i] > X[-1]:
            Zi[:, i] = np.full(Zi.shape[0], np.nan)
            i -= 1
        j = 0
        while Yi[j] > Y[0]:
            Zi[j, :] = np.full(Zi.shape[1], np.nan)
            j += 1
        j = -1
        while Yi[j] < Y[-1]:
            Zi[j, :] = np.full(Zi.shape[1], np.nan)
            j -= 1

    return Zi


def interp2_scipy(X, Y, Z, Xi, Yi, method, borderNaNs=True,
                  griddata=False,
                  SBS=False,
                  RGI=False, extrap=True, RGI_fillVal=None,
                  CLT=False, CLT_fillVal=np.nan,
                  RBS=False):
    """
    Aims to provide similar functionality to interp2_gdal using SciPy's
    interpolation library. However, initial tests show that interp2_gdal
    both runs more quickly and produces output more similar to MATLAB's
    interp2 function for every method required by Ian's mosaicking script.
    griddata, SBS, and CLT interpolation methods are not meant to be used
    for the resampling of a large grid as is done here.
    """
    order = {
        'linear'   : 1,
        'quadratic': 2,
        'cubic'    : 3,
        'quartic'  : 4,
        'quintic'  : 5,
    }

    if griddata:
        # Supports nearest, linear, and cubic interpolation methods.
        # Has errored out with "QH7074 qhull warning: more than 16777215 ridges.
        #   ID field overflows and two ridges may have the same identifier."
        #   when used on large arrays. Fails to draw a convex hull of input points.
        # Needs more testing, but seems to handle NaN input. Output for linear and
        # cubic methods shows NaN borders when interpolating out of input domain.
        xx,  yy  = np.meshgrid(X, Y)
        xxi, yyi = np.meshgrid(Xi, Yi)
        Zi = scipy.interpolate.griddata((xx.flatten(),   yy.flatten()), Z.flatten(),
                                        (xxi.flatten(), yyi.flatten()), method)
        Zi.resize((Yi.size, Xi.size))

    elif SBS:
        # Supports all 5 orders of spline interpolation.
        # Can't handle NaN input; results in all NaN output.
        xx,  yy  = np.meshgrid(X, Y)
        xxi, yyi = np.meshgrid(Xi, Yi)
        fn = scipy.interpolate.SmoothBivariateSpline(xx.flatten(), yy.flatten(), Z.flatten(),
                                                     kx=order[method], ky=order[method])
        Zi = fn.ev(xxi, yyi)
        Zi.resize((Yi.size, Xi.size))

    elif (method == 'nearest') or ((method == 'linear') and np.any(np.isnan(Z))) or RGI:
        # Supports nearest and linear interpolation methods.
        xxi, yyi = np.meshgrid(Xi, Yi[::-1])
        pi = np.column_stack((yyi.flatten(), xxi.flatten()))
        fn = scipy.interpolate.RegularGridInterpolator((Y[::-1], X), Z, method=method,
                                                       bounds_error=(not extrap), fill_value=RGI_fillVal)
        Zi = fn(pi, method=method)
        Zi.resize((Yi.size, Xi.size))

    elif ((method == 'cubic') and np.any(np.isnan(Z))) or CLT:
        # Performs cubic interpolation of data,
        # but includes logic to first perform a nearest resampling of input NaNs.
        # Produces the same error as scipy.interpolate.griddata when used on large arrays.
        if np.any(np.isnan(Z)):
            Zi = interp2_scipy(X, Y, Z, Xi, Yi, 'nearest')
            Zi_data = np.where(~np.isnan(Zi))
            Z_data  = np.where(~np.isnan(Z))
            p  = np.column_stack((Z_data[0],   Z_data[1]))
            pi = np.column_stack((Zi_data[0], Zi_data[1]))
            fn = scipy.interpolate.CloughTocher2DInterpolator(p, Z[Z_data], fill_value=CLT_fillVal)
            Zi[Zi_data] = fn(pi)
        else:
            xx,  yy  = np.meshgrid(X, Y)
            xxi, yyi = np.meshgrid(Xi, Yi)
            p  = np.column_stack((xx.flatten(), yy.flatten()))
            pi = np.column_stack((xxi.flatten(), yyi.flatten()))
            fn = scipy.interpolate.CloughTocher2DInterpolator(p, Z.flatten(), fill_value=CLT_fillVal)
            Zi = fn(pi)
            Zi.resize((Yi.size, Xi.size))

    elif (method in ('quadratic', 'quartic')) or RBS:
        # Supports all 5 orders of spline interpolation.
        # Can't handle NaN input; results in all NaN output.
        fn = scipy.interpolate.RectBivariateSpline(Y[::-1], X, Z,
                                                   kx=order[method], ky=order[method])
        Zi = fn(Yi[::-1], Xi, grid=True)

    else:
        # Supports linear, cubic, and quintic interpolation methods.
        # Can't handle NaN input; results in all NaN output.
        # Default interpolator for its presumed efficiency.
        fn = scipy.interpolate.interp2d(X, Y[::-1], Z, kind=method)
        Zi = fn(Xi, Yi)

    if borderNaNs:
        # Rows and columns of Zi outside the domain of Z are made NaN.
        i = 0
        while Xi[i] < X[0]:
            Zi[:, i] = np.full(Zi.shape[0], np.nan)
            i += 1
        i = -1
        while Xi[i] > X[-1]:
            Zi[:, i] = np.full(Zi.shape[0], np.nan)
            i -= 1
        j = 0
        while Yi[j] > Y[0]:
            Zi[j, :] = np.full(Zi.shape[1], np.nan)
            j += 1
        j = -1
        while Yi[j] < Y[-1]:
            Zi[j, :] = np.full(Zi.shape[1], np.nan)
            j -= 1

    return Zi


def imresize(array, size, method='bicubic', use_gdal='auto', gdal_fix='scipy'):
    """
    This function is meant to replicate MATLAB's imresize function.

    Uses SciPy's scipy.misc.imresize function, GDAL's gdal.ReprojectImage function
    (through a call of the local interp2_gdal method), or a combination of both.

    For particular sets of input/output array shapes (dependent on resizing scale factor),
    the GDAL method can produce results for some interpolation methods that match MATLAB's
    imresize function pixel-for-pixel, with the exception of the last row and last column which
    currently can't be interpolated through the GDAL method (and are instead set to zeros) without
    applying some sort of fix.
    """
    gdal_fix_choices = (None, 'pad', 'pad_merge', 'rotate', 'scipy')

    # If a resize factor is provided for size, round up the x, y pixel
    # sizes for the output array to match MATLAB's imresize function.
    new_shape = size if type(size) == tuple else np.ceil(np.dot(size, array.shape)).astype(int)
    if use_gdal == 'auto':
        use_gdal = False if np.any((np.array(new_shape) - np.array(array.shape)) >= 0) else True
    if gdal_fix not in gdal_fix_choices:
        raise InvalidArgumentError("gdal_fix must be one of {}".format(gdal_fix_choices))

    old_array = array
    new_array = None

    if not use_gdal or gdal_fix == 'scipy':
        PILmode = 'L' if array.dtype in (np.bool, np.uint8) else 'F'
        if PILmode == 'L' and old_array.dtype != np.uint8:
            old_array = old_array.astype(np.uint8)
        new_array = scipy.misc.imresize(old_array, new_shape, method, PILmode)
        if 'uint' in str(array.dtype):
            new_array[new_array < 0] = 0

    if use_gdal:
        if gdal_fix == 'scipy':
            new_array_edgefix = new_array.astype(array.dtype)

        if gdal_fix in ('pad', 'pad_merge'):
            # Pad the right and bottom sides of the array with zeros before resizing,
            # then cut them off in the resized array before returning.
            pad = 1
            cur_array = old_array
            cur_shape = cur_array.shape
            res_shape = [s + pad for s in new_shape]
            pad_shape = map(lambda new_size, old_size: int((new_size+pad)*(old_size/new_size)), new_shape, cur_shape)
            pad_array = np.concatenate((cur_array, np.zeros((cur_shape[0], pad_shape[1]-cur_shape[1]))), axis=1)
            pad_array = np.concatenate((pad_array, np.zeros((pad_shape[0]-cur_shape[0], pad_shape[1]))), axis=0)
            old_array = pad_array
            new_shape = res_shape

        # Set up grid coordinate arrays, then run interp2_gdal.
        X = np.arange(old_array.shape[1]) + 1
        Y = np.arange(old_array.shape[0]) + 1
        Xi = np.linspace(X[0], X[-1], num=new_shape[1])
        Yi = np.linspace(Y[0], Y[-1], num=new_shape[0])
        new_array = interp2_gdal(X, Y, old_array, Xi, Yi, method, borderNaNs=False)

        if gdal_fix in ('pad', 'pad_merge'):
            new_array = new_array[0:-pad, 0:-pad]
            if gdal_fix == 'pad_merge':
                new_array_edgefix = new_array
                new_array = imresize(array, size, method, use_gdal=True, gdal_fix=None)

        if gdal_fix == 'rotate':
            # Rotate the input array 180 degrees and interpolate it again to retrieve
            # good values for all but the upper-right and lower-left corner pixels.
            new_array_edgefix = np.rot90(interp2_gdal(X, Y, np.rot90(old_array, 2), Xi, Yi, method, borderNaNs=False), 2)

        if 'new_array_edgefix' in vars():
            new_array[-1, :] = new_array_edgefix[-1, :]
            new_array[:, -1] = new_array_edgefix[:, -1]

        if 'uint' in str(array.dtype):
            new_array[new_array < 0] = 0

    return new_array.astype(array.dtype)


def conv2(array, kernel, shape='full', method='auto',
          default_double_out=True, allow_flipped_processing=True):
    """
    Convolve two 2D arrays.

    Parameters
    ----------
    array : ndarray, 2D
        Primary array to convolve.
    kernel : ndarray, 2D
        Secondary array to convolve with the primary array.
    shape : str; 'full', 'same', or 'valid'
        See documentation for scipy.signal.convolve. [1]
    method : str; 'direct', 'fft', or 'auto'
        See documentation for scipy.signal.convolve. [1]
    default_double_out : bool
        If True, returns an array of type np.float64
        unless the input array is of type np.float32,
        in which case an array of type np.float32 is returned.
        The sole purpose of this option is to allow this function
        to most closely replicate the corresponding MATLAB array method. [2]
    allow_flipped_processing : bool
        If True and at least one of the kernel array's sides
        has an even length, rotate both input array and kernel
        180 degrees before performing convolution, then rotate
        the result array 180 degrees before returning.
        The sole purpose of this option is to allow this function
        to most closely replicate the corresponding MATLAB array method. [2]

    Returns
    -------
    conv2 : ndarray, 2D
        A 2D array containing the convolution of input array and kernel.

    NOTES
    -----
    This function is meant to replicate MATLAB's conv2 function. [2]

    Scipy's convolution functionS cannot handle NaN input as it results in all NaN output.
    In comparison, MATLAB's conv2 function takes a sensible approach by letting NaN win out
    in all calculations involving pixels with NaN values in the input array.
    To replicate this, we set all NaN values to zero before performing convolution,
    then mask our result array with NaNs according to a binary dilation of ALL NaN locations
    in the input array, dilating using a structure of ones with same shape as the provided 'kernel'.

    For large arrays, this function will use an FFT method for convolution that results in
    FLOP errors on the order of 10^-12.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html
    .. [2] https://www.mathworks.com/help/matlab/ref/conv2.html

    """
    if default_double_out:
        dtype_out = None
        if isinstance(array.dtype.type(1), np.floating):
            dtype_out = array.dtype
            if (isinstance(kernel.dtype.type(1), np.floating)
                and int(str(kernel.dtype).replace('float', '')) > int(str(dtype_out).replace('float', ''))):
                warn("Since default_double_out=True, kernel with floating dtype ({}) at greater precision than"
                     " array floating dtype ({}) is cast to array dtype".format(str(kernel.dtype), str(dtype_out)))
                kernel = kernel.astype(dtype_out)
        else:
            dtype_out = np.float64

    rotation_flag = False
    if allow_flipped_processing:
        array, kernel, rotation_flag = rotate_array_if_kernel_has_even_sidelength(array, kernel)

    fixnans_flag = False
    if isinstance(array.dtype.type(1), np.floating):
        array_nans = np.isnan(array)
        if np.any(array_nans):
            fixnans_flag = True
            array[array_nans] = 0
        else:
            del array_nans

    if method == 'auto':
        method = scipy.signal.choose_conv_method(array, kernel, shape)
    result = scipy.signal.convolve(array, kernel, shape, method)
    if method != 'direct' and isinstance(result.dtype.type(1), np.floating):
        # Fix FLOP error from FFT method where we assume zero was the desired result.
        result[(-1.0e-12 < result) & (result < 10.0e-12)] = 0

    if fixnans_flag:
        result[imdilate(array_nans, np.ones(kernel.shape), shape)] = np.nan
        # Return the input array to its original state.
        array[array_nans] = np.nan

    if default_double_out and result.dtype != dtype_out:
        result = result.astype(dtype_out)

    return fix_array_if_rotation_was_applied(result, rotation_flag)


def moving_average(array, kernel_size=None, kernel=None, shape='same', method='auto',
                   float_dtype=None, allow_flipped_processing=True):
    # FIXME: Rewrite docstring in new standard.
    """
    Given an input array of any type, returns an array of the same size with each pixel containing
    the average of the surrounding [kernel_size x kernel_size] neighborhood (or a that of a custom
    boolean neighborhood specified through input 'kernel').
    Arguments 'mode' and allow_rotation are forwarded through to the convolution function.
    """
    float_dtype_choices = (None, np.float16, np.float32, np.float64)

    if kernel_size is None and kernel is None:
        raise InvalidArgumentError("Either kernel_size or kernel must be provided")
    if float_dtype not in float_dtype_choices:
        raise InvalidArgumentError("float_dtype must be one of {}".format(float_dtype_choices))

    float_dtype_bits = int(str(float_dtype(1).dtype).replace('float', '')) if float_dtype is not None else np.inf
    array_float_bits = int(str(array.dtype).replace('float', '')) if isinstance(array.dtype.type(1), np.floating) else 0
    if array_float_bits > float_dtype_bits:
        warn("Input float_dtype ({}) is lower precision than input array floating dtype ({})".format(
             str(float_dtype(1).dtype), str(array.dtype))
             + "\n-> Casting array to float_dtype")
        array = array.astype(float_dtype)
    if float_dtype is None:
        if array_float_bits > 0:
            float_dtype_bits = array_float_bits
        else:
            float_dtype_bits = 64
        float_dtype = eval('np.float{}'.format(float_dtype_bits))

    if kernel is not None:
        if np.any(~np.logical_or(kernel == 0, kernel == 1)):
            raise InvalidArgumentError("kernel may only contain zeros and ones")

        if kernel.dtype != float_dtype:
            kernel = kernel.astype(float_dtype)

        conv_kernel = np.rot90(kernel / np.sum(kernel), 2)
    else:
        conv_kernel = np.ones((kernel_size, kernel_size), dtype=float_dtype) / (kernel_size**2)

    return conv2(array, conv_kernel, shape, method,
                 default_double_out=False, allow_flipped_processing=allow_flipped_processing)


def imerode(array, structure, shape='same', mode='conv', allow_flipped_processing=True):
    # TODO: Write docstring.
    """
    This function is meant to replicate MATLAB's imerode function.
    """
    mode_choices = ('conv', 'skimage', 'scipy', 'scipy_grey')

    if structure.dtype != np.bool and np.any(~np.logical_or(structure == 0, structure == 1)):
        raise InvalidArgumentError("structure contains values other than 0 and 1")
    if mode not in mode_choices:
        raise InvalidArgumentError("'mode' must be one of {}".format(mode_choices))

    if mode == 'conv':
        if not isinstance(array.dtype.type(1), np.floating):
            prod_bitdepth = math.log(np.prod(structure.shape)+1, 2)
            dtype_bitdepths = (1, 8, 16, 32, 64)
            conv_bitdepth = None
            for b in dtype_bitdepths:
                if prod_bitdepth <= b:
                    conv_bitdepth = b
                    break
            if conv_bitdepth == 1:
                structure = structure.astype(np.bool)
            else:
                structure = structure.astype(eval('np.uint{}'.format(conv_bitdepth)))
        structure = np.rot90(structure, 2)

    rotation_flag = False
    if allow_flipped_processing:
        array, structure, rotation_flag = rotate_array_if_kernel_has_even_sidelength(array, structure)

    if mode == 'skimage':
        pady, padx = np.array(structure.shape) / 2
        pady, padx = int(pady), int(padx)

    if array.dtype == np.bool:
        if mode == 'conv':
            result = (conv2(~array, structure, shape, method='auto',
                            default_double_out=False, allow_flipped_processing=False) == 0)
        elif mode in ('scipy', 'scipy_grey'):
            result = sp_ndimage.binary_erosion(array, structure, border_value=1)
        elif mode == 'skimage':
            array = np.pad(array, ((pady, pady), (padx, padx)), 'constant', constant_values=1)
            result = sk_morphology.binary_erosion(array, structure)[pady:-pady, padx:-padx]

    elif mode == 'scipy_grey':
        if np.any(structure != 1):
            if not isinstance(structure.dtype.type(1), np.floating):
                structure = structure.astype(np.float32)
            result = sp_ndimage.grey_erosion(array, structure=(structure - 1))
        else:
            result = sp_ndimage.grey_erosion(array, size=structure.shape)

    else:
        array_vals = np.unique(array)

        array_vals_nans = np.isnan(array_vals)
        has_nans = np.any(array_vals_nans)
        if has_nans:
            array_nans = np.isnan(array)
            # Remove possible multiple occurrences of "nan" in results of np.unique().
            array_vals = np.delete(array_vals, np.where(np.isnan(array_vals)))
            array_vals = np.append(array_vals, np.nan)

        if mode == 'skimage':
            padval = np.inf if isinstance(array.dtype.type(1), np.floating) else np.iinfo(array.dtype).max
            array = np.pad(array, ((pady, pady), (padx, padx)), 'constant', constant_values=padval)

        result = np.full_like(array, array_vals[0])
        for val in array_vals[1:]:
            if not np.isnan(val):
                mask_val = (array >= val) if not has_nans else np.logical_or(array >= val, array_nans)
            else:
                mask_val = array_nans if mode != 'skimage' else np.logical_or(array_nans, array == np.inf)

            if mode == 'conv':
                result_val = (conv2(~mask_val, structure, shape, method='auto',
                                    default_double_out=False, allow_flipped_processing=False) == 0)
            elif mode == 'scipy':
                result_val = sp_ndimage.binary_erosion(mask_val, structure, border_value=1)
            elif mode == 'skimage':
                result_val = sk_morphology.binary_erosion(mask_val, structure)

            result[result_val] = val

        if mode == 'skimage':
            result = result[pady:-pady, padx:-padx]

    return fix_array_if_rotation_was_applied(result, rotation_flag)


def imdilate(array, structure, shape='same', mode='conv', allow_flipped_processing=True):
    # TODO: Write docstring.
    """
    This function is meant to replicate MATLAB's imdilate function.
    """
    mode_choices = ('conv', 'skimage', 'scipy', 'scipy_grey')

    if structure.dtype != np.bool and np.any(~np.logical_or(structure == 0, structure == 1)):
        raise InvalidArgumentError("structure contains values other than 0 and 1")
    if mode not in mode_choices:
        raise InvalidArgumentError("'mode' must be one of {}".format(mode_choices))

    if mode == 'conv':
        if not isinstance(array.dtype.type(1), np.floating):
            prod_bitdepth = math.log(np.prod(structure.shape)+1, 2)
            dtype_bitdepths = (1, 8, 16, 32, 64)
            conv_bitdepth = None
            for b in dtype_bitdepths:
                if prod_bitdepth <= b:
                    conv_bitdepth = b
                    break
            if conv_bitdepth == 1:
                structure = structure.astype(np.bool)
            else:
                structure = structure.astype(eval('np.uint{}'.format(conv_bitdepth)))

    rotation_flag = False
    if mode in ('scipy', 'scipy_grey', 'skimage') and allow_flipped_processing:
        array, structure, rotation_flag = rotate_array_if_kernel_has_even_sidelength(array, structure)

    if array.dtype == np.bool:
        if mode == 'conv':
            result = (conv2(array, structure, shape, method='auto',
                            default_double_out=False, allow_flipped_processing=False) > 0)
        elif mode in ('scipy', 'scipy_grey'):
            result = sp_ndimage.binary_dilation(array, structure, border_value=0)
        elif mode == 'skimage':
            result = sk_morphology.binary_dilation(array, structure)

    elif mode == 'scipy_grey':
        if np.any(structure != 1):
            if not isinstance(structure.dtype.type(1), np.floating):
                structure = structure.astype(np.float32)
            result = sp_ndimage.grey_dilation(array, structure=(structure - 1))
        else:
            result = sp_ndimage.grey_dilation(array, size=structure.shape)

    else:
        array_vals = np.unique(array)
        array_vals_nans = np.isnan(array_vals)
        has_nans = np.any(array_vals_nans)
        if has_nans:
            # Remove possible multiple occurrences of "nan" in results of np.unique().
            array_vals = np.delete(array_vals, np.where(np.isnan(array_vals)))
            array_vals = np.append(array_vals, np.nan)

        result = np.full_like(array, array_vals[0])
        for val in array_vals[1:]:
            mask_val = (array == val) if not np.isnan(val) else np.isnan(array)

            if mode == 'conv':
                result_val = (conv2(mask_val, structure, shape, method='auto',
                                    default_double_out=False, allow_flipped_processing=False) > 0)
            elif mode == 'scipy':
                result_val = sp_ndimage.binary_dilation(mask_val, structure, border_value=0)
            elif mode == 'skimage':
                result_val = sk_morphology.binary_dilation(mask_val, structure)

            result[result_val] = val

    return fix_array_if_rotation_was_applied(result, rotation_flag)


def bwareaopen(array, size_tolerance, connectivity=8, in_place=False):
    # TODO: Write docstring.
    if array.dtype == np.bool:
        binary_array = array
    else:
        warn("Input array to `bwareaopen` is not a boolean array"
             "\n-> Casting array to np.bool before performing function;"
             " beware that in_place argument cannot modify input array")
        binary_array = array.astype(np.bool)
        in_place = True
    return sk_morphology.remove_small_objects(binary_array, size_tolerance, connectivity/4, in_place)


def bwboundaries_array(array, side='inner', connectivity=8, noholes=False,
                       edge_boundaries=True, grey_boundaries=False):
    """
    Return a binary array with 1-px borders of ones highlighting
    boundaries between areas of differing values in the input array.

    Parameters
    ----------
    array : ndarray, 2D
        Binary array from which to extract black-white boundaries.
    side : str, 'inner' or 'outer'
        If 'inner', boundaries are on the side of ones.
        If 'outer', boundaries are on the side of zeros.
    connectivity : int, 4 or 8
        For drawing boundaries...
        If 4, only pixels with touching edges are considered connected.
        If 8, pixels with touching edges and corners are considered connected.
    noholes : bool
        If True, do not draw boundaries of zero clusters surrounded by ones.
    edge_boundaries : bool
        If True, take all nonzero pixels on the edges of the array to be boundaries.
    grey_boundaries : bool
        If True and a non-boolean array is provided,
        boundaries between areas of different values are drawn.

    Returns
    -------
    bwboundaries_array : ndarray of bool, same shape as input array
        A binary array with 1-px borders of ones highlighting boundaries
        between areas of differing values in the input array.

    Notes
    -----
    This function is meant to replicate MATLAB's bwboundaries function. [1]

    References
    ----------
    .. [1] https://www.mathworks.com/help/images/ref/bwboundaries.html

    """
    side_choices = ('inner', 'outer')
    conn_choices = (4, 8)
    if side not in side_choices:
        raise InvalidArgumentError("'side' must be one of {}".format(side_choices))
    if connectivity not in conn_choices:
        raise InvalidArgumentError("connectivity must be one of {}".format(conn_choices))

    binary_array = array.astype(np.bool) if (array.dtype != np.bool and not grey_boundaries) else array
    fn = imerode if side == 'inner' else imdilate
    structure = np.zeros((3, 3), dtype=np.int8)
    if connectivity == 8:
        structure[:, 1] = 1
        structure[1, :] = 1
    elif connectivity == 4:
        structure[:, :] = 1

    if noholes:
        array_filled = sp_ndimage.binary_fill_holes(binary_array)
        result = (array_filled != fn(array_filled, structure=structure))
    else:
        result = (binary_array != fn(binary_array, structure=structure))

    if edge_boundaries:
        result[ 0, np.nonzero(binary_array[ 0, :])] = True
        result[-1, np.nonzero(binary_array[-1, :])] = True
        result[np.nonzero(binary_array[:,  0]),  0] = True
        result[np.nonzero(binary_array[:, -1]), -1] = True

    return result


def entropyfilt(array, kernel, bin_bitdepth=8, nbins=None):
    # TODO: Write docstring.

    if bin_bitdepth is None and nbins is None:
        raise InvalidArgumentError("Either bin_bitdepth or nbins must be provided")
    if nbins is None:
        if type(bin_bitdepth) == int and 1 <= bin_bitdepth <= 64:
            nbins = 2**bin_bitdepth
        else:
            raise InvalidArgumentError("bin_bitdepth must be an integer between 1 and 64, inclusive")
    else:
        if type(nbins) == int and 2 <= nbins <= 2**64:
            bin_bitdepth = math.log(nbins, 2)
        else:
            raise InvalidArgumentError("nbins must be an integer between 2 and 2**64, inclusive")

    array_dtype_str = str(array.dtype)
    array_gentype = None
    array_dtype_bitdepth = None
    try:
        if array_dtype_str == 'bool':
            array_gentype = 'bool'
        elif array_dtype_str.startswith('int'):
            array_gentype = 'int'
        elif array_dtype_str.startswith('uint'):
            array_gentype = 'uint'
        elif array_dtype_str.startswith('float'):
            array_gentype = 'float'
        else:
            raise ValueError
        if array_gentype == 'bool':
            array_dtype_bitdepth = 1
        elif array_gentype in ('int, uint'):
            # The following throw ValueError if an invalid input array dtype is encountered.
            array_dtype_bitdepth = int(array_dtype_str.split('int')[-1])
        elif array_gentype == 'float':
            array_dtype_bitdepth = np.inf
    except ValueError:
        raise InvalidArgumentError("array dtype np.{} is not supported".format(array_dtype_str))

    bin_array = None
    if array_dtype_bitdepth <= bin_bitdepth and array_gentype != 'float':
        bin_array = array
    else:
        if nbins is None:
            nbins = 2**bin_bitdepth

        bin_array_max = nbins - 1
        bin_array = array.astype(np.float64) if array_dtype_bitdepth > 16 else array.astype(np.float32)
        bin_array = bin_array - np.nanmin(bin_array)
        array_range = np.nanmax(bin_array)
        if array_range > bin_array_max:
            bin_array = round_half_up(array / array_range * bin_array_max)

        dtype_bitdepths = (1, 8, 16, 32, 64)
        bin_array_bitdepth = None
        for b in dtype_bitdepths:
            if bin_bitdepth <= b:
                bin_array_bitdepth = b
                break
        if bin_array_bitdepth == 1:
            bin_array = bin_array.astype(np.bool)
        else:
            bin_array = bin_array.astype(eval('np.uint{}'.format(bin_array_bitdepth)))

    return entropy(bin_array, kernel)


def convex_hull_image_offsets_diamond(ndim):
    # TODO: Remove this function once skimage package is updated to include it.
    offsets = np.zeros((2 * ndim, ndim))
    for vertex, (axis, offset) in enumerate(product(range(ndim), (-0.5, 0.5))):
        offsets[vertex, axis] = offset
    return offsets


def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10):
    # TODO: Remove this function once skimage package is updated to include it.
    """Compute the convex hull image of a binary image.
    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    Parameters
    ----------
    image : array
        Binary input image. This array is cast to bool before processing.
    offset_coordinates : bool, optional
        If ``True``, a pixel at coordinate, e.g., (4, 7) will be represented
        by coordinates (3.5, 7), (4.5, 7), (4, 6.5), and (4, 7.5). This adds
        some "extent" to a pixel when computing the hull.
    tolerance : float, optional
        Tolerance when determining whether a point is inside the hull. Due
        to numerical floating point errors, a tolerance of 0 can result in
        some points erroneously being classified as being outside the hull.

    Returns
    -------
    hull : (M, N) array of bool
        Binary image with pixels in convex hull set to True.

    References
    ----------
    .. [1] http://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/

    """
    ndim = image.ndim

    # In 2D, we do an optimisation by choosing only pixels that are
    # the starting or ending pixel of a row or column.  This vastly
    # limits the number of coordinates to examine for the virtual hull.
    if ndim == 2:
        coords = sk_morphology._convex_hull.possible_hull(image.astype(np.uint8))
    else:
        coords = np.transpose(np.nonzero(image))
        if offset_coordinates:
            # when offsetting, we multiply number of vertices by 2 * ndim.
            # therefore, we reduce the number of coordinates by using a
            # convex hull on the original set, before offsetting.
            hull0 = scipy.spatial.ConvexHull(coords)
            coords = hull0.points[hull0.vertices]

    # Add a vertex for the middle of each pixel edge
    if offset_coordinates:
        offsets = convex_hull_image_offsets_diamond(image.ndim)
        coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim)

    # ERIK'S NOTE: Added the following conditional barrier for speed.
    if offset_coordinates or ndim != 2:
        # repeated coordinates can *sometimes* cause problems in
        # scipy.spatial.ConvexHull, so we remove them.
        coords = unique_rows(coords)

    # Find the convex hull
    hull = ConvexHull(coords)
    vertices = hull.points[hull.vertices]

    # If 2D, use fast Cython function to locate convex hull pixels
    if ndim == 2:
        # ERIK'S NOTE: Substituted grid_points_in_poly() for the following for speed.
        # mask = grid_points_in_poly(image.shape, vertices)
        hull_perim_r, hull_perim_c = polygon_perimeter(vertices[:, 0], vertices[:, 1])
        mask = np.zeros(image.shape, dtype=np.bool)
        mask[hull_perim_r, hull_perim_c] = True
        mask = sp_ndimage.morphology.binary_fill_holes(mask)
    else:
        gridcoords = np.reshape(np.mgrid[tuple(map(slice, image.shape))],
                                (ndim, -1))
        # A point is in the hull if it satisfies all of the hull's inequalities
        coords_in_hull = np.all(hull.equations[:, :ndim].dot(gridcoords) +
                                hull.equations[:, ndim:] < tolerance, axis=0)
        mask = np.reshape(coords_in_hull, image.shape)

    return mask


def concave_hull_image_traverse_alpha_length(boundary_points, boundary_res, convex_hull, indices, indptr):
    # TODO: Write docstring.

    alpha_min = boundary_res
    alpha_max = boundary_res
    edge_info = {}
    revisit_edges = deque()
    amin_edges = set()

    for k1, k2 in convex_hull:
        next_edge = (k1, k2) if k1 < k2 else (k2, k1)
        p1, p2 = boundary_points[[k1, k2]]
        p1_p2 = p2 - p1
        next_alpha = np.sqrt(np.sum(np.square(p2 - p1)))
        alpha_max = max(alpha_max, next_alpha)
        if abs(p1_p2[0]) > boundary_res or abs(p1_p2[1]) > boundary_res:
            k3 = set(indptr[indices[k1]:indices[k1+1]]).intersection(
                 set(indptr[indices[k2]:indices[k2+1]])).pop()
            edge_info[next_edge] = [next_alpha, next_alpha, k3]
            revisit_edges.append((next_edge, k3))

            while len(revisit_edges) > 0:
                next_edge, revisit_k3 = revisit_edges.pop()
                k1, k2 = next_edge
                k3 = revisit_k3
                revisit_edge_info = edge_info[next_edge]
                local_mam = revisit_edge_info[1]
                p1, p2, p3 = boundary_points[[k1, k2, k3]]

                while True:
                    forward_edges = []
                    edge_1_3 = None
                    p1_p3 = p3 - p1
                    edge_2_3 = None
                    p2_p3 = p3 - p2
                    if abs(p1_p3[0]) > boundary_res or abs(p1_p3[1]) > boundary_res:
                        edge_1_3 = (k1, k3) if k1 < k3 else (k3, k1)
                        forward_edges.append(edge_1_3)
                    if abs(p2_p3[0]) > boundary_res or abs(p2_p3[1]) > boundary_res:
                        edge_2_3 = (k2, k3) if k2 < k3 else (k3, k2)
                        forward_edges.append(edge_2_3)

                    next_edge = None
                    for fedge in forward_edges:
                        ka, kb = fedge
                        fedge_k3 = set(indptr[indices[ka]:indices[ka+1]]).intersection(
                                   set(indptr[indices[kb]:indices[kb+1]])).difference({k1, k2})
                        if not fedge_k3:
                            # We've arrived at a convex hull edge.
                            if fedge == edge_1_3:
                                edge_1_3 = None
                            else:
                                edge_2_3 = None
                            continue
                        fedge_k3 = fedge_k3.pop()

                        if fedge not in edge_info:
                            fedge_alpha = np.sqrt(np.sum(np.square(p1_p3 if fedge == edge_1_3 else p2_p3)))
                            fedge_mam = min(local_mam, fedge_alpha)
                            edge_info[fedge] = [fedge_alpha, fedge_mam, fedge_k3]

                        else:
                            fedge_info = edge_info[fedge]
                            fedge_alpha, fedge_mam_old, _ = fedge_info
                            fedge_mam = min(local_mam, fedge_alpha)
                            if fedge_mam > fedge_mam_old:
                                fedge_info[1] = fedge_mam
                                fedge_info[2] = fedge_k3
                            else:
                                if fedge_mam > alpha_min:
                                    alpha_min = fedge_mam
                                    amin_edges.add(fedge)
                                if fedge == edge_1_3:
                                    edge_1_3 = None
                                else:
                                    edge_2_3 = None
                                continue

                        if next_edge is None:
                            next_edge = fedge
                            next_mam = fedge_mam
                            next_k3 = fedge_k3

                    if next_edge is not None:
                        if edge_1_3 is not None:
                            if next_edge[0] == k1:
                                # p1 = p1
                                p2 = p3
                            else:
                                p2 = p1
                                p1 = p3

                            if edge_2_3 is not None:
                                revisit_edges.append((edge_2_3, fedge_k3))
                        else:
                            if next_edge[0] == k2:
                                p1 = p2
                                p2 = p3
                            else:
                                p1 = p3
                                # p2 = p2

                        k1, k2 = next_edge
                        k3 = next_k3
                        p3 = boundary_points[next_k3]
                        local_mam = next_mam

                        if revisit_k3:
                            revisit_edge_info[2] = revisit_k3
                            revisit_k3 = None
                    else:
                        break

    return alpha_min, alpha_max, edge_info, amin_edges


def concave_hull_image(image, concavity, fill=True,
                       data_boundary_res=3, alpha_cutoff_mode='unique',
                       debug=False):
    # TODO: Add more comments.
    # TODO: Write docstring.
    """

    Parameters
    ----------
    image :
    concavity :
    fill :
    data_boundary_res : positive int
        Minimum coordinate-wise distance between two points in a triangle for that edge to be
        traversed and allow the triangle on the other side of the edge to be considered for erosion.
    alpha_cutoff_mode :
    debug :

    Returns
    -------

    """
    if 0 <= concavity <= 1:
        pass
    else:
        raise InvalidArgumentError("concavity must be between 0 and 1, inclusive")
    if alpha_cutoff_mode not in ('mean', 'median', 'unique'):
        raise UnsupportedMethodError("alpha_cutoff_mode='{}'".format(alpha_cutoff_mode))

    # Find data coverage boundaries.
    data_boundary = bwboundaries_array(image, connectivity=8, noholes=True, side='inner')
    boundary_points = np.argwhere(data_boundary)

    if debug:
        import matplotlib.pyplot as plt
    else:
        del data_boundary

    tri = scipy.spatial.Delaunay(boundary_points)

    if debug in (True, 1):
        print "[DEBUG] concave_hull_image_alpha_line (1): Initial triangulation plot"
        plt.triplot(boundary_points[:, 1], -boundary_points[:, 0], tri.simplices.copy(), lw=1)
        plt.plot(boundary_points[:, 1], -boundary_points[:, 0], 'o', ms=1)
        plt.show()

    hull_convex = tri.convex_hull
    indices, indptr = tri.vertex_neighbor_vertices

    alpha_min, alpha_max, edge_info, amin_edges = concave_hull_image_traverse_alpha_length(
        boundary_points, data_boundary_res, hull_convex, indices, indptr
    )

    alpha_cut = None
    if concavity == 0 or alpha_min == alpha_max:
        alpha_cut = np.inf
    elif alpha_cutoff_mode == 'mean':
        alpha_cut = (alpha_min + alpha_max) / 2
    elif alpha_cutoff_mode in ('median', 'unique'):
        mam_allowed = [einfo[1] for einfo in edge_info.values() if einfo[1] > alpha_min]
        if not mam_allowed:
            warn("Of {} total edges in edge_info, none have mam > alpha_min={}".format(len(edge_info), alpha_min))
            alpha_cut = np.inf
        else:
            if alpha_cutoff_mode == 'unique':
                mam_allowed = list(set(mam_allowed))
            mam_allowed.sort()
            alpha_cut = mam_allowed[-int(np.ceil(len(mam_allowed) * concavity))]
        del mam_allowed

    if debug in (True, 2):
        print "[DEBUG] concave_hull_image_alpha_line (2): Triangulation traversal"
        print "alpha_min = {}".format(alpha_min)
        print "alpha_max = {}".format(alpha_max)
        print "concavity = {}".format(concavity)
        print "alpha_cut = {}".format(alpha_cut)
        while True:
            eaten_simplices = []
            eaten_tris_mam = []
            mam_instances = {}
            for edge in edge_info:
                einfo = edge_info[edge]
                if einfo[1] >= alpha_cut:
                    eaten_simplices.append([edge[0], edge[1], einfo[2]])
                    eaten_tris_mam.append(einfo[1])
                if einfo[1] == alpha_min:
                    mam_tris = []
                    mam_instances[edge] = mam_tris
                    for k1 in edge:
                        amin_neighbors = indptr[indices[k1]:indices[k1+1]]
                        for k2 in amin_neighbors:
                            possible_k3 = set(indptr[indices[k1]:indices[k1+1]]).intersection(set(indptr[indices[k2]:indices[k2+1]]))
                            for k3 in possible_k3:
                                mam_tris.append([k1, k2, k3])
            plt.triplot(boundary_points[:, 1], -boundary_points[:, 0], tri.simplices.copy(), lw=1)
            if eaten_simplices:
                plt.triplot(boundary_points[:, 1], -boundary_points[:, 0], eaten_simplices, color='black', lw=1)
                plt.tripcolor(boundary_points[:, 1], -boundary_points[:, 0], eaten_simplices, facecolors=np.array(eaten_tris_mam), lw=1)
            for amin_edge in amin_edges:
                plt.plot(boundary_points[amin_edge, 1], -boundary_points[amin_edge, 0], 'r--', lw=1)
            for mam_edge in mam_instances:
                mam_tris = mam_instances[mam_edge]
                plt.triplot(boundary_points[:, 1], -boundary_points[:, 0], mam_tris, color='red', lw=1)
            plt.plot(boundary_points[:, 1], -boundary_points[:, 0], 'o', ms=1)
            for hull_edge in hull_convex:
                plt.plot(boundary_points[hull_edge, 1], -boundary_points[hull_edge, 0], 'yo', lw=1.5)
            for mam_edge in mam_instances:
                plt.plot(boundary_points[mam_edge, 1], -boundary_points[mam_edge, 0], 'ro', lw=1.5)
            plt.show()
            user_input = raw_input("Modify params? (y/n): ")
            if user_input.lower() != "y":
                break
            validInput = False
            while not validInput:
                try:
                    user_input = raw_input("concavity = ")
                    if user_input == "":
                        break
                    else:
                        user_input_num = float(user_input)
                    if 0 <= user_input_num <= 1:
                        pass
                    else:
                        raise ValueError
                    concavity = user_input_num

                    alpha_cut = None
                    if concavity == 0 or alpha_min == alpha_max:
                        alpha_cut = np.inf
                    elif alpha_cutoff_mode == 'mean':
                        alpha_cut = (alpha_min + alpha_max) / 2
                    elif alpha_cutoff_mode in ('median', 'unique'):
                        mam_allowed = [einfo[1] for einfo in edge_info.values() if einfo[1] > alpha_min]
                        if not mam_allowed:
                            warn("Of {} total edges in edge_info, none have mam > alpha_min={}".format(len(edge_info), alpha_min))
                            alpha_cut = np.inf
                        else:
                            if alpha_cutoff_mode == 'unique':
                                mam_allowed = list(set(mam_allowed))
                            mam_allowed.sort()
                            alpha_cut = mam_allowed[-int(np.ceil(len(mam_allowed) * concavity))]
                        del mam_allowed

                    validInput = True
                    print "alpha_cut = {}".format(alpha_cut)
                except ValueError:
                    print "concavity must be an int or float between 0 and 1"
            while not validInput:
                try:
                    user_input = raw_input("alpha_cut = ")
                    if user_input == "":
                        break
                    else:
                        user_input_num = float(user_input)
                    alpha_cut = user_input_num
                    validInput = True
                except ValueError:
                    print "alpha_cut must be an int or float"

    eaten_tris = []
    mam_instances = []
    for edge in edge_info:
        einfo = edge_info[edge]
        if einfo[1] >= alpha_cut:
            eaten_tris.append(shapely.geometry.Polygon(boundary_points[[edge[0], edge[1], einfo[2]]]))
        if einfo[1] == alpha_min:
            mam_indices = []
            mam_instances.append(mam_indices)
            for k1 in edge:
                amin_neighbors = indptr[indices[k1]:indices[k1+1]]
                for k2 in amin_neighbors:
                    possible_k3 = set(indptr[indices[k1]:indices[k1+1]]).intersection(set(indptr[indices[k2]:indices[k2+1]]))
                    for k3 in possible_k3:
                        mam_indices.extend([k1, k2, k3])

    eaten_poly = shapely.ops.unary_union(eaten_tris)
    mam_poly = shapely.ops.unary_union(
        [shapely.geometry.MultiPoint(boundary_points[np.unique(indices)]).convex_hull for indices in mam_instances]
    )
    hull_convex_poly = shapely.geometry.MultiPoint(boundary_points[np.unique(hull_convex)]).convex_hull

    hull_concave_poly = hull_convex_poly.difference(eaten_poly.difference(mam_poly))
    if type(hull_concave_poly) == shapely.geometry.polygon.Polygon:
        hull_concave_poly = [hull_concave_poly]
    else:
        warn("Concave hull is broken into multiple polygons; try increasing data_boundary_res")

    del eaten_poly, mam_poly, hull_convex_poly

    mask = np.zeros(image.shape, dtype=np.bool)
    for poly in hull_concave_poly:
        cchull_r, cchull_c = poly.exterior.coords.xy
        cchull_r = np.array(cchull_r)
        cchull_c = np.array(cchull_c)

        if debug in (True, 3):
            print "[DEBUG] concave_hull_image_alpha_line (3): Concave hull boundary points"
            plt.triplot(boundary_points[:, 1], -boundary_points[:, 0], tri.simplices.copy(), lw=1)
            if eaten_simplices:
                plt.triplot(boundary_points[:, 1], -boundary_points[:, 0], eaten_simplices, color='red', lw=1)
            plt.plot(boundary_points[:, 1], -boundary_points[:, 0], 'o', ms=1)
            plt.plot(cchull_c, -cchull_r, 'ro', ms=3)
            i = 0
            for xy in np.column_stack((cchull_c, -cchull_r)):
                i += 1
                plt.annotate(s=str(i), xy=xy)
            plt.show()

        draw_r, draw_c = polygon_perimeter(cchull_r, cchull_c)
        mask[draw_r, draw_c] = 1

    if fill:
        mask = sp_ndimage.morphology.binary_fill_holes(mask)

    # TODO: Remove the following before sharing algorithm.
    if debug in (True, 4):
        debug_mask = np.zeros(image.shape, dtype=np.int8)
        debug_mask[mask] = 1
        debug_mask[data_boundary] += 2
        test.saveImage(debug_mask, 'debug_concave_hull_image')

    return mask


def getWindow(array, window_shape, x_y_tup, one_based_index=True):
    # TODO: Write docstring.
    # FIXME: Needs error checking on array bounds.
    window_ysize, window_xsize = window_shape
    colNum, rowNum = x_y_tup
    if one_based_index:
        rowNum -= 1
        colNum -= 1
    return array[int(rowNum-np.floor((window_ysize-1)/2)):int(rowNum+np.ceil((window_ysize-1)/2)+1),
                 int(colNum-np.floor((window_xsize-1)/2)):int(colNum+np.ceil((window_xsize-1)/2)+1)]



################################
# Data Boundary Polygonization #
################################


def getFPvertices(array, X, Y,
                  tolerance_start=100, nodataVal=np.nan, method='convhull'):
    """
    Polygonizes the generalized (hull) boundary of all data clusters in a
    NumPy 2D array with supplied grid coordinates [X, Y] (ranges in 1D array form)
    and simplifies this boundary until it contains 80 or fewer vertices.
    These 'footprint' vertices are returned as a tuple containing lists
    of all x-coordinates and all y-coordinates of these (ordered) points.
    """
    if nodataVal != np.nan:
        array_data = (array != nodataVal)
    else:
        array_data = ~np.isnan(array)

    # Get the data boundary ring.
    if method == 'convhull':
        # Fill interior nodata holes.
        array_filled = sp_ndimage.morphology.binary_fill_holes(array_data)
        try:
            ring = getFPring_nonzero(array_filled, X, Y)
        except MemoryError:
            print "MemoryError on call to getFPring_convhull in raster_array_tools:"
            print_exc()
            print "-> Defaulting to getFPring_nonzero"
            del array_filled
            ring = getFPring_nonzero(array_data, X, Y)

    elif method == 'nonzero':
        ring = getFPring_nonzero(array_data, X, Y)

    else:
        raise UnsupportedMethodError("method='{}'".format(method))

    del array_data
    if 'array_filled' in vars():
        del array_filled

    numVertices = ring.GetPointCount()
    if numVertices > 80:
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # Simplify the geometry until it has 80 or fewer points.
        toler = tolerance_start
        while numVertices > 80:
            poly = poly.SimplifyPreserveTopology(toler)
            ring = poly.GetGeometryRef(0)
            numVertices = ring.GetPointCount()
            if numVertices > 400:
                toler += 1000
            elif numVertices > 200:
                toler += 500
            elif numVertices > 100:
                toler += 300
            else:
                toler += 200

    boundary_points = ring.GetPoints()

    points_xlist = map(lambda point_tup: point_tup[0], boundary_points)
    points_ylist = map(lambda point_tup: point_tup[1], boundary_points)

    return points_xlist, points_ylist


def getFPring_convhull(array_filled, X, Y):
    """
    Traces the boundary of a (large) pre-hole-filled data mass in array_filled
    using a convex hull function.
    Returns an OGRGeometry object in ogr.wkbLinearRing format representing
    footprint vertices of the data mass, using [X, Y] grid coordinates.
    """
    # Derive data cluster boundaries (in array representation).
    data_boundary = (array_filled != sp_ndimage.binary_erosion(array_filled))

    boundary_points = np.argwhere(data_boundary)
    del data_boundary

    # Convex hull method.
    convex_hull = scipy.spatial.ConvexHull(boundary_points)
    hull_points = boundary_points[convex_hull.vertices]
    del convex_hull

    # Assemble the geometry.
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for p in hull_points:
        ring.AddPoint_2D(X[p[1]], Y[p[0]])  # Make points (x-coord, y-coord)
    # Close the ring.
    ring.AddPoint_2D(X[hull_points[0][1]],
                     Y[hull_points[0][0]])

    return ring


def getFPring_nonzero(array_data, X, Y):
    """
    Traces a simplified boundary of a (large) pre-hole-filled data mass
    in array_filled by making one scan across the columns of the array
    and recording the top and bottom data points found in each column.
    Returns an OGRGeometry object in ogr.wkbLinearRing format representing
    footprint vertices of the data mass, using [X, Y] grid coordinates.
    """
    # Scan left to right across the columns in the binary data array,
    # building top and bottom routes that are simplified because they
    # cannot represent the exact shape of some 'eaten-out' edges.
    top_route = []
    bottom_route = []
    for colNum in range(array_data.shape[1]):
        rowNum_data = np.nonzero(array_data[:, colNum])[0]
        if rowNum_data.size > 0:
            top_route.append((rowNum_data[0], colNum))
            bottom_route.append((rowNum_data[-1], colNum))

    # Prepare the two routes (check endpoints) for connection.
    bottom_route.reverse()
    if top_route[-1] == bottom_route[0]:
        del top_route[-1]
    if bottom_route[-1] != top_route[0]:
        bottom_route.append(top_route[0])

    # Assemble the geometry.
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for p in top_route:
        ring.AddPoint_2D(X[p[1]], Y[p[0]])  # Make points (x-coord, y-coord)
    for p in bottom_route:
        ring.AddPoint_2D(X[p[1]], Y[p[0]])  # Make points (x-coord, y-coord)

    return ring


def getDataBoundariesPoly(array, X, Y, nodataVal=np.nan, coverage='all',
                          erode=False, BBS=True):
    """
    Polygonizes the boundaries of all data clusters in a NumPy 2D array,
    using supplied [X, Y] grid coordinates (ranges in 1D array form) for vertices.
    Returns an OGRGeometry object in ogr.wkbPolygon format.
    If coverage='outer', interior nodata holes are filled before tracing.
    If erode=True, data spurs are eliminated before tracing.
    If BBS=True, Bi-directional Boundary Skewing preprocessing is done.
    --Utilizes a fast boundary tracing method: outline.c
    """
    if nodataVal != np.nan:
        data_array = (array != nodataVal)
    else:
        data_array = ~np.isnan(array)

    if BBS:
        # Pad data array with zeros and extend grid coordinates arrays
        # in preparation for Bi-directional Boundary Skewing.
        dx = X[1]-X[0]
        dy = Y[1]-Y[0]
        data_array = np.pad(data_array, 2, 'constant')
        X_ext = np.concatenate((np.array([X[0]-2*dx, X[0]-dx]), X, np.array([X[-1]+dx, X[-1]+2*dx])))
        Y_ext = np.concatenate((np.array([Y[0]-2*dy, Y[0]-dy]), Y, np.array([Y[-1]+dy, Y[-1]+2*dy])))
    else:
        X_ext, Y_ext = X, Y

    if coverage == 'outer':
        # Fill interior nodata holes.
        data_array = sp_ndimage.morphology.binary_fill_holes(data_array)

    if erode:
        # Erode data regions to eliminate *any* data spurs (possible 1 or 2 pixel-
        # width fingers that stick out from data clusters in the original array).
        # This should make the tracing of data boundaries more efficient since
        # the rings that are traced will be more substantial.
        data_array = sp_ndimage.binary_erosion(data_array, structure=np.ones((3, 3)))
        if ~np.any(data_array):
            # No data clusters large enough to have a traceable boundary exist.
            return None
        # Reverse the erosion process to retrieve a data array that does not
        # contain data spurs, yet remains true to data coverage.
        data_array = sp_ndimage.binary_dilation(data_array, structure=np.ones((3, 3)))

    if BBS:
        # Bi-directional Boundary Skewing
        # To represent data coverage fairly for most data pixels in the raster image,
        # the right and bottom edges of all data boundaries must grow by one pixel so
        # that their full extents may be recorded upon grid coordinate lookup of data
        # boundary nodes after each boundary ring is traced.
        print "Performing Bi-directional Boundary Skewing"
        outer_boundary = (
            sp_ndimage.binary_dilation(data_array, structure=np.ones((3, 3))) != data_array
        )
        outer_boundary_nodes = [(row[0], row[1]) for row in np.argwhere(outer_boundary)]
        # In skew_check, 0 is the location of an outer boundary node to be checked, 'n'.
        # If there is a 1 in the data array at any of the three corresponding neighbor
        # locations, n will be set in the data array for data boundary tracing.
        skew_check = np.array([[1,1],
                               [1,0]])
        skew_additions = np.zeros_like(data_array)
        for n in outer_boundary_nodes:
            window = data_array[n[0]-1:n[0]+1, n[1]-1:n[1]+1]
            if np.any(skew_check & window):
                skew_additions[n] = 1
        data_array = (data_array | skew_additions)

    # Find data coverage boundaries.
    data_boundary = (data_array != sp_ndimage.binary_erosion(data_array))

    # Create polygon.
    poly = ogr.Geometry(ogr.wkbPolygon)
    ring_count = 0
    for colNum in range(data_boundary.shape[1]):
        rowNum_dataB = np.where(data_boundary[:, colNum])[0]
        while rowNum_dataB.size > 0:
            rowNum = rowNum_dataB[0]
            home_node = (rowNum, colNum)

            # Trace the data cluster.
            print "Tracing ring from home node {}".format(home_node)
            ring_route = outline(data_boundary, 1, start=home_node)

            # Create ring geometry.
            data_ring = ogr.Geometry(ogr.wkbLinearRing)
            for p in ring_route:
                data_ring.AddPoint_2D(X_ext[p[1]], Y_ext[p[0]])  # Make points (x-coord, y-coord)
                data_boundary[p] = 0    # Fill in ring in data boundary array
                                        # to mark that this ring is captured.
            # # (Alternative method of filling in ring.)
            # ring_route = np.array(ring_route).T  ## DOES preserve point order.
            # data_boundary[ring_route[0], ring_route[1]] = 0

            # Add ring to the polygon.
            poly.AddGeometry(data_ring)
            ring_count += 1

            # Search for more rings that may intersect this column in the data boundary array.
            rowNum_dataB = np.where(data_boundary[:, colNum])[0]

    print "Found {} rings!".format(ring_count)
    return poly


def outline(array, every, start=None, pass_start=False, complete_ring=True):
    """
    Taking an (binary) array as input, finds the first set node in the array
    (by scanning down each column as it scans left to right across the array)
    [may instead be specified by giving the argument start=(row, col)]
    and traces the inner boundary of the structure (of set nodes) it is
    connected to, winding up back at this starting node.
    The path taken is returned as an ordered list of nodes in (row, col) form
    where every "every"-th node is reported.
    If pass_start=False, the path is complete when the first node is revisited.
    If pass_start=True, the path is complete when the first AND second nodes are revisited.
    If complete_ring=True, the first and last nodes in the returned route will match.
    --Utilizes a fast boundary tracing method: (outline.c / outline_every1.c)
    --Does not adhere to either connectivity 1 or 2 rule in data boundary path-finding.
    Both this function and outline.c have been modified from their original forms, found at:
    http://stackoverflow.com/questions/14110904/numpy-binary-raster-image-to-polygon-transformation
    """
    if type(array) != np.ndarray:
        raise InvalidArgumentError("'array' must be of type numpy.ndarray")
    if type(every) != int or every < 1:
        raise InvalidArgumentError("'every' must be a positive integer")

    if len(array) == 0:
        return np.array([])

    # Set up arguments to (outline.c / outline_every1.c)
    if start is not None:
        rows, cols = array.shape
        starty, startx = start
        if starty < 0 or starty >= rows or startx < 0 or startx >= cols:
            raise InvalidArgumentError("Invalid 'start' node: {}".format(start))
        starty += 1
        startx += 1
    else:
        starty, startx = -1, -1
    pass_start = int(pass_start)
    data = np.pad(array, 1, 'constant')
    rows, cols = data.shape

    if every != 1:
        padded_route = scipy.weave.inline(
            _outline, ['data', 'rows', 'cols', 'every', 'starty', 'startx', 'pass_start'],
            type_converters=scipy.weave.converters.blitz
        )
        if complete_ring and (len(padded_route) > 0) and (padded_route[0] != padded_route[-1]):
            padded_route.append(padded_route[0])
    else:
        padded_route = scipy.weave.inline(
            _outline_every1, ['data', 'rows', 'cols', 'starty', 'startx', 'pass_start'],
            type_converters=scipy.weave.converters.blitz
        )

    fixed_route = [(row[0], row[1]) for row in (np.array(padded_route) - 1)]

    return fixed_route


def connectEdges(edge_collection, allow_modify_deque_input=True):
    # TODO: Test function.
    """
    Takes a collection of edges, each edge being an ordered collection of vertex numbers,
    and recursively connects them by linking edges with endpoints that have matching vertex numbers.
    A reduced list of edges (each edge as a deque) is returned. This list will contain multiple edge
    components if the input edges cannot be connected to form a single unbroken edge.
    If allow_modify_deque_input=True, an input edge_list containing deque edge components will be
    modified.
    """
    edges_input = None
    if type(edge_collection[0]) == deque:
        edges_input = edge_collection if allow_modify_deque_input else copy.deepcopy(edge_collection)
    else:
        edges_input = [deque(e) for e in edge_collection]

    while True:
        edges_output = []
        for edge_in in edges_input:
            edge_in_end_l, edge_in_end_r = edge_in[0], edge_in[-1]
            connected = False
            for edge_out in edges_output:
                if edge_out[0] == edge_in_end_l:
                    edge_out.popleft()
                    edge_out.extendleft(edge_in)
                    edge_in_added = True
                    break
                if edge_out[0] == edge_in_end_r:
                    edge_out.popleft()
                    edge_in.reverse()
                    edge_out.extendleft(edge_in)
                    edge_in_added = True
                    break
                if edge_out[-1] == edge_in_end_l:
                    edge_out.pop()
                    edge_out.extend(edge_in)
                    edge_in_added = True
                    break
                if edge_out[-1] == edge_in_end_r:
                    edge_out.pop()
                    edge_in.reverse()
                    edge_out.extend(edge_in)
                    edge_in_added = True
                    break
            if not connected:
                edges_output.append(edge_in)
        if len(edges_output) == len(edges_input):
            return edges_output
        else:
            edges_input = edges_output
