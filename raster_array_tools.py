#!/usr/bin/env python2

# Version 3.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2017


from __future__ import division
import os
from collections import deque
from subprocess import check_call
from traceback import print_exc

import gdal, ogr, osr
import numpy as np
import scipy
from scipy import ndimage
from skimage import morphology as sk_morphology
from skimage.filters.rank import entropy

import test

_outline = open("outline.c", "r").read()
_outline_every1 = open("outline_every1.c", "r").read()


RASTER_PARAMS = ['ds', 'shape', 'z', 'array', 'x', 'y', 'dx', 'dy', 'res', 'geo_trans', 'corner_coords', 'proj_ref', 'spat_ref', 'geom', 'geom_sr']


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


# Legacy; For quick instruction of useful GDAL raster information extraction methods.
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
                print ("WARNING: Spatial reference could not be extracted from raster dataset,"
                       " so extracted geometry has not been assigned a spatial reference.")
        value_list.append(value)

    if len(value_list) == 1:
        value_list = value_list[0]
    return value_list


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
        'bool'      : (np.bool,       gdal.GDT_Byte,     1),  # **GDAL byte is unsigned**
        'int'       : (np.int,        gdal.GDT_Int32,    1),  # :: GDAL doesn't have int64,
        'intc'      : (np.intc,       gdal.GDT_Int32,    1),  # :: so I assume int32
        'intp'      : (np.intp,       gdal.GDT_Int32,    1),  # :: works for these.
        'int8'      : (np.int8,       gdal.GDT_Byte,     1),  # **GDAL byte is unsigned**
        'int16'     : (np.int16,      gdal.GDT_Int16,    1),
        'int32'     : (np.int32,      gdal.GDT_Int32,    1),
        'int64'     : (np.int64,      gdal.GDT_Int32,    0),  # No int64
        'uint8'     : (np.uint8,      gdal.GDT_Byte,     1),
        'uint16'    : (np.uint16,     gdal.GDT_UInt16,   1),
        'uint32'    : (np.uint32,     gdal.GDT_UInt32,   1),
        'uint64'    : (np.uint64,     gdal.GDT_UInt32,   0),  # No uint64
        'float'     : (np.float,      gdal.GDT_Float64,  1),
        'float16'   : (np.float16,    gdal.GDT_Float32,  1),  # No float16
        'float32'   : (np.float32,    gdal.GDT_Float32,  1),
        'float64'   : (np.float64,    gdal.GDT_Float64,  1),
        'complex'   : (np.complex,    gdal.GDT_CFloat64, 1),  # :: Not sure if these
        'complex64' : (np.complex64,  gdal.GDT_CFloat32, 1),  # :: complex lookups
        'complex128': (np.complex128, gdal.GDT_CFloat64, 1),  # :: are correct.
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
        raise UnsupportedMethodError("The following output data type format is not supported: '{}'".format(form_out))

    return dtype_out


def saveArrayAsTiff(array, dest,
                    X=None, Y=None, proj_ref=None, geotrans_rot_tup=(0, 0),
                    like_rasterFile=None,
                    nodataVal=None, dtype_out=None, force_dtype=False, skip_casting=False):
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
    dest_temp = dest.replace('.tif', '_temp.tif')

    dtype_in = str(array.dtype)
    array_out = array
    if dtype_out is not None:
        if dtype_in != dtype_out:
            print "WARNING: Input array data type ({}) differs from output raster data type ({})".format(
                dtype_in, dtype_out)
            if not skip_casting:
                print "Casting array to output data type"
                array_out = array.astype(dtype_np2gdal(dtype_out, form_out='numpy'))
        dtype = dtype_np2gdal(dtype_out, force_conversion=force_dtype)
    else:
        dtype = dtype_np2gdal(dtype_in, force_conversion=force_dtype)

    if proj_ref is not None and type(proj_ref) == osr.SpatialReference:
        proj_ref = proj_ref.ExportToWkt()

    shape = array.shape
    geo_trans = None
    if like_rasterFile is not None:
        ds_like = gdal.Open(like_rasterFile, gdal.GA_ReadOnly)
        if shape[0] != ds_like.RasterYSize or shape[1] != ds_like.RasterXSize:
            raise InvalidArgumentError("Shape of like_rasterFile '{}' ({}, {}) does not match"
                                       " the shape of 'array' to be saved ({})".format(
                like_rasterFile, ds_like.RasterYSize, ds_like.RasterXSize, shape)
            )
        geo_trans = ds_like.GetGeoTransform()
        if proj_ref is None:
            proj_ref = ds_like.GetProjectionRef()
    else:
        if shape[0] != Y.size or shape[1] != X.size:
            raise InvalidArgumentError("Lengths of [Y, X] grid coordinates ({}, {}) do not match"
                                       " the shape of 'array' to be saved ({})".format(Y.size, X.size, shape))
        geo_trans = (X[0], X[1]-X[0], geotrans_rot_tup[0],
                     Y[0], geotrans_rot_tup[1], Y[1]-Y[0])

    # Create and write the output dataset to a temporary file.
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(dest_temp, shape[1], shape[0], 1, dtype)
    ds_out.SetGeoTransform(geo_trans)
    if proj_ref is not None:
        ds_out.SetProjection(proj_ref)
    else:
        print "WARNING: Missing projection reference for saved raster '{}'".format(dest)
    ds_out.GetRasterBand(1).WriteArray(array_out)
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

    dtype = dtype_np2gdal(Z.dtype)
    mem_drv = gdal.GetDriverByName('MEM')

    ds_in = mem_drv.Create('', X.size, Y.size, 1, dtype)
    ds_in.SetGeoTransform((X[0], X[1]-X[0], 0,
                           Y[0], 0, Y[1]-Y[0]))
    ds_in.GetRasterBand(1).WriteArray(Z)

    ds_out = mem_drv.Create('', Xi.size, Yi.size, 1, dtype)
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


def imresize(array, size, method, PILmode=None, gdal_fix='scipy'):
    """
    This function is meant to replicate MATLAB's imresize function.
    Uses either SciPy's scipy.misc.imresize function or GDAL's gdal.ReprojectImage function
    (through a call of the local interp2_gdal method). To use SciPy's function, specify the
    PIL image mode based on the input array's data type ('F' for float32, 'L' for 8-bit, etc.)
    and set gdal_fix=None.
    WARNING: INPUT PIL IMAGE MODES OTHER THAN 'F' WILL CAUSE ARRAY DATA TO BE SCALED TO UINT8
    BEFORE RESIZING, POTENTIALLY DESTROYING ARRAY DATA IN THE PROCESS.

    For particular sets of input/output array shapes (dependent on resizing scale factor),
    the GDAL method can produce results for some interpolation methods that match MATLAB's
    imresize function pixel-for-pixel, with the exception of the last row and last column which
    currently can't be interpolated through the GDAL method (and are instead set to zeros) without
    applying some sort of fix.
    """
    if gdal_fix not in (None, 'pad', 'pad_merge', 'rotate', 'scipy'):
        raise InvalidArgumentError("gdal_fix must be 'pad', 'pad_merge', 'rotate', 'scipy', or None")
    if gdal_fix == 'scipy' and PILmode is None:
        PILmode = 'F'

    # If a percentage or fraction is given for size, round up the x, y pixel
    # sizes for the output array to match MATLAB's imresize function.
    new_shape = size if type(size) == tuple else np.ceil(np.dot(size, array.shape)).astype(int)
    old_array = array
    new_array = None

    if PILmode is not None:
        new_array = scipy.misc.imresize(old_array, new_shape, method, PILmode)
        if 'uint' in str(array.dtype):
            new_array[new_array < 0] = 0

    if PILmode is None or gdal_fix == 'scipy':
        if gdal_fix == 'scipy':
            new_array_edgefix = new_array.astype(array.dtype)

        if gdal_fix in ('pad', 'pad_merge'):
            # Fix Method 1: Pad the right and bottom sides of the array with zeros before resizing,
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
                new_array = imresize(array, size, method, PILmode=None, gdal_fix=None)

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


def conv2(array, kernel, mode='full', method='auto', allow_flipped_processing=True):
    """
    This function is meant to replicate MATLAB's conv2 function.
    Scipy's fftconvolve funciton cannot handle NaN input (it results in all NaN output),
    so we must set all NaN values to zero before performing convolution. In comparison, MATLAB's
    conv2 function takes a sensible approach by letting NaN win out in all calculations involving
    pixels with NaN values in the input array. This results in masking our result array with NaNs
    according to a binary dilation (with a structure of ones, same shape as the provided 'kernel')
    of NaN locations in the input array.
    The larger the kernel size, the closer this function can reproduce MATLAB results.
    """
    rotation_flag = False
    if allow_flipped_processing:
        array, kernel, rotation_flag = rotate_array_if_kernel_has_even_sidelength(array, kernel)

    fixnans_flag = False
    array_nans = np.isnan(array)
    if np.any(array_nans):
        fixnans_flag = True
        array[array_nans] = 0
    else:
        del array_nans

    if method == 'auto':
        method = scipy.signal.choose_conv_method(array, kernel, mode)
    result = scipy.signal.convolve(array, kernel, mode, method)
    if method != 'direct':
        result[(-1.0e-12 < result) & (result < 10.0e-12)] = 0

    if fixnans_flag:
        result[imdilate_binary(array_nans, np.ones(kernel.shape), allow_flipped_processing=False)] = np.nan
        # Return the input array to its original state.
        array[array_nans] = np.nan

    return fix_array_if_rotation_was_applied(result, rotation_flag)


def moving_average(array, kernel_size, kernel_binary=None, mode='same', allow_flipped_processing=True):
    """
    Given an input array of any type, returns an array of the same size with each pixel containing
    the average of the surrounding [kernel_size x kernel_size] neighborhood (or a that of a custom
    boolean neighborhood specified through input "kernel").
    Arguments "mode" and allow_rotation are forwarded through to the convolution function.
    """
    if kernel_binary is None:
        return conv2(array, np.ones((kernel_size, kernel_size))/(kernel_size**2),
                     mode=mode, allow_flipped_processing=allow_flipped_processing)
    else:
        return conv2(array, kernel_binary/np.sum(kernel_binary),
                     mode=mode, allow_flipped_processing=allow_flipped_processing)


def imerode_binary(array, structure, allow_flipped_processing=True):
    """
    This function is meant to copy the binary erosion functionality of MATLAB's imerode function.
    It is assumed that both inputs 'array' and 'structure' are binary arrays.

    NOTE: The following applies when one of the commented-out SciPy/scikit-image "binary_erosion"
    methods are in use, but NOT when the SciPy convolution method is in use.

    When at least one of the structuring array's sides has an even length, we rotate the input array
    180 degrees before performing erosion, then rotate it back into place before returning the result.
    This correction works because imerode, like most MATLAB functions that work on arrays, slides the
    structuring window first down a column, then moves from left to right across rows -- this is
    opposite of most Python functions, which move first left to right across a row, then down columns.
    """
    rotation_flag = False
    # if allow_flipped_processing:
    #     array, structure, rotation_flag = rotate_array_if_kernel_has_even_sidelength(array, structure)
    # result = ndimage.binary_erosion(array, structure, border_value=1)
    # result = sk_morphology.binary_erosion(array, structure)
    result = (scipy.signal.convolve(array, structure, mode='same', method='auto')
              > (np.count_nonzero(structure) - 0.5))
    return fix_array_if_rotation_was_applied(result, rotation_flag)


def imdilate_binary(array, structure, allow_flipped_processing=True):
    """
    This function is meant to copy the binary dilation functionality of MATLAB's imdilate function.
    It is assumed that both inputs 'array' and 'structure' are binary arrays.

    NOTE: The following applies when one of the commented-out SciPy/scikit-image "binary_dilation"
    methods are in use, but NOT when the SciPy convolution method is in use.

    When at least one of the structuring array's sides has an even length, we rotate the input array
    180 degrees before performing dilation, then rotate it back into place before returning the result.
    This correction works because imdilate, like most MATLAB functions that work on arrays, slides the
    structuring window first down a column, then moves from left to right across rows -- this is
    opposite of most Python functions, which move first left to right across a row, then down columns.
    """
    rotation_flag = False
    # if allow_flipped_processing:
    #     array, structure, rotation_flag = rotate_array_if_kernel_has_even_sidelength(array, structure)
    # result = ndimage.binary_dilation(array, structure, border_value=0)
    # result = sk_morphology.binary_dilation(array, structure)
    result = (scipy.signal.convolve(array, np.rot90(structure, 2), mode='same', method='auto')
              > 0.5)
    return fix_array_if_rotation_was_applied(result, rotation_flag)


def bwareaopen(binary_array, size_tolerance, connectivity=8, in_place=False):
    # TODO: Write docstring.
    return sk_morphology.remove_small_objects(binary_array, size_tolerance, connectivity/4, in_place)


def bwboundaries_array(array, side='inner', connectivity=8, noholes=False, matlab=True):
    """
    This function is meant to replicate MATLAB's bwboundaries function.
    Returns a boolean array of the same shape as the input array
    with pixels on the boundary of [nonzero values in the input array] set to 1.
    """
    side_choices = ('inner', 'outer')
    connectivity_duo = {4, 8}
    if side not in side_choices:
        raise InvalidArgumentError("'side' must be 'inner' or 'outer'")
    if connectivity not in connectivity_duo:
        raise InvalidArgumentError("connectivity must be 4 or 8")

    fn = ndimage.binary_erosion if side == 'inner' else ndimage.binary_dilation

    structure = np.zeros((3, 3), dtype=np.bool)
    if connectivity == 8:
        structure[:, 1] = 1
        structure[1, :] = 1
    elif connectivity == 4:
        structure[:, :] = 1

    if noholes or matlab:
        array_filled = ndimage.binary_fill_holes(array)

    if noholes:
        return (array_filled != fn(array_filled, structure=structure))
    else:
        if matlab:
            return bwboundaries_array((array != fn(array_filled, structure=structure)),
                                      side=side,
                                      connectivity=list(connectivity_duo.difference({connectivity}))[0],
                                      noholes=False, matlab=False)
        else:
            return (array != fn(array, structure=structure))


def entropyfilt(array, kernel):
    # TODO: Write docstring.
    reduced_array = None
    if array.dtype in (np.uint8, np.int8, np.bool):
        reduced_array = array
    else:
        array_dtype_float = True if 'float' in str(array.dtype) else False
        array_dtype_max = np.finfo(array.dtype) if array_dtype_float else np.iinfo(array.dtype)
        if not array_dtype_float:
            reduced_array = array.astype(np.float32)
        reduced_array = np.round(reduced_array / array_dtype_max * np.iinfo(np.uint8).max).astype(np.uint8)
    return entropy(reduced_array, kernel)


def getDataDensityMap(array, kernel_size=11):
    # TODO: Write docstring.
    return moving_average(array, kernel_size)


def getWindow(array, window_shape, x_y_tup, one_based_index=True):
    # TODO: Write docstring.
    # FIXME: Do error checking.
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
        array_filled = ndimage.morphology.binary_fill_holes(array_data)
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
        raise UnsupportedMethodError("method='{}' in call to raster_array_tools.getFPvertices".format(method))

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
    data_boundary = (array_filled != ndimage.binary_erosion(array_filled))

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
        data_array = ndimage.morphology.binary_fill_holes(data_array)

    if erode:
        # Erode data regions to eliminate *any* data spurs (possible 1 or 2 pixel-
        # width fingers that stick out from data clusters in the original array).
        # This should make the tracing of data boundaries more efficient since
        # the rings that are traced will be more substantial.
        data_array = ndimage.binary_erosion(data_array, structure=np.ones((3, 3)))
        if ~np.any(data_array):
            # No data clusters large enough to have a traceable boundary exist.
            return None
        # Reverse the erosion process to retrieve a data array that does not
        # contain data spurs, yet remains true to data coverage.
        data_array = ndimage.binary_dilation(data_array, structure=np.ones((3, 3)))

    if BBS:
        # Bi-directional Boundary Skewing
        # To represent data coverage fairly for most data pixels in the raster image,
        # the right and bottom edges of all data boundaries must grow by one pixel so
        # that their full extents may be recorded upon grid coordinate lookup of data
        # boundary nodes after each boundary ring is traced.
        print "Performing Bi-directional Boundary Skewing"
        outer_boundary = (
            ndimage.binary_dilation(data_array, structure=np.ones((3, 3))) != data_array
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
    data_boundary = (data_array != ndimage.binary_erosion(data_array))

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


def connectEdges(edges_array):
    """
    A helper function for connectEdges_inplace.
    Takes a 2D NumPy array (or some other 2D object with iterable sub-objects)
    and creates a 2D list of edge lists from it to pass to connectEdges_inplace.
    The input list is NOT modified in the process.
    """
    edges_list = [list(e) for e in edges_array]
    return connectEdges_inplace(edges_list)


def connectEdges_inplace(edges_list):
    """
    Takes a 2D list of edge lists, where points are identified by vertex number,
    and recursively connects them by linking edges with endpoints that have
    matching vertex numbers.
    If the edges can be connected to form a single ring, it is returned as a
    1D list of vertices. If multiple rings exist, they are returned as a
    2D list of their vertices as separate lists.
    The input list is modified in the process.
    """
    input_edgeNum = len(edges_list)
    if input_edgeNum < 2:
        # Return the single ring.
        return list(edges_list[0])

    # print "Connecting {} edges".format(input_edgeNum)  ## For testing purposes.

    lines = [deque(edges_list[0])]
    edges_list = edges_list[1:]
    i = 0
    for e in edges_list:
        start = i
        endpoints = (e[0], e[-1])
        connected = False

        if lines[i][0] in endpoints:
            if lines[i][0] == endpoints[1]:
                del e[-1]
                e.reverse()
                lines[i].extendleft(e)
            else:
                del e[0]
                lines[i].extendleft(e)
            connected = True
        elif lines[i][-1] in endpoints:
            if lines[i][-1] == endpoints[0]:
                del e[0]
                lines[i].extend(e)
            else:
                del e[-1]
                e.reverse()
                lines[i].extend(e)
            connected = True
        i += 1
        while not connected and i != start:
            try:
                if lines[i][0] in endpoints:
                    if lines[i][0] == endpoints[1]:
                        del e[-1]
                        e.reverse()
                        lines[i].extendleft(e)
                    else:
                        del e[0]
                        lines[i].extendleft(e)
                    connected = True
                elif lines[i][-1] in endpoints:
                    if lines[i][-1] == endpoints[0]:
                        del e[0]
                        lines[i].extend(e)
                    else:
                        del e[-1]
                        e.reverse()
                        lines[i].extend(e)
                    connected = True
                i += 1
            except IndexError:
                i = 0
        if not connected:
            lines.append(deque(e))
        else:
            i -= 1

    if len(lines) < input_edgeNum:
        # Try another pass to connect edges.
        return connectEdges(lines)
    else:
        # More than one ring exists.
        return [list(deq) for deq in lines]
