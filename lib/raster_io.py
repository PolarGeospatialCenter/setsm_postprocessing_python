
# Version 3.1; Erik Husby; Polar Geospatial Center, University of Minnesota; 2019


from __future__ import division
import math
import os
import sys
import traceback
from warnings import warn

import numpy as np
from osgeo import gdal_array, gdalconst
from osgeo import gdal, ogr, osr

gdal.UseExceptions()


class RasterIOError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class UnsupportedDataTypeError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class UnsupportedMethodError(Exception):
    def __init__(self, msg=""):
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


def openRaster(file_or_ds, target_EPSG=None):
    """
    Open a raster image as a GDAL dataset object.

    Parameters
    ----------
    file_or_ds : str (file path) or osgeo.gdal.Dataset
        File path of the raster image to open as a GDAL dataset object,
        or the GDAL dataset itself.

    Returns
    -------
    ds : osgeo.gdal.Dataset
        The raster image as a GDAL dataset.

    Notes
    -----
    If `rasterFile_or_ds` is a GDAL dataset,
    it is returned without modification.

    """
    ds = None
    if type(file_or_ds) == gdal.Dataset:
        ds = file_or_ds
    elif isinstance(file_or_ds, str):
        if not os.path.isfile(file_or_ds):
            raise RasterIOError("No such rasterFile: '{}'".format(file_or_ds))
        try:
            ds = gdal.Open(file_or_ds, gdal.GA_ReadOnly)
        except RuntimeError:
            print("RuntimeError when opening file/dataset: {}".format(file_or_ds))
            raise
    else:
        raise InvalidArgumentError("Invalid input type for `file_or_ds`: {}".format(
                                   type(file_or_ds)))

    if target_EPSG is not None:
        target_sr = osr.SpatialReference()
        target_sr.ImportFromEPSG(target_EPSG)
        ds = reprojectGDALDataset(ds, target_sr, 'nearest')

    return ds


def reprojectGDALDataset(ds_in, sr_out, interp_str):
    # FIXME: Finish this function.

    # dtype_gdal, promote_dtype = dtype_np2gdal(Z.dtype)
    # if promote_dtype is not None:
    #     Z = Z.astype(promote_dtype)

    interp_gdal = interp_str2gdal(interp_str)

    mem_drv = gdal.GetDriverByName('MEM')

    sr_in = osr.SpatialReference()

    # ds_in = mem_drv.Create('', X.size, Y.size, 1, dtype_gdal)
    # ds_in.SetGeoTransform((X[0], X[1]-X[0], 0,
    #                        Y[0], 0, Y[1]-Y[0]))
    # ds_in.GetRasterBand(1).WriteArray(Z)

    ds_out = mem_drv.Create('', ds_in.RasterXSize, ds_in.RasterYSize, 1)

    gdal.ReprojectImage(ds_in, ds_out, '', '', interp_gdal)

    return ds_out


def gdalReadAsArraySetsmSceneBand(raster_band, make_nodata_nan=False):
    """Read full GDAL raster band from a SETSM DEM raster into a NumPy array,
    converting data type from scaled integer (Int32) to floating point (Float32)
    if necessary.

    The data type conversion is necessary before working with raw elevation
    values from DEM rasters that are stored in scaled integer format, chiefly
    the `*_dem.tif`, `*_ortho.tif`, and `*_matchtag.tif` rasters from 50cm
    scene DEM results. These rasters are stored in this format with a custom
    LERC & ZSTD compression applied to achive the greatest space savings for
    long term, high data volume storage.

    Rasters that do not have internal 'scale' or 'offset' metadata information
    visible to GDAL will not have their values adjusted, so it should be safe
    to replace all GDAL `ReadAsArray()` calls on SETSM DEM rasters with this
    function.

    Parameters
    ----------
    raster_band : GDALRasterBand
        SETSM DEM raster band to be read.
    make_nodata_nan : boolean, optional
        Convert NoData values in the raster band to NaN in the returned NumPy
        array.

    Returns
    -------
    array_data : numpy.ndarray
        The NumPy array containing adjusted (if necessary) values read from the
        input raster band.
    """
    scale = raster_band.GetScale()
    offset = raster_band.GetOffset()
    if scale is None:
        scale = 1.0
    if offset is None:
        offset = 0.0
    if scale == 1.0 and offset == 0.0:
        array_data = raster_band.ReadAsArray()
        if make_nodata_nan:
            nodata_val = raster_band.GetNoDataValue()
            if nodata_val is not None:
                array_data[array_data == nodata_val] = np.nan
    else:
        if raster_band.DataType != gdalconst.GDT_Int32:
            raise RasterIOError(
                "Expected GDAL raster band with scale!=1.0 or offset!=0.0 to be of Int32 data type"
                " (scaled int LERC_ZSTD-compressed 50cm DEM), but data type is {}".format(
                    gdal.GetDataTypeName(raster_band.DataType)
                )
            )
        if scale == 0.0:
            raise RasterIOError(
                "GDAL raster band has invalid parameters: scale={}, offset={}".format(scale, offset)
            )
        nodata_val = raster_band.GetNoDataValue()
        array_data = raster_band.ReadAsArray(buf_type=gdalconst.GDT_Float32)
        adjust_where = (array_data != nodata_val) if nodata_val is not None else True
        if scale != 1.0:
            np.multiply(array_data, scale, out=array_data, where=adjust_where)
        if offset != 0.0:
            np.add(array_data, offset, out=array_data, where=adjust_where)
        if make_nodata_nan:
            array_nodata = np.logical_not(adjust_where, out=adjust_where)
            array_data[array_nodata] = np.nan
        del adjust_where

    if array_data is None:
        raise RasterIOError("`raster_band.ReadAsArray()` returned None")

    return array_data


def getCornerCoords(gt, shape):
    """
    Retrieve the georeferenced corner coordinates of a raster image.

    The corner coordinates of the raster are calculated from
    the rasters's geometric transformation specifications and
    the dimensions of the raster.

    Parameters
    ----------
    gt : numeric tuple `(top_left_x, dx_x, dx_y, top_left_y, dy_x, dy_y)`
        The affine geometric transformation ("geotransform" or "geo_trans")
        describing the relationship between pixel coordinates and
        georeferenced coordinates.
        Pixel coordinates start at `(0, 0)` [row, col] for the top left pixel
        in the raster image, increasing down rows and right across columns.
        Georeferenced coordinates `(x_geo, y_geo)` are calculated for pixels
        in the image by the pixel coordinates `(pix_row, pix_col)` as follows:
        `x_geo = top_left_x + pix_row*dx_x + pix_col*dx_y`
        `y_geo = top_left_y + pix_row*dy_x + pix_col*dy_y`
    shape : tuple of positive int, 2 elements
        Dimensions of the raster image in (num_rows, num_cols) format.

    Returns
    -------
    corner_coords : ndarray (5, 2)
        Georeferenced corner coordinates of the raster image,
        in (x, y) coordinate pairs, starting and ending at the
        top left corner, clockwise.

    """
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


def coordsToWkt(point_coords):
    """
    Retrieve a WKT polygon representation of an ordered list of
    point coordinates.

    Parameters
    ----------
    point_coords : 2D sequence of floats/ints like ndarray
                   of shape (npoints, ndim)
        Ordered list of points, each represented by a list of
        coordinates that define its position in space.

    Returns
    -------
    wkt : str
        WKT polygon representation of `point_coords`.

    """
    return 'POLYGON (({}))'.format(
        ','.join([" ".join([str(c) for c in xy]) for xy in point_coords])
    )


def wktToCoords(wkt):
    """
    Create an array of point coordinates from a WKT polygon string.

    Parameters
    ----------
    wkt : str
        WKT polygon representation of points with coordinate data
        to be extracted.

    Returns
    -------
    point_coords : ndarray of shape (npoints, ndim)
        Ordered list of point coordinates extracted from `wkt`.

    """
    coords_list = eval(
        wkt.replace('POLYGON ','').replace('(','[').replace(')',']').replace(',','],[').replace(' ',',')
    )
    return np.array(coords_list)


def extractRasterData(rasterFile_or_ds, *params):
    """
    Extract information from a single-band raster image file.

    Parameters
    ----------
    rasterFile_or_ds : str (file path) or osgeo.gdal.Dataset
        File path of the raster image to open as a GDAL dataset object,
        or the GDAL dataset itself.
    params : str
        Names of parameters to be extracted from the raster dataset.
        'array'/'z' ------ matrix of image pixel values as ndarray (2D)
        'shape'----------- pixel shape of image as tuple (nrows, ncols)
        'x' -------------- georeferenced grid coordinates corresponding to
                           each column of pixels in image as ndarray (1D)
        'y' -------------- georeferenced grid coordinates corresponding to
                           each row of pixels in image as ndarray (1D)
        'dx' ------------- x length of each pixel in georeferenced pixel-grid coordinates,
                           corresponding to x[1] - x[0] from 'x' param (dx may be negative)
        'dy' ------------- y length of each pixel in georeferenced pixel-grid coordinates,
                           corresponding to y[1] - y[0] from 'y' param (dy may be negative)
        'res' ------------ (absolute) resolution of square pixels in image
                           (NaN if pixels are not square)
        'geo_trans' ------ affine geometric transformation
                           (see documentation for `getCornerCoords`)
        'corner_coords' -- georeferenced corner coordinates of image extent
                           (see documentation for `getCornerCoords`)
        'proj_ref' ------- projection definition string in OpenGIS WKT format
                           (None if projection definition is not available)
        'spat_ref' ------- spatial reference as osgeo.osr.SpatialReference object
                           (None if spatial reference is not available)
        'geom' ----------- polygon geometry of image extent as osgeo.ogr.Geometry object
        'geom_sr' -------- polygon geometry of image extent as osgeo.ogr.Geometry object
                           with spatial reference assigned (if available)
        'nodata_val' ----- pixel value that should be interpreted as "No Data"
        'dtype_val' ------ GDAL type code for numeric data type of pixel values (integer)
        'dtype_str' ------ GDAL type name for numeric data type of pixel values (string)

    Returns
    -------
    value_list : list
        List of parameter data with length equal to the number
        of parameter name arguments given in the function call.
        The order of returned parameter data corresponds directly to
        the order of the parameter name arguments.
        If only one parameter name argument is provided, the single
        datum is returned itself, not in a list.

    Examples
    --------
    >>> f = 'my_raster.tif'
    >>> image_data, resolution = extractRasterData(f, 'array', 'res')
    >>> resolution
    2
    >>> extractRasterData(f, 'dy')
    -2

    """
    ds = openRaster(rasterFile_or_ds)
    pset = set(params)
    invalid_pnames = pset.difference({'ds', 'shape', 'z', 'array', 'x', 'y',
                                      'dx', 'dy', 'res', 'geo_trans', 'corner_coords',
                                      'proj_ref', 'spat_ref', 'geom', 'geom_sr',
                                      'nodata_val', 'dtype_val', 'dtype_str'})
    if invalid_pnames:
        raise InvalidArgumentError("Invalid parameter(s) for extraction: {}".format(invalid_pnames))

    if pset.intersection({'z', 'array', 'nodata_val', 'dtype_val', 'dtype_str'}):
        band = ds.GetRasterBand(1)
    if pset.intersection({'z', 'array'}):
        try:
            array_data = gdalReadAsArraySetsmSceneBand(band)
        except RasterIOError as e:
            traceback.print_exc()
            print("Error reading raster: {}".format(rasterFile_or_ds))
            raise
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
    if pset.intersection({'nodata_val'}):
        nodata_val = band.GetNoDataValue()
    if pset.intersection({'dtype_val', 'dtype_str'}):
        dtype_val = band.DataType
    if pset.intersection({'dtype_str'}):
        dtype_str = gdal.GetDataTypeName(dtype_val)

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
            value = abs(geo_trans[1])
        elif pname == 'dy':
            value = abs(geo_trans[5])
        elif pname == 'res':
            value = abs(geo_trans[1]) if abs(geo_trans[1]) == abs(geo_trans[5]) else np.nan
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
                warn("Spatial reference could not be extracted from raster dataset, "
                     "so extracted geometry has not been assigned a spatial reference.")
        elif pname == 'nodata_val':
            value = nodata_val
        elif pname == 'dtype_val':
            value = dtype_val
        elif pname == 'dtype_str':
            value = dtype_str
        value_list.append(value)

    if len(value_list) == 1:
        value_list = value_list[0]
    return value_list


# Legacy; Retained for a visual aid of equivalences between NumPy and GDAL data types.
# Use gdal_array.NumericTypeCodeToGDALTypeCode to convert from NumPy to GDAL data type.
def dtype_np2gdal_old(dtype_in, form_out='gdal', force_conversion=False):
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
                print(errmsg_unsupported_dtype)
            else:
                raise UnsupportedDataTypeError(errmsg_unsupported_dtype)
        dtype_out = dtype_tup[1]
    elif form_out.lower() == 'numpy':
        dtype_out = dtype_tup[0]
    else:
        raise UnsupportedDataTypeError("The following output data type format is not supported: '{}'".format(form_out))

    return dtype_out


def dtype_np2gdal(dtype_np):
    # TODO: Write docstring.

    if dtype_np == np.bool:
        promote_dtype = np.uint8
    elif dtype_np == np.int8:
        promote_dtype = np.int16
    elif dtype_np == np.float16:
        promote_dtype = np.float32
    else:
        promote_dtype = None

    if promote_dtype is not None:
        warn("NumPy array data type ({}) does not have equivalent GDAL data type and is not "
             "supported, but can be safely promoted to {}".format(dtype_np, promote_dtype(1).dtype))
        dtype_np = promote_dtype

    dtype_gdal = gdal_array.NumericTypeCodeToGDALTypeCode(dtype_np)
    if dtype_gdal is None:
        raise InvalidArgumentError("NumPy array data type ({}) does not have equivalent "
                                   "GDAL data type and is not supported".format(dtype_np))

    return dtype_gdal, promote_dtype


def interp_str2gdal(interp_str):
    # TODO: Write docstring.

    interp_choices = ('nearest', 'linear', 'cubic', 'spline', 'lanczos', 'average', 'mode')

    interp_dict = {
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

    if interp_str not in interp_dict:
        raise UnsupportedMethodError("`interp` must be one of {}, but was '{}'".format(interp_choices, interp_str))

    return interp_dict[interp_str]


def saveArrayAsTiff(array, dest,
                    X=None, Y=None, proj_ref=None, geotrans_rot_tup=(0, 0),
                    nodata_val='like_raster', dtype_out=None, nbits=None,
                    co_args='compress', co_predictor=None,
                    like_raster=None):
    """
    Save a NumPy 2D array as a single-band raster image in GeoTiff format.

    Parameters
    ----------
    array : ndarray, 2D
        Array containing the values of pixels to be saved in the image,
        one value per pixel.
    dest : str (file path)
        File path where the raster image will be saved.
        If a file already exists at this path, it will be overwritten.
    X : None or (ndarray, 1D)
        Grid coordinates corresponding to all columns in the raster image,
        from left to right, such that `X[j]` specifies the x-coordinate for
        all pixels in `array[:, j]`.
        If None, `like_raster` must be provided.
    Y : None or (ndarray, 1D)
        Grid coordinates corresponding to all rows in the raster image,
        from top to bottom, such that `Y[i]` specifies the y-coordinate for
        all pixels in `array[i, :]`
        If None, `like_raster` must be provided.
    proj_ref : None, str (WKT or Proj4), or osr.SpatialReference
        Projection reference of the raster image to be saved, specified as
        either a WKT/Proj4 string or an osr.SpatialReference object.
        If None, `like_raster` must be provided.
    geotrans_rot_tup : None or tuple (2 floats)
        The third and fifth elements of the geometric transformation tuple
        that specify rotation from north-up of the raster image to be saved.
        If a north-up output is desired, let both elements be zero.
        See documentation for `getCornerCoords` for more information on the
        geometric transformation tuple.
        If None, `like_raster` must be provided.
    nodata_val : 'like_raster', None, or int/float
        Non-NaN value in `array` that will be classified as "no data" in the
        output raster image.
        If 'like_raster', allow this value to be set equal to the nodata value
        of `like_raster`.
    dtype_out : data type as str (e.g. 'uint16'), NumPy data type
                (e.g. np.uint16), or numpy.dtype object (e.g. from arr.dtype)
        Numeric type of values in the output raster image.
        If 'n-bit', write output raster image in an unsigned integer GDAL
        data type with ['NBITS=n'] option in driver, where n is set to `nbits`
        if `nbits` is not None. If `nbits` is None, n is calculated to be only
        as large as necessary to capture the maximum value of `array`, and the
        output array data type is unsigned integer of minimal bitdepth.
    nbits : None or 1 <= int <= 32
        Only applies when `dtype_out='nbits'`.
    co_args : None, 'compress', or list of '[ARG_NAME]=[ARG_VALUE]' strings
        Creation Option arguments to pass to the `Create` method of the GDAL
        Geotiff driver that instantiates the output raster dataset.
        If 'compress', the following default arguments are used:
          'TILED=YES'
          'BIGTIFF=YES'
          'COMPRESS=LZW'
          'PREDICTOR=X' (where `X` is automatically derived from input and
                         output array data types)
        The 'NBITS=X' argument may not be used -- that is set by the `nbits`
        argument for this function.
        A list of Creation Option arguments may be found here: [1].
    co_predictor : None, or int in range [1,3]
        GeoTIFF Creation Option PREDICTOR value to override the default that
        would be automatically set when `co_args` is 'compress'.
        Has no effect if `co_args` is not 'compress'.
    like_raster : None, str (file path), or osgeo.gdal.Dataset
        File path or GDAL dataset for a raster image of identical dimensions,
        geographic location/extent, spatial reference, and nodata value as
        the raster image that will be saved.
        If provided, `X`, `Y`, `proj_ref`, and `geotrans_rot_tup` should not
        be provided, as these metrics will be taken from the like raster.

    Returns
    -------
    None

    Notes
    -----
    The OSGeo `gdal_translate` program [1] must be callable by name
    from the current working directory at the time this function is called.

    References
    ----------
    .. [1] https://www.gdal.org/frmt_gtiff.html

    """
    spat_ref = None
    projstr_wkt = None
    projstr_proj4 = None
    if proj_ref is None:
        pass
    elif type(proj_ref) == osr.SpatialReference:
        spat_ref = proj_ref
    elif isinstance(proj_ref, str):
        spat_ref = osr.SpatialReference()
        if proj_ref.lstrip().startswith('PROJCS'):
            projstr_wkt = proj_ref
            spat_ref.ImportFromWkt(projstr_wkt)
        elif proj_ref.lstrip().startswith('+proj='):
            projstr_proj4 = proj_ref
            spat_ref.ImportFromProj4(projstr_proj4)
        else:
            raise InvalidArgumentError("`proj_ref` of string type has unknown format: '{}'".format(proj_ref))
    else:
        raise InvalidArgumentError("`proj_ref` must be a string or osr.SpatialReference object, "
                                   "but was of type {}".format(type(proj_ref)))

    dtype_is_nbits = (dtype_out is not None and type(dtype_out) is str and dtype_out == 'nbits')

    if co_args is not None and co_args != 'compress':
        if type(co_args) != list:
            raise InvalidArgumentError("`co_args` must be a list of strings, but was {}".format(co_args))
        if dtype_is_nbits:
            for arg in co_args:
                if arg.startswith('NBITS='):
                    raise InvalidArgumentError("`co_args` cannot include 'NBITS=X' argument. "
                                               "Please use this function's `nbits` argument.")

    shape = array.shape
    dtype_out_gdal = None
    if like_raster is not None:
        ds_like = openRaster(like_raster)
        if shape[0] != ds_like.RasterYSize or shape[1] != ds_like.RasterXSize:
            raise InvalidArgumentError("Shape of `like_rasterFile` '{}' ({}, {}) does not match "
                                       "the shape of `array` {}".format(
                like_raster, ds_like.RasterYSize, ds_like.RasterXSize, shape)
            )
        geo_trans = extractRasterData(ds_like, 'geo_trans')
        if proj_ref is None:
            spat_ref = extractRasterData(ds_like, 'spat_ref')
        if nodata_val == 'like_raster':
            nodata_val = extractRasterData(ds_like, 'nodata_val')
        if dtype_out is None:
            dtype_out_gdal = extractRasterData(ds_like, 'dtype_val')
    else:
        if shape[0] != Y.size or shape[1] != X.size:
            raise InvalidArgumentError("Lengths of [`Y`, `X`] grid coordinates ({}, {}) do not match "
                                       "the shape of `array` ({})".format(Y.size, X.size, shape))
        geo_trans = (X[0], X[1]-X[0], geotrans_rot_tup[0],
                     Y[0], geotrans_rot_tup[1], Y[1]-Y[0])

    if nodata_val == 'like_raster':
        nodata_val = None

    dtype_in_np = array.dtype

    dtype_in_general = None
    dtype_out_general = None

    if dtype_in_np == bool:
        dtype_in_general = 'bool'
    elif np.issubdtype(dtype_in_np, np.integer):
        dtype_in_general = 'int'
    elif np.issubdtype(dtype_in_np, np.floating):
        dtype_in_general = 'float'

    if dtype_out is not None:
        if dtype_is_nbits:
            if nbits is None:
                nbits = int(math.floor(math.log(float(max(1, np.max(array))), 2)) + 1)
            elif type(nbits) != int or nbits < 1:
                raise InvalidArgumentError("`nbits` must be an integer in the range [1,32]")
            if nbits <= 8:
                dtype_out_gdal = gdal.GDT_Byte
            elif nbits <= 16:
                dtype_out_gdal = gdal.GDT_UInt16
            elif nbits <= 32:
                dtype_out_gdal = gdal.GDT_UInt32
            else:
                raise InvalidArgumentError("Output array requires {} bits of precision, "
                                           "but GDAL supports a maximum of 32 bits")
            dtype_out_general = 'int'
        else:
            if type(dtype_out) is str:
                dtype_out = eval('np.{}'.format(dtype_out.lower()))
            dtype_out_gdal = gdal_array.NumericTypeCodeToGDALTypeCode(dtype_out)
            if dtype_out_gdal is None:
                raise InvalidArgumentError("Output array data type ({}) does not have equivalent "
                                           "GDAL data type and is not supported".format(dtype_out))
            if np.issubdtype(dtype_out, np.integer):
                dtype_out_general = 'int'
            elif np.issubdtype(dtype_out, np.floating):
                dtype_out_general = 'float'

    dtype_in_gdal, promote_dtype = dtype_np2gdal(dtype_in_np)
    if promote_dtype is not None:
        array = array.astype(promote_dtype)
        dtype_in_np = promote_dtype(1).dtype

    if dtype_out_general is None:
        if np.issubdtype(dtype_in_np, np.integer):
            dtype_out_general = 'int'
        elif np.issubdtype(dtype_in_np, np.floating):
            dtype_out_general = 'float'

    if dtype_out is not None:
        if dtype_is_nbits:
            if not np.issubdtype(dtype_in_np, np.unsignedinteger):
                warn("Input array data type ({}) is not unsigned and may be incorrectly saved "
                     "with n-bit precision".format(dtype_in_np))
        elif dtype_in_np != dtype_out:
            warn("Input array NumPy data type ({}) differs from output "
                 "NumPy data type ({})".format(dtype_in_np, dtype_out(1).dtype))
    elif dtype_out_gdal is not None and dtype_out_gdal != dtype_in_gdal:
        warn("Input array GDAL data type ({}) differs from output "
             "GDAL data type ({})".format(gdal.GetDataTypeName(dtype_in_gdal),
                                          gdal.GetDataTypeName(dtype_out_gdal)))
    if dtype_out_gdal is None:
        dtype_out_gdal = dtype_in_gdal

    if co_args == 'compress':
        if co_predictor is not None:
            compress_predictor = co_predictor
        elif dtype_in_general == 'bool':
            compress_predictor = 1
        elif dtype_out_general == 'int':
        # elif dtype_out_general == 'int' or dtype_in_general == 'int':
            compress_predictor = 2
        elif dtype_out_general == 'float':
            compress_predictor = 3
        else:
            compress_predictor = 1

    sys.stdout.write("Saving Geotiff {} ...".format(dest))
    sys.stdout.flush()

    # Create the output raster dataset in memory.
    if co_args is None:
        co_args = []
    if co_args == 'compress':
        co_args = []
        co_args.extend(['TILED=YES'])         # Force creation of tiled TIFF files.
        co_args.extend(['BIGTIFF=YES'])       # Will create BigTIFF
                                              # if the resulting file *might* exceed 4GB.
        co_args.extend(['COMPRESS=LZW'])      # Do LZW compression on output image.
        co_args.extend(['PREDICTOR={}'.format(compress_predictor)])
        # co_args.extend(['BLOCKXSIZE=256'])
        # co_args.extend(['BLOCKYSIZE=256'])
    if dtype_is_nbits:
        co_args.extend(['NBITS={}'.format(nbits)])

    if spat_ref is not None:
        if projstr_wkt is None:
            projstr_wkt = spat_ref.ExportToWkt()
        if projstr_proj4 is None:
            projstr_proj4 = spat_ref.ExportToProj4()
    sys.stdout.write(" GDAL data type: {}, NoData value: {}, Creation Options: {}, Projection (Proj4): {} ...".format(
        gdal.GetDataTypeName(dtype_out_gdal), nodata_val, ' '.join(co_args) if co_args else None, projstr_proj4.strip())
    )
    sys.stdout.flush()

    sys.stdout.write(" creating file ...")
    sys.stdout.flush()
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(dest, shape[1], shape[0], 1, dtype_out_gdal, co_args)
    ds_out.SetGeoTransform(geo_trans)
    if projstr_wkt is not None:
        ds_out.SetProjection(projstr_wkt)
    band = ds_out.GetRasterBand(1)
    if nodata_val is not None:
        band.SetNoDataValue(nodata_val)

    sys.stdout.write(" writing array values ...")
    sys.stdout.flush()
    band.WriteArray(array)

    # Write the output raster dataset to disk.
    sys.stdout.write(" finishing file ...")
    sys.stdout.flush()
    ds_out = None  # Dereference dataset to initiate write to disk of intermediate image.
    sys.stdout.write(" done!\n")
    sys.stdout.flush()
