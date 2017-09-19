# Version 3.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2017

from __future__ import division
import numbers
import os
import sys
from collections import deque
from operator import itemgetter
from subprocess import check_call

import gdal, ogr, osgeo, osr
import numpy as np
import scipy
from scipy import ndimage

_outline = open("outline.c", "r").read()
_outline_every1 = open("outline_every1.c", "r").read()


PROJREF_POLAR_STEREO = """PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-70],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]"""
RASTER_DEFAULT_PROJREF = PROJREF_POLAR_STEREO


class RasterIOError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

class UnsupportedDataTypeError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

class InvalidArgumentError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

class UnsupportedMethodError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)



#############
# Raster IO #
#############


class Raster:
    """
    *** NOTE THAT ONLY 'NORTH-UP' GEOTIFF IMAGES ARE FULLY SUPPORTED AT THIS TIME ***

    Contains methods to extract pixel data, geotransform, projection, corner coordinates, geometry,
    and other associated information from a raster image, built on the framework provided by GDAL's
    GDALDataset class.
    Additionally, 'smart' getter and setter methods are provided for all data members listed in the
    class initialization (data members are referred to as a raster's 'parameters' hereafter) that
    make it possible to store and modify the values of useful parameters while maintaining a
    self-consistent dataset.

    A Raster instance starts with all parameters set to None, except for those whose names are
    provided to the initialization call as additional arguments beyond the first.
    As for the first argument, if it is a path to a valid raster file (or equivalently, if it is an
    osgeo.gdal.Dataset object), all values of those parameters which are to be set will be extracted
    directly from the provided raster dataset. If 'ds' is not included in that list (or equivalently,
    'all' and 'no-ds' are included), the class will not keep a reference to the raster dataset in
    its 'ds' parameter after initialization is complete.
    If the first argument is instead None, those parameters which are to be set will be set to their
    respective default values (as retrieved from the getter methods mentioned later).

    After initialization, setting individual parameters should be done via the Raster.set_param()
    method. Since changing the value of one parameter of a raster image (such as the 'x' array of
    horizontal grid coordinates) may affect the (x-)size of the image in pixel SHAPE, the RESOLUTION
    of the image pixels (in the x-direction, dx), and the geographic EXTENT of the image (in its
    x-coordinates), to maintain a self-consistent dataset any modifications should be propagated to
    those parameters that are based on the same core values of SHAPE, RESOLUTION, or EXTENT.
    This is done by default, but may be turned off by passing the 'prop' keyword argument as False.
    Core values for each property -- SHAPE ('shape'), RESOLUTION ('dx', 'dy', 'res', the dx and dy
    parts of 'geo_trans'), EXTENT (the xmin and ymax parts of 'geo_trans') -- may be set (remember,
    every parameter is initialized to None unless specifically set) automatically by passing the
    'set_core' keyword argument as True when using Raster.set_param() to set a higher-level
    (non-core) parameter.
    Furthermore...
    When setting a parameter that directly sets a value(s) in only one of the three Raster property
    domains SHAPE, RESOLUTION, or EXTENT, it must be determined which of the other two properties
    will be held constant (or as close as possible to constant in the case of changing SHAPE/EXTENT
    when RESOLUTION is held constant). By default, EXTENT is preserved when setting SHAPE/RESOLUTION
    and RESOLUTION is preserved when setting EXTENT. This behavior may be changed when setting any
    applicable parameter by passing the 'hold' keyword argument as the name of the property you wish
    to preserve ('shape', 'res', or 'extent').

    Setting a parameter with Raster.set_param() in 'default mode' (by passing None as the 'value'
    argument with the 'set_default' keyword argument set to True) will attempt to use the values of
    other already-set parameters to determine a value for the new parameter. This is done to try to
    keep the Raster in a self-consistent state. Getter methods for each parameter work to accomplish
    this task, and may be used by themselves to extract wanted information from the Raster without
    setting any unneeded parameters.
    NOTE: These getter methods will give no warning if there are inconsistencies among the parameter
    values, and should be used at the risk of the programmer.

    Since no copying is done when setting parameters to values that are mutable objects, multiple
    references may exist in a program that point to the value of a Raster parameter and one must be
    careful. However, since it is highly beneficial to be able to make direct modifications to such
    items (without copying, modifying, and passing the result into Raster.set_param() over and over),
    calling Raster.prop_param() after making direct modifications to the value of a parameter will
    essentially propagate those changes to other parameters in the Raster by forcing the getter
    methods to ignore the modified parameter when looking for values that should be held constant
    through the propagation.

    At this time, changes are not propagated through to the pixel data parameter 'z' after z is set.
    """
    def __init__(self, rasterFile_or_ds=None, *set_params):
        self.ds = None
        self.shape = None
        self.z = None
        self.x = None
        self.y = None
        self.dx = None
        self.dy = None
        self.res = None
        self.geo_trans = None
        self.corner_coords = None
        self.proj_ref = None
        self.spat_ref = None
        self.geom = None

        set_params_copy = list(set(set_params))
        if 'no-ds' in set_params_copy:
            set_params_copy.remove('no-ds')

        if rasterFile_or_ds is not None:
            self.set_param('ds', self.open_ds(rasterFile_or_ds))
            if set_params:
                self.extract_and_set(*set_params_copy)
                if 'no-ds' in set_params or \
                   ('ds' not in set_params and 'all' not in set_params):
                    self.ds = None
        elif set_params:
            if 'ds' in set_params:
                raise InvalidArgumentError("ds parameter cannot be set when rasterFile_or_ds"
                                           " argument is None")
            self.set_params(*set_params_copy)


    @staticmethod
    def open_ds(rasterFile_or_ds):
        ds = None
        if type(rasterFile_or_ds) == str:
            ds = gdal.Open(rasterFile_or_ds, gdal.GA_ReadOnly)
        elif type(rasterFile_or_ds) == osgeo.gdal.Dataset:
            ds = rasterFile_or_ds
        else:
            raise InvalidArgumentError("Invalid input type for 'rasterFile_or_ds': {}".format(
                                       type(rasterFile_or_ds)))
        return ds


    def extract_shape(self):
        return (self.ds.RasterYSize, self.ds.RasterXSize) if self.ds is not None else None

    def extract_z(self):
        return self.ds.GetRasterBand(1).ReadAsArray() if self.ds is not None else None

    def extract_geo_trans(self):
        return np.array(self.ds.GetGeoTransform()) if self.ds is not None else None

    def extract_proj_ref(self):
        return self.ds.GetProjectionRef() if self.ds is not None else None


    def wkt(self, corner_coords=None):
        if corner_coords is None:
            corner_coords = self.get_corner_coords()
        return 'POLYGON (({}))'.format(
            ','.join([" ".join([str(c) for c in cc]) for cc in corner_coords])
        )

    def wkt_to_coords(self, wkt):
        eval_str = 'np.array({})'.format(
            wkt.replace('POLYGON ','').replace('(','[').replace(')',']').replace(',','],[').replace(' ',',')
        )
        return eval(eval_str)


    def extract_param(self, pname):
        if self.ds is None:
            raise RasterIOError("Raster must have a raster dataset reference in its 'ds'"
                                " data member before parameters may be extracted")
        value = None

        if pname in ('shape', 'x', 'y', 'corner_coords'):
            shape = self.extract_shape()
        if pname in ('x', 'y', 'dx', 'dy', 'res', 'geo_trans', 'corner_coords'):
            geo_trans = self.extract_geo_trans()
        if pname in ('proj_ref', 'spat_ref'):
            proj_ref = self.extract_proj_ref()

        if pname == 'ds':
            value = self.ds
        elif pname == 'shape':
            value = shape
        elif pname == 'z':
            value = self.extract_z()
        elif pname == 'x':
            value = geo_trans[0] + np.arange(shape[1]) * geo_trans[1]
        elif pname == 'y':
            value = geo_trans[3] + np.arange(shape[0]) * geo_trans[5]
        elif pname == 'dx':
            value = geo_trans[1]
        elif pname == 'dy':
            value = -geo_trans[5]
        elif pname == 'res':
            value = geo_trans[1] if geo_trans[1] == -geo_trans[5] else np.nan
        elif pname == 'geo_trans':
            value = geo_trans
        elif pname == 'corner_coords':
            value = self.get_corner_coords(geo_trans, shape)
        elif pname == 'proj_ref':
            value = proj_ref
        elif pname == 'spat_ref':
            value = osr.SpatialReference(proj_ref) if proj_ref is not None else None
        elif pname == 'geom':
            value = ogr.Geometry(wkt=self.wkt(self.extract_param('corner_coords')))
        elif pname == 'geom_sr':
            value = self.extract_param('geom')
            spat_ref = self.extract_param('spat_ref')
            if spat_ref is not None:
                value.AssignSpatialReference(spat_ref)
            else:
                print ("WARNING: Spatial reference could not be extracted from raster dataset,"
                       " so extracted geometry has not been assigned a spatial reference.")
        else:
            raise InvalidArgumentError("Invalid parameter for extraction: {}".format(pname))

        return value


    def extract_params(self, *params):
        if self.ds is None:
            raise RasterIOError("Raster must have a raster dataset reference in its 'ds'"
                                " data member before parameters may be extracted")
        pset = set(params)
        valid_pnames = vars(self).keys()
        valid_pnames.append('geom_sr')
        invalid_pnames = pset.difference(set(valid_pnames))
        if invalid_pnames:
            raise InvalidArgumentError("Invalid parameter(s) for extraction: {}".format(invalid_pnames))

        if pset.intersection({'shape', 'x', 'y', 'corner_coords', 'geom', 'geom_sr'}):
            shape = self.extract_shape()
        if pset.intersection({'x', 'y', 'dx', 'dy', 'res', 'geo_trans', 'corner_coords', 'geom', 'geom_sr'}):
            geo_trans = self.extract_geo_trans()
        if pset.intersection({'proj_ref', 'spat_ref', 'geom_sr'}):
            proj_ref = self.extract_proj_ref()
        if pset.intersection({'corner_coords', 'geom', 'geom_sr'}):
            corner_coords = self.get_corner_coords(geo_trans, shape)
        if pset.intersection({'spat_ref', 'geom_sr'}):
            spat_ref = osr.SpatialReference(proj_ref) if proj_ref is not None else None
        if pset.intersection({'geom', 'geom_sr'}):
            geom = ogr.Geometry(wkt=self.wkt(corner_coords))

        value_list = []
        for pname in params:
            value = None
            if pname == 'ds':
                value = self.ds
            elif pname == 'shape':
                value = shape
            elif pname == 'z':
                value = self.extract_z()
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

        return value_list


    def set_params(self, *params):
        set_core = False
        params_copy = None
        if 'all' in params:
            params_copy = ('z', 'x', 'y', 'corner_coords', 'spat_ref', 'geom')
            # params_copy = vars(self).keys()
            set_core = True
        params_copy = tuple(set(params_copy))
        for p in params_copy:
            self.set_param(p, set_core=set_core)


    def set_params_and_values(self, *pname_value):
        pnames = list(pname_value[0::2])
        values = pname_value[1::2]
        if len(pnames) != len(values):
            raise InvalidArgumentError("Unequal number of parameter names and parameter values")
        valid_parameters = vars(self).keys()
        for i in range(len(pnames)):
            p = pnames[i]
            if type(p) == str:
                if p in valid_parameters:
                    continue
                elif p == 'geom_sr':
                    pnames[i] = 'geom'
                    continue
            raise InvalidArgumentError("Starting with the first argument, every other argument"
                                       " must be a valid string name of a Raster parameter")
        for i in range(len(pnames)):
            exec('self.{} = values[{}]').format(pnames[i], i)


    def extract_and_set(self, *params):
        self.set_params_and_values(*[a for b in zip(params, self.extract_params(*params)) for a in b])


    def clear_params(self):
        params = vars(self).keys()
        params.remove('ds')
        for p in params:
            exec('self.{} = None'.format(p))


    def get_shape(self, caller_function=None):
        if self.shape is not None:
            return self.shape
        elif self.z is not None:
            return self.z.shape
        elif caller_function == 'get_res':
            return None

        xsize, ysize = None, None
        if self.x is not None:
            xsize = len(self.x)
        if self.y is not None:
            ysize = len(self.y)

        if (xsize is None or ysize is None) and self.corner_coords is not None:
            if xsize is None:
                dx = self.get_res('dx', 'get_shape')
                if not np.isnan(dx):
                    cc_x = self.corner_coords[:, 0]
                    if cc_x[2] is not None and cc_x[0] is not None:
                        xsize = (cc_x[2] - cc_x[0]) / dx
            if ysize is None:
                dy = self.get_res('dy', 'get_shape')
                if not np.isnan(dy):
                    cc_y = self.corner_coords[:, 1]
                    if cc_y[2] is not None and cc_y[0] is not None:
                        ysize = -(cc_y[2] - cc_y[0]) / dy

        if xsize is None:
            xsize = 0
        if ysize is None:
            ysize = 0
        return ysize, xsize


    def get_res(self, param='res', caller_function=None):
        if param not in ('dx', 'dy', 'res'):
            raise InvalidArgumentError("Invalid 'param' argument: {}".format(param))
        value = eval('self.{}'.format(param))
        if value is not None:
            return value

        if param in ('dx', 'dy'):
            if self.res is not None and not np.isnan(self.res):
                value = self.res

            elif param == 'dx':
                if self.geo_trans is not None:
                    value = self.geo_trans[1]
                elif self.corner_coords is not None and caller_function != 'get_shape':
                    cc_x = self.corner_coords[:, 0]
                    shape = self.get_shape('get_res')
                    if shape is not None:
                        xsize = shape[1]
                        value = np.nan if xsize == 0 else (cc_x[2] - cc_x[0]) / xsize
                elif self.x is not None:
                    value = (self.x[1] - self.x[0]) if len(self.x) > 1 else np.nan

            elif param == 'dy':
                if self.geo_trans is not None:
                    value = -self.geo_trans[5]
                elif self.corner_coords is not None and caller_function != 'get_shape':
                    cc_y = self.corner_coords[:, 1]
                    shape = self.get_shape('get_res')
                    if shape is not None:
                        ysize = shape[0]
                        value = np.nan if ysize == 0 else -(cc_y[2] - cc_y[0]) / ysize
                elif self.y is not None:
                    value = (self.y[0] - self.y[1]) if len(self.y) > 1 else np.nan

        elif param == 'res':
            dx = self.get_res('dx')
            dy = self.get_res('dy')
            value = dx if dx == dy else np.nan

        if value is None:
            value = np.nan
        return value


    def get_xmin_ymax(self):
        xmin, ymax = None, None
        if self.geo_trans is not None:
            xmin, ymax = itemgetter(0, 3)(self.geo_trans)
        elif self.corner_coords is not None:
            xmin, ymax = self.corner_coords[0]
        else:
            if self.geom is not None:
                corner_coords = self.wkt_to_coords(self.geom.ExportToWkt())
                if corner_coords.shape[0] == 5:
                    xmin, ymax = corner_coords[0]
            if xmin is None or ymax is None:
                xmin = self.x[0] if (self.x is not None and len(self.x) > 0) else np.nan
                ymax = self.y[0] if (self.y is not None and len(self.y) > 0) else np.nan
        return np.array([xmin, ymax])


    def get_xmax_ymin(self):
        xmax, ymin = None, None
        if self.corner_coords is not None:
            xmax, ymin = self.corner_coords[2]
        else:
            if self.geom is not None:
                corner_coords = self.wkt_to_coords(self.geom.ExportToWkt())
                if corner_coords.shape[0] == 5:
                    xmax, ymin = corner_coords[2]
            if xmax is None or ymin is None:
                dx = self.get_res('dx')
                dy = self.get_res('dy')
                xmax = (self.x[-1] + dx) if (self.x is not None and len(self.x) > 0) else np.nan
                ymin = (self.y[-1] - dy) if (self.y is not None and len(self.y) > 0) else np.nan
                if np.isnan(xmax) or np.isnan(ymin):
                    xmin, ymax = self.get_xmin_ymax()
                    ysize, xsize = self.get_shape()
                    if np.isnan(xmax):
                        xmax = xmin + xsize*dx
                    if np.isnan(ymin):
                        ymin = ymax - ysize*dy
        return np.array([xmax, ymin])


    def get_x(self, xmin=None, xsize=None, dx=None):
        if self.x is not None \
           and (xmin is None and xsize is None and dx is None):
            return self.x
        else:
            if xmin is None:
                xmin = self.get_xmin_ymax()[0]
            if xsize is None:
                xsize = self.get_shape()[1]
            if dx is None:
                dx = self.get_res('dx')
            x = xmin + np.arange(xsize)*dx
            if xsize > 0:
                x[0] = xmin
            return x


    def get_y(self, ymax=None, ysize=None, dy=None):
        if self.y is not None \
           and (ymax is None and ysize is None and dy is None):
            return self.y
        else:
            if ymax is None:
                ymax = self.get_xmin_ymax()[1]
            if ysize is None:
                ysize = self.get_shape()[0]
            if dy is None:
                dy = self.get_res('dy')
            y = ymax - np.arange(ysize)*dy
            if ysize > 0:
                y[0] = ymax
            return y


    def get_geo_trans(self):
        if self.geo_trans is not None:
            return self.geo_trans
        else:
            xmin, ymax = self.get_xmin_ymax()
            dx = self.get_res('dx')
            dy = self.get_res('dy')
            rot1, rot2 = 0, 0
            geo_trans = np.array([
                xmin,
                dx,
                rot1,
                ymax,
                rot2,
                -dy
            ]).astype(float)
            return geo_trans


    def get_corner_coords(self, geo_trans=None, shape=None):
        if geo_trans is None and self.corner_coords is not None:
            return self.corner_coords
        else:
            if geo_trans is None and self.geom is not None:
                corner_coords = self.wkt_to_coords(self.geom.ExportToWkt())
                if corner_coords.shape[0] == 5:
                    return corner_coords

            gt = geo_trans if geo_trans is not None else self.geo_trans
            if gt is not None and (geo_trans is not None or (gt[2] != 0 or gt[4] != 0)):
                top_left_x = np.full((5, 1), gt[0])
                top_left_y = np.full((5, 1), gt[3])
                top_left_mat = np.concatenate((top_left_x, top_left_y), axis=1)

                ysize, xsize = shape if shape is not None else self.get_shape()
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

            else:
                xmin, ymax = self.get_xmin_ymax()
                xmax, ymin = self.get_xmax_ymin()
                corner_coords = np.array([
                    [xmin, ymax],
                    [xmax, ymax],
                    [xmax, ymin],
                    [xmin, ymin],
                    [xmin, ymax]
                ])
                return corner_coords


    def get_proj_ref(self):
        if self.proj_ref is not None:
            return self.proj_ref
        else:
            proj_ref = None
            spat_ref = self.spat_ref
            if spat_ref is None and self.geom is not None:
                spat_ref = self.geom.GetSpatialReference()
            if spat_ref is not None:
                proj_ref = spat_ref.ExportToWkt()
            return proj_ref


    def get_spat_ref(self):
        if self.spat_ref is not None:
            return self.spat_ref
        else:
            spat_ref = None
            if self.proj_ref is not None:
                spat_ref = osr.SpatialReference(self.proj_ref)
            elif self.geom is not None:
                spat_ref = self.geom.GetSpatialReference()
            return spat_ref


    def get_geom(self):
        if self.geom is not None:
            return self.geom
        else:
            geom_cc = self.get_corner_coords()
            if np.any(np.isnan(geom_cc)):
                geom_cc = np.array([[0, 0]])
            geom = ogr.Geometry(wkt=self.wkt(geom_cc))
            spat_ref = self.spat_ref
            if spat_ref is None and self.proj_ref is not None:
                spat_ref = osr.SpatialReference(self.proj_ref)
            if spat_ref is not None:
                geom.AssignSpatialReference(spat_ref)
            return geom


    def set_shape(self, shape, hold, set_core=True):
        if type(shape) not in (tuple, list) or len(shape) != 2 \
           or False in [(type(n) in (int, long) and n >= 0) for n in shape]:
            raise InvalidArgumentError("shape must be a numeric tuple or list of length 2")

        if hold != 'off':
            new_ysize, new_xsize = shape
            xmin, ymax = self.get_xmin_ymax()
            dx = None
            dy = None
            if hold == 'res':
                dx = self.get_res('dx')
                dy = self.get_res('dy')
                self.set_extent((xmin, ymax), (xmin + new_xsize*dx, ymax - new_ysize*dy), 'off', False)
            elif hold == 'extent':
                xmax, ymin = self.get_xmax_ymin()
                new_dx = (xmax-xmin)/new_xsize
                new_dy = (ymax-ymin)/new_ysize
                self.set_res('dx', new_dx, 'off', False)
                self.set_res('dy', new_dy, 'off', False)
                dx, dy = new_dx, new_dy
            else:
                raise InvalidArgumentError("Invalid 'hold' argument: {}".format(hold))
            if self.x is not None and new_xsize != len(self.x):
                self.set_param('x', self.get_x(xmin, new_xsize, dx), False)
            if self.y is not None and new_ysize != len(self.y):
                self.set_param('y', self.get_y(ymax, new_ysize, dy), False)

        if self.shape is not None or set_core:
            self.shape = shape


    def set_res(self, pname, res, hold, set_core=True, skip_gt=False):
        if pname not in ('dx', 'dy', 'res'):
            raise InvalidArgumentError("Invalid 'pname' argument: {}".format(pname))
        if not isinstance(res, numbers.Number) or res < 0 or res == float('inf'):
            raise InvalidArgumentError("{} must be a positive, finite number".format(pname))

        new_dx = res if pname in ('dx', 'res') else self.get_res('dx')
        new_dy = res if pname in ('dy', 'res') else self.get_res('dy')

        if hold != 'off':
            xmin, ymax = self.get_xmin_ymax()
            ysize, xsize = None, None
            if hold == 'shape':
                ysize, xsize = self.get_shape()
                self.set_extent((xmin, ymax), (xmin + xsize*new_dx, ymax - ysize*new_dy), 'off', False)
            elif hold == 'extent':
                xmax, ymin = self.get_xmax_ymin()
                new_xsize = (xmax-xmin)/new_dx
                new_ysize = (ymax-ymin)/new_dy
                new_xsize = int(new_xsize) if not np.isnan(new_xsize) else 0
                new_ysize = int(new_ysize) if not np.isnan(new_ysize) else 0
                self.set_shape((new_ysize, new_xsize), 'off', False)
                self.set_extent((xmin, ymax), (xmin + new_xsize*new_dx, ymax - new_ysize*new_dy), 'off', False)
                ysize, xsize = new_ysize, new_xsize
            else:
                raise InvalidArgumentError("Invalid 'hold' argument: {}".format(hold))
            if self.x is not None and len(self.x) > 1 and new_dx != (self.x[1]-self.x[0]):
                self.set_param('x', self.get_x(xmin, xsize, new_dx), False)
            if self.y is not None and len(self.y) > 1 and new_dy != (self.y[0]-self.y[1]):
                self.set_param('y', self.get_y(ymax, ysize, new_dy), False)

        if not skip_gt and (self.geo_trans is not None or set_core):
            if self.geo_trans is None:
                self.set_param('geo_trans')
            new_geo_trans = np.array([
                self.geo_trans[0],
                new_dx,
                self.geo_trans[2],
                self.geo_trans[3],
                self.geo_trans[4],
                -new_dy
            ])
            self.set_param('geo_trans', new_geo_trans, False)

        if eval('self.{}'.format(pname)) is not None or set_core:
            exec('self.{} = res'.format(pname))

        if pname == 'res':
            if self.dx is not None or set_core:
                self.dx = res
            if self.dy is not None or set_core:
                self.dy = res
        elif self.res is not None or set_core:
            if self.dx == self.dy and self.dx is not None:
                self.res = self.dx
            else:
                self.res = np.nan


    def set_extent(self, xmin_ymax, xmax_ymin, hold, set_core=True,
                   skip_gt=False, skip_cc=False, skip_geom=False):
        if hold in ('off', 'shape', 'res'):
            pass
        elif hold is None and xmax_ymin is None:
            pass
        else:
            raise InvalidArgumentError("Invalid 'hold' argument: {}".format(hold))
        arg_check = [np.array(xmin_ymax)]
        if xmax_ymin is None:
            # Translation will be performed.
            hold = None
        else:
            arg_check.append(np.array(xmax_ymin))
        if True in [(p.ndim != 1 or len(p) != 2 or not np.issubdtype(p.dtype, np.number))
                    for p in arg_check]:
            raise InvalidArgumentError("xmin_ymax, xmax_ymin must be convertible into a"
                                       " numeric numpy.ndarray with ndim=1 and length 2")

        new_xmin, new_ymax = xmin_ymax
        new_xmax, new_ymin = None, None
        if xmax_ymin is not None:
            new_xmax, new_ymin = xmax_ymin
        else:
            ysize, xsize = self.get_shape()
            new_xmax = new_xmin + xsize*self.get_res('dx')
            new_ymin = new_ymax - ysize*self.get_res('dy')

        littleX = True if (self.x is not None and len(self.x) < 2) else False
        littleY = True if (self.y is not None and len(self.y) < 2) else False

        if hold != 'off':
            ysize, xsize = None, None
            dx = None
            dy = None
            if hold == 'shape':
                ysize, xsize = self.get_shape()
                new_dx = (new_xmax-new_xmin)/xsize
                new_dy = (new_ymax-new_ymin)/ysize
                self.set_res('dx', new_dx, 'off', False)
                self.set_res('dy', new_dy, 'off', False)
                dx, dy = new_dx, new_dy
            elif hold == 'res':
                dx = self.get_res('dx')
                dy = self.get_res('dy')
                new_xsize = (new_xmax-new_xmin)/dx
                new_ysize = (new_ymax-new_ymin)/dy
                new_xsize = int(new_xsize) if not np.isnan(new_xsize) else 0
                new_ysize = int(new_ysize) if not np.isnan(new_ysize) else 0
                self.set_shape((new_ysize, new_xsize), 'off', False)
                new_xmax = new_xmin + new_xsize*dx
                new_ymin = new_ymax - new_ysize*dy
                ysize, xsize = new_ysize, new_xsize

            if hold is None:
                # Perform translation.
                if xmax_ymin is None:
                    if not littleX and self.x is not None and new_xmin != self.x[0]:
                        self.set_param('x', self.x + (new_xmin - self.x[0]), False)
                        self.x[0] = new_xmin
                    if not littleY and self.y is not None and new_ymax != self.y[0]:
                        self.set_param('y', self.y + (new_ymax - self.y[0]), False)
                        self.y[0] = new_ymax
            else:
                if not littleX and self.x is not None \
                   and (new_xmin != self.x[0] or new_xmax != (self.x[-1] + (self.x[1] - self.x[0]))):
                    self.set_param('x', self.get_x(new_xmin, xsize, dx), False)
                if not littleY and self.y is not None \
                   and (new_ymax != self.y[0] or new_ymin != (self.y[-1] - (self.y[0] - self.y[1]))):
                    self.set_param('y', new_ymax - np.arange(ysize)*dy, False)

            if littleX and len(self.x) == 1:
                self.set_param('x', self.get_x(new_xmin, 1, 0), False)
            if littleY and len(self.y) == 1:
                self.set_param('y', self.get_y(new_ymax, 1, 0), False)

        if not skip_gt and (self.geo_trans is not None or set_core):
            if self.geo_trans is None:
                self.set_param('geo_trans')
            new_geo_trans = np.array([
                new_xmin,
                self.geo_trans[1],
                self.geo_trans[2],
                new_ymax,
                self.geo_trans[4],
                self.geo_trans[5]
            ])
            self.set_param('geo_trans', new_geo_trans, False)

        if not (skip_cc and skip_geom) and (self.corner_coords is not None or self.geom is not None):
            corner_coords = np.array([
                [new_xmin, new_ymax],
                [new_xmax, new_ymax],
                [new_xmax, new_ymin],
                [new_xmin, new_ymin],
                [new_xmin, new_ymax]
            ])
            if not skip_cc and self.corner_coords is not None:
                self.set_param('corner_coords', corner_coords, False)
            if not skip_geom and self.geom is not None:
                spat_ref = self.geom.GetSpatialReference()
                geom_cc = corner_coords if not np.any(np.isnan(corner_coords)) else np.array([[0, 0]])
                self.set_param('geom', ogr.Geometry(wkt=self.wkt(geom_cc)), False)
                if spat_ref is not None:
                    self.geom.AssignSpatialReference(spat_ref)


    def set_projection(self, proj_ref, set_core=True, skip_sr=False, skip_geom=False):
        try:
            spat_ref = osr.SpatialReference(proj_ref)
            spat_ref.IsProjected()
        except:
            raise InvalidArgumentError("proj_ref must be a WKT projection string that can be"
                                       " converted into an osgeo.osr.SpatialReference object")

        if not skip_sr and self.spat_ref is not None:
            self.set_param('spat_ref', spat_ref, False)
        if not skip_geom and self.geom is not None:
            self.geom.AssignSpatialReference(spat_ref)

        if self.proj_ref is not None or set_core:
            self.proj_ref = proj_ref


    def set_param(self, pname, value=None, prop=True, hold=None, set_core=False, set_default=True):
        if pname not in vars(self).keys():
            raise InvalidArgumentError("Raster does not have param '{}'".format(pname))

        if value is None:
            # Set default value for parameter.
            if not set_default:
                return
            elif eval('self.{}'.format(pname)) is not None:
                # The parameter is already set. Without a value argument, there is nothing to do.
                print "This Raster's '{}' data member is already set".format(pname)
                return
        elif type(value) == str and value == 'extract':
            value = self.extract_param(pname)

        if value is None:
            prop = False
        if not prop:
            hold = 'off'
        if set_core:
            prop = True

        errmsg = None

        if pname in ('all', 'no-ds'):
            pass

        elif pname == 'ds':
            if value is None:
                raise InvalidArgumentError("ds has no default to be set")
            ds = value
            if type(ds) != osgeo.gdal.Dataset:
                errmsg = "{} must be an osgeo.gdal.Dataset"
            else:
                self.ds = ds

        elif pname == 'shape':
            shape = value if value is not None else self.get_shape()
            if hold is None:
                hold = 'extent'
            self.set_shape(shape, hold)

        elif pname == 'z':
            z = value if value is not None else np.zeros(self.get_shape())
            if type(z) != np.ndarray or not np.issubdtype(z.dtype, np.number) or z.ndim != 2:
                errmsg = "{} must be a numeric numpy.ndarray with ndim=2".format(pname)
            else:
                if prop:
                    if hold is None:
                        hold = 'extent'
                    self.set_shape(z.shape, hold, set_core)
                self.z = z

        elif pname == 'x':
            x = value if value is not None else self.get_x()
            if type(x) != np.ndarray or not np.issubdtype(x.dtype, np.number) or x.ndim != 1 \
               or (len(x) > 1 and np.any(~np.isnan(x)) \
                   and len(np.unique(np.round((x[1:] - x[:-1]), 8))) > 1):
                errmsg = "{} must be a numeric numpy.ndarray with ndim=1 and regular spacing".format(pname)
            else:
                if prop:
                    old_ysize, old_xsize = self.get_shape()
                    old_dx = self.get_res('dx')
                    old_xmin, old_ymax = self.get_xmin_ymax()
                    old_xmax, old_ymin = self.get_xmax_ymin()
                    new_xsize = len(x)
                    new_dx = None
                    if len(x) == 0:
                        new_dx = np.nan
                    elif len(x) == 1:
                        new_dx = old_dx
                    else:
                        new_dx = (x[1] - x[0])
                    new_xmin = x[0] if len(x) > 0 else np.nan
                    new_xmax = new_xmin + new_xsize*new_dx
                    if new_xsize != old_xsize:
                        self.set_shape((old_ysize, new_xsize), 'off', set_core)
                    if new_dx != old_dx:
                        self.set_res('dx', new_dx, 'off', set_core)
                    if new_xmin != old_xmin or new_xmax != old_xmax:
                        self.set_extent((new_xmin, old_ymax), (new_xmax, old_ymin), 'off', set_core)
                self.x = x

        elif pname == 'y':
            y = value if value is not None else self.get_y()
            if type(y) != np.ndarray or not np.issubdtype(y.dtype, np.number) or y.ndim != 1 \
               or (len(y) > 1 and np.any(~np.isnan(y)) \
                   and len(np.unique(np.round((y[1:] - y[:-1]), 8))) > 1):
                errmsg = "{} must be of type numpy.ndarray with ndim=1 and regular spacing".format(pname)
            else:
                if prop:
                    old_ysize, old_xsize = self.get_shape()
                    old_dy = self.get_res('dy')
                    old_xmin, old_ymax = self.get_xmin_ymax()
                    old_xmax, old_ymin = self.get_xmax_ymin()
                    new_ysize = len(y)
                    new_dy = None
                    if len(y) == 0:
                        new_dy = np.nan
                    elif len(y) == 1:
                        new_dy = old_dy
                    else:
                        new_dy = (y[0] - y[1])
                    new_ymax = y[0] if len(y) > 0 else np.nan
                    new_ymin = new_ymax - new_ysize*new_dy
                    if new_ysize != old_ysize:
                        self.set_shape((new_ysize, old_xsize), 'off', set_core)
                    if new_dy != old_dy:
                        self.set_res('dy', new_dy, 'off', set_core)
                    if new_ymax != old_ymax or new_ymin != old_ymin:
                        self.set_extent((old_xmin, new_ymax), (old_xmax, new_ymin), 'off', set_core)
                self.y = y

        elif pname in ('dx', 'dy', 'res'):
            val = value if value is not None else self.get_res(pname)
            if prop:
                if hold is None:
                    hold = 'extent'
                self.set_res(pname, value, hold)
            else:
                if not isinstance(val, numbers.Number) or val < 0 or val == float('inf'):
                    errmsg = "{} must be a positive, finite number".format(pname)
                else:
                    exec('self.{} = val'.format(pname))

        elif pname == "geo_trans":
            geo_trans = value if value is not None else self.get_geo_trans()
            if type(geo_trans) != np.ndarray or not np.issubdtype(geo_trans.dtype, np.number) \
               or geo_trans.shape != (6,):
                errmsg = "{} must be a numeric numpy.ndarray with shape (6,)".format(pname)
            else:
                if prop:
                    if hold is None:
                        hold = 'extent'
                    old_xmin, old_ymax = self.get_xmin_ymax()
                    old_dx = self.get_res('dx')
                    old_dy = self.get_res('dy')
                    new_xmin, new_ymax = itemgetter(0, 3)(geo_trans)
                    new_dx =  geo_trans[1]
                    new_dy = -geo_trans[5]
                    if new_dx != old_dx:
                        self.set_res('dx', new_dx, hold, set_core, skip_gt=True)
                    if new_dy != old_dy:
                        self.set_res('dy', new_dy, hold, set_core, skip_gt=True)
                    if new_xmin != old_xmin or new_ymax != old_ymax:
                        self.set_extent((new_xmin, new_ymax), None, None, set_core, skip_gt=True)
                self.geo_trans = geo_trans

        elif pname == 'corner_coords':
            corner_coords = value if value is not None else self.get_corner_coords()
            if type(corner_coords) != np.ndarray or not np.issubdtype(corner_coords.dtype, np.number) \
               or not corner_coords.shape == (5, 2):
                errmsg = "{} must be a numeric numpy.ndarray with shape (5, 2)".format(pname)
            else:
                if prop:
                    if hold is None:
                        hold = 'res'
                    self.set_extent(corner_coords[0], corner_coords[2], hold, set_core, skip_cc=True)
                self.corner_coords = corner_coords

        elif pname == 'proj_ref':
            proj_ref = value if value is not None else RASTER_DEFAULT_PROJREF
            if prop:
                self.set_projection(proj_ref)
            else:
                try:
                    spat_ref = osr.SpatialReference(proj_ref)
                    spat_ref.IsProjected()
                    self.proj_ref = proj_ref
                except:
                    raise InvalidArgumentError("{} must be a WKT projection string that can be"
                                               " converted into an osgeo.osr.SpatialReference"
                                               " object".format(pname))

        elif pname == 'spat_ref':
            spat_ref = value if value is not None else osr.SpatialReference(RASTER_DEFAULT_PROJREF)
            try:
                if type(spat_ref) != osgeo.osr.SpatialReference:
                    raise InvalidArgumentError
                spat_ref.IsProjected()
            except:
                errmsg = "{} must be a projected osgeo.osr.SpatialReference object".format(pname)
            if errmsg is None:
                if prop:
                    self.set_projection(spat_ref.ExportToWkt(), set_core, skip_sr=True)
                self.spat_ref = spat_ref

        elif pname in ('geom', 'geom_sr'):
            geom = value if value is not None else self.get_geom()
            try:
                if type(geom) != osgeo.ogr.Geometry \
                   or geom.GetDimension() != 2 or geom.GetCoordinateDimension() != 2:
                    raise InvalidArgumentError
                wkt = geom.ExportToWkt()
                if len(wkt.split(',')) != 5:
                    prop = False
            except:
                errmsg = "{} must be a 2D osgeo.ogr.Geometry object"\
                         " containing 5 pairs of 2D coordinates".format(pname)
            if errmsg is None:
                if prop:
                    if hold is None:
                        hold = 'res'
                    corner_coords = self.wkt_to_coords(wkt)
                    self.set_extent(corner_coords[0], corner_coords[2], hold, set_core, skip_geom=True)
                    spat_ref = self.geom.GetSpatialReference()
                    if spat_ref is not None:
                        self.set_projection(spat_ref.ExportToWkt(), set_core, skip_geom=True)
                self.geom = geom

        else:
            errmsg = "No setter mechanism has been implemented yet for parameter '{}'".format(pname)

        if errmsg is not None:
            if value is not None:
                raise InvalidArgumentError(errmsg)
            else:
                raise RasterIOError(errmsg)


    def refresh_param(self, pname):
        if pname not in vars(self).keys():
            raise InvalidArgumentError("Raster does not have param '{}'".format(pname))
        exec('self.{} = None'.format(pname))
        self.set_param(pname)


    def prop_param(self, pname, hold=None, set_core=False):
        if pname not in vars(self).keys():
            raise InvalidArgumentError("Raster does not have param '{}'".format(pname))
        value = eval('self.{}'.format(pname))
        if value is None:
            print "No value is stored in this Raster's '{}' parameter to propagate".format(pname)
            return
        exec('self.{} = None'.format(pname))
        self.set_param(pname, value, True, hold, set_core, False)



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
        'int64'     : (np.int64,      gdal.GDT_Int32,    1),  # No int64
        'uint8'     : (np.uint8,      gdal.GDT_Byte,     1),
        'uint16'    : (np.uint16,     gdal.GDT_UInt16,   1),
        'uint32'    : (np.uint32,     gdal.GDT_UInt32,   1),
        'uint64'    : (np.uint64,     gdal.GDT_UInt32,   1),  # No uint64
        'float'     : (np.float,      gdal.GDT_Float64,  1),
        'float16'   : (np.float16,    gdal.GDT_Float32,  1),  # No float16
        'float32'   : (np.float32,    gdal.GDT_Float32,  1),
        'float64'   : (np.float64,    gdal.GDT_Float64,  1),
        'complex'   : (np.complex,    gdal.GDT_CFloat64, 1),  # :: Not sure if these
        'complex64' : (np.complex64,  gdal.GDT_CFloat32, 1),  # :: complex lookups
        'complex128': (np.complex128, gdal.GDT_CFloat64, 1),  # :: are correct.
    }

    try:
        dtype_tup = dtype_dict[str(dtype_in).lower()]
    except KeyError:
        raise UnsupportedDataTypeError(
            "No such NumPy data type in lookup table: '{}'".format(dtype_in)
        )

    if form_out.lower() == 'gdal':
        if dtype_tup[2] == 1 or force_conversion:
            dtype_out = dtype_tup[1]
        else:
            raise UnsupportedDataTypeError(
                "Conversion of NumPy data type '{}' to GDAL is not supported".format(dtype_in)
            )

    elif form_out.lower() == 'numpy':
        dtype_out = dtype_tup[0]

    else:
        raise UnsupportedMethodError(
            "The following output data type format is not supported: '{}'".format(form_out)
        )

    return dtype_out


def oneBandImageToArrayZXY(rasterFile):
    """
    Opens a single-band raster image as a NumPy 2D array [Z] and returns it along
    with [X, Y] coordinate ranges of pixels in the raster grid as NumPy 1D arrays.
    """
    if not os.path.isfile(rasterFile):
        raise RasterIOError("No such rasterFile: '{}'".format(rasterFile))

    ds = gdal.Open(rasterFile, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()

    xmin, ymax = gt[0], gt[3]
    dx, dy     = gt[1], gt[5]

    X = xmin + np.arange(ds.RasterXSize) * dx
    Y = ymax + np.arange(ds.RasterYSize) * dy

    Z = ds.GetRasterBand(1).ReadAsArray()

    return Z, X, Y


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


def oneBandImageToArray_res(rasterFile):
    """
    Opens a single-band raster image as a NumPy 2D array and returns it,
    along with the (x-axis) resolution of the image.
    """
    if not os.path.isfile(rasterFile):
        raise RasterIOError("No such rasterFile: '{}'".format(rasterFile))

    ds = gdal.Open(rasterFile, gdal.GA_ReadOnly)
    array = ds.GetRasterBand(1).ReadAsArray()

    gt = ds.GetGeoTransform()
    res = gt[1]  # Technically this is dx, pixel size on the x-axis.

    return array, res


def oneBandImageToArray(rasterFile):
    """
    Opens a single-band raster image as a NumPy 2D array and returns it.
    """
    if not os.path.isfile(rasterFile):
        raise RasterIOError("No such rasterFile: '{}'".format(rasterFile))

    ds = gdal.Open(rasterFile, gdal.GA_ReadOnly)
    array = ds.GetRasterBand(1).ReadAsArray()

    return array


def getXYarrays(rasterFile):
    """
    Returns full x, y coordinate ranges of pixels in the raster grid
    as a pair of NumPy 1D arrays.
    """
    if not os.path.isfile(rasterFile):
        raise RasterIOError("No such rasterFile: '{}'".format(rasterFile))

    ds = gdal.Open(rasterFile, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()

    xmin, ymax = gt[0], gt[3]
    dx, dy     = gt[1], gt[5]

    X = xmin + np.arange(ds.RasterXSize) * dx
    Y = ymax + np.arange(ds.RasterYSize) * dy

    return X, Y


def getRes(rasterFile):
    """
    Returns the resolution of the raster image in the units of the raster
    as a numeric value.
    """
    if not os.path.isfile(rasterFile):
        raise RasterIOError("No such rasterFile: '{}'".format(rasterFile))

    ds = gdal.Open(rasterFile, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    res = gt[1]  # Technically this is dx, pixel size on the x-axis.

    return res


def getProjRef(rasterFile):
    """
    Returns the projection definition string for the raster dataset
    in OpenGIS WKT format.
    """
    if not os.path.isfile(rasterFile):
        raise RasterIOError("No such rasterFile: '{}'".format(rasterFile))

    ds = gdal.Open(rasterFile, gdal.GA_ReadOnly)
    proj_ref = ds.GetProjectionRef()

    return proj_ref


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

    driver = gdal.GetDriverByName('GTiff')

    if like_rasterFile is not None:
        ds_like = gdal.Open(like_rasterFile, gdal.GA_ReadOnly)
        if array.shape[1] == ds_like.RasterXSize and array.shape[0] == ds_like.RasterYSize:
            ds_out = driver.Create(dest_temp, ds_like.RasterXSize, ds_like.RasterYSize, 1, dtype)
            ds_out.SetGeoTransform(ds_like.GetGeoTransform())
            if proj_ref is not None:
                ds_out.SetProjection(proj_ref)
            else:
                ds_out.SetProjection(ds_like.GetProjectionRef())
        else:
            raise InvalidArgumentError(
                "Shape of like_rasterFile '{}' does not match the shape of 'array' to be saved".format(
                like_rasterFile)
            )

    else:
        if array.shape[1] == X.size and array.shape[0] == Y.size:
            ds_out = driver.Create(dest_temp, X.size, Y.size, 1, dtype)
            ds_out.SetGeoTransform((X[0], X[1]-X[0], geotrans_rot_tup[0],
                                    Y[0], geotrans_rot_tup[1], Y[1]-Y[0]))
            if proj_ref is not None:
                ds_out.SetProjection(proj_ref)
            else:
                print "WARNING: Missing projection reference for saved raster '{}'".format(dest)
        else:
            raise InvalidArgumentError(
                "Lengths of [X, Y] grid coordinates do not match the shape of 'array' to be saved"
            )

    ds_out.GetRasterBand(1).WriteArray(array_out)
    ds_out = None  # Dereference dataset to initiate write to disk of intermediate image.

    ###################################################
    # Run gdal_translate with the following arguments #
    ###################################################
    args = ["gdal_translate", dest_temp, dest]

    if nodataVal is not None:
        args.extend(["-a_nodata", str(nodataVal)])  # Create internal nodata mask.

    args.extend(["-co", "BIGTIFF=IF_SAFER"])        # Will create BigTIFF
                                                    # :: if the resulting file *might* exceed 4GB.
    args.extend(["-co", "COMPRESS=LZW"])            # Do LZW compression on output image.
    args.extend(["-co", "TILED=YES"])               # Force creation of tiled TIFF files.

    print "Calling gdal_translate:"
    check_call(args)
    os.remove(dest_temp)  # Delete the intermediate image.
    # print "'{}' saved".format(dest)



######################
# Grid Interpolation #
######################


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
        'nearest'  : gdal.GRA_NearestNeighbour,
        'linear'   : gdal.GRA_Bilinear,
        'bilinear' : gdal.GRA_Bilinear,
        'cubic'    : gdal.GRA_Cubic,
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


# TODO: Compare the results of different interpolation methods with those
# -t    of MATLAB's imresize method.
def my_imresize(array, size, method, PILmode=None):
    """
    Resizes a NumPy 2D array using either SciPy's scipy.misc.imresize function
    or GDAL's gdal.ReprojectImage (through a call of local interp2_gdal).
    Specify the size of the resized array as an int (percent of current size),
    float (fraction of current size), or tuple ([col, row] size of output array).
    Interpolation method for resampling must also be specified.
    To use SciPy's function, specify the necessary PIL image mode based on
    the input array's data type ('I' for integer, 'F' for floating, etc.)
    WARNING: INPUT PIL IMAGE MODES OTHER THAN 'F' WILL CAUSE ARRAY DATA TO
      TO BE SCALED TO UINT8 BEFORE RESIZING, POTENTIALLY DESTROYING DATA.
    """
    # If a percentage or fraction is given for size, round up the x, y pixel
    # sizes for the output array to match MATLAB's imresize function.
    if type(size) == int:
        new_shape = np.ceil(np.dot(size/100, array.shape)).astype(int)
    elif type(size) == float:
        new_shape = np.ceil(np.dot(size, array.shape)).astype(int)
    else:
        new_shape = size

    if PILmode is not None:
        array_r = scipy.misc.imresize(array, new_shape, method, mode=PILmode.upper())
        # NOTE: scipy.misc.imresize 'nearest' has slightly different results compared to
        #   MATLAB's imresize 'nearest'. Most significant is a 1-pixel SE shift.
        if method == 'nearest':
            # Correct for 1-pixel SE shift by concatenating a copy of
            # the last row and column to the last row and column,
            # then delete the first row and column.
            array_r = np.row_stack((array_r, array_r[-1, :]))
            array_r = np.column_stack((array_r, array_r[:, -1]))
            array_r = np.delete(array_r, 0, 0)
            array_r = np.delete(array_r, 0, 1)

    else:
        X = np.arange(array.shape[1]) + 1
        Y = np.arange(array.shape[0]) + 1
        Xi = np.linspace(X[0], X[-1], num=new_shape[1])
        Yi = np.linspace(Y[0], Y[-1], num=new_shape[0])
        array_r = interp2_gdal(X, Y, array, Xi, Yi, method, borderNaNs=False)

    return array_r



######################
# Array Calculations #
######################


def getDataDensityMap(array, n=11):
    """
    Given a NumPy 2D boolean array, returns an array of the same size
    with each node describing the fraction of nodes containing ones
    within a [n x n]-size kernel (or "window") of the input array.
    """
    P = scipy.signal.fftconvolve(array, np.ones((n, n)), mode='same')
    P = P / n**2

    return P



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
        except MemoryError as err:
            print >>sys.stderr, repr(err)
            print "MemoryError on call to getFPring_convhull in raster_array_tools"
            print "Defaulting to getFPring_nonzero"
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
