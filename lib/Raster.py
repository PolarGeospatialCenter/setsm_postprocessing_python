# Version 1.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2017

from __future__ import division
import os
import numbers
from operator import itemgetter

import numpy as np
import osgeo
from osgeo import gdal, ogr, osr

gdal.UseExceptions()


PROJREF_POLAR_STEREO = """PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-70],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]"""
RASTER_DEFAULT_PROJREF = PROJREF_POLAR_STEREO


class RasterIOError(Exception):
    def __init__(self, msg=""):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)


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

        set_params_unique = list(set(set_params))
        if 'all' in set_params_unique:
            set_params_unique = ['ds', 'shape', 'z', 'x', 'y', 'dx', 'dy', 'res',
                                 'geo_trans', 'corner_coords', 'proj_ref', 'spat_ref', 'geom']
        if 'no-ds' in set_params:
            if 'ds' in set_params_unique:
                set_params_unique.remove('ds')
            if 'no-ds' in set_params_unique:
                set_params_unique.remove('no-ds')

        if rasterFile_or_ds is not None:
            self.set_param('ds', self.open_ds(rasterFile_or_ds))
            if set_params_unique:
                self.extract_and_set(*set_params_unique)
                if 'ds' not in set_params_unique:
                    self.ds = None

        elif set_params_unique:
            if 'ds' in set_params_unique:
                raise InvalidArgumentError("`ds` parameter cannot be set when `rasterFile_or_ds`"
                                           " argument is None")
            self.set_params(*set_params_unique)


    @staticmethod
    def open_ds(rasterFile_or_ds):
        ds = None
        if isinstance(rasterFile_or_ds, str):
            if not os.path.isfile(rasterFile_or_ds):
                raise RasterIOError("No such `rasterFile`: '{}'".format(rasterFile_or_ds))
            ds = gdal.Open(rasterFile_or_ds, gdal.GA_ReadOnly)
        elif type(rasterFile_or_ds) == osgeo.gdal.Dataset:
            ds = rasterFile_or_ds
        else:
            raise InvalidArgumentError("Invalid input type for `rasterFile_or_ds`: {}".format(
                                       type(rasterFile_or_ds)))
        return ds


    def extract_z(self):
        return self.ds.GetRasterBand(1).ReadAsArray() if self.ds is not None else None

    def extract_shape(self):
        return (self.ds.RasterYSize, self.ds.RasterXSize) if self.ds is not None else None

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
        pname = pname.lower()
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
            value = abs(geo_trans[1])
        elif pname == 'dy':
            value = abs(geo_trans[5])
        elif pname == 'res':
            value = abs(geo_trans[1]) if abs(geo_trans[1]) == abs(geo_trans[5]) else np.nan
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
                print("WARNING: Spatial reference could not be extracted from raster dataset,"
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
            pname = pname.lower()
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
                    print("WARNING: Spatial reference could not be extracted from raster dataset,"
                          " so extracted geometry has not been assigned a spatial reference.")
            value_list.append(value)

        return value_list


    def set_params(self, *params):
        set_core = False
        params_copy = None
        if 'all' in params:
            params_copy = ('z', 'x', 'y', 'corner_coords', 'spat_ref', 'geom')
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
            if isinstance(p, str):
                if p in valid_parameters:
                    continue
                elif p == 'geom_sr':
                    pnames[i] = 'geom'
                    continue
            raise InvalidArgumentError("Starting with the first argument, every other argument "
                                       "must be a valid string name of a Raster parameter")
        for i in range(len(pnames)):
            exec('self.{} = values[{}]'.format(pnames[i], i))


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
            raise InvalidArgumentError("Invalid `param` argument: {}".format(param))
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
            raise InvalidArgumentError("`shape` must be a numeric tuple or list of length 2")

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
                raise InvalidArgumentError("Invalid `hold` argument: {}".format(hold))
            if self.x is not None and new_xsize != len(self.x):
                self.set_param('x', self.get_x(xmin, new_xsize, dx), False)
            if self.y is not None and new_ysize != len(self.y):
                self.set_param('y', self.get_y(ymax, new_ysize, dy), False)

        if self.shape is not None or set_core:
            self.shape = shape


    def set_res(self, pname, res, hold, set_core=True, skip_gt=False):
        if pname not in ('dx', 'dy', 'res'):
            raise InvalidArgumentError("Invalid `pname` argument: {}".format(pname))
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
                raise InvalidArgumentError("Invalid `hold` argument: {}".format(hold))
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
            raise InvalidArgumentError("Invalid `hold` argument: {}".format(hold))
        arg_check = [np.array(xmin_ymax)]
        if xmax_ymin is None:
            # Translation will be performed.
            hold = None
        else:
            arg_check.append(np.array(xmax_ymin))
        if True in [(p.ndim != 1 or len(p) != 2 or not np.issubdtype(p.dtype, np.number))
                    for p in arg_check]:
            raise InvalidArgumentError("`xmin_ymax`, `xmax_ymin` must be convertible into a "
                                       "numeric numpy.ndarray with ndim=1 and length 2")

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
            raise InvalidArgumentError("`proj_ref` must be a WKT projection string that can be "
                                       "converted into an osgeo.osr.SpatialReference object")

        if not skip_sr and self.spat_ref is not None:
            self.set_param('spat_ref', spat_ref, False)
        if not skip_geom and self.geom is not None:
            self.geom.AssignSpatialReference(spat_ref)

        if self.proj_ref is not None or set_core:
            self.proj_ref = proj_ref


    def set_param(self, pname, value=None, prop=True, hold=None, set_core=False, set_default=True):
        if pname not in vars(self).keys():
            raise InvalidArgumentError("Raster does not have param `pname` '{}'".format(pname))

        if value is None:
            # Set default value for parameter.
            if not set_default:
                return
            elif eval('self.{}'.format(pname)) is not None:
                # The parameter is already set. Without a value argument, there is nothing to do.
                print("This Raster's '{}' data member is already set".format(pname))
                return
        elif isinstance(value, str) and value == 'extract':
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
                raise InvalidArgumentError("`ds` has no default to be set")
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
            raise InvalidArgumentError("Raster does not have param `pname` '{}'".format(pname))
        exec('self.{} = None'.format(pname))
        self.set_param(pname)


    def prop_param(self, pname, hold=None, set_core=False):
        if pname not in vars(self).keys():
            raise InvalidArgumentError("Raster does not have param `pname` '{}'".format(pname))
        value = eval('self.{}'.format(pname))
        if value is None:
            print("No value is stored in this Raster's '{}' parameter to propagate".format(pname))
            return
        exec('self.{} = None'.format(pname))
        self.set_param(pname, value, True, hold, set_core, False)
