# Erik Husby; Polar Geospatial Center, University of Minnesota; 2018


import os
import sys

import gdal
import osr


srs_base = None


items = sys.argv[1:]

for item in items:

    srs_compare = osr.SpatialReference()


    if os.path.isfile(item):
        rasterFile = item
        raster_ds = gdal.Open(rasterFile, gdal.GA_ReadOnly)
        srs_compare.ImportFromWkt(raster_ds.GetProjectionRef())

    elif item.startswith('EPSG:') or item.isdigit():
        epsg_code = item
        srs_compare.ImportFromEPSG(int(epsg_code.replace('EPSG:', '')))

    else:
        proj4_str = item
        srs_compare.ImportFromProj4(proj4_str)


    if srs_base is None:
        srs_base = srs_compare
    else:
        compare_result = srs_compare.IsSame(srs_base)
        if compare_result == 1:
            continue
        elif compare_result == 0:
            sys.exit(1)
        elif compare_result != 1:
            sys.exit(compare_result)


sys.exit(0)
