
import textwrap
import warnings

from osgeo import gdal, osr
gdal.UseExceptions()
osr.UseExceptions()


SETSM_SRS_LIST = []


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)
class GdalError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


## GDAL error handler setup
def GdalErrorHandler(err_level, err_no, err_msg):
    error_message = (
        "Caught GDAL error (err_level={}, err_no={}) "
        "where level >= gdal.CE_Warning({}); error message below:\n{}".format(
            err_level, err_no, gdal.CE_Warning, err_msg
        )
    )
    if err_level == gdal.CE_Warning:
        warnings.warn(error_message)
    if err_level > gdal.CE_Warning:
        raise GdalError(error_message)
gdal.PushErrorHandler(GdalErrorHandler)
gdal.UseExceptions()


def osr_srs_preserve_axis_order(osr_srs):
    try:
        # Revert to GDAL 2.X axis conventions to maintain consistent results if GDAL 3+ used
        osr_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except AttributeError:
        pass
    return osr_srs


def get_setsm_epsg_list():
    # Polar projections
    epsg_list = [
        3413,  # WGS 84 / NSIDC Sea Ice Polar Stereographic North
        3031,  # WGS 84 / Antarctic Polar Stereographic
    ]

    # UTM projections, generally for "non-polar"
    # (between 60N and 60S latitudes)
    for hemi_n in (6, 7):
        for zone_n in range(1, 61):
            epsg = 32000 + 100*hemi_n + zone_n
            epsg_list.append(epsg)

    return epsg_list


def get_setsm_srs_list():
    srs_list = []
    for epsg in get_setsm_epsg_list():
        srs = osr_srs_preserve_axis_order(osr.SpatialReference())
        rc = srs.ImportFromEPSG(epsg)
        if rc != 0:
            raise GdalError(
                "Non-zero return code ({}) from"
                " OGRSpatialReference.ImportFromEPSG({})".format(
                    rc, epsg
                ))
        srs_list.append(srs)
    return srs_list


def get_matching_srs(srs_or_proj4, compare_srs_list=None):
    if compare_srs_list is None:
        global SETSM_SRS_LIST
        SETSM_SRS_LIST = get_setsm_srs_list()
        compare_srs_list = SETSM_SRS_LIST

    if isinstance(srs_or_proj4, osr.SpatialReference):
        input_srs = srs_or_proj4
    elif isinstance(srs_or_proj4, str):
        input_srs = osr_srs_preserve_axis_order(osr.SpatialReference())
        rc = input_srs.ImportFromProj4(srs_or_proj4)
        if rc != 0:
            raise GdalError(
                "Non-zero return code ({}) from"
                " OGRSpatialReference.ImportFromProj4('{}')".format(
                    rc, srs_or_proj4
                )
            )
    else:
        raise InvalidArgumentError(
            "Argument `srs_or_proj4` must be an `osr.SpatialReference` object or string,"
            " but type was {}".format(type(srs_or_proj4))
        )

    compare_srs_match = None
    for compare_srs in compare_srs_list:
        rc = input_srs.IsSame(compare_srs)
        if rc == 0:
            # SRS are not the same
            pass
        elif rc == 1:
            # SRS are the same
            compare_srs_match = compare_srs
            break
        else:
            input_srs_proj4 = input_srs.ExportToProj4()
            compare_srs_proj4 = compare_srs.ExportToProj4()
            raise GdalError(textwrap.fill(textwrap.dedent(rf"""
                Return code ({rc}) from OGRSpatialReference.IsSame
                is outside range of valid return codes [0,1]
                when comparing input SRS (1) and compare SRS (2) SRS
                (PROJ.4 strings as follows):
                \n  1) {input_srs_proj4}
                \n  2) {compare_srs_proj4}
                """
            ), width=float('inf')).replace(r'\n', '\n'))

    return compare_srs_match
