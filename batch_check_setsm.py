
# Erik Husby; Polar Geospatial Center, University of Minnesota; 2019


from __future__ import division
from lib import script_utils

PYTHON_VERSION_ACCEPTED_MIN = "2.7"  # supports multiple dot notation
if script_utils.PYTHON_VERSION < script_utils.VersionString(PYTHON_VERSION_ACCEPTED_MIN):
    raise script_utils.VersionError("Python version ({}) is below accepted minimum ({})".format(
        script_utils.PYTHON_VERSION, PYTHON_VERSION_ACCEPTED_MIN))


import argparse
import copy
import glob
import os
import re
import shutil
import subprocess
import sys
import traceback
import warnings
from time import sleep
if script_utils.PYTHON_VERSION < script_utils.VersionString(3):
    from StringIO import StringIO
else:
    from io import StringIO

from lib.script_utils import eprint
from lib.script_utils import ScriptArgumentError, DeveloperError


##############################

## Core globals

SCRIPT_VERSION_NUM = 1.0

# Paths
SCRIPT_FILE = os.path.realpath(__file__)
SCRIPT_FNAME = os.path.basename(SCRIPT_FILE)
SCRIPT_NAME, SCRIPT_EXT = os.path.splitext(SCRIPT_FNAME)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE)
SCRIPT_RUNCMD = ' '.join(sys.argv)+'\n'
PYTHON_EXE = 'python -u'

HOSTNAME = os.getenv('HOSTNAME')
if HOSTNAME is not None:
    HOSTNAME = HOSTNAME.lower()
    RUNNING_AT_PGC = True if True in [s in HOSTNAME for s in ['rookery', 'nunatak']] else False
else:
    RUNNING_AT_PGC = False

##############################

## Argument globals

# Argument strings
ARGSTR_SRC = 'src'
ARGSTR_DEPTH = '--depth'
ARGSTR_SRC_SUFFIX = '--src-suffix'
ARGSTR_CHECK_METHOD = '--check-method'
ARGSTR_CHECK_SETSM_VALIDRANGE = '--check-setsm-validrange'
ARGSTR_CHECK_SETSM_ALLOW_INVALID = '--check-setsm-allow-invalid'
ARGSTR_CHECKFILE_OFF = '--checkfile-off'
ARGSTR_CHECKFILE = '--checkfile'
ARGSTR_CHECKFILE_ROOT = '--checkfile-root'
ARGSTR_CHECKFILE_ROOT_REGEX = '--checkfile-root-regex'
ARGSTR_CHECK_SPECIAL = '--check-special'
ARGSTR_CHECK_SPECIAL_DEMTYPE = '--check-special-demtype'
ARGSTR_CHECKFILE_EXT = '--checkfile-ext'
ARGSTR_ERRFILE_EXT = '--errfile-ext'
ARGSTR_ALLOW_MISSING_SUFFIX = '--allow-missing-suffix'
ARGSTR_ALLOW_MISSING_ORTHO2 = '--allow-missing-ortho2'
ARGSTR_RETRY_ERRORS = '--retry-errors'
ARGSTR_KEEP_CHECKFILE_WITH_ERRORS = '--keep-checkfile-with-errors'
ARGSTR_SUPPRESS_ERRFILE_EXISTS = '--suppress-errfile-exists'
ARGSTR_SUPPRESS_MISSING_SUFFIX = '--suppress-missing-suffix'
ARGSTR_SUPPRESS_MISSING_CHECKED = '--suppress-missing-checked'
ARGSTR_SUPPRESS_NEW_SOURCE = '--suppress-new-source'
ARGSTR_REMOVE_TYPE = '--remove-type'
ARGSTR_RMWHERE_ERRFILE_EXISTS = '--rmwhere-errfile-exists'
ARGSTR_RMWHERE_MISSING_SUFFIX = '--rmwhere-missing-suffix'
ARGSTR_RMWHERE_MISSING_CHECKED = '--rmwhere-missing-checked'
ARGSTR_RMWHERE_NEW_SOURCE = '--rmwhere-new-source'
ARGSTR_REMOVE_ONLY = '--remove-only'
ARGSTR_STATS_ONLY = '--stats-only'
ARGSTR_SCHEDULER = '--scheduler'
ARGSTR_JOBSCRIPT = '--jobscript'
ARGSTR_TASKS_PER_JOB = '--tasks-per-job'
ARGSTR_SCRATCH = '--scratch'
ARGSTR_WD = '--wd'
ARGSTR_LOGDIR = '--logdir'
ARGSTR_EMAIL = '--email'
ARGSTR_DO_DELETE = '--do-delete'
ARGSTR_DRYRUN = '--dryrun'

# Argument groups
ARGGRP_OUTDIR = [ARGSTR_LOGDIR, ARGSTR_SCRATCH]
ARGGRP_BATCH = [ARGSTR_SCHEDULER, ARGSTR_JOBSCRIPT, ARGSTR_TASKS_PER_JOB, ARGSTR_EMAIL]
ARGGRP_CHECK_REGULAR = [ARGSTR_CHECKFILE, ARGSTR_CHECKFILE_ROOT, ARGSTR_CHECKFILE_ROOT_REGEX]
ARGGRP_CHECK_OTHER = [ARGSTR_CHECK_SPECIAL]
ARGGRP_CHECK_ALL = ARGGRP_CHECK_REGULAR + ARGGRP_CHECK_OTHER
ARGGRP_RMWHERE = [
    ARGSTR_RMWHERE_ERRFILE_EXISTS,
    ARGSTR_RMWHERE_MISSING_SUFFIX,
    ARGSTR_RMWHERE_MISSING_CHECKED,
    ARGSTR_RMWHERE_NEW_SOURCE
]
ARGGRP_REQUIRES_RMWHERE = [ARGSTR_DO_DELETE, ARGSTR_REMOVE_ONLY]

# Argument choices
ARGCHO_CHECK_METHOD_READ = 'read'
ARGCHO_CHECK_METHOD_CHECKSUM = 'checksum'
ARGCHO_CHECK_METHOD = [
    ARGCHO_CHECK_METHOD_READ,
    ARGCHO_CHECK_METHOD_CHECKSUM
]
ARGCHO_CHECK_SPECIAL_ALL_TOGETHER = 'altogether'
ARGCHO_CHECK_SPECIAL_ALL_SEPARATE = 'separate'
ARGCHO_CHECK_SPECIAL_SCENEPAIRS = 'scenes'
ARGCHO_CHECK_SPECIAL_PAIRNAMES = 'pairnames'
ARGCHO_CHECK_SPECIAL_STRIPSEGMENTS = 'strip-segments'
ARGCHO_CHECK_SPECIAL_STRIPS = 'strips'
ARGCHO_CHECK_SPECIAL_SCENEMETA = 'scene-meta'
ARGCHO_CHECK_SPECIAL_STRIPMETA = 'strip-meta'
ARGCHO_CHECK_SPECIAL = [
    ARGCHO_CHECK_SPECIAL_ALL_TOGETHER,
    ARGCHO_CHECK_SPECIAL_ALL_SEPARATE,
    ARGCHO_CHECK_SPECIAL_SCENEPAIRS,
    ARGCHO_CHECK_SPECIAL_PAIRNAMES,
    ARGCHO_CHECK_SPECIAL_STRIPSEGMENTS,
    ARGCHO_CHECK_SPECIAL_STRIPS,
    ARGCHO_CHECK_SPECIAL_SCENEMETA,
    ARGCHO_CHECK_SPECIAL_STRIPMETA
]
ARGCHO_CHECK_SPECIAL_DEMTYPE_REGULAR = 'non-lsf'
ARGCHO_CHECK_SPECIAL_DEMTYPE_SMOOTH = 'lsf'
ARGCHO_CHECK_SPECIAL_DEMTYPE_BOTH = 'both'
ARGCHO_CHECK_SPECIAL_DEMTYPE = [
    ARGCHO_CHECK_SPECIAL_DEMTYPE_REGULAR,
    ARGCHO_CHECK_SPECIAL_DEMTYPE_SMOOTH,
    ARGCHO_CHECK_SPECIAL_DEMTYPE_BOTH
]
ARGCHO_REMOVE_TYPE_CHECKFILES = 'checkfiles'
ARGCHO_REMOVE_TYPE_SOURCEFILES = 'sourcefiles'
ARGCHO_REMOVE_TYPE_BOTH = 'both'
ARGCHO_REMOVE_TYPE = [
    ARGCHO_REMOVE_TYPE_CHECKFILES,
    ARGCHO_REMOVE_TYPE_SOURCEFILES,
    ARGCHO_REMOVE_TYPE_BOTH
]

# Argument choice groups
ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM_SCENELEVEL = [
    ARGCHO_CHECK_SPECIAL_SCENEPAIRS,
    ARGCHO_CHECK_SPECIAL_PAIRNAMES
]
ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM_STRIPLEVEL = [
    ARGCHO_CHECK_SPECIAL_STRIPSEGMENTS,
    ARGCHO_CHECK_SPECIAL_STRIPS
]
ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM = (
      ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM_SCENELEVEL
    + ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM_STRIPLEVEL
)
ARGCHOGRP_CHECK_SPECIAL_SETSM_SCENELEVEL = ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM_SCENELEVEL + [
    ARGCHO_CHECK_SPECIAL_SCENEMETA
]
ARGCHOGRP_CHECK_SPECIAL_SETSM_STRIPLEVEL = ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM_STRIPLEVEL + [
    ARGCHO_CHECK_SPECIAL_STRIPMETA
]
ARGCHOGRP_CHECK_SPECIAL_SETSM = (
      ARGCHOGRP_CHECK_SPECIAL_SETSM_SCENELEVEL
    + ARGCHOGRP_CHECK_SPECIAL_SETSM_STRIPLEVEL
)

# Argument choice settings
CHECK_SPECIAL_DEM_SUFFIX_ORTHO2 = 'ortho2.tif'
CHECK_SPECIAL_DEM_SUFFIX_SCENELEVEL_MATCHTAG_SET = {'matchtag_mt.tif', 'meta_mt.txt'}
CHECK_SPECIAL_DEM_SUFFIX_OPTIONAL_SCENELEVEL_SET = {
    'mask.tif',
    'meta_or.txt'
}
CHECK_SPECIAL_DEM_SUFFIX_OPTIONAL_STRIPLEVEL_SET = {
    'bitmask_10m.tif',
    'ortho_10m.tif',
    'dem_10m.tif',
    'dem_10m_masked.tif',
    'dem_10m_shade.tif',
    'dem_10m_shade_masked.tif',
    'dem_40m_masked.tif',
    'dem_40m_coverage.tif'
}
ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_META = 'meta.txt'
ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_SCENELEVEL = '/'.join([
    # DEM suffix(es) will be set by --check-special-demtype script argument
    'matchtag.tif',
    'ortho.tif',
    CHECK_SPECIAL_DEM_SUFFIX_ORTHO2,
    ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_META,
    '/'.join(CHECK_SPECIAL_DEM_SUFFIX_SCENELEVEL_MATCHTAG_SET),
    '/'.join(CHECK_SPECIAL_DEM_SUFFIX_OPTIONAL_SCENELEVEL_SET)
])
ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_STRIPLEVEL = '/'.join([
    'dem.tif',
    'matchtag.tif',
    'ortho.tif',
    CHECK_SPECIAL_DEM_SUFFIX_ORTHO2,
    'bitmask.tif',
    ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_META,
    '/'.join(CHECK_SPECIAL_DEM_SUFFIX_OPTIONAL_STRIPLEVEL_SET)
])
# ARGCHOSET_CHECK_SPECIAL_DEM_REGEX_SCENELEVEL = "(?P<scenepairname>(?P<strippairname>(?P<sensor>[A-Z0-9]{4})_(?P<timestamp>\d{8})_(?P<catid1>[A-Z0-9]{16})_(?P<catid2>[A-Z0-9]{16}))_(?P<tile1>R\d+C\d+-)?(?P<order1>\d{12}_\d{2})_(?P<part1>P\d{3})_(?P<tile2>R\d+C\d+-)?(?P<order2>\d{12}_\d{2})_(?P<part2>P\d{3})_(?P<res>\d{1}))_(?P<suffix>[_a-z0-9]+)\.(?P<ext>\w+)"
ARGCHOSET_CHECK_SPECIAL_DEM_REGEX_SCENELEVEL   = "^([A-Z0-9]{4}_\d{8}_[0-9A-F]{16}_[0-9A-F]{16}_(R\d+C\d+-)?\d{12}_\d{2}_P\d{3}_(R\d+C\d+-)?\d{12}_\d{2}_P\d{3}_\d{1})_[a-z0-9_]+\.\w+$"
ARGCHOSET_CHECK_SPECIAL_DEM_REGEX_STRIPLEVEL   = "^([A-Z0-9]{4}_\d{8}_[0-9A-F]{16}_[0-9A-F]{16}).*$"
ARGCHOSET_CHECK_SPECIAL_DEM_REGEX_STRIPSEGMENT = "^([A-Z0-9]{4}_\d{8}_[0-9A-F]{16}_[0-9A-F]{16}_\d+c?m(_lsf)?_seg\d+)_[a-z0-9_]+\.\w+$"

ARGCHOSET_CHECK_SPECIAL_SETTING_DICT = {
    ARGCHO_CHECK_SPECIAL_ALL_TOGETHER: [
        (ARGSTR_CHECKFILE_ROOT, "")
    ],
    ARGCHO_CHECK_SPECIAL_ALL_SEPARATE: [
        (ARGSTR_CHECKFILE_ROOT_REGEX, "^(.*)$")
    ],
    ARGCHO_CHECK_SPECIAL_SCENEPAIRS: [
        (ARGSTR_SRC_SUFFIX, ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_SCENELEVEL),
        (ARGSTR_CHECKFILE_ROOT_REGEX, ARGCHOSET_CHECK_SPECIAL_DEM_REGEX_SCENELEVEL)
    ],
    ARGCHO_CHECK_SPECIAL_PAIRNAMES: [
        (ARGSTR_SRC_SUFFIX, ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_SCENELEVEL),
        (ARGSTR_CHECKFILE_ROOT_REGEX, ARGCHOSET_CHECK_SPECIAL_DEM_REGEX_STRIPLEVEL)
    ],
    ARGCHO_CHECK_SPECIAL_STRIPSEGMENTS: [
        (ARGSTR_SRC_SUFFIX, ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_STRIPLEVEL),
        (ARGSTR_CHECKFILE_ROOT_REGEX, ARGCHOSET_CHECK_SPECIAL_DEM_REGEX_STRIPSEGMENT)
    ],
    ARGCHO_CHECK_SPECIAL_STRIPS: [
        (ARGSTR_SRC_SUFFIX, ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_STRIPLEVEL),
        (ARGSTR_CHECKFILE_ROOT_REGEX, ARGCHOSET_CHECK_SPECIAL_DEM_REGEX_STRIPLEVEL)
    ],
    ARGCHO_CHECK_SPECIAL_SCENEMETA: [
        (ARGSTR_SRC_SUFFIX, ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_META),
        (ARGSTR_CHECKFILE_OFF, True),
    ],
    ARGCHO_CHECK_SPECIAL_STRIPMETA: [
        (ARGSTR_SRC_SUFFIX, ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_META),
        (ARGSTR_CHECKFILE_OFF, True),
    ]
}
ARGCHOSET_CHECK_SPECIAL_SUBGROUP_DICT = {
    ARGCHO_CHECK_SPECIAL_PAIRNAMES: [
        ARGCHO_CHECK_SPECIAL_SCENEPAIRS
    ],
    ARGCHO_CHECK_SPECIAL_STRIPS: [
        ARGCHO_CHECK_SPECIAL_STRIPSEGMENTS
    ]
}
ARGCHOSET_CHECK_SPECIAL_DEMTYPE_SUFFIX_DICT = {
    ARGCHO_CHECK_SPECIAL_DEMTYPE_REGULAR: 'dem.tif',
    ARGCHO_CHECK_SPECIAL_DEMTYPE_SMOOTH: 'dem_smooth.tif/smooth_result.txt',
}
ARGCHOSET_CHECK_SPECIAL_DEMTYPE_SUFFIX_DICT[ARGCHO_CHECK_SPECIAL_DEMTYPE_BOTH] = '/'.join(
    ARGCHOSET_CHECK_SPECIAL_DEMTYPE_SUFFIX_DICT.values())

# Argument defaults
ARGDEF_SRC_SUFFIX = '.tif'
ARGDEF_DEPTH = script_utils.ARGNUM_POS_INF
ARGDEF_CHECKFILE_EXT = '.check'
ARGDEF_CHECKERROR_EXT = '.err'
ARGDEF_SCRATCH = os.path.join(os.path.expanduser('~'), 'scratch', 'task_bundles')

##############################

## Batch settings

JOBSCRIPT_DIR = os.path.join(SCRIPT_DIR, 'jobscripts')
JOBSCRIPT_INIT = os.path.join(JOBSCRIPT_DIR, 'init.sh')
JOB_ABBREV = 'Check'
BATCH_ARGDEF_WD = '/local' if RUNNING_AT_PGC else None
JOB_WALLTIME_HR = 30
JOB_MEMORY_GB = 20

##############################

## Custom globals

GDAL_RASTER_SUFFIXES = ['.tif', '.tiff']

SETSM_RASTER_SUFFIX_VALIDRANGE_DICT = {
    '_dem.tif': [-8000, 100000],
    '_dem_smooth.tif': [-8000, 100000],
    '_matchtag.tif': [0, 1],
    '_matchtag_mt.tif': [0, 1],
}

SETSM_META_SUFFIX = '_meta.txt'
SETSM_STRIPMETA_SCENEMETA_SECTION_HEADER = 'Scene Metadata'
SETSM_STRIPMETA_SCENEMETA_ITEM_HEADER_REGEX = re.compile("^\s*scene \d+ name=.*$")

SETSM_META_REQUIRED_DICT = dict()
SETSM_META_KEY_TOKEN_DELIM_RE = '(?: +|_+)'
SETSM_META_SPACE_RE = '[ \t]*?'
SETSM_META_NEWLINE_START_RE = '(?:\r\n|\r|\n)'
SETSM_META_NEWLINE_END_RE = '(?=(?:\r\n|\r|\n))'
SETSM_META_KEY_PREFIX_IMAGE = 'image'.strip().lower()
SETSM_META_KEY_PREFIX_IMAGE_1 = ' '.join([SETSM_META_KEY_PREFIX_IMAGE, str(1)])
SETSM_META_KEY_PREFIX_IMAGE_2 = ' '.join([SETSM_META_KEY_PREFIX_IMAGE, str(2)])
SETSM_META_IMAGE_PREFIX_RE = SETSM_META_KEY_TOKEN_DELIM_RE.join([SETSM_META_KEY_PREFIX_IMAGE, '[12]'])
SETSM_META_WV_CORRECT_SATIDS = ['WV01', 'WV02']

def get_setsm_meta_item_regex(key_str, value_re, allow_missing_image_prefix=False):
    if key_str is None:
        key_re = SETSM_META_IMAGE_PREFIX_RE
    else:
        key_re = SETSM_META_KEY_TOKEN_DELIM_RE.join(key_str.replace('_', ' ').split())
        if allow_missing_image_prefix:
            key_re = '(?:{}|{})'.format(SETSM_META_KEY_TOKEN_DELIM_RE.join([SETSM_META_IMAGE_PREFIX_RE, key_re]), key_re)
        else:
            key_re = SETSM_META_KEY_TOKEN_DELIM_RE.join([SETSM_META_IMAGE_PREFIX_RE, key_re])
    item_re = SETSM_META_SPACE_RE.join([SETSM_META_NEWLINE_START_RE, key_re, '=', value_re, SETSM_META_NEWLINE_END_RE])
    return re.compile(item_re, re.I)

SETSM_META_ITEM_IS_KEY_VALUE = True
SETSM_META_ITEM_IS_NOT_KEY_VALUE = False
SETSM_META_ITEM_COUNT_SINGLE = 1
SETSM_META_ITEM_COUNT_PAIR = 2

SETSM_META_KEY = 'Image path'
SETSM_META_KEY_IMAGE_PATH = SETSM_META_KEY
SETSM_META_ITEM_RE = get_setsm_meta_item_regex(None, "[\d\w_\-/]+\.tif")
SETSM_META_REQUIRED_DICT[SETSM_META_KEY] = (SETSM_META_ITEM_RE, SETSM_META_ITEM_IS_KEY_VALUE, SETSM_META_ITEM_COUNT_PAIR)

SETSM_META_KEYGRP_GSD = [
    # 'Mean_row_GSD',
    # 'Mean_col_GSD',
    # 'Mean_GSD'
]

for SETSM_META_KEY in SETSM_META_KEYGRP_GSD + [
    'Mean_sun_azimuth_angle',
    'Mean_sun_elevation',
    'Mean_sat_azimuth_angle'
]:
    SETSM_META_VALUE_RE = "\d+\.?\d*"
    SETSM_META_ITEM_RE = get_setsm_meta_item_regex(SETSM_META_KEY, SETSM_META_VALUE_RE, allow_missing_image_prefix=True)
    SETSM_META_REQUIRED_DICT[SETSM_META_KEY] = (SETSM_META_ITEM_RE, SETSM_META_ITEM_IS_KEY_VALUE, SETSM_META_ITEM_COUNT_PAIR)

SETSM_META_KEY = 'Mean_sat_elevation'
SETSM_META_ITEM_RE = get_setsm_meta_item_regex(SETSM_META_KEY, "\-?\d+\.?\d*", allow_missing_image_prefix=True)
SETSM_META_REQUIRED_DICT[SETSM_META_KEY] = (SETSM_META_ITEM_RE, SETSM_META_ITEM_IS_KEY_VALUE, SETSM_META_ITEM_COUNT_PAIR)

for SETSM_META_KEY in [
    'effbw',
    'abscalfact'
]:
    SETSM_META_VALUE_RE = "\d+\.?\d*"
    SETSM_META_ITEM_RE = get_setsm_meta_item_regex(SETSM_META_KEY, SETSM_META_VALUE_RE)
    SETSM_META_REQUIRED_DICT[SETSM_META_KEY] = (SETSM_META_ITEM_RE, SETSM_META_ITEM_IS_KEY_VALUE, SETSM_META_ITEM_COUNT_PAIR)

for SETSM_META_KEY in [
    'tdi',
    'min',
    'max'
]:
    SETSM_META_VALUE_RE = "\d+"
    SETSM_META_ITEM_RE = get_setsm_meta_item_regex(SETSM_META_KEY, SETSM_META_VALUE_RE)
    SETSM_META_REQUIRED_DICT[SETSM_META_KEY] = (SETSM_META_ITEM_RE, SETSM_META_ITEM_IS_KEY_VALUE, SETSM_META_ITEM_COUNT_PAIR)

SETSM_META_KEY = 'wv_correct'
SETSM_META_KEY_WV_CORRECT = SETSM_META_KEY
SETSM_META_ITEM_RE = get_setsm_meta_item_regex(SETSM_META_KEY, "[01]")
SETSM_META_REQUIRED_DICT[SETSM_META_KEY] = (SETSM_META_ITEM_RE, SETSM_META_ITEM_IS_KEY_VALUE, SETSM_META_ITEM_COUNT_PAIR)

SETSM_META_KEY = 'ASP build ID'
SETSM_META_ITEM_RE = get_setsm_meta_item_regex(SETSM_META_KEY, "(?:[0-9A-F]+)?")
SETSM_META_REQUIRED_DICT[SETSM_META_KEY] = (SETSM_META_ITEM_RE, SETSM_META_ITEM_IS_KEY_VALUE, SETSM_META_ITEM_COUNT_PAIR)

SETSM_META_REQUIRED_KEY_SORTED_LIST = sorted(SETSM_META_REQUIRED_DICT.keys())

del SETSM_META_KEY, SETSM_META_ITEM_RE, SETSM_META_VALUE_RE

##############################


class SETSMMetaParseError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

class RasterFileReadError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


def argparser_init():

    parser = argparse.ArgumentParser(
        formatter_class=script_utils.RawTextArgumentDefaultsHelpFormatter,
        description=' '.join([
            "Check existence and integrity of data files in batch."
        ])
    )

    # Positional arguments

    parser.add_argument(
        ARGSTR_SRC,
        # type=script_utils.ARGTYPE_PATH(
        #     argstr=ARGSTR_SRC,
        #     existcheck_fn=os.path.exists,
        #     existcheck_reqval=True),
        type=str,
        help=' '.join([
            "Path to source file directory or single input file to check.",
            "Accepts a task bundle text file listing paths to checkfile root paths."
        ])
    )

    # Optional arguments

    parser.add_argument(
        ARGSTR_DEPTH,
        type=script_utils.ARGTYPE_NUM(
            argstr=ARGSTR_DEPTH,
            numeric_type=int,
            allow_neg=False,
            allow_zero=False,
            allow_inf=True),
        default=ARGDEF_DEPTH,
        help=' '.join([
            "Depth of recursive search into source directory for files to check.",
        ])
    )

    parser.add_argument(
        ARGSTR_SRC_SUFFIX,
        type=str,
        default=ARGDEF_SRC_SUFFIX,
        help=' '.join([
            "'/'-delimited list of accepted source file suffixes to be checked.",
        ])
    )

    parser.add_argument(
        ARGSTR_CHECK_METHOD,
        type=str,
        choices=ARGCHO_CHECK_METHOD,
        default=ARGCHO_CHECK_METHOD_CHECKSUM,
        help=' '.join([
            "Method used to check integrity of source rasters.",
            "\nIf '{}', simply attempt to read raster band(s).".format(ARGCHO_CHECK_METHOD_READ),
            "\nIf '{}', attempt to compute checksum of each raster band.".format(ARGCHO_CHECK_METHOD_CHECKSUM),
            "\n"
        ])
    )
    parser.add_argument(
        ARGSTR_CHECK_SETSM_VALIDRANGE,
        action='store_true',
        help=' '.join([
            "After successfully opening a source raster ending with a filename suffix listed in",
            "script 'Custom globals' dictionary variable SETSM_RASTER_SUFFIX_VALIDRANGE_DICT, check that all",
            "non-NoData values fall the raster band fall within the corresponding numerical range",
            "(inclusive)."
        ])
    )

    parser.add_argument(
        ARGSTR_CHECKFILE_OFF,
        action='store_true',
        help=' '.join([
            "Ignore existing checkfiles and check all files, saving error files but not checkfiles."
        ])
    )
    parser.add_argument(
        ARGSTR_CHECKFILE,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_CHECKFILE,
            existcheck_fn=os.path.isdir,
            existcheck_reqval=False),
        default=None,
        help=' '.join([
            "Path to single checkfile (which may already exist) used to store filenames of",
            "passing source file(s) selected by arguments {} and {}.".format(ARGSTR_SRC, ARGSTR_SRC_SUFFIX),
            "Due to the issue of multiple processes attempting to write to a text file at once,",
            "this argument is incompatible with job scheduler options.",
        ])
    )
    parser.add_argument(
        ARGSTR_CHECKFILE_ROOT,
        type=str,
        default=None,
        help=' '.join([
            "Filename prefix by which to group source files for checking.",
            "The default path of the checkfile becomes '[{}]/[{}].[{}]'".format(ARGSTR_SRC, ARGSTR_CHECKFILE_ROOT, ARGSTR_CHECKFILE_EXT),
            "Use only if argument {} is a directory.".format(ARGSTR_SRC),
            "Due to the issue of multiple processes attempting to write to a text file at once,",
            "this argument is incompatible with job scheduler options."
        ])
    )
    parser.add_argument(
        ARGSTR_CHECKFILE_ROOT_REGEX,
        type=str,
        default=None,
        help=' '.join([
            "Regex for filename prefix by which to group source files for checking.",
            "Regex must contain one group for matching, which becomes the filename prefix for"
            "a single bundle of source files to check.",
            "The default path of each checkfile thus becomes '[{}]/[regex match group].[{}]'".format(ARGSTR_SRC, ARGSTR_CHECKFILE_ROOT, ARGSTR_CHECKFILE_EXT),
            "Use only if argument {} is a directory.".format(ARGSTR_SRC),
            "In the context of the job scheduler {} option, each unique regex match group becomes".format(ARGSTR_SCHEDULER),
            "a single task by passing it as the {} argument to a 'fork' of this batch script".format(ARGSTR_CHECKFILE_ROOT)
        ])
    )
    parser.add_argument(
        ARGSTR_CHECK_SPECIAL,
        type=str,
        choices=ARGCHO_CHECK_SPECIAL,
        default=None,
        help=' '.join([
            "Popular options for quickly setting {} and {} arguments.".format(ARGSTR_SRC_SUFFIX, ARGSTR_CHECKFILE_ROOT_REGEX),
        ])
    )
    parser.add_argument(
        ARGSTR_CHECK_SPECIAL_DEMTYPE,
        type=str,
        choices=ARGCHO_CHECK_SPECIAL_DEMTYPE,
        default=ARGCHO_CHECK_SPECIAL_DEMTYPE_BOTH,
        help=' '.join([
            "Used in conjunction with argument {}, this determines which DEM file suffix(es)".format(ARGSTR_CHECK_SPECIAL),
            "are set for argument {} source file selection".format(ARGSTR_SRC_SUFFIX)
        ])
    )

    parser.add_argument(
        ARGSTR_CHECKFILE_EXT,
        type=str,
        default=ARGDEF_CHECKFILE_EXT,
        help=' '.join([
            "File extension of checkfile(s), unless argument {} is used, in which case the extension"
            "is considered to be included/excluded in the provided checkfile file path."
        ])
    )
    parser.add_argument(
        ARGSTR_ERRFILE_EXT,
        type=str,
        default=ARGDEF_CHECKERROR_EXT,
        help=' '.join([
            "File extension of error files created when source files are deemed invalid during",
            "checking procedures, containing error messages describing issues with the source file.",
            "The full file path of an error file is constructed by simply appending this string",
            "to the full file path of the corresponding source file."
        ])
    )

    parser.add_argument(
        ARGSTR_ALLOW_MISSING_SUFFIX,
        action='store_true',
        help=' '.join([
            "Allow checking of check groups that are missing source file suffixes."
        ])
    )
    parser.add_argument(
        ARGSTR_ALLOW_MISSING_ORTHO2,
        action='store_true',
        help=' '.join([
            "Allow checking of SETSM DEM check groups that are missing the '{}' source file.".format(
                CHECK_SPECIAL_DEM_SUFFIX_ORTHO2
            )
        ])
    )
    parser.add_argument(
        ARGSTR_RETRY_ERRORS,
        action='store_true',
        help=' '.join([
            "Attempt checking source files & groups with existing error files."
        ])
    )
    parser.add_argument(
        ARGSTR_KEEP_CHECKFILE_WITH_ERRORS,
        action='store_true',
        help=' '.join([
            "Continue writing group checkfile after errors in source files have been discovered."
        ])
    )

    parser.add_argument(
        ARGSTR_SUPPRESS_ERRFILE_EXISTS,
        action='store_true',
        help=' '.join([
            "Suppress printing all cases of existing error files among check group source files."
        ])
    )
    parser.add_argument(
        ARGSTR_SUPPRESS_MISSING_SUFFIX,
        action='store_true',
        help=' '.join([
            "Suppress printing all cases of source file suffixes missing from check group."
        ])
    )
    parser.add_argument(
        ARGSTR_SUPPRESS_MISSING_CHECKED,
        action='store_true',
        help=' '.join([
            "Suppress printing all files that are listed in checkfiles but cannot be found in source directory."
        ])
    )
    parser.add_argument(
        ARGSTR_SUPPRESS_NEW_SOURCE,
        action='store_true',
        help=' '.join([
            "Suppress printing all new source files that are to be added to existing checkfiles."
        ])
    )

    parser.add_argument(
        ARGSTR_REMOVE_TYPE,
        type=str,
        choices=ARGCHO_REMOVE_TYPE,
        default=ARGCHO_REMOVE_TYPE_CHECKFILES,
        help=' '.join([
            "Specify which files can be removed by the following arguments:",
            ARGSTR_RMWHERE_ERRFILE_EXISTS,
            ARGSTR_RMWHERE_MISSING_SUFFIX,
            ARGSTR_RMWHERE_MISSING_CHECKED,
            ARGSTR_RMWHERE_NEW_SOURCE
        ])
    )

    parser.add_argument(
        ARGSTR_RMWHERE_ERRFILE_EXISTS,
        action='store_true',
        help=' '.join([
            "Remove existing check/source files when error files exist among check group source files.",
            "Use {} argument to specify which files can be removed.".format(ARGSTR_REMOVE_TYPE)
        ])
    )
    parser.add_argument(
        ARGSTR_RMWHERE_MISSING_SUFFIX,
        action='store_true',
        help=' '.join([
            "Remove existing check/source files when source file suffixes are missing from check group.",
            "Use {} argument to specify which files can be removed.".format(ARGSTR_REMOVE_TYPE)
        ])
    )
    parser.add_argument(
        ARGSTR_RMWHERE_MISSING_CHECKED,
        action='store_true',
        help=' '.join([
            "Remove existing check/source files when files listed in checkfile cannot be found in source directory.",
            "Use {} argument to specify which files can be removed.".format(ARGSTR_REMOVE_TYPE)
        ])
    )
    parser.add_argument(
        ARGSTR_RMWHERE_NEW_SOURCE,
        action='store_true',
        help=' '.join([
            "Remove existing check/source files when new source files are to be added to checkfile.",
            "Use {} argument to specify which files can be removed.".format(ARGSTR_REMOVE_TYPE)
        ])
    )

    parser.add_argument(
        ARGSTR_REMOVE_ONLY,
        action='store_true',
        help="Scan check/source files and possibly perform removal actions, then exit."
    )
    parser.add_argument(
        ARGSTR_STATS_ONLY,
        action='store_true',
        help="Scan check/source files and report task completion status, then exit."
    )

    parser.add_argument(
        ARGSTR_SCHEDULER,
        type=str,
        choices=script_utils.SCHED_SUPPORTED,
        default=None,
        help="Submit tasks to job scheduler."
    )
    parser.add_argument(
        ARGSTR_JOBSCRIPT,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_JOBSCRIPT,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=True),
        default=None,
        help=' '.join([
            "Script to run in job submission to scheduler.",
            "(default scripts are found in {})".format(JOBSCRIPT_DIR)
        ])
    )
    parser.add_argument(
        ARGSTR_TASKS_PER_JOB,
        type=int,
        choices=None,
        default=None,
        help=' '.join([
            "Number of tasks to bundle into a single job.",
            "(requires {} option)".format(ARGSTR_SCHEDULER)
        ])
    )
    parser.add_argument(
        ARGSTR_SCRATCH,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_SCRATCH,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=False),
        default=ARGDEF_SCRATCH,
        help="Scratch directory to build task bundle text files."
    )
    parser.add_argument(
        ARGSTR_WD,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_WD,
            existcheck_fn=os.path.isdir,
            existcheck_reqval=True),
        default=None,
        help=' '.join([
            "Copy source files to this directory before checking, run checks on these copies,",
            "then clean up the copies before moving on.",
            "At PGC, this argument is meant to be used with {} argument to minimize the impact of".format(ARGSTR_SCHEDULER),
            "file I/O on the network."
        ])
    )
    parser.add_argument(
        ARGSTR_LOGDIR,
        type=script_utils.ARGTYPE_PATH(
            argstr=ARGSTR_LOGDIR,
            existcheck_fn=os.path.isfile,
            existcheck_reqval=False),
        default=None,
        help=' '.join([
            "Directory to which standard output/error log files will be written for batch job runs.",
            "\nIf not provided, default scheduler (or jobscript #CONDOPT_) options will be used.",
            "\n**Note:** Due to implementation difficulties, this directory will also become the",
            "working directory for the job process. Since relative path inputs are always changed",
            "to absolute paths in this script, this should not be an issue."
        ])
    )
    parser.add_argument(
        ARGSTR_EMAIL,
        type=script_utils.ARGTYPE_BOOL_PLUS(
            parse_fn=str),
        nargs='?',
        help="Send email to user upon end or abort of the LAST SUBMITTED task."
    )

    parser.add_argument(
        ARGSTR_DO_DELETE,
        action='store_true',
        help="Perform file removal actions."
    )
    parser.add_argument(
        ARGSTR_DRYRUN,
        action='store_true',
        help="Print actions without executing."
    )

    return parser


def endswith_one_of_coll(check_string, string_ending_coll, case_sensitive=True, return_match=False):
    for s_end in string_ending_coll:
        if check_string.endswith(s_end) or (not case_sensitive and check_string.lower().endswith(s_end.lower())):
            return s_end if return_match else True
    return None if return_match else False


def ends_one_of_coll(string_ending, string_coll, case_sensitive=True, return_match=False):
    for s in string_coll:
        if s.endswith(string_ending) or (not case_sensitive and s.lower().endswith(string_ending.lower())):
            return s if return_match else True
    return None if return_match else False


def checkfile_incomplete(args,
                         checkfile_root, checkfile_ext, errfile_ext, src_suffixes,
                         src_rasters=None, return_incomplete_src_rasters=False,
                         srcfile_count=None, errfile_count=None,
                         missing_suffix_flag=None, checkfile_removed_flag=None,
                         warn_missing_suffix=True, warn_errfile_exists=True,
                         warn_missing_checked=True, warn_new_source=True):

    checkfile = checkfile_root+checkfile_ext if checkfile_ext is not None else checkfile_root
    if checkfile_ext is None and src_rasters is None:
        raise DeveloperError("Checkfile {}; cannot locate corresponding source files when checkfile"
                             "is a full file path (assuming argument {} was provided)".format(checkfile, ARGSTR_CHECKFILE))
    checkfile_dir = os.path.dirname(checkfile)
    checkfile_exists = os.path.isfile(checkfile)
    if src_rasters is not None and type(src_rasters) is list:
        src_rasters = set(src_rasters)

    checkfname = os.path.basename(checkfile)
    check_group_is_xtrack = checkfname[1].isdigit()

    find_src_rasters = (   return_incomplete_src_rasters
                        or warn_missing_suffix or args.get(ARGSTR_RMWHERE_MISSING_SUFFIX)
                        or warn_errfile_exists or args.get(ARGSTR_RMWHERE_ERRFILE_EXISTS))
    delete_files = False

    if checkfile_exists and not args.get(ARGSTR_CHECKFILE_OFF):
        with open(checkfile, 'r') as checkfile_fp:
            src_rasters_checked = set(checkfile_fp.read().splitlines())

        if src_rasters is None:
            src_rasters = {os.path.basename(f) for f in glob.glob(checkfile_root+'*') if endswith_one_of_coll(f, src_suffixes)}

        src_rasters_to_check = src_rasters.difference(src_rasters_checked)
        if src_rasters_to_check:
            warnings.warn("There are more (new?) source files to be added to an existing checkfile")
            if warn_new_source:
                eprint("Checkfile {}; {} more (new?) source files are to be added to existing checkfile".format(
                    checkfile, len(src_rasters_to_check)))
                for f in sorted(list(src_rasters_to_check)):
                    eprint(f)
            delete_files = (delete_files or args.get(ARGSTR_RMWHERE_NEW_SOURCE))

        src_rasters_checked_missing = src_rasters_checked.difference(src_rasters)
        if src_rasters_checked_missing:
            warnings.warn("Files listed in a checkfile were not captured in source selection")
            if warn_missing_checked:
                eprint("Checkfile {}; {} source files listed in checkfile are missing from source selection:".format(
                    checkfile, len(src_rasters_checked_missing)))
                for f in sorted(list(src_rasters_checked_missing)):
                    eprint(f)
            delete_files = (delete_files or args.get(ARGSTR_RMWHERE_MISSING_CHECKED))

    elif return_incomplete_src_rasters or find_src_rasters:
        if src_rasters is None:
            src_rasters = {os.path.basename(f) for f in glob.glob(checkfile_root+'*') if endswith_one_of_coll(f, src_suffixes)}
        src_rasters_to_check = src_rasters

    else:
        src_rasters_to_check = True

    if src_rasters is not None:
        check_special_missing_subgroups = [None]
        if args.get(ARGSTR_CHECK_SPECIAL) is not None and args.get(ARGSTR_CHECK_SPECIAL) in ARGCHOSET_CHECK_SPECIAL_SUBGROUP_DICT:
            check_special_missing_subgroups = check_special_missing_subgroups + ARGCHOSET_CHECK_SPECIAL_SUBGROUP_DICT[args.get(ARGSTR_CHECK_SPECIAL)]

        if type(srcfile_count) is list and len(srcfile_count) == 1:
            srcfile_count[0] = len(src_rasters)

        for check_special_option in check_special_missing_subgroups:
            if check_special_option is None:
                cssgroup_ffileroot_srcfname_dict = {checkfile_root: src_rasters}
                src_suffixes_subgroup = src_suffixes
            else:
                cssgroup_ffileroot_srcfname_dict = dict()
                for check_special_set_argstr, check_special_set_value in ARGCHOSET_CHECK_SPECIAL_SETTING_DICT[check_special_option]:
                    if check_special_set_argstr == ARGSTR_SRC_SUFFIX:
                        src_suffixes = [s.strip() for s in check_special_set_value.split('/')]
                    elif check_special_set_argstr == ARGSTR_CHECKFILE_ROOT_REGEX:
                        subgroup_root_regex = check_special_set_value
                        for srcfname in src_rasters:
                            match = re.match(subgroup_root_regex, srcfname)
                            if match is None:
                                eprint("No regex match for check special subgroup {}='{}' setting {}='{}' with filename: {}".format(
                                    ARGSTR_CHECK_SPECIAL, check_special_option, ARGSTR_CHECKFILE_ROOT_REGEX, subgroup_root_regex.pattern, srcfname
                                ))
                            else:
                                cf_root_name = match.group(1)
                                cf_root_full = os.path.join(checkfile_dir, cf_root_name)
                                if cf_root_full not in cssgroup_ffileroot_srcfname_dict:
                                    cssgroup_ffileroot_srcfname_dict[cf_root_full] = set()
                                cssgroup_ffileroot_srcfname_dict[cf_root_full].add(srcfname)
                    else:
                        eprint("No option to handle check special subgroup {}={} setting {}={}, exiting".format(
                            ARGSTR_CHECK_SPECIAL, check_special_option, check_special_set_argstr, check_special_set_value
                        ))
                        sys.exit(1)
            for checkfile_root_subgroup, src_rasters_subgroup in cssgroup_ffileroot_srcfname_dict.items():
                # if (    (len(src_rasters_subgroup) == 1 and src_rasters_subgroup.pop().endswith('meta.txt'))
                #     and (check_special_option is not None and check_special_option == ARGCHO_CHECK_SPECIAL_SCENEPAIRS)):
                #     warnings.showwarning = script_utils.showwarning_stdout
                #     warnings.warn("Stray metadata file detected in check special 'scene' subgroup."
                #                   " Stray metadata files are ignored for the purpose of flagging"
                #                   " higher-level check special groups as incomplete due to missing suffixes.")
                #     warnings.showwarning = script_utils.showwarning_stderr
                #     continue

                missing_suffixes = [s for s in src_suffixes_subgroup if not ends_one_of_coll(s, src_rasters_subgroup)]
                if missing_suffixes and args.get(ARGSTR_CHECK_SPECIAL) is not None:
                    missing_suffixes_set = set(missing_suffixes)
                    if args.get(ARGSTR_CHECK_SPECIAL) in ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM_SCENELEVEL:
                        if CHECK_SPECIAL_DEM_SUFFIX_SCENELEVEL_MATCHTAG_SET.issubset(missing_suffixes_set):
                            missing_suffixes_set.difference_update(CHECK_SPECIAL_DEM_SUFFIX_SCENELEVEL_MATCHTAG_SET)
                        missing_suffixes_set.difference_update(CHECK_SPECIAL_DEM_SUFFIX_OPTIONAL_SCENELEVEL_SET)
                    if missing_suffixes_set and args.get(ARGSTR_CHECK_SPECIAL) in ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM:
                        if (    CHECK_SPECIAL_DEM_SUFFIX_ORTHO2 in missing_suffixes_set
                            and ((not check_group_is_xtrack) or args.get(ARGSTR_ALLOW_MISSING_ORTHO2))):
                            missing_suffixes_set.remove(CHECK_SPECIAL_DEM_SUFFIX_ORTHO2)
                    if missing_suffixes_set and args.get(ARGSTR_CHECK_SPECIAL) in ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM_STRIPLEVEL:
                        missing_suffixes_set.difference_update(CHECK_SPECIAL_DEM_SUFFIX_OPTIONAL_STRIPLEVEL_SET)
                    missing_suffixes = [s for s in missing_suffixes if s in missing_suffixes_set]

                if missing_suffixes:
                    warnings.warn("Source file suffixes for a check group were not found")
                    if warn_missing_suffix:
                        eprint("Check {}group {}; missing the following source file suffixes: {}".format(
                            "special '{}' sub".format(check_special_option)*(check_special_option is not None),
                            checkfile_root_subgroup, missing_suffixes
                        ))
                    if type(missing_suffix_flag) is list and len(missing_suffix_flag) == 1:
                        missing_suffix_flag[0] = True
                    delete_files = (delete_files or args.get(ARGSTR_RMWHERE_MISSING_SUFFIX))

            if missing_suffix_flag[0] and check_special_option is None:
                break

        src_raster_errfnames = [f+errfile_ext for f in src_rasters if os.path.isfile(os.path.join(checkfile_dir, f+errfile_ext))]
        if src_raster_errfnames:
            warnings.warn("Error files were found among source files for a check group")
            if warn_errfile_exists:
                eprint("Check group {}; {} error files were found among source selection:".format(
                    checkfile_root, len(src_raster_errfnames)))
                for f in sorted(list(src_raster_errfnames)):
                    eprint(f)
            if type(errfile_count) is list and len(errfile_count) == 1:
                errfile_count[0] = len(src_raster_errfnames)
            delete_files = (delete_files or args.get(ARGSTR_RMWHERE_ERRFILE_EXISTS))

        delete_dryrun = (args.get(ARGSTR_DRYRUN) or not args.get(ARGSTR_DO_DELETE))

        if (    (delete_files and checkfile_exists)
            and args.get(ARGSTR_REMOVE_TYPE) in [ARGCHO_REMOVE_TYPE_CHECKFILES, ARGCHO_REMOVE_TYPE_BOTH]):
            eprint("Removing checkfile"+" (dryrun)"*delete_dryrun)
            cmd = "rm {}".format(checkfile)
            if args.get(ARGSTR_DO_DELETE):
                eprint(cmd)
            if not delete_dryrun:
                os.remove(checkfile)
            if type(checkfile_removed_flag) is list and len(checkfile_removed_flag) == 1:
                checkfile_removed_flag[0] = True
            src_rasters_to_check = src_rasters

        if (    delete_files
            and args.get(ARGSTR_REMOVE_TYPE) in [ARGCHO_REMOVE_TYPE_SOURCEFILES, ARGCHO_REMOVE_TYPE_BOTH]):
            eprint("Removing source files"+" (dryrun)"*delete_dryrun)
            srcfnames_to_remove = list(src_rasters) + src_raster_errfnames
            for fn in srcfnames_to_remove:
                srcfile_to_remove = os.path.join(checkfile_dir, fn)
                cmd = "rm {}".format(srcfile_to_remove)
                if args.get(ARGSTR_DO_DELETE):
                    eprint(cmd)
                if not delete_dryrun:
                    os.remove(srcfile_to_remove)
            return -1

    return list(src_rasters_to_check) if return_incomplete_src_rasters else bool(src_rasters_to_check)


def main():

    # Invoke argparse argument parsing.
    arg_parser = argparser_init()
    try:
        args = script_utils.ArgumentPasser(PYTHON_EXE, SCRIPT_FILE, arg_parser, sys.argv)
    except ScriptArgumentError as e:
        arg_parser.error(e)


    ## Further parse/adjust argument values.

    src = args.get(ARGSTR_SRC)
    search_depth = args.get(ARGSTR_DEPTH)
    print(search_depth)
    checkfile_ext = args.get(ARGSTR_CHECKFILE_EXT)
    errfile_ext = args.get(ARGSTR_ERRFILE_EXT)
    allow_missing_suffix = args.get(ARGSTR_ALLOW_MISSING_SUFFIX)
    retry_errors = args.get(ARGSTR_RETRY_ERRORS)
    warn_errfile_exists = (not args.get(ARGSTR_SUPPRESS_ERRFILE_EXISTS) or args.get(ARGSTR_RMWHERE_ERRFILE_EXISTS))
    warn_missing_suffix = (not args.get(ARGSTR_SUPPRESS_MISSING_SUFFIX) or args.get(ARGSTR_RMWHERE_MISSING_SUFFIX))
    warn_missing_checked = (not args.get(ARGSTR_SUPPRESS_MISSING_CHECKED) or args.get(ARGSTR_RMWHERE_MISSING_CHECKED))
    warn_new_source = (not args.get(ARGSTR_SUPPRESS_NEW_SOURCE) or args.get(ARGSTR_RMWHERE_NEW_SOURCE))
    try_removal = (True in args.get(ARGGRP_RMWHERE))
    allow_remove_checkfiles = args.get(ARGSTR_REMOVE_TYPE) in [ARGCHO_REMOVE_TYPE_CHECKFILES, ARGCHO_REMOVE_TYPE_BOTH]
    allow_remove_sourcefiles = args.get(ARGSTR_REMOVE_TYPE) in [ARGCHO_REMOVE_TYPE_SOURCEFILES, ARGCHO_REMOVE_TYPE_BOTH]
    delete_dryrun = (args.get(ARGSTR_DRYRUN) or not args.get(ARGSTR_DO_DELETE))

    if args.get(ARGSTR_SCHEDULER) is not None:
        if args.get(ARGSTR_JOBSCRIPT) is None:
            jobscript_default = os.path.join(JOBSCRIPT_DIR, 'head_{}.sh'.format(args.get(ARGSTR_SCHEDULER)))
            if not os.path.isfile(jobscript_default):
                arg_parser.error(
                    "Default jobscript ({}) does not exist, ".format(jobscript_default)
                    + "please specify one with {} argument".format(ARGSTR_JOBSCRIPT))
            else:
                args.set(ARGSTR_JOBSCRIPT, jobscript_default)
                print("argument {} set automatically to: {}".format(ARGSTR_JOBSCRIPT, args.get(ARGSTR_JOBSCRIPT)))


    ## Validate argument values.

    argstr_mutexl_checkfile = [
        ARGSTR_CHECKFILE,
        ARGSTR_CHECKFILE_ROOT,
        ARGSTR_CHECKFILE_ROOT_REGEX,
        ARGSTR_CHECK_SPECIAL
    ]
    argstr_incompat_sched = [ARGSTR_CHECKFILE, ARGSTR_CHECKFILE_ROOT]

    if args.get(argstr_mutexl_checkfile).count(None) < (len(argstr_mutexl_checkfile)-1):
        arg_parser.error("Only one of the following checkfile arguments may be provided: {}".format(argstr_mutexl_checkfile))

    if args.get(ARGSTR_CHECK_SPECIAL) is not None:
        check_special_option = args.get(ARGSTR_CHECK_SPECIAL)
        for check_special_set_argstr, check_special_set_value in ARGCHOSET_CHECK_SPECIAL_SETTING_DICT[check_special_option]:
            if args.provided(check_special_set_argstr):
                continue
            if check_special_option in ARGCHOGRP_CHECK_SPECIAL_SETSM_DEM_SCENELEVEL and check_special_set_argstr == ARGSTR_SRC_SUFFIX:
                check_special_set_value = '/'.join([
                    ARGCHOSET_CHECK_SPECIAL_DEMTYPE_SUFFIX_DICT[args.get(ARGSTR_CHECK_SPECIAL_DEMTYPE)],
                    check_special_set_value
                ])
            args.set(check_special_set_argstr, check_special_set_value)
            print("via provided argument {}={}, argument {} set automatically to: '{}'".format(
                ARGSTR_CHECK_SPECIAL, args.get(ARGSTR_CHECK_SPECIAL),
                check_special_set_argstr, args.get(check_special_set_argstr)))

    for removal_argstr in ARGGRP_REQUIRES_RMWHERE:
        if args.get(removal_argstr) and not try_removal:
            arg_parser.error("{} option can only be used in conjunction with one of the following "
                             "removal arguments: {}".format(removal_argstr, ARGGRP_RMWHERE))

    if args.get(ARGSTR_SCHEDULER) is not None and args.get(argstr_incompat_sched).count(None) < len(argstr_incompat_sched):
        arg_parser.error("{} option is incompatible with the following arguments: {}".format(
            ARGSTR_SCHEDULER, argstr_incompat_sched
        ))
    if args.get(ARGSTR_TASKS_PER_JOB) is not None and not args.get(ARGSTR_SCHEDULER):
        arg_parser.error("{} option requires {} option".format(ARGSTR_TASKS_PER_JOB, ARGSTR_SCHEDULER))

    src_suffixes = [s.strip() for s in args.get(ARGSTR_SRC_SUFFIX).split('/')]

    if (    endswith_one_of_coll(SETSM_META_SUFFIX, src_suffixes, case_sensitive=False)
        and args.get(ARGSTR_CHECK_SPECIAL) not in ARGCHOGRP_CHECK_SPECIAL_SETSM):
        arg_parser.error("argument {} suffix '{}' that could match SETSM meta suffix '{}' "
                         "may only be provided when argument {} is set to one of the following SETSM options: {}".format(
            ARGSTR_SRC_SUFFIX, endswith_one_of_coll(SETSM_META_SUFFIX, src_suffixes, case_sensitive=False, return_match=True),
            SETSM_META_SUFFIX, ARGSTR_CHECK_SPECIAL, ARGCHOGRP_CHECK_SPECIAL_SETSM
        ))

    checkfile_root_regex = (re.compile(args.get(ARGSTR_CHECKFILE_ROOT_REGEX))
                            if args.get(ARGSTR_CHECKFILE_ROOT_REGEX) is not None else None)


    ## Scan source dir/file input to determine which source files should be checked.
    checkffileroot_srcfnamechecklist_dict = None
    srcffile_checklist = None
    num_srcfiles = 0
    num_checkgroups = None
    srcfile_count = [None]
    errfile_count = [None]
    missing_suffix_flag = [False]
    checkfile_removed_flag = [False]

    print("-----")
    if not args.get(ARGSTR_CHECKFILE_OFF):
        print("Checkfile extension: {}".format(checkfile_ext))
    print("Error file extension: {}".format(errfile_ext))
    print("Accepted source file suffixes: {}".format(src_suffixes))
    print("-----")
    print("Any check group warnings would appear here:")

    srcdir = None

    if os.path.isdir(src):
        srcdir = src

        if (    args.get(ARGSTR_CHECKFILE_ROOT_REGEX) is not None
            and args.get(ARGSTR_CHECK_SPECIAL) != ARGCHO_CHECK_SPECIAL_ALL_SEPARATE):
            checkffileroot_srcfnamechecklist_dict = dict()
            for root, dnames, fnames in os.walk(srcdir):
                if root[len(srcdir):].count(os.sep) < search_depth:
                    for srcfname in fnames:
                        if endswith_one_of_coll(srcfname, src_suffixes):
                            match = re.match(checkfile_root_regex, srcfname)
                            if match is None:
                                eprint("No regex match for filename matching suffix criteria in source directory: {}".format(srcfname))
                            else:
                                cf_root_name = match.group(1)
                                cf_root_full = os.path.join(root, cf_root_name)
                                if cf_root_full not in checkffileroot_srcfnamechecklist_dict:
                                    checkffileroot_srcfnamechecklist_dict[cf_root_full] = []
                                checkffileroot_srcfnamechecklist_dict[cf_root_full].append(srcfname)

        elif args.get(ARGSTR_CHECKFILE_ROOT) is not None:
            checkffileroot_srcfnamechecklist_dict = dict()
            cf_root_full = os.path.join(srcdir, args.get(ARGSTR_CHECKFILE_ROOT))
            checkffileroot_srcfnamechecklist_dict[cf_root_full] = [
                os.path.basename(f) for f in glob.glob(cf_root_full+'*') if endswith_one_of_coll(f, src_suffixes)]

        else:  # if argument --checkfile was provided or if each source raster is allotted a checkfile
            srcffile_checklist = []
            for root, dnames, fnames in os.walk(srcdir):
                if root[len(srcdir):].count(os.sep) < search_depth:
                    for srcfname in fnames:
                        if endswith_one_of_coll(srcfname, src_suffixes):
                            srcffile_checklist.append(os.path.join(root, srcfname))
            missing_suffixes = [s for s in src_suffixes if not ends_one_of_coll(s, srcffile_checklist)]
            if missing_suffixes:
                warnings.warn("Source file suffixes were not found")
                if warn_missing_suffix:
                    eprint("Source directory is missing the following file suffixes: {}".format(missing_suffixes))
                    missing_suffix_flag[0] = True

    elif os.path.isfile(src):
        if src.endswith('.txt') and not src.endswith(ARGCHOSET_CHECK_SPECIAL_DEM_SUFFIX_META):
            bundle_file = src
            task_list = script_utils.read_task_bundle(bundle_file)
            if args.get(ARGSTR_CHECK_SPECIAL) == ARGCHO_CHECK_SPECIAL_ALL_SEPARATE:
                srcffile_checklist = task_list
                if args.get(ARGSTR_CHECKFILE_ROOT) is not None:
                    srcffile_checklist = [srcffile for srcffile in srcffile_checklist if
                                          os.path.basename(srcffile.startswith(ARGSTR_CHECKFILE_ROOT))]
                elif args.get(ARGSTR_CHECKFILE_ROOT_REGEX) is not None:
                    srcffile_checklist = [srcffile for srcffile in srcffile_checklist if
                                          re.match(checkfile_root_regex, os.path.basename(srcffile)) is not None]
            else:
                argstr_incompat_srcfile_cfroots = [ARGSTR_CHECKFILE, ARGSTR_CHECKFILE_ROOT]
                if args.get(argstr_incompat_srcfile_cfroots).count(None) < len(argstr_incompat_srcfile_cfroots):
                    arg_parser.error("argument {} text file containing checkfile roots is "
                                     "incompatible with the following arguments: {}".format(
                        ARGSTR_SRC, argstr_incompat_srcfile_cfroots
                    ))
                checkffileroot_list = task_list

                srcffiles = []
                for cff_root in checkffileroot_list:
                    srcffiles.extend(glob.glob(cff_root+'*'))

                if args.get(ARGSTR_CHECKFILE) is not None:
                    srcffile_checklist = srcffiles
                elif args.get(ARGSTR_CHECKFILE_ROOT_REGEX) is not None:
                    checkffileroot_srcfnamechecklist_dict = dict()
                    for srcffile in srcffiles:
                        if endswith_one_of_coll(srcffile, src_suffixes):
                            srcfdir, srcfname = os.path.split(srcffile)
                            match = re.match(checkfile_root_regex, srcfname)
                            if match is None:
                                eprint("No regex match for file matching suffix criteria pulled from "
                                       "source text file containing checkfile roots: {}".format(srcffile))
                            else:
                                cf_root_name = match.group(1)
                                cf_root_full = os.path.join(srcfdir, cf_root_name)
                                if cf_root_full not in checkffileroot_srcfnamechecklist_dict:
                                    checkffileroot_srcfnamechecklist_dict[cf_root_full] = []
                                checkffileroot_srcfnamechecklist_dict[cf_root_full].append(srcfname)
                else:
                    checkffileroot_srcfnamechecklist_dict = {cf_root_full: None for cf_root_full in checkffileroot_list}

                num_srcfiles = None

        else:
            argstr_incompat_srcfile = [ARGSTR_CHECKFILE_ROOT, ARGSTR_CHECKFILE_ROOT_REGEX, ARGSTR_CHECK_SPECIAL]
            if args.get(argstr_incompat_srcfile).count(None) < len(argstr_incompat_srcfile):
                arg_parser.error("argument {} source file is incompatible with the following arguments: {}".format(
                    ARGSTR_SRC, argstr_incompat_srcfile
                ))
            srcffile_checklist = [src]
            warn_missing_checked = False
            warn_missing_suffix = False

    else:
        args.set(ARGSTR_CHECKFILE_ROOT, src)
        srcdir = os.path.dirname(src)
        print("via non-(directory/file) argument {}, argument {} set automatically to: '{}'".format(
            ARGSTR_SRC, ARGSTR_CHECKFILE_ROOT, args.get(ARGSTR_CHECKFILE_ROOT)))
        checkffileroot_srcfnamechecklist_dict = dict()
        cf_root_full = args.get(ARGSTR_CHECKFILE_ROOT)
        checkffileroot_srcfnamechecklist_dict[cf_root_full] = [
            os.path.basename(f) for f in glob.glob(cf_root_full+'*') if endswith_one_of_coll(f, src_suffixes)]

    num_srcfiles_to_check = None
    num_checkgroups_to_check = None
    num_srcfiles_to_run = None
    num_checkgroups_to_run = None
    num_srcfiles_err_exist = 0
    num_srcfiles_err_skip = 0
    num_checkgroups_err_exist = 0
    num_checkgroups_err_skip = 0
    num_srcfiles_suf_skip = 0
    num_checkgroups_suf_miss = 0
    num_checkgroups_suf_skip = 0
    num_srcfiles_removed = 0
    num_checkgroups_removed = 0
    num_checkfiles_removed = 0

    check_items = None

    if checkffileroot_srcfnamechecklist_dict is not None:

        num_checkgroups = len(checkffileroot_srcfnamechecklist_dict.keys())

        return_incomplete_src_rasters = (args.get(ARGSTR_SCHEDULER) is None)
        if return_incomplete_src_rasters:
            num_srcfiles_to_check = 0
            num_srcfiles_to_run = 0
        num_checkgroups_to_check = 0
        num_checkgroups_to_run = 0

        for cff_root in checkffileroot_srcfnamechecklist_dict:
            cff_root_src_rasters = checkffileroot_srcfnamechecklist_dict[cff_root]

            srcfile_count[0] = None
            errfile_count[0] = None
            missing_suffix_flag[0] = False
            checkfile_removed_flag[0] = False

            checkffileroot_srcfnamechecklist_dict[cff_root] = checkfile_incomplete(args,
                cff_root, checkfile_ext, errfile_ext, src_suffixes,
                checkffileroot_srcfnamechecklist_dict[cff_root], return_incomplete_src_rasters,
                srcfile_count, errfile_count,
                missing_suffix_flag, checkfile_removed_flag,
                warn_missing_suffix, warn_errfile_exists,
                warn_missing_checked, warn_new_source
            )
            if checkfile_removed_flag[0]:
                num_checkfiles_removed += 1

            cff_root_src_rasters_to_check = checkffileroot_srcfnamechecklist_dict[cff_root]
            if type(cff_root_src_rasters_to_check) is int and cff_root_src_rasters_to_check == -1:
                checkffileroot_srcfnamechecklist_dict[cff_root] = None
                num_checkgroups -= 1
                num_checkgroups_removed += 1
                num_srcfiles_removed += srcfile_count[0]
                continue
            elif srcfile_count[0] is not None:
                num_srcfiles += srcfile_count[0]

            if (    cff_root_src_rasters is not None
                and (   errfile_count[0] is None
                     or (not retry_errors and args.get(ARGSTR_CHECKFILE_OFF) and type(cff_root_src_rasters_to_check) is list))):
                cff_dir = os.path.join(os.path.dirname(cff_root))
                srcfname_errlist = [fn for fn in cff_root_src_rasters if os.path.isfile(os.path.join(cff_dir, fn+errfile_ext))]
                errfile_count[0] = len(srcfname_errlist)

            if errfile_count[0] is not None:
                num_srcfiles_err_exist += errfile_count[0]

            if cff_root_src_rasters_to_check:
                num_checkgroups_to_check += 1
            if type(cff_root_src_rasters_to_check) is list:
                num_srcfiles_to_check_this_group = len(cff_root_src_rasters_to_check)
                num_srcfiles_to_check += num_srcfiles_to_check_this_group
            else:
                num_srcfiles_to_check_this_group = None

            if (   (not allow_missing_suffix and missing_suffix_flag[0])
                or (not retry_errors and errfile_count[0])):
                cff_root_src_rasters_to_check_backup = cff_root_src_rasters_to_check
                if not retry_errors and errfile_count[0]:
                    if args.get(ARGSTR_CHECKFILE_OFF):
                        if type(cff_root_src_rasters_to_check) is list:
                            cff_root_src_rasters_to_check = list(set(cff_root_src_rasters_to_check).difference(set(srcfname_errlist)))
                            num_srcfiles_err_skip += (num_srcfiles_to_check_this_group - len(cff_root_src_rasters_to_check))
                            if len(cff_root_src_rasters_to_check) == 0:
                                if num_srcfiles_to_check_this_group > 0:
                                    num_checkgroups_err_skip += 1
                    else:
                        if type(cff_root_src_rasters_to_check) is list:
                            cff_root_src_rasters_to_check = []
                            num_srcfiles_err_skip += num_srcfiles_to_check_this_group
                            num_checkgroups_err_exist += 1
                            if num_srcfiles_to_check_this_group > 0:
                                num_checkgroups_err_skip += 1
                        else:
                            num_checkgroups_err_exist += 1
                            if cff_root_src_rasters_to_check:
                                cff_root_src_rasters_to_check = False
                                num_checkgroups_err_skip += 1
                    checkffileroot_srcfnamechecklist_dict[cff_root] = cff_root_src_rasters_to_check
                if not allow_missing_suffix and missing_suffix_flag[0]:
                    if type(cff_root_src_rasters_to_check_backup) is list:
                        cff_root_src_rasters_to_check = []
                        num_srcfiles_suf_skip += num_srcfiles_to_check_this_group
                        num_checkgroups_suf_miss += 1
                        if num_srcfiles_to_check_this_group > 0:
                            num_checkgroups_suf_skip += 1
                    else:
                        num_checkgroups_suf_miss += 1
                        if cff_root_src_rasters_to_check_backup:
                            cff_root_src_rasters_to_check = False
                            num_checkgroups_suf_skip += 1
                checkffileroot_srcfnamechecklist_dict[cff_root] = cff_root_src_rasters_to_check

        checkffileroot_srcfnamechecklist_dict = {
            cff_root: f_list for cff_root, f_list in checkffileroot_srcfnamechecklist_dict.items() if f_list}

        check_items = checkffileroot_srcfnamechecklist_dict

        num_checkgroups_to_run = len(checkffileroot_srcfnamechecklist_dict.keys())
        if num_checkgroups_to_run == 0:
            num_srcfiles_to_run = 0
        elif type(next(iter(checkffileroot_srcfnamechecklist_dict))) is list:
            num_srcfiles_to_run = sum([len(file_list) for file_list in checkffileroot_srcfnamechecklist_dict.values()])

    elif srcffile_checklist is not None:
        num_srcfiles = len(srcffile_checklist)

        srcffile_errlist = [f for f in srcffile_checklist if os.path.isfile(f+errfile_ext)]
        num_srcfiles_err_exist = len(srcffile_errlist)

        if args.get(ARGSTR_CHECKFILE_OFF):
            num_srcfiles_to_check = len(srcffile_checklist)
        else:
            if args.get(ARGSTR_CHECKFILE):
                num_checkgroups = 1
                srcffile_checklist = checkfile_incomplete(args,
                    args.get(ARGSTR_CHECKFILE), None, errfile_ext, src_suffixes,
                    srcffile_checklist, True,
                    srcfile_count, errfile_count,
                    missing_suffix_flag, checkfile_removed_flag,
                    warn_missing_suffix, warn_errfile_exists,
                    warn_missing_checked, warn_new_source
                )
            else:
                num_checkgroups = num_srcfiles
                srcffile_checklist = [f for f in srcffile_checklist if not os.path.isfile(f+checkfile_ext)]

            num_srcfiles_to_check = len(srcffile_checklist)
            num_checkgroups_to_check = 1 if (args.get(ARGSTR_CHECKFILE) and num_srcfiles_to_check > 0) else num_srcfiles_to_check

        if num_srcfiles_err_exist > 0 and errfile_count[0] is None:
            warnings.warn("Error files were found among source files")
            if warn_errfile_exists:
                eprint("{} error files were found among source selection:".format(num_srcfiles_err_exist))
                for fn in sorted(list(srcffile_errlist)):
                    eprint(fn+errfile_ext)

        if not retry_errors and num_srcfiles_err_exist > 0:
            if args.get(ARGSTR_CHECKFILE):
                srcffile_checklist = []
                num_srcfiles_err_skip = num_srcfiles_to_check
                num_checkgroups_err_skip = num_checkgroups_to_check
            else:
                srcffile_checklist = list(set(srcffile_checklist).difference(set(srcffile_errlist)))
                num_srcfiles_err_skip = num_srcfiles_to_check - len(srcffile_checklist)
                num_checkgroups_err_skip = num_srcfiles_err_skip

        if not allow_missing_suffix and missing_suffix_flag[0]:
            srcffile_checklist = []
            num_srcfiles_suf_skip = num_srcfiles_to_check
            num_checkgroups_suf_skip = num_checkgroups_to_check

        check_items = srcffile_checklist

        num_srcfiles_to_run = len(check_items)
        num_checkgroups_to_run = 1 if (args.get(ARGSTR_CHECKFILE) and num_srcfiles_to_run > 0) else num_srcfiles_to_run

    else:
        raise DeveloperError("Neither `checkffileroot_srcfnamechecklist_dict` "
                             "nor `srcffile_checklist` have been initialized")

    num_errfiles_walk = 0
    print("-----")
    if not args.get(ARGSTR_CHECKFILE_OFF):
        print("Checkfile extension: {}".format(checkfile_ext))
    print("Error file extension: {}".format(errfile_ext))
    print("Accepted source file suffixes: {}".format(src_suffixes))
    if try_removal:
        print("-----")
        print("{} :: {}{}".format(
            ARGSTR_REMOVE_TYPE, args.get(ARGSTR_REMOVE_TYPE),
            " ({} and {})".format(ARGCHO_REMOVE_TYPE_CHECKFILES, ARGCHO_REMOVE_TYPE_SOURCEFILES)*(
                args.get(ARGSTR_REMOVE_TYPE) == ARGCHO_REMOVE_TYPE_BOTH)))
        if allow_remove_checkfiles:
            print("Number of checkfiles removed: {}".format(num_checkfiles_removed))
        if allow_remove_sourcefiles:
            print("Number of check groups removed: {}".format(num_checkgroups_removed))
            print("Total number of source files removed: {}".format(num_srcfiles_removed))
        if delete_dryrun:
            print("(dryrun; must turn on {} and turn off {} to do delete)".format(ARGSTR_DO_DELETE, ARGSTR_DRYRUN))
        if args.get(ARGSTR_REMOVE_ONLY):
            sys.exit(0)
    print("-----")
    if os.path.isdir(src):
        for root, dnames, fnames in os.walk(src):
            if root[len(srcdir):].count(os.sep) < search_depth:
                for srcfname in fnames:
                    if srcfname.endswith(errfile_ext):
                        num_errfiles_walk += 1
        print("{} existing error files found within source directory via os.walk".format(num_errfiles_walk))
    print("{} existing error files found among source selection".format(num_srcfiles_err_exist))
    if num_srcfiles is not None or num_srcfiles_to_check is not None:
        print("Number of source files: {}{}{}{}{}".format(
            num_srcfiles if num_srcfiles is not None else '',
            ', ' if (num_srcfiles is not None and num_srcfiles_to_check is not None) else '',
            '{} to check'.format(num_srcfiles_to_check) if num_srcfiles_to_check is not None else '',
            ' ({} skipped due to missing suffix)'.format(num_srcfiles_suf_skip) if num_srcfiles_suf_skip else '',
            ' ({} skipped due to existing error file)'.format(num_srcfiles_err_skip) if num_srcfiles_err_skip else ''
        ))
    if num_checkgroups is not None:
        print("Number of check groups: {}{}{}, {} to check{}{}".format(
            num_checkgroups,
            ' ({} with missing suffix)'.format(num_checkgroups_suf_miss) if num_checkgroups_suf_miss else '',
            ' ({} with existing error file)'.format(num_checkgroups_err_exist) if num_checkgroups_err_exist else '',
            num_checkgroups_to_check,
            ' ({} skipped due to missing suffix)'.format(num_checkgroups_suf_skip) if num_checkgroups_suf_skip else '',
            ' ({} skipped due to existing error file)'.format(num_checkgroups_err_skip) if num_checkgroups_err_skip else ''
        ))

    if args.get(ARGSTR_STATS_ONLY):
        sys.exit(0)

    print("--> Will run: {}{}{}".format(
        '{} check groups'.format(num_checkgroups_to_run) if num_checkgroups_to_run is not None else '',
        ', ' if (num_srcfiles_to_run is not None and num_checkgroups_to_run is not None) else '',
        '{} source files'.format(num_srcfiles_to_run) if num_srcfiles_to_run is not None else '',
    ))

    if (   (checkffileroot_srcfnamechecklist_dict is not None and len(checkffileroot_srcfnamechecklist_dict) == 0)
        or (srcffile_checklist is not None and len(srcffile_checklist) == 0)):
        sys.exit(0)


    # Pause for user review.
    print("-----")
    wait_seconds = 5
    print("Sleeping {} seconds before task submission".format(wait_seconds))
    sleep(wait_seconds)
    print("-----")


    ## Create output directories if they don't already exist.
    if not args.get(ARGSTR_DRYRUN):
        for dir_argstr, dir_path in list(zip(ARGGRP_OUTDIR, args.get_as_list(ARGGRP_OUTDIR))):
            if dir_path is not None and not os.path.isdir(dir_path):
                print("Creating argument {} directory: {}".format(dir_argstr, dir_path))
                os.makedirs(dir_path)
    if args.get(ARGSTR_CHECKFILE):
        checkfile_dir = os.path.dirname(args.get(ARGSTR_CHECKFILE))
        if not os.path.isdir(checkfile_dir):
            print("Creating directory to contain output checkfile: {}".format(checkfile_dir))
            os.makedirs(checkfile_dir)


    ## Check rasters.

    if check_items is checkffileroot_srcfnamechecklist_dict:
        check_items_sorted = sorted(checkffileroot_srcfnamechecklist_dict)
    elif check_items is srcffile_checklist:
        check_items.sort()
        check_items_sorted = check_items

    if args.get(ARGSTR_SCHEDULER) is not None:
        # Check rasters in batch.

        tasks_per_job = args.get(ARGSTR_TASKS_PER_JOB)
        check_units = (check_items_sorted if tasks_per_job is None else
                       script_utils.write_task_bundles(check_items_sorted, tasks_per_job,
                                                       args.get(ARGSTR_SCRATCH),
                                                        '{}_{}'.format(JOB_ABBREV, ARGSTR_SRC)))

        jobnum_fmt = script_utils.get_jobnum_fmtstr(check_units)
        last_job_email = args.get(ARGSTR_EMAIL)

        args_batch = args
        args_single = copy.deepcopy(args)
        args_single.unset(ARGGRP_BATCH)
        if args.get(ARGSTR_WD) is None and BATCH_ARGDEF_WD is not None:
            args_single.set(ARGSTR_WD, BATCH_ARGDEF_WD)
            print("argument {} set to default value for batch run with {} option: {}".format(
                ARGSTR_WD, ARGSTR_SCHEDULER, args_single.get(ARGSTR_WD)
            ))

        if check_items is srcffile_checklist:
            args_single.set(ARGSTR_CHECK_SPECIAL, ARGCHO_CHECK_SPECIAL_ALL_SEPARATE)
        if args.get(ARGSTR_CHECK_SPECIAL) is not None:
            args_single.unset(ARGGRP_CHECK_REGULAR)

        job_num = 0
        num_jobs = len(check_units)
        for unit in check_units:
            job_num += 1

            args_single.set(ARGSTR_SRC, unit)
            if last_job_email and job_num == num_jobs:
                args_single.set(ARGSTR_EMAIL, last_job_email)
            cmd_single = args_single.get_cmd()

            job_name = JOB_ABBREV+jobnum_fmt.format(job_num)
            cmd = args_single.get_jobsubmit_cmd(
                args_batch.get(ARGSTR_SCHEDULER),
                jobscript=args_batch.get(ARGSTR_JOBSCRIPT),
                jobname=job_name, time_hr=JOB_WALLTIME_HR, memory_gb=JOB_MEMORY_GB, email=args.get(ARGSTR_EMAIL),
                envvars=[args_batch.get(ARGSTR_JOBSCRIPT), JOB_ABBREV, cmd_single, PYTHON_VERSION_ACCEPTED_MIN]
            )

            print(cmd)
            if not args_batch.get(ARGSTR_DRYRUN):
                subprocess.call(cmd, shell=True, cwd=args_batch.get(ARGSTR_LOGDIR))

    else:
        error_trace = None
        try:
            # Check rasters in serial.

            if check_items is checkffileroot_srcfnamechecklist_dict:
                for i, cff_root in enumerate(check_items_sorted):
                    checkfile_dir = os.path.dirname(cff_root)
                    cf_rasterffile_list = [os.path.join(checkfile_dir, rasterfname) for rasterfname in
                                           checkffileroot_srcfnamechecklist_dict[cff_root]]
                    cf_rasterffile_list.sort()
                    checkfile = cff_root+checkfile_ext
                    print("Check group ({}/{}), {} files to check: {}*".format(
                        i+1, num_checkgroups_to_check, len(cf_rasterffile_list), cff_root))
                    if not args.get(ARGSTR_DRYRUN):
                        check_rasters(cf_rasterffile_list, checkfile, args)

            elif check_items is srcffile_checklist:
                for i, src_rasterffile in enumerate(check_items_sorted):
                    checkfile = src_rasterffile+checkfile_ext
                    print("Check source file ({}/{}): {}".format(i+1, num_srcfiles_to_check, src_rasterffile))
                    if not args.get(ARGSTR_DRYRUN):
                        check_rasters(src_rasterffile, checkfile, args)

        except KeyboardInterrupt:
            raise

        except Exception as e:
            with script_utils.capture_stdout_stderr() as out:
                traceback.print_exc()
            caught_out, caught_err = out
            error_trace = caught_err
            eprint(error_trace)
            if e.__class__ is ImportError:
                print("\nFailed to import necessary module(s)")
                print("If running on a Linux system where the jobscripts/init.sh file has been properly"
                      " set up, try running the following command to activate a working environment"
                      " in your current shell session:\n{}".format("source {} {}".format(JOBSCRIPT_INIT, JOB_ABBREV)))
                print('')

        if type(args.get(ARGSTR_EMAIL)) is str:
            # Send email notification of script completion.
            email_body = SCRIPT_RUNCMD
            if error_trace is not None:
                email_status = "ERROR"
                email_body += "\n{}\n".format(error_trace)
            else:
                email_status = "COMPLETE"
            email_subj = "{} - {}".format(email_status, SCRIPT_FNAME)
            script_utils.send_email(args.get(ARGSTR_EMAIL), email_subj, email_body)

        if error_trace is not None:
            sys.exit(1)


def check_rasters(raster_ffiles, checkfile, args):
    import gdal
    import numpy as np
    gdal.UseExceptions()

    if args.get(ARGSTR_CHECKFILE) is not None:
        checkfile = args.get(ARGSTR_CHECKFILE)

    checkfile_write = (not args.get(ARGSTR_CHECKFILE_OFF))
    checkfile_exists = os.path.isfile(checkfile)
    if checkfile_exists:
        print("Checkfile already exists: {}".format(checkfile))

    raster_ffile_list = raster_ffiles
    checkfile_group_fp = None
    if type(raster_ffiles) is not list:
        # Input is a single source file to check.
        raster_ffile_list = [raster_ffiles]
    else:
        # Input is a list of source files in a single check group.
        raster_ffile_list = raster_ffiles
        if checkfile_write:
            if checkfile_exists:
                with open(checkfile, 'r') as checkfile_group_fp:
                    rasters_checked = set(checkfile_group_fp.read().splitlines())
                    rasters_to_check = set([os.path.basename(f) for f in raster_ffile_list])
                    rasters_already_checked = rasters_checked.intersection(rasters_to_check)
                    if len(rasters_already_checked) > 0:
                        raise DeveloperError("The following source files have already been checked: {}".format(
                            rasters_already_checked))
            print("Opening group checkfile in append mode: {}".format(checkfile))
            checkfile_group_fp = open(checkfile, 'a')


    # Check each input source file.
    for raster_ffile in raster_ffile_list:

        raster_ffile_err = raster_ffile+args.get(ARGSTR_ERRFILE_EXT)
        if os.path.isfile(raster_ffile_err):
            print("Removing existing error file: {}".format(raster_ffile_err))
            os.remove(raster_ffile_err)
        errmsg_list = []

        if not os.path.isfile(raster_ffile):
            errmsg_print_and_list(errmsg_list,
                "Source file to check does not exist: {}".format(raster_ffile))

        else:
            if raster_ffile.endswith(SETSM_META_SUFFIX) or raster_ffile.lower().endswith(SETSM_META_SUFFIX.lower()):
                meta_ffile = raster_ffile

                if args.get(ARGSTR_CHECK_SPECIAL) in ARGCHOGRP_CHECK_SPECIAL_SETSM_SCENELEVEL:
                    print("Checking SETSM scene metadata file: {}".format(meta_ffile))
                    try:
                        with open(meta_ffile, 'r') as scenemeta_fp:
                            meta_errmsg_list = check_setsm_meta(scenemeta_fp)
                            errmsg_list = meta_errmsg_list
                    except RuntimeError as e:
                        errmsg_print_and_list(errmsg_list,
                            "Text file read error: {}".format(e))

                elif args.get(ARGSTR_CHECK_SPECIAL) in ARGCHOGRP_CHECK_SPECIAL_SETSM_STRIPLEVEL:
                    print("Checking SETSM strip metadata file: {}".format(meta_ffile))
                    try:
                        with open(meta_ffile, 'r') as stripmeta_fp:
                            in_scenemeta_section = False
                            current_scenemeta_name = None
                            scenemeta_txt = ''
                            for line in stripmeta_fp:
                                if not in_scenemeta_section:
                                    if line.strip() == SETSM_STRIPMETA_SCENEMETA_SECTION_HEADER:
                                        in_scenemeta_section = True
                                elif re.match(SETSM_STRIPMETA_SCENEMETA_ITEM_HEADER_REGEX, line) is not None:
                                    if current_scenemeta_name is not None:
                                        meta_errmsg_list = check_setsm_meta(StringIO(scenemeta_txt))
                                        errmsg_list.extend(["{}: {}".format(current_scenemeta_name, err) for err in meta_errmsg_list])
                                        scenemeta_txt = ''
                                    current_scenemeta_name = line.strip()
                                elif current_scenemeta_name is not None:
                                    scenemeta_txt += line
                            if current_scenemeta_name is not None:
                                meta_errmsg_list = check_setsm_meta(StringIO(scenemeta_txt))
                                errmsg_list.extend(["{}: {}".format(current_scenemeta_name, err) for err in meta_errmsg_list])
                    except RuntimeError as e:
                        errmsg_print_and_list(errmsg_list,
                            "Text file read error: {}".format(e))

                else:
                    errmsg_print_and_list(errmsg_list, ' '.join([
                        "SETSM metadata text file (matching suffix '{}') could not be checked".format(SETSM_META_SUFFIX),
                        "because script argument {} is not one of the following SETSM options: {}".format(
                            ARGSTR_CHECK_SPECIAL, ARGCHOGRP_CHECK_SPECIAL_SETSM)
                    ]))


            elif endswith_one_of_coll(raster_ffile, GDAL_RASTER_SUFFIXES, case_sensitive=False):
                working_on_copy = False
                raster_ffile_wd = None
                try:
                    if args.get(ARGSTR_WD) is not None:
                        raster_ffile_wd = os.path.join(args.get(ARGSTR_WD), os.path.basename(raster_ffile))
                        print("Copying source raster to working directory: {} -> {}".format(raster_ffile, raster_ffile_wd))
                        try:
                            shutil.copy2(raster_ffile, raster_ffile_wd)
                            raster_ffile = raster_ffile_wd
                            working_on_copy = True
                        except shutil.SameFileError as e:
                            raster_ffile_wd = None
                            print(e)

                    print("Checking raster: {}".format(raster_ffile))

                    setsm_suffix = None
                    if args.get(ARGSTR_CHECK_SETSM_VALIDRANGE):
                        for suffix in SETSM_RASTER_SUFFIX_VALIDRANGE_DICT:
                            if raster_ffile.endswith(suffix):
                                setsm_suffix = suffix
                                break

                    try:
                        ds = gdal.Open(raster_ffile, gdal.GA_ReadOnly)
                    except RuntimeError as e:
                        errmsg_print_and_list(errmsg_list,
                            "Raster file read error: {}".format(e))
                        raise RasterFileReadError()

                    num_bands = ds.RasterCount
                    print("{} bands{}".format(
                        num_bands, ', check SETSM suffix: {}'.format(setsm_suffix) if setsm_suffix is not None else ''))

                    if setsm_suffix is not None and num_bands > 1:
                        errmsg_print_and_list(errmsg_list, ' '.join([
                            "SETSM raster has {} bands, more than expected (1 band).".format(num_bands),
                            "All bands will be checked for valid SETSM data range."
                        ]))

                    for band_index in range(num_bands):
                        band_num = band_index + 1
                        band = ds.GetRasterBand(band_num)
                        print("Processing Band {}".format(band_num))

                        if args.get(ARGSTR_CHECK_METHOD) == ARGCHO_CHECK_METHOD_CHECKSUM:
                            try:
                                print("Doing checksum")
                                checksum = band.Checksum()
                                print("Checksum succeeded: {}".format(checksum))
                            except RuntimeError as e:
                                errmsg_print_and_list(errmsg_list,
                                    "Band {} checksum error: {}".format(band_num, e))

                        if args.get(ARGSTR_CHECK_METHOD) == ARGCHO_CHECK_METHOD_READ or setsm_suffix is not None:
                            try:
                                print("Reading band data array")
                                data_array = band.ReadAsArray()
                                print("Data read succeeded")
                            except RuntimeError as e:
                                errmsg_print_and_list(errmsg_list,
                                    "Band {} data read error: {}".format(band_num, e))
                                print("Continuing to next band")
                                continue

                            if setsm_suffix is not None:
                                valid_range = SETSM_RASTER_SUFFIX_VALIDRANGE_DICT[setsm_suffix]
                                nodata_val = band.GetNoDataValue()

                                print("Checking SETSM suffix '{}' valid range: {} (NoData value: {})".format(
                                    setsm_suffix, valid_range, nodata_val))

                                valid_min, valid_max = valid_range
                                data_array_invalid = np.logical_or(data_array < valid_min, data_array > valid_max)
                                if nodata_val is not None:
                                    data_array_nodata = (np.isnan(data_array) if np.isnan(nodata_val)
                                                         else (data_array == nodata_val))
                                    data_array_invalid[data_array_nodata] = False

                                if not np.any(data_array_invalid):
                                    print("SETSM check succeeded")
                                else:
                                    errmsg_print_and_list(errmsg_list,
                                        "Band {} failed SETSM suffix '{}' valid range check of {}".format(
                                        band_num, setsm_suffix, valid_range))

                                    shape = (ds.RasterYSize, ds.RasterXSize)
                                    geo_trans = ds.GetGeoTransform()
                                    grid_x = geo_trans[0] + np.arange(shape[1]) * geo_trans[1]
                                    grid_y = geo_trans[3] + np.arange(shape[0]) * geo_trans[5]

                                    invalid_image_coords = [(i, j) for i, j in np.argwhere(data_array_invalid)]
                                    invalid_geo_coords = [(grid_x[j], grid_y[i]) for i, j in invalid_image_coords]
                                    invalid_values = [v for v in data_array[np.where(data_array_invalid)]]

                                    errmsg_setsm_details_list = [
                                        "Invalid (i, j) image coordinates: {}".format(invalid_image_coords),
                                        "Invalid (x, y) georeferenced coordinates: {}".format(invalid_geo_coords),
                                        "Invalid values: {}".format(invalid_values)
                                    ]
                                    for line in errmsg_setsm_details_list:
                                        print(line)
                                    errmsg_list.extend(errmsg_setsm_details_list)

                except RasterFileReadError:
                    pass
                except:
                    raise
                finally:
                    if args.get(ARGSTR_WD) is not None and working_on_copy and raster_ffile_wd is not None:
                        print("Removing working copy of source raster: {}".format(raster_ffile_wd))
                        os.remove(raster_ffile_wd)

            else:
                # File to check is neither a raster nor a SETSM metadata file.
                # As long as the file exists, it passes.
                pass


        if len(errmsg_list) > 0:
            print("Writing{} error file: {}".format(
                ' over existing' if os.path.isfile(raster_ffile_err) else '', raster_ffile_err))
            with open(raster_ffile_err, 'w') as raster_ffile_err_fp:
                for line in errmsg_list:
                    raster_ffile_err_fp.write(line+'\n')
            if checkfile_write and checkfile_group_fp is not None and not args.get(ARGSTR_KEEP_CHECKFILE_WITH_ERRORS):
                if checkfile_group_fp is not None:
                    checkfile_group_fp.close()
                if os.path.isfile(checkfile):
                    print("Removing checkfile after encountering source file errors: {}".format(checkfile))
                    os.remove(checkfile)
                if checkfile_write:
                    print("No longer writing to checkfile after encountering source file errors: {}".format(checkfile))
                    print("To continue writing to checkfile despite encountering source file errors, "
                          "please pass the {} script argument".format(ARGSTR_KEEP_CHECKFILE_WITH_ERRORS))
                    checkfile_write = False
        else:
            print("Source file passed check(s)")
            if checkfile_write:
                if checkfile_group_fp is None:
                    print("Writing single checkfile: {}".format(checkfile))
                    with open(checkfile, 'w'):
                        pass
                else:
                    print("Adding filename to group checkfile list: {}".format(checkfile))
                    checkfile_group_fp.write(os.path.basename(raster_ffile)+'\n')

    if checkfile_group_fp is not None:
        checkfile_group_fp.close()

    print("Done!")


def check_setsm_meta(meta_fp):
    errmsg_list = []
    meta_txt_buf = meta_fp.read()
    meta_fp.close()

    image1_satID = None
    image2_satID = None
    image1_wv_correct_value = None
    image2_wv_correct_value = None

    for meta_key in SETSM_META_REQUIRED_KEY_SORTED_LIST:
        item_regex, item_is_key_value, item_req_count = SETSM_META_REQUIRED_DICT[meta_key]
        search_message = "Searching metadata text for item '{}' (item regex = {})".format(meta_key, repr(item_regex.pattern))
        print(search_message)
        errmsg_list_this_key = []

        item_matches_stripped = [item.strip() for item in re.findall(item_regex, meta_txt_buf)]
        num_matches = len(item_matches_stripped)

        match_results = "Item '{}'; {} of {} instances found: {}".format(
                meta_key, num_matches, item_req_count, item_matches_stripped)
        print(match_results)

        if num_matches != item_req_count:
            errmsg_print_and_list(errmsg_list_this_key, match_results)

        if not item_is_key_value:
            if len(set([item.lower() for item in item_matches_stripped])) < len(item_matches_stripped):
                errmsg_print_and_list(errmsg_list_this_key,
                    "Item '{}'; duplicate items found: {}""".format(meta_key, item_matches_stripped))

        else:
            item_matches_parts = [[s.strip() for s in item.split('=')] for item in item_matches_stripped]
            split_issue = False
            for item_matches_index, item_parts in enumerate(item_matches_parts):
                if len(item_parts) != 2:
                    split_issue = True
                    errmsg_print_and_list(errmsg_list_this_key,
                        "Key/value item '{}'; splitting item string by '=' character did not result in two parts: {}".format(
                        meta_key, item_matches_stripped[item_matches_index]))

            if not split_issue:
                item_keys_norm = [' '.join(item_parts[0].lower().replace('_', ' ').split()) for item_parts in item_matches_parts]
                item_values = [item_parts[1] for item_parts in item_matches_parts]

                item_keys_contains_image_prefix_count = [
                    key.startswith(SETSM_META_KEY_PREFIX_IMAGE) for key in item_keys_norm
                ].count(True)

                if 0 < item_keys_contains_image_prefix_count < len(item_keys_norm):
                    errmsg_print_and_list(errmsg_list_this_key,
                        "Key/value item '{}'; item matches are inconsistent "
                        "in starting with Image 1/2 prefix: {}".format(meta_key, item_matches_stripped))

                elif item_keys_contains_image_prefix_count > 0:
                    if len(set(item_keys_norm)) < len(item_keys_norm):
                        errmsg_print_and_list(errmsg_list_this_key,
                            "Key/value item '{}'; duplicate keys found: {}".format(meta_key, item_matches_stripped))


                if meta_key == SETSM_META_KEY_IMAGE_PATH:
                    for item_matches_index in range(len(item_matches_stripped)):
                        satID = os.path.basename(item_values[item_matches_index])[0:4].upper()
                        if item_keys_norm[item_matches_index] == SETSM_META_KEY_PREFIX_IMAGE_1:
                            if image1_satID is not None:
                                errmsg_print_and_list(errmsg_list_this_key,
                                    "Key/value item '{}'; two {} keys found: {}".format(
                                    meta_key, SETSM_META_KEY_PREFIX_IMAGE_1, item_matches_stripped))
                                break
                            image1_satID = satID
                        elif item_keys_norm[item_matches_index] == SETSM_META_KEY_PREFIX_IMAGE_2:
                            if image2_satID is not None:
                                errmsg_print_and_list(errmsg_list_this_key,
                                    "Key/value item '{}'; two {} keys found: {}".format(
                                    meta_key, SETSM_META_KEY_PREFIX_IMAGE_2, item_matches_stripped))
                                break
                            image2_satID = satID
                    if image1_satID is None or image2_satID is None:
                        errmsg_print_and_list(errmsg_list_this_key,
                            "Key/value item '{}'; could not parse satID for {}{}{}: {}".format(
                            meta_key,
                            SETSM_META_KEY_PREFIX_IMAGE_1 if image1_satID is None else '',
                            ' or ' if image1_satID is None and image2_satID is None else '',
                            SETSM_META_KEY_PREFIX_IMAGE_2 if image2_satID is None else '',
                            item_matches_stripped))

                elif meta_key in SETSM_META_KEYGRP_GSD:
                    for item_matches_index, value in enumerate(item_values):
                        if float(value) >= 1.5:
                            errmsg_print_and_list(errmsg_list_this_key,
                                "Key/value item '{}'; value {} >= 1.5: {}".format(
                                meta_key, value, item_matches_stripped[item_matches_index]))

                elif meta_key == SETSM_META_KEY_WV_CORRECT:
                    for item_matches_index in range(len(item_matches_stripped)):
                        wv_correct = int(item_values[item_matches_index])
                        if item_keys_norm[item_matches_index].startswith(SETSM_META_KEY_PREFIX_IMAGE_1):
                            if image1_wv_correct_value is not None:
                                errmsg_print_and_list(errmsg_list_this_key,
                                    "Key/value item '{}'; two {} keys found: {}".format(
                                    meta_key, SETSM_META_KEY_PREFIX_IMAGE_1, item_matches_stripped))
                                break
                            image1_wv_correct_value = wv_correct
                        elif item_keys_norm[item_matches_index].startswith(SETSM_META_KEY_PREFIX_IMAGE_2):
                            if image2_wv_correct_value is not None:
                                errmsg_print_and_list(errmsg_list_this_key,
                                    "Key/value item '{}'; two {} keys found: {}".format(
                                    meta_key, SETSM_META_KEY_PREFIX_IMAGE_2, item_matches_stripped))
                                break
                            image2_wv_correct_value = wv_correct
                    if image1_wv_correct_value is None or image2_wv_correct_value is None:
                        errmsg_print_and_list(errmsg_list_this_key,
                            "Key/value item '{}'; could not parse wv_correct value for {}{}{}: {}".format(
                            meta_key,
                            SETSM_META_KEY_PREFIX_IMAGE_1 if image1_wv_correct_value is None else '',
                            ' or ' if image1_wv_correct_value is None and image2_wv_correct_value is None else '',
                            SETSM_META_KEY_PREFIX_IMAGE_2 if image2_wv_correct_value is None else '',
                            item_matches_stripped))


        if len(errmsg_list_this_key) > 0:
            errmsg_list_this_key.insert(0, search_message)
            errmsg_list.extend(errmsg_list_this_key)


    if image1_satID in SETSM_META_WV_CORRECT_SATIDS and image1_wv_correct_value != 1:
        errmsg_print_and_list(errmsg_list, "Image 1 with satID '{}' requires wv_correct application, but {}{}".format(image1_satID,
            'Image 1 {} meta key was not found'.format(SETSM_META_KEY_WV_CORRECT) if image1_wv_correct_value is None else '',
            'Image 1 {} flag value is {}'.format(SETSM_META_KEY_WV_CORRECT, image1_wv_correct_value) if image1_wv_correct_value is not None else ''))
    if image2_satID in SETSM_META_WV_CORRECT_SATIDS and image2_wv_correct_value != 1:
        errmsg_print_and_list(errmsg_list, "Image 2 with satID '{}' requires wv_correct application, but {}{}".format(image2_satID,
            'Image 2 {} meta key was not found'.format(SETSM_META_KEY_WV_CORRECT) if image2_wv_correct_value is None else '',
            'Image 2 {} flag value is {}'.format(SETSM_META_KEY_WV_CORRECT, image2_wv_correct_value) if image2_wv_correct_value is not None else ''))


    return errmsg_list


def errmsg_print_and_list(errmsg_list, errmsg):
    print("ERROR: {}".format(errmsg))
    errmsg_list.append(errmsg)



if __name__ == '__main__':
    main()
