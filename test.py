#!/usr/bin/env python2

# Version 1.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2017


from __future__ import division
import inspect
import os
import platform
import re
import warnings
from glob import glob
from warnings import warn

import ogr, osr
import numpy as np
from PIL import Image
from scipy.misc import imread as scipy_imread
from tifffile import imread, imsave

import mask_scene
import raster_array_tools as rat


SYSTYPE = platform.system()
if SYSTYPE == 'Windows':
    TESTDIR = 'C:/Users/husby036/Documents/Cprojects/test_s2s/testFiles/'
elif SYSTYPE == 'Linux':
    TESTDIR = '/mnt/pgc/data/scratch/erik/test_s2s/testFiles/'

PREFIX_RUNNUM = 'CURRENT_RUNNUM_'

PROJREF_POLAR_STEREO = """PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-70],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]"""


warnings.simplefilter('always', UserWarning)

class InvalidArgumentError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)

class TestingError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


def stringifyThisFunctionForExec(*args):
    exec_script = ''

    this_funcName = inspect.stack()[0][3]
    this_funcReturn = 'return {}('.format(this_funcName)
    caller_funcName = inspect.stack()[1][3]
    caller_funcDef = 'def {}('.format(caller_funcName)

    this_file_fp = open(__file__, 'r')
    line = this_file_fp.readline()
    indent = ''

    # Find the function definition in this file.
    found = False
    while not found and line != '':
        if line.startswith(caller_funcDef):
            found = True
        line = this_file_fp.readline()
    if not found:
        raise TestingError("Could not find function definition matching '{}'".format(caller_funcDef))

    # Find the return statement that called this function.
    found = False
    while not found and line != '':
        if line.lstrip().startswith(this_funcReturn):
            found = True
            # Assuming the return statement is indented once beyond the function definition,
            # capture the indentation schema so that one indent may be removed from every line of
            # the string of code that is returned.
            indent = line[:line.find(this_funcReturn)]
        line = this_file_fp.readline()
    if not found:
        raise TestingError("Could not find return statement matching '{}' within function '{}'".format(
                           this_funcReturn, this_funcName))

    # Add all code that follows that first return statement to a string variable,
    # stopping when the next function definition is read or EOF is reached.
    done = False
    while not done and line != '':
        if line.startswith('def '):
            done = True
        else:
            exec_script += line.replace(indent, '', 1)
            line = this_file_fp.readline()

    this_file_fp.close()

    # Place all arguments into their proper places in the script.
    # NOTE: Arguments must be evaluated to perform these substitutions, SO BE CAREFUL!!
    for i, arg in enumerate(args):
        exec_script = exec_script.replace('__arg{}__'.format(i), arg)

    return exec_script


def cv():
    """
    Check Vars
    *** to be executed while debugging ***
    """
    return stringifyThisFunctionForExec()

    cv_test_vars = None
    cv_test_var = None
    cv_test_expr = None
    cv_test_var_shape = None

    cv_test_vars = (
        'x', 'y', 'z', 'm', 'o', 'md', '-',
        'X', 'Y', 'Z', 'M', 'O', '-',
        'Xsub', 'Ysub', 'Zsub', 'Msub', 'Osub'
    )

    print
    for cv_test_var in cv_test_vars:
        if cv_test_var in vars():

            cv_test_expr = 'str({}.dtype)'.format(cv_test_var)
            print '> {}.dtype = {}'.format(cv_test_var, eval(cv_test_expr))

            cv_test_expr = '{}.shape'.format(cv_test_var)
            cv_test_var_shape = eval(cv_test_expr)
            if len(cv_test_var_shape) == 1:
                cv_test_var_shape = (1, cv_test_var_shape[0])
            print '    shape = {}'.format(str(cv_test_var_shape).replace('L', ''))

            cv_test_expr = 'np.nanmin({})'.format(cv_test_var)
            print '    min = {}'.format(eval(cv_test_expr))

            cv_test_expr = 'np.nanmax({})'.format(cv_test_var)
            print '    max = {}'.format(eval(cv_test_expr))

        elif cv_test_var == '-':
            print '------------------'

    print

    del cv_test_vars, cv_test_var, cv_test_var_shape, cv_test_expr


def sg(varNames_csv):
    """
    Set Globals
    *** to be executed while debugging ***
    ::
    varNames_csv must be a comma-delimited string of variable names
    accessible in the current namespace.
    """
    if type(varNames_csv) != str:
        raise InvalidArgumentError("`varNames_csv` must be a string")
    return stringifyThisFunctionForExec('"{}"'.format(varNames_csv))

    sg_varNames_list = None
    sg_testVname = None
    sg_i = None
    sg_v = None

    if 'sg_testVnames_list' in vars():
        for sg_v in sg_testVnames_list:
            exec('del {}'.format(sg_v))
    sg_testVnames_list = []

    sg_varNames_list = __arg0__.split(',')
    for sg_i, sg_v in enumerate(sg_varNames_list):
        sg_testVname = '{}{}_{}'.format('sg_testVar_', sg_i, sg_v.strip())
        exec('global {}'.format(sg_testVname))
        exec('{} = {}'.format(sg_testVname, sg_v))
        sg_testVnames_list.append(sg_testVname)

    del sg_varNames_list, sg_testVname, sg_i, sg_v


def getTestVarsFromGlobals(debug_globals):
    testVname_pattern_str = "{}\d+_(.+)".format('sg_testVar_')
    testVname_pattern = re.compile(testVname_pattern_str)
    testVar_names = []
    testVar_values = []
    g_keys = debug_globals.keys()
    g_keys.sort()
    for varName in g_keys:
        m = re.match(testVname_pattern, varName)
        if m is not None:
            testVar_names.append(m.group(1))
            testVar_values.append(debug_globals[varName])
    return testVar_names, testVar_values


def splitTupleString(tup_string):
    tup_pattern = re.compile("\((.*)\)")
    search_result = re.search(tup_pattern, tup_string)
    if search_result is None:
        return None
    else:
        return tuple(element.strip() for element in search_result.group(1).split(','))


def splitArgsString(args_string):
    lefts  = '([{'
    rights = ')]}'
    lefts_count  = np.array([0, 0, 0])
    rights_count = np.array([0, 0, 0])
    quotes = ("'", '"')
    curr_string_type = -1

    args = []
    arg_start_index = 0
    i = 0
    for c in args_string:

        if curr_string_type == -1:
            if c == ',':
                if np.array_equal(lefts_count, rights_count):
                    args.append(args_string[arg_start_index:i])
                    arg_start_index = i + 1
            elif c in lefts:
                lefts_count[lefts.find(c)] += 1
            elif c in rights:
                rights_count[rights.find(c)] += 1
            elif c in quotes:
                curr_string_type = quotes.index(c)

        elif c == quotes[curr_string_type]:
            # We've reached the end of a string.
            curr_string_type = -1

        i += 1

    if arg_start_index < i:
        args.append(args_string[arg_start_index:i])

    return tuple(a.strip() for a in args)


# Doesn't work correctly in newest release of Python2.7... :'(
def getCalledFunctionArgs(depth=1, funcName=None):
    stack = inspect.stack()
    func_frame_record = None

    try:
        if depth == 0 or depth == float('inf'):
            if funcName is None:
                raise InvalidArgumentError("`funcName` must be provided when depth is not certain")
            stack_iterable = xrange(len(stack))
            if depth != 0:
                stack_iterable = reversed(stack_iterable)
            for i in stack_iterable:
                fr_funcName = stack[i][3]
                if fr_funcName == funcName:
                    func_frame_record = stack[i+1]
                    break
            if func_frame_record is None:
                raise InvalidArgumentError("`funcName` '{}' could not be found in the stack".format(funcName))
        else:
            try:
                func_frame_record = stack[depth+1]
            except IndexError:
                raise InvalidArgumentError("Invalid `depth` index for stack: {}".format(depth))
            if funcName is not None and stack[depth][3] != funcName:
                raise InvalidArgumentError("`funcName` '{}' could not be found in the stack "
                                           "at index {}".format(funcName, depth))

            funcCall = ''.join([str(line).strip() for line in func_frame_record[4]])

    except:
        print "STACK AT ERROR:"
        for fr in stack:
            print fr
        raise

    args_pattern_str = "\w" if funcName is None else funcName
    args_pattern_str += "\s*\((.+)\)"
    args_pattern = re.compile(args_pattern_str)
    search_result = re.search(args_pattern, funcCall)
    if search_result is None:
        return ()
    else:
        args_string = search_result.group(1)

    return splitArgsString(args_string)


def findFile(testFile):
    if not os.path.isfile(testFile):
        testFile_temp = testFile+'.tif'
        if not os.path.isfile(testFile_temp):
            testFile_temp = os.path.join(TESTDIR, testFile)
            if not os.path.isfile(testFile_temp):
                testFile_temp += '.tif'
                if not os.path.isfile(testFile_temp):
                    raise InvalidArgumentError("Cannot find `testFile`: '{}'".format(testFile))
        testFile = testFile_temp
    return testFile


def getRunnum():
    runnum = -1
    runnumFiles = glob(os.path.join(TESTDIR, PREFIX_RUNNUM+'*'))
    try:
        if len(runnumFiles) == 0:
            runnum = setRunnum()
        elif len(runnumFiles) == 1:
            runnum_fname = os.path.basename(runnumFiles[0])
            runnum = int(runnum_fname[15:18])
        else:
            raise ValueError
    except ValueError:
        raise TestingError(
            "One dummy file must exist in the test directory"
            " with a name indicating the current runnum for comparison!"
            " e.g. 'CURRENT_RUNNUM_001'"
        )
    return runnum


def getLastRunnum():
    testFiles = glob(os.path.join(TESTDIR, 'run*'))
    runnums = [int(os.path.basename(f)[3:6]) for f in testFiles]
    last_runnum = max(runnums) if len(testFiles) > 0 else None
    return last_runnum


def setRunnum(new_runnum=None, increment=False, concurrent=False):
    if new_runnum is None:
        new_runnum = getLastRunnum()
        if new_runnum is None:
            new_runnum = 0
        if increment:
            new_runnum += 1

    runnumFile_new = os.path.join(TESTDIR, PREFIX_RUNNUM+'{:03d}{}'.format(new_runnum, '_CC'*concurrent))

    runnumFiles = glob(os.path.join(TESTDIR, PREFIX_RUNNUM+'*'))
    if len(runnumFiles) == 0:
        runnumFile_fp = open(runnumFile_new, 'w')
        runnumFile_fp.close()
    elif len(runnumFiles) == 1:
        runnumFile_current = runnumFiles[0]
        if concurrent:
            runnumFname_current = os.path.basename(runnumFile_current)
            if '_CC' in runnumFname_current:
                runnumFile_new = os.path.join(TESTDIR, runnumFname_current.replace('_CC', ''))
        os.rename(runnumFile_current, runnumFile_new)
    else:
        getRunnum()  # Get error message from this function.

    return new_runnum


def incRunnum(concurrent=False):
    return setRunnum(increment=True, concurrent=concurrent)


def getNextImgnum(runnum=getRunnum(), compare=False, concurrent=False):
    next_imgnum = -1

    testFiles = glob(os.path.join(TESTDIR, 'run{:03d}_*'.format(runnum)))
    if concurrent:
        testFiles = [f for f in testFiles if '_py_' in f]

    if len(testFiles) == 0:
        next_imgnum = 1 if not compare else None
    else:
        imgnums = [int(os.path.basename(f)[7:10]) for f in testFiles]
        next_imgnum = max(imgnums)+1 if not compare else max(imgnums)

    return next_imgnum


def getTestFileFromFname(fname):
    if not fname.endswith('.tif'):
        fname += '.tif'
    testFile = os.path.join(TESTDIR, fname)
    while os.path.isfile(testFile):
        opt = raw_input("Test file '{}' already exists. Overwrite? (y/n): ".format(testFile.replace(TESTDIR, '{TESTDIR}/')))
        if opt.strip().lower() == 'y':
            break
        else:
            opt = raw_input("Append description to filename (or press [ENTER] to cancel): ")
            if opt == '':
                return None
            else:
                testFile = testFile.replace('.tif', '~'+opt.replace(' ', '-')+'.tif')
    return testFile


def interpretImageRasterFlavor(flavor):
    flavor_name = ''
    image_PILmode = None
    raster_format = None
    raster_nodata = None

    if flavor is not None and flavor != '':
        if flavor in ('dem', 'dem_array', 'z', 'Z', 'Zsub'):
            flavor_name = 'dem'
            image_PILmode = 'F'
            raster_format = 'float32'
            raster_nodata = -9999
        elif flavor in ('match', 'match_array', 'm', 'M', 'Msub'):
            flavor_name = 'match'
            image_PILmode = 'L'
            raster_format = 'uint8'
            raster_nodata = 0
        elif flavor in ('ortho', 'ortho_array', 'o', 'or', 'O', 'Osub'):
            flavor_name = 'ortho'
            image_PILmode = 'I'
            raster_format = 'int16'
            raster_nodata = 0
        elif flavor in ('mask', 'mask_array'):
            flavor_name = 'mask'
            image_PILmode = 'L'
            raster_format = 'uint8'
            raster_nodata = 0
        elif flavor in ('data', 'data_array', 'md', 'd'):
            flavor_name = 'data'
            image_PILmode = 'L'
            raster_format = 'uint8'
            raster_nodata = 0
        elif flavor in ('edge', 'edge_array', 'me', 'e'):
            flavor_name = 'edge'
            image_PILmode = 'L'
            raster_format = 'uint8'
            raster_nodata = 0
        else:
            raise InvalidArgumentError("Invalid `flavor`: {}".format(flavor))

    return flavor_name, image_PILmode, raster_format, raster_nodata


def handleBatchImageRasterAuto(arrays, flavor, matchkey, descr, compare, concurrent, *X_Y_pref):
    # array_names = splitTupleString(getCalledFunctionArgs(2)[0])
    array_names = None
    if type(arrays) == dict:
        # Assume 'arrays' is a dictionary of global variables that contains test array variables.
        array_names, arrays = getTestVarsFromGlobals(arrays)
        if arrays is None:
            raise TestingError("No global variables with accepted test variable names to be found")
    if (flavor == 'auto' or matchkey == 'auto') and array_names is None:
        raise InvalidArgumentError("Global variables with accepted test variable names must be set "
                                   "in order to automatically determine `flavor` or `matchkey`")

    flavor_order = None
    if flavor.startswith('-'):
        if len(flavor) != (1+len(arrays)):
            raise InvalidArgumentError("`flavor` argument starting with '-' must be followed by "
                                       "a number of characters (flavor abbreviations) equal to "
                                       "the number of arrays in `arrays`")
        flavor_order = flavor[1:len(flavor)]
    else:
        flavor_order = [flavor]*len(arrays)

    key_order = [matchkey]*len(arrays)

    if array_names is None:
        array_names = [None]*len(arrays)

    if not X_Y_pref:
        sia(arrays[0], flavor_order[0], key_order[0], descr, compare, concurrent, array_names[0])
        for i in range(1, len(arrays)):
            sia(arrays[i], flavor_order[i], key_order[i], descr, True, concurrent, array_names[i])
    else:
        X, Y, proj_ref = X_Y_pref
        sra(arrays[0], X, Y, flavor_order[0], key_order[0], descr, compare, concurrent, proj_ref, array_names[0])
        for i in range(1, len(arrays)):
            sra(arrays[i], X, Y, flavor_order[i], key_order[i], descr, True, concurrent, proj_ref, array_names[i])
    return


def getImageRasterAutoFname(array, flavor_name, matchkey, descr, compare, concurrent, isRaster):
    runnum = getRunnum()
    imgnum = getNextImgnum(runnum, compare, concurrent)
    if imgnum is None:
        imgnum = 1
    filetype = 'ras' if isRaster else 'img'
    flavor_name = '{:_<5}'.format(flavor_name)
    if matchkey != '':
        matchkey = '_'+matchkey.replace(' ', '-').replace('~', '-')
    if descr != '':
        descr = '~'+descr.replace(' ', '-')

    testFname = 'run{:03d}_{:03d}_py_{}_{}_{}x{}{}{}.tif'.format(
        runnum, imgnum, filetype, flavor_name, array.shape[0], array.shape[1], matchkey, descr
    )
    return testFname


def saveImage(array, fname='testImage_py.tif'):
    testFile = getTestFileFromFname(fname)
    if testFile is None:
        return

    if array.dtype == np.bool:
        image = Image.frombytes(mode='1', size=array.shape[::-1], data=np.packbits(array, axis=1))
        image.save(testFile)
    else:
        imsave(testFile, array)
    print "'{}' saved".format(testFile.replace(TESTDIR, '{TESTDIR}/'))


def sia(array, flavor='auto', matchkey='auto', descr='', compare=False, concurrent=False, array_name=None):
    """
    Save Image Auto
    ::
    Saves an indexed image in the test file directory specified by global TESTDIR.
    """
    if type(array) in (tuple, list, dict):
        # If 'array' is a dictionary, assume it is one of global variables that contains test array variables.
        handleBatchImageRasterAuto(array, flavor, matchkey, descr, compare, concurrent)
        return
    # array_name = getCalledFunctionArgs()[0]
    if (flavor == 'auto' or matchkey == 'auto') and array_name is None:
        raise InvalidArgumentError("`array_name` must be provided to automatically "
                                   "determine `flavor` or `matchkey` for a single array")

    # Determine the correct data type for saving the image data.
    flavor_name = ''
    fmtstr = None
    if flavor is not None:
        if flavor == 'auto':
            flavor = array_name
        flavor_name, _, _, _ = interpretImageRasterFlavor(flavor)

    if matchkey is not None:
        if matchkey == 'auto':
            matchkey = array_name
    else:
        matchkey = ''

    testFname = getImageRasterAutoFname(array, flavor_name, matchkey, descr, compare, concurrent, False)
    saveImage(array, testFname)


def sia_one(array, flavor=None, matchkey=None, descr='', compare=False, concurrent=False, array_name=None):
    """
    Save Image Auto -- (For) One (Image)
    ::
    Saves an indexed image in the test file directory specified by global TESTDIR.
    """
    sia(array, flavor, matchkey, descr, compare, concurrent, array_name)


def saveRaster(Z, X=None, Y=None, fname='testRaster_py.tif',
               proj_ref=None, geotrans_rot_tup=(0, 0),
               like_rasterFile=None,
               nodata_val=None, dtype_out=None):
    testFile = getTestFileFromFname(fname)
    if testFile is None:
        return

    if proj_ref is None:
        warn("No proj_ref argument given to saveRaster()"
             "\n-> Using default global PROJREF_POLAR_STEREO")
        proj_ref = PROJREF_POLAR_STEREO

    rat.saveArrayAsTiff(Z, testFile,
                        X, Y, proj_ref, geotrans_rot_tup,
                        like_rasterFile,
                        nodata_val, dtype_out)
    print "'{}' saved".format(testFile.replace(TESTDIR, '{TESTDIR}/'))


def sra(Z, X, Y, flavor='auto', matchkey='auto', descr='', compare=False, concurrent=False, proj_ref=None, array_name=None):
    """
    Save Raster Auto
    ::
    Saves an indexed raster image in the test file directory specified by global TESTDIR.
    """
    if type(Z) in (tuple, list):
        raise InvalidArgumentError("tuple/list argument for `Z` is not supported")
    elif type(Z) == dict:
        # If 'Z' is a dictionary, assume it is one of global variables that contains test array variables.
        handleBatchImageRasterAuto(Z, flavor, matchkey, descr, compare, concurrent, X, Y, proj_ref)
        return
    # array_name = getCalledFunctionArgs()[0]
    if (flavor == 'auto' or matchkey == 'auto') and array_name is None:
        raise InvalidArgumentError("`array_name` must be provided to automatically "
                                   "determine `flavor` or `matchkey` for a single array `Z`")

    # Determine the correct data type for saving the raster data.
    if flavor == 'auto':
        flavor = array_name
    flavor_name, _, fmtstr, nodata = interpretImageRasterFlavor(flavor)

    if matchkey is not None:
        if matchkey == 'auto':
            matchkey = array_name
    else:
        matchkey = ''

    testFname = getImageRasterAutoFname(Z, flavor_name, matchkey, descr, compare, concurrent, True)
    if nodata is not None:
        Z_copy = np.copy(Z)
        Z_copy[np.where(np.isnan(Z_copy))] = nodata
    else:
        Z_copy = Z
    saveRaster(Z_copy, X, Y, fname=testFname, proj_ref=proj_ref, nodata_val=nodata, dtype_out=fmtstr)


def waitForComparison(expected_imgnum):

    last_imgnum = getNextImgnum(compare=True, concurrent=True)
    if last_imgnum != expected_imgnum:
        warn("last_imgnum ({}) != expected_imgnum ({}) in test file comparison!!".format(last_imgnum, expected_imgnum))

    FILE_COMPARE_READY = os.path.join(TESTDIR, 'COMPARE_READY')
    FILE_COMPARE_WAIT  = os.path.join(TESTDIR, 'COMPARE_WAIT')

    # Wait for the concurrent MATLAB script to catch up, if necessary.
    while not os.path.isfile(FILE_COMPARE_READY):
        pass
    # Wait for the user to end comparison in MATLAB before continuing.
    while os.path.isfile(FILE_COMPARE_WAIT):
        pass
    # Take care of removing the READY file so that weird things don't happen.
    os.remove(FILE_COMPARE_READY)

    return


def readImage(imgFile='testImage_ml.tif'):
    try:
        in_array = imread(findFile(imgFile))
    except (ValueError, Image.DecompressionBombWarning):
        warn("Error in reading image with tifffile.imread()"
             "\n-> Assuming image is logical; opening with scipy.misc.imread() and casting array to np.bool")
        in_array = scipy_imread(findFile(imgFile)).astype(np.bool)
    return in_array


def readRasterZ(rasterFile='testRaster_ml.tif'):
    return rat.extractRasterParams(findFile(rasterFile), 'z')


def doMasking(matchFile):
    mask_scene.generateMasks(findFile(matchFile))


def getFP(demFile):
    demFile = findFile(demFile)

    Z, X, Y = rat.extractRasterParams(demFile, 'z', 'x', 'y')
    fp_vertices = rat.getFPvertices(Z, X, Y, nodata_val=-9999)
    num = len(fp_vertices[0])

    test_str = (
"""demFile: {}
Strip Footprint Vertices
X: {}
Y: {}
""".format(
        demFile,
        str(fp_vertices[0]).replace(',', '')[1:-1],
        str(fp_vertices[1]).replace(',', '')[1:-1],
        )
    )


    return num, test_str


def saveDBP(demFile):
    demFile = findFile(demFile)
    shapefileFile = demFile.replace('dem.tif', 'dem_boundary.shp')

    Z, X, Y, proj_ref = rat.extractRasterParams(demFile, 'z', 'x', 'y', 'proj_ref')
    poly = rat.getDataBoundariesPoly(Z, X, Y, nodata_val=-9999)
    if not poly:
        raise TestingError("Failed to create data cluster boundaries polygon")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(shapefileFile)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj_ref)

    layer = ds.CreateLayer("data_clusters", srs, ogr.wkbPolygon)
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)

    # Dereference datasource to initiate write to disk.
    ds = None
    print "'{}' saved".format(shapefileFile)
