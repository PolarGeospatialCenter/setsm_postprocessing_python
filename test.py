# Version 1.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2017

from __future__ import division
import inspect
import os
import platform
import re
from glob import glob

import ogr, osr
import numpy as np
from scipy import misc

from mask_scene import generateMasks
import raster_array_tools as rat


SYSTYPE = platform.system()
if SYSTYPE == 'Windows':
    TESTDIR = 'C:/Users/husby036/Documents/Cprojects/test_s2s/testFiles/'
elif SYSTYPE == 'Linux':
    TESTDIR = '/mnt/pgc/data/scratch/erik/test_s2s/testFiles/'

PREFIX_RUNNUM = 'CURRENT_RUNNUM_'

PROJREF_POLAR_STEREO = """PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-70],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]"""


class InvalidArgumentError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

class TestingError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)


def stringifyThisFunctionForExec():
    exec_script = ''

    this_funcName = inspect.stack()[0][3]
    this_funcReturn = 'return {}()'.format(this_funcName)
    caller_funcName = inspect.stack()[1][3]
    caller_funcDef = 'def {}('.format(caller_funcName)

    this_file = open(__file__, 'r')
    line = this_file.readline()
    indent = ''

    # Find the function definition in this file.
    found = False
    while line != '' and not found:
        if line.startswith(caller_funcDef):
            found = True
        line = this_file.readline()
    if not found:
        raise TestingError("Could not find function definition matching '{}'".format(caller_funcName))

    # Find the return statement that called this function.
    found = False
    while line != '' and not found:
        if line.lstrip().startswith(this_funcReturn):
            found = True
            # Assuming the return statement is indented once beyond the function definition,
            # capture the indentation schema so that one indent may be removed from every line of
            # the string of code that is returned.
            indent = line[:line.find(this_funcReturn)]
        line = this_file.readline()
    if not found:
        raise TestingError("Could not find return statement matching '{}' within function '{}'".format(
                           this_funcReturn, caller_funcDef))

    # Add all code that follows that first return statement to a string variable,
    # stopping when the next function definition is read or EOF is reached.
    done = False
    while line != '' and not done:
        if line.startswith('def '):
            done = True
        else:
            exec_script += line.replace(indent, '', 1)
            line = this_file.readline()

    return exec_script


def checkVars():
    return stringifyThisFunctionForExec()

    array_vars = (
        'x', 'y', 'z', 'm', 'o', 'md', '-',
        'X', 'Y', 'Z', 'M', 'O', '-',
        'Xsub', 'Ysub', 'Zsub', 'Msub', 'Osub'
    )

    print
    for i in range(len(array_vars)):
        var = array_vars[i]
        if var in vars():

            expr = 'str({}.dtype)'.format(var)
            print '> {}.dtype = {}'.format(var, eval(expr))

            expr = '{}.shape'.format(var)
            shape = eval(expr)
            if len(shape) == 1:
                shape = (1, shape[0])
            print '    shape = {}'.format(str(shape).replace('L', ''))

            expr = 'np.nanmin({})'.format(var)
            print '    min = {}'.format(eval(expr))

            expr = 'np.nanmax({})'.format(var)
            print '    max = {}'.format(eval(expr))

        elif var == '-':
            print '------------------'

    print


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


def getCalledFunctionArgs(depth=1, funcName=None):
    stack = inspect.stack()
    func_frame_record = None

    try:
        if depth == 0 or depth == float('inf'):
            if funcName is None:
                raise InvalidArgumentError("Must be given a function name to search for when depth is not certain")
            stack_iterable = xrange(len(stack))
            if depth != 0:
                stack_iterable = reversed(stack_iterable)
            for i in stack_iterable:
                fr_funcName = stack[i][3]
                if fr_funcName == funcName:
                    func_frame_record = stack[i+1]
                    break
            if func_frame_record is None:
                raise InvalidArgumentError("Function name '{}' could not be found in the stack".format(funcName))
        else:
            try:
                func_frame_record = stack[depth+1]
            except IndexError:
                raise InvalidArgumentError("Invalid 'depth' index for stack: {}".format(depth))
            if funcName is not None and stack[depth][3] != funcName:
                raise InvalidArgumentError("Function name '{}' could not be found in the stack"
                                           " at index {}".format(funcName, depth))
    except InvalidArgumentError:
        print "STACK AT ERROR:"
        for fr in stack:
            print fr
        raise

    funcCall = ''.join([str(line).strip() for line in func_frame_record[4]])

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
                    raise InvalidArgumentError("Cannot find test file: '{}'".format(testFile))
        testFile = testFile_temp
    return testFile


def openRasterZ(rasterFile):
    rasterFile = findFile(rasterFile)
    Z, _, _ = rat.oneBandImageToArrayZXY(rasterFile)
    return Z


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
        runnum_file = open(runnumFile_new, 'w')
        runnum_file.close()
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
        if flavor in ('dem', 'z'):
            flavor_name = 'dem'
            image_PILmode = 'F'
            raster_format = 'float32'
            raster_nodata = -9999
        elif flavor in ('match', 'm'):
            flavor_name = 'match'
            image_PILmode = 'L'
            raster_format = 'uint8'
            raster_nodata = 0
        elif flavor in ('ortho', 'o', 'or'):
            flavor_name = 'ortho'
            image_PILmode = 'I'
            raster_format = 'int16'
            raster_nodata = 0
        elif flavor in ('mask', 'md', 'd'):
            flavor_name = 'mask'
            image_PILmode = 'L'
            raster_format = 'uint8'
            raster_nodata = 0
        else:
            raise InvalidArgumentError("Invalid image/raster 'flavor': {}".format(flavor))

    return flavor_name, image_PILmode, raster_format, raster_nodata


def handleBatchImageRasterAuto(arrays, flavor, matchkey, descr, compare, concurrent, *X_Y_pref):
    array_names_tuple = splitTupleString(getCalledFunctionArgs(2)[0])

    flavor_order = None
    if flavor is None:
        flavor_order = [None]*len(arrays)
    elif flavor == 'auto':
        flavor_order = array_names_tuple
    elif flavor.startswith('-'):
        if len(flavor) != (1+len(arrays)):
            raise InvalidArgumentError("'flavor' argument starting with '-' must be followed by"
                                       " a number of characters (flavor abbreviations) equal to"
                                       " the number of input 'arrays'")
        flavor_order = flavor[1:len(flavor)]
    else:
        flavor_order = [flavor]*len(arrays)

    key_order = None
    if matchkey is None:
        key_order = [None]*len(arrays)
    elif matchkey == 'auto':
        key_order = array_names_tuple
    elif matchkey == 'flavor':
        key_order = [interpretImageRasterFlavor(f)[0] for f in array_names_tuple]
    else:
        key_order = [matchkey]*len(arrays)

    if not X_Y_pref:
        saveImageAuto(arrays[0], flavor_order[0], key_order[0], descr, compare, concurrent)
        for i in range(1, len(arrays)):
            saveImageAuto(arrays[i], flavor_order[i], key_order[i], descr, True, concurrent)
    else:
        X, Y, proj_ref = X_Y_pref
        saveRasterAuto(arrays[0], X, Y, flavor_order[0], key_order[0], descr, compare, concurrent, proj_ref)
        for i in range(1, len(arrays)):
            saveRasterAuto(arrays[i], X, Y, flavor_order[i], key_order[i], descr, True, concurrent, proj_ref)
    return


def getImageRasterAutoFname(array, array_name, flavor_name, matchkey, descr, compare, concurrent, isRaster):
    runnum = getRunnum()
    imgnum = getNextImgnum(runnum, compare, concurrent)
    if imgnum is None:
        imgnum = 1
    flavor = '{:_<5}'.format(flavor_name)
    filetype = 'ras' if isRaster else 'img'
    key = ''
    if matchkey is not None:
        if matchkey == 'auto':
            key = array_name
        elif matchkey == 'flavor':
            key = interpretImageRasterFlavor(array_name)[0]
        else:
            key = matchkey
        key = key.replace(' ', '-').replace('~', '-')
    description = '~'+descr.replace(' ', '-') if descr != '' else ''

    testFname = 'run{:03d}_{:03d}_py_{}_{}_{}_{}x{}{}.tif'.format(
        runnum, imgnum, filetype, flavor, key, array.shape[0], array.shape[1], description
    )
    return testFname


def saveImage(array, PILmode='F', fname='testImage_py.tif'):
    testFile = getTestFileFromFname(fname)
    if testFile is None:
        return
    image = misc.toimage(array, high=np.max(array), low=np.min(array), mode=PILmode.upper())
    image.save(testFile)
    print "'{}' saved".format(testFile.replace(TESTDIR, '{TESTDIR}/'))


def saveImageAuto(array, flavor='auto', matchkey='auto', descr='', compare=False, concurrent=False):
    if type(array) in (tuple, list):
        handleBatchImageRasterAuto(array, flavor, matchkey, descr, compare, concurrent)
        return
    array_name = getCalledFunctionArgs()[0]

    # Determine the correct data type for saving the raster data.
    if flavor == 'auto':
        flavor = array_name
    flavor_name, PILmode, _, _ = interpretImageRasterFlavor(flavor)

    testFname = getImageRasterAutoFname(array, array_name, flavor_name, matchkey, descr, compare, concurrent, False)
    saveImage(array, PILmode, fname=testFname)


def saveRaster(Z, X=None, Y=None, fname='testRaster_py.tif',
               proj_ref=None, geotrans_rot_tup=(0, 0),
               like_rasterFile=None,
               nodataVal=None, dtype_out=None, force_dtype=False, skip_casting=False):
    testFile = getTestFileFromFname(fname)
    if testFile is None:
        return

    if proj_ref is None:
        print "WARNING: No proj_ref argument given to saveRaster()"
        print "-> Using default global PROJREF_POLAR_STEREO."
        proj_ref = PROJREF_POLAR_STEREO

    rat.saveArrayAsTiff(Z, testFile,
                        X, Y, proj_ref, geotrans_rot_tup,
                        like_rasterFile,
                        nodataVal, dtype_out, force_dtype, skip_casting)
    print "'{}' saved".format(testFile.replace(TESTDIR, '{TESTDIR}/'))


def saveRasterAuto(Z, X, Y, flavor='auto', matchkey='auto', descr='', compare=False, concurrent=False, proj_ref=None):
    if type(Z) in (tuple, list):
        handleBatchImageRasterAuto(Z, flavor, matchkey, descr, compare, concurrent, X, Y, proj_ref)
        return
    array_name = getCalledFunctionArgs()[0]

    # Determine the correct data type for saving the raster data.
    if flavor == 'auto':
        flavor = array_name
    flavor_name, _, fmt, nodata = interpretImageRasterFlavor(flavor)

    testFname = getImageRasterAutoFname(Z, array_name, flavor_name, matchkey, descr, compare, concurrent, True)
    if nodata is not None:
        Z_copy = np.copy(Z)
        Z_copy[np.where(np.isnan(Z_copy))] = nodata
    else:
        Z_copy = Z
    saveRaster(Z_copy, X, Y, fname=testFname, proj_ref=proj_ref, nodataVal=nodata, dtype_out=fmt)


def waitForComparison(expected_imgnum):

    last_imgnum = getNextImgnum(compare=True, concurrent=True)
    if last_imgnum != expected_imgnum:
        print "WARNING: last_imgnum ({}) != expected_imgnum ({}) in test file comparison!!".format(last_imgnum, expected_imgnum)

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


def doMasking(matchFile):
    matchFile = findFile(matchFile)
    generateMasks(matchFile)


def getFP(demFile):
    demFile = findFile(demFile)

    Z, X, Y = rat.oneBandImageToArrayZXY(demFile)
    fp_vertices = rat.getFPvertices(Z, X, Y, nodataVal=-9999)
    num = len(fp_vertices[0])

    test_str = \
"""demFile: {}
Strip Footprint Vertices
X: {}
Y: {}
""".format(
        demFile,
        str(fp_vertices[0]).replace(',', '')[1:-1],
        str(fp_vertices[1]).replace(',', '')[1:-1],
    )

    return num, test_str


def saveDBP(demFile):
    demFile = findFile(demFile)
    shapefileFile = demFile.replace('dem.tif', 'dem_boundary.shp')

    Z, X, Y, proj_ref = rat.oneBandImageToArrayZXY_projRef(demFile)
    poly = rat.getDataBoundariesPoly(Z, X, Y, nodataVal=-9999)
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
