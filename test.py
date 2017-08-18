# Version 1.0; Erik Husby; Polar Geospatial Center, University of Minnesota; 2017

from __future__ import division
import inspect
import os
import platform
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
        print "ERROR: Could not find function definition matching '{}'".format(caller_funcName)
        return

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
        print "ERROR: Could not find return statement matching '{}' within '{}'".format(
            this_funcReturn, caller_funcDef)
        return

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

    array_vars = ('x', 'y', 'z', 'm', 'o', 'md',
                  'X', 'Y', 'Z', 'M', 'O')

    print
    for i in range(len(array_vars)):
        var = array_vars[i]
        if var in vars():

            expr = 'str({}.dtype)'.format(var)
            print '> {}.dtype = {}'.format(var, eval(expr))

            expr = 'np.nanmin({})'.format(var)
            print '    min = {}'.format(eval(expr))

            expr = 'np.nanmax({})'.format(var)
            print '    max = {}'.format(eval(expr))

    print


def findFile(testFile):
    if not os.path.isfile(testFile):
        testFile_temp = os.path.join(TESTDIR, testFile)
        if not os.path.isfile(testFile_temp):
            testFile_temp += '.tif'
            if not os.path.isfile(testFile_temp):
                raise InvalidArgumentError("No such test file: '{}'".format(testFile))
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


def saveImage(array, PILmode, fname='testImage_py.tif'):
    if not fname.endswith('.tif'):
        fname += '.tif'

    testFile = os.path.join(TESTDIR, fname)
    while os.path.isfile(testFile):
        opt = raw_input("Test image '{}' already exists. Overwrite? (y/n): ".format(testFile))
        if opt.strip().lower() == 'y':
            break
        else:
            opt = raw_input("Append description to filename (or press [ENTER] to cancel): ")
            if opt == '':
                return
            else:
                testFile = testFile.replace('.tif', '_'+opt.replace(' ', '-')+'.tif')

    image = misc.toimage(array, high=np.max(array), low=np.min(array), mode=PILmode.upper())
    image.save(testFile)
    print "'{}' saved".format(testFile)


def saveImageAuto(array, flavor, keyword='', descr='', compare=False, concurrent=False):
    # Determine the correct data type for saving the image data.
    PILmode = None
    if flavor == 'dem':
        PILmode = 'F'
    elif flavor in ('match', 'edge', 'data'):
        PILmode = 'L'
    elif flavor == 'ortho':
        PILmode = 'I'
    else:
        raise InvalidArgumentError("Invalid raster 'flavor': {}".format(flavor))

    if keyword == '':
        keyword = flavor
    keyword = '{:_<5}'.format(keyword[:5].replace('~', '-'))
    description = '' if descr == '' else '~'+descr.replace(' ', '-')

    # Save the test image.
    runnum = getRunnum()
    imgnum = getNextImgnum(runnum, compare, concurrent)
    if imgnum is None:
        imgnum = 1
    testFname = 'run{:03d}_{:03d}_py_img_{}_{}x{}{}.tif'.format(
        runnum, imgnum, keyword, array.shape[0], array.shape[1], description
    )
    saveImage(array, PILmode, fname=testFname)


def saveRaster(Z, X=None, Y=None, fname='testRaster_py.tif',
                    proj_ref=None, geotrans_rot_tup=(0, 0),
                    like_rasterFile=None,
                    nodataVal=None, dtype_out=None, force_dtype=False, skip_casting=False):
    if not fname.endswith('.tif'):
        fname += '.tif'

    if proj_ref is None:
        print "WARNING: No proj_ref argument given to saveRaster()."
        print "Using default global PROJREF_POLAR_STEREO."
        proj_ref = PROJREF_POLAR_STEREO

    testFile = os.path.join(TESTDIR, fname)
    while os.path.isfile(testFile):
        opt = raw_input("Test raster '{}' already exists. Overwrite? (y/n): ")
        if opt.strip().lower() == 'y':
            break
        else:
            opt = raw_input("Append description to filename (or press [ENTER] to cancel): ")
            if opt == '':
                return
            else:
                testFile = testFile.replace('.tif', '_'+opt.replace(' ', '-')+'.tif')

    rat.saveArrayAsTiff(Z, testFile,
                        X, Y, proj_ref, geotrans_rot_tup,
                        like_rasterFile,
                        nodataVal, dtype_out, force_dtype, skip_casting)

    print "'{}' saved".format(testFile)


def saveRasterAuto(Z, X, Y, flavor='', keyword='', descr='', compare=False, concurrent=False, proj_ref=None):
    if type(Z) in (tuple, list):
        array_order = ('dem', 'match', 'ortho', 'mask')
        saveRasterAuto(Z[0], X, Y, array_order[0], keyword, descr, compare, concurrent, proj_ref)
        for i in range(1, len(Z)):
            saveRasterAuto(Z[i], X, Y, array_order[i], keyword, descr, True, concurrent, proj_ref)
        return

    # Determine the correct data type for saving the raster data.
    fmt = None
    nodata = None
    if flavor != '':
        if flavor == 'dem':
            fmt = 'float32'
            nodata = -9999
        elif flavor in ('match', 'mask'):
            fmt = 'uint8'
            nodata = 0
        elif flavor == 'ortho':
            fmt = 'int16'
            nodata = 0
        else:
            raise InvalidArgumentError("Invalid raster 'flavor': {}".format(flavor))

    if keyword == '':
        keyword = flavor
    keyword = '{:_<5}'.format(keyword[:5].replace('~', '-'))
    description = '' if descr == '' else '~'+descr.replace(' ', '-')

    # Save the test raster.
    runnum = getRunnum()
    imgnum = getNextImgnum(runnum, compare, concurrent)
    if imgnum is None:
        imgnum = 1
    testFname = 'run{:03d}_{:03d}_py_ras_{}_{}x{}{}.tif'.format(
        runnum, imgnum, keyword, len(Y), len(X), description
    )
    Z_copy = np.copy(Z)
    Z_copy[np.where(np.isnan(Z_copy))] = nodata
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
        print "ERROR: Failed to create data cluster boundaries polygon"
        return

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
