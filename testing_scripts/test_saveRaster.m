function test_saveRaster(Z, X, Y, fmt, nodata, projstr, fname)

if ~exist('projstr', 'var') || isempty(projstr)
    fprintf("WARNING: No proj_ref argument given to test_saveRaster()\n");
    fprintf("-> Using default 'polar stereo north'.\n");
    projstr = 'polar stereo north';
end
if ~exist('fname', 'var') || isempty(fname)
    fname = 'testRaster_ml.tif';
end

test_setGlobals();
global TESTDIR;


testFile = test_getTestFileFromFname(fname);
if isempty(testFile)
    return;
end

writeGeotiff(testFile, X, Y, Z, fmt, nodata, projstr);
fprintf("'%s' saved\n", strrep(testFile, TESTDIR, '{TESTDIR}'));
