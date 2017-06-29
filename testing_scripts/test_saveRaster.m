function test_saveRaster(Z, X, Y, fmt, nodata, projstr, fname)

if ~exist('projstr', 'var') || isempty(projstr)
    projstr = 'polar stereo north';
end
if ~exist('fname', 'var') || isempty(fname)
    fname = 'testRaster_ml.tif';
end

test_setGlobals();
global TESTDIR;


if ~contains(fname, '.tif')
    fname = strcat(fname, '.tif');
end

testFile = [TESTDIR,'/',fname];
while exist(testFile, 'file') == 2
    opt = input(sprintf('Test raster "%s" already exists. Overwrite? (y/n): ', testFile), 's');
    if strcmpi(opt, 'y')
        break;
    else
        opt = input('Append description to filename (or press [ENTER] to cancel): ', 's');
        if isempty(opt)
            return;
        else
            testFile = strrep(testFile, '.tif', ['_',strrep(opt,' ','-'),'.tif']);
        end
    end
end

writeGeotiff(testFile, X, Y, Z, fmt, nodata, projstr);
fprintf("'%s' saved\n", testFile);
