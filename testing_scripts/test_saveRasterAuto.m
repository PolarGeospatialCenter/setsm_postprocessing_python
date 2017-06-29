function test_saveRasterAuto(Z, X, Y, flavor, keyword, descr, compare, concurrent, projstr)
% Saves an indexed raster image in a test file directory speicifed by test_setGlobals.m.

if ~exist('keyword', 'var') || isempty(keyword)
    keyword = '';
end
if ~exist('descr', 'var') || isempty(descr)
    descr = '';
end
if ~exist('compare', 'var') || isempty(compare)
    compare = false;
end
if ~exist('concurrent', 'var') || isempty(concurrent)
    concurrent = false;
end
if ~exist('projstr', 'var') || isempty(projstr)
    projstr = 'polar stereo north';
end


% Determine the correct data type for saving the raster data.
fmt = -1;
nodata = -1;
if strcmp(flavor, 'dem')
    fmt = 4;
    nodata = -9999;
elseif strcmp(flavor, 'ortho')
    fmt = 2;
    nodata = 0;
elseif strcmp(flavor, 'match') || strcmp(flavor, 'edge') || strcmp(flavor, 'data')
    fmt = 1;
    nodata = 0;
else
    error('Invalid argument flavor for raster: %s', flavor);
end

if strcmp(keyword, '')
    keyword = flavor;
end
keyword = pad(strrep(keyword(1:min(5, length(keyword))), '~', '-'), 5, '_');

description = '';
if ~strcmp(descr, '')
    description = ['~',strrep(descr, ' ', '-')];
end

% Save the test raster.
runnum = test_getRunnum();
imgnum = test_getNextImgnum(runnum, compare, concurrent);
if isempty(imgnum)
    imgnum = 1;
end
testFname = sprintf('run%03d_%03d_ml_ras_%s_%dx%d%s.tif', ...
    runnum, imgnum, keyword, length(Y), length(X), description);
Z(isnan(Z)) = -9999;
test_saveRaster(Z, X, Y, fmt, nodata, projstr, testFname);
Z(Z == -9999) = NaN;
