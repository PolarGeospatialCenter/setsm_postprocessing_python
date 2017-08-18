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


if isstruct(Z)
    array_order = ["dem", "match", "ortho", "mask"];
    array_names = fieldnames(Z);
    test_saveRasterAuto(eval(['Z.',char(array_names(1))]), X, Y, array_order(1), ...
        keyword, descr, compare, concurrent, projstr);
    for i = 2:length(array_names)
        test_saveRasterAuto(eval(['Z.',char(array_names(i))]), X, Y, array_order(i), ...
            keyword, descr, true, concurrent, projstr);
    end
    return;
end

if iscell(Z)
    Z = Z{1,1};
end

% Determine the correct data type for saving the raster data.
fmt = -1;
nodata = -1;
if strcmp(flavor, 'dem')
    fmt = 4;
    nodata = -9999;
elseif strcmp(flavor, 'match') || strcmp(flavor, 'mask')
    fmt = 1;
    nodata = 0;
elseif strcmp(flavor, 'ortho')
    fmt = 2;
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
Z_copy = Z;
Z_copy(isnan(Z_copy)) = nodata;
test_saveRaster(Z_copy, X, Y, fmt, nodata, projstr, testFname);
