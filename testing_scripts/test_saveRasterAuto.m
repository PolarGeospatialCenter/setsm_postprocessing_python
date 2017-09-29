function test_saveRasterAuto(Z, X, Y, flavor, matchkey, descr, compare, concurrent, projstr)
% Saves an indexed raster image in a test file directory speicifed by test_setGlobals.m.

if ~exist('flavor', 'var') || isempty(flavor)
    flavor = 'auto';
end
if ~exist('matchkey', 'var')
    matchkey = [];
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
    projstr = [];
end


array_name = [];
if isstruct(Z)
    if length(fieldnames(Z)) > 1
        test_handleBatchImageRasterAuto(Z, flavor, matchkey, descr, compare, concurrent, X, Y, projstr);
        return;
    else
        array_name = fieldnames(Z);
        Z = Z.(array_name{1});
        if iscell(Z)
            Z = Z{1,1};
        end
        array_name = array_name(1);
        array_name = array_name{1,1};
    end
else
    array_name = inputname(1);
end

% Determine the correct data type for saving the raster data.
if strcmp(flavor, 'auto')
    flavor = array_name;
end
[flavor_name, fmt, nodata] = test_interpretImageRasterFlavor(flavor);

testFname = test_getImageRasterAutoFname(Z, array_name, flavor_name, matchkey, descr, compare, concurrent, true);
if ~isempty(nodata)
    Z_copy = Z;
    Z_copy(isnan(Z_copy)) = nodata;
else
    Z_copy = Z;
end
test_saveRaster(Z_copy, X, Y, fmt, nodata, projstr, testFname);
