function test_sra(Z, X, Y, flavor, matchkey, descr, compare, concurrent, projstr)
% Save Raster Auto :: Saves an indexed raster image in the test file directory specified by test_setGlobals.m.

if ~exist('flavor', 'var')
    flavor = 'auto';
end
if ~exist('matchkey', 'var')
    matchkey = 'auto';
end
if ~exist('descr', 'var')
    descr = '';
end
if ~exist('compare', 'var')
    compare = false;
end
if ~exist('concurrent', 'var')
    concurrent = false;
end
if ~exist('projstr', 'var')
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

if (strcmp(flavor, 'auto') || (~isempty(matchkey) && strcmp(matchkey, 'auto'))) && isempty(array_name)
    error(['"array_name" must be provided to automatically determine' ...
           ' image flavor or matchkey for a single array']);
end

% Determine the correct data type for saving the raster data.
if strcmp(flavor, 'auto')
    flavor = array_name;
end
[flavor_name, fmt, nodata] = test_interpretImageRasterFlavor(flavor);

if ~isempty(matchkey)
    if strcmp(matchkey, 'auto')
        matchkey = array_name;
    end
else
    matchkey = '';
end

testFname = test_getImageRasterAutoFname(Z, flavor_name, matchkey, descr, compare, concurrent, true);
if ~isempty(nodata)
    Z_copy = Z;
    Z_copy(isnan(Z_copy)) = nodata;
else
    Z_copy = Z;
end
test_saveRaster(Z_copy, X, Y, fmt, nodata, projstr, testFname);
