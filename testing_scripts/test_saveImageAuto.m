function test_saveImageAuto(array, flavor, matchkey, descr, compare, concurrent)
% Saves an indexed image in a test file directory speicifed by test_setGlobals.m.

if ~exist('flavor', 'var') || isempty(flavor)
    flavor = 'auto';
end
if ~exist('matchkey', 'var') || isempty(matchkey)
    matchkey = 'auto';
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


array_name = [];
if isstruct(array)
    if length(fieldnames(array)) > 1
        test_handleBatchImageRasterAuto(array, flavor, matchkey, descr, compare, concurrent);
        return;
    else
        array_name = fieldnames(array);
        array = array.(array_name{1});
        if iscell(array)
            array = array{1,1};
        end
        array_name = array_name(1);
        array_name = array_name{1,1};
    end
else
    array_name = inputname(1);
end

% Determine the correct data type for saving the raster data.
flavor_name = '';
if strcmp(flavor, 'auto')
    flavor = array_name;
end
try
    [flavor_name,~,~] = test_interpretImageRasterFlavor(flavor);
catch ME
    if startsWith(ME.message, 'Invalid image/raster "flavor":')
        ;
    else
        rethrow(ME);
    end
end

testFname = test_getImageRasterAutoFname(array, array_name, flavor_name, matchkey, descr, compare, concurrent, false);
test_saveImage(array, testFname);
