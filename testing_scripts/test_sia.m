function test_sia(array, flavor, matchkey, descr, compare, concurrent)
% Save Image Auto :: Saves an indexed image in the test file directory specified by test_setGlobals.m.

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

if (strcmp(flavor, 'auto') || (~isempty(matchkey) && strcmp(matchkey, 'auto'))) && isempty(array_name)
    error(['"array_name" must be provided to automatically determine' ...
           ' image flavor or matchkey for a single array']);
end

% Determine the correct data type for saving the image data.
if ~isempty(flavor)
    if strcmp(flavor, 'auto')
        flavor = array_name;
    end
    [flavor_name,~,~] = test_interpretImageRasterFlavor(flavor);
else
    flavor_name = '';
end

if ~isempty(matchkey)
    if strcmp(matchkey, 'auto')
        matchkey = array_name;
    end
else
    matchkey = '';
end

testFname = test_getImageRasterAutoFname(array, flavor_name, matchkey, descr, compare, concurrent, false);
test_saveImage(array, testFname);
