function test_handleBatchImageRasterAuto(arrays, flavor, matchkey, descr, compare, concurrent, X, Y, projstr)

if ~exist('X', 'var') || isempty(X)
    X = [];
end
if ~exist('Y', 'var') || isempty(Y)
    Y = [];
end
if ~exist('pref', 'var') || isempty(projstr)
    projstr = [];
end


array_names = fieldnames(arrays);

flavor_order = [];
flavor = char(flavor);
if isempty(flavor)
    flavor_order = strings(1, length(array_names));
elseif startsWith(flavor, '-')
    if length(flavor) ~= (1+length(array_names))
        error(['"flavor" argument starting with "-" must be followed by' ...
               ' a number of characters (flavor abbreviations) equal to' ...
               ' the number of input "arrays"']);
    end
    flavor_order = flavor(2:length(flavor));
else
    flavor_order = repmat(string(flavor), 1, length(array_names));
end

key_order = [];
if isempty(matchkey)
    key_order = strings(1, length(array_names));
else
    key_order = repmat(string(matchkey), 1, length(array_names));
end

if isempty(X)
    array_struct = test_namedStruct(char(array_names(1)), arrays.(array_names{1}));
    test_saveImageAuto(array_struct, flavor_order(1), key_order(1), descr, compare, concurrent);
    for i = 2:length(array_names)
        array_struct = test_namedStruct(char(array_names(i)), arrays.(array_names{i}));
        test_saveImageAuto(array_struct, flavor_order(i), key_order(i), descr, true, concurrent);
    end
else
    array_struct = test_namedStruct(char(array_names(1)), arrays.(array_names{1}));
    test_saveRasterAuto(array_struct, X, Y, flavor_order(1), key_order(1), descr, compare, concurrent, projstr);
    for i = 2:length(array_names)
        array_struct = test_namedStruct(char(array_names(i)), arrays.(array_names{i}));
        test_saveRasterAuto(array_struct, X, Y, flavor_order(i), key_order(i), descr, true, concurrent, projstr);
    end
end
