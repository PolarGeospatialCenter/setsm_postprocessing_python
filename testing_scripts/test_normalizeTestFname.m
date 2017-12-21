function [resultstr] = test_normalizeTestFname(testFname, no_matchkey)

if ~exist('no_matchkey', 'var') || isempty(no_matchkey)
    no_matchkey = false;
end


testFname = char(testFname);

i = [];
if no_matchkey
    i_underscore = strfind(testFname, '_');
    i_matchkey = i_underscore > 24;
    if any(i_matchkey)
        i = i_underscore(find(i_matchkey));
    end
    if length(i) > 1
        i = i(1);
    end
end
if isempty(i)
    i = strfind(testFname, '~');
    if isempty(i)
        i = strfind(testFname, '.tif');
    end
end

if isempty(i) || i < 2
    resultstr = [];
    return;
else
    resultstr = testFname(1:i-1);
end

resultstr = strrep(resultstr, '_ml_img_', '_**_***_');
resultstr = strrep(resultstr, '_ml_ras_', '_**_***_');
resultstr = strrep(resultstr, '_py_img_', '_**_***_');
resultstr = strrep(resultstr, '_py_ras_', '_**_***_');
