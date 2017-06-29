function [resultstr] = test_normalizeTestFname(testFname)

testFname = char(testFname);

i = strfind(testFname, '~');
if isempty(i)
    i = strfind(testFname, '.tif');
end

if isempty(i) || i < 2
    resultstr = [];
    return;
end

resultstr = strrep(strrep(testFname(1:i-1), '_ml_', '_**_'), '_py_', '_**_');
