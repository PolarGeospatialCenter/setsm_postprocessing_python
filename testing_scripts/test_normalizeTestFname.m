function [resultstr] = test_normalizeTestFname(testFname)

testFname = char(testFname);

i = strfind(testFname, '~');
if isempty(i)
    i = strfind(testFname, '.tif');
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
