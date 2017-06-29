function [array] = test_readArray(imgFile)

test_setGlobals();
global TESTDIR;


imgFile = char(imgFile);
if ~(exist(imgFile, 'file') == 2)
    imgFile = [TESTDIR,'/',imgFile];
    if ~(exist(imgFile, 'file') == 2)
        error('Argument imgFile image file does not exist.');
    end
end

array = imread(imgFile);
