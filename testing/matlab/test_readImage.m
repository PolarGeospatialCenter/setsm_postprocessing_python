function [array] = test_readImage(fname_or_file)

if ~exist('fname_or_file', 'var') || isempty(fname_or_file)
    fname_or_file = 'testImage_py.tif';
end


testFile = test_findTestFile(fname_or_file);
array = imread(testFile);
