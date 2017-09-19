function [array] = test_readImage(imgFile)

if ~exist('imgFile', 'var') || isempty(imgFile)
    imgFile = 'testImage_py.tif';
end


imgFile = test_findFile(imgFile);
array = imread(imgFile);
