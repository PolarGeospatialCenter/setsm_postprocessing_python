function [d] = test_readRaster(rasterFile)

if ~exist('rasterFile', 'var') || isempty(rasterFile)
    rasterFile = 'testRaster_py.tif';
end


rasterFile = test_findTestFile(rasterFile);
d = readGeotiff(rasterFile);
