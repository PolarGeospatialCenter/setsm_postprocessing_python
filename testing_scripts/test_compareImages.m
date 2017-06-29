function [arr1, arr2, diff, diff_bool] = test_compareImages(imgFile1, imgFile2, figtitle, isRaster)
% test_compareImages Reads two input image files into arrays and compares them. If only one image file is given, the image is displayed.

if ~exist('figtitle', 'var') || isempty(figtitle)
    figtitle = '';
end
if ~exist('isRaster', 'var') || isempty(isRaster)
    isRaster = false;
end

test_setGlobals();
global TESTDIR;


single = true;

diff = [];
diff_bool = [];

imgFile1 = char(imgFile1);
if ~(exist(imgFile1, 'file') == 2)
    imgFile1 = [TESTDIR,'/',imgFile1];
    if ~(exist(imgFile1, 'file') == 2)
        error('Argument imgFile1 image file does not exist.');
    end
end

if isRaster
    ds = readGeotiff(imgFile1);
    arr1 = ds.z;
else
    arr1 = imread(imgFile1);
end

if exist('imgFile2', 'var') && ~isempty(imgFile2)
    imgFile2 = char(imgFile2);
    if ~isempty(imgFile2)
        if ~(exist(imgFile2, 'file') == 2)
            imgFile2 = [TESTDIR,'/',imgFile2];
            if ~(exist(imgFile2, 'file') == 2)
                error('Argument imgFile2 image file does not exist.');
            end
        end
        
        single = false;
        
        if isRaster
            ds = readGeotiff(imgFile2);
            arr2 = ds.z;
        else
            arr2 = imread(imgFile2);
        end
        
        [diff, diff_bool] = test_compareArrays(arr1, arr2, imgFile1, imgFile2, figtitle);
    end    
end

if single
    test_showArray(arr1, imgFile1);
end
