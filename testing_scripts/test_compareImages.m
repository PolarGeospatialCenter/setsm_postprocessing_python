function [arr1, arr2, diff, diff_bool] = test_compareImages(imgFile1, imgFile2, figtitle, isRaster)
% test_compareImages Reads two input image files into arrays and compares them. If only one image file is given, the image is displayed.

if ~exist('figtitle', 'var') || isempty(figtitle)
    figtitle = '';
end
if ~exist('isRaster', 'var') || isempty(isRaster)
    isRaster = false;
end


single = true;

diff = [];
diff_bool = [];

try
    imgFile1 = test_findFile(imgFile1);
catch ME
    if strcmp(ME.message, 'cannot find test file')
        error('Argument imgFile1 test file does not exist.');
    else
        rethrow(ME);
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
        try
            imgFile2 = test_findFile(imgFile2);
        catch ME
            if strcmp(ME.message, 'cannot find test file')
                error('Argument imgFile2 test file does not exist.');
            else
                rethrow(ME);
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
    figure('Name', figtitle);
    test_showArray(arr1, imgFile1);
end
