function [arr1, arr2, diff, diff_bool] = test_compareImages(imgFile1, imgFile2, figtitle, isRaster, display_image, display_histogram, display_casting, display_split, display_difflate, display_small)
% test_compareImages Reads two input image files into arrays and compares them. If only one image file is given, the image is displayed.

if ~exist('figtitle', 'var') || isempty(figtitle)
    figtitle = '';
end
if ~exist('isRaster', 'var') || isempty(isRaster)
    isRaster = false;
end
if ~exist('display_image', 'var') || isempty(display_image)
    display_image = true;
end
if ~exist('display_histogram', 'var') || isempty(display_histogram)
    display_histogram = true;
end
if ~exist('display_casting', 'var') || isempty(display_casting)
    display_casting = false;
end
if ~exist('display_split', 'var') || isempty(display_split)
    display_split = false;
end
if ~exist('display_difflate', 'var') || isempty(display_difflate)
    display_difflate = false;
end
if ~exist('display_small', 'var') || isempty(display_small)
    display_small = false;
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

[~, imgFname1, ~] = fileparts(imgFile1);

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
        
        [~, imgFname2, ~] = fileparts(imgFile2);
        
        [diff, diff_bool] = test_compareArrays(arr1, arr2, imgFname1, imgFname2, figtitle, display_image, display_histogram, display_casting, display_split, display_difflate, display_small);
    end    
end

if single
    figure_args_extra = {};
    if ~display_small
        figure_args_extra = [figure_args_extra, {'units','normalized','outerposition',[0 0 1 1]}];
    end

    if display_histogram
        if any(strcmp(class(arr1), ["single", "double"]))
            arr1(isnan(arr1)) = -inf;
        end
        figure('Name', sprintf('HIST: %s', figtitle));
        test_histArray(arr1, imgFname1);
    end
    if display_image
        figure_args = [{'Name', sprintf('VIEW: %s', figtitle)}, figure_args_extra];
        figure(figure_args{:});
        test_viewArray(arr1, imgFname1, true);
    end
end
