function [array] = test_readImage(imgFile)

try
    imgFile = test_findFile(imgFile);
catch ME
    if strcmp(ME.message, 'cannot find test file')
        error('Argument imgFile test file does not exist.');
    else
        rethrow(ME);
    end
end

array = imread(imgFile);
