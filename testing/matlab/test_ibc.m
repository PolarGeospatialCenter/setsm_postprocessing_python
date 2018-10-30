function test_ibc
% Image Batch Compare :: A command line program for comparing images between two directories that contain an identical number of TIF files (with matching filename pairs between the two) that meet a certain search criteria set in code.


%%%%%% MAKE CHANGES HERE %%%%%%

tifDir_1_name = 'Original PIL';
tifDir_2_name = 'New PIL';


tifFnameSearch = '*_bitmask.tif';

tifDir_1 = 'V:\pgc\data\elev\dem\setsm\klassen\region-2018jun13\tif_results\2m\bitmask_lsf_pil';
tifDir_2 = 'V:\pgc\data\elev\dem\setsm\klassen\region-2018jun13\tif_results\2m';

% tifDir_1 = 'D:\test_s2s\reg34_masks\matlab\';
% tifDir_2 = 'D:\test_s2s\reg34_masks\python\';

% tifDir_1 = 'V:\pgc\data\scratch\erik\test_s2s\PBS\region_31_alaska_south\tif_results\old_mask_sel\';
% tifDir_2 = 'V:\pgc\data\scratch\erik\test_s2s\PBS\region_31_alaska_south\tif_results\new_mask_sel\';

% tifDir_1 = 'V:\pgc\data\scratch\erik\test_s2s\matlab\setsm\REMA\region\region_01_subantarctic_islands\strips\8m';
% tifDir_2 = 'V:\pgc\data\scratch\erik\test_s2s\python\setsm\REMA\region\region_01_subantarctic_islands\strips\8m';

% tifDir_1 = 'V:\pgc\data\scratch\erik\test_s2s\matlab\setsm\ArcticDEM\region\region_02_greenland_southeast\tif_results\2m';
% tifDir_2 = 'V:\pgc\data\scratch\erik\test_s2s\python\setsm\ArcticDEM\region\region_02_greenland_southeast\tif_results\2m';

% tifDir_1 = 'C:\Users\husby036\Documents\Cprojects\test_s2s\region_02_greenland_southeast\matlab\tif_results\2m';
% tifDir_2 = 'C:\Users\husby036\Documents\Cprojects\test_s2s\region_02_greenland_southeast\python\tif_results\2m';

% tifDir_1 = 'C:/Users/husby036/Documents/Cprojects/test_s2s/russia_central_east/matlab/tif_results/2m';
% tifDir_2 = 'C:/Users/husby036/Documents/Cprojects/test_s2s/russia_central_east/python/tif_results/python_masks';

% tifDir_1 = 'C:/Users/husby036/Documents/Cprojects/test_s2s/russia_central_east/matlab/strips/2m';
% tifDir_2 = 'C:/Users/husby036/Documents/Cprojects/test_s2s/russia_central_east/python/strips/strips_pymask';

% tifDir_1 = 'C:/Users/husby036/Documents/Cprojects/test_s2s/subantarctic_islands/matlab/strips/strips2a';
% tifDir_2 = 'C:/Users/husby036/Documents/Cprojects/test_s2s/subantarctic_islands/python/strips/strips2a_matmask';

% tifDir_1 = 'C:/Users/husby036/Documents/Cprojects/test_s2s/subantarctic_islands/matlab/strips/strips2a';
% tifDir_2 = 'C:/Users/husby036/Documents/Cprojects/test_s2s/subantarctic_islands/python/strips/strips2a_pymask';


nodata_val = [];
% nodata_val = -9999;
mask_nans = true;

display_image = true;
display_histogram = true;

display_casting = false;
display_split = false;
display_difflate = false;
display_small = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


compare = true;

if ~exist('tifDir_1', 'var') || isempty(tifDir_1)
    compare = false;
    if ~exist('tifDir_2', 'var') || isempty(tifDir_2)
        error('No directories specified.');
    end
    tifDir_1 = tifDir_2;
    tifDir_1_name = tifDir_2_name;
end
if ~exist('tifDir_2', 'var') || isempty(tifDir_2)
    compare = false;
end

fprintf(2, ['' ...
        '\n----- IBC COMMANDS -----\n' ...
        'next (or empty command) :: Compare at current index.\n' ...
        'rerun :: Redo comparison at previous index.\n' ...
        'list :: Show indices of images for comparison.\n' ...
        'figclose :: Close all figures.\n' ...
        'quit :: Exit without closing figures.\n' ...
        'close :: Close all figures and exit.\n']);

tifFiles_1 = dir([tifDir_1,'/',tifFnameSearch]);
tifFnames_1 = {tifFiles_1.name};

if compare
    tifFiles_2 = dir([tifDir_2,'/',tifFnameSearch]);
    tifFnames_2 = {tifFiles_2.name};
    
    tifFnames = intersect(tifFnames_1, tifFnames_2);
    
    tifFiles_1 = cellfun(@(x) [tifDir_1,'/',x], tifFnames, 'uniformoutput', false);
    tifFiles_2 = cellfun(@(x) [tifDir_2,'/',x], tifFnames, 'uniformoutput', false);
else
    tifFnames = tifFnames_1;
end
    
num_tifs = length(tifFnames);

digits = floor(log10(num_tifs)) + 1;
num_format = ['%-',num2str(digits),'d'];

if compare
    fprintf('\n%s dir: %s\n', tifDir_1_name, tifDir_1);
    fprintf(  '%s dir: %s\n', tifDir_2_name, tifDir_2);
else
    fprintf('\n%s dir: %s\n', tifDir_1_name, tifDir_1);
end
fprintf('Search string: %s\n', tifFnameSearch);

fprintf('\n%d tifs found:\n\n', num_tifs);
for i=1:num_tifs
    fprintf([num_format,': %s\n'], i, char(tifFnames(i)));
end
fprintf('\n');

index = 0;
while (0 <= index) && (index <= num_tifs)
    if index > 0
        figtitle = char(tifFnames(index));
        fprintf([num_format,': %s\n'], index, figtitle);
        
        img_1 = char(tifFiles_1(index));
        if compare
            img_2 = char(tifFiles_2(index));
        end

        try
            if compare
                test_compareArrays(imread(img_1), imread(img_2), tifDir_1_name, tifDir_2_name, figtitle, nodata_val, mask_nans, display_image, display_histogram, display_casting, display_split, display_difflate, display_small);
            else
                test_compareImages(img_1, '', figtitle, 0, nodata_val, mask_nans, display_image, display_histogram, display_casting, display_split, display_difflate, display_small);
            end
        catch ME
            if strcmp(ME.message, 'Input arrays for comparison differ in shape.')
                fprintf('\n');
                fprintf(2, ME.message);
                fprintf('\n\n');
            else
                rethrow(ME);
            end
        end
    end
    
    last_index = index;
    index = index + 1;
    
    while true
        fprintf(2, 'index = %d; IBC>> ', index);
        if index > num_tifs
            fprintf(2, '(end) ');
        end
        command = input('', 's');
        if isempty(command)
            close all;
            break;
        elseif strcmp(command, 'next')
            break;
        elseif strcmp(command, 'rerun')
            index = last_index;
            break;
        elseif strcmp(command, 'list')
            fprintf('\n');
            for i=1:num_tifs
                fprintf([num_format,': %s\n'], i, char(tifFnames(i)));
            end
            fprintf('\n');
        elseif strcmp(command, 'figclose')
             close all;
        elseif strcmp(command, 'quit') || strcmp(command, 'close')
            if strcmp(command, 'close')
                close all;
            end
            index = inf;
            break;
        elseif ~isempty(command)
            try
                eval(command);
            catch ME
                fprintf(2, "\n%s\n\n", getReport(ME));
            end
        end
    end
end

fprintf(2, '------------------------\n\n');
end
