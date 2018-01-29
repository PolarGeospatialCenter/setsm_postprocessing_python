function test_ibc
% Image Batch Compare :: A command line program for comparing images between two directories that contain an identical number of TIF files (with matching filename pairs between the two) that meet a certain search criteria set in code.


tifDir_ml = 'C:/Users/husby036/Documents/Cprojects/test_s2s/subantarctic_islands/matlab/tif_results/matlab_masks';
tifDir_py = 'C:/Users/husby036/Documents/Cprojects/test_s2s/subantarctic_islands/python_linked/tif_results/python_masks';
tifFnameSearch = '*_mask2a.tif';

tifFiles_ml = dir([tifDir_ml,'/',tifFnameSearch]);
tifFnames = {tifFiles_ml.name};

tifFiles_py = dir([tifDir_py,'/',tifFnameSearch]);
if length(tifFiles_py) ~= length(tifFiles_ml)
    error('%d tifs found in MATLAB dir, %d found in Python dir.', length(tifFiles_ml), length(tifFiles_py));
end
if any(~strcmp(tifFnames, {tifFiles_py.name}))
    error('Tif filenames mismatch between MATLAB and Python dirs.');
end

tifFiles_ml = cellfun(@(x) [tifDir_ml,'/',x], tifFnames, 'uniformoutput', false);
tifFiles_py = cellfun(@(x) [tifDir_py,'/',x], tifFnames, 'uniformoutput', false);

fprintf(2, ['' ...
        '\n----- IBC COMMANDS -----\n' ...
        'next (or empty command) :: compare at current index\n' ...
        'rerun :: redo comparison at previous index\n' ...
        'list :: show indices of images for comparison\n' ...
        'figclose :: close all figures\n' ...
        'quit :: exit without closing figures\n' ...
        'close :: close all figures and exit\n']);

num_tifs = length(tifFnames);

digits = floor(log10(num_tifs)) + 1;
num_format = ['%-',num2str(digits),'d'];

fprintf('\nMATLAB dir: %s\n', tifDir_ml);
fprintf('Python dir: %s\n', tifDir_py);
fprintf('Search string: %s\n', tifFnameSearch);

fprintf('\n%d tifs found:\n\n', num_tifs);
for i=1:num_tifs
    fprintf([num_format,': %s\n'], i, char(tifFnames(i)));
end
fprintf('\n');

display_image = true;
display_histogram = false;
display_casting = false;
display_split = false;
display_difflate = false;
display_fullscreen = true;

index = 0;
while (0 <= index) && (index <= num_tifs)
    if index > 0
        figtitle = char(tifFnames(index));
        fprintf([num_format,': %s\n'], index, figtitle);

        arr_ml = imread(char(tifFiles_ml(index)));
        arr_py = imread(char(tifFiles_py(index)));

        test_compareArrays(arr_ml, arr_py, 'MATLAB', 'Python', figtitle, display_image, display_histogram, display_casting, display_split, display_difflate, display_fullscreen);
    end
    
    last_index = index;
    index = index + 1;
    
    while true
        fprintf(2, 'index = %d; IBC>> ', index);
        if index > num_tifs
            fprintf(2, '(end) ');
        end
        command = input('', 's');
        if isempty(command) || strcmp(command, 'next')
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
            eval(command);
        end
    end
end

fprintf(2, '------------------------\n\n');
end
