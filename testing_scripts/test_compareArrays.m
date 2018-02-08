function [diff, diff_bool] = test_compareArrays(arr1, arr2, title1, title2, figtitle, nodata_val, mask_nans, display_image, display_histogram, display_casting, display_split, display_difflate, display_small)
% Computes difference maps between two input image arrays and displays them with the original images. To better allow for comparison of nodata (NaN) pixel locations, all NaN values in both input arrays are converted to -Inf before differencing (arr2 - arr1).

if ~exist('title1', 'var') || isempty(title1)
    title1 = 'arr1';
end
if ~exist('title2', 'var') || isempty(title2)
    title2 = 'arr2';
end
if ~exist('figtitle', 'var') || isempty(figtitle)
    figtitle = '';
end
if ~exist('nodata_val', 'var') || isempty(nodata_val)
    nodata_val = [];
end
if ~exist('mask_nans', 'var') || isempty(mask_nans)
    mask_nans = false;
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


if ~isequal(size(arr1), size(arr2))
    error('Input arrays for comparison differ in shape.');
end

if ~isempty(nodata_val)
    arr1_nodata = (arr1 == nodata_val);
    if any(arr1_nodata(:))
        if ~any(strcmp(class(arr1), ["single", "double"]))
            arr1 = single(arr1);
        end
        arr1(arr1_nodata) = NaN;
    end
    arr2_nodata = (arr2 == nodata_val);
    if any(arr2_nodata(:))
        if ~any(strcmp(class(arr2), ["single", "double"]))
            arr2 = single(arr2);
        end
        arr2(arr2_nodata) = NaN;
    end
end

arr1_dtype_in = class(arr1);
arr2_dtype_in = class(arr2);
arr1_dtype_out = [];
arr2_dtype_out = [];
has_nans = false;
arr1_casted = false;
arr2_casted = false;

class_order = ["logical", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "single", "double"];
array_ranks = cellfun(@(x) find(strcmp(class(x), class_order)), {arr1, arr2});
compare_rank = max(array_ranks);
compare_dtype = class_order(compare_rank);
if strcmp(compare_dtype, 'logical') || contains(compare_dtype, 'uint')
    compare_rank = compare_rank + 1;
    compare_dtype = class_order(compare_rank);
end

if strcmp(arr1_dtype_in, compare_dtype)
    arr1_dtype_out = arr1_dtype_in;
else
    if display_casting
        arr1_precast = arr1;
        arr1_precast_dtype = arr1_dtype_in;
    end
    arr1_dtype_out = [arr1_dtype_in, sprintf(' -> %s', compare_dtype)];
    eval(sprintf('arr1 = %s(arr1);', compare_dtype));
    arr1_casted = true;
end

if strcmp(arr2_dtype_in, compare_dtype)
    arr2_dtype_out = arr2_dtype_in;
else
    if display_casting
        arr2_precast = arr2;
        arr2_precast_dtype = arr2_dtype_in;
    end
    arr2_dtype_out = [arr2_dtype_in, sprintf(' -> %s', compare_dtype)];
    eval(sprintf('arr2 = %s(arr2);', compare_dtype));
    arr2_casted = true;
end

if array_ranks(1) == array_ranks(2)
    flag_arr1_cast = false;
    flag_arr2_cast = false;
else
    flag_arr1_cast = arr1_casted;
    flag_arr2_cast = arr2_casted;
end

if any(strcmp(arr1_dtype_in, ["single", "double"]))
    arr1_nans = isnan(arr1);
    arr1_nancount = sum(arr1_nans(:));
    if arr1_nancount > 0
        has_nans = true;
        if ~mask_nans
            arr1(arr1_nans) = -inf;
        end
    end
end
if any(strcmp(arr2_dtype_in, ["single", "double"]))
    arr2_nans = isnan(arr2);
    arr2_nancount = sum(arr2_nans(:));
    if arr2_nancount > 0
        has_nans = true;
        if ~mask_nans
            arr2(arr2_nans) = -inf;
        end
    end
end
    
fprintf("--> (UL) '%s' array class: ", title1);
fprintf(flag_arr1_cast + 1, "%s",  arr1_dtype_out);
if exist('arr1_nancount', 'var')
    fprintf(" (%e NaNs)", arr1_nancount);
end
fprintf("\n");

fprintf("--> (UR) '%s' array class: ", title2);
fprintf(flag_arr2_cast + 1, "%s",  arr2_dtype_out);
if exist('arr2_nancount', 'var')
    fprintf(" (%e NaNs)", arr2_nancount);
end
fprintf("\n");

has_nans = false;
if any(cellfun(@(x) any(strcmp(class(x), ["single", "double"])), {arr1, arr2}))
    arr1_nans = isnan(arr1);
    arr1_nancount = sum(arr1_nans(:));
    arr2_nans = isnan(arr2);
    arr2_nancount = sum(arr2_nans(:));
    if arr1_nancount > 0
        has_nans = true;
        if ~mask_nans
            arr1(arr1_nans) = -inf;
        end
    end
    if arr2_nancount > 0
        has_nans = true;
        if ~mask_nans
            arr2(arr2_nans) = -inf;
        end
    end
end

try
    diff = arr2 - arr1;
catch ME
    fprintf(2, "*** Caught the following error during array differencing step ***\n");
    fprintf(2, "%s\n", getReport(ME));
    fprintf(2, "--> Casting both arrays to single and re-attempting differencing...\n");
    diff = single(arr2) - single(arr1);
end

UL_nans = 0;
UR_nans = 0;
if has_nans
    if ~mask_nans
        UL_nans = sum(diff(:) == inf);
        UR_nans = sum(diff(:) == -inf);
    else
        both_nans = arr1_nans & arr2_nans;
        UL_nans = xor(arr1_nans, both_nans);
        UR_nans = xor(arr2_nans, both_nans);
        UL_nans = sum(UL_nans(:));
        UR_nans = sum(UR_nans(:));
    end
end
diff(isnan(diff)) = 0;

diff_bool = (diff ~= 0);
vals_diff_bool = unique(diff_bool);
cnts_diff_bool = histcounts(diff_bool);

diff_bool_disp = [];
diff_bool_title = [];
if display_difflate && (display_image || display_split)
%     diffl_factor = 0.006;
    diffl_factor = 0.003;
    if display_split
        diffl_factor = diffl_factor / 2;
    end
%     diffl_se_sz = max(ceil(size(diff_bool) * diffl_factor));
%     diff_bool_disp = imdilate(diff_bool, ones(diffl_se_sz));
    diffl_se_sz = prod(size(diff_bool)) * diffl_factor;
    diff_bool_disp = imdilate(diff_bool, ones(ceil(sqrt(diffl_se_sz))));
    diff_bool_title = sprintf('Boolean Difference (dilated by ones(%d))', diffl_se_sz);
else
    diff_bool_disp = diff_bool;
    diff_bool_title = 'Boolean Difference';
end

figure_args_extra = {};
if ~display_small
    figure_args_extra = [figure_args_extra, {'units','normalized','outerposition',[0 0 1 1]}];
end

diff_title = sprintf('Difference (%s - %s)', title2, title1);

if display_casting && (arr1_casted || arr2_casted)
    figure_args = [{'Name', sprintf('CAST: %s', figtitle)}, figure_args_extra];
    figure(figure_args{:});
    subplot(2,2,1);
    test_viewArray(arr1, sprintf('%s (%s)', title1, compare_dtype), false);
    subplot(2,2,2);
    test_viewArray(arr2, sprintf('%s (%s)', title2, compare_dtype), false);
    subplot(2,2,3);
    if arr1_casted
        test_viewArray(arr1_precast, sprintf('%s (%s)', title1, arr1_precast_dtype), false);
    end
    subplot(2,2,4);
    if arr2_casted
        test_viewArray(arr2_precast, sprintf('%s (%s)', title2, arr2_precast_dtype), false);
    end
end

if display_image
    figure_args = [{'Name', sprintf('COMP: %s', figtitle)}, figure_args_extra];
    figure(figure_args{:});
    subplot(2,2,1);
    test_viewArray(arr1, title1, false);
    subplot(2,2,2);
    test_viewArray(arr2, title2, false);
    subplot(2,2,3);
    test_viewArray(diff, diff_title, true);
    subplot(2,2,4);
    test_viewArray(diff_bool_disp, diff_bool_title, false);
end

if display_split
    figure_args = [{'Name', sprintf('DIFF: %s', figtitle)}, figure_args_extra];
    figure(figure_args{:});
    test_viewArray(diff, diff_title, ~display_image);
    figure_args = [{'Name', sprintf('BOOL: %s', figtitle)}, figure_args_extra];
    figure(figure_args{:});
    test_viewArray(diff_bool_disp, diff_bool_title, false);
end

if display_histogram && any(vals_diff_bool == 1)
    figure('Name', sprintf('HIST: %s', figtitle));
    test_histArray(diff, diff_title);
end

fprintf("--- statistics for boolean difference map ---\n");
fprintf("(value, count):");
for i = 1:length(vals_diff_bool)
    fprintf(vals_diff_bool(i) + 1, ...
        " (%d, %e)", vals_diff_bool(i), cnts_diff_bool(i));
end
fprintf("\n");
if UL_nans > 0 || UR_nans > 0
    fprintf(2, "[(UL unique NaNs, %e) (UR unique NaNs, %e)]\n", ...
        UL_nans, UR_nans);
end
fprintf("---------------------------------------------\n");
