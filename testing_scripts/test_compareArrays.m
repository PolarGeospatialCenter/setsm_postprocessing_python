function [diff, diff_bool] = test_compareArrays(arr1, arr2, title1, title2, figtitle, display_image, display_histogram, display_casting, display_split, display_difflate)
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


if ~isequal(size(arr1), size(arr2))
    error('Input arrays for comparison differ in shape.');
end

arr1_casted = false;
arr2_casted = false;
arr1_dtype = class(arr1);
arr2_dtype = class(arr2);
class_order = ["logical", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "single", "double"];
array_ranks = cellfun(@(x) find(strcmp(class(x), class_order)), {arr1, arr2});
compare_rank = max(array_ranks);
compare_dtype = class_order(compare_rank);
if strcmp(compare_dtype, 'logical') || contains(compare_dtype, 'uint')
    compare_rank = compare_rank + 1;
    compare_dtype = class_order(compare_rank);
end
if ~strcmp(arr1_dtype, compare_dtype)
    if display_casting
        arr1_precast = arr1;
        arr1_precast_dtype = arr1_dtype;
    end
    arr1_dtype = [arr1_dtype, sprintf(' -> %s', compare_dtype)];
    eval(sprintf('arr1 = %s(arr1);', compare_dtype));
    arr1_casted = true;
end
if ~strcmp(arr1_dtype, compare_dtype)
    if display_casting
        arr2_precast = arr2;
        arr2_precast_dtype = arr2_dtype;
    end
    arr2_dtype = [arr2_dtype, sprintf(' -> %s', compare_dtype)];
    eval(sprintf('arr2 = %s(arr2);', compare_dtype));
    arr2_casted = true;
end
fprintf("--> '%s' array class: %s\n", title1, arr1_dtype);
fprintf("--> '%s' array class: %s\n", title2, arr2_dtype);  

has_nans = false;
if any(cellfun(@(x) any(strcmp(class(x), ["single", "double"])), {arr1, arr2}))
    arr1_nans = isnan(arr1);
    arr2_nans = isnan(arr2);
    if any(arr1_nans(:))
        has_nans = true;
        arr1(arr1_nans) = -inf;
    end
    if any(arr2_nans(:))
        has_nans = true;
        arr2(arr2_nans) = -inf;
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
    diff(isnan(diff)) = 0;
    UL_nans = sum(diff(:) == inf);
    UR_nans = sum(diff(:) == -inf);
end

diff_bool = (diff ~= 0);
vals_diff_bool = unique(diff_bool);
cnts_diff_bool = histcounts(diff_bool);

diff_bool_disp = [];
diff_bool_title = [];
if display_difflate
    diffl_factor = 0.006;
    if display_split
        diffl_factor = diffl_factor / 2;
    end
    diffl_se_sz = max(ceil(size(diff_bool) * diffl_factor));
    diff_bool_disp = imdilate(diff_bool, ones(diffl_se_sz));
    diff_bool_title = sprintf('Boolean Difference (dilated by ones(%d))', diffl_se_sz);
else
    diff_bool_disp = diff_bool;
    diff_bool_title = 'Boolean Difference';
end

diff_title = sprintf('Difference (%s - %s)', title2, title1);

if display_casting && (arr1_casted || arr2_casted)
    figure('Name', sprintf('CAST: %s', figtitle));
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
    figure('Name', sprintf('COMP: %s', figtitle));
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
    figure('Name', sprintf('DIFF: %s', figtitle));
    test_viewArray(diff, diff_title, true);
    figure('Name', sprintf('BOOL: %s', figtitle));
    test_viewArray(diff_bool_disp, diff_bool_title, false);
end

if display_histogram && length(vals_diff_bool) > 1
    figure('Name', sprintf('HIST: %s', figtitle));
    test_histArray(diff, diff_title);
end

fprintf("--- statistics for boolean difference map ---\n");
fprintf("(value, count):");
for i = 1:length(vals_diff_bool)
    fprintf(vals_diff_bool(i) + 1, ...
        " (%d, %e)", vals_diff_bool(i), cnts_diff_bool(i));
end
fprintf("\n---------------------------------------------\n");

if UL_nans > 0 || UR_nans > 0
    fprintf(2, "[(UL unique NaNs, %d) (UR unique NaNs, %d)]\n", ...
        UL_nans, UR_nans);
end
