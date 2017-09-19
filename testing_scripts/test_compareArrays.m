function [diff, diff_bool] = test_compareArrays(arr1, arr2, title1, title2, figtitle)
% Computes difference maps between two input image arrays and displays them with the original images. To better allow for comparison of nodata (NaN) pixel locations, all NaN values in both input arrays are converted to -Inf before differencing (arr2 - arr1).

if ~exist('title1', 'var') || isempty(title1)
    title1 = 'arr1';
end
if ~exist('title2', 'var') || isempty(title2)
    title2 = 'arr2';
end
% if ~exist('figtitle', 'var') || isempty(figtitle)
%     figtitle = '';
% end


if ~isequal(size(arr1), size(arr2))
    error('Input arrays for comparison differ in shape.');
end

arr1(isnan(arr1)) = -inf;
arr2(isnan(arr2)) = -inf;
diff = arr2 - arr1;
diff(isnan(diff)) = 0;
UL_nans = sum(diff(:) == inf);
UR_nans = sum(diff(:) == -inf);

diff_bool = (diff ~= 0);

figure('Name', figtitle);
subplot(2,2,1);
test_showArray(arr1, title1);
subplot(2,2,2);
test_showArray(arr2, title2);
subplot(2,2,3);
test_showArray(diff, 'Difference (UR-UL)');
subplot(2,2,4);
test_showArray(diff_bool, 'Boolean Difference');

vals_diff_bool = unique(diff_bool);
cnts_diff_bool = histcounts(diff_bool);

if length(vals_diff_bool) > 1
    figure('Name', figtitle);
    test_histArray(diff);
    title('Difference (UR-UL)');
end

fprintf("--- statistics for boolean difference map ---\n");
fprintf("(value, count):");
for i = 1:length(vals_diff_bool)
    fprintf(vals_diff_bool(i) + 1, ...
        " (%d, %e)", vals_diff_bool(i), cnts_diff_bool(i));
end
fprintf("\n");

if UL_nans > 0 || UR_nans > 0
    fprintf(2, "[(UL unique NaNs, %d) (UR unique NaNs, %d)]\n", ...
        UL_nans, UR_nans);
end
