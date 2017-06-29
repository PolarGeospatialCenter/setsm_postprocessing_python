function [diff, diff_bool] = test_compareArrays(arr1, arr2, title1, title2, figtitle)
% Computes difference maps between two input image arrays and displays them with the original images.

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

diff = single(arr2) - single(arr1);

diff_bool = diff;
diff_bool(isnan(arr1) & isnan(arr2)) = 0;
diff_bool = (diff_bool ~= 0);

figure('Name', figtitle);
subplot(2,2,1);
test_showArray(arr1, title1);
subplot(2,2,2);
test_showArray(arr2, title2);
subplot(2,2,3);
test_showArray(diff, 'Difference');
subplot(2,2,4);
test_showArray(diff_bool, 'Boolean Difference');

figure('Name', figtitle);
test_histArray(diff);
title('Difference');

vals_diff_bool = unique(diff_bool);
cnts_diff_bool = histcounts(diff_bool);

fprintf("--- Statistics for boolean difference map ---\n");
fprintf("(Value, Count):");
for i = 1:length(vals_diff_bool)
    fprintf(" (%d, %d)", vals_diff_bool(i), cnts_diff_bool(i));
end
fprintf("\n");
