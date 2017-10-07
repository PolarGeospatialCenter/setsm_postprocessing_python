function [hist] = test_histArray(arr)
% Displys a histogram with bins for all unique values in an array.

vals = unique(arr);
if length(vals) == 1
    vals = [vals; vals(1)+1];
else
    vals = [vals; (vals(end)+(vals(end)-vals(end-1)))];
end
hist = histogram(arr, vals);
set(gca,'yscale','log');
