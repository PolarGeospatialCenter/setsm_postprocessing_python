function [hist] = test_histArray(arr)
% Displys a histogram with bins for all unique values in an array.

vals = unique(arr);
vals = [vals; vals+1];
vals = unique(vals);
hist = histogram(arr, vals);
