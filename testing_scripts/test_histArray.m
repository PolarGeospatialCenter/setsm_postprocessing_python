function [hist] = test_histArray(array, TITLE)
% Displys a histogram with bins for all unique values in an array.

if ~exist('TITLE', 'var') || isempty(TITLE)
    TITLE = '';
end


vals = unique(array);
if length(vals) == 1
    vals = [vals; vals(1)+1];
else
    vals = [vals; (vals(end)+(vals(end)-vals(end-1)))];
end
hist = histogram(array, vals);
set(gca,'yscale','log');

title(strrep(TITLE, '_', '\_'));
