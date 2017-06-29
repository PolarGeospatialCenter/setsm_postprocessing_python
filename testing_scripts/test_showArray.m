function test_showArray(array, TITLE)
% Displays a 2D array as a scaled image.

if ~exist('TITLE', 'var') || isempty(TITLE)
    TITLE = '';
end


if isa(array, 'single')
    array(array == -9999) = NaN;
end
image(array, 'CDataMapping','scaled')
title(strrep(TITLE, '_', '\_'));
