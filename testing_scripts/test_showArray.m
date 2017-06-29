function test_showArray(arr, TITLE)
% Displays a 2D array as a scaled image.

if ~exist('TITLE', 'var') || isempty(TITLE)
    TITLE = '';
end


image(arr, 'CDataMapping','scaled')
title(strrep(TITLE, '_', '\_'));
