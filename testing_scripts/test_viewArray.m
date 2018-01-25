function test_viewArray(array, TITLE, print_dtype)
% Displays a 2D array as a scaled image.

if ~exist('TITLE', 'var') || isempty(TITLE)
    TITLE = '';
end
if ~exist('print_dtype', 'var') || isempty(print_dtype)
    print_dtype = true;
end


if print_dtype
    fprintf("--> '%s' array class: %s\n", TITLE, class(array));
end

if any(isnan(array(:)))
    array_min = min(array(:));
    array_max = max(array(:));
    image(array, 'CDataMapping','scaled');
    caxis([array_min - (array_max-array_min)/30, array_max]);
else
    image(array, 'CDataMapping','scaled');
end

colorbar;
title(strrep(TITLE, '_', '\_'));
