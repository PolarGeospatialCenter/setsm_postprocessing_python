function test_viewArray(array, TITLE)
% Displays a 2D array as a scaled image.

if ~exist('TITLE', 'var') || isempty(TITLE)
    TITLE = '';
end


if isa(array, 'single') && any(array(:) == -9999)
    array_nodata = (array == -9999);
    array(array_nodata) = nan;
end

if any(isnan(array(:)))
    array_min = min(array(:));
    array_max = max(array(:));
    if exist('array_nodata', 'var')
        array(array_nodata) = -9999;
    end
    image(array, 'CDataMapping','scaled');
    caxis([array_min - (array_max-array_min)/30, array_max]);
else
    image(array, 'CDataMapping','scaled');
end

colorbar;
title(strrep(TITLE, '_', '\_'));
