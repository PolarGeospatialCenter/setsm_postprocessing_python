function test_viewArray(array, TITLE, print_dtype, nancount)
% Displays a 2D array as a scaled image.

if ~exist('TITLE', 'var') || isempty(TITLE)
    TITLE = '';
end
if ~exist('print_dtype', 'var') || isempty(print_dtype)
    print_dtype = true;
end
if ~exist('nancount', 'var') || isempty(nancount)
    nancount = [];
end


if print_dtype
    array_dtype = class(array);
    
    if any(strcmp(array_dtype, ["single", "double"]))
        array_nans = isnan(array);
        array_nancount = sum(array_nans(:));
    end
    
    fprintf("--> '%s' array class: %s", TITLE, array_dtype);
    if ~isempty(nancount)
        fprintf(" (%e NaNs)", nancount);
    end
    fprintf("\n");
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
