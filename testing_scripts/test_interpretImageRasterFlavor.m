function [flavor_name, raster_format, raster_nodata] = test_interpretImageRasterFlavor(flavor)

flavor_name = '';
raster_format = [];
raster_nodata = [];

if ~isempty(char(flavor))
    if any(strcmp(flavor, ["dem", "z"]))
        flavor_name = 'dem';
        raster_format = 4;
        raster_nodata = -9999;
    elseif any(strcmp(flavor, ["match", "m"]))
        flavor_name = 'match';
        raster_format = 1;
        raster_nodata = 0;
    elseif any(strcmp(flavor, ["ortho", "o", "or"]))
        flavor_name = 'ortho';
        raster_format = 2;
        raster_nodata = 0;
    elseif strcmp(flavor, 'mask')
        flavor_name = 'mask';
        raster_format = 1;
        raster_nodata = 0;
    elseif any(strcmp(flavor, ["data", "md", "d"]))
        flavor_name = 'data';
        raster_format = 1;
        raster_nodata = 0;
    elseif any(strcmp(flavor, ["edge", "me", "e"]))
        flavor_name = 'edge';
        raster_format = 1;
        raster_nodata = 0;
    else
        error('Invalid image/raster "flavor": %s', flavor);
    end
end
