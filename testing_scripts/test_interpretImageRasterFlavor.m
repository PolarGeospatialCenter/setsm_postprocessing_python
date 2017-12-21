function [flavor_name, raster_format, raster_nodata] = test_interpretImageRasterFlavor(flavor)

flavor_name = '';
raster_format = [];
raster_nodata = [];

if ~isempty(char(flavor))
    if any(strcmp(flavor, ["dem", "dem_array", "z", "Z", "Zsub"]))
        flavor_name = 'dem';
        raster_format = 4;
        raster_nodata = -9999;
    elseif any(strcmp(flavor, ["match", "match_array", "m", "mt", "M", "Msub"]))
        flavor_name = 'match';
        raster_format = 1;
        raster_nodata = 0;
    elseif any(strcmp(flavor, ["ortho", "ortho_array", "o", "or", "O", "Osub"]))
        flavor_name = 'ortho';
        raster_format = 2;
        raster_nodata = 0;
    elseif any(strcmp(flavor, ["mask", "mask_array"]))
        flavor_name = 'mask';
        raster_format = 1;
        raster_nodata = 0;
    elseif any(strcmp(flavor, ["data", "data_array", "md", "d"]))
        flavor_name = 'data';
        raster_format = 1;
        raster_nodata = 0;
    elseif any(strcmp(flavor, ["edge", "edge_array", "me", "e"]))
        flavor_name = 'edge';
        raster_format = 1;
        raster_nodata = 0;
    else
        error('Invalid image/raster "flavor": %s', flavor);
    end
end
