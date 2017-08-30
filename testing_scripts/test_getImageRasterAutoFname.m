function [testFname] = test_getImageRasterAutoFname(array, array_name, flavor_name, matchkey, descr, compare, concurrent, isRaster)

runnum = test_getRunnum();
imgnum = test_getNextImgnum(runnum, compare, concurrent);
if isempty(imgnum)
    imgnum = 1;
end
flavor = pad(flavor_name(1:min(5, length(flavor_name))), 5, '_');
if isRaster
    filetype = 'ras';
else
    filetype = 'img';
end
key = '';
if ~isempty(char(matchkey))
    if strcmp(matchkey, 'auto')
        key = array_name;
    elseif strcmp(matchkey, 'flavor')
        [key,~,~] = test_interpretImageRasterFlavor(array_name);
    else
        key = matchkey;
    end
    key = strrep(strrep(key, '~', '-'), '_', '-');
end
description = '';
if ~strcmp(descr, '')
    description = ['~',strrep(descr, ' ', '-')];
end

sz = size(array);
testFname = sprintf('run%03d_%03d_ml_%s_%s_%s_%dx%d%s.tif', ...
    runnum, imgnum, filetype, flavor, key, sz(1), sz(2), description);
