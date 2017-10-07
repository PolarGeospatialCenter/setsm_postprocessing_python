function [testFname] = test_getImageRasterAutoFname(array, flavor_name, matchkey, descr, compare, concurrent, isRaster)

runnum = test_getRunnum();
imgnum = test_getNextImgnum(runnum, compare, concurrent);
if isempty(imgnum)
    imgnum = 1;
end
if isRaster
    filetype = 'ras';
else
    filetype = 'img';
end
flavor_name = pad(flavor_name(1:min(5, length(flavor_name))), 5, '_');
matchkey = strrep(strrep(matchkey, '~', '-'), '_', '-');
if ~strcmp(descr, '')
    descr = ['~',strrep(descr, ' ', '-')];
end

sz = size(array);
testFname = sprintf('run%03d_%03d_ml_%s_%s_%s_%dx%d%s.tif', ...
    runnum, imgnum, filetype, flavor_name, matchkey, sz(1), sz(2), descr);
