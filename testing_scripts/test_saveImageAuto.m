function test_saveImageAuto(array, keyword, descr, compare, concurrent)
% Saves an indexed image in a test file directory speicifed by test_setGlobals.m.

if ~exist('keyword', 'var') || isempty(keyword)
    keyword = '';
end
if ~exist('descr', 'var') || isempty(descr)
    descr = '';
end
if ~exist('compare', 'var') || isempty(compare)
    compare = false;
end
if ~exist('concurrent', 'var') || isempty(concurrent)
    concurrent = false;
end


keyword = pad(strrep(keyword(1:min(5, length(keyword))), '~', '-'), 5, '_');

description = '';
if ~strcmp(descr, '')
    description = ['~',strrep(descr, ' ', '-')];
end

% Save the test image.
runnum = test_getRunnum();
imgnum = test_getNextImgnum(runnum, compare, concurrent);
if isempty(imgnum)
    imgnum = 1;
end
sz = size(array);
testFname = sprintf('run%03d_%03d_ml_img_%s_%dx%d%s.tif', ...
    runnum, imgnum, keyword, sz(1), sz(2), description);
test_saveImage(array, testFname);
