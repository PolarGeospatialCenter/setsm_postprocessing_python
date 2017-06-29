function [next_imgnum] = test_getNextImgnum(runnum, compare, concurrent)

if ~exist('runnum', 'var') || isempty(runnum)
    runnum = test_getRunnum();
end
if ~exist('compare', 'var') || isempty(compare)
    compare = false;
end
if ~exist('concurrent', 'var') || isempty(concurrent)
    concurrent = false;
end

test_setGlobals();
global TESTDIR;


next_imgnum = -1;

testFiles = dir([TESTDIR,'/',sprintf('run%03d_*', runnum)]);
if concurrent
    testFiles = test_findFnamesMatchingSubstring(testFiles, '_ml_');
end

if isempty(testFiles)
    if compare
        next_imgnum = [];
    else
        next_imgnum = 1;
    end
else
    imgnums = char(testFiles.name);
    imgnums = str2num(imgnums(:, 8:10));
    if compare
        next_imgnum = max(imgnums);
    else
        next_imgnum = max(imgnums)+1;
    end
end
