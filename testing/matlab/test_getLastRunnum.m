function [last_runnum] = test_getLastRunnum()

test_setGlobals();
global TESTDIR;


last_runnum = [];

testFiles = dir([TESTDIR,'/run*']);
if ~isempty(testFiles)
    runnums = char(testFiles.name);
    runnums = str2num(runnums(:, 4:6));
    last_runnum = max(runnums);
end
