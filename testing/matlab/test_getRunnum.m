function [runnum] = test_getRunnum()

test_setGlobals();
global TESTDIR;
global PREFIX_RUNNUM;


runnum = -1;

runnumFiles = dir([TESTDIR,'/',PREFIX_RUNNUM,'*']);
if isempty(runnumFiles)
    runnum = test_setRunnum();
elseif length(runnumFiles) == 1
    runnum_fname = runnumFiles(1).name;
    runnum = str2num(runnum_fname(16:18));
else
    runnum = [];
end

if isempty(runnum)
    error(['One dummy file must exist in the test directory ' ...
        'with a name indicating the current runnum for comparison! ' ...
        'e.g. "CURRENT_RUNNUM_001"']);
end
