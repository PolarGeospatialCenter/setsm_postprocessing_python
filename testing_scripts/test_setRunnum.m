function [new_runnum] = test_setRunnum(new_runnum, increment, concurrent)

if ~exist('increment', 'var') || isempty(increment)
    increment = false;
end
if ~exist('new_runnum', 'var') || isempty(new_runnum)
    new_runnum = test_getLastRunnum();
    if isempty(new_runnum)
        new_runnum = 0;
    end
    if increment
        new_runnum = new_runnum + 1;
    end
end
if ~exist('concurrent', 'var') || isempty(concurrent)
    concurrent = false;
end

test_setGlobals();
global TESTDIR;
global PREFIX_RUNNUM;


cc_str = '';
if concurrent
    cc_str = '_CC';
end

runnumFile_new = sprintf('%s/%s%03d%s', TESTDIR, PREFIX_RUNNUM, new_runnum, cc_str);

runnumFiles = dir([TESTDIR,'/',PREFIX_RUNNUM,'*']);
if isempty(runnumFiles)
    runnum_file = fopen(runnumFile_new, 'wt');
    fclose(runnum_file);
elseif length(runnumFiles) == 1
    runnumFname_current = runnumFiles(1).name;
    if concurrent && contains(runnumFname_current, cc_str)
        runnumFile_new = [TESTDIR,'/',strrep(runnumFname_current,cc_str,'')];
    end
    try
        movefile([TESTDIR,'/',runnumFname_current], runnumFile_new);
    catch ME
        if ~strcmp(ME.identifier, 'MATLAB:MOVEFILE:SourceAndDestinationSame')
            rethrow(ME);
        end
    end
else
    test_getRunnum();  % Get error message from this function.
end
