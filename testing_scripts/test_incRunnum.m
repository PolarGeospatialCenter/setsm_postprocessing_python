function [new_runnum] = test_incRunnum(concurrent)

if ~exist('concurrent', 'var') || isempty(concurrent)
    concurrent = false;
end


new_runnum = test_setRunnum([], true, concurrent);
