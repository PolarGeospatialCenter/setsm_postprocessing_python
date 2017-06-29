function [matches] = test_findFnamesMatchingSubstring(dirfiles, substring)

x = arrayfun(@(fname) contains(char(fname), substring), {dirfiles.name}, 'UniformOutput', false);
match = cell2mat(x);
matches = dirfiles(match);
