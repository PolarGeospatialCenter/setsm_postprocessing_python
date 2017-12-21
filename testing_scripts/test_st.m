function [var_struct] = test_st(varargin)
% Test Struct

for i = 1:nargin
    array = varargin(i);
    if iscell(array)
        array = array{1,1};
    end
    if isstruct(array)
        array = array.z;
    end
    eval(['var_struct.',inputname(i),' = array;']);
end
