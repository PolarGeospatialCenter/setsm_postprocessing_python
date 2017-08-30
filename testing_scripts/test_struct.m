function [var_struct] = test_struct(varargin)

for i = 1:nargin
    eval(['var_struct.',inputname(i),' = varargin(i);']);
end
