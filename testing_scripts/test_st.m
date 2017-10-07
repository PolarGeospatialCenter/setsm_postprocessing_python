function [var_struct] = test_st(varargin)
% Test Struct

for i = 1:nargin
    eval(['var_struct.',inputname(i),' = varargin(i);']);
end
