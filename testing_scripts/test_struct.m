function [Z] = test_struct(varargin)

for i = 1:nargin
    eval(['Z.',char(96+i),' = varargin(i);']);
end
