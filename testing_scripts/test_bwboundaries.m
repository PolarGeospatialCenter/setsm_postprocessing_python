function [boundaries_array] = test_bwboundaries(array, connectivity, options)
% test_bwboundaries Takes the same arguments as bwboundaries, but outputs a logical array where the boundary indices are set to 1 and all else are 0.

if ~exist('connectivity', 'var') || isempty(connectivity)
    connectivity = 8;
end
if ~exist('options', 'var') || isempty(options)
    options = 'holes';
end


B = cell2mat(bwboundaries(array, connectivity, options));
boundaries_array = zeros(size(array), 'logical');
boundaries_array(sub2ind(size(array), B(:,1), B(:,2))) = 1;
