function test_sia_one(array, flavor, matchkey, descr, compare, concurrent)
% Save Image Auto -- (For) One (Image) :: Saves an indexed image in the test file directory specified by test_setGlobals.m.

if ~exist('flavor', 'var')
    flavor = [];
end
if ~exist('matchkey', 'var')
    matchkey = [];
end
if ~exist('descr', 'var')
    descr = '';
end
if ~exist('compare', 'var')
    compare = false;
end
if ~exist('concurrent', 'var')
    concurrent = false;
end
array_name = [];


test_sia(array, flavor, matchkey, descr, compare, concurrent);
