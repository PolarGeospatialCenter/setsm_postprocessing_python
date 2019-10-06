function [testdir] = test_getTestDir()

systype = computer;

if contains(systype, 'WIN')
    homedir = getenv('USERPROFILE');
elseif contains(systype, 'LNX')
    homedir = getenv('HOME');
end

testdir = fullfile(homedir, 'scratch', 'setsm_postprocessing_testFiles');

% if contains(systype, 'WIN')
%     testdir = 'E:\scratch\test\s2s\testFiles';
% %    testdir = 'V:\pgc\data\scratch\erik\test\s2s\testFiles';
% elseif contains(systype, 'LNX')
%    testdir = '/mnt/pgc/data/scratch/erik/test/s2s/testFiles';
% end
