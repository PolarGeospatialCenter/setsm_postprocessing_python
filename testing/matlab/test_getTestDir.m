function [testDir] = test_getTestDir()

systype = computer;

if contains(systype, 'WIN')
    homedir = getenv('USERPROFILE');
elseif contains(systype, 'LNX')
    homedir = getenv('HOME');
end

testdir = fullfile(homedir, 'scratch', 'setsm_postprocessing_testFiles');

%if contains(systype, 'WIN')
%%     testDir = 'D:\setsm_postprocessing_tests\testFiles';
%    testDir = 'V:\pgc\data\scratch\erik\setsm_postprocessing_tests\testFiles';
%elseif contains(systype, 'LNX')
%    testDir = '/mnt/pgc/data/scratch/erik/setsm_postprocessing_tests/testFiles';
%end
