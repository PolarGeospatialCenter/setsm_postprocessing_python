function [testDir] = test_getTestDir()

systype = computer;

if contains(systype, 'WIN')
%     testDir = 'D:\setsm_postprocessing_tests\testFiles';
    testDir = 'V:\pgc\data\scratch\erik\setsm_postprocessing_tests\testFiles';
elseif contains(systype, 'LNX')
    testDir = '/mnt/pgc/data/scratch/erik/setsm_postprocessing_tests/testFiles';
end