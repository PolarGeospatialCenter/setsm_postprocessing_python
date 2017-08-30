function [testFile] = test_findFile(testFile)

test_setGlobals();
global TESTDIR;


testFile = char(testFile);
if ~(exist(testFile, 'file') == 2)
    testFile_temp = [testFile,'.tif'];
    if ~(exist(testFile_temp, 'file') == 2)
        testFile_temp = [TESTDIR,'/',testFile];
        if ~(exist(testFile_temp, 'file') == 2)
            testFile_temp = [testFile_temp,'.tif'];
            if ~(exist(testFile_temp, 'file') == 2)
                error('cannot find test file');
            end
        end
    end
    testFile = testFile_temp;
end
