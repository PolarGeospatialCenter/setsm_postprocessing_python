function [testFile] = test_findTestFile(fname_or_file)

test_setGlobals();
global TESTDIR;


testFile = char(fname_or_file);
if ~(exist(testFile, 'file') == 2)
    testFile = [testFile,'.tif'];
    if ~(exist(testFile, 'file') == 2)
        testFile = [TESTDIR,'/',char(fname_or_file)];
        if ~(exist(testFile, 'file') == 2)
            testFile = [testFile,'.tif'];
            if ~(exist(testFile, 'file') == 2)
                error("Cannot find `testFile`: '%s'", fname_or_file);
            end
        end
    end
end
