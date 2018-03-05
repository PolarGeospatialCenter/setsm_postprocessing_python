function [testFile] = test_getTestFileFromFname(fname)

test_setGlobals();
global TESTDIR;


if ~contains(fname, '.tif')
    fname = strcat(fname, '.tif');
end
testFile = [TESTDIR,'/',fname];
while exist(testFile, 'file') == 2
    opt = input(sprintf('Test file "%s" already exists. Overwrite? (y/n): ', strrep(testFile, TESTDIR, '{TESTDIR}')), 's');
    if strcmpi(opt, 'y')
        break;
    else
        opt = input('Append description to filename (or press [ENTER] to cancel): ', 's');
        if isempty(opt)
            testFile = [];
            return;
        else
            testFile = strrep(testFile, '.tif', ['~',strrep(opt,' ','-'),'.tif']);
        end
    end
end
