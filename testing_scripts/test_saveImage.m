function test_saveImage(array, fname)

if ~exist('fname', 'var') || isempty(fname)
    fname = 'testImage_ml.tif';
end

test_setGlobals();
global TESTDIR;


if ~contains(fname, '.tif')
    fname = strcat(fname, '.tif');
end

testFile = [TESTDIR,'/',fname];
while exist(testFile, 'file') == 2
    opt = input(sprintf('Test image "%s" already exists. Overwrite? (y/n): ', testFile), 's');
    if strcmpi(opt, 'y')
        break;
    else
        opt = input('Append description to filename (or press [ENTER] to cancel): ', 's');
        if isempty(opt)
            return;
        else
            testFile = strrep(testFile, '.tif', ['_',strrep(opt,' ','-'),'.tif']);
        end
    end
end

if isa(array, 'single')
    t = Tiff(testFile, 'w');
    setTag(t, 'ImageLength', size(array, 1))
    setTag(t, 'ImageWidth',  size(array, 2))
    setTag(t, 'Photometric', Tiff.Photometric.MinIsBlack);
    setTag(t, 'SampleFormat', Tiff.SampleFormat.IEEEFP);
    setTag(t, 'BitsPerSample', 32)
    setTag(t, 'SamplesPerPixel', 1)
    setTag(t, 'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);
    t.write(array);
else
    imwrite(array, testFile);
end
fprintf("'%s' saved\n", testFile);
