function test_saveImage(array, fname)

if ~exist('fname', 'var') || isempty(fname)
    fname = 'testImage_ml.tif';
end

test_setGlobals();
global TESTDIR;


testFile = test_getTestFileFromFname(fname);
if isempty(testFile)
    return;
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
fprintf("'%s' saved\n", strrep(testFile, TESTDIR, '{TESTDIR}'));
