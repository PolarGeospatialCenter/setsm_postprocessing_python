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

array_class = class(array);

if strcmp(array_class, 'logical')
    imwrite(array, testFile);
    
else
    
sf = [];
array_genclass = [];
if contains(array_class, 'uint')
    array_genclass = 'uint';
    sf = Tiff.SampleFormat.UInt;
elseif contains(array_class, 'int')
    array_genclass = 'int';
    sf = Tiff.SampleFormat.Int;
elseif any(strcmp(array_class, ["single", "double"]))
    sf = Tiff.SampleFormat.IEEEFP;
else
    error('Invalid array class: %s\n', array_class);
end
bps = [];
if ~isempty(array_genclass)
    bps = str2num(strrep(array_class, array_genclass, ''));
elseif strcmp(array_class, 'single')
    bps = 32;
elseif strcmp(array_class, 'double')
    bps = 64;
else
    error('Invalid array class: %s\n', array_class);
end

t = Tiff(testFile, 'w');
setTag(t, 'ImageLength', size(array, 1))
setTag(t, 'ImageWidth',  size(array, 2))
setTag(t, 'Photometric', Tiff.Photometric.MinIsBlack);
setTag(t, 'BitsPerSample', bps);
setTag(t, 'SampleFormat', sf);
setTag(t, 'SamplesPerPixel', 1);
setTag(t, 'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);
t.write(array);
    
end

fprintf("'%s' saved\n", strrep(testFile, TESTDIR, '{TESTDIR}'));
