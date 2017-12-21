function test_saveImage(array, fname, dtype_out, do_casting)

if ~exist('fname', 'var') || isempty(fname)
    fname = 'testImage_ml.tif';
end
if ~exist('dtype_out', 'var') || isempty(dtype_out)
    dtype_out = [];
end
if ~exist('do_casting', 'var') || isempty(do_casting)
    do_casting = false;
end

test_setGlobals();
global TESTDIR;


testFile = test_getTestFileFromFname(fname);
if isempty(testFile)
    return;
end

dtype_in = class(array);
array_out = array;
if ~isempty(dtype_out)
    if ~strcmp(dtype_in, dtype_out)
        warning("Input array data type (%s) differs from specified output data type (%s)\n", ...
            dtype_in, dtype_out);
        if do_casting
            fprintf("Casting array to output data type");
            array_out = eval(sprintf('%s(array);', dtype_out));
        end
    end
end

array_class = class(array_out);

if strcmp(array_class, 'logical')
    imwrite(array_out, testFile);
    
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
setTag(t, 'ImageLength', size(array_out, 1))
setTag(t, 'ImageWidth',  size(array_out, 2))
setTag(t, 'Photometric', Tiff.Photometric.MinIsBlack);
setTag(t, 'SampleFormat', sf);
setTag(t, 'BitsPerSample', bps);
setTag(t, 'SamplesPerPixel', 1);
setTag(t, 'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);
t.write(array_out);
    
end

fprintf("'%s' saved\n", strrep(testFile, TESTDIR, '{TESTDIR}'));
