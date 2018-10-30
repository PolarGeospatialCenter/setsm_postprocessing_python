import cProfile
import os
import re
import numpy as np
from sys import stdout

from testing import TESTDIR
from testing import test
from lib.raster_array_tools import imresize, imerode_slow, imdilate_slow

testImageFile = os.path.join(TESTDIR, 'image.tif')
statsFile = os.path.join(TESTDIR, 'timing_stats.csv')


import contextlib
@contextlib.contextmanager
def capture():
    import sys
    from cStringIO import StringIO
    oldout, olderr = sys.stdout, sys.stderr
    try:
        out=[StringIO(), StringIO()]
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()

cprof_time_re = re.compile(".*\s*\d+ function calls in (\d*\.\d+) seconds.*")


fcall_template = "{}(arr, struct, mode='{}')"
function_modes = ['conv', 'scipy', 'skimage']
# function_modes = ['auto']
functions = ['imdilate_slow', 'imerode_slow']
structures = [np.ones((x, x), dtype=np.float32) for x in [5, 9, 13, 17, 19, 21, 25]]
array_sizes = [0.1, 0.25, 0.5, 0.75, 1]
# array_sizes = [1, 1.25, 1.5, 1.75, 2]

col_names = ['array_elements', 'structure_elements', 'function', 'mode', 'time']
row_entry = [''] * len(col_names)

repeat_number = 1
repeat_times = np.zeros((repeat_number, 1))


array = test.readImage(testImageFile)
print("Writing to '{}'".format(statsFile))

statsFile_fp = open(statsFile, 'w')
statsFile_fp.write(','.join(col_names)+'\n')
for arr_sz_factor in array_sizes:
    print(">>> arr_sz_factor: {} <<<".format(arr_sz_factor))
    arr = imresize(array, arr_sz_factor)
    row_entry[0] = str(np.prod(arr.shape))
    for struct in structures:
        print(">> structure shape: {} <<".format(struct.shape))
        row_entry[1] = str(np.prod(struct.shape))
        for fun in functions:
            print("~ function: {} ~".format(fun))
            row_entry[2] = fun
            for mode in function_modes:
                stdout.write("mode: {}".format(mode))
                row_entry[3] = mode
                # stdout.write("mode: ")

                for i in range(repeat_number):

                    with capture() as out:
                        cProfile.run(fcall_template.format(fun, mode))

                    # mode = out[0].split('\n')[0]
                    # stdout.write(mode)
                    # row_entry[3] = mode

                    match = re.match(cprof_time_re, out[0])
                    time = match.group(1)
                    stdout.write(" --> {} sec".format(time))

                    repeat_times[i] = float(time)

                time = str(np.average(repeat_times))

                row_entry[4] = time
                statsFile_fp.write(','.join(row_entry)+'\n')

                stdout.write("\n".format(time))

statsFile_fp.close()
