import argparse
import glob
import os
import sys

import numpy as np

_script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '../..'))
from lib.raster_array_tools import extractRasterData


class InvalidArgumentError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


def main():
    parser = argparse.ArgumentParser(description=(
        "Make sure matchtags only have zeros and ones."
        "\nPrint filenames of all invalid matchtags."))

    parser.add_argument('dir',
        help="Path to directory containing *_matchtag.tif files.")

    # Parse and validate arguments.
    args = parser.parse_args()
    matchdir = os.path.abspath(args.dir)

    if not os.path.isdir(matchdir):
        parser.error("matchdir must be a directory")

    matchFiles = glob.glob(os.path.join(matchdir, '*_matchtag.tif'))

    for m in matchFiles:
        array = extractRasterData(m, 'array')
        if np.any(np.right_shift(array, 1)):
            print os.path.basename(m)



if __name__ == '__main__':
    main()
