import argparse
import glob
import os
import sys

import numpy as np

_script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '../..'))
from lib.raster_array_tools import extractRasterData


def main():
    parser = argparse.ArgumentParser(description=(
        "Make sure matchtags contain only zeros and ones."
        "\nPrint filenames of all invalid matchtags and "
        "store results in a text file."))

    parser.add_argument('src',
        help="Path to directory containing *_matchtag.tif files.")

    parser.add_argument('-o', '--out', default=os.path.join(os.getcwd(), 'check_match_results.txt'),
        help="File path of results text file (default is './check_match_results.txt').")

    # Parse and validate arguments.
    args = parser.parse_args()
    srcdir = os.path.abspath(args.src)
    outfile = os.path.abspath(args.out)
    outdir = os.path.dirname(outfile)

    if not os.path.isdir(srcdir):
        parser.error("src must be a directory")
    if os.path.isfile(outfile):
        parser.error("out file already exists")
    if not os.path.isdir(os.path.dirname(outdir)):
        print "Creating directory for output results file: {}".format(outdir)
        os.makedirs(outdir)

    matchFiles = glob.glob(os.path.join(srcdir, '*_matchtag.tif'))

    results_fp = open(outfile, 'w')
    for m in matchFiles:
        array = extractRasterData(m, 'array')
        if np.any(array > 1):
            affected_fname = os.path.basename(m)
            print affected_fname
            results_fp.write(affected_fname+'\n')
    results_fp.close()

    print "Done!"



if __name__ == '__main__':
    main()
