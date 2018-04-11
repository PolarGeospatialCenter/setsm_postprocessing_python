import argparse
import glob
import os

import numpy as np


np.set_printoptions(suppress=True)


class InvalidArgumentError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


def main():
    parser = argparse.ArgumentParser(description=(
        "Compares 'Mean/Median Vertical Residual' and "
        "'Translation Vector' sections of "
        "strip registration text files between two directories."))

    parser.add_argument('dir1',
        help="Path to first strips directory containing *_reg.txt files.")
    parser.add_argument('dir2',
        help="Path to second strips directory containing *_reg.txt files.")

    parser.add_argument('-o', '--out', default=os.path.join(os.getcwd(), 'diff_stripreg_results.txt'),
        help="File path of results text file (default is './diff_stripreg_results.txt').")
    parser.add_argument('--skip-flist',
        help=("Text file containing a list of *_reg.txt file names to be "
              "removed from consideration, one on each line."))

    # Parse and validate arguments.
    args = parser.parse_args()
    dir1 = os.path.abspath(args.dir1)
    dir2 = os.path.abspath(args.dir2)
    outFile = os.path.abspath(args.out)
    outDir = os.path.dirname(outFile)
    if args.skip_flist is not None:
        skip_flist = os.path.abspath(args.skip_flist)
    else:
        skip_flist = None

    if not os.path.isdir(dir1):
        parser.error("dir1 must be a directory")
    if not os.path.isdir(dir2):
        parser.error("dir2 must be a directory")
    # if os.path.isfile(outFile):
    #     parser.error("out file already exists")
    if skip_flist is not None and not os.path.isfile(skip_flist):
        parser.error("skip list file does not exist")
    if not os.path.isdir(os.path.dirname(outDir)):
        print "Creating directory for output results file: {}".format(outDir)
        os.makedirs(outDir)

    dir1_fnames = set([os.path.basename(p) for p in glob.glob(os.path.join(dir1, '*_reg.txt'))])
    dir2_fnames = set([os.path.basename(p) for p in glob.glob(os.path.join(dir2, '*_reg.txt'))])

    fnames_comm = dir1_fnames.intersection(dir2_fnames)
    if skip_flist is not None:
        skip_flist_fp = open(skip_flist, 'r')
        skip_fnames = skip_flist_fp.read().strip()
        skip_flist_fp.close()
        skip_fnames = set(skip_fnames.splitlines())
        fnames_comm = fnames_comm.difference(skip_fnames)
    fnames_comm = list(fnames_comm)
    fnames_comm.sort()

    unifnames1 = list(dir1_fnames.difference(dir2_fnames))
    unifnames2 = list(dir2_fnames.difference(dir1_fnames))
    unifnames1.sort()
    unifnames2.sort()

    diff_fp = open(outFile, 'w')
    diff_fp.write("\n")

    if len(unifnames1) > 0 or len(unifnames2) > 0:
        diff_fp.write("*** Strip segmentation differs between dir1 and dir2 ***\n")
        if len(unifnames1) > 0:
            diff_fp.write("\nSegments unique to dir1:\n\n")
            for fname in unifnames1:
                diff_fp.write(fname + "\n")
        if len(unifnames2) > 0:
            diff_fp.write("\nSegments unique to dir2:\n\n")
            for fname in unifnames2:
                diff_fp.write(fname + "\n")
        diff_fp.write("\n")

    diff_fp.write("\n%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%\n"
                    "#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#\n\n")

    val_diffs_all = []

    num_jobs = len(fnames_comm)
    print("Found {} *_reg.txt files in common".format(num_jobs))
    jobnum = 0
    for stripnum, fname in enumerate(fnames_comm):
        jobnum += 1
        print "({}/{}) {}".format(jobnum, num_jobs, fname)

        metaFile1 = os.path.join(dir1, fname)
        metaFile2 = os.path.join(dir2, fname)

        meta_fp1 = open(metaFile1, 'r')
        meta_fp2 = open(metaFile2, 'r')

        skip_lines(meta_fp1, meta_fp2, 1)
        header1, header2 = read_lines(meta_fp1, meta_fp2, 3)
        data1, data2 = read_lines(meta_fp1, meta_fp2, 3)

        meta_fp1.close()
        meta_fp2.close()

        mean_vert_res1, median_vert_res1, trans1 = data1.strip().split('\n')
        mean_vert_res2, median_vert_res2, trans2 = data2.strip().split('\n')

        values1 = np.insert(np.fromstring(trans1.split("=")[1].strip(), sep=', '), 0,
                            [float(mean_vert_res1.split("=")[1].strip()),
                             float(median_vert_res1.split("=")[1].strip())])
        values2 = np.insert(np.fromstring(trans2.split("=")[1].strip(), sep=', '), 0,
                            [float(mean_vert_res2.split("=")[1].strip()),
                             float(median_vert_res2.split("=")[1].strip())])

        val_diffs = values2 - values1
        val_diffs_all.append(val_diffs)

        diff_fp.write("\n\n{}, strip segment name = {}\n\n".format(stripnum+1, fname))

        dir = dir1
        header = header1
        data = data1
        for i in range(2):
            diff_fp.write("\n{}:\n\n".format(dir))

            diff_fp.write("{}{}\n".format(header, data))

            dir = dir2
            header = header2
            data = data2

            if i == 0:
                diff_fp.write("---------------------------------------------------------\n")

        diff_fp.write("\n{}\n".format(np.array2string(val_diffs, max_line_width=np.inf, threshold=np.inf)))

        diff_fp.write("\n\n%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%\n"
                          "#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#\n")

    diff_fp.write("\n\n\n")

    val_diffs_all = np.vstack(val_diffs_all)

    col_labels = "Mean Vertical Residual, Median Vertical Residual, dz, dx, dy"

    line_format = '{:<'+str(len(str(val_diffs_all.shape[0])))+'} {}'
    diff_fp.write("Per-segment difference (dir2 - dir1):\n\n{}\n{}\n\n".format(
        line_format.format("#", " "+np.array2string(np.array([s.strip() for s in col_labels.split(',')]))),
        '\n'.join([line_format.format(n+1, line) for n, line in enumerate(np.array2string(val_diffs_all, max_line_width=np.inf, threshold=np.inf).split('\n'))])
    ))

    stats_array = val_diffs_all

    diff_fp.write("\n".join([
        "Mean   : "+np.array2string(np.nanmean(stats_array, axis=0), max_line_width=np.inf, threshold=np.inf),
        "Median : "+np.array2string(np.nanmedian(stats_array, axis=0), max_line_width=np.inf, threshold=np.inf),
        "StdDev : "+np.array2string(np.nanstd(stats_array, axis=0), max_line_width=np.inf, threshold=np.inf),
        "AbsMax : "+np.array2string(stats_array[np.nanargmax(np.abs(stats_array), axis=0), np.arange(stats_array.shape[1])], max_line_width=np.inf, threshold=np.inf),
        "AbsMax#: "+np.array2string(1+np.nanargmax(np.abs(stats_array), axis=0), max_line_width=np.inf, threshold=np.inf)
        ]
    ))

    diff_fp.write("\n")

    diff_fp.close()

    np.savetxt(outFile+'_Diff.csv', stats_array, delimiter=',', fmt='%0.3f')

    print "Done!"


def read_lines(fp1, fp2, num=0):
    str1, str2 = "", ""
    for i in range(num):
        str1 += fp1.readline()
        str2 += fp2.readline()
    return str1, str2


def skip_lines(fp1, fp2, num=0):
    for i in range(num):
        fp1.readline()
        fp2.readline()



if __name__ == '__main__':
    main()
