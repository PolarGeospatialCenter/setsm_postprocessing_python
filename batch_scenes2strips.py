# Version 1.0; Erik Husby, Claire Porter; Polar Geospatial Center, University of Minnesota; 2017

import argparse
import glob
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filters scene dems in a source directory,"
            " then mosaics them into strips and saves the results."
            "\nBatch work is done in units of strip ID (<catid1_catid2>), as parsed from scene dem"
            " filenames."
        )
    )
    parser.add_argument(
        'srcdir',
        help=(
            "Path to source directory containing scene dems to process."
            " If --dstdir is not specified, this path should contain the substring 'tif_results'."
        )
    )
    parser.add_argument(
        'res',
        choices=['2', '8'],
        help="Resolution of target dems (2 or 8 meters)."
    )
    parser.add_argument(
        '--dstdir',
        help=(
            "Path to directory for output mosaicked strip data"
            " (default is srcdir.replace('tif_results', 'strips'))."
        )
    )
    parser.add_argument(
        '--no-entropy',
        action='store_true',
        default=False,
        help="Use filter without entropy protection."
    )
    parser.add_argument(
        '--remask',
        action='store_true',
        default=False,
        help="Recreate and overwrite all existing edgemasks and datamasks (filtering output)."
    )
    parser.add_argument(
        '--restrip',
        action='store_true',
        default=False,
        help="Recreate and overwrite all existing output strips (mosaicking output)."
    )
    parser.add_argument(
        '--pbs',
        action='store_true',
        default=False,
        help="Submit tasks to PBS."
    )
    parser.add_argument(
        '--qsubscript',
        help=(
            "Path to qsub script to use in PBS submission"
            " (default is qsub_scenes2strips.sh in script root folder)."
        )
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        default=False,
        help="Print actions without executing."
    )
    args = parser.parse_args()

    abs_scriptdir = os.path.abspath(os.path.dirname(sys.argv[0]))
    single_script = os.path.join(abs_scriptdir, 'scenes2strips_single.py')

    # Verify source directory.
    if not os.path.isdir(args.srcdir):
        parser.error("srcdir must be a directory.")
    abs_srcdir = os.path.abspath(args.srcdir)

    # Verify output directory.
    if args.dstdir is None:
        # TODO: Use the following regex method of determining strip output directory?
        # dirtail_pattern = re.compile("[/\\\]tif_results[/\\\]\d+m[/\\\]?$")
        # result = re.search(dirtail_pattern, abs_srcdir)
        # if result:
        #     dirtail = result.group(0)
        #     abs_dstdir = re.sub(
        #         dirtail_pattern,
        #         dirtail.replace('tif_results', 'strips').replace('\\', '\\\\'),
        #         abs_srcdir
        #     )
        # else:
        #     parser.error(
        #         "srcdir does not meet naming convention requirements for setting a default dstdir."
        #     )
        abs_dstdir = abs_srcdir.replace('tif_results', 'strips')
        print "Strip output directory set to: {}".format(abs_dstdir)
    else:
        abs_dstdir = os.path.abspath(args.dstdir)

    # Verify qsub script.
    abs_qsubpath = (os.path.abspath(args.qsubscript) if args.qsubscript is not None
                    else os.path.join(abs_scriptdir, 'qsub_scenes2strips.sh'))
    if not os.path.isfile(abs_qsubpath):
        parser.error("qsubscript path is not valid: {}".format(abs_qsubpath))

    # Find scene dems to be merged into strips.
    src_dems = glob.glob(os.path.join(abs_srcdir, '*dem.tif'))
    if not src_dems:
        print "No scene dems found to merge. Exiting."
        sys.exit(1)

    # Create strip output directory if it doesn't already exist.
    isFirstRun = False
    if not os.path.isdir(abs_dstdir):
        isFirstRun = True
        os.makedirs(abs_dstdir)

    # Find unique strip IDs (<catid1_catid2>).
    stripids = list(set([os.path.basename(s)[14:47] for s in src_dems]))
    stripids.sort()

    # Process each strip ID.
    i = 0
    for stripid in stripids:
        i += 1

        # TODO: Remove the following skip check that that renders existence checks in
        # -t    scenes2strips_single.py useless.
        # If output does not already exist, add to task list.
        dst_dems = glob.glob(os.path.join(abs_dstdir, '*'+stripid+'_seg*_dem.tif'))
        if not dst_dems and args.restrip:
            print "{} output files exist, skipping".format(stripid)

        else:

            # If PBS, submit to scheduler.
            if args.pbs:
                job_name = 's2s{:04g}'.format(i)
                cmd = r'qsub -N {0} -v p1={1},p2={2},p3={3},p4={4},p5={5},p6={6},p7={7},p8={8} {9}'.format(
                    job_name,
                    single_script,
                    abs_srcdir,
                    stripid,
                    args.res,
                    '--no-entropy' * args.no_entropy,
                    '--remask' * args.remask,
                    '--restrip' * args.restrip,
                    '--first-run' * isFirstRun,
                    abs_qsubpath
                )
                print cmd

            # ...else run Python.
            else:
                cmd = r'{0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(
                    'python',
                    single_script,
                    abs_srcdir,
                    stripid,
                    args.res,
                    '--no-entropy' * args.no_entropy,
                    '--remask' * args.remask,
                    '--restrip' * args.restrip,
                    '--first-run' * isFirstRun
                )
                print r'{}, {}'.format(i, cmd)

            if not args.dryrun:
                # TODO: Switch to the following subprocess call once testing is complete:
                # subprocess.call(cmd, shell=True)
                subprocess.call(cmd)



if __name__ == '__main__':
    main()
