import argparse
import os
import subprocess
import glob


def main():
    parser = argparse.ArgumentParser(description=(
        "Link a selection of files from one directory to another."))

    parser.add_argument('srcdir',
        help="Source directory containing files to be linked.")
    parser.add_argument('dstdir',
        help="Destination directory for linked files.")
    parser.add_argument('flist',
        help=("Text file containing filename prefixes of source "
              "files to be linked, each on a separate line."))

    parser.add_argument("-v", "--verbose", action="store_true", default=False,
        help="Print actions.")
    parser.add_argument("--dryrun", action="store_true", default=False,
        help="Print actions without executing.")

    # Parse and validate arguments.
    args = parser.parse_args()
    srcdir = os.path.abspath(args.srcdir)
    dstdir = args.dstdir
    flist = args.flist

    if not os.path.isdir(srcdir):
        parser.error("srcdir must be a valid directory")
    if not os.path.isfile(flist):
        parser.error("flist does not exist")
    if not os.path.isdir(dstdir):
        os.makedirs(dstdir)
    dstdir = os.path.abspath(dstdir)

    # Open file list and start linking!
    flist_fp = open(flist, 'r')
    fname = flist_fp.readline().strip()
    while fname != "":
        for fname_full in [os.path.basename(f) for f in glob.glob("{}/{}*".format(srcdir, fname))]:

            srcfile = os.path.join(srcdir, fname_full)
            dstfile = os.path.join(dstdir, fname_full)

            cmd = r"ln {} {}".format(srcfile, dstfile)

            if args.dryrun or args.verbose:
                print cmd

            if not args.dryrun:
                subprocess.call(cmd, shell=True)

        fname = flist_fp.readline().strip()

    flist_fp.close()

    print "Done!"



if __name__ == '__main__':
    main()
