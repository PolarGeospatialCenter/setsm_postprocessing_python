#!/usr/bin/env python2


import argparse
import os
import subprocess


SRCDIR = r"C:\Users\husby036\Documents\Git\setsm_postprocessing"
DSTDIR = r"C:\Users\husby036\Documents\Git\setsm_postprocessing_python\testing\matlab"
DEPTH_LIMIT = 0
COPYROOTFOLDER = False

EXCLUDE_DNAMES = [
    "",
]
EXCLUDE_FNAMES = [
    "",
]
# NOTE: The following must be absolute paths.
EXCLUDE_DPATHS = [
    r"",
]
EXCLUDE_FPATHS = [
    r"",
]

FNAME_PREFIX = "test_"
DNAME_PREFIX = ""
FNAME_CONTAINS = ""
DNAME_CONTAINS = ""
FNAME_SUFFIX = ""
DNAME_SUFFIX = ""


default_srcdir = SRCDIR if SRCDIR is not None and SRCDIR != "" else None
default_dstdir = DSTDIR if DSTDIR is not None and DSTDIR != "" else os.getcwd()
default_depth = str(DEPTH_LIMIT)
default_copyrootfolder = COPYROOTFOLDER

default_fprefix = FNAME_PREFIX if FNAME_PREFIX is not None and FNAME_PREFIX != "" else None
default_dprefix = DNAME_PREFIX if DNAME_PREFIX is not None and DNAME_PREFIX != "" else None
default_fcontains = FNAME_CONTAINS if FNAME_CONTAINS is not None and FNAME_CONTAINS != "" else None
default_dcontains = DNAME_CONTAINS if DNAME_CONTAINS is not None and DNAME_CONTAINS != "" else None
default_fsuffix = FNAME_SUFFIX if FNAME_SUFFIX is not None and FNAME_SUFFIX != "" else None
default_dsuffix = DNAME_SUFFIX if DNAME_SUFFIX is not None and DNAME_SUFFIX != "" else None


def main():
    parser = argparse.ArgumentParser(description=(
        "Creates hardlinks within dst directory of all files within src directory,"
        " preserving directory structure of the source directory file tree."))

    parser.add_argument('src', nargs='?', default=default_srcdir,
        help=("Path to source directory containing files to link to."))

    parser.add_argument('dst', nargs='?', default=default_dstdir,
        help="Path to destination directory where file links will be created.")

    parser.add_argument('--fprefix', default=default_fprefix,
        help="Only include files with a name that starts with this string.")

    parser.add_argument('--dprefix', default=default_dprefix,
        help="Only include directories with a name that starts with this string.")

    parser.add_argument('--fcontains', default=default_fcontains,
        help="Only include files with a name that contains this string.")

    parser.add_argument('--dcontains', default=default_dcontains,
        help="Only include directories with a name that contains this string.")

    parser.add_argument('--fsuffix', default=default_fsuffix,
        help="Only include files with a name that ends with this string.")

    parser.add_argument('--dsuffix', default=default_dsuffix,
        help="Only include directories with a name that ends with this string.")

    parser.add_argument('--depth', default=default_depth,
        help="Depth of recursion, in terms of directory levels below the level of the root."
             " Value of 0 will link only files in dst directory. Value of 'inf' (sans quotes)"
             " will traverse the whole directory tree.")

    parser.add_argument('--copyrootfolder', action='store_true', default=default_copyrootfolder,
        help="Create the link file tree within the root folder 'dst/{basename(src)}/'."
             " If false, create the link file tree within 'dst/'")

    # Parse arguments.
    args = parser.parse_args()
    srcdir = os.path.abspath(args.src)
    dstdir = os.path.abspath(args.dst)
    FNAME_PREFIX = args.fprefix
    DNAME_PREFIX = args.dprefix
    FNAME_CONTAINS = args.fcontains
    DNAME_CONTAINS = args.dcontains
    FNAME_SUFFIX = args.fsuffix
    DNAME_SUFFIX = args.dsuffix
    copyrootfolder = args.copyrootfolder
    depth_limit = args.depth

    # Validate arguments.
    if not os.path.isdir(srcdir):
        parser.error("src must be a directory")
    if not os.path.isdir(dstdir):
        parser.error("dst must be a directory")
    try:
        depth_limit = int(depth_limit) if str(depth_limit) != 'inf' else float(depth_limit)
        if depth_limit < 0:
            raise ValueError
    except ValueError:
        parser.error("depth must be 'inf' (sans quotes) or a positive integer")
    DEPTH_LIMIT = depth_limit

    if copyrootfolder:
        link_rootdir = os.path.join(dstdir, os.path.basename(srcdir))
        os.makedirs(link_rootdir)
        dstdir = link_rootdir

    linkDir(srcdir, dstdir, 0)


def linkDir(srcdir, dstdir, depth):
    for dirent in os.listdir(srcdir):
        main_dirent = os.path.join(srcdir, dirent)
        link_dirent = os.path.join(dstdir, dirent)

        if os.path.isdir(main_dirent) and depth < DEPTH_LIMIT \
           and dirent not in EXCLUDE_DNAMES and main_dirent not in EXCLUDE_DPATHS \
           and (DNAME_PREFIX is None or dirent.startswith(DNAME_PREFIX)) \
           and (DNAME_CONTAINS is None or DNAME_CONTAINS in dirent) \
           and (DNAME_SUFFIX is None or dirent.endswith(DNAME_SUFFIX)):
            # The directory entry is a subdirectory to traverse.
            if not os.path.isdir(link_dirent):
                os.makedirs(link_dirent)
            linkDir(main_dirent, link_dirent, depth+1)

        elif not os.path.isfile(link_dirent) \
           and dirent not in EXCLUDE_FNAMES and main_dirent not in EXCLUDE_FPATHS \
           and (FNAME_PREFIX is None or dirent.startswith(FNAME_PREFIX)) \
           and (FNAME_CONTAINS is None or FNAME_CONTAINS in dirent) \
           and (FNAME_SUFFIX is None or dirent.endswith(FNAME_SUFFIX)):
            # The directory entry is a file to link.
            cmd = r'mklink /h {} {}'.format(link_dirent, main_dirent)
            subprocess.call(cmd, shell=True)



if __name__ == '__main__':
    main()
