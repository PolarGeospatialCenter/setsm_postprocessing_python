#!/usr/bin/env python2

# Erik Husby, 2018


import argparse
import os
import platform
import subprocess


###### DO NOT MODIFY ######
LINK_TYPE_HARDLINK = 0
LINK_TYPE_SYMLINK = 1
CMD_RAW = None
###########################


###### SET ARG DEFAULTS ######
SRCDIR = r""
DSTDIR = r""
FNAME_PREFIX = ""
DNAME_PREFIX = ""
FNAME_CONTAINS = ""
DNAME_CONTAINS = ""
FNAME_SUFFIX = ""
DNAME_SUFFIX = ""
DEPTH_LIMIT = 'inf'
COPYROOTFOLDER = False
LINK_TYPE = LINK_TYPE_HARDLINK
##############################


###### SET EXCLUSIONS ######
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
############################


default_srcdir = SRCDIR if SRCDIR is not None and SRCDIR != "" else None
default_dstdir = DSTDIR if DSTDIR is not None and DSTDIR != "" else os.getcwd()
default_depth = str(DEPTH_LIMIT)
default_copyrootfolder = COPYROOTFOLDER
default_hardlink = True if LINK_TYPE == LINK_TYPE_HARDLINK else False
default_symlink = True if LINK_TYPE == LINK_TYPE_SYMLINK else False

default_fprefix = FNAME_PREFIX if FNAME_PREFIX is not None and FNAME_PREFIX != "" else None
default_dprefix = DNAME_PREFIX if DNAME_PREFIX is not None and DNAME_PREFIX != "" else None
default_fcontains = FNAME_CONTAINS if FNAME_CONTAINS is not None and FNAME_CONTAINS != "" else None
default_dcontains = DNAME_CONTAINS if DNAME_CONTAINS is not None and DNAME_CONTAINS != "" else None
default_fsuffix = FNAME_SUFFIX if FNAME_SUFFIX is not None and FNAME_SUFFIX != "" else None
default_dsuffix = DNAME_SUFFIX if DNAME_SUFFIX is not None and DNAME_SUFFIX != "" else None


def main():
    parser = argparse.ArgumentParser(description=(
        "Creates links within dst directory to files within src directory, "
        "preserving directory structure of the source directory file tree."))

    parser.add_argument('src', nargs='?', default=default_srcdir,
        help=("Path to source directory containing files to link to."))

    parser.add_argument('dst', nargs='?', default=default_dstdir,
        help="Path to destination directory where file links will be created.")

    parser.add_argument('--fprefix', default=default_fprefix,
        help="Only include files with a name that starts with this string.")

    parser.add_argument('--fcontains', default=default_fcontains,
        help="Only include files with a name that contains this string.")

    parser.add_argument('--fsuffix', default=default_fsuffix,
        help="Only include files with a name that ends with this string.")

    parser.add_argument('--dprefix', default=default_dprefix,
        help="Only include directories with a name that starts with this string.")

    parser.add_argument('--dcontains', default=default_dcontains,
        help="Only include directories with a name that contains this string.")

    parser.add_argument('--dsuffix', default=default_dsuffix,
        help="Only include directories with a name that ends with this string.")

    parser.add_argument('--depth', default=default_depth,
        help="Depth of recursion, in terms of directory levels below the level of the root."
             " Value of 0 will link only files in src directory. Value of 'inf' (sans quotes)"
             " will traverse the whole directory tree.")

    parser.add_argument('--copyrootfolder', action='store_true', default=default_copyrootfolder,
        help="Create the link file tree within the root folder 'dst/{basename(src)}/'."
             " If false, create the link file tree within 'dst/'.")

    parser.add_argument('--hardlink', action='store_true', default=default_hardlink,
        help="Create hard links.")

    parser.add_argument('--symlink', action='store_true', default=default_symlink,
        help="Create symbolic links.")

    supported_systypes = ('Windows', 'Linux')
    systype = platform.system()
    if systype not in supported_systypes:
        parser.error("Only supported systems are {0}, "
                     "but detected {1}".format(supported_systypes, systype))

    global CMD_RAW
    global FNAME_PREFIX, FNAME_CONTAINS, FNAME_SUFFIX
    global DNAME_PREFIX, DNAME_CONTAINS, DNAME_SUFFIX

    # Parse arguments.
    args = parser.parse_args()
    if args.src is None or args.dst is None:
        parser.error("src and dst must both be specified")
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
    try:
        depth_limit = int(depth_limit) if str(depth_limit) != 'inf' else float(depth_limit)
        if depth_limit < 0:
            raise ValueError
    except ValueError:
        parser.error("depth must be 'inf' (sans quotes) or a positive integer")
    DEPTH_LIMIT = depth_limit
    if args.hardlink and args.symlink:
        parser.error("--hardlink and --symlink options are mutually exclusive")
    elif not (args.hardlink or args.symlink):
        parser.error("one of --hardlink and --symlink options must be specified")

    # Set syntax of linking command, to be evaluated in recursive linkDir method.
    link_cmd = None
    if systype == 'Windows':
        if args.hardlink:
            CMD_RAW = r"r'mklink /h {0} {1}'.format(link_dirent, main_dirent)"
        elif args.symlink:
            CMD_RAW = r"r'mklink {0} {1}'.format(link_dirent, main_dirent)"
    elif systype == 'Linux':
        if args.hardlink:
            CMD_RAW = r"r'ln {0} {1}'.format(main_dirent, link_dirent)"
        elif args.symlink:
            CMD_RAW = r"r'ln -s {0} {1}'.format(main_dirent, link_dirent)"

    if not os.path.isdir(dstdir):
        os.makedirs(dstdir)

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
            cmd = eval(CMD_RAW)
            print cmd
            subprocess.call(cmd, shell=True)



if __name__ == '__main__':
    main()
