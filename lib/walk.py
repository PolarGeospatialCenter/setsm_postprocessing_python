
import os

try:
    WALK_LIST_FUNCTION_DEFAULT = os.scandir
except AttributeError:
    WALK_LIST_FUNCTION_DEFAULT = os.listdir

class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)

def walk(srcdir, mindepth=1, maxdepth=float('inf'), list_function=WALK_LIST_FUNCTION_DEFAULT):
    if not os.path.isdir(srcdir):
        raise InvalidArgumentError("`srcdir` directory does not exist: {}".format(srcdir))
    if mindepth < 0 or maxdepth < 0:
        raise InvalidArgumentError("depth arguments must be >= 0")
    srcdir = os.path.abspath(srcdir)
    if mindepth == 0:
        updir = os.path.dirname(srcdir)
        srcdname = os.path.basename(srcdir)
        yield updir, [srcdname], []
    for x in _walk(srcdir, 1, mindepth, maxdepth, list_function):
        yield x

def _walk(rootdir, depth, mindepth, maxdepth, list_function):
    if depth > maxdepth:
        return
    dnames, fnames = [], []
    for dirent in list_function(rootdir):
        if list_function is os.listdir:
            pname = dirent
            dirent_is_dir = os.path.isdir(os.path.join(rootdir, pname))
        else:
            pname = dirent.name
            dirent_is_dir = dirent.is_dir()
        (dnames if dirent_is_dir else fnames).append(pname)
    if mindepth <= depth <= maxdepth:
        yield rootdir, dnames, fnames
    if depth < maxdepth:
        for dname in dnames:
            for x in _walk(os.path.join(rootdir, dname), depth+1, mindepth, maxdepth, list_function):
                yield x
