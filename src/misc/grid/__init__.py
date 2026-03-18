"""\
Generic read module which determines format from extension.
"""
import logging
import os.path as pth


# module specific log; add a null handler so that we won't get an
# error if the main program hasn't set up a log
log = logging.getLogger(__name__)  # pylint: disable=invalid-name
log.addHandler(logging.NullHandler())


def read_grid(filename, cache_dir=None):
    """
    Read a grid file and optionally cache it for faster future reads.

    Parameters
    ----------
    filename : str
        Name of the grid file to read, including path.
    cache_dir : str
        Path to a directory where a cache of the grid may be stored to ensure faster read next time.
    """
    # allow shortcut to home directories to be used in paths
    fullname = pth.expanduser(filename)

    # split the filename into directory, name and extension
    base, ext = pth.splitext(fullname)

    if ext.lower() == '.grdecl':
        from misc import grdecl as grdecl
        log.info("Reading corner point grid from \"%s\"", fullname)
        grid = grdecl.read(fullname)

    elif ext.lower() == '.egrid':
        # in case we only have a simulation available, with the binary
        # output from the restart, we can read this directly
        from misc import ecl as ecl
        log.info("Reading binary Eclipse grid from \"%s\"", fullname)
        egrid = ecl.EclipseGrid(base)
        grid = {'DIMENS': egrid.shape[::-1],
                'COORD': egrid.coord,
                'ZCORN': egrid.zcorn,
                'ACTNUM': egrid.actnum}

    elif ext.lower() == '.pickle':
        # direct import of pickled file (should not do this, prefer to read
        # it through a cache directory
        import pickle
        log.info("Reading binary grid dump from \"%s\"", fullname)
        with open(fullname, 'rb') as f:
            grid = pickle.load(f)

    else:
        raise ValueError(
            "File format with extension \"{0}\" is unknown".format(ext))

    return grid
