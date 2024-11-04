"""\
Extract a sector from an existing cornerpoint grid.
"""
import argparse
import collections
import logging
import numpy
import re
import sys

import misc.grid as pyr
import misc.grdecl as grdecl


# add a valid log in case we are not run through the main program which
# sets up one for us
log = logging.getLogger(__name__)  # pylint: disable=invalid-name
log.addHandler(logging.NullHandler())

# three consecutive integers, separated by comma, and perhaps some spaces
# thrown in for readability, optionally enclosed in parenthesis
tuple_format = re.compile(r'\(?([0-9]+)\ *\,\ *([0-9]+)\ *\,\ *([0-9]+)\)?')


def parse_tuple(corner):
    """
    Parse a coordinate specification string into a tuple of zero-based coordinates.

    Parameters
    ----------
    corner : str
        Coordinate specification in the format "(i1,j1,k1)".

    Returns
    -------
    tuple of int
        The parsed tuple, converted into zero-based coordinates and in Python-matrix order: (k, j, i).
    """
    # let the regular expression engine parse the string
    match = re.match(tuple_format, corner.strip())

    # if the string matched, then we know that each of the sub-groups can be
    # parsed into strings successfully. group 0 is the entire string, so we
    # get natural numbering into the parenthesized expressions. subtract one
    # to get zero-based coordinates
    if match:
        i = int(match.group(1)) - 1
        j = int(match.group(2)) - 1
        k = int(match.group(3)) - 1
        return (k, j, i)
    # if we didn't got any valid string, then return a bottom value
    else:
        return None


def sort_tuples(corner, opposite):
    """
    Parameters
    ----------
    corner : tuple of int
        Coordinates of one corner.
    opposite : tuple of int
        Coordinates of the opposite corner.

    Returns
    -------
    tuple of tuple of int
        The two tuples, but with coordinates interchanged so that one corner is always in the lower, left, back and the other is in the upper, right, front.
    """
    # pick out the most extreme variant in either direction, into each its own
    # variable; this may be the same as the input or not, but at least we know
    # for sure when we return from this method
    least = (min(corner[0], opposite[0]),
             min(corner[1], opposite[1]),
             min(corner[2], opposite[2]))
    most = (max(corner[0], opposite[0]),
            max(corner[1], opposite[1]),
            max(corner[2], opposite[2]))
    return (least, most)


def extract_dimens(least, most):
    """
    Build a new dimension tuple for a submodel.

    Parameters
    ----------
    least : tuple of int
        Lower, left-most, back corner of submodel, (k1, j1, i1).
    most : tuple of int
        Upper, right-most, front corner of submodel, (k2, j2, i2).

    Returns
    -------
    numpy.ndarray
        Dimensions of the submodel.
    """
    # split the corners into constituents
    k1, j1, i1 = least
    k2, j2, i2 = most

    # make an array out of the cartesian distance of the two corners
    sector_dimens = numpy.array([i2-i1+1, j2-j1+1, k2-k1+1], dtype=numpy.int32)
    return sector_dimens


def extract_coord(coord, least, most):
    """
    Extract the coordinate pillars for a submodel.

    Parameters
    ----------
    coord : numpy.ndarray
        Coordinate pillars for the entire grid with shape (nj+1, ni+1, 2, 3).
    least : tuple of int
        Lower, left-most, back corner of submodel, (k1, j1, i1).
    most : tuple of int
        Upper, right-most, front corner of submodel, (k2, j2, i2).

    Returns
    -------
    numpy.ndarray
        Coordinate pillars for the submodel with shape (j2-j1+2, i2-i1+2, 2, 3).
    """
    # split the corners into constituents
    k1, j1, i1 = least
    k2, j2, i2 = most

    # add one to get the pillar on the other side of the element, so that
    # we include the last element, add one more since Python indexing is
    # end-exclusive
    sector_coord = coord[j1:(j2+2), i1:(i2+2), :, :]
    return sector_coord


def extract_zcorn(zcorn, least, most):
    """
    Extract hinge depth values for a submodel from the entire grid.

    Parameters
    ----------
    zcorn : numpy.ndarray
        Hinge depth values for the entire grid with shape (nk, 2, nj, 2, ni, 2).
    least : tuple of int
        Lower, left-most, back corner of submodel, (k1, j1, i1).
    most : tuple of int
        Upper, right-most, front corner of submodel, (k2, j2, i2).

    Returns
    -------
    numpy.ndarray
        Hinge depth values for the submodel with shape (k2-k1+1, 2, j2-j1+1, 2, i2-i1+1).
    """
    # split the corners into constituents
    k1, j1, i1 = least
    k2, j2, i2 = most

    # add one since Python-indexing is end-exclusive, and we want to include
    # the element in the opposite corner. all eight hinges are returned for
    # each element (we only select in every other dimension)
    sector_zcorn = zcorn[k1:(k2+1), :, j1:(j2+1), :, i1:(i2+1), :]
    return sector_zcorn


def extract_cell_prop(prop, least, most):
    """
    Extract the property values for a submodel.

    Parameters
    ----------
    prop : numpy.ndarray
        Property values for each cell in the entire grid with shape (nk, nj, ni).
    least : tuple of int
        Lower, left-most, back corner of submodel, (k1, j1, i1).
    most : tuple of int
        Upper, right-most, front corner of submodel, (k2, j2, i2).

    Returns
    -------
    numpy.ndarray
        Property values for each cell in the submodel with shape (k2-k1+1, j2-j1+1, i2-i1+1).
    """
    # split the corners into constituents
    k1, j1, i1 = least
    k2, j2, i2 = most

    # add one since Python-indexing is end-exclusive, and we want to include
    # the element in the opposite corner.
    sector_prop = prop[k1:(k2+1), j1:(j2+1), i1:(i2+1)]
    return sector_prop


def extract_grid(grid, least, most):
    """
    Extract a submodel from a full grid.

    Parameters
    ----------
    grid : dict
        Attributes of the full grid, such as COORD, ZCORN, ACTNUM.
    least : tuple of int
        Lower, left-most, back corner of the submodel, (k1, j1, i1).
    most : tuple of int
        Upper, right-most, front corner of the submodel, (k2, j2, i2).

    Returns
    -------
    dict
        Attributes of the sector model.
    """
    # create a new grid and fill extract standard properties
    sector = collections.OrderedDict()
    sector['DIMENS'] = extract_dimens(least, most)
    sector['COORD'] = extract_coord(grid['COORD'], least, most)
    sector['ZCORN'] = extract_zcorn(grid['ZCORN'], least, most)
    sector['ACTNUM'] = extract_cell_prop(grid['ACTNUM'], least, most)

    # then extract all extra cell properties, such as PORO, PERMX that
    # may have been added
    for prop_name in grid:
        # ignore the standard properties; they have already been converted
        # by specialized methods above
        prop_upper = prop_name.upper()
        std_prop = ((prop_upper == 'DIMENS') or (prop_upper == 'COORD') or
                    (prop_upper == 'ZCORN') or (prop_upper == 'ACTNUM'))
        # use the generic method to convert this property
        if not std_prop:
            sector[prop_name] = extract_cell_prop(grid[prop_name], least, most)

    return sector


def main(*args):
    """Read a data file to see if it parses OK."""
    # setup simple logging where we prefix each line with a letter code
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname).1s: %(message).76s")

    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", metavar="infile.grdecl", help="Input model")
    parser.add_argument("outfile", metavar="outfile",
                        help="Output sector model")
    parser.add_argument("first", metavar="i1,j1,k1", type=parse_tuple,
                        help="A corner of the sector model (one-based)")
    parser.add_argument("last", metavar="i2,j2,k2", type=parse_tuple,
                        help="The opposite corner of the sector (one-based)")
    parser.add_argument("--dialect", choices=['ecl', 'cmg'], default='ecl')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--quiet", action='store_true')
    cmd_args = parser.parse_args(*args)

    # adjust the verbosity of the program
    if cmd_args.verbose:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    if cmd_args.quiet:
        logging.getLogger(__name__).setLevel(logging.NOTSET)

    # read the input file
    log.info('Reading full grid')
    grid = pyr.read_grid(cmd_args.infile)

    # get the two opposite corners that defines the submodel
    log.info('Determining scope of sector model')
    least, most = sort_tuples(cmd_args.first, cmd_args.last)
    log.info('Sector model is (%d, %d, %d)-(%d, %d, %d)',
             least[2]+1, least[1]+1, least[0]+1,
             most[2]+1, most[1]+1, most[0]+1)

    # extract the data for the submodel into a new grid
    log.info('Extracting sector model from full grid')
    submodel = extract_grid(grid, least, most)

    # write this grid to output
    log.info('Writing sector model to disk')
    grdecl.write(cmd_args.outfile, submodel, cmd_args.dialect,
                 multi_file=True)


# if this module is called as a standalone program, then pass all the
# arguments, except the program name
if __name__ == "__main__":
    main(sys.argv[1:])
