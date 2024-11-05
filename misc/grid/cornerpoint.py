"""Common functionality for visualization of corner-point grids.

import pyresito.grid.cornerpoint as cp
"""
import itertools as it
import logging
import numpy as np
import numpy.ma as ma


# add a valid log in case we are not run through the main program which
# sets up one for us
log = logging.getLogger(__name__)  # pylint: disable=invalid-name
log.addHandler(logging.NullHandler())


def scatter(cell_field):
    """
    Duplicate all items in every dimension.

    Use this method to duplicate values associated with each cell to
    each of the corners in the cell.

    Parameters
    ----------
    cell_field : numpy.ndarray
        Property per cell in the grid, shape = (nk, nj, ni)

    Returns
    -------
    numpy.ndarray
        Property per corner in the cube, shape = (nk, 2, nj, 2, ni, 2)

    Examples
    --------
    >>> scatter(np.array([[[1, 2],
                           [3, 4]],
                          [[5, 6],
                           [7, 8]]]))
    array([[[[[[1, 1], [2, 2]],
              [[1, 1], [2, 2]]],
             [[[3, 3], [4, 4]],
              [[3, 3], [4, 4]]]],

            [[[[1, 1], [2, 2]],
              [[1, 1], [2, 2]]],
             [[[3, 3], [4, 4]],
              [[3, 3], [4, 4]]]]],

           [[[[[5, 5], [6, 6]],
              [[5, 5], [6, 6]]],
             [[[7, 7], [8, 8]],
              [[7, 7], [8, 8]]]],

            [[[[5, 5], [6, 6]],
              [[5, 5], [6, 6]]],
             [[[7, 7], [8, 8]],
              [[7, 7], [8, 8]]]]]])
    """
    # get the dimensions of the cube in easily understood aliases
    nk, nj, ni = cell_field.shape  # pylint: disable=invalid-name

    # create an extra dimension so that we can duplicate individual values
    flat = np.reshape(cell_field, (nk, nj, ni, 1))

    # duplicate along each of the axis
    dup_i = np.tile(flat, (1, 1, 1, 2))
    dup_j = np.tile(dup_i, (1, 1, 2, 1))
    dup_k = np.tile(dup_j, (1, 2, 1, 1))

    # reformat to a cube with the appropriate dimensions
    corn_field = np.reshape(dup_k, (nk, 2, nj, 2, ni, 2))
    return corn_field


def inner_dup(pillar_field):
    """
    Duplicate all inner items in both dimensions.

    Use this method to duplicate values associated with each pillar to
    the corners on each side(s) of the pillar; four corners for all
    interior pillars, two corners for all pillars on the rim and only
    one (element) corner for those pillars that are on the grid corners.

    Parameters
    ----------
    pillar_field : numpy.ndarray
        Property per pillar in the grid, shape = (m+1, n+1)

    Returns
    -------
    numpy.ndarray
        Property per corner in a grid plane, shape = (2*m, 2*n)

    Examples
    --------
    >>> inner_dup(np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]))
    array([[1, 2, 2, 3],
           [4, 5, 5, 6],
           [4, 5, 5, 6],
           [7, 8, 8, 9]])
    """
    # extract the inner horizontal part of the plane
    horz_part = pillar_field[:, 1:-1]

    # tile it to a separate plane
    horz_dupd = np.tile(horz_part, (2, 1, 1))

    # shift the tile to the end so it can be rolled
    # into the second dimension
    horz_shft = np.transpose(horz_dupd, (1, 2, 0))

    # roll this into the second dimension
    horz_roll = np.reshape(horz_shft,
                           (horz_shft.shape[0],
                            horz_shft.shape[1] * 2))

    # add back the first and last column
    horz = np.column_stack([pillar_field[:, 0],
                            horz_roll,
                            pillar_field[:, -1]])

    # extract the inner vertical part of the plane
    vert_part = horz[1:-1, :]

    # tile it to a separate plane
    vert_dupd = np.tile(vert_part, (1, 1, 2))

    # roll this into the first dimension
    vert_roll = np.reshape(vert_dupd,
                           (vert_dupd.shape[1] * 2,
                            vert_dupd.shape[2] // 2))

    # add back the first and last row
    result = np.vstack([horz[0, :],
                        vert_roll,
                        horz[-1, :]])
    return result


def elem_vtcs_ndcs(nk, nj, ni):  # pylint: disable=invalid-name
    """
    List zcorn indices used by every element in an nk*nj*ni grid.

    Parameters
    ----------
    nk : int
        Number of layers in Z direction.
    nj : int
        Number of elements in the Y direction.
    ni : int
        Number of elements in the X direction.

    Returns
    -------
    ndarray
        Zero-based indices for the hexahedral element corners, 
        with shape (nk*nj*ni, 8) and dtype int.
    """
    # hex_perm is the order a hexahedron should be specified to the
    # gridding engine (it is the same for VTK and Ensight Gold formats)
    # relative to the order they are in the Eclipse format. the latter
    # four is listed first because we turn the z order around with a
    # transform.
    # local indices:      0, 1, 2, 3, 4, 5, 6, 7
    hex_perm = np.array([4, 5, 7, 6, 0, 1, 3, 2], dtype=int)

    # kji is the global (k, j, i) index for every element; we tile
    # this into 8 corners per element, with 3 indices per corner;
    # this this matrix contains repeats of 8 identical permutations
    # of the values 0..(ni-1), 0..(nj-1) and 0..(nk-1)
    kji = np.array(list(it.product(range(nk),
                                   range(nj),
                                   range(ni))))
    kji = np.reshape(np.tile(kji, [1, hex_perm.shape[0]]), [-1, 3])

    # bfr is the local (bottom, front, right) index for every corner;
    # this is a block of 8 local corners which are picked out of the
    # zcorn matrix to then correspond with the gridding engines order.
    # we tile this block for every element
    bfr = np.array(list(it.product([0, 1], repeat=3)))[hex_perm, :]
    bfr = np.tile(bfr, [nk*nj*ni, 1])

    # calculate running number for every corner, combining the element
    # indices kji with the local corner indices bfr. the zcorn hypercube
    # has dimensions nk * 2 * nj * 2 * ni * 2; the indexing is simply a
    # flattened view of this hypercube.
    ndx = (((2*kji[:, 0] + bfr[:, 0]) * 2*nj +
            (2*kji[:, 1] + bfr[:, 1])) * 2*ni +
           2*kji[:, 2] + bfr[:, 2])

    # we then collect the indices so that we get 8 of them (all those that
    # belongs to the same element) on a row
    ndx = np.reshape(ndx, [-1, hex_perm.shape[0]])
    return ndx


# pylint: disable=invalid-name, multiple-statements, too-many-locals
def corner_coordinates(coord, zcorn):
    """
    Generate (x, y, z) coordinates for each corner-point.

    Parameters
    ----------
    coord : numpy.ndarray
        Pillar geometrical information.
        Shape: (nj+1, ni+1, 2, 3)
    zcorn : numpy.ndarray
        Depth values along each pillar.
        Shape: (nk, 2, nj, 2, ni, 2)

    Returns
    -------
    numpy.ndarray
        Coordinate values for each corner.
        Shape: (3, nk*2*nj*2*ni*2)
    """
    # indices to treat the pillar matrix as a record
    X = 0
    Y = 1
    Z = 2
    START = 0
    END = 1

    # calculate depth derivatives for the pillar slope
    dx = coord[:, :, END, X] - coord[:, :, START, X]
    dy = coord[:, :, END, Y] - coord[:, :, START, Y]
    dz = coord[:, :, END, Z] - coord[:, :, START, Z]

    # make the pillar information compatible with a plane of corners,
    # having a singleton dimension at the front (instead tiling 2*nk)
    x0 = inner_dup(coord[:, :, START, X])[None, :, :]
    y0 = inner_dup(coord[:, :, START, Y])[None, :, :]
    z0 = inner_dup(coord[:, :, START, Z])[None, :, :]
    dxdz = inner_dup(dx / dz)[None, :, :]
    dydz = inner_dup(dy / dz)[None, :, :]

    # infer the grid size from the dimensions of coord
    nj = coord.shape[0] - 1
    ni = coord.shape[1] - 1

    # make the formal dimensions of the depths compatible with the plane
    # of vectors created from the pillars; notice that the first dimension
    # is not singular
    z = np.reshape(zcorn, (-1, nj*2, ni*2))

    # calculate the point in the plane where the horizon intersect with the
    # pillar; we get the (x, y)-coordinates for each corner. notice that the
    # difference z - z0 is for each point, whereas dz is for the pillar!
    x = (z - z0) * dxdz + x0
    y = (z - z0) * dydz + y0

    # reformat the coordinates into the final coordinates
    xyz = np.vstack([np.reshape(x, (1, -1)),
                     np.reshape(y, (1, -1)),
                     np.reshape(z, (1, -1))])
    return xyz


# Enumeration of the dimensions in a point tuple
Dim = type('Dim', (), {
    'X': 0,
    'Y': 1,
    'Z': 2,
})


# Enumeration of the faces of a hexahedron. The contents are the local
# indices in each row of 8 corners that are set up for the elements by
# the elem_vtcs_ndcs function. (The indices here are the numbers that
# are in the comment "local indices" above the hex_perm selector)
Face = type('Face', (), {
    'DOWN':  np.array([0, 1, 2, 3]),
    'UP':    np.array([4, 5, 6, 7]),
    'FRONT': np.array([4, 7, 3, 0]),
    'BACK':  np.array([5, 6, 2, 1]),
    'LEFT':  np.array([7, 6, 2, 3]),
    'RIGHT': np.array([4, 5, 1, 0]),
    'ALL':   np.array([0, 1, 2, 3, 4, 5, 6, 7]),
})


def cp_cells(grid, face):
    """
    Make a cell array from a cornerpoint grid. The cells will be in the
    same order as the Cartesian enumeration of the grid.

    Parameters
    ----------
    grid : dict
        Pillar coordinates and corner depths. Must contain 'coord' and 'zcorn' properties.
    face : Face enum
        Which face that should be extracted, e.g. Face.UP.

    Returns
    -------
    dict
        Set of geometrical objects that can be sent to rendering. Contains:
        - 'points': ndarray, shape (nverts, 3)
        - 'cells': ndarray, shape (nelems, ncorns), where ncorns is either 8 
          (hexahedron volume) or 4 (quadrilateral face), depending on the face parameter.
    """
    src = {}

    log.debug("Generating points for corner depths")
    # notice that VTK wants shape (nverts, 3), where as the code sets up
    # for Ensight format which is (3, nverts).
    src['points'] = corner_coordinates(grid['COORD'],
                                       grid['ZCORN']).T.copy()

    # order of the dimensions are different from the structure to the call
    ni, nj, nk = grid['DIMENS']

    # extract only the columns of the face we want to show. the shape was
    # originally (nelem, 8). copy is necessary to get a contiguous array
    # for VTK
    log.debug("Generating cells")
    src['cells'] = elem_vtcs_ndcs(nk, nj, ni)[:, face].copy()
    return src


def cell_filter(grid, func):
    """
    Create a filter for a specific grid.

    Parameters
    ----------
    grid : dict
        Grid structure that should be filtered.
    func : Callable[(int, int, int), bool]
        Lambda function that takes i, j, k indices (one-based) and returns
        boolean for whether the cells should be included or not. The function
        should be vectorized.

    Returns
    -------
    layr : ndarray
        Filtered grid layer.

    Examples
    --------
    >>> layr = cell_filter(grid, lambda i, j, k: np.greater_equal(k, 56))
    """
    # extract the dimensions; notice that the i-dimensions is first
    # in this list, since it is directly from the grid
    ni, nj, nk = grid['DIMENS']  # pylint: disable=invalid-name

    # setup a grid of the i, j and k address for each of the cells;
    # notice that now the order of the dimensions are swapped so that
    # they fit with the memory layout of a loaded Numpy array
    kji = np.mgrid[1:(nk+1), 1:(nj+1), 1:(ni+1)]

    # call the filter function on all these addresses, and simply
    # return the boolean array of those
    filter_flags = func(kji[2], kji[1], kji[0])
    masked = filter_flags.astype(np.bool)

    # get the mask of active cells, and combine this with the masked
    # cells from the filter, giving us a flag for all visible nodes
    active = grid['ACTNUM']
    visible = np.logical_and(active, masked)

    return visible


def face_coords(grid):
    """
    Get (x, y, z) coordinates for each corner-point.

    Parameters
    ----------
    grid : dict
        Cornerpoint-grid.

    Returns
    -------
    numpy.ndarray
        Coordinate values for each corner. Use the Face enum to index
        the first dimension, k, j, i coordinates to index the next
        three, and Dim enum to index the last dimension. Note that the
        first point in a face is not necessarily the point that is
        closest to the origin of the grid. Shape = (8, nk, nj, ni, 3).

    Examples
    --------
    >>> import numpy as np
    >>> import pyresito.io.ecl as ecl
    >>> import pyresito.grid.cornerpoint as cp
    >>> case = ecl.EclipseCase("FOO")
    >>> coord_fijkd = cp.face_coords(case.grid())
    >>> # get the midpoint of the upper face in each cell
    >>> up_mid = np.average(coord_fijkd[cp.Face.UP, :, :, :, :], axis=0)
    """
    # get coordinates in native corner-point format
    xyz = corner_coordinates(grid['COORD'], grid['ZCORN'])

    # get extent of grid
    ni, nj, nk = grid['DIMENS']

    # break up the latter dimension into a many-dimensional hypercube; this
    # is an inexpensive operation since it won't have to reshuffle the memory
    xyz = np.reshape(xyz, (3, nk, 2, nj, 2, ni, 2))

    # move all the dimensions local within an element to the front
    xyz = np.transpose(xyz, (2, 4, 6, 1, 3, 5, 0))

    # coalesce the local dimensions so that they end up in one dimension
    # which can be indexed by the Face enum to extract a certain field
    xyz = np.reshape(xyz, (8, nk, nj, ni, 3))

    return xyz


def horizon(grid, layer=0, top=True):
    """
    Extract the points that are in a certain horizon and average them so
    that the result is per cell and not per pillar.

    Parameters
    ----------
    grid : dict
        Grid structure.
    layer : int, optional
        The K index of the horizon. Default is the layer at the top.
    top : bool, optional
        Whether the top face should be exported. If this is False,
        then the bottom face is exported instead.

    Returns
    -------
    numpy.ma.array
        Depth at the specified horizon for each cell center, with shape (nk, nj, ni)
        and dtype numpy.float64.
    """
    # get the dimensions of the grid. notice that this array is always given
    # in Fortran order (since Eclipse works that way) and not in native order
    num_i, num_j, _ = grid['DIMENS']

    # zcorn is nk * (up,down) * nj * (front,back) * ni * (left,right). we first
    # project the particular layer and face that we want to work with, ending
    # up with four corners for each cell.
    vertical = 0 if top else 1
    corners = grid['ZCORN'][layer, vertical, :, :, :, :]

    # move the second and fourth dimension (front/back and left/right) to the
    # back and collapse them into one dimension, meaning that we get
    # (nj, ni, 4). then take the average across this last dimension, ending up
    # with an (nj, ni) grid, which is what we return
    heights = np.average(np.reshape(np.transpose(
        corners, axes=(0, 2, 1, 3)), (num_j, num_i, 2*2)), axis=2)

    # if there are any inactive elements, then mask out the elevation at that
    # spot (it is not really a part of the grid)
    actnum = grid['ACTNUM'][layer, :, :]
    return ma.array(data=heights, mask=np.logical_not(actnum))


def _reduce_corners(plane, red_op):
    """
    Reduce the corners of a plane using a specified reduction operator.

    Parameters
    ----------
    plane : numpy.ndarray
        Z-coordinates for a specific horizon in the grid with shape (nj, 2, ni, 2).
    red_op : numpy.ufunc
        Reduction operator to merge coordinates, either numpy.minimum or numpy.maximum.

    Returns
    -------
    numpy.ndarray
        Extremal value of each point around the corners with shape (nj+1, ni+1).

    Examples
    --------
    >>> plane = np.reshape(np.arange(36) + 1, (3, 2, 3, 2))
    >>> _reduce_corners(plane, np.minimum)
    """
    # constants
    NORTH = 1
    SOUTH = 0
    EAST = 1
    WEST = 0

    # get dimensions of the domain
    nj, _, ni, _ = plane.shape

    # allocate output memory for all pillars
    out = np.empty((nj + 1, ni + 1), dtype=np.float64)

    # views into the original plane, getting a specific corner in every cell.
    # these views then have dimensions (nj, ni)
    nw = plane[:, NORTH, :, WEST]
    ne = plane[:, NORTH, :, EAST]
    sw = plane[:, SOUTH, :, WEST]
    se = plane[:, SOUTH, :, EAST]

    # four corners of the output grid only have a single value. the views have
    # indices 0..n-1, the output array have indices 0..n
    out[0,  0] = sw[0,    0]
    out[nj,  0] = nw[nj-1,    0]
    out[0, ni] = se[0, ni-1]
    out[nj, ni] = ne[nj-1, ni-1]

    # four edges contains values from two windows; first two horizontal edges,
    # (varying along i), then two vertical edges (varying along j). recall that
    # *ranges* in Python have inclusive start, exclusive end.
    out[0, 1:ni] = red_op(sw[0, 1:ni], se[0, 0:ni-1])
    out[nj, 1:ni] = red_op(nw[nj-1, 1:ni], ne[nj-1, 0:ni-1])
    out[1:nj,    0] = red_op(nw[0:nj-1,    0], sw[1:nj,      0])
    out[1:nj,   ni] = red_op(ne[0:nj-1, ni-1], se[1:nj,   ni-1])

    # interior have two nested reduction operations
    out[1:nj, 1:ni] = red_op(
        red_op(nw[0:nj-1,   1:ni], sw[1:nj,   1:ni]),
        red_op(ne[0:nj-1, 0:ni-1], se[1:nj, 0:ni-1]))

    return out


def horizon_pillars(grid, layer=0, top=True):
    """
    Extract heights where a horizon crosses the pillars.

    Parameters
    ----------
    grid : dict
        Grid structure, containing both COORD and ZCORN properties.
    layer : int, optional
        The K index of the horizon. Default is the layer at the top.
    top : bool, optional
        Whether the top face should be exported. If False, then the bottom face is exported instead.

    Returns
    -------
    numpy.ndarray
        Heights of the horizon at each pillar, in the same format as the COORD matrix.
        Shape: (nj+1, ni+1, 2, 3)
    """
    # NumPy's average operations finds the average within arrays, but we want
    # an operation that can take the average across two arrays
    def avg_op(a, b):
        return 0.5 * (a + b)

    # zcorn is nk * (up,down) * nj * (front,back) * ni * (left,right). we first
    # project the particular layer and face that we want to work with, ending
    # up with four corners for each cell. take the average of the heights that
    # are hinged up on each pillar to get a "smooth" surface. (this will not be
    # the same surface that the cornerpoint grid was created from).
    vertical = 0 if top else 1
    points = _reduce_corners(grid['ZCORN'][layer, vertical, :, :, :, :],
                             avg_op)
    return points


def snugfit(grid):
    """
    Create coordinate pillars that match exactly the top and bottom horizon of the grid cells.

    This version assumes that the pillars in the grid are all strictly vertical,
    i.e., the x- and y-coordinates will not be changed.

    Parameters
    ----------
    grid : dict
        Grid structure, containing both COORD and ZCORN properties.

    Returns
    -------
    numpy.ndarray
        Pillar coordinates, in the same format as the COORD matrix.
        Note that a new matrix is returned, the original grid is not altered/updated.
        Shape: (nj+1, ni+1, 2, 3)
    """
    # placement in the plane of each pillar is unchanged
    x = grid['COORD'][:, :, 0, 0]
    y = grid['COORD'][:, :, 0, 1]

    # get implicit dimension of grid
    njp1, nip1 = x.shape

    # get new top and bottom from the z-coordinates; smaller z is shallower
    top = _reduce_corners(grid['ZCORN'][0, 0, :, :, :, :], np.minimum)
    btm = _reduce_corners(grid['ZCORN'][-1, 1, :, :, :, :], np.maximum)

    # combine coordinates to a pillar grid with new z-coordinates
    coord = np.reshape(np.dstack((
        x.ravel(), y.ravel(), top.ravel(),
        x.ravel(), y.ravel(), btm.ravel())), (njp1, nip1, 2, 3))

    return coord


def bounding_box(corn, filtr):
    """
    Calculate the bounding box of the grid.

    This function assumes that the grid is aligned with the geographical axes.

    Parameters
    ----------
    corn : numpy.ndarray
        Coordinate values for each corner. This matrix can be constructed with the `corner_coordinates` function.
        Shape: (3, nk*2*nj*2*ni*2)
    filtr : numpy.ndarray
        Active corners; use scatter of ACTNUM if no filtering.
        Shape: (nk, 2, nj, 2, ni, 2), dtype: numpy.bool

    Returns
    -------
    numpy.ndarray
        Bottom front right corner and top back left corner. Since the matrix is returned with C ordering, it is specified the opposite way of what is natural for mathematical matrices.
        Shape: (2, 3), dtype: numpy.float64

    Examples
    --------
    >>> corn = corner_coordinates(grid['COORD'], grid['ZCORN'])
    >>> bbox = bounding_box(corn, scatter(grid['ACTNUM']))
    >>> p, q = bbox[0, :], bbox[1, :]
    >>> diag = np.sqrt(np.dot(q - p, q - p))
    """
    # scatter the filter mask from each cell to its corners
    mask = np.logical_not(filtr).ravel()

    # build arrays of each separate coordinate based on its inclusion
    x_vals = ma.array(data=corn[0, :], mask=mask)
    y_vals = ma.array(data=corn[1, :], mask=mask)
    z_vals = ma.array(data=corn[2, :], mask=mask)

    # construct bounding box that includes all coordinates in visible grid
    bbox = np.array([[np.min(x_vals), np.min(y_vals), np.min(z_vals)],
                     [np.max(x_vals), np.max(y_vals), np.max(z_vals)]],
                    dtype=corn.dtype)
    return bbox


def mass_center(corn, filtr):
    """
    Mass center of the grid.

    This function will always assume that the density is equal throughout the field.

    Parameters
    ----------
    corn : numpy.ndarray
        Coordinate values for each corner. This matrix can be constructed with the 
        `corner_coordinates` function. Shape = (3, nk*2*nj*2*ni*2).
    filtr : numpy.ndarray
        Active corners; use scatter of ACTNUM if no filtering. Shape = (nk, 2, nj, 2, ni, 2), 
        dtype = numpy.bool.

    Returns
    -------
    numpy.ndarray
        Center of mass. This should be the focal point of the grid. Shape = (3,), dtype = np.float64.

    Examples
    --------
    >>> corn = cp.corner_coordinates(grid['COORD'], grid['ZCORN'])
    >>> focal_point = cp.mass_center(corn, cp.scatter(grid['ACTNUM']))
    """
    # scatter the filter mask from each cell to its corners
    mask = np.logical_not(filtr).ravel()

    # build arrays of each separate coordinate based on its inclusion
    x_vals = ma.array(data=corn[0, :], mask=mask)
    y_vals = ma.array(data=corn[1, :], mask=mask)
    z_vals = ma.array(data=corn[2, :], mask=mask)

    # use this as a coarse approximation to find the element center
    center = np.array([np.average(x_vals),
                       np.average(y_vals),
                       np.average(z_vals)], dtype=corn.dtype)
    return center
