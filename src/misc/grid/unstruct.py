"""
Convert cornerpoint grids to unstructured grids.

Examples
--------
>>> import pyresito.grid.unstruct as us
>>> import pyresito.io.grdecl as grdecl
>>> g = grdecl.read('~/proj/cmgtools/bld/overlap.grdecl')
"""
# pylint: disable=too-few-public-methods, multiple-statements
import numpy as np


class Ridge (object):
    """A ridge consists of two points, anchored in each their pillar. We only
    need to store the z-values, because the x- and y- values are determined by
    the pillar themselves.
    """
    __slots__ = ['left', 'right']

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def is_not_below(self, other):
        """
        Weak ordering of ridges based on vertical placement.

        Parameters
        ----------
        other : Ridge
            Ridge to be compared to this object.

        Returns
        -------
        bool or None
            True if no point on self is below any on the other,
            None if the ridges cross, and False if there is a point
            on the other ridge that is above any on self.
        """
        # test each side separately. if self is at the same level as other,
        # this should count positively towards the test (i.e. it is regarded
        # as "at least as high"), so use less-or-equal.
        left_above = other.left <= self.left
        right_above = other.right <= self.right

        # if both sides are at least as high, then self is regarded as at
        # least as high as other, and vice versa. if one side is lower and
        # one side is higher, then the ridges cross.
        if left_above:
            return True if right_above else None
        else:
            return None if right_above else False


class Face (object):
    """A (vertical) face consists of two ridges, because all of the faces in a
    hexahedron can be seen as (possibly degenerate) quadrilaterals.
    """
    __slots__ = ['top', 'btm']

    def __init__(self, top, btm):
        self.top = top
        self.btm = btm

    def is_above(self, other):
        """
        Weak ordering of faces based on vertical placement.

        Parameters
        ----------
        other : Face
            Face to be compared to this object.

        Returns
        -------
        bool
            True if all points in face self are above all points in face other, False otherwise.
        """
        # if the bottom of self is aligned with the top of other, then the
        # face itself is considered to be above. since the ridge test also
        # has an indeterminate result, we test explicitly like this
        return True if self.btm.is_not_below(other.top) else False


def conv(grid):
    """
    Convert a cornerpoint grid to an unstructured grid.

    Parameters
    ----------
    grid : dict
        Cornerpoint grid to be converted. Should contain 'COORD', 'ZCORN', 'ACTNUM'.

    Returns
    -------
    dict
        Unstructured grid.
    """
    # extract the properties of interest from the cornerpoint grid
    zcorn = grid['ZCORN']
    actnum = grid['ACTNUM']
    ni, nj, nk = grid['DIMENS']

    # zcorn has now dimensionality (k, b, j, f, i, r) and actnum is (k, j, i);
    # permute the cubes to get the (b, k)-dimensions varying quickest, to avoid
    # page faults when we move along pillars/columns
    zcorn = np.transpose(zcorn,  axes=[2, 3, 4, 5, 1, 0])
    actnum = np.transpose(actnum, axes=[1, 2, 0])

    # memory allocation: number of unique cornerpoints along each pillar, and
    # the index of each cornerpoint into the global list of vertices
    num_cp = np.empty((nj + 1, ni + 1), dtype=np.int32)
    ndx_cp = np.empty((nk, 2, nj, 2, ni, 2), dtype=np.int32)

    # each pillar is connected to at most 2*2 columns (front/back, right/left),
    # and each column has at most 2*nk (one top and one bottom) corners
    corn_z = np.empty((2, 2, 2, nk), dtype=np.float32)
    corn_i = np.empty((2, 2, 2, nk), dtype=np.int32)
    corn_j = np.empty((2, 2, 2, nk), dtype=np.int32)
    corn_a = np.empty((2, 2, 2, nk), dtype=np.bool)

    # get all unique points that are hinged to a certain pillar (p, q)
    for q, p in np.ndindex((nj + 1, ni + 1)):
        # reset number of corners found for this column
        num_corn_z = 0

        for f, r in np.ndindex((2, 2)):  # front/back, right/left
            # calculate the cell index of the column at this position
            j = q - f
            i = p - r

            for b in range(2):  # bottom/top

                # copy depth values for this corner; notice that we have
                # pivoted the zcorn matrix so that values going upwards are in
                # last dim.
                corn_z[f, r, b, :] = zcorn[j, f, i, r, b, :]

                # same for active numbers, but this is reused for top/bottom;
                # if the cell is inactive, then both top and bottom point are.
                corn_a[f, r, b, :] = actnum[j, i, :]

                # save the original indices into these auxiliary arrays so that
                # we can figure out where each point came from after they are
                # sorted
                corn_i[f, r, b] = i
                corn_j[f, r, b] = j
