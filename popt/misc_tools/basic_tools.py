"""
Collection of simple, yet useful Python tools
"""


def index2d(list2d, value):
    """
    Search in a 2D list for pattern or value and return is (i, j) index. If the
    pattern/value is not found, (None, None) is returned

    Example:

    >>> l = [['string1', 1], ['string2', 2]]
    >>> print index2d(l, 'string1')
    (0, 0)

    Parameters
    ----------
    list2d : list of lists
        2D list.

    value : object
        Pattern or value to search for.

    Returns
    -------
    ind : tuple
        Indices (i, j) of the value.
    """
    return next(((i, j) for i, lst in enumerate(list2d) for j, x in enumerate(lst) if x == value), None)
