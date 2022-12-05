import sys
import numpy as np
from scipy.linalg import toeplitz


def fast_gaussian(dimension, sdev, corr):

    # print('Fast Gaussian!')

    """
    Generates random vector from distribution satisfying Gaussian variogram in dimension up to 3-d.

    Input:
    - dimension  Dimension of grid
    - sdev       Standard deviation
    - corr       Correlation length, in units of block length.
                 Corr may be replaced with a vector of length 3 with correlation length in x-, y- and z-direction.

    Output:
    - x          Random vector.

    The parametrization of the grid is assumed to have size dimension, if dimension is a vector,
    or [dimension,1] if dimension is scalar. The coefficients of the grid is assumed to be reordered
    columnwise into the parameter vector. The grid is assumed to have a local basis.

    Example of use:

    Want to generate a field on a 3-d grid with dimension m x n x p, with correlation length a along first coordinate
    axis, b along second coordinate axis, c alone third coordinate axis, and standard deviation sigma:

      x=fast_gaussian(np.array([m, n, p]),np.array([sigma]),np.array([a b c]))

    If the dimension is n x 1 one can write

      x=fast_gaussian(np.array([n]),np.array([sigma]),np.array([a]))

    If the correlation length is the same in all directions:

      x=fast_gaussian(np.array([m n p]),np.array([sigma]),np.array([a]))

    The properties on the Kronecker product behind this algorithm can be found in
    Horn & Johnson: Topics in Matrix Analysis, Cambridge UP, 1991.

    Note that we add a small number on the diagonal of the covariance matrix to avoid numerical problems with Cholesky
    decomposition (a nugget effect).

    Also note that reshape with order='F' is used to keep the code identical to the Matlab code.

    The method was invented and implemented in Matlab by Geir Nævdal in 2011.
    """

    # Initialize dimension:
    if len(dimension) == 0:
        sys.exit("fast_gaussian: Wrong input, dimension should have length at least 1")
    m = dimension[0]
    n = 1
    p = None
    if len(dimension) > 1:
        n = dimension[1]
    dim = m * n
    if len(dimension) > 2:
        p = dimension[2]
        dim = dim * p
    if len(dimension) > 3:
        sys.exit("fast_gaussian: Wrong input, dimension should have length at most 3")

    # Compute standard deviation
    if len(sdev) > 1:  # check input
        std = 1
    else:
        std = sdev  # the variance will come out through the kronecker product.

    # Initialize correlation length
    if len(corr) == 0:
        sys.exit("fast_gaussian: Wrong input, corr should have length at least 1")
    if len(corr) == 1:
        corr = np.append(corr, corr[0])
    if len(corr) == 2 and p is not None:
        corr = np.append(corr, corr[1])
    corr = np.maximum(corr, 1)

    # first generate the covariance matrix for first dimension
    dist1 = np.arange(m)
    dist1 = dist1 / corr[0]
    t1 = toeplitz(dist1)
    # to avoid problem with Cholesky factorization when the matrix is close to singular we add a small number on the
    # diagonal entries
    t1 = std * np.exp(-t1 ** 2) + 1e-10 * np.eye(m)
    # Cholesky decomposition
    cholt1 = np.linalg.cholesky(t1)

    # generate the covariance matrix for the second dimension
    # to save time - use a copy if possible
    if corr[0] == corr[1] and n == m:
        cholt2 = cholt1
    else:
        dist2 = np.arange(n)
        dist2 = dist2 / corr[1]
        t2 = toeplitz(dist2)
        t2 = std * np.exp(-t2 ** 2) + 1e-10 * np.eye(n)
        cholt2 = np.linalg.cholesky(t2)

    # generate the covariance matrix for the third dimension if required
    cholt3 = None
    if p is not None:
        # use std = 1 to get the correct value
        dist3 = np.arange(p)
        dist3 = dist3 / corr[2]
        t3 = toeplitz(dist3)
        t3 = np.exp(-t3 ** 2) + 1e-10 * np.eye(p)
        cholt3 = np.linalg.cholesky(t3)

    # draw a random variable
    x = np.random.randn(dim, 1)

    # adjust to get the correct covariance matrix, applying Lemma 4.3.1. in Horn & Johnson:
    # we need to adjust to get the correct covariance matrix
    if p is None:  # 2-d
        x = np.dot(np.dot(cholt1, np.reshape(x, (m, n))), cholt2.T)
    else:  # 3-d
        # either dimension 1 and 2 or 2 and 3 need to be grouped together.
        if n <= p:
            x = np.dot(np.dot(np.kron(cholt2, cholt1), np.reshape(x, (m * n, p))), cholt3.T)
        else:
            x = np.dot(np.dot(cholt1, np.reshape(x, (m, n * p))), np.kron(cholt3.T, cholt2.T))

    # reshape back
    x = np.reshape(x, (dim,), order='F')

    if len(sdev) > 1:
        if len(sdev) == len(x):
            x = sdev * x
        else:
            sys.exit('fast_gaussian: Inconsistent dimension of sdev')

    return x
