"""Random draws for PET: which stream they come from, and the sampler that makes them.

Every draw PET makes -- prior realisations, perturbed observations, outlier
and crash replacement, the auto-adaptive localization's shuffle, popt's
control perturbations -- goes through the ensemble's ``rng``. That is a
``numpy.random.RandomState`` seeded from the ensemble config's ``seed`` when
one is given, so a run is reproducible on its own and leaves NumPy's global
state untouched; without a seed it is :class:`GlobalRandomStream`, which
draws from the global functions exactly as PET always did, so
``np.random.seed(...)`` keeps controlling a run.
"""

import numpy as np
from scipy import linalg

__all__ = ["GlobalRandomStream", "random_stream", "gen_real"]


class GlobalRandomStream:
    """NumPy's global random functions behind a ``RandomState``-shaped object.

    Exists for two reasons: the ``numpy.random`` module itself cannot be
    pickled, and the ensemble is pickled by its emergency dump; and a named
    object makes it visible in code that a draw comes from the global stream.
    """

    def randn(self, *shape):
        """As ``numpy.random.randn``, on the global stream."""
        return np.random.randn(*shape)

    def standard_normal(self, size=None):
        """As ``numpy.random.standard_normal``, on the global stream."""
        return np.random.standard_normal(size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """As ``numpy.random.normal``, on the global stream."""
        return np.random.normal(loc, scale, size)

    def rand(self, *shape):
        """As ``numpy.random.rand``, on the global stream."""
        return np.random.rand(*shape)

    def uniform(self, low=0.0, high=1.0, size=None):
        """As ``numpy.random.uniform``, on the global stream."""
        return np.random.uniform(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        """As ``numpy.random.choice``, on the global stream."""
        return np.random.choice(a, size=size, replace=replace, p=p)

    def permutation(self, x):
        """As ``numpy.random.permutation``, on the global stream."""
        return np.random.permutation(x)

    def multivariate_normal(self, mean, cov, size=None):
        """As ``numpy.random.multivariate_normal``, on the global stream."""
        return np.random.multivariate_normal(mean, cov, size)

    def get_state(self):
        """As ``numpy.random.get_state``, on the global stream."""
        return np.random.get_state()

    def set_state(self, state):
        """As ``numpy.random.set_state``, on the global stream."""
        np.random.set_state(state)

    def __reduce__(self):
        return (GlobalRandomStream, ())


def random_stream(seed=None):
    """The stream a run draws from: a private ``RandomState`` if ``seed`` is given, else the global one."""
    if seed is None:
        return GlobalRandomStream()
    return np.random.RandomState(int(seed))


def gen_real(mean, var, number, rng=None, limits=None, return_chol=False):
    """Realisations of a Gaussian with the given mean and (co)variance.

    Draw for draw the same as ``geostat.decomp.Cholesky.gen_real`` -- the
    same shapes drawn in the same order with the same arithmetic -- so runs
    are bit-identical to what geostat produced; only the stream is a
    parameter now.

    Parameters
    ----------
    mean : array-like, shape (n,)
        Mean vector.
    var : array-like
        Variance vector ``(n,)``, covariance matrix ``(n, n)``, or a scalar
        when ``mean`` has one element.
    number : int
        Number of realisations.
    rng : RandomState-like, optional
        The stream to draw from; the global one by default.
    limits : dict, optional
        ``{'lower': ..., 'upper': ...}`` to clip the realisations to.
    return_chol : bool, optional
        Also return the factor used: ``sqrt(var)`` for a diagonal, the upper
        Cholesky factor otherwise.

    Returns
    -------
    ndarray, shape (n, number), and the factor when ``return_chol``.
    """
    rng = random_stream() if rng is None else rng
    var = np.asarray(var)
    if len(mean) == 1 or var.ndim == 1:
        factor = np.sqrt(var)
    elif np.count_nonzero(var - np.diagonal(var)) == 0:
        factor = np.sqrt(var)                       # diagonal: no factorisation needed
    else:
        factor = linalg.cholesky(var)               # upper triangular, var = factor.T @ factor

    if var.ndim == 1:
        real = (np.dot(np.expand_dims(mean, axis=1), np.ones((1, number)))
                + np.expand_dims(factor, axis=1) * rng.randn(np.size(mean), number))
    else:
        real = (np.tile(np.reshape(mean, (len(mean), 1)), (1, number))
                + np.dot(factor.T, rng.randn(np.size(mean), number)))

    if limits is not None:
        real[real > limits['upper']] = limits['upper']
        real[real < limits['lower']] = limits['lower']

    return (real, factor) if return_chol else real
