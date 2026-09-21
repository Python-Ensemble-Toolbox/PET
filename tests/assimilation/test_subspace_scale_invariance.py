"""A weight-space update must not depend on the units of the data.

Scaling the predictions, the observations and the data scaling by the same
factor changes nothing about the problem, so the ensemble weights must come
out identical. ``subspace_update`` once took the SVD of the *unwhitened*
anomalies while whitening the residual and the observation perturbations, and
the weights depended on the units of the data; this pins the fix.
"""

import numpy as np

from pipt.update_schemes.analysis.subspace import subspace_update


class Scheme:
    """Plain attributes only: exactly the context subspace_update reads."""

    def __init__(self, scale, ne):
        self.scale_data = scale
        self.trunc_energy = 0.99
        self.iteration = 0
        self.lam = 0
        self.proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)


def _w_step(pred, obs, scale):
    scheme = Scheme(scale, pred.shape[1])
    return subspace_update(scheme).update(np.zeros((3, pred.shape[1])), pred, obs).w_step


def test_weights_are_invariant_to_the_units_of_the_data():
    rng = np.random.default_rng(0)
    nd, ne = 30, 12
    pred = rng.standard_normal((nd, ne)) * 3 + 1
    obs = pred.mean(1)[:, None] + rng.normal(0, 0.5, size=pred.shape)
    scale = 0.2 + 3 * rng.random(nd)

    reference = _w_step(pred, obs, scale)
    rescaled = _w_step(4 * pred, 4 * obs, 4 * scale)

    np.testing.assert_allclose(rescaled, reference, rtol=1e-10)
