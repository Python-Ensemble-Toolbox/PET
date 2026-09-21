"""Rescaling a state variable must rescale its update, and nothing else.

Both analyses work in a scaled state space: anomalies are divided by the prior
standard deviation (``state_scaling``) and the step is multiplied back. If any
term forgets one half of that, changing the units of a variable changes the
update of the others, or its own update by the wrong factor. Multiplying one
variable's ensemble, prior and standard deviation by ``c`` must therefore
multiply that variable's rows of the step by ``c`` and leave the other rows
untouched.
"""

import numpy as np
import pytest

from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.analysis.full import full_update


class NoLocalization:
    name = None


class Scheme:
    """Plain attributes: the context approx_update and full_update read."""

    def __init__(self, enX_prior, state_scaling, cov, ne, seed=3):
        self.lam = 0.5
        self.trunc_energy = 0.99
        self.keys_da = {"emp_cov": False}
        self.localization = NoLocalization()
        self.proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)
        self.prior_enX = enX_prior
        self.state_scaling = state_scaling
        self.cov_data = cov
        self.scale_data = np.sqrt(cov)
        self.Am = None


def _case(seed=0, nx=6, nd=20, ne=10):
    rng = np.random.default_rng(seed)
    prior = rng.standard_normal((nx, ne)) * np.array([1, 1, 1, 5, 5, 5])[:, None]
    enX = prior + 0.3 * rng.standard_normal((nx, ne))
    enY = rng.standard_normal((nd, ne)) * 2 + 1
    enE = enY.mean(1)[:, None] + rng.normal(0, 0.4, size=enY.shape)
    std = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    cov = 0.1 + rng.random(nd)
    return prior, enX, enY, enE, std, cov


@pytest.mark.parametrize("analysis", [approx_update, full_update], ids=["approx", "full"])
def test_rescaling_one_variable_rescales_only_its_rows_of_the_step(analysis):
    prior, enX, enY, enE, std, cov = _case()
    ne = enX.shape[1]
    rows = slice(3, 6)      # the variable whose units we change
    c = 100.0

    step = analysis(Scheme(prior, std, cov, ne)).update(enX, enY, enE, prior=prior).step

    prior_c, enX_c, std_c = prior.copy(), enX.copy(), std.copy()
    prior_c[rows] *= c
    enX_c[rows] *= c
    std_c[rows] *= c
    step_c = analysis(Scheme(prior_c, std_c, cov, ne)).update(enX_c, enY, enE, prior=prior_c).step

    np.testing.assert_allclose(step_c[rows], c * step[rows], rtol=1e-9)
    np.testing.assert_allclose(step_c[:3], step[:3], rtol=1e-9)
