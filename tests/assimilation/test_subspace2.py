"""The ensemble-transform IES flavour, and what pins it.

`subspace2` has no reference output to check against, so the anchor is an identity:
it is exactly `margis` with the marginalised error scale `Ratio` fixed at 1, i.e.
with the data uncertainty taken as known. That identity is what the method *is*, so
a test of it documents the method as well as checking it.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pipt.update_schemes.analysis import ANALYSES, available_analyses
from pipt.update_schemes.analysis.margis import margIS_update
from pipt.update_schemes.analysis.subspace2 import subspace2_update
from pipt.update_schemes.enkf import EnKF
from pipt.update_schemes.enrml import GNEnRML, LMEnRML
from pipt.update_schemes.es import ES
from pipt.update_schemes.esmda import ESMDA

ND, NE = 12, 8


def _scheme(scale_data, *, iteration=0, lam=0.0, current_W=None):
    proj = (np.eye(NE) - np.ones((NE, NE)) / NE) / np.sqrt(NE - 1)
    layout = SimpleNamespace(row_datatypes=lambda: np.array(["d"] * ND, dtype=object))
    scheme = SimpleNamespace(
        ne=NE, iteration=iteration, lam=lam, proj=proj,
        scale_data=scale_data, data_layout=layout,
    )
    if current_W is not None:
        scheme.current_W = current_W
    return scheme


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(ND, NE)), rng.normal(size=(ND, NE))


# --------------------------------------------------------------------------
# The identity that defines the method
# --------------------------------------------------------------------------

class _margis_ratio_one(margIS_update):
    """margis with the inverse-chi2 belief on the error scale switched off."""

    def update(self, enX, enY, enE, **kwargs):
        scheme = self.scheme
        ne = scheme.ne
        if scheme.iteration == 0:
            scheme.current_W = np.eye(ne)
            scheme.D = self.solve(scheme.scale_data, enE)

        sY = self.solve(scheme.scale_data, enY)
        Y = np.linalg.solve(scheme.current_W.T, sY.T).T
        Y = Y @ scheme.proj * np.sqrt(ne - 1)

        delta = scheme.D - sY
        ratio = 1.0                                     # <- the whole difference
        deltaD = (Y * ratio).T @ delta
        S = (Y * ratio).T @ Y + np.eye(ne) * (ne - 1)
        deltaM = (ne - 1) * (np.eye(ne) - scheme.current_W)

        from pipt.update_schemes.analysis.base import AnalysisResult
        return AnalysisResult(W_step=np.linalg.solve(S, deltaM + deltaD) / (1 + scheme.lam))


@pytest.mark.parametrize("scale_data", [
    np.linspace(0.5, 2.0, ND),                                   # diagonal
    np.diag(np.linspace(0.5, 2.0, ND)),                          # full matrix
])
@pytest.mark.parametrize("lam", [0.0, 3.0])
def test_subspace2_is_margis_with_the_error_scale_known(scale_data, lam):
    enY, enE = _inputs()

    a = subspace2_update(_scheme(scale_data, lam=lam)).update(None, enY, enE)
    b = _margis_ratio_one(_scheme(scale_data, lam=lam)).update(None, enY, enE)

    np.testing.assert_array_equal(a.W_step, b.W_step)


def test_the_identity_holds_away_from_the_first_iteration():
    """W = I only at the start; the transform enters through `solve(W.T, sY.T)`."""
    enY, enE = _inputs(seed=1)
    scale_data = np.linspace(0.5, 2.0, ND)
    W = np.eye(NE) + 0.1 * np.random.default_rng(2).normal(size=(NE, NE))

    scheme_a = _scheme(scale_data, iteration=1, current_W=W.copy())
    scheme_a.D = subspace2_update(scheme_a).solve(scale_data, enE)
    scheme_b = _scheme(scale_data, iteration=1, current_W=W.copy())
    scheme_b.D = scheme_a.D

    a = subspace2_update(scheme_a).update(None, enY, enE)
    b = _margis_ratio_one(scheme_b).update(None, enY, enE)

    np.testing.assert_array_equal(a.W_step, b.W_step)


# --------------------------------------------------------------------------
# The transform starts at I, and the data uncertainty is divided out
# --------------------------------------------------------------------------

def test_the_transform_is_initialised_to_the_identity():
    """`subspace` starts from W = 0 and `subspace2` from W = I; the reconstruction in
    `propose_state` differs accordingly, so getting this wrong is silent."""
    scheme = _scheme(np.ones(ND))
    enY, enE = _inputs()

    subspace2_update(scheme).update(None, enY, enE)

    np.testing.assert_array_equal(scheme.current_W, np.eye(NE))


def test_the_observations_are_rewhitened_every_call():
    """ES-MDA redraws the observations and their scale at every assimilation step, so
    a D cached on the first call would drive later steps with the first step's
    observations whitened by the first step's factor."""
    enY, enE = _inputs()
    scheme = _scheme(np.ones(ND), iteration=1, current_W=np.eye(NE))
    scheme.D = np.zeros((ND, NE))                # a stale cache, if one were consulted

    result = subspace2_update(scheme).update(None, enY, enE)
    expected = subspace2_update(_scheme(np.ones(ND), iteration=1, current_W=np.eye(NE))
                                ).update(None, enY, enE)

    np.testing.assert_array_equal(result.W_step, expected.W_step)


def test_the_data_scale_divides_out():
    """Scaling the observations and predictions together must not move the step."""
    enY, enE = _inputs(seed=3)
    plain  = subspace2_update(_scheme(np.ones(ND))).update(None, enY, enE)
    scaled = subspace2_update(_scheme(np.full(ND, 7.0))).update(None, 7.0 * enY, 7.0 * enE)

    np.testing.assert_allclose(plain.W_step, scaled.W_step, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def test_the_flavour_is_registered():
    assert "subspace2" in available_analyses()
    assert ANALYSES["subspace2"] is subspace2_update


@pytest.mark.parametrize("scheme", [ESMDA, LMEnRML, GNEnRML])
def test_the_schemes_that_accept_it(scheme):
    assert scheme.COMPATIBLE_ANALYSES["subspace2"] is subspace2_update


@pytest.mark.parametrize("scheme", [EnKF, ES])
def test_the_sequential_schemes_do_not(scheme):
    """Neither took it upstream, and the sequential path already fails for the
    weight-space flavours -- see the note on CASES in test_numerical_characterisation."""
    assert "subspace2" not in scheme.COMPATIBLE_ANALYSES
