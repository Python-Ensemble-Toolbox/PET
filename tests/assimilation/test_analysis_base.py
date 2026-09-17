"""Tests for the shared analysis base.

The three analysis flavours used to each carry a private copy of ``solve`` and
``sqrtm``. Those copies had drifted: ``approx_update`` used ``A.ndim`` while the
others used ``np.ndim(A)``, so only the latter tolerated a covariance supplied
as a plain list or scalar. These tests pin the consolidated behaviour.
"""

import numpy as np
import pytest

from pipt.update_schemes.analysis import AnalysisBase
from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.analysis.full import full_update
from pipt.update_schemes.analysis.subspace import subspace_update
from pipt.update_schemes.analysis.subspace2 import subspace2_update

FLAVOURS = [approx_update, full_update, subspace_update, subspace2_update]


@pytest.mark.parametrize("flavour", FLAVOURS, ids=lambda c: c.__name__)
def test_flavours_share_the_strategy_base(flavour):
    assert issubclass(flavour, AnalysisBase)


@pytest.mark.parametrize("flavour", FLAVOURS, ids=lambda c: c.__name__)
def test_flavours_no_longer_define_private_helpers(flavour):
    """Helpers must come from the base, not a per-file copy."""
    assert "solve" not in vars(flavour)
    assert "sqrtm" not in vars(flavour)


def test_base_is_abstract():
    with pytest.raises(TypeError):
        AnalysisBase()


# ----------------------------------------------------------------------
# solve
# ----------------------------------------------------------------------

def test_solve_diagonal_matches_dense_equivalent():
    diag = np.array([2.0, 4.0])
    B = np.array([[1.0, 3.0], [2.0, 8.0]])
    np.testing.assert_allclose(
        AnalysisBase.solve(diag, B),
        AnalysisBase.solve(np.diag(diag), B),
    )


def test_solve_dense_is_a_true_inverse_apply():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    B = np.array([[1.0], [2.0]])
    np.testing.assert_allclose(A @ AnalysisBase.solve(A, B), B, atol=1e-12)


def test_solve_accepts_list_covariance():
    """Regression: approx_update's old `A.ndim` raised AttributeError here."""
    out = AnalysisBase.solve([2.0, 4.0], np.ones((2, 2)))
    np.testing.assert_allclose(out, [[0.5, 0.5], [0.25, 0.25]])


# ----------------------------------------------------------------------
# sqrtm
# ----------------------------------------------------------------------

def test_sqrtm_diagonal():
    np.testing.assert_allclose(AnalysisBase.sqrtm(np.array([4.0, 9.0])), [2.0, 3.0])


def test_sqrtm_accepts_list():
    np.testing.assert_allclose(AnalysisBase.sqrtm([4.0, 9.0]), [2.0, 3.0])


def test_sqrtm_dense_squares_back():
    A = np.array([[4.0, 0.0], [0.0, 9.0]])
    root = AnalysisBase.sqrtm(A)
    np.testing.assert_allclose(root @ root, A, atol=1e-10)


# ----------------------------------------------------------------------
# Scheme + flavour combinations resolve to the right strategy
# ----------------------------------------------------------------------

def test_scheme_registry_selects_the_right_strategy():
    """Each ``(scheme, analysis)`` combination binds the matching strategy.

    The eighteen per-flavour classes (``esmda_approx``, ``lmenrml_full``, ...)
    used to *inherit* their strategy, so ``issubclass(esmda_approx,
    approx_update)`` held. They are gone now: ``ESMDA``/``LMEnRML``/``GNEnRML``
    take ``analysis`` as a constructor argument and *hold* an analysis
    instance instead. What matters -- which strategy a given combination
    uses -- is what this asserts.
    """
    from pipt.update_schemes.esmda import ESMDA
    from pipt.update_schemes.enrml import GNEnRML, LMEnRML
    from pipt.update_schemes.registry import get_scheme

    for scheme_name, flavour_name, algorithm, flavour_cls in [
        ("esmda", "approx", ESMDA, approx_update),
        ("lmenrml", "full", LMEnRML, full_update),
        ("gnenrml", "subspace", GNEnRML, subspace_update),
    ]:
        ctor = get_scheme(scheme_name, flavour_name)
        assert ctor.func is algorithm
        assert ctor.keywords == {"analysis": flavour_name}
        assert not issubclass(algorithm, AnalysisBase), (
            f"{algorithm.__name__} should hold an analysis, not inherit one"
        )
