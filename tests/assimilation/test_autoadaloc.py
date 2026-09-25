"""Comprehensive tests for localization methods and facade behavior."""

import numpy as np
from  pipt.misc_tools.analysis_tools import truncSVD
from pipt.update_schemes.analysis import approx_update
from pipt.localization import (
    AutoAdaptiveLocalization,
    build_localization_instance,
)


NX = 8
NY = 4
NE = 10

X = np.array([
    [1, 3, 2, 5, 4, 6, 7, 8, 9, 10],
    [2, 1, 4, 3, 6, 5, 8, 7, 10, 9],
    [5, 4, 6, 3, 7, 2, 8, 1, 10, 9],
    [3, 6, 2, 7, 1, 8, 4, 9, 5, 10],
    [7, 3, 8, 2, 9, 1, 10, 4, 6, 5],
    [1, 4, 3, 6, 2, 7, 5, 9, 8, 10],
    [8, 5, 9, 4, 10, 3, 7, 2, 6, 1],
    [4, 2, 6, 1, 7, 3, 8, 5, 10, 9],
], dtype=float) # shape: (NX, NE)

Y = np.array([
    [1, 2, 3, 5, 4, 6, 8, 7, 9, 10],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    [4, 6, 1, 8, 3, 7, 2, 10, 5, 9],
    [2, 8, 4, 7, 1, 9, 3, 6, 10, 5],
], dtype=float) # shape: (NY, NE)

X = X[:, :NE]  # shape: (NX, NE)
Y = Y[:, :NE]  # shape: (NY, NE)

# Correlation matrix
R = np.corrcoef(X, Y)[:NX, NX:] # Shape: (NX, NY)

def test_config_autoadaloc():
    loc_info = {
        "name": "autoadaloc",
        "field": [1, 5, 5],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 0.4,
        "type": "soft",
        "projection": "rank-r"
    }
    loc = build_localization_instance(loc_info)

    assert isinstance(loc, AutoAdaptiveLocalization)
    assert loc.name == "autoadaloc"
    assert loc.field == [1, 5, 5]
    assert loc.actnum is None
    assert loc.cutoff == 0.4
    assert loc.tapertype == "soft"
    assert loc.threshold == "fixed"


def test_autoadaloc_no_trunc():
    loc_info = {
        "name": "autoadaloc",
        "field": [4, 2],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 0.005,
        "type": "hard",
        "projection": "rank-r",
    }
    loc = AutoAdaptiveLocalization(loc_info)
    taper = loc(X, Y)
    assert taper.shape == (NX, NY)
    np.testing.assert_allclose(taper, np.ones((NX, NY)))


def test_autoadaloc_partial_trunc():
    loc_info = {
        "name": "autoadaloc",
        "field": [4, 2],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 0.4,
        "type": "hard",
        "projection": "rank-r"
    }
    loc = AutoAdaptiveLocalization(loc_info)
    taper_result = loc(X, Y)

    # Expected taper matrix
    taper_expected = np.where(np.abs(R) >= loc.cutoff, 1, 0)

    np.testing.assert_allclose(taper_result, taper_expected)


def test_autoadaloc_full_trunc():
    loc_info = {
        "name": "autoadaloc",
        "field": [4, 2],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 1.0,
        "type": "hard",
        "projection": "rank-r"
    }
    loc = AutoAdaptiveLocalization(loc_info)
    taper = loc(X, Y)
    assert taper.shape == (NX, NY)
    np.testing.assert_allclose(taper, np.zeros((NX, NY)))


def test_approx_update_with_autoadaloc():
    np.random.seed(128928)  # the perturbed observations below are drawn from the global state

    loc_info = {
        "name": "autoadaloc",
        "field": [4, 2],
        "actnum": None,
        "threshold": "fixed",
        "cutoff": 0.4,
        "type": "hard",
        "projection": "rank-r"
    }

    # Define ensemble matrices
    enX = X.copy()
    enY = Y.copy()
    enE = enY.mean(axis=1)[:, None] + np.random.normal(0, 0.1, size=enY.shape)
    Cdd = 0.1*np.ones(NY)

    # Fake scheme providing exactly the context approx_update reads. A real
    # scheme exposes ensemble-owned state as properties of its own, so a
    # strategy only ever reads scheme.<name> -- a double can be flat.
    class FakeScheme:
        lam = 1.0
        trunc_energy = 0.98
        cov_data = Cdd
        keys_da = {"emp_cov": False}

        def __init__(self, localization):
            self.localization = localization

    # Step with localization
    approx = approx_update(FakeScheme(AutoAdaptiveLocalization(loc_info)))
    step_loc = approx.update(enX, enY, enE).step

    # Step without localization
    approx_no_loc = approx_update(
        FakeScheme(type('localization', (object,), {'name': None})())
    )
    step_no_loc = approx_no_loc.update(enX, enY, enE).step

    # Calculate step manually without localization
    scy = np.sqrt(Cdd)
    PI = (np.eye(NE) - np.ones((NE, NE)) / NE)/ np.sqrt(NE-1)
    X_anom = enX @ PI
    Y_anom = (enY @ PI)  / scy[:, None]
    D_anom = (enE - enY) / scy[:, None]
    Ur, Sr, VrT = truncSVD(Y_anom, energy=0.98)
    X1 = Ur.T @ D_anom
    X2 = X1 / (1 + 1.0 + Sr**2)[:, None]
    X3 = VrT.T @ np.diag(Sr) @ X2
    step_expected_no_loc = X_anom @ X3

    # Calculate step manually with localization
    loc = AutoAdaptiveLocalization(loc_info)
    Y_anom_proj = np.diag(Sr) @ VrT
    taper = loc(X=X_anom, Y=Y_anom_proj)
    Cxy_loc = taper * (X_anom @ Y_anom_proj.T)
    step_loc_expected = Cxy_loc @ X2

    np.testing.assert_allclose(step_loc, step_loc_expected)
    np.testing.assert_allclose(step_no_loc, step_expected_no_loc)
    assert not np.array_equal(step_loc, step_no_loc)
