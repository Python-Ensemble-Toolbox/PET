"""
Tests for optimization workflows using Gaussian ensembles.

These tests validate:
1. Convergence of EnOpt on a quadratic objective
2. Line search optimization behavior
3. High-dimensional optimization (Rosenbrock function)
"""

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import rosen

from popt.ensembles import GaussianEnsemble
from popt.optimization_methods import EnOpt
from popt.optimization_methods import LineSearch
from popt.cost_functions.quadratic import quadratic


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

ENSEMBLE_CONFIG = {
    "ne": 10,
    "natural_gradient": False,
    "controls": {
        "x": {
            "mean": [5] * 2,
            "var": 1.0e-5,
            "limits": [-10, 10],
        }
    },
}

OPT_CONFIG = {
    "transform": True,
    "maxiter": 50,
    "tol": 1e-2,
    "alpha": 0.25,
    "alpha_maxiter": 4,
    "resample": 0,
    "optimizer": "GD",
    "restartsave": False,
    "restart": False,
    "save_data": ["alpha", "obj_func_values"],
}


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def prepare_test_environment(tmp_path: Path, seed: int):
    """
    Set working directory and initialize random seed.
    """
    np.random.seed(seed)
    os.chdir(tmp_path)


def create_ensemble(config, objective):
    """
    Initialize Gaussian ensemble and extract key components.
    """
    ensemble = GaussianEnsemble(config, None, objective)

    return {
        "ensemble": ensemble,
        "x0": ensemble.get_state(),
        "cov": ensemble.get_cov(),
        "bounds": ensemble.get_bounds(),
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_quadratic_enopt(tmp_path):
    """
    Verify EnOpt converges to optimum for quadratic function.
    """
    prepare_test_environment(tmp_path, seed=101122)

    data = create_ensemble(ENSEMBLE_CONFIG, quadratic)
    ensemble = data["ensemble"]

    res = EnOpt.minimize(
        x0=data["x0"],
        fun=ensemble.function,
        jac=ensemble.gradient,
        hess=ensemble.hessian,
        args=(data["cov"],),
        bounds=data["bounds"],
        **OPT_CONFIG,
    )

    np.testing.assert_array_almost_equal(
        res.x, [1.0, 1.0], decimal=1,
        err_msg="EnOpt failed to converge to expected optimum"
    )

    np.testing.assert_array_almost_equal(
        res.fun, [0.0], decimal=1,
        err_msg="Objective value not minimized as expected"
    )


def test_quadratic_linesearch(tmp_path):
    """
    Verify LineSearch converges on quadratic objective.
    """
    prepare_test_environment(tmp_path, seed=101122)

    data = create_ensemble(ENSEMBLE_CONFIG, quadratic)

    result = LineSearch.minimize(
        x0=data["x0"],
        fun=data["ensemble"].function,
        jac=data["ensemble"].gradient,
        args=(data["cov"],),
        bounds=data["bounds"],
        transform=True,
    )

    np.testing.assert_array_almost_equal(
        result.x, [1.0, 1.0], decimal=1,
        err_msg="LineSearch did not converge to expected optimum"
    )

    np.testing.assert_almost_equal(
        result.fun, 0.0, decimal=4,
        err_msg="Final objective value is too large"
    )

def test_rosenbrock_linesearch(tmp_path):
    """
    Verify LineSearch (BFGS) converges on high-dimensional Rosenbrock problem.
    """
    prepare_test_environment(tmp_path, seed=10_08_1997)

    dim = 100

    ensemble_config = {
        "ne": 100,
        "natural_gradient": False,
        "controls": {
            "x": {
                "mean": [-2] * dim,
                "var": 0.001,
                "limits": [-2, 2],
            }
        },
    }

    # Objective wrapped for compatibility with ensemble
    def rosenbrock(x, *args, **kwargs):
        return rosen(x)

    data = create_ensemble(ensemble_config, rosenbrock)

    result = LineSearch.minimize(
        x0=data["x0"],
        fun=data["ensemble"].function,
        jac=data["ensemble"].gradient,
        args=(data["cov"],),
        bounds=data["bounds"],
        method="BFGS",
        maxiter=1000,
        step_size=1.0,
        ftol=1e-8,
        step_size_adapt=0
    )
    print(result)
    expected = np.ones(dim)

    np.testing.assert_array_almost_equal(
        result.x, expected, decimal=0,
        err_msg="Solution deviates significantly from Rosenbrock optimum"
    )

    # Norm-based tolerance for high-dimensional case
    error_norm = np.linalg.norm(result.x - expected)
    tolerance = 0.1 * np.sqrt(dim)

    assert error_norm < tolerance, (
        f"Solution error too large: |x - x*| = {error_norm:.3f} "
        f">= {tolerance:.3f}"
    )



# ----------------------------------------------------------------------
# Objective-call dispatch
# ----------------------------------------------------------------------

def test_typeerror_inside_the_objective_is_not_swallowed(tmp_path):
    """An error from within the objective must surface, not trigger a retry.

    The optimizers support objectives that take only `x` as well as ones taking
    the covariance and extras. That used to be decided by calling the rich form
    and catching TypeError -- which cannot tell "rejected the arguments" from
    "raised TypeError halfway through". The whole evaluation was then repeated,
    and with a real simulator the repeat died on the scratch folders the first
    attempt had created, reporting FileExistsError and hiding the real error.
    """
    prepare_test_environment(tmp_path, seed=1)

    calls = []

    def raises_inside(x, **kwargs):
        calls.append(x)
        raise TypeError("deep inside the objective")

    data = create_ensemble(ENSEMBLE_CONFIG, raises_inside)

    with pytest.raises(TypeError, match="deep inside the objective"):
        EnOpt.minimize(
            x0=data["x0"],
            fun=data["ensemble"].function,
            jac=data["ensemble"].gradient,
            args=(data["cov"],),
            bounds=data["bounds"],
            **OPT_CONFIG,
        )

    assert len(calls) == 1, (
        f"objective evaluated {len(calls)} times for one evaluation; "
        f"the retry-on-TypeError path is back"
    )


def test_objective_taking_only_x_is_still_supported():
    """The case the fallback exists for: no covariance, no extras."""
    def fun(x):
        return float(np.sum((np.asarray(x) - 0.5) ** 2))

    def jac(x):
        return 2.0 * (np.asarray(x, dtype=float) - 0.5)

    res = EnOpt.minimize(
        x0=np.array([2.0]), fun=fun, jac=jac, args=(np.eye(1) * 1e-3,),
        bounds=[(-5, 5)], transform=True, maxiter=15, alpha=0.3, saveit=False,
    )

    np.testing.assert_array_almost_equal(res.x, [0.5], decimal=2)


def test_objective_accepting_neither_shape_is_reported_clearly():
    """Neither (x, *args, **kwargs) nor (x) -> say so, naming the signature.

    Previously this fell through to `func(x)` and failed with whatever
    TypeError that produced, which described the fallback rather than the
    mismatch the user has to fix.
    """
    def wrong(x, y, z):
        return 0.0

    with pytest.raises(TypeError, match=r"accepts neither"):
        EnOpt.minimize(
            x0=np.array([2.0]), fun=wrong, jac=lambda x: np.zeros_like(x),
            args=(np.eye(1) * 1e-3,), bounds=[(-5, 5)],
            transform=True, maxiter=1, saveit=False,
        )
