from pathlib import Path
from scipy.optimize import rosen, rosen_der, rosen_hess
import numpy as np
import pytest
import os

from popt.optimization_methods import LineSearch


def test_line_search_gradient_descent(tmp_path: Path):
    """Verify gradient descent converges with and without bound transforms."""
    os.chdir(tmp_path)

    x0 = np.array([-1.2, -1.0])
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    expected = np.array([1.0, 1.0])

    for transform in (False, True):
        res = LineSearch.minimize(
            x0,
            fun=rosen,
            jac=rosen_der,
            method="GD",
            bounds=bounds,
            maxiter=50_000,
            transform=transform,
        )
        np.testing.assert_allclose(res.x, expected, atol=1e-4)


def test_line_search_bfgs(tmp_path: Path):
    """Verify BFGS converges with and without bound transforms."""
    os.chdir(tmp_path)

    x0 = np.array([-1.2, -1.0])
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    expected = np.array([1.0, 1.0])

    for transform in (False, True):
        res = LineSearch.minimize(
            x0,
            fun=rosen,
            jac=rosen_der,
            method="BFGS",
            bounds=bounds,
            transform=transform,
        )
        np.testing.assert_allclose(res.x, expected, atol=1e-4)


def test_line_search_newton_cg(tmp_path: Path):
    """Verify Newton-CG converges with and without bound transforms."""
    os.chdir(tmp_path)

    x0 = np.array([-1.2, -1.0])
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    expected = np.array([1.0, 1.0])

    for transform in (False, True):
        res = LineSearch.minimize(
            x0,
            fun=rosen,
            jac=rosen_der,
            hess=rosen_hess,
            method="Newton-CG",
            bounds=bounds,
            transform=transform,
        )
        np.testing.assert_allclose(res.x, expected, atol=1e-4)


def test_line_search_restart(tmp_path: Path):
    """Verify restart save and resume behavior for BFGS line search."""
    restart_path = tmp_path / "line_search_restart.pkl"
    interrupt_iteration = 4
    x0 = np.array([-1.2, -1.0])

    def stop_after_checkpoint(opt):
        if opt.iteration == interrupt_iteration:
            raise RuntimeError("Intentional stop after checkpoint")

    with pytest.raises(RuntimeError, match="Intentional stop after checkpoint"):
        LineSearch.minimize(
            x0,
            fun=rosen,
            jac=rosen_der,
            method="BFGS",
            callback=stop_after_checkpoint,
            restartsave=True,
            restart_file=restart_path,
        )
    assert restart_path.exists(), "Expected callback to save a restart file."

    def verify_resume_progress(opt):
        assert opt.iteration >= interrupt_iteration, (
            "Expected resumed optimization to continue after the interrupted iteration."
        )

    # Resume the optimization from the saved restart file and verify it continues correctly.
    resumed = LineSearch.minimize(
        x0,
        fun=rosen,
        jac=rosen_der,
        method="BFGS",
        callback=verify_resume_progress,
        restart=True,
        restart_file=restart_path,
    )

    # Compare the resumed optimization result with a fresh optimization run to ensure they match.
    reference = LineSearch.minimize(
        x0,
        fun=rosen,
        jac=rosen_der,
        method="BFGS",
    )

    assert resumed.message == reference.message
    for key in ["x", "fun", "nfev", "njev", "nit"]:
        assert np.allclose(resumed[key], reference[key]), (
            f"Mismatch in {key} between resumed and reference optimization."
        )
