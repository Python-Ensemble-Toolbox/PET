import numpy as np
import pytest
import os
from scipy.optimize import rosen, rosen_der, rosen_hess
from pathlib import Path

from popt.optimization_methods import TrustRegion


def test_trust_region_iterative(tmp_path: Path):
    """Verify iterative trust-region converges with and without bound transforms."""
    os.chdir(tmp_path)

    x0 = np.array([-1.2, -1.0])
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    expected = np.array([1.0, 1.0])

    for transform in (False, True):
        res = TrustRegion.minimize(
            x0,
            fun=rosen,
            jac=rosen_der,
            hess=rosen_hess,
            method="iterative",
            bounds=bounds,
            transform=transform,
        )
        np.testing.assert_allclose(res.x, expected, atol=1e-4)


def test_trust_region_cg_steihaug(tmp_path: Path):
    """Verify CG-Steihaug trust-region converges with and without bound transforms."""
    os.chdir(tmp_path)

    x0 = np.array([-1.2, -1.0])
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    expected = np.array([1.0, 1.0])

    for transform in (False, True):
        res = TrustRegion.minimize(
            x0,
            fun=rosen,
            jac=rosen_der,
            hess=rosen_hess,
            method="CG-Steihaug",
            bounds=bounds,
            transform=transform,
        )
        np.testing.assert_allclose(res.x, expected, atol=1e-4)

def test_trust_region_bfgs(tmp_path: Path):
    """Verify BFGS trust-region converges with and without bound transforms."""
    os.chdir(tmp_path)

    x0 = np.array([-1.2, -1.0])
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    expected = np.array([1.0, 1.0])

    for method in ("iterative", "CG-Steihaug"):
        for transform in (False, True):
            res = TrustRegion.minimize(
                x0,
                fun=rosen,
                jac=rosen_der,
                hess="BFGS",
                method=method,
                bounds=bounds,
                transform=transform,
            )
            np.testing.assert_allclose(res.x, expected, atol=1e-4)


def test_trust_region_restart(tmp_path: Path):
    """Verify restart save and resume behavior for BFGS trust-region."""
    restart_path = tmp_path / "trust_region_restart.pkl"
    interrupt_iteration = 4
    x0 = np.array([-1.2, -1.0])

    def stop_after_checkpoint(opt):
        if opt.iteration == interrupt_iteration:
            raise RuntimeError("Intentional stop after checkpoint")

    with pytest.raises(RuntimeError, match="Intentional stop after checkpoint"):
        TrustRegion.minimize(
            x0,
            fun=rosen,
            jac=rosen_der,
            hess="BFGS",
            method="iterative",
            callback=stop_after_checkpoint,
            restartsave=True,
            restart_file=restart_path,
        )
    assert restart_path.exists(), "Expected callback to save a restart file."

    def verify_resume_progress(opt):
        assert opt.iteration >= interrupt_iteration, (
            "Expected resumed optimization to continue after the interrupted iteration."
        )

    resumed = TrustRegion.minimize(
        x0,
        fun=rosen,
        jac=rosen_der,
        hess="BFGS",
        method="iterative",
        callback=verify_resume_progress,
        restart=True,
        restart_file=restart_path,
    )

    reference = TrustRegion.minimize(
        x0,
        fun=rosen,
        jac=rosen_der,
        hess="BFGS",
        method="iterative",
    )

    assert resumed.message == reference.message
    for key in ["x", "fun", "nfev", "njev", "nit"]:
        assert np.allclose(resumed[key], reference[key]), (
            f"Mismatch in {key} between resumed and reference optimization."
        )
