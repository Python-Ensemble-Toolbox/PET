"""
Tests for Gaussian ensemble.
"""
import os
import numpy as np
from scipy.optimize import rosen, rosen_der
from popt.ensembles import GaussianEnsemble, GeneralizedEnsemble


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
X0 = np.array([5.0, 5.0])
NE = 10
CFG = {
    "ne": NE,
    "natural_gradient": False,
    "controls": {
        "x": {
            "mean": X0.tolist(),
            "var": 1.0e-5,
            "limits": [-2, 2],
        }
    },
}

def rosen_function_vectorized(x):
    return np.apply_along_axis(rosen, axis=0, arr=x)


def test_gaussian_ensemble_gradient(tmp_path):
    """
    Test the gradient estimation of the Gaussian ensemble.
    """
    os.chdir(tmp_path)

    # =============================================================
    # Compute ensmble gradient
    # =============================================================
    np.random.seed(42)
    ensemble = GaussianEnsemble(
        CFG,
        simulator = None,
        objective = rosen_function_vectorized
    )
    x0 = ensemble.get_state()
    f0 = ensemble.function(x0)
    cov = ensemble.get_cov()
    grad_ensemble = ensemble.gradient(x0, cov)
    # =============================================================

    # =============================================================
    # Compute ensmble gradient manually for comparison
    # =============================================================
    np.random.seed(42)
    enX = np.random.multivariate_normal(x0, cov, NE).T
    enX = enX - enX.mean(axis=1, keepdims=True) + x0[:, None]
    enX = np.clip(enX, -2, 2)
    enF = ensemble.function(enX)
    dF  = enF - f0
    dx  = enX - x0[:, None]
    grad_expected = np.linalg.solve(cov, dx @ dF / NE)
    # =============================================================

    np.testing.assert_array_equal(grad_ensemble, grad_expected)


def test_gaussian_ensemble_hessian(tmp_path):
    """
    Test the Hessian estimation of the Gaussian ensemble.
    """
    os.chdir(tmp_path)

    # =============================================================
    # Compute ensemble Hessian
    # =============================================================
    np.random.seed(42)
    ensemble = GaussianEnsemble(
        CFG,
        simulator = None,
        objective = rosen_function_vectorized
    )
    x0 = ensemble.get_state()
    f0 = ensemble.function(x0)
    # The return value is unused, but the call is required: hessian() below
    # reuses the ensemble (self.enF) that gradient() populates, and also
    # advances the global RNG that the manual comparison re-seeds against.
    # Deleting this line makes hessian() raise TypeError on self.enF.
    ensemble.gradient(x0, ensemble.get_cov())
    cov = ensemble.get_cov()
    hess_ensemble = ensemble.hessian(x0, cov)
    # =============================================================

    # =============================================================
    # Compute ensemble Hessian manually for comparison
    # =============================================================
    np.random.seed(42)
    enX = np.random.multivariate_normal(x0, cov, NE).T
    enX = enX - enX.mean(axis=1, keepdims=True) + x0[:, None]
    enX = np.clip(enX, -2, 2)
    enF = ensemble.function(enX)
    dF  = enF - f0
    dX  = enX - x0[:, None]
    hess_expected = (dX * dF) @ dX.T / NE - cov * np.mean(dF)
    hess_expected = np.linalg.solve(
        cov,
        np.linalg.solve(cov, hess_expected).T
    ).T
    # =============================================================

    np.testing.assert_array_equal(hess_ensemble, hess_expected)


def test_gaussian_ensemble_gradient_convergence(tmp_path):
    os.chdir(tmp_path)

    ne = 100_000
    cfg = {
        "ne": ne,
        "natural_gradient": False,
        "controls": {
            "x": {
                "mean": [-1.0, -1.0],
                "var": 1.0e-5,
                "limits": [-2, 2],
            }
        },
    }

    # =============================================================
    # Compute ensemble Gradient
    # =============================================================
    np.random.seed(42)
    ensemble = GaussianEnsemble(
        cfg,
        simulator = None,
        objective = rosen_function_vectorized
    )
    x0 = ensemble.get_state()
    cov = ensemble.get_cov()
    ensemble.function(x0)
    grad_ensemble = ensemble.gradient(x0, cov)
    # ============================================================

    # ============================================================
    # Compute true average gradient for comparison
    # ============================================================
    np.random.seed(42)
    enX = np.random.multivariate_normal(x0, cov, ne).T
    enX = enX - enX.mean(axis=1, keepdims=True) + x0[:, None]
    enX = np.clip(enX, -2, 2)

    grad_expected = np.mean(
        np.apply_along_axis(rosen_der, axis=0, arr=enX),
        axis=1
    )
    # ===========================================================

    np.testing.assert_allclose(
        grad_ensemble,
        grad_expected,
        rtol=1e-2,
        atol=1e-6,
    )


def test_generalized_ensemble_gradient_convergence(tmp_path):
    os.chdir(tmp_path)

    ne = 100_000
    cfg = {
        "ne": ne,
        #"marginal": "TruncGaussian",
        "controls": {
            "x": {
                "mean": [-1.0, -1.0],
                "var": 1.0e-5,
                "limits": [-2, 2],
            }
        },
    }

    # =============================================================
    # Compute ensemble Gradient
    # =============================================================
    np.random.seed(42)
    ensemble = GeneralizedEnsemble(
        cfg,
        simulator = None,
        objective = rosen_function_vectorized
    )
    x0 = ensemble.get_state()
    corr = ensemble.get_corr()
    theta = ensemble.get_theta()
    ensemble.function(x0)
    grad_ensemble = ensemble.gradient(x0, theta, corr)
    # ============================================================

    # ============================================================
    # Compute true average gradient for comparison
    # ============================================================
    np.random.seed(42)
    enX, _ = ensemble.sample(ne)
    enX = np.clip(enX.T, -2, 2)

    grad_expected = np.mean(
        np.apply_along_axis(rosen_der, axis=0, arr=enX),
        axis=1
    )
    # ===========================================================

    np.testing.assert_allclose(
        grad_ensemble,
        grad_expected,
        rtol=1e-2,
        atol=1e-6,
    )








