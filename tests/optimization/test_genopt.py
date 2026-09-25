"""GenOpt: the sampling distribution moves along with the controls.

EnOpt draws from a fixed Gaussian; GenOpt draws from the generalized ensemble's
marginals and advances `theta` and the correlation matrix as well, so an accepted
step has to change three things, not one.
"""

import numpy as np
import pytest
from scipy.optimize import rosen

from popt import CMA, GenOpt
from popt.ensembles import GeneralizedEnsemble

X0 = np.array([-1.0, -1.0])
NE = 60


def _rosen_vectorized(x, *args, **kwargs):
    return np.apply_along_axis(rosen, axis=0, arr=x)


def _ensemble(seed=42, ne=NE):
    np.random.seed(seed)
    cfg = {
        "ne": ne,
        "controls": {"x": {"mean": X0.tolist(), "var": 1.0e-2, "limits": [-2, 2]}},
    }
    return GeneralizedEnsemble(cfg, simulator=None, objective=_rosen_vectorized)


def _run(corr_adapt=None, *, maxiter=3, seed=42, **options):
    ensemble = _ensemble(seed)
    x0 = ensemble.get_state()
    ensemble.function(x0)
    return ensemble, GenOpt.minimize(
        x0, ensemble.function,
        jac=ensemble.gradient, jac_mut=ensemble.mutation_gradient,
        args=(ensemble.get_theta(), ensemble.get_corr()),
        corr_adapt=corr_adapt, bounds=[(-2, 2)] * X0.size,
        logit=False, maxiter=maxiter, **options,
    )


# --------------------------------------------------------------------------
# It runs, and it optimizes
# --------------------------------------------------------------------------

def test_genopt_reduces_the_objective(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensemble, result = _run()

    assert np.mean(result.fun) < rosen(X0)
    assert result.nit >= 1


def test_the_distribution_parameter_moves(tmp_path, monkeypatch):
    """theta follows its own gradient on every accepted step; if it never moves the
    method has silently degenerated into EnOpt with a fixed non-Gaussian sampler."""
    monkeypatch.chdir(tmp_path)
    ensemble = _ensemble()
    theta0 = np.array(ensemble.get_theta(), dtype=float)
    x0 = ensemble.get_state()
    ensemble.function(x0)

    optimizer = GenOpt(
        x0, ensemble.function, jac=ensemble.gradient, jac_mut=ensemble.mutation_gradient,
        args=(ensemble.get_theta(), ensemble.get_corr()),
        bounds=[(-2, 2)] * X0.size, logit=False, maxiter=3,
    )
    optimizer.run_optimization()

    assert not np.allclose(optimizer.theta, theta0)


# --------------------------------------------------------------------------
# Correlation adaptation
# --------------------------------------------------------------------------

def test_cma_adapts_the_correlation_matrix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensemble = _ensemble()
    corr0 = np.array(ensemble.get_corr(), dtype=float)
    x0 = ensemble.get_state()
    ensemble.function(x0)

    optimizer = GenOpt(
        x0, ensemble.function, jac=ensemble.gradient, jac_mut=ensemble.mutation_gradient,
        args=(ensemble.get_theta(), ensemble.get_corr()),
        corr_adapt=CMA(ne=NE, dim=X0.size, corr_update=True),
        bounds=[(-2, 2)] * X0.size, logit=False, maxiter=3,
    )
    optimizer.run_optimization()

    assert optimizer.corr.shape == corr0.shape
    assert not np.allclose(optimizer.corr, corr0)


def test_a_plain_callable_corr_adapt_is_descended_along(tmp_path, monkeypatch):
    """Anything callable works, not just CMA: its result is a descent direction for
    the correlation, scaled by `alpha_corr`."""
    monkeypatch.chdir(tmp_path)
    direction = np.array([[0.0, 1.0], [1.0, 0.0]])

    ensemble = _ensemble()
    corr0 = np.array(ensemble.get_corr(), dtype=float)
    x0 = ensemble.get_state()
    ensemble.function(x0)

    optimizer = GenOpt(
        x0, ensemble.function, jac=ensemble.gradient, jac_mut=ensemble.mutation_gradient,
        args=(ensemble.get_theta(), ensemble.get_corr()),
        corr_adapt=lambda: direction, alpha_corr=0.25,
        bounds=[(-2, 2)] * X0.size, logit=False, maxiter=1,
    )
    optimizer.run_optimization()

    steps = np.round((corr0 - optimizer.corr) / 0.25, 9)
    assert np.allclose(steps % 1, 0)                    # a whole number of alpha_corr steps
    assert not np.allclose(optimizer.corr, corr0)


def test_no_corr_adapt_leaves_the_correlation_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensemble = _ensemble()
    corr0 = np.array(ensemble.get_corr(), dtype=float)
    x0 = ensemble.get_state()
    ensemble.function(x0)

    optimizer = GenOpt(
        x0, ensemble.function, jac=ensemble.gradient, jac_mut=ensemble.mutation_gradient,
        args=(ensemble.get_theta(), ensemble.get_corr()),
        bounds=[(-2, 2)] * X0.size, logit=False, maxiter=2,
    )
    optimizer.run_optimization()

    np.testing.assert_array_equal(optimizer.corr, corr0)


# --------------------------------------------------------------------------
# The ensemble is drawn once, not twice
# --------------------------------------------------------------------------

def test_the_mutation_gradient_can_return_its_ensemble():
    """CMA needs the Gaussian samples and their objective values. Asking for them in
    the same call is what keeps GenOpt from simulating a second ensemble per step."""
    ensemble = _ensemble()
    x0 = ensemble.get_state()
    ensemble.function(x0)

    grad, matrices = ensemble.mutation_gradient(
        x0, ensemble.get_theta(), ensemble.get_corr(), return_ensembles=True
    )

    assert set(matrices) == {"gaussian", "objective"}
    assert matrices["gaussian"].shape[0] == NE
    assert matrices["objective"].shape[0] == NE
    np.testing.assert_array_equal(grad, ensemble.nat_grad)


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("missing, message", [
    ("jac", "requires a Jacobian"),
    ("jac_mut", "requires a jac_mut"),
])
def test_both_gradients_are_required(missing, message):
    kwargs = {"jac": lambda *a, **k: np.zeros(2), "jac_mut": lambda *a, **k: np.zeros(2)}
    kwargs.pop(missing)

    with pytest.raises(ValueError, match=message):
        GenOpt(X0, _rosen_vectorized, args=(np.ones((2, 2)), np.eye(2)), **kwargs)


def test_theta_and_corr_are_required():
    with pytest.raises(ValueError, match=r"args = \(theta, corr\)"):
        GenOpt(X0, _rosen_vectorized, jac=lambda *a, **k: np.zeros(2),
               jac_mut=lambda *a, **k: np.zeros(2), args=())


def test_an_unknown_optimizer_is_refused():
    with pytest.raises(ValueError, match="not recognized for GenOpt"):
        GenOpt(X0, _rosen_vectorized, jac=lambda *a, **k: np.zeros(2),
               jac_mut=lambda *a, **k: np.zeros(2),
               args=(np.ones((2, 2)), np.eye(2)), optimizer="Steihaug")
