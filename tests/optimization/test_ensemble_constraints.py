"""Constraints returned with the objective, and their ensemble Jacobian and Lagrangian Hessian."""

import numpy as np
import pytest
from scipy.optimize import rosen

from popt.ensembles import GaussianEnsemble
from popt.optimization_methods import EnOpt

NX = 3
A_LIN = np.array([1.0, -2.0, 0.5])


def _config(ne=10, natural=True, seed=7, var=1e-3):
    return {"ne": ne, "natural_gradient": natural, "seed": seed,
            "controls": {"x": {"mean": [0.5, -0.3, 1.2], "var": var, "limits": [-2, 2]}}}


def rosen_vectorized(x):
    return np.apply_along_axis(rosen, axis=0, arr=x)


def constrained(x, **kwargs):
    """Rosenbrock with one linear equality and two inequalities (one linear, one quadratic)."""
    return {
        "f": rosen_vectorized(x),
        "c_eq": A_LIN @ x - 1.0,                                    # 1-D: a single constraint
        "c_iq": np.vstack([x[0] - x[2] + 3.0, 4.0 - np.sum(x**2, axis=0)]),
    }


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_the_objective_values_are_returned_as_before_and_the_constraints_are_kept():
    ens = GaussianEnsemble(_config(), None, constrained)
    x = ens.get_state()

    f = ens.function(x)
    np.testing.assert_allclose(f, rosen(x))
    assert ens.stateC["c_eq"].shape == (1, 1) and ens.stateC["c_iq"].shape == (2, 1)

    enX = x[:, None] + 0.01 * np.arange(4)
    f = ens.function(enX)
    assert f.shape == (4,) and ens.enC["c_eq"].shape == (1, 4) and ens.enC["c_iq"].shape == (2, 4)

    values = ens.function_and_constraints(x)
    np.testing.assert_allclose(values["c_eq"][:, 0], A_LIN @ x - 1.0)
    np.testing.assert_allclose(values["c_iq"][:, 0], [x[0] - x[2] + 3.0, 4.0 - x @ x])


def test_constraints_of_the_wrong_shape_or_unknown_keys_are_refused():
    ens = GaussianEnsemble(_config(), None, lambda x, **kw: {"f": rosen_vectorized(x), "c_iq": np.zeros((2, 5))})
    with pytest.raises(ValueError, match="one column per evaluated control vector"):
        ens.function(ens.get_state())
    ens = GaussianEnsemble(_config(), None, lambda x, **kw: {"f": rosen_vectorized(x), "c": np.zeros((1, 1))})
    with pytest.raises(KeyError, match="Unknown keys"):
        ens.function(ens.get_state())


@pytest.mark.parametrize("natural, gradient, hessian", [
    (False, [46.98055305930163, 20.767874278819143, 171.0384042703416],
     [[2084.8469837219973, 2487.7629945058657, -2575.570315709927],
      [2487.7629945058657, -1241.1924320564835, -2362.5244201607466],
      [-2575.570315709927, -2362.5244201607466, -224.46553592990585]]),
    (True, [0.04698055305930163, 0.02076787427881914, 0.17103840427034162],
     [[0.002084846983722, 0.00248776299450587, -0.00257557031570993],
      [0.00248776299450587, -0.00124119243205648, -0.00236252442016075],
      [-0.00257557031570993, -0.00236252442016075, -0.00022446553592991]]),
])
def test_gradient_and_hessian_of_an_unconstrained_objective_are_what_they_were(natural, gradient, hessian):
    """Values from the code before the per-level loops were shared with the constraint Jacobian."""
    np.random.seed(42)
    config = _config(natural=natural)
    del config["seed"]
    ens = GaussianEnsemble(config, None, rosen_vectorized)
    x, cov = ens.get_state(), ens.get_cov()
    ens.function(x)

    np.testing.assert_allclose(ens.gradient(x, cov), gradient, rtol=1e-12)
    np.testing.assert_allclose(ens.hessian(x, cov), hessian, rtol=1e-12)


def test_the_jacobian_is_the_gradient_estimate_applied_to_each_constraint():
    ens = GaussianEnsemble(_config(), None, constrained)
    x, cov = ens.get_state(), ens.get_cov()
    ens.function(x)
    ens.gradient(x, cov)
    jac = ens.constraint_jacobian(x, cov)
    assert jac["c_eq"].shape == (1, NX) and jac["c_iq"].shape == (2, NX)

    # The same members, compared with the same reference, through `gradient` on each constraint alone
    for key, row, single in [("c_eq", 0, lambda x: A_LIN @ x - 1.0), ("c_iq", 1, lambda x: 4.0 - np.sum(x**2, axis=0))]:
        other = GaussianEnsemble(_config(), None, lambda x, **kw: single(x))
        other.function(x)
        np.testing.assert_allclose(jac[key][row], other.gradient(x, cov), rtol=1e-10)


def test_divided_by_the_sample_covariance_the_jacobian_of_a_linear_constraint_is_exact():
    ens = GaussianEnsemble(_config(), None, constrained)
    x, cov = ens.get_state(), ens.get_cov()
    ens.function(x)
    ens.gradient(x, cov)
    jac = ens.constraint_jacobian()

    exact = np.linalg.solve(ens.sample_covariance(), jac["c_eq"][0])
    np.testing.assert_allclose(exact, A_LIN, rtol=1e-8)
    # Divided by the covariance the members were drawn from, it carries their sampling error
    assert np.max(np.abs(np.linalg.solve(cov, jac["c_eq"][0]) - A_LIN)) > 1e-3


def test_without_natural_gradient_the_jacobian_is_divided_by_the_covariance():
    x = np.array([0.5, -0.3, 1.2])
    jacobians = {}
    for natural in (True, False):
        ens = GaussianEnsemble(_config(natural=natural), None, constrained)
        cov = ens.get_cov()
        ens.function(x)
        ens.gradient(x, cov)
        jacobians[natural] = ens.constraint_jacobian(x, cov)
    for key in ("c_eq", "c_iq"):
        np.testing.assert_allclose(jacobians[False][key], np.linalg.solve(cov, jacobians[True][key].T).T, rtol=1e-10)


def test_the_jacobian_approaches_the_analytic_one_with_many_members():
    ens = GaussianEnsemble(_config(ne=20000, natural=False), None, constrained)
    x, cov = ens.get_state(), ens.get_cov()
    ens.function(x)
    ens.gradient(x, cov)
    jac = ens.constraint_jacobian(x, cov)
    np.testing.assert_allclose(jac["c_eq"][0], A_LIN, atol=0.1)
    np.testing.assert_allclose(jac["c_iq"], [[1.0, 0.0, -1.0], -2.0 * x], atol=0.1)


def test_the_lagrangian_hessian_is_the_hessian_of_the_lagrangian():
    lam = {"c_eq": np.array([0.7]), "c_iq": np.array([0.0, 1.3])}

    def lagrangian(x, **kwargs):
        v = constrained(x)
        return v["f"] - lam["c_eq"][0] * v["c_eq"] - lam["c_iq"] @ v["c_iq"]

    hessians = []
    for objective in (constrained, lagrangian):
        ens = GaussianEnsemble(_config(), None, objective)
        x, cov = ens.get_state(), ens.get_cov()
        ens.function(x)
        ens.gradient(x, cov)
        if objective is constrained:
            hessians.append(ens.lagrangian_hessian(x, cov, multipliers=lam, detrend=False))
            # No multipliers: the Hessian of the objective
            np.testing.assert_allclose(ens.lagrangian_hessian(x, cov, detrend=False), ens.hessian(x, cov), rtol=1e-12)
        else:
            hessians.append(ens.hessian(x, cov))
    np.testing.assert_allclose(hessians[0], hessians[1], rtol=1e-8, atol=1e-14)


def test_removing_the_linear_trend_makes_the_hessian_far_less_noisy():
    """f = 5 a.x + |x|^2 has Hessian 2 I; with small perturbations the trend's sampling error swamps it."""
    def objective(x, **kwargs):
        return 5.0 * (A_LIN @ x) + np.sum(x**2, axis=0)

    errors = {}
    for detrend in (True, False):
        ens = GaussianEnsemble(_config(ne=2000, natural=False, var=1e-4), None, objective)
        x, cov = ens.get_state(), ens.get_cov()
        ens.function(x)
        ens.gradient(x, cov)
        errors[detrend] = np.linalg.norm(ens.lagrangian_hessian(x, cov, detrend=detrend) - 2.0 * np.eye(NX))
    assert errors[True] < 0.5 and errors[True] < 0.05 * errors[False]


def test_a_member_with_a_failed_constraint_is_left_out_of_every_estimate():
    def objective(x, **kwargs):
        values = constrained(x)
        if x.shape[1] > 1:
            values["c_iq"][1, 2] = np.nan
        return values

    ens = GaussianEnsemble(_config(), None, objective)
    x, cov = ens.get_state(), ens.get_cov()
    ens.function(x)
    gradient = ens.gradient(x, cov)
    jac = ens.constraint_jacobian(x, cov)

    assert np.all(np.isfinite(gradient)) and all(np.all(np.isfinite(j)) for j in jac.values())
    keep = np.arange(10) != 2
    dx = ens.enX[:, keep] - x[:, None]
    np.testing.assert_allclose(gradient, dx @ (ens.enF[0][keep] - rosen(x)) / 9, rtol=1e-10)


def test_the_reference_handed_in_replaces_the_latest_single_evaluation():
    """After a rejected trial point `stateF`/`stateC` are that point's; the optimizer passes its own values."""
    ens = GaussianEnsemble(_config(), None, constrained)
    x, cov = ens.get_state(), ens.get_cov()
    at_x = ens.function_and_constraints(x)
    ens.function(x + 0.3)                       # a trial point, rejected
    ens.gradient(x, cov, reference=at_x)

    np.testing.assert_allclose(ens.stateF, rosen(x))
    np.testing.assert_allclose(ens.stateC["c_eq"][:, 0], A_LIN @ x - 1.0)


def test_an_optimizer_that_knows_nothing_of_constraints_runs_on_such_an_objective():
    ens = GaussianEnsemble(_config(natural=True), None, constrained)
    res = EnOpt.minimize(ens.get_state(), ens.function, jac=ens.gradient, args=(ens.get_cov(),),
                         bounds=ens.get_bounds(), transform=True, maxiter=3, logit=False)
    assert np.mean(res.fun) < rosen(np.array([0.5, -0.3, 1.2]))
