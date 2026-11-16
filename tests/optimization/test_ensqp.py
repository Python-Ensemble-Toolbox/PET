"""EnSQP: sequential quadratic programming with derivatives from an ensemble."""

import numpy as np
import pytest
from scipy.optimize import minimize

from popt.ensembles import GaussianEnsemble
from popt.optimization_methods import EnSQP

TARGET = np.array([2.0, 1.0])


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------
# Exact derivatives: the SQP iteration itself
# --------------------------------------------------------------------------
def _objective(x):
    return float(np.sum((x - TARGET) ** 2))


def _gradient(x):
    return 2.0 * (np.asarray(x, dtype=float) - TARGET)


def _problem(kind):
    """fun, con_jac, solution, multipliers (c_eq, c_iq) for min |x - (2, 1)|^2 under one constraint."""
    if kind == "equality":            # x0 + x1 = 1
        return (lambda x: {"f": _objective(x), "c_eq": [x[0] + x[1] - 1.0]},
                lambda x: {"c_eq": np.array([[1.0, 1.0]])}, [1.0, 0.0], ([-2.0], []))
    if kind == "active inequality":   # x0 + x1 <= 1
        return (lambda x: {"f": _objective(x), "c_iq": [1.0 - x[0] - x[1]]},
                lambda x: {"c_iq": np.array([[-1.0, -1.0]])}, [1.0, 0.0], ([], [2.0]))
    if kind == "inactive inequality":  # x0 + x1 <= 10
        return (lambda x: {"f": _objective(x), "c_iq": [10.0 - x[0] - x[1]]},
                lambda x: {"c_iq": np.array([[-1.0, -1.0]])}, [2.0, 1.0], ([], [0.0]))
    if kind == "nonlinear":           # x0 + x1 = 2 and |x|^2 <= 2.5, both active at (1.5, 0.5)
        return (lambda x: {"f": _objective(x), "c_eq": [x[0] + x[1] - 2.0], "c_iq": [2.5 - x @ x]},
                lambda x: {"c_eq": np.array([[1.0, 1.0]]), "c_iq": -2.0 * x[None, :]}, [1.5, 0.5], None)
    raise ValueError(kind)


@pytest.mark.parametrize("transform", [False, True])
@pytest.mark.parametrize("kind", ["equality", "active inequality", "inactive inequality", "nonlinear"])
def test_exact_derivatives_reach_the_constrained_minimum(kind, transform):
    fun, con_jac, solution, multipliers = _problem(kind)

    res = EnSQP.minimize(np.array([0.0, 0.0]), fun, jac=_gradient, con_jac=con_jac,
                         bounds=[(-3, 3)] * 2, transform=transform, logit=False)

    np.testing.assert_allclose(res.x, solution, atol=1e-5)
    assert res.violation <= 1e-6
    assert "constraint violation" in res.message       # stopped on the KKT test
    if multipliers is not None:
        for name, expected in zip(("lam_eq", "lam_iq"), multipliers):
            if expected:
                np.testing.assert_allclose(res[name], expected, atol=1e-4)


@pytest.mark.parametrize("transform", [False, True])
def test_an_active_bound_is_handled_by_the_qp(transform):
    """No constraints but the bounds: the minimum is on the upper bound of x0."""
    res = EnSQP.minimize(np.array([0.0, 0.0]), _objective, jac=_gradient,
                         bounds=[(-1.0, 1.5)] * 2, transform=transform, logit=False)
    np.testing.assert_allclose(res.x, [1.5, 1.0], atol=1e-5)
    assert res.violation == 0.0


def test_an_infeasible_start_on_a_nonlinear_equality():
    """min x0 + x1 on the circle |x|^2 = 2, from a point off it: solution (-1, -1), multiplier -1/2."""
    res = EnSQP.minimize(np.array([2.0, 0.5]), lambda x: {"f": x[0] + x[1], "c_eq": [x @ x - 2.0]},
                         jac=lambda x: np.ones(2), con_jac=lambda x: {"c_eq": 2.0 * x[None, :]},
                         bounds=[(-3, 3)] * 2, logit=False, maxiter=50)
    np.testing.assert_allclose(res.x, [-1.0, -1.0], atol=1e-4)
    np.testing.assert_allclose(res.lam_eq, [-0.5], atol=1e-3)
    assert res.violation <= 1e-6


def test_a_small_change_does_not_end_the_run_at_an_infeasible_point():
    """ftol = 1 would stop any other optimizer after one step; here only once the constraint holds."""
    fun, con_jac, solution, _ = _problem("nonlinear")
    res = EnSQP.minimize(np.array([0.0, 0.0]), fun, jac=_gradient, con_jac=con_jac, bounds=[(-3, 3)] * 2,
                         max_step=0.3, ftol=1.0, logit=False)
    assert res.violation <= 1e-6


def test_the_options_that_cannot_work_are_refused():
    fun, con_jac, _, _ = _problem("equality")
    x0 = np.zeros(2)
    with pytest.raises(ValueError, match="epf"):
        EnSQP(x0, fun, jac=_gradient, con_jac=con_jac, epf={"r": 1.0}, logit=False)
    with pytest.raises(ValueError, match="hessian='ensemble' needs"):
        EnSQP(x0, fun, jac=_gradient, con_jac=con_jac, hessian="ensemble", logit=False)
    with pytest.raises(ValueError, match="not recognized"):
        EnSQP(x0, fun, jac=_gradient, con_jac=con_jac, hessian="newton", logit=False)
    with pytest.raises(ValueError, match="con_jac"):
        EnSQP.minimize(x0, fun, jac=_gradient, logit=False)
    with pytest.raises(ValueError, match="gradient"):
        EnSQP(x0, fun, logit=False)


def test_restart_continues_to_the_same_result(tmp_path):
    fun, con_jac, _, _ = _problem("nonlinear")
    restart_path = tmp_path / "ensqp_restart.pkl"
    kwargs = dict(jac=_gradient, con_jac=con_jac, bounds=[(-3, 3)] * 2, max_step=0.4, logit=False)
    x0 = np.array([0.0, 0.0])

    def stop(opt):
        if opt.iteration == 2:
            raise RuntimeError("stopped")

    with pytest.raises(RuntimeError, match="stopped"):
        EnSQP.minimize(x0, fun, callback=stop, restartsave=True, restart_file=restart_path, **kwargs)
    assert restart_path.exists()

    resumed = EnSQP.minimize(x0, fun, restart=True, restart_file=restart_path, **kwargs)
    reference = EnSQP.minimize(x0, fun, **kwargs)

    assert resumed.message == reference.message and resumed.nit == reference.nit
    np.testing.assert_allclose(resumed.x, reference.x, atol=1e-10)
    np.testing.assert_allclose(resumed.lam_eq, reference.lam_eq, atol=1e-8)


# --------------------------------------------------------------------------
# Derivatives from an ensemble
# --------------------------------------------------------------------------
NX = 6
WEIGHT = np.linspace(0.5, 2.0, NX)
CENTRE = np.linspace(0.2, 0.8, NX)


def _parts(x):
    f = np.sum(WEIGHT[:, None] * (x - CENTRE[:, None]) ** 2, axis=0)
    c_eq = np.sum(x, axis=0) - 0.4 * NX                              # linear equality
    c_iq = np.vstack([0.05 - np.sum((x[:3] - 0.1) ** 2, axis=0),      # nonlinear, active at the solution
                      x[4] - x[5] + 0.5])                            # linear, inactive
    return f, c_eq, c_iq


def ensemble_objective(x, **kwargs):
    f, c_eq, c_iq = _parts(x)
    return {"f": f, "c_eq": c_eq, "c_iq": c_iq}


def _reference_solution(nx=NX):
    def one(x):
        return _parts(np.asarray(x)[:, None])

    res = minimize(lambda x: one(x)[0][0], np.full(nx, 0.5), bounds=[(0, 1)] * nx, method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda x: one(x)[1]},
                                {"type": "ineq", "fun": lambda x: one(x)[2][:, 0]}], options={"ftol": 1e-12})
    assert res.success
    return res


def _run(ne, seed, natural=True, **options):
    config = {"ne": ne, "natural_gradient": natural, "seed": seed,
              "controls": {"x": {"mean": [0.5] * NX, "var": 1e-4, "limits": [0, 1]}}}
    ens = GaussianEnsemble(config, None, ensemble_objective)
    return EnSQP.minimize(ens.get_state(), ens.function_and_constraints, jac=ens.gradient,
                          con_jac=ens.constraint_jacobian, hess=ens.lagrangian_hessian, args=(ens.get_cov(),),
                          bounds=ens.get_bounds(), transform=True, logit=False, **options)


@pytest.mark.parametrize("natural", [True, False])
@pytest.mark.parametrize("hessian", ["bfgs", "ensemble"])
def test_ensemble_derivatives_reach_the_constrained_minimum(hessian, natural):
    reference = _reference_solution()
    assert _parts(reference.x[:, None])[2][0, 0] == pytest.approx(0.0, abs=1e-6)     # the nonlinear one is active
    for seed in range(4):
        res = _run(ne=30, seed=seed, natural=natural, hessian=hessian)
        assert res.violation <= 1e-3                  # the nonlinear constraint, to the noise of the ensemble
        assert abs(np.mean(res.fun) - reference.fun) < 5e-3
        np.testing.assert_allclose(res.x, reference.x, atol=0.05)


def test_natural_and_averaged_ensemble_derivatives_give_the_same_run():
    """The optimizer divides a natural gradient by the covariance itself, so the two settings only differ in round-off."""
    natural, averaged = _run(ne=30, seed=0, natural=True), _run(ne=30, seed=0, natural=False)
    np.testing.assert_allclose(natural.x, averaged.x, atol=1e-6)


def test_with_fewer_members_than_controls_the_step_stays_in_their_subspace_and_ends_feasible():
    reference = _reference_solution()
    start = _parts(np.full((NX, 1), 0.5))[0][0]
    assert start - reference.fun > 0.2
    for seed in range(8):
        res = _run(ne=4, seed=seed, maxiter=60)
        assert res.violation <= 1e-3
        assert abs(np.mean(res.fun) - reference.fun) < 0.05


def test_the_sample_covariance_scaling_meets_a_linear_constraint_exactly_and_the_drawn_one_does_not():
    """Dividing by the drawn covariance leaves the members' sampling error in the Jacobian of even a linear constraint."""
    sample = [abs(_run(ne=10, seed=seed, scaling="sample").c_eq[0]) for seed in range(4)]
    drawn = [abs(_run(ne=10, seed=seed, scaling="cov").c_eq[0]) for seed in range(4)]
    assert max(sample) <= 1e-9
    assert max(drawn) > 1e-6
    with pytest.raises(ValueError, match="scaling"):
        _run(ne=10, seed=0, scaling="other")


def test_the_result_carries_constraints_multipliers_and_jacobian():
    res = _run(ne=30, seed=1)
    assert res.c_eq.shape == (1,) and res.c_iq.shape == (2,)
    assert res.lam_eq.shape == (1,) and res.lam_iq.shape == (2,)
    assert res.jac_c_eq.shape == (1, NX) and res.jac_c_iq.shape == (2, NX)
    np.testing.assert_allclose(res.jac_c_eq, 1.0, atol=1e-6)      # linear constraint: exact with the sample scaling
    assert res.lam_iq[0] > 0 and res.lam_iq[1] == pytest.approx(0.0, abs=1e-6)
    assert res.njev_con == res.njev


def test_robust_optimization_averages_objective_and_constraints_over_the_models():
    """Two models whose constraint differs by +-0.2: the mean constraint is the one enforced."""
    def objective(x, **kwargs):
        offset = np.where(np.arange(x.shape[1]) < x.shape[1] / 2, 0.2, -0.2)     # members are grouped by model
        return {"f": np.sum((x - 0.8) ** 2, axis=0), "c_iq": 0.9 - x[0] - x[1] + offset}

    config = {"ne": 20, "num_models": 2, "seed": 3,
              "controls": {"x": {"mean": [0.3, 0.3], "var": 1e-4, "limits": [0, 1]}}}
    ens = GaussianEnsemble(config, None, objective)
    res = EnSQP.minimize(ens.get_state(), ens.function_and_constraints, jac=ens.gradient,
                         con_jac=ens.constraint_jacobian, args=(ens.get_cov(),), bounds=ens.get_bounds(),
                         transform=True, logit=False)
    np.testing.assert_allclose(res.x, [0.45, 0.45], atol=0.02)
    assert res.violation <= 1e-5
