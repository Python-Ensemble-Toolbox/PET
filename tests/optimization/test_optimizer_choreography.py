"""The base owns everything around a step; an optimizer is `update_step` plus `log_columns`."""

import numpy as np
import pytest
from scipy.optimize import rosen, rosen_der

from popt.optimization_methods import EnOpt, LineSearch
from popt.optimization_methods.optimizer_base import OptimizerBase, StepReport


def quadratic(x):
    return float(np.sum((np.asarray(x) - 1.0) ** 2))


def quadratic_jac(x):
    return 2.0 * (np.asarray(x, dtype=float) - 1.0)


class FixedStepDescent(OptimizerBase):
    """The smallest optimizer the contract allows."""

    NAME = "Fixed-step descent"

    def update_step(self) -> StepReport:
        x_new = self.xk - 0.25 * self.jk
        f_new = self.fun(x_new)
        if f_new >= np.mean(self.fk):
            return StepReport(False, "no descent along the gradient")
        self._commit_step(x_new, f_new, jac=self.jac(x_new))
        return StepReport(True)


def test_an_optimizer_is_a_step_and_the_base_does_the_rest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = []
    res = FixedStepDescent.minimize(np.array([4.0, -2.0]), quadratic, jac=quadratic_jac,
                                    callback=lambda opt: seen.append(opt.iteration), logit=False, xtol=1e-12, ftol=1e-12)

    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-4)
    assert res.message.startswith("Projected gradient norm")      # the base's gradient check, no override needed
    assert seen == list(range(1, res.nit + 1))                     # callback once per accepted step
    assert res.nfev == res.nit + 1 and res.njev == res.nit + 1     # start evaluation + one per step


def test_a_rejected_step_stops_the_run_with_its_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Stuck(OptimizerBase):
        def update_step(self):
            return StepReport(False, "nothing works here")

    calls = []
    res = Stuck.minimize(np.array([0.0]), quadratic, jac=quadratic_jac, callback=lambda opt: calls.append(1), logit=False)
    assert res.message == "nothing works here"
    assert res.nit == 0 and calls == []


def test_commit_step_keeps_the_previous_iterate_for_the_convergence_checks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FixedStepDescent(np.array([3.0]), quadratic, jac=quadratic_jac, logit=False)
    assert opt.fk is None                                          # nothing is evaluated until the run starts
    opt._start()
    opt._commit_step(np.array([2.0]), 1.0, jac=np.array([2.0]))
    assert opt.xk_old == np.array([3.0]) and opt.fk_old == 4.0
    assert opt.xk == np.array([2.0]) and opt.fk == 1.0 and opt.jk == np.array([2.0])


def test_minimize_and_the_constructor_take_the_same_arguments_in_the_same_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x0, cov = np.array([2.0]), np.eye(1) * 1e-3
    kwargs = dict(bounds=[(-5, 5)], transform=True, maxiter=15, alpha=0.3, logit=False)

    via_minimize = EnOpt.minimize(x0, quadratic, quadratic_jac, args=(cov,), **kwargs)
    built = EnOpt(x0, quadratic, quadratic_jac, args=(cov,), **kwargs)
    built.run_optimization()

    np.testing.assert_array_equal(built.optimize_results.x, via_minimize.x)
    assert built.optimize_results.nit == via_minimize.nit


def test_line_search_recomputes_the_gradient_and_retries(tmp_path, monkeypatch):
    """`recompute_jac` cleared the gradient and then took `-None` as the next direction."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def flaky_der(x):
        calls.append(1)
        return -rosen_der(x) if len(calls) == 1 else rosen_der(x)   # first call: an ascent direction

    res = LineSearch.minimize(np.array([-1.2, 1.0]), rosen, jac=flaky_der, method="GD", lsmethod=0,
                              lsmaxiter=5, recompute_jac=1, maxiter=3, logit=False)
    assert res.nit >= 1
    assert len(calls) >= 3                                          # the bad one, the recomputed one, the accepted step's


@pytest.mark.parametrize("cls", [FixedStepDescent])
def test_log_columns_default_names_the_iteration_and_objective(cls, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = cls(np.array([3.0]), quadratic, jac=quadratic_jac, logit=False)
    opt._start()
    assert opt.log_columns() == {"iter.": 0, "fun": 4.0}
