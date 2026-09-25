"""A crashed simulation costs the trial point, not the optimization run.

The optimizer proposes control vectors, some of which the simulator cannot run. Ending
the run on the first of them throws away every iteration that came before it. Reporting
`inf` instead lets backtracking reject the point and carry on from the last good one.
"""

from types import SimpleNamespace

import numpy as np

from popt.ensembles.ensemble_base import EnsembleOptimizationBase

NX, NE = 3, 4


def _host(*, sim_success, ne=NE):
    """The attributes `function` reads, and nothing else."""
    logged = []
    return SimpleNamespace(
        ne=ne,
        num_models=1,
        num_samples=ne,
        aux_input=None,
        idX={"x": (0, NX)},
        save_prediction=None,
        sim=SimpleNamespace(input_dict={}, true_order=None),
        sim_data=None,
        logger=SimpleNamespace(error=logged.append, info=logged.append),
        calc_prediction=lambda x, save_prediction=None: sim_success,
        obj_func=lambda *a, **k: np.arange(ne, dtype=float),
        _aux_input=lambda: 1,
        _reorganize_multilevel_ensemble=lambda x: x,
        stateF=np.array([7.0]),
        enF=None,
    ), logged


def test_a_crashed_ensemble_evaluation_reports_inf_instead_of_raising():
    host, logged = _host(sim_success=False)

    values = EnsembleOptimizationBase.function(host, np.zeros((NX, NE)))

    assert np.all(np.isinf(values))
    assert any("reject it" in m for m in logged)


def test_a_crashed_single_point_leaves_the_current_objective_alone():
    """`gradient` computes `enF - repeat(stateF, nr)`, so writing inf into stateF
    would poison every later gradient rather than just rejecting this point."""
    host, _ = _host(sim_success=False)
    before = host.stateF.copy()

    values = EnsembleOptimizationBase.function(host, np.zeros(NX))

    assert np.all(np.isinf(values))
    np.testing.assert_array_equal(host.stateF, before)


def test_a_successful_evaluation_still_updates_the_state_objective():
    host, _ = _host(sim_success=True)

    values = EnsembleOptimizationBase.function(host, np.zeros(NX))

    assert np.all(np.isfinite(values))
    np.testing.assert_array_equal(host.stateF, values)


def test_a_successful_ensemble_evaluation_still_updates_the_ensemble_objective():
    host, _ = _host(sim_success=True)

    values = EnsembleOptimizationBase.function(host, np.zeros((NX, NE)))

    np.testing.assert_array_equal(host.enF, values)
    np.testing.assert_array_equal(host.stateF, np.array([7.0]))   # untouched


# --------------------------------------------------------------------------
# End to end: the optimizer rejects the point and keeps going
# --------------------------------------------------------------------------

def test_the_optimizer_backtracks_past_a_crashed_trial_point(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from popt.optimization_methods.optimizer_base import OptimizerBase, StepReport

    class _Descent(OptimizerBase):
        NAME = "descent"

        def update_step(self) -> StepReport:
            for shrink in (1.0, 0.5, 0.25):
                x_new = self.xk - shrink * 0.5 * self.jk
                f_new = self.fun(x_new)
                if np.mean(f_new) < np.mean(self.fk):
                    self._commit_step(x_new, f_new, jac=self.jac(x_new))
                    return StepReport(True)
            return StepReport(False, "no improving step")

    crashed = []

    def objective(x, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        # The full step from (2, 2) lands on (0, 0), which the "simulator" cannot run;
        # backtracking halves it to (1, 1), which it can.
        if np.allclose(x, np.array([0.0, 0.0]), atol=1e-9):
            crashed.append(tuple(x))
            return np.inf
        return float(np.sum(x ** 2))

    result = _Descent.minimize(
        np.array([2.0, 2.0]), objective, jac=lambda x: 2.0 * np.asarray(x, dtype=float),
        logit=False, maxiter=3, xtol=1e-12, ftol=1e-12,
    )

    assert crashed, "the crashing point was never proposed; the test proves nothing"
    assert np.all(np.isfinite(result.x))
    assert np.mean(result.fun) < np.sum(np.array([2.0, 2.0]) ** 2)
    assert result.nit >= 1
