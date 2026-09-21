"""The outer EPF loop converges on the constraint violation, not on the step size.

Measuring the relative change in the controls answered the wrong question: the loop
declared success whenever the inner optimizer stalled, however badly the constraints
were still violated, and refused to finish while one control kept jittering.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from popt.optimization_methods.optimizer_base import OptimizerBase, StepReport


def _host(penalty, **epf):
    """The attributes `check_epf_convergence` reads, and nothing else."""
    log = []
    options = {'r': 2.0, 'r_factor': 2.0, 'tol_factor': 0.9}
    options.update(epf)
    if penalty is not None:
        options['penalty'] = penalty
    return SimpleNamespace(
        epf=options, epf_iteration=1, epf_maxiter=10, ftol=1e-4, logger=log.append
    ), log


def test_a_satisfied_constraint_ends_the_loop():
    host, log = _host(np.array([1e-6, 1e-6]), conv_crit=1e-3)   # mean/r = 5e-7

    assert OptimizerBase.check_epf_convergence(host) is True
    assert any('penalty term smaller than' in m for m in log)


def test_a_violated_constraint_tightens_the_penalty_and_continues():
    host, log = _host(np.array([4.0, 6.0]), conv_crit=1e-3)     # mean/r = 2.5

    assert OptimizerBase.check_epf_convergence(host) is False
    assert host.epf['r'] == 4.0                                  # r doubled
    assert host.ftol == pytest.approx(9e-5)                      # tolerance tightened


def test_a_stalled_but_infeasible_point_is_not_convergence():
    """The old criterion read `|xk - xk_old| / |xk_old|`, so an inner loop that stopped
    moving reported success at a point that never satisfied the constraints."""
    host, _ = _host(np.array([100.0]), conv_crit=1e-5)
    host.xk = host.xk_old = np.array([1.0, 2.0])                 # nothing moved at all

    assert OptimizerBase.check_epf_convergence(host) is False


def test_the_maximum_outer_iteration_count_still_wins():
    host, log = _host(np.array([100.0]), conv_crit=1e-5)
    host.epf_iteration = host.epf_maxiter

    assert OptimizerBase.check_epf_convergence(host) is True
    assert any('Maximum number of outer EPF iterations' in m for m in log)


def test_an_objective_that_never_writes_a_penalty_is_an_error():
    host, _ = _host(None, conv_crit=1e-3)

    with pytest.raises(KeyError, match="must write it"):
        OptimizerBase.check_epf_convergence(host)


def test_an_empty_penalty_is_an_error():
    host, _ = _host(np.array([]), conv_crit=1e-3)

    with pytest.raises(ValueError, match='penalty is empty'):
        OptimizerBase.check_epf_convergence(host)


def test_conv_crit_defaults_when_the_config_omits_it():
    """1e-5 was chosen for a dimensionless relative state change and is kept, so a
    config written for the old criterion still runs. It now means an absolute penalty
    magnitude, which is why CHANGELOG records the changed meaning."""
    host, _ = _host(np.array([1e-9]))                             # mean/r = 5e-10 < 1e-5

    assert OptimizerBase.check_epf_convergence(host) is True


# --------------------------------------------------------------------------
# End to end: the outer loop runs exactly max_epf_iter times
# --------------------------------------------------------------------------

class _FixedStep(OptimizerBase):
    NAME = "fixed step"

    def update_step(self) -> StepReport:
        x_new = self.xk - 0.25 * self.jk
        self._commit_step(x_new, self.fun(x_new), jac=self.jac(x_new))
        return StepReport(True)


def test_the_outer_loop_runs_exactly_max_epf_iter_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    penalty_factors = []

    def objective(x, **kwargs):
        epf = kwargs['epf']
        epf['penalty'] = np.array([1e3])          # never satisfied, so the loop runs out
        penalty_factors.append(epf['r'])
        return float(np.sum(np.asarray(x) ** 2))

    _FixedStep.minimize(
        np.array([2.0, -1.0]), objective, jac=lambda x: 2.0 * np.asarray(x, dtype=float),
        logit=False, maxiter=2,
        epf={'r': 1.0, 'r_factor': 2.0, 'tol_factor': 0.9, 'conv_crit': 1e-5, 'max_epf_iter': 3},
    )

    assert sorted(set(penalty_factors)) == [1.0, 2.0, 4.0]        # three outer passes, r doubling
