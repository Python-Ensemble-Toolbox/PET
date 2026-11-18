"""``tol`` is EnOpt's absolute improvement test; ``ftol`` the base loop's relative stop.

``tol`` used to override an explicit ``ftol``, so the relative stop could not be relaxed or
switched off without also changing the step-acceptance threshold. An explicit ``ftol`` now wins,
and ``ftol = 0`` reproduces the old ``popt.loop`` behaviour of stopping only when no improving
step is found (or at ``maxiter``).
"""

import numpy as np
import pytest

from popt.optimization_methods import EnOpt, GenOpt, SmcOpt

X0 = np.array([0.5, 0.5])
BOUNDS = [(0.0, 1.0), (0.0, 1.0)]
COV = 0.01 * np.eye(2)
QUIET = dict(restart=False, restartsave=False, saveit=False, logit=False)


def _fun(x, *args, **kwargs):
    return np.atleast_1d(np.sum(np.asarray(x) ** 2))


def _jac(x, cov, *args, **kwargs):
    return cov @ (2 * np.asarray(x))


def _enopt(**options):
    return EnOpt(X0, _fun, jac=_jac, args=(COV,), bounds=BOUNDS, **QUIET, **options)


def test_tol_alone_sets_both_tolerances():
    opt = _enopt(tol=1e-3)
    assert opt.obj_func_tol == 1e-3
    assert opt.ftol == 1e-3


def test_an_explicit_ftol_wins_over_tol_for_the_relative_stop_only():
    opt = _enopt(tol=1e-3, ftol=1e-7)
    assert opt.obj_func_tol == 1e-3
    assert opt.ftol == 1e-7


def test_ftol_zero_disables_the_relative_function_stop():
    strict = _enopt(tol=1e-3)
    relaxed = _enopt(tol=1e-3, ftol=0.0)
    for opt in (strict, relaxed):
        opt.fk_old = np.array([-5.893])
        opt.fk = np.array([-5.898])          # improvement 0.005 > tol, but < tol * |f_prev|
    assert strict.check_function_convergence()
    assert not relaxed.check_function_convergence()


def test_ftol_zero_runs_to_maxiter_while_steps_keep_improving():
    res = EnOpt.minimize(X0, _fun, jac=_jac, args=(COV,), bounds=BOUNDS, **QUIET,
                         tol=1e-12, ftol=0.0, maxiter=4, alpha=0.1, resample=0)
    assert res.nit == 4
    assert "Maximum number of iterations" in res.message


@pytest.mark.parametrize("cls", [GenOpt, SmcOpt])
def test_the_other_ensemble_optimizers_follow_the_same_precedence(cls):
    kwargs = dict(x0=X0, fun=_fun, bounds=BOUNDS, tol=1e-3, ftol=1e-7, **QUIET)
    if cls is GenOpt:   # args = (theta, corr): marginal parameter and correlation matrix
        opt = cls(jac=_jac, jac_mut=_jac, args=(np.zeros(2), np.eye(2)), **kwargs)
    else:
        opt = cls(sens=_jac, args=(COV,), **kwargs)
    assert opt.obj_func_tol == 1e-3
    assert opt.ftol == 1e-7
