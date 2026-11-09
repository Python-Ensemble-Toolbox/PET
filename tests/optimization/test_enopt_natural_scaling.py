"""The ensemble gradient scales like a state perturbation under the unit-cube transform.

EnOpt's gradient is Cov(F, x), a covariance-preconditioned gradient. In unit-cube coordinates
u = (x - lb) / db that is Cov(F, u) = Cov(F, x) / db, while a derivative transforms the other way,
dF/du = dF/dx * db. The optimizer applied the derivative rule to the ensemble gradient, so the
unit-cube components came out a factor db^2 off and controls with wide bounds dominated the step.
"""

import numpy as np

from popt.optimization_methods import BoundTransformHandler, EnOpt

LB = np.array([0.0, 0.0])
DB = np.array([10.0, 1000.0])          # very different ranges
BOUNDS = [(lo, lo + d) for lo, d in zip(LB, DB)]
SIGMA_U = 0.05                         # perturbation std in the unit cube
ALPHA = 0.3


def test_the_handler_divides_an_ensemble_gradient_by_the_range_and_a_derivative_is_multiplied():
    h = BoundTransformHandler(BOUNDS, transform=True)
    g = np.array([2.0, 3.0])
    np.testing.assert_allclose(h.natural_jac_to_unit_cube(g), g / DB)
    np.testing.assert_allclose(h.jac_to_unit_cube(g), g * DB)
    np.testing.assert_allclose(h.natural_jac_from_unit_cube(h.natural_jac_to_unit_cube(g)), g)
    H = np.outer(g, g)
    np.testing.assert_allclose(h.natural_hess_to_unit_cube(H), H / np.outer(DB, DB))
    np.testing.assert_allclose(h.natural_hess_from_unit_cube(h.natural_hess_to_unit_cube(H)), H)
    assert h.natural_jac_to_unit_cube(None) is None
    h_off = BoundTransformHandler(BOUNDS, transform=False)
    np.testing.assert_allclose(h_off.natural_jac_to_unit_cube(g), g)


def _linear_problem(natural):
    """F decreases with the same slope per unit-cube coordinate in both controls.

    The exact ensemble gradient of a linear F is cov_phys @ dF/dx; a true gradient is dF/dx."""
    slope_u = -np.ones(2)                       # dF/du
    slope_x = slope_u / DB                      # dF/dx
    cov_phys = np.diag((SIGMA_U * DB) ** 2)

    def fun(x, *args, **kwargs):
        return np.atleast_1d(slope_x @ np.asarray(x))

    def jac(x, cov, *args, **kwargs):
        return cov @ slope_x if natural else slope_x

    return fun, jac, cov_phys


def _one_step(natural, **extra):
    fun, jac, cov = _linear_problem(natural)
    x0 = LB + 0.5 * DB
    res = EnOpt.minimize(x0=x0, fun=fun, jac=jac, args=(cov,), bounds=BOUNDS,
                         transform=True, maxiter=1, alpha=ALPHA, tol=1e-12, resample=0,
                         restart=False, restartsave=False, saveit=False, logit=False, **extra)
    return (np.asarray(res.x) - x0) / DB        # the step in unit-cube coordinates


def test_a_natural_gradient_moves_every_control_by_alpha_regardless_of_its_range():
    step_u = _one_step(natural=True, natural_gradient=True)
    np.testing.assert_allclose(step_u, [ALPHA, ALPHA], rtol=1e-6)


def test_a_true_gradient_gives_the_same_step_with_the_derivative_rule():
    step_u = _one_step(natural=False, natural_gradient=False)
    np.testing.assert_allclose(step_u, [ALPHA, ALPHA], rtol=1e-6)


def test_the_derivative_rule_on_a_natural_gradient_is_what_went_wrong():
    """Kept as the record of the defect: the wide-range control takes the whole step."""
    step_u = _one_step(natural=True, natural_gradient=False)
    assert step_u[1] > 0.99 * ALPHA
    assert step_u[0] < ALPHA * (DB[0] / DB[1]) ** 2 * 1.01


def test_the_setting_is_read_from_the_ensemble_the_gradient_is_bound_to():
    class Ensemble:
        def __init__(self, natural):
            self.keys_en = {'natural_gradient': natural}

        def gradient(self, x, cov, *args, **kwargs):
            return np.zeros(2)

    assert EnOpt._ensemble_uses_natural_gradient(Ensemble(False).gradient) is False
    assert EnOpt._ensemble_uses_natural_gradient(Ensemble(True).gradient) is True
    assert EnOpt._ensemble_uses_natural_gradient(lambda x, cov: x) is True     # no ensemble: default
