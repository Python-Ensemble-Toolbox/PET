"""Regression tests for verified bugs in popt's numerical subroutines."""

import numpy as np
import pytest

from popt.misc_tools import optim_tools as ot
from popt.optimization_methods.subroutines.optimizers import Steihaug
from popt.optimization_methods.subroutines.subroutines import newton_cg


def test_steihaug_tau_lands_exactly_on_the_trust_region_boundary():
    """Only the square root used to be divided by ||d||^2."""
    rng = np.random.default_rng(0)
    rule = Steihaug(delta0=1.0)
    rule.delta = 1.0
    for _ in range(5):
        p = rng.standard_normal(4) * 0.3          # inside the region
        d = rng.standard_normal(4)
        tau = rule.get_tau(p, d)
        assert tau > 0
        assert np.linalg.norm(p + tau * d) == pytest.approx(1.0, rel=1e-12)


def test_newton_cg_returns_a_direction_when_it_runs_out_of_iterations():
    """It used to fall off the loop and return None."""
    H = np.diag([1.0, 10.0, 100.0])              # needs three CG steps
    g = np.array([1.0, 1.0, 1.0])
    d = newton_cg(g, H, maxiter=1, logger=lambda *a: None)
    assert isinstance(d, np.ndarray) and d.shape == g.shape
    assert np.dot(d, g) < 0                       # still a descent direction


def test_newton_cg_with_no_iterations_falls_back_to_steepest_descent():
    g = np.array([1.0, -2.0])
    np.testing.assert_array_equal(newton_cg(g, np.eye(2), maxiter=0, logger=lambda *a: None), -g)


@pytest.mark.parametrize(
    "bounds, expected",
    [
        ([(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0]),          # zero bounds used to be treated as no bounds
        ([(None, 1.0), (-1.0, None)], [1.0, -1.0]),      # None means open on that side
        ([(None, None), (None, None)], [5.0, -5.0]),     # fully open: unchanged
        ([(-2.0, 2.0), (-2.0, 2.0)], [2.0, -2.0]),
    ],
)
def test_clip_state_respects_every_kind_of_bound(bounds, expected):
    np.testing.assert_array_equal(ot.clip_state(np.array([5.0, -5.0]), bounds), expected)


# ----------------------------------------------------------------------
# Backtracking factor: every step rule takes it, and EnOpt passes it through.
# ----------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from scipy.optimize import rosen, rosen_der  # noqa: E402

from popt.optimization_methods import EnOpt, LineSearch  # noqa: E402
from popt.optimization_methods.subroutines.optimizers import Adam, GradientDescent  # noqa: E402


@pytest.mark.parametrize("make, attr", [
    (lambda: Adam(0.1, 0.0), "_step_size"),
    (lambda: Steihaug(delta0=3.0), "delta"),
    (lambda: GradientDescent(0.1, 0.0), "_step_size"),
])
def test_every_step_rule_scales_by_the_backtracking_factor(make, attr):
    rule = make()
    before = getattr(rule, attr)
    rule.apply_backtracking(1.0)
    assert getattr(rule, attr) == before                  # factor 1 is a no-op ...
    rule.apply_backtracking(0.25)
    assert getattr(rule, attr) == pytest.approx(0.25 * before)   # ... and the factor is honoured


def test_enopt_no_longer_halves_adam_before_the_first_attempt():
    host = SimpleNamespace(optimizer=Adam(0.1, 0.0))
    EnOpt._apply_optimizer_backtracking(host, 1.0)
    assert host.optimizer._step_size == 0.1


def test_line_search_iteration_cap_reaches_the_line_search():
    """`lsmaxiter` was stored under a key the subroutines never read."""
    ls = LineSearch(np.array([-1.2, 1.0]), rosen, jac=rosen_der, lsmaxiter=3, maxiter=1)
    assert ls.line_search_options["maxiter"] == 3
    assert "lsmaxiter" not in ls.line_search_options
