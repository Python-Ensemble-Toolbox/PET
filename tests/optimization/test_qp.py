"""The QP solver behind EnSQP, against scipy on random strictly convex problems."""

import numpy as np
import pytest
from scipy.optimize import minimize

from popt.optimization_methods.subroutines.qp import solve_qp


def _random_qp(rng, n, m_eq, m_iq, bounded):
    M = rng.standard_normal((n, n))
    B = M @ M.T + 0.1 * np.eye(n)
    g = rng.standard_normal(n)
    # Constraints that hold at a known point inside the bounds, so the problem is feasible
    d_feasible = rng.uniform(-0.5, 0.5, n)
    A_eq = rng.standard_normal((m_eq, n))
    b_eq = -A_eq @ d_feasible
    A_iq = rng.standard_normal((m_iq, n))
    b_iq = -A_iq @ d_feasible + rng.uniform(0.0, 1.0, m_iq) * rng.integers(0, 2, m_iq)
    lb, ub = (-np.ones(n), np.ones(n)) if bounded else (None, None)
    return B, g, A_eq, b_eq, A_iq, b_iq, lb, ub


def _scipy_solution(B, g, A_eq, b_eq, A_iq, b_iq, lb, ub):
    constraints = []
    if b_eq.size:
        constraints.append({'type': 'eq', 'fun': lambda d: A_eq @ d + b_eq, 'jac': lambda d: A_eq})
    if b_iq.size:
        constraints.append({'type': 'ineq', 'fun': lambda d: A_iq @ d + b_iq, 'jac': lambda d: A_iq})
    bounds = None if lb is None else list(zip(lb, ub))
    res = minimize(lambda d: g @ d + 0.5 * d @ B @ d, np.zeros(g.size), jac=lambda d: g + B @ d,
                   constraints=constraints, bounds=bounds, method='SLSQP', options={'ftol': 1e-14, 'maxiter': 500})
    assert res.success
    return res.x


@pytest.mark.parametrize('n, m_eq, m_iq, bounded', [
    (4, 0, 0, False), (4, 0, 0, True), (5, 2, 0, False), (5, 0, 4, False),
    (6, 2, 5, True), (8, 3, 10, True), (3, 0, 6, True), (10, 4, 4, False),
])
def test_the_solution_is_scipys_and_satisfies_the_kkt_conditions(n, m_eq, m_iq, bounded):
    rng = np.random.default_rng(100 * n + 10 * m_eq + m_iq)
    for _ in range(10):
        B, g, A_eq, b_eq, A_iq, b_iq, lb, ub = _random_qp(rng, n, m_eq, m_iq, bounded)

        d, lam, info = solve_qp(B, g, A_eq, b_eq, A_iq, b_iq, lb, ub)

        assert info['converged'] and info['slack'] == 0.0
        np.testing.assert_allclose(d, _scipy_solution(B, g, A_eq, b_eq, A_iq, b_iq, lb, ub), atol=1e-5)

        # Stationarity, feasibility, sign and complementarity
        stationarity = g + B @ d - A_eq.T @ lam['c_eq'] - A_iq.T @ lam['c_iq'] - lam['lb'] + lam['ub']
        np.testing.assert_allclose(stationarity, 0.0, atol=1e-7)
        np.testing.assert_allclose(A_eq @ d + b_eq, 0.0, atol=1e-8)
        assert np.all(A_iq @ d + b_iq >= -1e-8)
        assert np.all(lam['c_iq'] >= 0) and np.all(lam['lb'] >= 0) and np.all(lam['ub'] >= 0)
        np.testing.assert_allclose(lam['c_iq'] * (A_iq @ d + b_iq), 0.0, atol=1e-7)


def test_without_constraints_it_is_the_newton_step():
    B = np.array([[2.0, 0.5], [0.5, 1.0]])
    g = np.array([1.0, -3.0])
    d, lam, info = solve_qp(B, g)
    np.testing.assert_allclose(d, -np.linalg.solve(B, g), atol=1e-10)
    assert lam['c_eq'].size == 0 and lam['c_iq'].size == 0 and info['slack'] == 0.0


def test_constraints_that_cannot_hold_still_give_a_step_and_report_the_slack():
    """d_0 >= 1 and d_0 <= -1 at once: the step settles between them and says so."""
    B, g = np.eye(2), np.zeros(2)
    A_iq = np.array([[1.0, 0.0], [-1.0, 0.0]])
    b_iq = np.array([-1.0, -1.0])

    d, lam, info = solve_qp(B, g, A_iq=A_iq, b_iq=b_iq)

    assert np.all(np.isfinite(d)) and abs(d[0]) <= 1.0 + 1e-8
    assert info['slack'] == pytest.approx(2.0, abs=1e-3)       # the total violation cannot go below 2


def test_a_violated_constraint_is_restored_in_one_step_when_it_can_be():
    """Start violating d_0 + d_1 >= 2 (b = -2): the step lands on it, the nearest point for B = I."""
    d, lam, info = solve_qp(np.eye(2), np.zeros(2), A_iq=np.array([[1.0, 1.0]]), b_iq=np.array([-2.0]))
    np.testing.assert_allclose(d, [1.0, 1.0], atol=1e-8)
    assert lam['c_iq'][0] == pytest.approx(1.0, abs=1e-7) and info['slack'] == 0.0
