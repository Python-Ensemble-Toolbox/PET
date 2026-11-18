"""A small dense solver for the strictly convex quadratic programs of an SQP method."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

__all__ = ['solve_qp']


def solve_qp(B, g, A_eq=None, b_eq=None, A_iq=None, b_iq=None, lb=None, ub=None,
             rho=None, rho_max=None, tol=1e-9, maxiter=None):
    r"""
    Solve the quadratic program of an SQP step,

    $$ \min_d \; g^T d + \tfrac{1}{2} d^T B d \quad \text{s.t.} \quad
       A_{eq} d + b_{eq} = 0, \quad A_{iq} d + b_{iq} \geq 0, \quad lb \leq d \leq ub. $$

    The linear constraints are relaxed with non-negative slacks that cost ``rho`` each (an
    exact penalty), which does two things. A feasible starting point always exists
    (``d = 0`` with the slacks taking up the violation), so a primal active-set method needs
    no first phase; and constraints that cannot all hold, which happens for the
    linearisation of a nonlinear problem far from the solution, still give a step that
    reduces their violation. ``rho`` is raised until the slacks vanish or stop shrinking.

    Parameters
    ----------
    B : ndarray, shape (n, n)
        Symmetric positive definite Hessian.
    g : ndarray, shape (n,)
        Gradient.
    A_eq, b_eq : ndarray, shape (m_eq, n) and (m_eq,), optional
        Equality constraints.
    A_iq, b_iq : ndarray, shape (m_iq, n) and (m_iq,), optional
        Inequality constraints.
    lb, ub : ndarray, shape (n,), optional
        Bounds on ``d``; infinite entries are free. ``lb <= 0 <= ub`` is expected.
    rho : float, optional
        First penalty on the slacks. Default: ten times the largest gradient element, at least 1.
    rho_max : float, optional
        Largest penalty tried. Default: ``1e8 * rho``.
    tol : float, optional
        Tolerance on steps, multipliers and slacks.
    maxiter : int, optional
        Active-set iterations per solve. Default: 50 times the number of variables and constraints.

    Returns
    -------
    d : ndarray, shape (n,)
        The step.
    multipliers : dict
        ``'c_eq'``, ``'c_iq'``, ``'lb'`` and ``'ub'``: multipliers of the equalities (either
        sign) and of the inequalities and bounds (non-negative), in the convention
        ``g + B d = A_eq^T lam_eq + A_iq^T lam_iq + lam_lb - lam_ub``.
    info : dict
        ``'slack'``: what is left of the linear constraints' violation (zero when they can
        hold), ``'rho'``: the penalty used, ``'iterations'``, and ``'converged'``.
    """
    g = np.asarray(g, dtype=float)
    n = g.size
    B = np.asarray(B, dtype=float).reshape(n, n)
    A_eq, b_eq = _linear_part(A_eq, b_eq, n)
    A_iq, b_iq = _linear_part(A_iq, b_iq, n)
    m_eq, m_iq = b_eq.size, b_iq.size
    lb = np.full(n, -np.inf) if lb is None else np.minimum(np.asarray(lb, dtype=float), 0.0)
    ub = np.full(n, np.inf) if ub is None else np.maximum(np.asarray(ub, dtype=float), 0.0)

    # Variables: z = [d, s_plus, s_minus, s_iq], the slacks relaxing
    #   A_eq d + b_eq + s_plus - s_minus = 0   and   A_iq d + b_iq + s_iq >= 0.
    n_slack = 2 * m_eq + m_iq
    N = n + n_slack
    curvature = max(np.trace(B) / n, np.finfo(float).tiny)
    H = np.zeros((N, N))
    H[:n, :n] = 0.5 * (B + B.T)
    # A little curvature on the slacks keeps H positive definite. The penalty stays exact:
    # its slope at zero slack is still rho.
    H[n:, n:] = 1e-3 * curvature * np.eye(n_slack)

    E = np.hstack([A_eq, np.eye(m_eq), -np.eye(m_eq), np.zeros((m_eq, m_iq))])
    e = -b_eq

    has_lb, has_ub = np.isfinite(lb), np.isfinite(ub)
    identity = np.eye(N)
    G = np.vstack([
        np.hstack([A_iq, np.zeros((m_iq, 2 * m_eq)), np.eye(m_iq)]),   # relaxed inequalities
        identity[n:],                                                  # slacks >= 0
        identity[:n][has_lb],                                          # d >= lb
        -identity[:n][has_ub],                                         # -d >= -ub
    ])
    h = np.concatenate([-b_iq, np.zeros(n_slack), lb[has_lb], -ub[has_ub]])

    # Feasible start: no step, slacks equal to the violation.
    z = np.zeros(N)
    z[n:n + m_eq] = np.maximum(-b_eq, 0.0)
    z[n + m_eq:n + 2 * m_eq] = np.maximum(b_eq, 0.0)
    z[n + 2 * m_eq:] = np.maximum(-b_iq, 0.0)

    if rho is None:
        rho = 10.0 * max(1.0, np.linalg.norm(g, np.inf))
    if rho_max is None:
        rho_max = 1e8 * rho
    if maxiter is None:
        maxiter = 50 * (N + h.size)
    slack_tol = tol * (1.0 + np.linalg.norm(np.concatenate([b_eq, b_iq, [0.0]]), np.inf))

    iterations = 0
    previous_slack = np.inf
    while True:
        q = np.concatenate([g, np.full(n_slack, rho)])
        z, lam_E, lam_G, converged, used = _active_set(H, q, E, e, G, h, z, tol, maxiter)
        iterations += used
        slack = float(np.sum(z[n:]))
        # Slacks that are gone, that a ten times larger penalty did not shrink (the
        # constraints cannot all hold), or no room to raise the penalty: done.
        if n_slack == 0 or slack <= slack_tol or slack > 0.99 * previous_slack or 10.0 * rho > rho_max:
            break
        previous_slack = slack
        rho *= 10.0

    lam_lb, lam_ub = np.zeros(n), np.zeros(n)
    lam_lb[has_lb] = lam_G[m_iq + n_slack:m_iq + n_slack + has_lb.sum()]
    lam_ub[has_ub] = lam_G[m_iq + n_slack + has_lb.sum():]
    multipliers = {'c_eq': lam_E, 'c_iq': lam_G[:m_iq], 'lb': lam_lb, 'ub': lam_ub}
    info = {'slack': slack if slack > slack_tol else 0.0, 'rho': rho, 'iterations': iterations, 'converged': converged}
    return np.clip(z[:n], lb, ub), multipliers, info


def _linear_part(A, b, n):
    """``A`` as an (m, n) array and ``b`` as an (m,) array; no constraints gives m = 0."""
    if A is None or np.size(A) == 0:
        return np.zeros((0, n)), np.zeros(0)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if A.shape != (b.size, n):
        raise ValueError(f'Constraint matrix has shape {A.shape}; expected ({b.size}, {n}).')
    return A, b


def _active_set(H, q, E, e, G, h, z, tol, maxiter):
    """Primal active-set method for ``min 1/2 z'Hz + q'z`` s.t. ``E z = e``, ``G z >= h`` from a feasible ``z``.

    ``H`` is positive definite, so each equality-constrained subproblem is solved through the
    Schur complement ``A H^-1 A'`` of the working constraints ``A``. Returns the solution, the
    multipliers of ``E`` and of ``G`` (zero for inactive rows), whether it converged, and the
    number of iterations.
    """
    m_E, m_G = E.shape[0], G.shape[0]
    factor = cho_factor(H)
    z = z.copy()

    # Start from the constraints active at z, leaving out any that depend linearly on those
    # already taken: the multipliers of a dependent set are not unique, and their signs
    # decide which constraint is released.
    working = np.zeros(m_G, dtype=bool)
    rank = np.linalg.matrix_rank(E) if m_E else 0
    residual = G @ z - h
    for i in np.flatnonzero(residual <= tol * (1.0 + np.abs(h))):
        candidate = working.copy()
        candidate[i] = True
        new_rank = np.linalg.matrix_rank(np.vstack([E, G[candidate]]))
        if new_rank > rank:
            working, rank = candidate, new_rank

    lam = np.zeros(m_E)
    rows = np.zeros(0, dtype=int)
    converged = False
    iteration = 0
    for iteration in range(1, maxiter + 1):
        gradient = H @ z + q
        rows = np.flatnonzero(working)
        A = np.vstack([E, G[rows]])
        Hinv_grad = cho_solve(factor, gradient)
        if A.shape[0]:
            schur = A @ cho_solve(factor, A.T)
            lam = np.linalg.lstsq(schur, A @ Hinv_grad, rcond=None)[0]
            p = -(Hinv_grad - cho_solve(factor, A.T @ lam))
        else:
            lam = np.zeros(0)
            p = -Hinv_grad

        if np.linalg.norm(p, np.inf) <= tol * (1.0 + np.linalg.norm(z, np.inf)):
            # Stationary on the working set: done unless an inequality wants to leave it.
            lam_working = lam[m_E:]
            if lam_working.size == 0 or lam_working.min() >= -tol * (1.0 + np.linalg.norm(gradient, np.inf)):
                converged = True
                break
            working[rows[np.argmin(lam_working)]] = False
            continue

        # Longest step along p that keeps the inactive inequalities satisfied.
        Gp = G @ p
        residual = np.maximum(G @ z - h, 0.0)
        blocking = (~working) & (Gp < -1e-14 * (1.0 + np.abs(G).sum(axis=1) * np.linalg.norm(p, np.inf)))
        alpha, hit = 1.0, None
        if blocking.any():
            ratios = np.full(m_G, np.inf)
            ratios[blocking] = residual[blocking] / (-Gp[blocking])
            hit = int(np.argmin(ratios))
            if ratios[hit] < 1.0:
                alpha = ratios[hit]
            else:
                hit = None
        z = z + alpha * p
        if hit is not None:
            working[hit] = True

    lam_G = np.zeros(m_G)
    lam_G[rows] = lam[m_E:]
    return z, lam[:m_E], np.maximum(lam_G, 0.0), converged, iteration
