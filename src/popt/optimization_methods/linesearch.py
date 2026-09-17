"""Line-search-based deterministic optimization methods.

This module implements gradient-based algorithms that share a common line
search interface, including gradient descent, BFGS, and Newton-CG.
"""

import numpy as np

# Internal imports
from popt.optimization_methods.subroutines import line_search, line_search_backtracking, bfgs_update, newton_cg
from popt.optimization_methods.optimizer_base import OptimizerBase, StepReport

__author__ = "Mathias Methlie Nilsen"
__all__ = ["LineSearch"]

# -----------------------------------------
# Some symbols for logger
# -----------------------------------------
subk = 'ₖ'
sup2 = '²'
jac_inf_symbol = f'‖jac(x{subk})‖∞'
fun_xk_symbol  = f'fun(x{subk})'
nabla_symbol = "∇"


class LineSearch(OptimizerBase):
    """Line-search optimizer compatible with OptimizerBase.

    The class supports gradient descent, BFGS, and Newton-CG search
    directions, together with either Wolfe or backtracking line search.
    It can operate with bounds, optional state transformations, logging,
    result persistence, and restart checkpoints.
    """

    VALID_METHODS = ("GD", "BFGS", "Newton-CG")
    LS_METHODS = {
        0: line_search_backtracking, # Backtracking line search
        1: line_search,              # Wolfe line search
    }


    def __init__(self, x0, fun, method='GD', jac=None, hess=None, args=(), bounds=None, callback=None, **options):
        """Initialize a line-search optimizer instance.

        Parameters
        ----------
        x0 : ndarray
            Initial parameter vector.
        fun : callable
            Objective function.
        method : {'GD', 'BFGS', 'Newton-CG'}, optional
            Search-direction method.
        jac : callable
            Gradient function.
        hess : callable, optional
            Hessian function, required by ``Newton-CG``.
        args : tuple, optional
            Extra positional arguments passed to the wrapped callables.
        bounds : sequence, optional
            Lower and upper bounds for each state variable.
        callback : callable, optional
            Callback invoked after successful updates.
        **options
            Line-search configuration, plus everything :class:`OptimizerBase` takes.
            - step_size: Initial step size (default: None, auto-scaled).
            - step_size_max: Maximum step size (default: 1e5).
            - step_size_adapt: Step size adaptation strategy (0: none, 1: function-based, 2: gradient-based). Default is 1 (function-based).
            - c1: Armijo condition constant (default: 1e-4).
            - c2: Curvature condition constant (default: 0.9).
            - rho: Step size reduction factor for backtracking (default: 0.5).
            - lsmaxiter: Maximum line search iterations (default: 10).
            - lsmethod: Line search method (0: backtracking, 1: Wolfe, default: 1).
            - normalize: Whether to normalize the search direction (default: False).
            - recompute_jac: Number of gradient recomputation attempts on line search failure (default: 0).
            - hess0_inv: Initial inverse-Hessian approximation for BFGS (default: identity).
        """

        if jac is None:
            raise ValueError("LineSearch requires a Jacobian (gradient) function for the specified methods.")

        # Initialize the base class
        super().__init__(x0, fun, jac, hess, args, bounds, callback, **options)

        # Validate method and required callables
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Valid options are: {self.VALID_METHODS}")
        if method == "Newton-CG" and hess is None:
            raise ValueError(f"Method '{method}' requires a Hessian function.")

        # Line search specific attributes
        self.method = method
        self.NAME = f"Line Search ({method})"  # the banner names the search direction

        # Set options for step-size
        self.step_size       = options.get('step_size', None)
        self.step_size_max   = options.get('step_size_max', 1e5)
        self.step_size_adapt = options.get('step_size_adapt', 1)
        self.step_taken      = None  # the step length of the last accepted step

        # Line search specific options
        self.line_search_options = {
            'c1': options.get('c1', 1e-4),              # Armijo condition constant
            'c2': options.get('c2', 0.9),               # Curvature condition constant
            'rho': options.get('rho', 0.5),             # Step size reduction factor for backtracking
            'amax': self.step_size_max,                 # Max step size for line search
            'maxiter': options.get('lsmaxiter', 10),    # Max line search iterations (the subroutines read 'maxiter')
            'logger': self.logger,                      # Logger instance

        }
        try:
            lsmethod = options.get('lsmethod', 1)
            self.line_search_fn = self.LS_METHODS[lsmethod]
        except KeyError:
            raise ValueError(f"Invalid line search method: {lsmethod}")

        # Other options
        self.recompute_jac = options.get('recompute_jac', 0)
        self.normalize = options.get('normalize', False)
        self.jk_old = None
        self.pk_old = None

        if self.method == 'BFGS':
            self.bk = options.get('hess0_inv', np.eye(self.xk.size))  # BFGS approximation of the inverse Hessian

    def update_step(self) -> StepReport:
        """
        Perform one optimization step.

        The method computes a search direction, performs a line search, and
        commits the new iterate on success. When enabled, it can recompute the
        gradient and retry if the line search fails.
        """
        iter_jac_recompute = 0  # Reset recompute counter for this step

        # Perform line-search step (with optional recompute loop)
        while iter_jac_recompute <= self.recompute_jac:
            # Gradient and Hessian at the current iterate, if not already on hand
            # (the Hessian is invalidated after every accepted step, the
            # gradient when a retry asks for a fresh one).
            self._evaluate_missing_derivatives()
            pk = self._compute_search_direction()
            step_size, fk_new, jk_new = self._run_line_search(pk)

            # SUCCESS --> accept step and return
            if step_size:
                self._accept_step(pk, step_size, fk_new, jk_new)
                return StepReport(True)

            # FAILURE --> recompute or exit
            if iter_jac_recompute < self.recompute_jac:
                if self.logger:
                    self.logger('Recomputing gradient and retrying line search...')
                self.jk = None
                iter_jac_recompute += 1
            else:
                return StepReport(False, 'Line search failed to find a suitable step size')

    def _run_line_search(self, pk) -> tuple[float, float, np.ndarray]:
        """Run the line search algorithm to find an acceptable step size."""
        step_size = self._set_step_size(pk, self.step_size_max)
        step_size, fk_new, jk_new, _, _ = self.line_search_fn(
            step_size=step_size,
            xk=self.xk,
            pk=pk,
            fun=lambda x, *a, **kw: np.mean(self.fun(x, *a, **kw)),
            jac=self.jac,
            fk=np.mean(self.fk),
            jk=self.jk,
            **self.line_search_options
        )

        return step_size, fk_new, jk_new

    def _accept_step(self, pk, step_size, fk_new, jk_new) -> None:
        """Make the line-search point current and update what the next direction needs."""
        self.jk_old = self.jk
        self.pk_old = pk
        self.step_taken = step_size

        self._commit_step(self.bound_handler.project_to_bounds(self.xk + step_size * pk), fk_new, jac=jk_new)

        if self.method == 'BFGS':
            sk = self.xk - self.xk_old
            yk = self.jk - self.jk_old
            if self.iteration == 1:
                self.bk = np.dot(yk,sk)/np.dot(yk,yk) * np.eye(sk.size)
            self.bk = bfgs_update(self.bk, sk, yk)

        # The Hessian on hand belongs to the previous iterate; it is
        # recomputed at the next step if the method needs one.
        self.hk = None

    def _get_restart_state(self) -> dict:
        state = {
            'step_size': self.step_size,
            'jk_old': self.jk_old,
            'pk_old': self.pk_old,
        }
        if self.method == 'BFGS':
            state['bk'] = self.bk
        return state

    def _set_restart_state(self, state: dict) -> None:
        self.step_size = state.get('step_size', self.step_size)
        self.jk_old = state.get('jk_old', self.jk_old)
        self.pk_old = state.get('pk_old', self.pk_old)
        if self.method == 'BFGS' and 'bk' in state:
            self.bk = state['bk']


    def _compute_search_direction(self) -> np.ndarray:
        if self.method == 'GD':
            return -self.jk
        elif self.method == 'BFGS':
            return - np.matmul(self.bk, self.jk)
        elif self.method == 'Newton-CG':
            return newton_cg(self.jk, self.hk)
        else:
            raise ValueError(f"Unsupported method: {self.method}")


    def _set_step_size(self, pk, amax) -> float:
        if self.step_size is None:
            self.step_size = 0.25 / np.linalg.norm(pk, np.inf)

        alpha = float(np.asarray(self.step_size).reshape(-1)[0])

        if self.iteration > 1:
            slope = np.dot(pk, self.jk)
            if self.step_size_adapt == 1 and slope != 0:
                fk = float(np.asarray(np.mean(self.fk)).reshape(-1)[0])
                fk_old = float(np.asarray(np.mean(self.fk_old)).reshape(-1)[0])
                alpha = 2 * (fk - fk_old) / slope
            elif self.step_size_adapt == 2 and slope != 0:
                slope_old = np.dot(self.pk_old, self.jk_old)
                alpha = self.step_size * slope_old / slope

        alpha = float(abs(alpha))

        if alpha >= amax:
            alpha = 0.75 * amax

        return alpha

    def log_columns(self) -> dict:
        """The row of the iteration log: iteration, objective, gradient infinity norm, step length taken."""
        return {
            'iter.': self.iteration,
            fun_xk_symbol: self.fk,
            jac_inf_symbol: np.linalg.norm(self.jk, np.inf),
            'step-size': self.step_taken if self.step_taken is not None else self.step_size,
        }
