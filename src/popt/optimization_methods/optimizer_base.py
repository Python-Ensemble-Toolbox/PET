'''Shared OptimizerBase for iterative optimization algorithms.'''
import inspect
import pprint
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult
from abc import ABC, abstractmethod
from functools import wraps

# Internal imports
import popt.misc_tools.optim_tools as ot
from ensemble.checkpoint import RestartMixin
from ensemble.logger import PetLogger

__author__ = "Mathias Methlie Nilsen, Rolf J. Lorentzen"
__all__ = [
    'OptimizerBase',
    'StepReport',
    'BoundTransformHandler',
    'OptimizerRestartMixin'
]


def _accepts_arguments(func, x, args, kwargs) -> bool:
    """Whether ``func`` can be called as ``func(x, *args, **kwargs)``.

    Answered from the signature, without calling. The optimizers support two
    kinds of objective -- a rich one taking the covariance and extras, and a
    plain one taking only the control vector -- and this is what tells them
    apart.

    Deciding it by calling and catching ``TypeError`` cannot: an objective that
    runs an ensemble of simulations and then raises ``TypeError`` internally is
    indistinguishable from one that rejected the arguments, so the error is
    swallowed and the entire evaluation repeated. The repeat then trips over
    the simulator scratch folders the first attempt created and reports
    ``FileExistsError``, with the real error nowhere to be seen.

    A callable whose signature cannot be inspected -- some builtins and C
    extensions -- is assumed to accept them, so the full call is attempted and
    any error propagates rather than being hidden.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind(x, *args, **kwargs)
    except TypeError:
        return False
    return True


def _describe_signature(func) -> str:
    """``func``'s signature for an error message, or '' if unavailable."""
    try:
        return str(inspect.signature(func))
    except (TypeError, ValueError):
        return ""


@dataclass(frozen=True)
class StepReport:
    """What one call to :meth:`OptimizerBase.update_step` produced.

    ``accepted`` says the optimizer committed a new iterate (through
    :meth:`OptimizerBase._commit_step`); the loop then does the bookkeeping
    every optimizer used to repeat. ``message`` is why it stopped when it did
    not, and becomes the result's ``message``.
    """

    accepted: bool
    message: str = ""


class OptimizerRestartMixin(RestartMixin):
    """Checkpoint/restart behaviour for optimizers.

    The implementation is shared with PIPT via
    :class:`ensemble.checkpoint.RestartMixin`; this subclass exists so the
    optimizer-facing name stays stable.
    """


class BoundTransformHandler:
    """
    Transform states between the original parameter domain and the unit cube.

    Notes
    -----
    All bounds must be finite whenever bounds are supplied.
    """

    def __init__(self, bounds=None, transform=False):
        '''
        Initialize the BoundTransformHandler.

        Parameters
        ----------
        bounds : sequence of (lower, upper) pairs, optional
            Lower and upper bounds for each state variable.
        transform : bool, optional
            If True, transform the optimization problem to the unit cube [0, 1]^n.
        '''
        self.transform = transform

        # ------------------------------------------------------------------
        # Bounds
        # ------------------------------------------------------------------
        if bounds is None:
            self.bounds = None
            self.lb = None
            self.ub = None
            return

        self.bounds = bounds
        self.lb, self.ub = np.asarray(bounds, dtype=float).T
        self.db = self.ub - self.lb
        self._validate_bounds()

    # ----------------------------------------------------------------------
    # Validation helpers
    # ----------------------------------------------------------------------

    def _validate_bounds(self):
        """Validate lower and upper bounds."""
        if not np.all(np.isfinite(self.lb)):
            raise ValueError("All lower bounds must be finite.")
        if not np.all(np.isfinite(self.ub)):
            raise ValueError("All upper bounds must be finite.")
        if np.any(self.ub <= self.lb):
            raise ValueError(
                "Every upper bound must be strictly greater "
                "than its lower bound."
            )

    def _validate_state(self, x):
        """Validate a state vector in the original parameter space."""
        if x.shape != self.lb.shape:
            raise ValueError(
                f"Expected shape {self.lb.shape}, got {x.shape}."
            )
        if np.any(np.isnan(x)):
            raise ValueError("State vector contains NaN values.")
        if np.any(x < self.lb) or np.any(x > self.ub):
            raise ValueError(
                "State vector is outside the specified bounds."
            )

    def _validate_unit_cube(self, u):
        """Validate a vector in unit-cube coordinates."""
        if u.shape != self.lb.shape:
            raise ValueError(
                f"Expected shape {self.lb.shape}, got {u.shape}."
            )
        if np.any(np.isnan(u)):
            raise ValueError(
                "Unit-cube vector contains NaN values."
            )
        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError(
                "Unit-cube coordinates must lie in [0, 1]."
            )

    # ----------------------------------------------------------------------
    # Coordinate transforms
    # ----------------------------------------------------------------------

    def state_to_unit_cube(self, x):
        """Transform original coordinates to unit-cube coordinates."""
        if (not self.transform) or (self.bounds is None):
            return x
        x = np.asarray(x, dtype=float)
        self._validate_state(x)
        return (x - self.lb) / self.db

    def unit_cube_to_state(self, u):
        """Transform unit-cube coordinates to original coordinates."""
        if (not self.transform) or (self.bounds is None):
            return u
        u = np.asarray(u, dtype=float)
        self._validate_unit_cube(u)
        return self.lb + u * self.db

    # ----------------------------------------------------------------------
    # Feasibility operations
    # ----------------------------------------------------------------------

    def project_to_bounds(self, x):
        """Project a vector onto the feasible domain."""
        x = np.asarray(x, dtype=float)
        if self.bounds is None:
            return x
        if self.transform:
            return np.clip(x, 0.0, 1.0)
        return np.clip(x, self.lb, self.ub)

    def project_gradient(self, x, g, tol=1e-8):
        """Project a gradient to respect active bound constraints."""
        if self.bounds is None:
            return g

        g_proj = g.copy()

        if self.transform:
            lower_bound = tol
            upper_bound = 1.0 - tol
        else:
            lower_bound = self.lb + tol
            upper_bound = self.ub - tol

        at_lower = x <= lower_bound
        at_upper = x >= upper_bound

        g_proj[at_lower] = np.minimum(g_proj[at_lower], 0.0)
        g_proj[at_upper] = np.maximum(g_proj[at_upper], 0.0)

        return g_proj

    # ----------------------------------------------------------------------
    # Derivative transforms
    # ----------------------------------------------------------------------
    def jac_to_unit_cube(self, jac):
        """Transform a gradient to unit-cube coordinates."""
        if (not self.transform) or (self.bounds is None):
            return jac
        return jac * self.db

    def jac_from_unit_cube(self, jac):
        """Transform a gradient from unit-cube coordinates."""
        if (not self.transform) or (self.bounds is None):
            return jac
        if jac is None:
            return None
        return jac / self.db

    def hess_to_unit_cube(self, hess):
        """Transform a Hessian to unit-cube coordinates."""
        if (not self.transform) or (self.bounds is None):
            return hess
        if hess is None:
            return None
        return hess * np.outer(self.db, self.db)

    def hess_from_unit_cube(self, hess):
        """Transform a Hessian from unit-cube coordinates."""
        if (not self.transform) or (self.bounds is None):
            return hess
        if hess is None:
            return None
        return hess / np.outer(self.db, self.db)



class OptimizerBase(OptimizerRestartMixin, ABC):
    """The iteration every optimizer shares; a subclass supplies the step.

    A subclass implements :meth:`update_step`, committing an improving point
    with :meth:`_commit_step` and returning a :class:`StepReport`, and names
    what its log row shows in :meth:`log_columns`. Everything else -- the
    starting evaluation, the callback, recording and saving the result, the
    log row, the function, state and projected-gradient convergence checks,
    restart checkpoints and the EPF outer loop -- happens here.
    """

    NAME = "Optimizer"
    """Shown in the start-of-run banner."""

    def __init__(self, x0, fun, jac=None, hess=None, args=(), bounds=None, callback=None, **options):
        """
        Parameters
        ----------
        x0 : ndarray
            Initial parameter vector.
        fun : callable
            Objective function.
        jac : callable, optional
            Gradient function.
        hess : callable, optional
            Hessian function.
        args : tuple, optional
            Extra positional arguments passed to callables: `fun`, `jac`, `hess`.
        bounds : sequence, optional
            Lower and upper bounds for each state variable.
        callback : callable, optional
            Called with the optimizer after every accepted step.
        **options
            Optimizer configuration such as tolerances, logging, restart, and
            persistence options.
            - maxiter: Maximum number of iterations (default: 100)
            - ftol: Relative function tolerance for convergence (default: 1e-5)
            - xtol: Relative change in state for convergence (default: 1e-8)
            - gtol: Projected-gradient infinity-norm tolerance for convergence (default: 1e-5)
            - fun0, jac0, hess0: Initial objective, gradient and Hessian values to reuse instead of evaluating them
            - logit: Enable logging (default: True)
            - logger_name: Log file name (default: 'OPTIM.log')
            - restart: Enable restart from file (default: False)
            - restartsave: Save restart file after each iteration (default: False)
            - restart_file: Path for restart file (default: '{optimizer_name}_restart.pkl')
            - epf: Dictionary of EPF options (default: None)
                - r: Initial penalty factor
                - r_factor: Penalty factor update multiplier (default: 2)
                - tol_factor: Function tolerance update multiplier (default: 0.9)
                - conv_crit: EPF convergence criterion, compared against the mean
                  penalty with the penalty factor divided out (default: 1e-5). The
                  objective must write `penalty` into the epf dict it is handed.
            - transform: Enable [lb, ub] --> [0, 1] transformation for optimization (default: False)
            - saveit: Save intermediate results after each iteration (default: False)
            - savefolder (or save_folder): Folder for those results (default: 'Iteration_Results')
        """
        # Store user configuration first.
        self.options = options
        self.args = args
        self.callback = callback if callable(callback) else None

        # Bounds and optional unit-cube transform.
        self.transform = options.get('transform', False)
        self.bound_handler = BoundTransformHandler(bounds, transform=self.transform)
        self.xk = self.bound_handler.state_to_unit_cube(x0) if self.transform else x0

        # Wrapped objective-related callables with evaluation counters.
        self.fun  = self._wrap_callable(fun, "fun")
        self.jac  = self._wrap_callable(jac, "jac", self.bound_handler.jac_to_unit_cube)
        self.hess = self._wrap_callable(hess, "hess", self.bound_handler.hess_to_unit_cube)

        # Core iteration controls.
        self.iteration = 0
        self.maxiter = options.get('maxiter', 100)

        # Restart/checkpoint controls.
        self.restart = options.get('restart', False)
        self.restartsave = options.get('restartsave', False)
        self.restart_file = options.get(
            'restart_file',
            options.get('restartfile', f'{type(self).__name__.lower()}_restart.pkl')
        )
        self._restart_loaded = False

        # EPF controls.
        self.epf = options.get('epf', {})
        self.epf_maxiter = self.epf.get('max_epf_iter', 10) if self.epf else 1
        self.epf_iteration = 0

        # Convergence tolerances.
        self.ftol = options.get('ftol', 1e-5)  # Relative function tolerance
        self.xtol = options.get('xtol', 1e-8)  # Relative state-change tolerance
        self.gtol = options.get('gtol', 1e-5)  # Projected-gradient infinity norm

        # Iteration state. Initial values may be handed in; whatever is
        # missing is evaluated when the run starts (see `_start`).
        self.fk = options.get('fun0', None)
        self.jk = options.get('jac0', None)
        self.hk = options.get('hess0', None)
        self.fk_old = None
        self.xk_old = None
        self._started = False

        # Logging.
        self.logger = None
        if options.get('logit', True):
            self.logger = PetLogger(options.get('logger_name', 'OPTIM.log'))

        # Result container and persistence.
        self.conv_msg = ''
        self.optimize_results = OptimizeResult()
        self.saveit = options.get('saveit', False)
        self.savefolder = options.get('savefolder', options.get('save_folder', 'Iteration_Results'))

    @classmethod
    def minimize(cls, x0, fun, *args, **kwargs) -> OptimizeResult:
        """Construct the optimizer with these arguments, run it, and return its result.

        The arguments are the constructor's, in the constructor's order; see
        the class for what each optimizer takes.
        """
        optimizer = cls(x0, fun, *args, **kwargs)
        optimizer.run_optimization()
        return optimizer.optimize_results

    @abstractmethod
    def update_step(self) -> StepReport:
        """Take one step from the current iterate.

        Find a better point and make it current with :meth:`_commit_step`,
        which also keeps the previous iterate for the convergence checks; then
        return ``StepReport(True)``. The loop runs the callback, records and
        saves the result, logs a row and checks convergence -- none of that
        is the step's job. Return ``StepReport(False, why)`` when no
        acceptable step exists: the run stops and ``why`` is its message.
        """

    def run_optimization(self):
        """Run this optimizer to completion.

        Named for the job rather than the mechanism; the counterpart in pipt is
        ``AssimilationScheme.run_assimilation``.

        The loop handles restart restoration, the starting evaluation, optional
        EPF outer iterations, repeated calls to ``update_step()``, and the
        shared convergence checks. When enabled, restart files are updated
        after successful iterations and after EPF penalty updates.
        """

        if self.restart and not self._restart_loaded:
            self.load_restart()
        elif not self.restart:
            self.clear_restart()

        if not (self._restart_loaded or self._started):
            self._start()

        if self.epf_iteration == 0:
            self.epf_iteration = 1

        # EPF outer loop
        while self.epf_iteration <= self.epf_maxiter:

            self._refresh_epf_function_value()

            # Main optimization loop
            update_step_failed = False
            optimization_converged = False
            while self.iteration < self.maxiter:
                self.iteration += 1

                report = self.update_step()
                if not report.accepted:
                    self.conv_msg = report.message
                    update_step_failed = True
                    break

                # The step is committed; this is the bookkeeping that follows every accepted step.
                if self.callback is not None:
                    self.callback(self)
                self._record_results()
                self._log_iteration()

                # Check function tolerance convergence
                if self.check_function_convergence():
                    optimization_converged = True
                # Check state tolerance convergence
                elif self.check_state_convergence():
                    optimization_converged = True
                # Check subclass-specific convergence criteria (if any)
                elif self.check_convergence():
                    optimization_converged = True

                # Save restart file if enabled
                if self.restartsave:
                    self.save_restart()

                if optimization_converged:
                    break

            if (self.iteration == self.maxiter) and (not optimization_converged):
                self.conv_msg = 'Maximum number of iterations reached'

            if update_step_failed or (not self.epf):
                # If the update step failed or EPF is not enabled, we exit the loop.
                break

            # Check if EPF convergence is met
            if self.check_epf_convergence():
                break

            # Update iteration counters
            self.iteration = 0
            self.epf_iteration += 1

            if self.restartsave:
                self.save_restart()

        # Set convergence message
        self.optimize_results['message'] = self.conv_msg

        # Log convergence message
        self._log_convergence()

    # ==========================================
    # What the loop does around a step
    # ==========================================
    def _start(self):
        """Evaluate what the first step needs and record the starting point."""
        self._started = True
        if self.logger:
            self.logger(f'========== Starting {self.NAME} Minimization ==========')
            if self.options:
                self.logger(f'\n\nUSER-SPECIFIED OPTIONS:\n{pprint.pformat(OptimizeResult(self.options))}\n')

        if self.fk is None:
            if self.logger:
                self.logger('Computing initial function value...')
            self.fk = self._objective_value(self.xk)
        self._evaluate_missing_derivatives()

        self._log_iteration()
        self._record_results()

    def _objective_value(self, x):
        """The objective at ``x`` as this optimizer keeps it (``fun``'s value as returned, by default)."""
        return self.fun(x)

    def _evaluate_missing_derivatives(self):
        """Evaluate the gradient and Hessian at the current iterate when the optimizer has none.

        Used at the start and by optimizers that invalidate them between
        steps. One that computes its derivatives differently (EnOpt's
        ensemble gradient needs the covariance) overrides this.
        """
        if self.jk is None and self.jac is not None:
            self.jk = self.jac(self.xk)
        if self.hk is None and self.hess is not None:
            self.hk = self.hess(self.xk)

    def _commit_step(self, x_new, f_new, jac=None, hess=None):
        """Make ``x_new`` the current iterate; the one it replaces becomes ``xk_old``/``fk_old``.

        The convergence checks compare the two, so a step that skipped either
        assignment used to iterate and log normally while never converging.
        Pass ``jac``/``hess`` when the step evaluated them at the new point.
        """
        self.xk_old = self.xk
        self.fk_old = self.fk
        self.xk = x_new
        self.fk = f_new
        if jac is not None:
            self.jk = jac
        if hess is not None:
            self.hk = hess

    def _record_results(self):
        """Refresh the result object and, if asked, save it."""
        self.optimize_results = self._update_optimize_result()
        if self.saveit:
            ot.save_optimize_results(self.optimize_results, folder=self.savefolder)

    def log_columns(self) -> dict:
        """One row of the iteration log. Optimizers override to show their own quantities."""
        return {'iter.': self.iteration, 'fun': float(np.mean(self.fk))}

    def _log_iteration(self):
        if self.logger:
            columns = self.log_columns()
            if self.epf:
                columns['EPF iter.'] = self.epf_iteration
            self.logger(**columns)

    # ==========================================
    # Convergence
    # ==========================================
    def check_convergence(self) -> bool:
        """Optimizer-specific criteria; by default the projected gradient against ``gtol``.

        Runs after the function and state checks. An optimizer with more
        criteria extends this; one without a gradient gets ``False``.
        """
        if self.jk is None:
            return False
        proj_jac = self.bound_handler.project_gradient(self.xk, self.jk)
        if np.linalg.norm(proj_jac, np.inf) < self.gtol:
            self.conv_msg = f'Projected gradient norm ‖g‖∞ < {self.gtol}.'
            return True
        return False

    def check_function_convergence(self) -> bool:
        """Check convergence based on relative change in objective value."""
        if self.fk_old is not None:
            df = np.mean(self.fk) - np.mean(self.fk_old)
            if abs(df) < self.ftol*np.abs(np.mean(self.fk_old)):
                self.conv_msg = f'Function change satisfies |Δf| < {self.ftol}·|f_prev|'
                return True
        return False

    def check_state_convergence(self) -> bool:
        """Check convergence based on the norm of the state update."""
        if self.xk_old is not None:
            dx = np.linalg.norm(self.xk - self.xk_old)
            if dx < self.xtol:
                self.conv_msg = f'State change norm ‖Δx‖₂ < {self.xtol}'
                return True
        return False

    def check_epf_convergence(self):
        """Evaluate convergence of the outer EPF iteration.

        The loop stops once the constraints are satisfied, measured as the mean of
        ``self.epf['penalty']`` with the penalty factor ``r`` divided back out. The
        objective is responsible for writing ``penalty`` into the ``epf`` dict it is
        handed; without it there is nothing to converge on and this raises.

        Returns
        -------
        bool
            ``True`` when the EPF loop should terminate, otherwise ``False``.
        """
        if self.epf_iteration == self.epf_maxiter:
            if self.logger:
                self.logger('─────> Maximum number of outer EPF iterations reached')
            return True

        # Mean penalty magnitude with the penalty factor divided back out, so the test
        # asks whether the constraints are still violated rather than whether the
        # controls happened to move. The objective writes `penalty` into the epf dict it
        # is handed; `cost_functions.epf.epf` returns r * 0.5 * (...), so dividing by r
        # leaves the violation itself.
        if 'penalty' not in self.epf:
            raise KeyError(
                "EPF convergence needs self.epf['penalty']; the objective must write it "
                "into the epf dict it is passed."
            )
        penalty = np.asarray(self.epf['penalty'])
        if penalty.size == 0:
            raise ValueError('EPF penalty is empty; cannot compute the convergence criterion.')
        mean_penalty = np.mean(penalty) / self.epf['r']
        conv_crit = self.epf.get('conv_crit', 1e-5)
        if mean_penalty > conv_crit:

            # Update penalty factor
            rold = self.epf['r']
            rnew = rold * self.epf.get('r_factor', 2)
            self.epf['r'] = rnew
            if self.logger:
                self.logger(f'EPF penalty factor updated: {rold} ─────> {rnew}')

            # Update function tolerance
            ftol_old = self.ftol
            ftol_new = ftol_old * self.epf.get('tol_factor', 0.9)
            self.ftol = ftol_new
            if self.logger:
                self.logger(f'Function tolerance updated: {ftol_old} ─────> {ftol_new}')

            return False
        else:
            if self.logger:
                self.logger(f'Outer EPF loop converged ─────> penalty term smaller than {conv_crit}')
            return True

    # ==========================================
    # Internal utility functions
    # ==========================================
    def _update_optimize_result(self):
        xk = self.bound_handler.project_to_bounds(self.xk)
        xk = self.bound_handler.unit_cube_to_state(xk)
        result = OptimizeResult({
            'x': xk,
            'fun': self.fk,
            'jac': self.bound_handler.jac_from_unit_cube(self.jk),
            'hess': self.bound_handler.hess_from_unit_cube(self.hk),
            'nit': self.iteration,
            'nfev': self.fun.nfev,
            'njev': self.jac.nfev if self.jac else 0,
            'nhev': self.hess.nfev if self.hess else 0,
        })
        return result

    def _refresh_epf_function_value(self):
        if self.epf_iteration <= 1 or self.iteration != 0:
            return

        self.fk = self.fun(self.xk)
        self._record_results()

    def _wrap_callable(self, func, name, transform_result=None):
        if func is None:
            return None

        if not callable(func):
            raise ValueError(f"The {name} must be callable.")

        @wraps(func)
        def wrapper(x, *args, **kwargs):
            wrapper.nfev += 1

            x = self.bound_handler.project_to_bounds(x)
            x = self.bound_handler.unit_cube_to_state(x)

            # check if args empty, if so, don't pass them to func
            if not args:
                args = self.args
            kwargs["epf"] = self.epf

            # A plain objective may accept only `x`. Decide that from the
            # signature rather than by calling and catching TypeError: the
            # objective runs a full ensemble of simulations, and a TypeError
            # raised *inside* it would otherwise be swallowed and the whole
            # evaluation silently repeated. The repeat then failed on the
            # scratch folders the first attempt had already created, reporting
            # FileExistsError and hiding the real error completely.
            if _accepts_arguments(func, x, args, kwargs):
                result = func(x, *args, **kwargs)
            elif _accepts_arguments(func, x, (), {}):
                result = func(x)
            else:
                raise TypeError(
                    f"The {name} {getattr(func, '__name__', func)!r} "
                    f"{_describe_signature(func)} accepts neither "
                    f"(x, *args, **kwargs) nor (x). It must take either the "
                    f"control vector alone, or the control vector plus the "
                    f"optimizer's args and keywords."
                )

            if (transform_result is not None) and self.transform:
                result = transform_result(result)

            return result

        wrapper.nfev = 0
        return wrapper

    def _log_convergence(self):
        if self.logger:
            self.logger('==========================================================================')
            self.logger(f'  Reason for convergence: {self.conv_msg}')
            self.logger(f'  Final function value: {np.mean(self.fk):.4f}')
            self.logger(f'  Total iterations: {self.iteration}')
            self.logger(f'  Total function evaluations: {self.fun.nfev}')
            if self.jac:
                self.logger(f'  Total jacobian evaluations: {self.jac.nfev}')
            if self.hess:
                self.logger(f'  Total hessian evaluations: {self.hess.nfev}')
            if self.epf:
                self.logger(f'  Total EPF iterations: {self.epf_iteration}')
            self.logger('==========================================================================')

    # =============================================
    # Restart state hooks consumed by RestartMixin
    # =============================================

    def _get_base_restart_state(self):
        return {
            'xk': self.xk,
            'fk': self.fk,
            'jk': self.jk,
            'hk': self.hk,
            'xk_old': self.xk_old,
            'fk_old': self.fk_old,
            'iteration': self.iteration,
            'epf_iteration': self.epf_iteration,
            'ftol': self.ftol,
            'xtol': self.xtol,
            'epf': self.epf,
            'conv_msg': self.conv_msg,
            'optimize_results': dict(self.optimize_results),
            'nfev': getattr(self.fun, 'nfev', 0),
            'njev': getattr(self.jac, 'nfev', 0) if self.jac else 0,
            'nhev': getattr(self.hess, 'nfev', 0) if self.hess else 0,
        }

    def _set_base_restart_state(self, state):
        self.xk = state['xk']
        self.fk = state['fk']
        self.jk = state['jk']
        self.hk = state['hk']
        self.xk_old = state['xk_old']
        self.fk_old = state['fk_old']
        self.iteration = state['iteration']
        self.epf_iteration = state['epf_iteration']
        self.ftol = state['ftol']
        self.xtol = state['xtol']
        self.epf = state['epf']
        self.conv_msg = state.get('conv_msg', '')
        self.optimize_results = OptimizeResult(state.get('optimize_results', {}))

        self.fun.nfev = state.get('nfev', getattr(self.fun, 'nfev', 0))
        if self.jac:
            self.jac.nfev = state.get('njev', getattr(self.jac, 'nfev', 0))
        if self.hess:
            self.hess.nfev = state.get('nhev', getattr(self.hess, 'nfev', 0))
