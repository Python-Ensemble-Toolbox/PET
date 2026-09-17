"""Non-Gaussian generalisation of EnOpt: the sampling distribution adapts as well."""

import numpy as np

from popt.optimization_methods.optimizer_base import OptimizerBase, StepReport
from popt.optimization_methods.subroutines.cma import CMA
import popt.optimization_methods.subroutines.optimizers as opt

__all__ = ["GenOpt"]


class GenOpt(OptimizerBase):
    """Generalized ensemble optimization with an adapting mutation distribution.

    EnOpt draws its ensemble from a Gaussian whose covariance is fixed apart from an
    optional Hessian-driven update. GenOpt draws from the marginals of
    :class:`~popt.ensembles.ensemble_generalized.GeneralizedEnsemble` -- Beta,
    logistic, truncated Gaussian -- and moves the distribution itself along with the
    controls: ``theta`` (the marginal's shape) follows its own gradient, and the
    correlation matrix follows ``corr_adapt``.

    So each accepted step updates three things rather than one: the controls from
    ``jac``, ``theta`` from ``jac_mut``, and ``corr`` from ``corr_adapt`` -- which is
    either a :class:`CMA` instance, called with the ensemble the mutation gradient
    was built from, or any callable returning a matrix to descend along.

    Examples
    --------
    ```python
    ensemble = GeneralizedEnsemble(options, simulator, objective)
    cma = CMA(ne=ensemble.num_samples, dim=x0.size, corr_update=True)
    result = GenOpt.minimize(
        x0, ensemble.function,
        jac=ensemble.gradient, jac_mut=ensemble.mutation_gradient,
        args=(ensemble.get_theta(), ensemble.get_corr()),
        corr_adapt=cma, bounds=bounds,
    )
    ```
    """

    NAME = "GenOpt"
    VALID_OPTIMIZERS = ("GD", "Adam")

    def __init__(self, x0, fun, jac=None, jac_mut=None, corr_adapt=None,
                 args=(), bounds=None, callback=None, **options):
        """
        Parameters
        ----------
        x0 : ndarray
            Initial control vector.
        fun : callable
            Objective function.
        jac : callable
            Ensemble gradient, called as ``jac(x, theta, corr)``.
        jac_mut : callable
            Mutation gradient, called as ``jac_mut(x, theta, corr)``. For a
            :class:`CMA` ``corr_adapt`` it is called with ``return_ensembles=True``
            and must then also return ``{'gaussian': ..., 'objective': ...}``.
        corr_adapt : CMA or callable, optional
            Correlation-matrix adaptation. A :class:`CMA` instance is called with the
            ensemble; any other callable is called with no arguments and its result
            is descended along with step size ``alpha_corr``. ``None`` leaves the
            correlation fixed.
        args : tuple
            ``(theta, corr)``: the initial marginal parameter and correlation matrix.
        bounds : sequence, optional
            (min, max) per control.
        callback : callable, optional
            Invoked after each accepted step.
        **options
            GenOpt configuration, plus everything :class:`OptimizerBase` takes.

            - tol: objective improvement required to accept a step (default: 1e-6).
            - alpha: initial step size for the controls (default: 0.1).
            - alpha_theta: step size for the marginal parameter (default: 0.1).
            - alpha_corr: step size for the correlation, for a non-CMA ``corr_adapt`` (default: 0.1).
            - beta: momentum (default: 0.0).
            - nesterov: evaluate the gradients at the momentum-extrapolated point (default: False).
            - alpha_maxiter: backtracking trials per iteration (default: 5).
            - resample: resampling attempts when backtracking fails (default: 0).
            - normalize: scale both gradients by their inf-norm (default: True).
            - cov_factor: shrink factor applied to theta when resampling (default: 0.5).
            - optimizer: ``GD`` or ``Adam`` (default: ``GD``).
        """
        if jac is None:
            raise ValueError("GenOpt requires a Jacobian (ensemble gradient) callable.")
        if jac_mut is None:
            raise ValueError("GenOpt requires a jac_mut (mutation gradient) callable; "
                             "without it the distribution never adapts and this is EnOpt.")
        if len(args) < 2:
            raise ValueError("GenOpt needs args = (theta, corr): the initial marginal "
                             "parameter and correlation matrix.")

        super().__init__(x0, fun, jac=jac, args=(), bounds=bounds, callback=callback, **options)

        self.jac_mut = jac_mut
        self.corr_adapt = corr_adapt

        self.obj_func_tol = options.get("tol", 1e-6)
        self.ftol = options.get("tol", options.get("ftol", self.ftol))
        self.alpha = options.get("step_size", options.get("alpha", 0.1))
        self.alpha_theta = options.get("alpha_theta", 0.1)
        # Upstream read 'alpha_theta' for this too, so `alpha_corr` silently did
        # nothing and the correlation moved at the theta step size.
        self.alpha_corr = options.get("alpha_corr", 0.1)
        self.beta = options.get("beta", 0.0)
        self.nesterov = options.get("nesterov", False)
        self.alpha_iter_max = options.get("alpha_maxiter", 5)
        self.max_resample = options.get("resample", 0)
        self.normalize = options.get("normalize", True)
        self.cov_factor = options.get("cov_factor", 0.5)

        self.theta = np.asarray(args[0], dtype=float)
        self.corr = np.asarray(args[1], dtype=float)
        self.state_step = np.zeros_like(self.xk, dtype=float)
        self.theta_step = np.zeros_like(self.theta, dtype=float)
        self.alpha_iter = 0

        self.optimizer_name = options.get("optimizer", "GD")
        self.optimizer = self._build_optimizer(self.optimizer_name)

    # ------------------------------------------------------------------
    # The step
    # ------------------------------------------------------------------

    def update_step(self) -> StepReport:
        """One GenOpt step: controls by backtracking, then theta and the correlation."""
        self.optimizer.restore_parameters()
        resampling_iter = 0
        new_func_values = self.fk

        while resampling_iter <= self.max_resample:
            shrink = self.cov_factor ** resampling_iter
            self.jk, theta_gradient, ensembles = self._compute_search_quantities(shrink)

            self.alpha_iter = 0
            while self.alpha_iter <= self.alpha_iter_max:
                new_state, new_step = self.optimizer.apply_update(
                    self.xk, self.jk, iter=self.iteration
                )
                new_state = self.bound_handler.project_to_bounds(new_state)
                new_func_values = self.fun(new_state)

                if np.mean(self.fk) - np.mean(new_func_values) > self.obj_func_tol:
                    self._accept_step(new_state, new_func_values, new_step,
                                      theta_gradient, ensembles)
                    return StepReport(True)

                if self.alpha_iter < self.alpha_iter_max:
                    self.optimizer.apply_backtracking()
                    self.alpha_iter += 1
                else:
                    break

            if (resampling_iter < self.max_resample) and (np.mean(new_func_values) > np.mean(self.fk)):
                resampling_iter += 1
                self.optimizer.restore_parameters()
                continue

            return StepReport(False, "GenOpt failed to find an improving step.")

        return StepReport(False, "GenOpt exhausted all resampling attempts.")

    def _evaluate_missing_derivatives(self):
        # Both gradients take theta and corr and are evaluated inside the step,
        # never at the bare iterate. Same reason as EnOpt.
        pass

    def _compute_search_quantities(self, shrink):
        theta_step = self.beta * self.theta_step if self.nesterov else 0.0
        state_step = self.beta * self.state_step if self.nesterov else 0.0

        theta = shrink * (self.theta + theta_step)
        x_for_grad = self.xk + state_step

        gradient = self.jac(x_for_grad, theta, self.corr, epf=self.epf)

        # CMA needs the Gaussian samples and their objective values. Ask for them
        # in the same call so the ensemble is drawn -- and simulated -- once.
        if isinstance(self.corr_adapt, CMA):
            theta_gradient, ensembles = self.jac_mut(
                x_for_grad, theta, self.corr, return_ensembles=True
            )
        else:
            theta_gradient, ensembles = self.jac_mut(x_for_grad, theta, self.corr), None

        theta_gradient = np.asarray(theta_gradient, dtype=float)
        if self.normalize:
            gradient = gradient / np.maximum(np.linalg.norm(gradient, np.inf), 1e-12)
            theta_gradient = theta_gradient / np.maximum(
                np.linalg.norm(theta_gradient, np.inf), 1e-12
            )

        return gradient, theta_gradient, ensembles

    def _accept_step(self, new_state, new_func_values, new_step, theta_gradient, ensembles):
        self._commit_step(new_state, new_func_values)
        self.state_step = new_step
        if hasattr(self.optimizer, "get_step_size"):
            self.alpha = self.optimizer.get_step_size()

        # Theta is not backtracked; it follows its own gradient once the controls
        # have found a step that improves the objective.
        self.theta_step = self.beta * self.theta_step - self.alpha_theta * theta_gradient
        self.theta = self.theta + self.theta_step

        self._adapt_correlation(new_step, ensembles)

        if self.xk.size == 1 and hasattr(self.optimizer, "step_size"):
            self.optimizer.step_size /= 2

        self.optimizer.restore_parameters()

    def _adapt_correlation(self, new_step, ensembles):
        if isinstance(self.corr_adapt, CMA):
            # `step / alpha` is the unit-length direction the evolution path wants;
            # alpha can be zero if the step rule collapsed, so guard the division.
            alpha = self.alpha if self.alpha else 1.0
            self.corr = self.corr_adapt(
                cov=self.corr,
                step=new_step / alpha,
                X=ensembles["gaussian"],
                J=ensembles["objective"],
            )
        elif callable(self.corr_adapt):
            self.corr = self.corr - self.alpha_corr * self.corr_adapt()

    # ------------------------------------------------------------------
    # Base-class hooks
    # ------------------------------------------------------------------

    def _build_optimizer(self, optimizer_name):
        if optimizer_name not in self.VALID_OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{optimizer_name}' not recognized for GenOpt. "
                f"Valid options are: {self.VALID_OPTIMIZERS}."
            )
        if optimizer_name == "GD":
            return opt.GradientDescent(self.alpha, self.beta)
        return opt.Adam(self.alpha, self.beta)

    def _get_restart_state(self) -> dict:
        return {
            "theta": self.theta,
            "corr": self.corr,
            "state_step": self.state_step,
            "theta_step": self.theta_step,
            "alpha": self.alpha,
            "alpha_iter": self.alpha_iter,
            "obj_func_tol": self.obj_func_tol,
            "optimizer_name": self.optimizer_name,
            "optimizer_state": dict(self.optimizer.__dict__),
        }

    def _set_restart_state(self, state: dict) -> None:
        self.theta = state.get("theta", self.theta)
        self.corr = state.get("corr", self.corr)
        self.state_step = state.get("state_step", self.state_step)
        self.theta_step = state.get("theta_step", self.theta_step)
        self.alpha = state.get("alpha", self.alpha)
        self.alpha_iter = state.get("alpha_iter", self.alpha_iter)
        self.obj_func_tol = state.get("obj_func_tol", self.obj_func_tol)

        self.optimizer_name = state.get("optimizer_name", self.optimizer_name)
        self.optimizer = self._build_optimizer(self.optimizer_name)
        self.optimizer.__dict__.update(state.get("optimizer_state", {}))

    def log_columns(self) -> dict:
        """Iteration, backtracking attempts, objective, step size, and the correlation's spread."""
        off_diagonal = self.corr - np.eye(self.corr.shape[0])
        return {
            "iter.": self.iteration,
            "alpha_iter": self.alpha_iter,
            "obj_func": float(np.mean(self.fk)),
            "step-size": self.alpha,
            "max corr": float(np.max(off_diagonal)),
            "min corr": float(np.min(self.corr)),
        }
