"""Ensemble optimization methods compatible with OptimizerBase."""

import numpy as np

from popt.misc_tools import optim_tools as ot
from popt.optimization_methods.optimizer_base import OptimizerBase, StepReport
import popt.optimization_methods.subroutines.optimizers as opt

__author__ = ""
__all__ = ["EnOpt"]


class EnOpt(OptimizerBase):
    """Ensemble-based optimization (EnOpt)."""

    NAME = "EnOpt"
    VALID_OPTIMIZERS = ("GD", "Adam", "AdaMax", "Steihaug")

    def __init__(self, x0, fun, jac=None, hess=None, args=(), bounds=None, callback=None, **options):
        """Initialize an EnOpt optimizer instance.

        Parameters
        ----------
        x0 : ndarray
            Initial control/state vector.
        fun : callable
            Objective function.
        jac : callable
            Ensemble gradient function.
        hess : callable, optional
            Ensemble Hessian function.
        args : tuple, optional
            The first tuple element is interpreted as the initial covariance.
        bounds : sequence, optional
            Lower and upper bounds for each state variable.
        callback : callable, optional
            Callback invoked after successful updates.
        **options
            EnOpt configuration, plus everything :class:`OptimizerBase` takes.
            - tol: Convergence tolerance for objective improvement (default: 1e-6). Also used as ``ftol`` when given.
            - step_size: Initial optimizer step size. Overrides ``alpha`` when provided.
            - alpha: Initial optimizer step size (default: 0.1).
            - alpha_cov: Covariance update scaling factor (default: 0.001).
            - beta: Momentum parameter used in the optimizer and optional Nesterov updates (default: 0.0).
            - nesterov: Whether to evaluate search quantities with Nesterov momentum (default: False).
            - alpha_maxiter: Maximum number of backtracking trials per iteration (default: 5).
            - resample: Number of covariance resampling attempts if no improvement is found (default: 0).
            - hessian: Whether to use the Hessian in the search direction computation (default: False).
            - normalize: Whether to normalize the gradient or Hessian-derived search quantities (default: True).
            - cov_factor: Covariance shrink factor applied during resampling (default: 0.5).
            - optimizer: Update rule name. Supported values are ``GD``, ``Adam``, ``AdaMax``, and ``Steihaug`` (default: ``GD``).
        """
        if jac is None:
            raise ValueError("EnOpt requires a Jacobian (ensemble gradient) callable.")

        # Keep args empty for wrapped callables to avoid duplicating covariance
        # (EnOpt passes covariance explicitly during each update).
        super().__init__(x0, fun, jac=jac, hess=hess, args=(), bounds=bounds, callback=callback, **options)

        # EnOpt controls
        self.obj_func_tol = options.get("tol", 1e-6)
        self.ftol = options.get("tol", options.get("ftol", self.ftol))
        self.alpha = options.get("step_size", options.get("alpha", 0.1))
        self.alpha_cov = options.get("alpha_cov", 0.001)
        self.beta = options.get("beta", 0.0)
        self.nesterov = options.get("nesterov", False)
        self.alpha_iter_max = options.get("alpha_maxiter", 5)
        self.max_resample = options.get("resample", 0)
        self.use_hessian = options.get("hessian", False)
        self.normalize = options.get("normalize", True)
        self.cov_factor = options.get("cov_factor", 0.5)

        # Dynamic EnOpt state
        self.cov = np.asarray(args[0], dtype=float)
        self.state_step = np.zeros_like(self.xk, dtype=float)
        self.cov_step = np.zeros_like(self.cov, dtype=float)
        self.alpha_iter = 0

        self.optimizer_name = options.get("optimizer", "GD")
        self.optimizer = self._build_optimizer(self.optimizer_name)

    @property
    def obj_func_values(self):
        """Legacy alias for ``fk``."""
        return self.fk

    def update_step(self) -> StepReport:
        """Perform one EnOpt step with backtracking and optional resampling."""
        self.optimizer.restore_parameters()
        resampling_iter = 0

        while resampling_iter <= self.max_resample:
            shrink = self.cov_factor ** resampling_iter
            self._apply_optimizer_backtracking(np.sqrt(self.cov_factor) ** resampling_iter)
            self.jk, self.hk = self._compute_search_quantities(shrink)

            self.alpha_iter = 0
            while self.alpha_iter <= self.alpha_iter_max:
                new_state, new_step = self.optimizer.apply_update(
                    self.xk,
                    self.jk,
                    hessian=self.hk,
                    iter=self.iteration,
                )
                new_state = self.bound_handler.project_to_bounds(new_state)
                new_func_values = self.fun(new_state)

                if np.mean(self.fk) - np.mean(new_func_values) > self.obj_func_tol:
                    self._accept_step(new_state, new_func_values, new_step, self.hk)
                    return StepReport(True)

                if self.alpha_iter < self.alpha_iter_max:
                    self._apply_optimizer_backtracking()
                    self.alpha_iter += 1
                else:
                    break

            if (resampling_iter < self.max_resample) and (np.mean(new_func_values) > np.mean(self.fk)):
                resampling_iter += 1
                self.optimizer.restore_parameters()
                continue

            return StepReport(False, "EnOpt failed to find an improving step.")

        return StepReport(False, "EnOpt exhausted all resampling attempts.")

    def _evaluate_missing_derivatives(self):
        # The ensemble gradient takes the covariance and is evaluated inside
        # every step (`_compute_search_quantities`), never at the bare iterate.
        pass

    def _compute_search_quantities(self, shrink):
        cov_step = self.beta * self.cov_step if self.nesterov else 0.0
        state_step = self.beta * self.state_step if self.nesterov else 0.0

        cov = shrink * (self.cov + cov_step)
        x_for_grad = self.xk + state_step

        gradient = self.jac(x_for_grad, cov, epf=self.epf)
        hessian = self.hess(x_for_grad, cov) if self.hess is not None else None

        if self.use_hessian:
            gradient = np.linalg.inv(hessian) @ (self.cov @ self.cov) @ gradient
        elif self.normalize:
            gradient = gradient / np.maximum(np.linalg.norm(gradient, np.inf), 1e-12)

        if self.normalize and hessian is not None:
            hessian = hessian / np.maximum(np.linalg.norm(hessian, np.inf), 1e-12)

        return gradient, hessian

    def _accept_step(self, new_state, new_func_values, new_step, hessian):
        self._commit_step(new_state, new_func_values)
        self.state_step = new_step
        if hasattr(self.optimizer, "get_step_size"):
            self.alpha = self.optimizer.get_step_size()

        if hessian is not None:
            grad_cov = self.bound_handler.hess_from_unit_cube(hessian)
            # Momentum on the covariance step, as for the state step (beta is
            # documented as the momentum parameter). `beta * self.cov` here
            # shrank the covariance by (1 - beta) every accepted step whatever
            # the gradient said.
            self.cov_step = self.alpha_cov * grad_cov + self.beta * self.cov_step
            self.cov = ot.get_sym_pos_semidef(self.cov - self.cov_step)

        if self.xk.size == 1 and hasattr(self.optimizer, "step_size"):
            self.optimizer.step_size /= 2

        self.optimizer.restore_parameters()

    def _build_optimizer(self, optimizer_name):
        if optimizer_name not in self.VALID_OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{optimizer_name}' not recognized for EnOpt. "
                f"Valid options are: {self.VALID_OPTIMIZERS}."
            )

        if optimizer_name == "GD":
            return opt.GradientDescent(self.alpha, self.beta)
        if optimizer_name == "Adam":
            return opt.Adam(self.alpha, self.beta)
        if optimizer_name == "AdaMax":
            self.normalize = False
            return opt.AdaMax(self.alpha, self.beta)
        return opt.Steihaug(delta0=3.0)

    def _apply_optimizer_backtracking(self, shrink=0.5):
        # Every step rule takes the factor. The TypeError fallback that used
        # to sit here halved Adam, AdaMax and Steihaug on the pre-trial call
        # with shrink = 1.0, i.e. before the first attempt of every iteration.
        self.optimizer.apply_backtracking(shrink)

    def _get_restart_state(self) -> dict:
        return {
            "cov": self.cov,
            "state_step": self.state_step,
            "cov_step": self.cov_step,
            "alpha": self.alpha,
            "alpha_iter": self.alpha_iter,
            "obj_func_tol": self.obj_func_tol,
            "optimizer_name": self.optimizer_name,
            "optimizer_state": dict(self.optimizer.__dict__),
        }

    def _set_restart_state(self, state: dict) -> None:
        self.cov = state.get("cov", self.cov)
        self.state_step = state.get("state_step", self.state_step)
        self.cov_step = state.get("cov_step", self.cov_step)
        self.alpha = state.get("alpha", self.alpha)
        self.alpha_iter = state.get("alpha_iter", self.alpha_iter)
        self.obj_func_tol = state.get("obj_func_tol", self.obj_func_tol)

        self.optimizer_name = state.get("optimizer_name", self.optimizer_name)
        self.optimizer = self._build_optimizer(self.optimizer_name)
        self.optimizer.__dict__.update(state.get("optimizer_state", {}))

    def log_columns(self) -> dict:
        """The row of the iteration log: iteration, backtracking attempts, objective, step size, first covariance entry."""
        return {
            "iter.": self.iteration,
            "alpha_iter": self.alpha_iter,
            "obj_func": float(np.mean(self.fk)),
            "step-size": self.alpha,
            "cov[0,0]": float(self.cov[0, 0]),
        }
