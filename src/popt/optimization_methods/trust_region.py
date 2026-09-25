"""Trust-region deterministic optimization methods.

This module implements a trust-region optimizer with optional
BFGS Hessian approximation and restart support.
"""

import numpy as np

# Internal imports
from popt.optimization_methods.optimizer_base import OptimizerBase, StepReport
from popt.optimization_methods.subroutines.subroutines import solve_trust_region_subproblem

__author__ = "Mathias Methlie Nilsen"
__all__ = ["TrustRegion"]

# Symbols for logger output
subk = "ₖ"
fun_xk_symbol = f"fun(x{subk})"
delta_k_symbol = f"Δ{subk}"
rho_symbol = f"ρ{subk}"
jac_inf_symbol = f"‖jac(x{subk})‖∞"


class TrustRegion(OptimizerBase):
    """Trust-region Optimizer.

    The class supports exact Hessian trust-region subproblems (iterative or
    CG-Steihaug) and optional BFGS Hessian approximation via ``hess='BFGS'``.
    """

    NAME = "Trust-Region"
    VALID_METHODS = ("iterative", "CG-Steihaug")

    def __init__(
        self,
        x0,
        fun,
        jac,
        hess,
        method="iterative",
        args=(),
        bounds=None,
        callback=None,
        **options,
    ):
        """Initialize a trust-region optimizer instance.

        Parameters
        ----------
        x0 : ndarray
            Initial parameter vector.
        fun : callable
            Objective function.
        jac : callable
            Gradient function.
        hess : callable or {'BFGS'}
            Hessian function, or ``'BFGS'`` to use a quasi-Newton Hessian approximation.
        method : {'iterative', 'CG-Steihaug'} or callable, optional
            Trust-region subproblem solver.
        args : tuple, optional
            Extra positional arguments passed to the wrapped callables.
        bounds : sequence, optional
            Lower and upper bounds for each state variable.
        callback : callable, optional
            Callback invoked after successful updates.
        **options
            Trust-region configuration, plus everything :class:`OptimizerBase` takes.
            - trust_radius: Initial trust-region radius (default: 1.0).
            - trust_radius_max: Maximum trust-region radius (default: ``100 * trust_radius``).
            - trust_radius_min: Minimum trust-region radius before termination (default: ``trust_radius / 1000``).
            - trust_radius_cuts: Maximum number of radius reductions before rejecting a step (default: 4).
            - rho_tol: Minimum ratio between actual and predicted reduction for step acceptance (default: 1e-6).
            - eta1: Threshold for rejecting a step (default: 0.05).
            - eta2: Threshold for increasing the trust-region radius (default: 0.5).
            - gam1: Factor used to decrease the trust-region radius (default: 0.5).
            - gam2: Factor used to increase the trust-region radius when the boundary is hit (default: 1.5).
            - resample: Whether to recompute gradient and Hessian after rejected steps (default: False).
            - convergence_criteria: Optional callable for custom convergence checks.
        """
        if jac is None:
            raise ValueError("TrustRegion requires a Jacobian (gradient) function.")

        use_bfgs = isinstance(hess, str) and hess.upper() == "BFGS"
        if (not use_bfgs) and (hess is None):
            raise ValueError("TrustRegion requires a Hessian function or hess='BFGS'.")

        super().__init__(x0, fun, jac, None if use_bfgs else hess, args, bounds, callback, **options)

        self.method = self._validate_method(method)
        self.quasi_newton = use_bfgs

        convergence_criteria = options.get("convergence_criteria", None)
        self.convergence_criteria = convergence_criteria if callable(convergence_criteria) else None

        # Trust-region controls
        self.trust_radius = options.get("trust_radius", 1.0)
        self.trust_radius_max = options.get("trust_radius_max", 100 * self.trust_radius)
        self.trust_radius_min = options.get("trust_radius_min", self.trust_radius / 1000)
        self.trust_radius_cuts = options.get("trust_radius_cuts", 4)

        # Acceptance and radius updates
        self.rho_tol = options.get("rho_tol", 1e-6)
        self.eta1 = options.get("eta1", 0.05) # Threshold for rejecting a step
        self.eta2 = options.get("eta2", 0.5)  # Threshold for increasing the trust-region radius
        self.gam1 = options.get("gam1", 0.5)  # Factor to decrease the trust-region radius when a step is rejected
        self.gam2 = options.get("gam2", 1.5)  # Factor to increase the trust-region radius when a step is accepted and hits the boundary
        self.rho = 0.0
        self.hits_boundary = None  # whether the last accepted step reached the trust-region boundary

        # Other options
        self.resample = options.get("resample", False)
        self.jk_old = None

    def update_step(self) -> StepReport:
        """Perform one trust-region step with optional radius reductions."""
        self._evaluate_missing_derivatives()
        return self._attempt_step(inner_iter=0)

    def check_convergence(self) -> bool:
        """The projected gradient, the trust-region radius, and any custom criterion."""
        if super().check_convergence():
            return True

        if self.trust_radius <= self.trust_radius_min:
            self.conv_msg = f"Trust-region radius {delta_k_symbol} <= {self.trust_radius_min}."
            return True

        if callable(self.convergence_criteria) and self.convergence_criteria(self):
            self.conv_msg = "Custom convergence criteria met."
            return True

        return False

    def _attempt_step(self, inner_iter: int) -> StepReport:
        if inner_iter > self.trust_radius_cuts:
            return StepReport(False, "Trust-region step rejected after radius cut attempts.")

        jk_proj = self.bound_handler.project_gradient(self.xk, self.jk)

        if self.quasi_newton and (self.hk is None) and (self.iteration == 1):
            sk = -jk_proj
            sk_norm = np.linalg.norm(sk, np.inf)
            if sk_norm > 0:
                sk = sk / sk_norm * self.trust_radius
            hits_boundary = True
        else:
            hk_step = self.hk
            if hk_step is None:
                hk_step = self.hess(self.xk)
                self.hk = hk_step

            if callable(self.method):
                sk, hits_boundary = self.method(
                    self.xk,
                    self.fk,
                    jk_proj,
                    hk_step,
                    self.trust_radius,
                    **self.options,
                )
            else:
                sk, hits_boundary = solve_trust_region_subproblem(
                    self.xk,
                    self.fk,
                    jk_proj,
                    hk_step,
                    self.trust_radius,
                    method=self.method,
                    **self.options,
                )

        xk_new = self.bound_handler.project_to_bounds(self.xk + sk)
        fk_new = self._objective_value(xk_new)

        df = self.fk - fk_new
        if self.quasi_newton and (self.iteration == 1) and (self.hk is None):
            dm = -np.dot(jk_proj, sk)
        else:
            hk_for_dm = self.hk
            if hk_for_dm is None:
                hk_for_dm = self.hess(self.xk)
            dm = -np.dot(jk_proj, sk) - 0.5 * np.dot(sk, hk_for_dm @ sk)

        self.rho = df / dm if dm != 0 else -np.inf

        if (self.rho > self.rho_tol) and (fk_new < self.fk):
            self._accept_step(xk_new, fk_new, sk, hits_boundary)
            return StepReport(True)

        if self.logger:
            if not (fk_new < self.fk):
                self.logger(
                    f"Function value not reduced: {fun_xk_symbol} = {fk_new:<10.4e} >= {self.fk:<10.4e}"
                )
            else:
                self.logger(
                    f"Step not successful: {rho_symbol} = {self.rho:<10.4e} < {self.rho_tol:<10.4e}"
                )

        old_radius = self.trust_radius
        self.trust_radius *= 0.25
        if self.logger:
            self.logger(
                f"Reducing {delta_k_symbol}: {old_radius:<10.4e} -> {self.trust_radius:<10.4e}"
            )

        if self.trust_radius < self.trust_radius_min:
            return StepReport(False, f"Trust-region radius {delta_k_symbol} below minimum.")

        if self.resample:
            self.jk = self.jac(self.xk)
            if not self.quasi_newton:
                self.hk = self.hess(self.xk)

        return self._attempt_step(inner_iter=inner_iter + 1)

    def _accept_step(self, xk_new, fk_new, sk, hits_boundary) -> None:
        self.jk_old = self.jk
        self._commit_step(xk_new, fk_new)
        self.jk = self.jac(self.xk)

        if self.quasi_newton:
            yk = self.jk - self.jk_old
            if self.hk is None:
                denom = np.dot(yk, sk)
                if denom > 0:
                    self.hk = np.dot(yk, yk) / denom * np.eye(self.xk.size)
                else:
                    self.hk = np.eye(self.xk.size)
            self.hk = self._bfgs_update(self.hk, sk, yk)
        else:
            self.hk = self.hess(self.xk)

        self._update_trust_radius(hits_boundary)
        self.hits_boundary = hits_boundary

    def _update_trust_radius(self, hits_boundary: bool) -> None:
        delta_old = self.trust_radius

        if (self.rho >= self.eta2) and hits_boundary:
            delta_new = min(self.gam2 * delta_old, self.trust_radius_max)
        elif self.rho < self.eta1:
            delta_new = self.gam1 * delta_old
        else:
            delta_new = delta_old

        self.trust_radius = np.clip(delta_new, self.trust_radius_min, self.trust_radius_max)

        if self.logger and (self.trust_radius != delta_old):
            d_delta = (self.trust_radius - delta_old) / delta_old * 100
            self.logger(
                f"Tr-radius {delta_k_symbol} updated: {delta_old:<10.4e} -> {self.trust_radius:<10.4e} ({d_delta:<.2f}%)"
            )

    def _objective_value(self, x) -> float:
        # The trust-region ratio needs a scalar; an ensemble objective returns one value per member.
        return float(np.mean(self.fun(x)))

    def _validate_method(self, method):
        if callable(method):
            if self.logger:
                self.logger("Using custom trust-region subproblem solver callable.")
            return method

        if not isinstance(method, str):
            raise ValueError("Method must be a string or a callable.")

        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid trust-region method '{method}'. Valid options are: {self.VALID_METHODS}."
            )

        return method

    def _bfgs_update(self, Bk, sk, yk):
        sk = sk.reshape(-1, 1)
        yk = yk.reshape(-1, 1)

        ykTsk = (yk.T @ sk).item()
        skTBksk = (sk.T @ Bk @ sk).item()
        if ykTsk <= 0 or skTBksk <= 0:
            return Bk

        term1 = np.matmul(yk, yk.T) / ykTsk
        term2 = np.matmul(np.matmul(Bk, sk), np.matmul(sk.T, Bk)) / skTBksk
        return Bk + term1 - term2

    def _get_restart_state(self) -> dict:
        return {
            "trust_radius": self.trust_radius,
            "rho": self.rho,
            "jk_old": self.jk_old,
            "quasi_newton": self.quasi_newton,
        }

    def _set_restart_state(self, state: dict) -> None:
        self.trust_radius = state.get("trust_radius", self.trust_radius)
        self.rho = state.get("rho", self.rho)
        self.jk_old = state.get("jk_old", self.jk_old)
        self.quasi_newton = state.get("quasi_newton", self.quasi_newton)

    def log_columns(self) -> dict:
        """The row of the iteration log: iteration, objective, trust radius, reduction ratio, whether the step hit the boundary."""
        columns = {
            "iter.": self.iteration,
            fun_xk_symbol: self.fk,
            delta_k_symbol: self.trust_radius,
            rho_symbol: self.rho,
        }
        if self.hits_boundary is not None:
            columns[f"‖p{subk}‖ = {delta_k_symbol}"] = "yes" if self.hits_boundary else "no"
        return columns
