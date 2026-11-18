"""Ensemble-based sequential quadratic programming (EnSQP) for constrained optimization."""

import numpy as np

from popt.optimization_methods.optimizer_base import OptimizerBase, StepReport
from popt.optimization_methods.subroutines.qp import solve_qp

__all__ = ["EnSQP"]

CONSTRAINT_KEYS = ("c_eq", "c_iq")


class EnSQP(OptimizerBase):
    r"""Ensemble-based sequential quadratic programming.

    Solves

    $$ \min_x f(x) \quad \text{s.t.} \quad c_{eq}(x) = 0, \quad c_{iq}(x) \geq 0, \quad lb \leq x \leq ub $$

    with every derivative estimated from one ensemble of perturbed controls per iteration:
    the gradient of the objective, the Jacobian of the constraints and, optionally, the
    Hessian of the Lagrangian $L = f - \lambda_{eq}^T c_{eq} - \lambda_{iq}^T c_{iq}$.
    Each iteration draws the ensemble, solves a quadratic program for the step and the
    multipliers, and backtracks on the $\ell_1$ merit function
    $f + \mu (\|c_{eq}\|_1 + \|\min(c_{iq}, 0)\|_1)$. The simulations per iteration are those
    of EnOpt: one ensemble plus the trial points of the line search.

    With fewer ensemble members than controls the derivatives are known only along the
    perturbations. The step is then sought in the subspace they span, where the linear model
    of objective and constraints is reliable, and the Hessian is updated with the curvature
    measured along each accepted step.

    The objective returns the constraint values with the objective value, from the same
    simulation: ``{'f': ..., 'c_eq': ..., 'c_iq': ...}``. With a popt ensemble::

        EnSQP.minimize(ens.get_state(), ens.function_and_constraints, jac=ens.gradient,
                       con_jac=ens.constraint_jacobian, hess=ens.lagrangian_hessian,
                       args=(ens.get_cov(),), bounds=ens.get_bounds(), **options)
    """

    NAME = "EnSQP"
    VALID_HESSIANS = ("bfgs", "ensemble")

    def __init__(self, x0, fun, jac=None, hess=None, args=(), bounds=None, callback=None, con_jac=None, **options):
        """Initialize an EnSQP optimizer instance.

        Parameters
        ----------
        x0 : ndarray
            Initial control vector.
        fun : callable
            Objective and constraints at a control vector: a dict with ``'f'`` and, for the
            kinds of constraint the problem has, ``'c_eq'`` and ``'c_iq'`` (one value per
            constraint, or one column per geological model). An inequality is satisfied when
            ``c_iq >= 0``. A bare objective value means no constraints other than the bounds.
        jac : callable
            Ensemble gradient of the objective.
        hess : callable, optional
            Ensemble Hessian of the Lagrangian, taking the multipliers as the keyword
            ``multipliers``. Needed with ``hessian='ensemble'``.
        args : tuple, optional
            The first element is the covariance the ensemble is drawn with. Leave empty for
            callables that take the control vector only.
        bounds : sequence, optional
            Lower and upper bounds for each control.
        callback : callable, optional
            Called with the optimizer after every accepted step.
        con_jac : callable, optional
            Ensemble Jacobian of the constraints, a dict with ``'c_eq'`` and/or ``'c_iq'``
            of shape (number of constraints, number of controls), computed from the ensemble
            of the last ``jac`` call. Needed when ``fun`` returns constraints.
        **options
            EnSQP configuration, plus everything :class:`OptimizerBase` takes except ``epf``.
            - hessian: ``'bfgs'`` (default) builds the Hessian of the Lagrangian from the changes in the ensemble
              gradients between iterations, with Powell's damping; ``'ensemble'`` takes it from ``hess`` on every
              iteration's ensemble and makes it positive definite. The sampling error of an ensemble Hessian is
              as large as the Hessian itself until the members far outnumber the controls, so ``'ensemble'`` is
              for few controls or large ensembles. With fewer members than controls (and ``scaling='sample'``)
              neither is possible, and the Hessian is built from the curvature along the accepted steps.
            - scaling: How an ensemble (natural) gradient or Jacobian, a covariance times the derivative, is turned
              into the derivative. ``'sample'`` (default) divides by the covariance of the perturbations as sampled,
              which is exact for linear functions, and with fewer members than controls looks for the step in
              the subspace the perturbations span. ``'cov'`` divides by the covariance they were drawn from, as
              EnOpt's averaged gradient does. ``'sample'`` needs the ensemble's ``sample_covariance``.
            - hess_min_eig: Smallest eigenvalue allowed in the ensemble Hessian, as a fraction of its largest
              (default: 1e-4). Eigenvalues are replaced by their absolute value and raised to this.
            - ctol: Largest constraint violation accepted at convergence (default: 1e-6).
            - max_step: Largest change of any control in one iteration, in the coordinates the optimizer works in
              (the unit cube with ``transform``). Default: no limit beyond the bounds.
            - alpha_maxiter: Maximum number of step halvings per iteration (default: 10).
            - armijo_c1: Fraction of the predicted decrease of the merit function a step must achieve (default: 1e-4).
            - resample: Number of new ensembles, with a smaller covariance, to try when no step is accepted (default: 0).
            - cov_factor: Covariance shrink factor applied on each resampling (default: 0.5).
            - mu0: Initial penalty parameter of the merit function (default: 0). It only grows, staying above the
              largest multiplier.
            - natural_gradient: Whether ``jac``, ``con_jac`` and ``hess`` return ensemble (natural) estimates, i.e.
              multiplied by the covariance, which are then divided by it. Default: the ``natural_gradient`` setting
              of the ensemble that ``jac`` is bound to, else False.
        """
        if jac is None:
            raise ValueError("EnSQP requires a gradient callable (jac).")
        if options.get("epf"):
            raise ValueError("EnSQP handles the constraints itself; do not combine it with the epf option.")

        # As in EnOpt: args stay empty for the wrapped callables, the covariance is passed on each call.
        super().__init__(x0, fun, jac=jac, hess=hess, args=(), bounds=bounds, callback=callback, **options)

        # Derivatives come back in physical coordinates; `_derivatives` converts them, because
        # an ensemble (natural) estimate has to be divided by the covariance first.
        self.jac = self._wrap_callable(jac, "jac")
        self.hess = self._wrap_callable(hess, "hess")
        self.con_jac = self._wrap_callable(con_jac, "con_jac")
        self._pass_reference = hasattr(getattr(jac, "__self__", None), "_set_reference")
        self._sample_covariance = getattr(getattr(jac, "__self__", None), "sample_covariance", None)
        self.scaling = str(options.get("scaling", "sample")).lower()
        if self.scaling not in ("sample", "cov"):
            raise ValueError(f"scaling '{self.scaling}' not recognized. Valid options are: ('sample', 'cov').")

        self.cov = np.asarray(args[0], dtype=float) if args else None
        self.natural_gradient = options.get("natural_gradient", self._ensemble_uses_natural_gradient(jac))
        if self.natural_gradient and self.cov is None:
            raise ValueError("Natural (ensemble) derivatives need the covariance as args[0].")

        # EnSQP controls
        self.hessian_method = str(options.get("hessian", "bfgs")).lower()
        if self.hessian_method not in self.VALID_HESSIANS:
            raise ValueError(f"hessian '{self.hessian_method}' not recognized. Valid options are: {self.VALID_HESSIANS}.")
        if self.hessian_method == "ensemble" and self.hess is None:
            raise ValueError("hessian='ensemble' needs the hess callable (the ensemble's lagrangian_hessian).")
        self.hess_min_eig = options.get("hess_min_eig", 1e-4)
        self.ctol = options.get("ctol", 1e-6)
        self.max_step = options.get("max_step", None)
        self.alpha_iter_max = options.get("alpha_maxiter", 10)
        self.armijo_c1 = options.get("armijo_c1", 1e-4)
        self.max_resample = options.get("resample", 0)
        self.cov_factor = options.get("cov_factor", 0.5)
        self.mu = float(options.get("mu0", 0.0))

        # Dynamic EnSQP state. The constraints come with the objective, so a `fun0` handed in
        # without them is not used.
        self.fk = None
        self.ck = None          # constraint values at xk, {'c_eq': (m_eq,), 'c_iq': (m_iq,)}
        self.ck_raw = None      # the same per geological model, as `fun` returned them
        self.lam = None         # multipliers, same keys as ck
        self.Ak = None          # constraint Jacobian at xk, in the optimizer's coordinates
        self._previous = None   # (gradient, Jacobian) of the iterate before xk, for the BFGS update
        self._first_update = True
        self._basis = None      # basis of the subspace the last ensemble spans, when that is not the whole space
        self._warned_small_ensemble = False
        self._small_function_changes = 0
        self._small_state_changes = 0
        self.alpha = 1.0
        self.alpha_iter = 0
        self.step_norm = np.nan
        self.kkt_norm = np.nan

    @staticmethod
    def _ensemble_uses_natural_gradient(jac):
        """The ``natural_gradient`` option of the ensemble ``jac`` is a method of (default True), or False."""
        options = getattr(getattr(jac, "__self__", None), "keys_en", None)
        if isinstance(options, dict):
            return bool(options.get("natural_gradient", True))
        return False

    # ==========================================
    # Evaluation
    # ==========================================
    def _evaluate(self, x):
        """Objective (as returned), constraints (averaged over the models) and constraints (as returned) at ``x``."""
        values = self.fun(x)
        if not isinstance(values, dict):
            return values, {}, {}
        raw = {key: np.asarray(values[key], dtype=float) for key in CONSTRAINT_KEYS if values.get(key) is not None}
        mean = {key: (c.mean(axis=1) if c.ndim == 2 else np.atleast_1d(c)) for key, c in raw.items()}
        return values["f"], mean, raw

    def _objective_value(self, x):
        f, self.ck, self.ck_raw = self._evaluate(x)
        if self.ck and self.con_jac is None:
            raise ValueError("The objective returns constraints; EnSQP needs their Jacobian (con_jac).")
        self.lam = {key: np.zeros(c.size) for key, c in self.ck.items()}
        return f

    def _evaluate_missing_derivatives(self):
        # The ensemble derivatives take the covariance and are evaluated inside every step.
        pass

    def _derivatives(self, shrink):
        """Gradient and constraint Jacobian at ``xk``, in the optimizer's coordinates, from a new ensemble."""
        cov = None if self.cov is None else shrink * self.cov
        cov_args = () if cov is None else (cov,)
        kwargs = {}
        if self._pass_reference:
            # The ensemble compares its members with the latest single evaluation, which
            # after a failed line search is a rejected trial point, not the current iterate.
            kwargs["reference"] = {"f": self.fk, **self.ck_raw}

        gradient = np.asarray(self.jac(self.xk, *cov_args, **kwargs), dtype=float)
        jacobian = self.con_jac(self.xk, *cov_args) if (self.ck and self.con_jac is not None) else {}
        missing = set(self.ck) - set(jacobian)
        if missing:
            raise ValueError(f"con_jac returned no Jacobian for {sorted(missing)}.")

        to_derivative = self._derivative_scaling(cov)
        gradient = self.bound_handler.jac_to_unit_cube(to_derivative(gradient))

        A = {}
        for key in self.ck:
            rows = np.atleast_2d(np.asarray(jacobian[key], dtype=float))
            A[key] = self.bound_handler.jac_to_unit_cube(to_derivative(rows.T).T)
        return gradient, A, to_derivative, cov_args

    def _derivative_scaling(self, cov):
        """The map from what the ensemble returned to a derivative estimate, for the ensemble just drawn.

        An ensemble (natural) gradient is a covariance times the gradient. With
        ``scaling='cov'`` it is divided by the covariance the perturbations were drawn from,
        as EnOpt's averaged gradient is. With ``scaling='sample'`` it is divided by the
        covariance of the perturbations as sampled, which makes the estimate exact for a
        linear function (a linear constraint gets its exact Jacobian) and, with fewer
        members than controls, confines it to the subspace the ensemble spans.
        """
        self._basis = None
        if self.scaling == "sample" and self._sample_covariance is not None:
            sample = self._sample_covariance()
            w, V = np.linalg.eigh(0.5 * (sample + sample.T))
            spanned = w > 1e-10 * w.max()
            inverse = (V[:, spanned] / w[spanned]) @ V[:, spanned].T
            if not spanned.all():
                # Fewer members than controls: the derivatives are known along the
                # perturbations only, so the step is sought in the subspace they span. In
                # unit-cube coordinates a perturbation is divided by the bound range.
                directions = V[:, spanned]
                if self.transform and self.bound_handler.bounds is not None:
                    directions = directions / self.bound_handler.db[:, None]
                self._basis = np.linalg.qr(directions)[0]
            if self.natural_gradient:
                return lambda v: inverse @ v
            return lambda v: inverse @ (cov @ v)
        if self.natural_gradient:
            return lambda v: np.linalg.solve(cov, v)
        return lambda v: v

    # ==========================================
    # Hessian of the Lagrangian
    # ==========================================
    def _lagrangian_gradient(self, gradient, A):
        for key, rows in A.items():
            gradient = gradient - rows.T @ self.lam[key]
        return gradient

    def _bfgs_hessian(self, gradient, A):
        """Powell-damped BFGS update of the Hessian with the step just taken and the new ensemble derivatives."""
        if self.hk is None:
            self.hk = np.eye(self.xk.size)
        if self._previous is None or self.xk_old is None:
            return self.hk

        B = self.hk
        s = self.xk - self.xk_old
        y = self._lagrangian_gradient(gradient, A) - self._lagrangian_gradient(*self._previous)
        self._previous = None  # one update per accepted step, also when the ensemble is redrawn

        sy = s @ y
        if self._first_update and sy > 1e-12:
            # Scale the initial Hessian to the curvature seen along the first step
            B = (y @ y) / sy * np.eye(s.size)
        self._first_update = False

        Bs = B @ s
        sBs = s @ Bs
        if sBs <= 1e-14:
            return B
        # Powell's damping keeps the update positive definite when s'y is small or negative,
        # which the noise of two ensemble gradients produces readily.
        theta = 1.0 if sy >= 0.2 * sBs else 0.8 * sBs / (sBs - sy)
        r = theta * y + (1.0 - theta) * Bs
        B = B - np.outer(Bs, Bs) / sBs + np.outer(r, r) / (s @ r)
        return 0.5 * (B + B.T)

    def _curvature_along_step(self, B, s, gradient, A, f_new, c_new):
        """Update of the Hessian for a step confined to the subspace of a small ensemble.

        Each ensemble then gives the gradient in its own subspace, and the difference of two
        such gradients, which BFGS is built on, is mostly the difference of the subspaces.
        What is known is the Lagrangian at both ends of the step and its slope along it at
        the start, hence its curvature along the step. The update sets the Hessian's
        curvature along the step to that (BFGS with ``y`` parallel to ``B s``), within a
        factor of ten of what it was, and leaves it positive definite.
        """
        Bs = B @ s
        sBs = s @ Bs
        if sBs <= 1e-14:
            return B
        change = float(np.mean(f_new)) - float(np.mean(self.fk))
        slope = gradient @ s
        for key in self.ck:
            change -= self.lam[key] @ (c_new[key] - self.ck[key])
            slope -= self.lam[key] @ (A[key] @ s)
        curvature = 2.0 * (change - slope)

        if self._first_update and curvature > 1e-12:
            # Scale the initial Hessian to the curvature seen along the first step
            self._first_update = False
            return curvature / (s @ s) * np.eye(s.size)
        self._first_update = False
        ratio = np.clip(curvature / sBs, 0.1, 10.0)
        return B + (ratio - 1.0) * np.outer(Bs, Bs) / sBs

    def _ensemble_hessian(self, cov_args):
        """Ensemble Hessian of the Lagrangian on the current ensemble, made positive definite."""
        H = np.asarray(self.hess(self.xk, *cov_args, multipliers=dict(self.lam)), dtype=float)
        # Always scaled by the covariance the perturbations were drawn from: the ensemble
        # Hessian subtracts that covariance times the mean change, so only then is it consistent.
        if self.natural_gradient:
            cov = cov_args[0]
            H = np.linalg.solve(cov, np.linalg.solve(cov, H).T).T
        H = self.bound_handler.hess_to_unit_cube(H)

        # The estimate is noisy and the Lagrangian need not be convex: reflect negative
        # eigenvalues and lift small ones, so the QP is strictly convex and no direction
        # gets an unbounded step.
        w, V = np.linalg.eigh(0.5 * (H + H.T))
        w = np.abs(w)
        if not np.all(np.isfinite(w)) or w.max() <= 1e-12:
            return np.eye(w.size)   # no curvature information in this ensemble
        floor = self.hess_min_eig * w.max()
        return (V * np.maximum(w, floor)) @ V.T

    # ==========================================
    # The step
    # ==========================================
    @staticmethod
    def _violation(c, norm=1):
        """Constraint violation: the l1 norm (merit function) or the largest element (convergence)."""
        parts = [np.abs(c.get("c_eq", np.zeros(0))), np.maximum(-c.get("c_iq", np.zeros(0)), 0.0)]
        v = np.concatenate([np.ravel(p) for p in parts])
        if v.size == 0:
            return 0.0
        return float(v.sum() if norm == 1 else v.max())

    def _step_bounds(self):
        """Bounds on the step: the bounds on the controls, and ``max_step``."""
        n = self.xk.size
        handler = self.bound_handler
        if handler.bounds is None:
            lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
        elif handler.transform:
            lower, upper = -self.xk, 1.0 - self.xk
        else:
            lower, upper = handler.lb - self.xk, handler.ub - self.xk
        if self.max_step is not None:
            lower, upper = np.maximum(lower, -self.max_step), np.minimum(upper, self.max_step)
        return lower, upper

    def _solve_subproblem(self, B, gradient, A_eq, c_eq, A_iq, c_iq, lower, upper):
        """The QP for the step: returns the step, the multipliers, the Hessian times the step, and the slack left.

        With a basis ``Q`` of the subspace the ensemble spans (fewer members than controls)
        the step is ``Q z`` and the QP is solved for ``z``; the bounds on the step are then
        general inequalities.
        """
        Q = self._basis
        if Q is None:
            d, lam, info = solve_qp(B, gradient, A_eq, c_eq, A_iq, c_iq, lower, upper)
            return d, lam, B @ d, info["slack"]

        has_lower, has_upper = np.isfinite(lower), np.isfinite(upper)
        A_bounds = np.vstack([Q[has_lower], -Q[has_upper]])
        b_bounds = np.concatenate([-lower[has_lower], upper[has_upper]])
        B_reduced = Q.T @ B @ Q
        z, lam, info = solve_qp(B_reduced, Q.T @ gradient, A_eq @ Q, c_eq,
                                np.vstack([A_iq @ Q, A_bounds]), np.concatenate([c_iq, b_bounds]))
        lam = {"c_eq": lam["c_eq"], "c_iq": lam["c_iq"][:c_iq.size]}
        return np.clip(Q @ z, lower, upper), lam, B_reduced @ z, info["slack"]

    def update_step(self) -> StepReport:
        """One SQP iteration: ensemble derivatives, QP, and a line search on the merit function."""
        for resampling_iter in range(self.max_resample + 1):
            gradient, A, to_derivative, cov_args = self._derivatives(self.cov_factor ** resampling_iter)
            # An ensemble with fewer members than controls holds no usable Hessian; the
            # curvature along the accepted steps is used instead (see `_curvature_along_step`).
            use_ensemble_hessian = self.hessian_method == "ensemble" and self._basis is None
            if self.hessian_method == "ensemble" and not use_ensemble_hessian and not self._warned_small_ensemble:
                self._warned_small_ensemble = True
                if self.logger:
                    self.logger("hessian='ensemble' needs more ensemble members than controls; "
                                "using the curvature along the accepted steps instead.")
            if use_ensemble_hessian:
                self.hk = self._ensemble_hessian(cov_args)
            else:
                self.hk = self._bfgs_hessian(gradient, A)
            self.jk, self.Ak = gradient, A
            B = self.hk

            n = self.xk.size
            A_eq, A_iq = A.get("c_eq", np.zeros((0, n))), A.get("c_iq", np.zeros((0, n)))
            c_eq, c_iq = self.ck.get("c_eq", np.zeros(0)), self.ck.get("c_iq", np.zeros(0))
            lower, upper = self._step_bounds()
            d, lam_qp, model_hessian_step, slack = self._solve_subproblem(
                B, gradient, A_eq, c_eq, A_iq, c_iq, lower, upper)
            if slack > 0:
                # The linearised constraints cannot all hold within the bounds on the step.
                # The QP's multipliers are then its penalty, not an estimate of the
                # Lagrange multipliers: keep the ones we have.
                lam_qp = {**lam_qp, **self.lam}

            # Converged when the step the model asks for is negligible at a feasible point.
            # g + B d is the gradient of the Lagrangian with the QP's multipliers, bounds
            # included, so |B d| is the size of the KKT residual.
            self.step_norm = float(np.linalg.norm(d, np.inf))
            self.kkt_norm = float(np.linalg.norm(model_hessian_step, np.inf))
            violation_inf = self._violation(self.ck, norm=np.inf)
            # In the subspace of a small ensemble a negligible step only says that this
            # subspace holds no descent, so there it is not taken as convergence.
            whole_space = self._basis is None
            if whole_space and violation_inf <= self.ctol and (self.kkt_norm < self.gtol or self.step_norm < self.xtol):
                self.lam = {key: lam_qp[key] for key in self.ck}
                self.iteration -= 1
                self._record_results()
                which = (f"KKT residual ‖∇L‖∞ < {self.gtol}" if self.kkt_norm < self.gtol
                         else f"SQP step ‖d‖∞ < {self.xtol}")
                return StepReport(False, f"{which}, constraint violation <= {self.ctol}.")

            # Merit function: the penalty parameter stays above the multipliers, and large
            # enough that the step is a descent direction for it.
            violation = self._violation(self.ck)
            linearised = self._violation({"c_eq": A_eq @ d + c_eq, "c_iq": A_iq @ d + c_iq})
            reduction = violation - linearised
            lam_max = max([np.max(np.abs(lam_qp[key]), initial=0.0) for key in self.ck], default=0.0)
            self.mu = max(self.mu, 1.1 * lam_max)
            if reduction > 1e-8 * max(1.0, violation):
                self.mu = max(self.mu, (gradient @ d + 0.5 * d @ B @ d) / (0.7 * reduction))
            slope = gradient @ d - self.mu * reduction
            merit = float(np.mean(self.fk)) + self.mu * violation

            if slope < 0:
                self.alpha, self.alpha_iter = 1.0, 0
                while self.alpha_iter <= self.alpha_iter_max:
                    x_new = self.bound_handler.project_to_bounds(self.xk + self.alpha * d)
                    f_new, c_new, c_raw = self._evaluate(x_new)
                    merit_new = float(np.mean(f_new)) + self.mu * self._violation(c_new)
                    if np.isfinite(merit_new) and merit_new <= merit + self.armijo_c1 * self.alpha * slope:
                        self._previous = (gradient, A)
                        self.lam = {key: self.lam[key] + self.alpha * (lam_qp[key] - self.lam[key]) for key in self.ck}
                        if self._basis is not None:
                            B = self._curvature_along_step(B, x_new - self.xk, gradient, A, f_new, c_new)
                            self._previous = None
                        self._commit_step(x_new, f_new, jac=gradient, hess=B)
                        self.ck, self.ck_raw = c_new, c_raw
                        return StepReport(True)
                    self.alpha *= 0.5
                    self.alpha_iter += 1

        return StepReport(False, "EnSQP failed to find a step that reduces the merit function.")

    # ==========================================
    # Convergence
    # ==========================================
    def _feasible(self):
        return self._violation(self.ck, norm=np.inf) <= self.ctol

    def check_convergence(self) -> bool:
        # The KKT test needs the QP at the current iterate and is made in `update_step`.
        return False

    def _patience(self):
        """How many iterations in a row a small change must be seen before it counts as convergence.

        Once, normally. When the step is confined to the subspace of a small ensemble, one
        iteration can make little progress merely because its subspace held little descent,
        so it takes as many as would span the controls, and one more.
        """
        if self._basis is None:
            return 1
        return int(np.ceil(self.xk.size / self._basis.shape[1])) + 1

    def check_function_convergence(self) -> bool:
        """A small change in the objective only counts at a feasible point (and see :meth:`_patience`)."""
        small = self._feasible() and super().check_function_convergence()
        self._small_function_changes = self._small_function_changes + 1 if small else 0
        return self._small_function_changes >= self._patience()

    def check_state_convergence(self) -> bool:
        """A small step only counts at a feasible point (and see :meth:`_patience`)."""
        small = self._feasible() and super().check_state_convergence()
        self._small_state_changes = self._small_state_changes + 1 if small else 0
        return self._small_state_changes >= self._patience()

    # ==========================================
    # Results, log and restart
    # ==========================================
    def _update_optimize_result(self):
        result = super()._update_optimize_result()
        result["violation"] = self._violation(self.ck or {}, norm=np.inf)
        for key in CONSTRAINT_KEYS:
            if self.ck and key in self.ck:
                result[key] = self.ck[key]
                result[f"lam_{key[2:]}"] = self.lam[key]
        if self.Ak:
            for key, rows in self.Ak.items():
                result[f"jac_{key}"] = self.bound_handler.jac_from_unit_cube(rows)
        result["njev_con"] = self.con_jac.nfev if self.con_jac else 0
        return result

    def log_columns(self) -> dict:
        """The row of the iteration log: iteration, step halvings, objective, violation, penalty, step and KKT residual."""
        return {
            "iter.": self.iteration,
            "alpha_iter": self.alpha_iter,
            "obj_func": float(np.mean(self.fk)),
            "violation": self._violation(self.ck or {}, norm=np.inf),
            "mu": self.mu,
            "|d|": self.step_norm,
            "|kkt|": self.kkt_norm,
        }

    def _log_convergence(self):
        if self.logger:
            self.logger(f"  Final constraint violation: {self._violation(self.ck or {}, norm=np.inf):.3e}")
        super()._log_convergence()

    def _get_restart_state(self) -> dict:
        return {
            "cov": self.cov,
            "ck": self.ck,
            "ck_raw": self.ck_raw,
            "lam": self.lam,
            "Ak": self.Ak,
            "mu": self.mu,
            "previous": self._previous,
            "first_update": self._first_update,
            "alpha": self.alpha,
            "alpha_iter": self.alpha_iter,
            "njev_con": self.con_jac.nfev if self.con_jac else 0,
        }

    def _set_restart_state(self, state: dict) -> None:
        self.cov = state.get("cov", self.cov)
        self.ck = state.get("ck", self.ck)
        self.ck_raw = state.get("ck_raw", self.ck_raw)
        self.lam = state.get("lam", self.lam)
        self.Ak = state.get("Ak", self.Ak)
        self.mu = state.get("mu", self.mu)
        self._previous = state.get("previous", None)
        self._first_update = state.get("first_update", True)
        self.alpha = state.get("alpha", self.alpha)
        self.alpha_iter = state.get("alpha_iter", self.alpha_iter)
        if self.con_jac:
            self.con_jac.nfev = state.get("njev_con", 0)
