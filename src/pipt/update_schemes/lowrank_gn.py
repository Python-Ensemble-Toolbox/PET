"""Low-rank Gauss-Newton, with a Laplace approximation of the posterior."""

import numpy as np

import pipt.misc_tools.extract_tools as extract
from pipt.update_schemes.core import AssimilationScheme, StepReport, restart_options
from pipt.update_schemes.laplace_prior import PriorBlock, apply_cov, apply_root, build_prior, prior_cost

__all__ = ["LowRankGN"]

#: Options this scheme cannot honour, and so refuses rather than ignores.
UNSUPPORTED_DATAASSIM = ("compress", "scale_data", "scale", "emp_cov", "localization", "localanalysis",
                         "remove_outliers", "post_process_forecast")


def data_basis(anomalies, energy, max_rank=None):
    """The leading left singular vectors of ``anomalies``: as many as hold ``energy`` of them, at most ``max_rank``."""
    U, s, _ = np.linalg.svd(anomalies, full_matrices=False)
    keep = s > 1e-10 * s[0]
    U, s = U[:, keep], s[keep]
    p = int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), energy)) + 1
    return U[:, :p if max_rank is None else min(p, max_rank)]


def damped_step(v, A, W, gram, lam):
    """Solve ((1 + λ) Q + AᵀA) step = −g, given v = B g and W = B Aᵀ."""
    alpha = 1 + lam
    return (W @ np.linalg.solve(gram + alpha * np.eye(len(gram)), A @ v) - v) / alpha


def laplace_perturbations(A, W, gram, prior_draws, data_noise):
    """Condition centred prior draws with independent standard-normal data noise."""
    return prior_draws - W @ np.linalg.solve(gram + np.eye(len(gram)), A @ prior_draws + data_noise)


class LowRankGN(AssimilationScheme):
    r"""Gauss-Newton on the posterior mode, in a low-rank data subspace, then its Laplace approximation.

    Where the ensemble schemes move every member, this one moves a single
    point, the *centre*, to the maximum of the posterior, and only then draws
    an ensemble: ``posterior_samples`` draws (default ``ne``) of the Gaussian that the Gauss-Newton Hessian
    describes about that point. The prior is a Gaussian of full rank, so the
    update is not confined to the span of an ensemble and needs no
    localization.

    The centre stays in physical coordinates. The objective is the sum of
    the prior cost and half the noise-weighted data misfit. The prior
    (``dataassim.laplace``) applies its covariance B without assembling it.

    The ensemble is simulated twice only: the prior, with the prior mean as
    its first member, and the samples at the end. The whitened anomalies of a
    set of ``ne`` predictions give a basis :math:`U` of the data directions the
    model responds in, truncated at ``energy`` and ``max_rank``. That set
    starts as the prior's predictions, and each accepted iteration puts the
    prediction of the new centre in place of a random one of them, so the basis
    follows the centre at no further simulation. The simulator's adjoint gives
    the sensitivity in the directions of the basis only,
    giving the noise-scaled physical Jacobian rows A: one backward
    sweep per direction, plus one for the gradient, rather than one per datum.
    The step uses W = B Aᵀ and solves a system of size rank with matrix
    (1 + λ) I + AW. It uses the full gradient even for a truncated basis
    and is kept if the objective falls, which takes one simulation, of the
    trial centre, to tell; :math:`\lambda` shrinks on an accepted step and
    grows, for another attempt from the same centre, on a rejected one. The
    samples use undamped curvature at the final centre.

    Parameters
    ----------
    keys_da : dict
        Parsed ``dataassim`` configuration, with ``scheme = "lowrankgn"`` and
        ``analysis = "laplace"``.
    keys_en : dict
        Parsed ``ensemble`` configuration. The prior ensemble may be drawn
        from the ``prior_<name>`` blocks or imported (``importstate``).
    sim : object
        Forward simulator with the optional ``run_vjp`` hook, ref
        :class:`ensemble.protocols.ForwardSimulator`.
    ensemble : optional
        A ready-made collaborator to run on instead of building one.

    Notes
    -----
    Configured through the ``iteration`` block of ``keys_da``:

    ``max_iter``
        Maximum accepted iterations.
    ``energy``
        Share of the prediction anomalies the basis keeps (default 0.95).
    ``max_rank``
        Upper bound on the size of the basis, hence on the backward sweeps of
        an iteration (default: none).
    ``full_jacobian``
        Do without the basis: take the sensitivity of every datum, at one
        backward sweep each, which makes the Hessian the full Gauss-Newton one
        (default: false). ``energy`` and ``max_rank`` then have no effect. For
        when the data are few enough to afford, or more than the ``ne - 1``
        directions a basis can hold.
    ``lambda``, ``lambda_factor``, ``lambda_min``
        Initial damping (default 10), the factor it changes by (default 10) and
        its lower bound (default 0.01).
    ``max_inner_iter``
        Attempts one iteration may make before the run gives up (default 10).
    ``data_misfit_tol``
        Relative change of the centre's data misfit treated as converged, as
        in LM-EnRML (default 0.01).

    The "Data Misfit" of the run table and of the result is the centre's.
    """

    RESTART_ATTRIBUTES = ("centre", "lam", "_converged", "centre_pred", "basis_pred", "mu", "prior_blocks")

    def __init__(self, keys_da, keys_en, sim, ensemble=None):
        ensemble = self.build_ensemble(keys_da, keys_en, sim, ensemble)
        # Zero tolerances switch off the base class's criteria; check_convergence() decides.
        super().__init__(ensemble, misfit_tol=0.0, step_tol=0.0, **restart_options(ensemble.keys_da))

        unsupported = [key for key in UNSUPPORTED_DATAASSIM if self.keys_da.get(key)]
        unsupported += [key for key in ("multilevel",) if key in ensemble.keys_en]
        if unsupported:
            raise NotImplementedError(f"lowrankgn does not support the option(s) {unsupported}.")
        if not hasattr(self.sim, "run_vjp"):
            raise TypeError(
                f"lowrankgn needs a simulator with a run_vjp(state, weights) method; "
                f"{type(self.sim).__name__} has none."
            )

        options = self.keys_da.get("iteration", {})
        self.maxiter = extract.extract_maxiter(self.keys_da)
        self.trunc_energy = options.get("energy", 0.95)
        if self.trunc_energy > 1:  # given as a percentage
            self.trunc_energy /= 100.0
        self.max_rank = options.get("max_rank", None)
        self.full_jacobian = extract.is_enabled(options.get("full_jacobian", False))
        self.lam = options.get("lambda", 10.0)
        self.lam_factor = options.get("lambda_factor", 10.0)
        self.lam_min = options.get("lambda_min", 0.01)
        self.max_inner_iter = options.get("max_inner_iter", 10)
        self.data_misfit_tol = options.get("data_misfit_tol", 0.01)
        self._converged = False
        self._linearization = None  # of the committed centre; None once the centre has moved
        self.centre_pred = None  # (nd,) prediction of the committed centre
        self.basis_pred = None  # (nd, ne) predictions whose anomalies span the data basis

        self.ensemble.check_assimindex_simultaneous()

        invalid = ~np.isfinite(self.obs_variance) | (self.obs_variance <= 0)
        if invalid.any():
            cells = [(r.label, r.datatype) for r in self.data_layout.rows if invalid[r.rows].any()]
            raise ValueError(f"lowrankgn needs finite, positive observation variance at {cells}.")
        self.ensemble.enX = np.asarray(self.enX, dtype=float)
        if self.enX.shape[1] != self.ne:
            raise ValueError(f"Expected {self.ne} prior members, got {self.enX.shape[1]}")
        laplace = self.keys_da.get("laplace", {})
        self.posterior_samples = laplace.get("posterior_samples", self.ne)
        if type(self.posterior_samples) is not int or self.posterior_samples < 1:
            raise ValueError("posterior_samples must be a positive integer")
        self.mu, self.prior_blocks = build_prior(
            self.enX, self.state_layout, self.prior_info, laplace, "importstate" in ensemble.keys_en,
            not ensemble.disable_tqdm,
        )
        self.limits = {key: self.prior_info[key].get("limits", (None, None)) for key in self.idX}

        # The centre starts at the prior mean, in place of the first member of the prior draw
        self.centre = self._bounded(self.mu)
        self.ensemble.enX[:, 0] = self.centre
        self.ensemble.prior_enX = self.enX.copy()

    # ------------------------------------------------------------------
    # The pieces of an iteration
    # ------------------------------------------------------------------
    def _bounded(self, x):
        """Project the state onto the configured physical limits."""
        states = x.copy()
        self.state_layout.clip(states, self.limits)
        return states

    def score(self, pred_data=None):
        """The centre's data misfit, against the observations as they are: nothing here perturbs them."""
        pred = self._as_matrix(self.pred_data if pred_data is None else pred_data)
        return np.sum((pred[:, :1] - self.obs_vector[:, None]) ** 2 / self.obs_variance[:, None], axis=0)

    def _misfit(self, pred):
        """The data misfit of the ``(nd,)`` prediction ``pred``."""
        return np.sum((pred - self.obs_vector) ** 2 / self.obs_variance)

    def _simulate(self, x):
        """The forecast of the single trial, or ``None`` if its simulation failed."""
        success = self.ensemble.calc_prediction(x[:, None], member_count=1)
        return self.ensemble._predicted_data() if success else None

    def _linearize(self):
        """Cache the covariance-weighted full gradient and data-space curvature for all damping trials."""
        sd = np.sqrt(self.obs_variance)
        residual = (self.centre_pred - self.obs_vector) / sd
        if self.full_jacobian:
            # One product per datum: the whole Jacobian, of which the gradient is a combination
            weights = np.diag(1 / sd)
        else:
            anomalies = (self.basis_pred - self.basis_pred.mean(axis=1, keepdims=True)) / sd[:, None]
            basis = data_basis(anomalies, self.trunc_energy, self.max_rank)
            # One product for the gradient of the data misfit, then one per direction of the basis
            weights = np.column_stack([residual, basis]) / sd[:, None]
        centre = self.state_layout.member_dicts(self.centre)[0]
        products = self.sim.run_vjp(centre, self.data_layout.to_frame(weights))
        missing = [name for name in self.state_layout.variables if name not in products]
        if missing:
            raise NotImplementedError(f"The simulator's run_vjp gives no sensitivity to {missing}.")
        rows = np.hstack([np.atleast_2d(products[name]) for name in self.state_layout.variables])
        weighted = apply_cov(self.prior_blocks, rows.T)
        if self.full_jacobian:
            A, W, data_term = rows, weighted, weighted @ residual
        else:
            A, W, data_term = rows[1:], weighted[:, 1:], weighted[:, 0]
        gram = A @ W
        v = (self.centre - self.mu) + data_term  # B Q (x − μ) = x − μ
        if not (np.isfinite(v).all() and np.isfinite(gram).all()):
            raise ValueError(f"Nonfinite Laplace gradient or curvature at MAP iteration {self.iteration}.")
        return v, A, W, 0.5 * (gram + gram.T)

    # ------------------------------------------------------------------
    # The scheme contract
    # ------------------------------------------------------------------
    def _set_restart_state(self, state):
        state = state.copy()
        if "C" in state and "prior_blocks" not in state:
            c = state.pop("C")
            state["prior_blocks"] = [PriorBlock(slice(0, c.shape[0]), c, precision=True)]
        if "xi" in state:
            state["centre"] = state["mu"] + apply_root(state["prior_blocks"], state.pop("xi"))
        if "centre" in state:
            state["centre"] = self._bounded(state["centre"])
        self._linearization = None
        super()._set_restart_state(state)

    def after_prior_forecast(self) -> None:
        """Keep the prior's predictions: the centre's, and all of them as the first basis."""
        if not np.array_equal(self.enX[:, 0], self.centre):  # replaced by another member's
            raise RuntimeError("The simulation of the prior mean failed.")
        self.basis_pred = self.pred_data.matrix.copy()
        self.centre_pred = self.basis_pred[:, 0].copy()
        super().after_prior_forecast()

    def update_step(self) -> StepReport:
        """One Gauss-Newton iteration: re-damp from the same centre until the objective falls."""
        if self._linearization is None:
            self._linearization = self._linearize()
        v, A, W, gram = self._linearization
        misfit = self._misfit(self.centre_pred)
        objective = prior_cost(self.prior_blocks, self.centre - self.mu) + 0.5 * misfit
        self.rank = len(gram)
        self.prev_data_misfit_mean = self.data_misfit_mean

        centre_output = self.ensemble.member_outputs[0][0]
        valid_trial = False
        for _ in range(self.max_inner_iter):
            self.lam_used = self.lam
            x = self.centre + damped_step(v, A, W, gram, self.lam)
            if np.isfinite(x).all():
                x = self._bounded(x)
            if np.array_equal(x, self.centre):
                # A blocked GN direction alone is not stationarity for a correlated prior.
                grad = np.concatenate([b.precision_product(v[b.rows]) for b in self.prior_blocks])
                if np.array_equal(self._bounded(self.centre - grad), self.centre):
                    self._converged = True
                    self.conv_msg = "Stationary at the feasible centre"
                    break
                x = self._bounded(self.centre - grad / (1 + self.lam))
            forecast = self._simulate(x) if np.isfinite(x).all() else None
            trial_misfit = np.inf if forecast is None else self._misfit(forecast.matrix[:, 0])
            trial_objective = np.inf
            if np.isfinite(trial_misfit):
                valid_trial = True
                trial_objective = prior_cost(self.prior_blocks, x - self.mu) + 0.5 * trial_misfit

            if trial_objective < objective:
                self.centre = x
                self.centre_pred = forecast.matrix[:, 0]
                # The basis follows the centre: its prediction takes the place of a random one
                self.basis_pred[:, self.ensemble.rng.choice(self.basis_pred.shape[1])] = self.centre_pred
                self.ensemble.pred_data = forecast
                self._linearization = None
                self.lam = max(self.lam / self.lam_factor, self.lam_min)
                if misfit > 0 and abs(1 - trial_misfit / misfit) < self.data_misfit_tol:
                    self._converged = True
                    self.conv_msg = f"Data misfit change satisfies |1 - d/d_prev| < {self.data_misfit_tol}"
                return StepReport(accepted=True, state=x[:, None], misfit=self.score(forecast))
            self.lam = max(self.lam * self.lam_factor, self.lam_min)
        else:
            self.why_stop["inner_stop"] = True
            self.conv_msg = (f"No step lowered the objective in {self.max_inner_iter} attempts" if valid_trial
                             else f"All {self.max_inner_iter} trial forecasts failed or were nonfinite")
            self.why_stop["all_trials_failed"] = not valid_trial
            if not valid_trial:
                self.ensemble.member_outputs = [[centre_output]]
                self.ensemble.pred_data = self.ensemble._predicted_data()
        return StepReport(accepted=False, state=self.enX, misfit=self.ensemble_misfit)

    def check_convergence(self) -> bool:
        return self._converged

    def log_columns(self, prior_run: bool = False) -> dict:
        return {"λ": "" if prior_run else self.lam_used, "rank": "" if prior_run else self.rank}

    def after_loop(self, converged: bool) -> None:
        """Replace the centre by the requested posterior draws, and forecast them."""
        if self.why_stop.get("all_trials_failed"):
            self.ensemble.enX = self.centre[:, None]
            if self._saving_enabled:
                self._save_stop_reason(converged)
            return
        if self._linearization is None:
            self._linearization = self._linearize()
        _, A, W, gram = self._linearization
        rng = self.ensemble.rng
        count = self.posterior_samples
        draws = apply_root(self.prior_blocks, rng.randn(self.mu.size, count))
        samples = self._bounded(self.centre[:, None] + laplace_perturbations(
            A, W, gram, draws, rng.randn(len(gram), count)))
        self.ensemble.forecast(samples, member_count=count)
        self.ensemble.enX = self.after_forecast(samples)
        super().after_loop(converged)
