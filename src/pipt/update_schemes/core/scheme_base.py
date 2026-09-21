"""The class every iterative ensemble data-assimilation scheme inherits.

This is the PIPT counterpart to
:mod:`popt.optimization_methods.optimizer_base`, and deliberately mirrors its
shape: the scheme object owns its own iteration loop, its convergence checks,
and its checkpoint/restart handling, while subclasses supply only the
algorithm-specific analysis step.

The two packages differ in what the iteration acts on. An optimizer is handed
callables (``fun``, ``jac``, ``hess``) and drives a control vector. An
assimilation scheme is handed an *ensemble* collaborator, which owns the state
realisations, the observed data, and the forward simulator.

Ensemble collaborator protocol
------------------------------
The scheme only relies on the following members, so anything satisfying them
can be substituted (a lightweight fake is used in the unit tests):

``ensemble.forecast()``
    Run the forward simulator on the current state and refresh ``pred_data``.
``ensemble.enX``
    State ensemble matrix, shape ``(nx, ne)``.
``ensemble.pred_data``
    Predicted data for the current state.
``ensemble.logger``
    A :class:`ensemble.logger.PetLogger`, a no-op :class:`ensemble.logger.NullLogger`
    (set when the ensemble's ``logit`` option is false), or ``None`` (e.g. a test
    double with no logger at all).
``ensemble.keys_da``
    The parsed ``dataassim`` config. Read at every hook, since which
    diagnostics and artifacts a run produces is a matter of configuration.
``ensemble.sim``
    The forward simulator. Only ``input_dict`` is read here, to decide whether
    QA/QC was asked for.
``ensemble._saving_enabled``
    Whether the run writes artifacts at all.

Reaching the ensemble's state
-----------------------------
A scheme reads plenty of ensemble state -- ``enX``, ``pred_data``,
``keys_da``, ``localization`` and friends -- and so do the analyses,
through the scheme. Rather than forwarding unknown attributes
at lookup time, each of those names is declared as an explicit
:class:`property` on :class:`AssimilationScheme` (see the block of
``_ensemble_attr`` / ``_own_or_ensemble_attr`` declarations below). The
scheme is therefore a *façade*: everything an analysis needs is
reachable as ``scheme.<name>``, whether the value lives on the scheme or on
its ensemble, and an analysis never has to know which.

Reads delegate; writes do not. Assigning ensemble state goes through
``self.ensemble.<name> = ...`` explicitly, because that is the object the
forecast reads back. The four names a scheme *may* legitimately compute for
itself (``cov_data``, ``scale_data``, ``proj``, ``Am``) are the exception
and have setters.

Relationship to the legacy design
---------------------------------
Historically a PIPT scheme *inherited* from ``pipt.loop.ensemble.Ensemble`` and
an external ``pipt.loop.assimilation.Assimilate`` object drove the loop. That
made every scheme simultaneously an algorithm and a data container, and made
the analysis flavour (``approx``/``full``/``subspace``) part of the class name.
Here the ensemble is a *collaborator* rather than a superclass, matching how
``OptimizerBase`` composes with its callables.
"""

import os
import pickle
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from misc.structures import PETDataFrame
from pipt.ensembles import AssimilationEnsemble
from ensemble.checkpoint import RestartMixin
from pipt.update_schemes.core.analysis_binding import AnalysisBindingMixin
from pipt.update_schemes.analysis.base import AnalysisResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # QAQC pulls in matplotlib; it is imported at runtime only inside
    # _build_qaqc, when the configuration actually asks for QA/QC.
    from pipt.misc_tools.qaqc_tools import QAQC
import pipt.misc_tools.analysis_tools as at
import pipt.misc_tools.extract_tools as extract

__all__ = ["AssimilationScheme", "AssimilationResult", "StepReport"]


def _ensemble_attr(name):
    """Read-only view of an ensemble attribute, as a real property.

    Used for the state a scheme reads but never owns. No setter: assigning
    raises ``AttributeError`` rather than quietly creating a scheme-local
    shadow that the ensemble -- and therefore the forecast -- would never
    see.
    """
    def getter(self):
        return getattr(self.ensemble, name)

    return property(getter, doc=f"``ensemble.{name}`` (owned by the ensemble).")


def _own_or_ensemble_attr(name):
    """The scheme's own value if it has set one, else the ensemble's.

    For the handful of names a scheme may legitimately recompute for itself
    (see the block where these are declared). Assigning stores on the
    scheme; reads fall through to the ensemble until it does.
    """
    slot = f"_own_{name}"

    def getter(self):
        try:
            return self.__dict__[slot]
        except KeyError:
            return getattr(self.ensemble, name)

    def setter(self, value):
        self.__dict__[slot] = value

    return property(
        getter,
        setter,
        doc=f"``{name}``: the scheme's own if it computed one, else the ensemble's.",
    )


@dataclass(slots=True)
class StepReport:
    """What one attempt produced. Returned by :meth:`update_step`.

    The base does not dictate how a scheme takes its step; this is what it
    needs back afterwards, to score convergence, log, and build the result.
    Required fields are positional, so forgetting one is a ``TypeError`` at
    construction rather than a ``None`` surfacing several iterations later.
    """

    accepted: bool
    """Keep this step? ``False`` says the scheme found no improving step and
    has exhausted the attempts it makes inside :meth:`update_step`, so the
    loop stops rather than asking for the same step again."""

    state: "Any"
    """The state this attempt produced, committed by the loop when
    ``accepted``. A scheme still writes it to ``ensemble.enX_temp`` first,
    because that is what the forecast predicts on -- but handing it back here
    is what lets the loop own the commit, rather than every scheme
    remembering the same two lines. Forgetting them used to give a run that
    iterated and logged normally while returning the prior untouched."""

    misfit: "np.ndarray"
    """Per-realisation data misfit **as of now**. The loop derives
    ``data_misfit`` and ``data_misfit_std`` from it, so the three can no
    longer drift apart the way separately-assigned attributes could.

    "As of now" matters for a scheme that gives up: LM-EnRML restores the last
    accepted misfit when it backs off, and returns *that*, so the value the
    loop records and logs is the one the run actually reached."""

    why_stop: dict | None = None
    """Criterion record, merged into ``result.why_stop``."""


class AssimilationResult(OptimizeResult):
    """Result of an assimilation run.

    A ``dict`` subclass with attribute access, mirroring
    :class:`scipy.optimize.OptimizeResult` so that PIPT and POPT results can be
    handled the same way. Typical fields:

    ``nit``
        Number of accepted iterations.
    ``success``
        Whether the run stopped on a convergence criterion rather than by
        exhausting ``maxiter``.
    ``message``
        Human-readable reason the run stopped.
    ``why_stop``
        Mapping of criterion name to whether it fired.
    ``data_misfit`` / ``prior_data_misfit``
        Final and initial mean data misfit.
    """


def restart_options(keys_da) -> dict:
    """The checkpoint settings of a config's ``[dataassim]`` block, as scheme options.

    ``restart`` (resume from the checkpoint), ``restartsave`` (write one after
    the prior forecast and every accepted iteration) and ``restart_file``
    (default ``<scheme>_restart.pkl``). Legacy ``yes``/``no`` strings are
    accepted. Schemes pass ``**restart_options(keys_da)`` to the base so the
    keys reach :class:`~ensemble.checkpoint.RestartMixin`; they used to stop
    at the ensemble, which loaded a pickle of itself and left the scheme's own
    state -- iteration, damping, misfit history -- at its initial values.
    """
    options = {
        "restart": extract.is_enabled(keys_da.get("restart", False)),
        "restartsave": extract.is_enabled(keys_da.get("restartsave", False)),
    }
    if "restart_file" in keys_da:
        options["restart_file"] = keys_da["restart_file"]
    elif "restartfile" in keys_da:   # a section that did not pass the config boundary
        options["restart_file"] = keys_da["restartfile"]
    return options


class AssimilationScheme(AnalysisBindingMixin, RestartMixin, ABC):
    """What every iterative ensemble data-assimilation scheme inherits.

    Subclasses implement :meth:`update_step`, which performs one iteration and
    reports what it produced. Everything else is here: the loop, convergence
    bookkeeping, restart files, the run table, the result object, and the
    diagnostics and artifact saving that surround a run.

    Those last two used to be a separate ``AssimilationWorkflowMixin`` that a
    combined class mixed in ahead of the loop. The split bought nothing --
    every shipped scheme wanted both halves -- and cost the reader two classes
    and one load-bearing MRO order, in which listing the mixin second silently
    stopped a run from saving anything.
    """

    PRIOR_FORECAST_FILE = "prior_forecast.pkl"
    POSTERIOR_STATE_FILE = "posterior_state_estimate.npz"
    POSTERIOR_FORECAST_FILE = "posterior_forecast.pkl"
    STOP_REASON_FILE = "why_iter_loop_stopped.pkl"

    qaqc: "QAQC | None" = None

    #: The ensemble a scheme builds when none is handed in. Multilevel ES-MDA
    #: overrides it with its per-level ensemble.
    ENSEMBLE_CLASS = AssimilationEnsemble

    @classmethod
    def build_ensemble(cls, keys_da, keys_en, sim, ensemble=None):
        """The collaborator to run on: ``ensemble`` if given, else a fresh ``ENSEMBLE_CLASS``.

        Handing one in lets two schemes share a prior and its forecasts, and
        lets a test substitute a stand-in without the config, data files and
        simulator a real ensemble needs.
        """
        return ensemble if ensemble is not None else cls.ENSEMBLE_CLASS(keys_da, keys_en, sim)

    def __init__(self, ensemble: AssimilationEnsemble, **options):
        """
        Parameters
        ----------
        ensemble : object
            Collaborator satisfying the ensemble protocol described in the
            module docstring. Owns the state, the observed data and the
            forward simulator.
        **options
            Scheme configuration.

            - maxiter: Maximum number of accepted iterations (default: 100).
            - misfit_tol: Relative data-misfit tolerance for convergence
              (default: 0.01). The assimilation counterpart of an optimizer's
              ``ftol``.
            - step_tol: Absolute tolerance on the norm of the state update
              (default: 1e-8). Counterpart of an optimizer's ``xtol``.
            - restart: Restore from a restart file on startup (default: False).
            - restartsave: Write a restart file after the prior forecast and
              each accepted iteration (default: False).
            - restart_file: Path for the restart file
              (default: '{scheme_name}_restart.pkl').
              Config-driven schemes take these three from the ``[dataassim]``
              block via :func:`restart_options`.
        """
        self.ensemble = ensemble
        self.options = options

        # Core iteration controls.
        self.iteration = 0
        self.maxiter = options.get("maxiter", 100)

        # Convergence tolerances.
        self.misfit_tol = options.get("misfit_tol", 0.01)
        self.step_tol = options.get("step_tol", 1e-8)

        # Restart/checkpoint controls.
        self.restart = options.get("restart", False)
        self.restartsave = options.get("restartsave", False)
        self.restart_file = options.get(
            "restart_file",
            options.get("restartfile", f"{type(self).__name__.lower()}_restart.pkl"),
        )
        self._restart_loaded = False

        # Iteration state. `data_misfit` is the assimilation analogue of an
        # optimizer's objective value; `enX` of its control vector.
        self.data_misfit_mean = None
        self.prior_data_misfit_mean = None
        self.data_misfit_std = None
        self.prev_data_misfit_mean = None
        self.enX_old = None

        # Logging. Owned by the ensemble (its logit/logger_name config
        # decides whether this is a real PetLogger or a no-op) -- adopt
        # whatever it has rather than building a separate one.
        self.logger = getattr(ensemble, "logger", None)

        # Result container and stop bookkeeping.
        self.conv_msg = ""
        self.why_stop = {}
        self.results = AssimilationResult()

        #: Whether the most recent step was accepted. Assigned by
        #: :meth:`run_assimilation` from what :meth:`update_step` returns, so
        #: it is always in step with the loop's own view. The
        #: Levenberg-Marquardt family also sets it in its scoring pass, and
        #: returns the same value.
        self.step_accepted = True

    # ------------------------------------------------------------------
    # Ensemble delegation
    # ------------------------------------------------------------------
    # Owned by the ensemble outright. No setter is deliberate: a stray
    # `self.enX = ...` raises instead of creating a shadow the forecast never
    # sees. Write ensemble state as `self.ensemble.enX = ...`.
    adjoints = _ensemble_attr("adjoints")
    data_df = _ensemble_attr("data_df")
    data_var_df = _ensemble_attr("data_var_df")
    enX = _ensemble_attr("enX")
    idX = _ensemble_attr("idX")
    keys_da = _ensemble_attr("keys_da")
    localization = _ensemble_attr("localization")
    ml_ne = _ensemble_attr("ml_ne")
    multilevel = _ensemble_attr("multilevel")
    ne = _ensemble_attr("ne")
    pred_data = _ensemble_attr("pred_data")
    data_layout = _ensemble_attr("data_layout")
    obs_vector = _ensemble_attr("obs_vector")
    obs_variance = _ensemble_attr("obs_variance")
    state_layout = _ensemble_attr("state_layout")
    prior_enX = _ensemble_attr("prior_enX")
    prior_info = _ensemble_attr("prior_info")
    save_folder = _ensemble_attr("save_folder")
    sim = _ensemble_attr("sim")
    sim_data = _ensemble_attr("sim_data")
    state = _ensemble_attr("state")
    state_scaling = _ensemble_attr("state_scaling")
    tot_level = _ensemble_attr("tot_level")
    _saving_enabled = _ensemble_attr("_saving_enabled")

    # The ensemble holds a default, but these four a scheme may compute for
    # itself, so they need setters:
    #   cov_data    EnKF rebuilds it each calc_analysis.
    #   scale_data  EnKF/ESMDA/esmda_hybrid redraw it each iteration.
    #   proj        esmda_hybrid holds one matrix *per level*, not one.
    #   Am          full_update caches it after computing it once.
    # Assigning shadows the ensemble from then on; until then reads fall
    # through.
    #
    # Do NOT make these write through. For three of them the scheme's value is
    # a different quantity that merely shares a name -- hybrid's per-level
    # `proj` list, ESMDA's alpha-inflated `scale_data` -- and `local_analysis`
    # and `perturb_observations` still read the ensemble's own version.
    Am = _own_or_ensemble_attr("Am")
    cov_data = _own_or_ensemble_attr("cov_data")
    proj = _own_or_ensemble_attr("proj")
    scale_data = _own_or_ensemble_attr("scale_data")

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @abstractmethod
    def update_step(self) -> "StepReport":
        """Perform one scheme-specific analysis step.

        Implementations compute the analysis update, apply it to the ensemble
        state, run the resulting forecast, and score the result. How they do
        that is entirely theirs -- the base calls this and nothing inside it.

        Returns
        -------
        StepReport
            ``accepted`` decides whether the loop advances or gives the scheme
            another attempt at the same iteration number, which is how the
            Levenberg-Marquardt schemes back off by increasing their damping
            parameter. ``misfit`` is the per-realisation data misfit as of now;
            the loop derives ``data_misfit`` and ``data_misfit_std`` from it.
        """

    def check_convergence(self) -> bool:
        """Check scheme-specific convergence criteria.

        Returns
        -------
        bool
            ``True`` if a subclass-specific stopping criterion is satisfied.
            The default implementation never stops the loop.
        """
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_assimilation(self) -> AssimilationResult:
        """Run this scheme's assimilation to completion.

        Named for the job rather than the mechanism, and matching the
        ``run_forecast`` already on this class. The counterpart in popt is
        ``OptimizerBase.run_optimization``.

        Restores a checkpoint if configured, forecasts and scores the prior,
        then calls :meth:`update_step` until a convergence criterion fires or
        ``maxiter`` iterations have been taken. One call is one iteration: a
        scheme that retries -- re-damping, backtracking a step length -- does
        so inside :meth:`update_step`, so a report coming back rejected means
        it has run out of attempts, and the run stops rather than asking again
        for a step it just said it could not find.

        Convergence is checked on rejected reports too, before that stop takes
        effect: a scheme's :meth:`check_convergence` can legitimately fire on
        a step it is about to reject (a stalled misfit that did not actually
        improve), and that verdict decides how the run is reported.

        Returns
        -------
        AssimilationResult
            Populated result object, also stored on ``self.results``.
        """
        if self.restart and not self._restart_loaded:
            self.load_restart()
            # Built by the prior-forecast hook on an ordinary run, which a resume skips.
            self.qaqc = self._build_qaqc()
        elif not self.restart:
            self.clear_restart()
            # The prior goes through the same post-forecast hook as every
            # later forecast, so outlier replacement applies to it too; that
            # hook can resample members, so its result is what gets committed.
            self.ensemble.enX = self.run_forecast(self.enX)
            self.record_prior_score()  # Scores through score(), below.
            self.after_prior_forecast()
            if self.restartsave:
                self.save_restart()  # the prior forecast is the expensive part of a short run

        converged = False

        while self.iteration < self.maxiter:
            # Guarded: enX is (nx, ne), so schemes that never opt in pay nothing.
            if self.step_tol > 0:
                self.enX_old = deepcopy(self.ensemble.enX)

            # Perform the scheme-specific update (in subclasses)
            step = self.update_step()
            assert isinstance(step, StepReport), (
                f"{type(self).__name__}.update_step() must return a StepReport, "
                f"not {type(step).__name__}"
            )
            self.step_accepted = step.accepted

            # Update the state ensemble. No copy: the report's state is the
            # array the scheme built and forecast on this iteration (enX + step,
            # or its outlier-resampled successor), and nothing mutates a state
            # matrix in place afterwards, so a copy would only double the
            # peak memory at commit for an (nx, ne) array nobody else changes.
            if self.step_accepted:
                self.ensemble.enX = step.state

            # Update the misfit and convergence bookkeeping
            misfit = np.asarray(step.misfit, dtype=float)
            self.ensemble_misfit  = misfit
            self.data_misfit_mean = float(misfit.mean())
            self.data_misfit_std  = float(misfit.std())

            if step.why_stop:
                self.why_stop.update(step.why_stop)

            if self.step_accepted:
                # Logged before the counter advances: the row is numbered
                # `iteration + 1`, so this is the iteration just finished.
                self.log_update(success=True)
                self.iteration += 1
                self.after_accepted_iteration()

            # After every attempt, not only accepted ones: a scheme can
            # converge on a step it is about to reject.
            if self.check_misfit_convergence():
                converged = True
            elif self.check_state_convergence():
                converged = True
            elif self.check_convergence(): # Subclass-specific convergence criteria.
                converged = True

            if self.step_accepted and self.restartsave:
                self.save_restart()

            if converged:
                break

            if not self.step_accepted:
                # The scheme has already retried as much as it intends to,
                # inside update_step(). Asking again would repeat the step it
                # just reported it could not improve on.
                self.conv_msg = self.conv_msg or "No improving step found"
                break

        if self.iteration >= self.maxiter and not converged:
            self.conv_msg = "Maximum number of iterations reached"

        self.after_loop(converged)
        return self._finalize(converged)

    # ------------------------------------------------------------------
    # The run table
    # ------------------------------------------------------------------

    def log_update(self, success=None, prior_run=False) -> None:
        """Log one row of the run table.

        Called by :meth:`run_assimilation` -- once for the prior and once per
        accepted iteration -- so a scheme gets its rows without asking, and
        the attempts it makes inside :meth:`update_step` stay its own
        business. The row is the same for every scheme apart from its control
        parameter, which :meth:`log_columns` supplies.
        """
        if self.logger is None:
            return
        info = {
            "Iteration"   : f"{0 if prior_run else self.iteration + 1}",
            "Status"      : "Success" if (prior_run or success) else "Failed",
            "Data Misfit" : self.data_misfit_mean,
            "Change (%)"  : "" if prior_run else
                            100 * (self.data_misfit_mean / self.prev_data_misfit_mean - 1),
        }
        info.update(self.log_columns(prior_run=prior_run))
        self.logger(**info)

    def log_columns(self, prior_run: bool = False) -> dict:
        """Trailing columns for the run table -- typically the scheme's
        control parameter, e.g. ``{"λ": self.lam}``. Empty by default."""
        return {}

    def score(self, pred_data=None) -> "np.ndarray | None":
        r"""Per-realisation data misfit of a forecast.

        Called every time a new state has been forecast and needs a number:
        once for the prior, by :meth:`record_prior_score`, and then by each
        scheme for every attempt it takes inside :meth:`update_step`. One
        definition per scheme, rather than the same expression repeated in a
        prior-scoring hook and again in the step.

        Parameters
        ----------
        pred_data : optional
            The forecast to score -- a ``PETDataFrame`` or an ``(nd, ne)``
            matrix. Defaults to ``self.pred_data``, which is what the
            ensemble's most recent forecast produced, so the usual call is
            ``self.score()`` straight after ``run_forecast``. Pass one
            explicitly to score a forecast the ensemble no longer holds.

        Returns
        -------
        np.ndarray or None
            ``(ne,)`` misfit per realisation, or ``None`` when the scheme has
            no observation ensemble bound -- a scheme that scores some other
            way overrides this, and one that reports no misfit at all (the
            base's own tests) leaves the loop's misfit bookkeeping alone.

        Notes
        -----
        The default is the objective function every shipped scheme uses,

        .. math::

            \Phi_j = (g(m_j) - d_j)^{\mathsf T} C_d^{-1} (g(m_j) - d_j),

        against the *perturbed* observations ``enObs`` and the data covariance
        ``cov_data``. ES-MDA overrides it to score against an un-inflated copy
        of the perturbations (``enObs_conv``); the multilevel scheme to score
        all fidelity levels at once.
        """
        pred = self.pred_data if pred_data is None else pred_data
        enObs = getattr(self, "enObs", None)
        if enObs is None or pred is None:
            return None
        return at.calc_objectivefun(enObs, self._as_matrix(pred), self.cov_data)

    @staticmethod
    def _as_matrix(pred) -> "np.ndarray":
        """A forecast as an ``(nd, ne)`` matrix: a ``PredictedData``, a legacy frame, or an array."""
        if hasattr(pred, "matrix"):
            return pred.matrix
        return pred.to_matrix() if hasattr(pred, "to_matrix") else np.asarray(pred)

    def record_prior_score(self) -> None:
        """Score the prior forecast and record it, before any iteration.

        Sets ``prior_data_misfit_mean``, ``data_misfit_mean`` and the
        per-realisation ``ensemble_misfit``, so the prior is described by the
        same attributes as every later iteration -- and early enough that the
        iteration-0 artifacts written by :meth:`after_prior_forecast` can
        capture them.

        This used to be a ``score_prior()`` hook that each scheme implemented,
        which meant every scheme spelled out both the misfit expression and
        the five assignments around it. The expression is now :meth:`score`
        and the bookkeeping is here; a scheme customises the former.

        Does nothing when :meth:`score` reports no misfit, which is how a
        scheme with nothing to score opts out.
        """
        misfit = self.score()
        if misfit is None:
            return

        misfit = np.asarray(misfit, dtype=float)
        self.ensemble_misfit = misfit
        self.data_misfit_mean = float(misfit.mean())
        self.data_misfit_std = float(misfit.std())
        self.prior_data_misfit_mean = self.data_misfit_mean
        self.prior_data_misfit_std = self.data_misfit_std

        self.log_update(success=True, prior_run=True)

    # ------------------------------------------------------------------
    # Points in a run
    # ------------------------------------------------------------------
    def run_forecast(self, state):
        """Forecast ``state``, then run the post-forecast step.

        Returns the state to carry forward -- the same one unless
        :meth:`after_forecast` replaced members in it.
        """
        self.ensemble.forecast(state)
        return self.after_forecast(state)

    def after_prior_forecast(self) -> None:
        """Handle the prior forecast: prior QA, saved artifacts.

        Outlier replacement is not done here. The prior goes through
        :meth:`after_forecast` like every other forecast, so it has already
        happened by the time this runs -- and before :meth:`record_prior_score`
        computes the misfit, which is the order that matters.
        """
        self.qaqc = self._build_qaqc()

        self._run_prior_quality_assurance()
        self._save_prior_forecast()
        if self._savedata_keys:
            self._save_iteration_data()
        if "iterinfo" in self.keys_da:
            self._save_iteration_information()

    def after_analysis(self) -> None:
        """Between analysis and forecast.

        The odd one out: it marks a point *inside* :meth:`update_step`, and
        this class does not dictate the shape of a step, so a scheme calls it
        itself. The rest of the hooks here are called by
        :meth:`run_assimilation`. Nothing runs here at present; it used to
        refresh QA/QC's variance after data screening, which is no longer
        supported.
        """

    def after_forecast(self, state):
        """Between forecast and scoring: replace outlier members.

        Ordering matters -- outliers are replaced before the misfit is scored,
        so the replacement feeds into the number the scheme sees. The
        resampled state is returned rather than written back, so the caller
        keeps ownership of what it is forecasting.
        """
        if "remove_outliers" in self.keys_da:
            return self.ensemble.remove_outliers(state)
        return state

    def after_accepted_iteration(self) -> None:
        """Persist iteration artifacts and run QA/QC after an accepted update."""
        if "iterinfo" in self.keys_da:
            self._save_iteration_information()
        if self._savedata_keys:
            self._save_iteration_data()

        if self.qaqc is not None:
            if "qc" in self.keys_da:
                self._set_qaqc()
                self.qaqc.calc_da_stat()
            if "qa" in self.keys_da:
                self._set_qaqc()
                self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))
                self.qaqc.calc_kg()


    def after_loop(self, converged: bool) -> None:
        """Save the posterior and the reason the run stopped."""
        if self._saving_enabled:
            self._save_posterior_results()
            self._save_stop_reason(converged)
        self._log_convergence_summary(converged)

    # ------------------------------------------------------------------
    # Shared convergence criteria
    # ------------------------------------------------------------------
    def check_misfit_convergence(self) -> bool:
        """Check convergence on the relative change in mean data misfit."""
        if self.prev_data_misfit_mean is None or self.data_misfit_mean is None:
            return False
        prev = np.mean(self.prev_data_misfit_mean)
        if prev == 0:
            return False
        change = abs(np.mean(self.data_misfit_mean) - prev)
        if change < self.misfit_tol * abs(prev):
            self.conv_msg = (
                f"Data misfit change satisfies |Δd| < {self.misfit_tol}·|d_prev|"
            )
            self.why_stop["misfit_tol"] = True
            return True
        return False

    def check_state_convergence(self) -> bool:
        """Check convergence on the norm of the state update.

        The counterpart of :meth:`popt.optimization_methods.optimizer_base.
        OptimizerBase.check_state_convergence`, which compares ``xk`` against
        ``xk_old``. ``enX_old`` is snapshotted by :meth:`run_assimilation`
        before each attempt, but only when ``step_tol > 0`` -- see there for
        why.

        Opt-in in practice: every shipped scheme passes ``step_tol=0.0``,
        because ``‖Δx‖₂`` over a state that mixes variables on different
        scales (log-permeability alongside saturations, say) has no tolerance
        that is meaningful across cases. The base default of ``1e-8`` is small
        enough to mean "the state did not move at all" rather than being a
        guess at a scale.
        """
        if self.enX_old is None:
            return False
        # A rejected step leaves enX untouched, so the norm would be exactly
        # zero -- convergence on every rejection, when the scheme in fact
        # failed to improve.
        if not self.step_accepted:
            return False
        step_norm = np.linalg.norm(np.asarray(self.ensemble.enX) - np.asarray(self.enX_old))
        if step_norm < self.step_tol:
            self.conv_msg = f"State change satisfies ‖Δx‖ < {self.step_tol}"
            self.why_stop["step_tol"] = True
            return True
        return False

    # ------------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------------
    def _finalize(self, converged: bool) -> AssimilationResult:
        """Populate the result object and log the stopping reason."""
        self.results["nit"] = self.iteration
        self.results["success"] = bool(converged)
        self.results["message"] = self.conv_msg
        self.results["why_stop"] = dict(self.why_stop)
        self.results["data_misfit"] = self.data_misfit_mean
        self.results["prior_data_misfit"] = self.prior_data_misfit_mean
        self.results["x"] = getattr(self.ensemble, "enX", None)

        if self.logger:
            self.logger(f"Assimilation finished after {self.iteration} iteration(s): "
                        f"{self.conv_msg or 'no stopping reason recorded'}")
        return self.results

    # ------------------------------------------------------------------
    # QA/QC
    # ------------------------------------------------------------------
    def _build_qaqc(self) -> "QAQC | None":
        """Create QA/QC helper only when requested by the configuration."""
        qaqc_requested = (
            "qa" in self.keys_da
            or "qa" in self.sim.input_dict
            or "qc" in self.keys_da
        )
        if not qaqc_requested:
            return None

        from pipt.misc_tools.qaqc_tools import QAQC  # heavy: matplotlib

        return QAQC(
            self.keys_da | self.sim.input_dict,
            self.data_df,
            self.data_var_df,
            logger=self.logger,
            prior_info=self.prior_info,
            sim=self.sim,
            ini_state=self.state_layout.to_dict(self.prior_enX),
            localization=self.localization,
            folder=Path(self.save_folder or ".") / "QAQC",
        )

    def _set_qaqc(self) -> None:
        # QA/QC reads predictions cell by cell; hand it the frame view.
        self.qaqc.set(self.pred_data.to_frame(), self.state_layout.to_dict(self.enX), self.lam)

    def _run_prior_quality_assurance(self) -> None:
        if self.qaqc is None or "qa" not in self.keys_da:
            return

        self._set_qaqc()
        self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))
        self.qaqc.calc_coverage()
        self.qaqc.calc_kg({"plot_all_kg": True, "only_log": False, "num_store": 5})

    # ------------------------------------------------------------------
    # From an analysis result to a trial state
    # ------------------------------------------------------------------
    def propose_state(self, result, step_scale=1.0):
        """The trial state an analysis result implies.

        Parameters
        ----------
        result : AnalysisResult or array-like
            What ``self.update(...)`` returned. A plain array is a
            state-space step.
        step_scale : float, optional
            Step length applied to the step (GN-EnRML's ``gamma``); 1 for
            schemes without one.

        Returns
        -------
        np.ndarray
            The state to forecast. Weight-space results also advance
            ``self.W`` from ``self.current_W``; the scheme commits ``W`` to
            ``current_W`` when it accepts the step.
        """
        result = AnalysisResult.coerce(result)
        self.step = result.step   # kept for ``savedata``; None for weight-space results
        if result.step is not None:
            return self.enX + step_scale * result.step
        if result.w_step is not None:
            # Ensemble subspace formulation (Evensen et al. 2019), W_0 = 0.
            self.W = self.current_W + step_scale * result.w_step
            return np.dot(self.prior_enX, (np.eye(self.ne) + self.W / np.sqrt(self.ne - 1)))
        # Matrix formulation (Raanes et al. 2019), W_0 = I.
        self.W = self.current_W + step_scale * result.W_step
        X_p = self.prior_enX @ self.proj * np.sqrt(self.ne - 1)
        return np.mean(self.prior_enX, axis=1, keepdims=True) + np.dot(X_p, self.W)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _save_prior_forecast(self) -> None:
        if not self._saving_enabled:
            return
        try:
            self.sim_data.to_pickle(self._save_path(self.PRIOR_FORECAST_FILE))
        except Exception:
            np.savez(self._save_path(self.PRIOR_FORECAST_FILE), sim_data=self.sim_data)

    def _save_posterior_results(self) -> None:
        """Save posterior state and forecast, falling back to pickle if needed."""
        try:
            np.savez(self._save_path(self.POSTERIOR_STATE_FILE), **self.state_layout.to_dict(self.enX))
            self.sim_data.to_pickle(self._save_path(self.POSTERIOR_FORECAST_FILE))
        except Exception:
            with open(self._save_path(self.POSTERIOR_STATE_FILE), "wb") as file:
                pickle.dump(self.state_layout.to_dict(self.enX), file)
            with open(self._save_path(self.POSTERIOR_FORECAST_FILE), "wb") as file:
                pickle.dump(self.sim_data, file)

    def _save_stop_reason(self, converged: bool) -> None:
        if converged:
            reason = "Convergence criteria met. Stopping assimilation loop."
        else:
            reason = "Maximum iterations reached without convergence."
        self.logger.info(reason)

        why = self.why_stop.copy() if isinstance(self.why_stop, dict) else self.why_stop
        if why is not None:
            why["conv_string"] = reason

        with open(self._save_path(self.STOP_REASON_FILE), "wb") as file:
            pickle.dump(why, file, protocol=4)

    def _log_convergence_summary(self, converged: bool) -> None:
        # `logger` is None for a collaborator that has none at all, which the
        # ensemble protocol allows; `log_update` guards the same way.
        if self.logger is None or self.prev_data_misfit_mean is None:
            return

        # Said "Convergence was met." whatever had happened, including a run
        # that stopped on the iteration limit.
        out_str = "\n Convergence was met." if converged else "\n Stopped without convergence."
        if self.prior_data_misfit_mean > self.data_misfit_mean:
            out_str += (
                f" Obj. function reduced from {self.prior_data_misfit_mean:0.1f} "
                f"to {self.data_misfit_mean:0.1f}"
            )
        self.logger(out_str)

    def _save_iteration_information(self) -> None:
        """Run configured iteration-info hooks."""
        for element in self._as_list(self.keys_da["iterinfo"]):
            if ".py" not in element:
                continue

            module_name = element.removesuffix(".py")
            iter_info_func = import_module(module_name)
            iter_info_func.main(self)

    @property
    def _savedata_keys(self) -> list[str]:
        """Variable names to record each iteration, from ``savedata``.

        ``analysisdebug`` is the old spelling and is still honoured, with a
        deprecation warning. The two are not merged: a config carrying both is
        almost certainly mid-migration, and silently unioning them would hide
        whichever one the user forgot to delete.
        """
        if "savedata" in self.keys_da:
            return self._as_list(self.keys_da["savedata"])
        if "analysisdebug" in self.keys_da:
            warnings.warn(
                "The 'analysisdebug' config key is deprecated; rename it to "
                "'savedata'. Output files are now 'assimilation_result_{i}.npz' "
                "rather than 'debug_analysis_step_{i}.npz'.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self._as_list(self.keys_da["analysisdebug"])
        return []

    def _save_iteration_data(self) -> None:
        """Save the scheme attributes named by ``savedata``.

        One file per iteration, ``assimilation_result_{iteration}.npz``, with
        iteration 0 describing the prior -- the assimilation counterpart of
        popt's ``optimize_result_{i}.npz``. ``state`` is special-cased: it
        expands to one array per state variable rather than a single entry.

        A name the scheme does not carry is reported and skipped rather than
        failing the run, since a variable can legitimately be absent for a
        given scheme -- ``lam`` exists for the Levenberg-Marquardt family and
        not for ES-MDA.
        """
        save_dict: dict[str, Any] = {}

        for save_type in self._savedata_keys:
            if hasattr(self, save_type):
                save_attr = getattr(self, save_type)
                if isinstance(save_attr, (pd.DataFrame, PETDataFrame)):
                    save_dict[save_type] = save_attr.to_dict(orient="records")
                elif hasattr(save_attr, "matrix"):
                    save_dict[save_type] = save_attr.matrix   # PredictedData: the (nd, ne) matrix
                else:
                    save_dict[save_type] = save_attr
            elif save_type == "state":
                save_dict.update(self._state_debug_dict())
            else:
                warnings.warn(
                    f"Cannot save '{save_type}' at iteration {self.iteration}: "
                    f"neither {type(self).__name__} nor its ensemble has an "
                    f"attribute by that name.",
                    stacklevel=2,
                )

        save_dict["savefolder"] = self.save_folder
        at.save_assimilation_result(self.iteration, **save_dict)

    def _state_debug_dict(self) -> dict[str, Any]:
        if getattr(self.ensemble, "multilevel", None) is not None:
            return {
                f"state_level{level}": self.state_layout.to_dict(self.enX[level])
                for level in range(self.ensemble.tot_level)
            }
        return self.state_layout.to_dict(self.enX)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else [value]

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _save_path(self, filename: str) -> str:
        if self.save_folder is None:
            raise RuntimeError("Cannot save results because saving is disabled.")
        os.makedirs(self.save_folder, exist_ok=True)
        return os.path.join(self.save_folder, filename)
    # ------------------------------------------------------------------
    # Restart hooks required by RestartMixin
    # ------------------------------------------------------------------
    RESTART_ATTRIBUTES: tuple = ()
    """Attributes a scheme needs restored to resume mid-run: what its
    iterations change and what it drew at construction (perturbed
    observations, a damping parameter). The loop's own bookkeeping and the
    ensemble's state are covered by the base state; a subclass only names what
    it adds. Missing names are skipped, so a scheme that has not yet set one
    of them checkpoints fine."""

    def _get_restart_state(self) -> dict:
        return {name: getattr(self, name) for name in self.RESTART_ATTRIBUTES if hasattr(self, name)}

    def _set_restart_state(self, state: dict) -> None:
        for name, value in state.items():
            setattr(self, name, value)

    def _get_base_restart_state(self) -> dict:
        """Serialize the loop's bookkeeping and the ensemble's state."""
        return {
            "iteration": self.iteration,
            "data_misfit": self.data_misfit_mean,
            "prior_data_misfit": self.prior_data_misfit_mean,
            "data_misfit_std": self.data_misfit_std,
            "prior_data_misfit_std": getattr(self, "prior_data_misfit_std", None),
            "prev_data_misfit": self.prev_data_misfit_mean,
            "prev_data_misfit_std": getattr(self, "prev_data_misfit_std", None),
            "ensemble_misfit": getattr(self, "ensemble_misfit", None),
            "conv_msg": self.conv_msg,
            "why_stop": dict(self.why_stop),
            "ensemble": self.ensemble.restart_state(),
        }

    def _set_base_restart_state(self, state: dict) -> None:
        """Restore the loop's bookkeeping and the ensemble's state."""
        self.iteration = state["iteration"]
        self.data_misfit_mean = state["data_misfit"]
        self.prior_data_misfit_mean = state["prior_data_misfit"]
        self.data_misfit_std = state["data_misfit_std"]
        self.prior_data_misfit_std = state.get("prior_data_misfit_std")
        self.prev_data_misfit_mean = state["prev_data_misfit"]
        self.prev_data_misfit_std = state.get("prev_data_misfit_std")
        if state.get("ensemble_misfit") is not None:
            self.ensemble_misfit = state["ensemble_misfit"]
        self.conv_msg = state.get("conv_msg", "")
        self.why_stop = dict(state.get("why_stop", {}))
        self.ensemble.restore_restart_state(state["ensemble"])

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------
    @classmethod
    def assimilate(cls, *args, **options) -> "AssimilationResult":
        """Construct the scheme and run it to completion.

        The assimilation counterpart of ``scipy.optimize.minimize``: one call
        that builds the scheme, runs every iteration, and returns the outcome.
        Use it when the scheme object itself is not needed afterwards; when it
        is, construct the class and call :meth:`run_assimilation` instead.

        Every argument is forwarded verbatim to the constructor, so this accepts
        whatever the scheme accepts rather than imposing a second signature.

        Parameters
        ----------
        *args
            Positional arguments for the constructor. For the shipped PIPT
            schemes that is ``(keys_da, keys_en, sim)`` -- the parsed
            data-assimilation config, the parsed ensemble config, and the
            forward simulator -- from which the scheme builds its own ensemble.
            A scheme written directly against the collaborator protocol is
            handed its ensemble here instead.
        **options
            Keyword arguments for the constructor, such as ``analysis`` to
            override the flavour named in the config.

        Returns
        -------
        AssimilationResult
            Outcome of the run. ``x`` is the posterior state ensemble, ``nit``
            the number of accepted iterations, ``data_misfit`` and
            ``prior_data_misfit`` the final and initial mean misfits, and
            ``message`` the reason the run stopped.

        Examples
        --------
        >>> keys_da, keys_sim, keys_en = read_config.read("case.toml")
        >>> result = ESMDA.assimilate(keys_da, keys_en, flow(keys_sim))
        >>> result.prior_data_misfit, result.data_misfit
        (539.2, 70.1)

        Overriding the flavour named in the config:

        >>> result = ESMDA.assimilate(keys_da, keys_en, sim, analysis="subspace")

        Notes
        -----
        ``success`` reports whether the run stopped on a convergence criterion
        rather than by exhausting ``maxiter``. Schemes with a fixed iteration
        schedule -- ES-MDA in particular -- therefore finish normally with
        ``success=False``, which is expected rather than a failure.

        See Also
        --------
        run_assimilation : Run an already-constructed scheme.
        """
        return cls(*args, **options).run_assimilation()
