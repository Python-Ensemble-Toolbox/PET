"""Tests for the shared assimilation scheme base class.

These exercise ``AssimilationScheme`` in isolation via a fake ensemble, so
the loop/convergence/restart machinery is covered without running a simulator.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from pipt.update_schemes.core import (
    AssimilationResult,
    AssimilationScheme,
    StepReport,
)


class FakeEnsemble:
    """Minimal object satisfying the ensemble collaborator protocol.

    ``keys_da``, ``sim`` and ``_saving_enabled`` are part of it because the
    scheme carries the run workflow -- QA/QC, artifact saving, outlier
    replacement -- and consults them at every hook. Saving is off, so nothing
    here touches the filesystem.
    """

    def __init__(self, nx=3, ne=5):
        self.enX = np.zeros((nx, ne))
        self.pred_data = None
        self.logger = None
        self.forecast_calls = 0
        self.keys_da = {}
        self.sim = SimpleNamespace(input_dict={})
        self._saving_enabled = False

    def forecast(self, enX):
        self.forecast_calls += 1
        self.pred_data = enX.copy()

    def restart_state(self):
        return {"enX": self.enX}

    def restore_restart_state(self, state):
        self.enX = state["enX"]


class DecreasingMisfitScheme(AssimilationScheme):
    """Scheme whose misfit halves each step, converging on misfit_tol."""

    def update_step(self):
        # The loop derives data_misfit from the reported array, so the shift
        # of current -> previous happens here, before the new value is sent.
        self.prev_data_misfit_mean = self.data_misfit_mean
        value = 100.0 if self.data_misfit_mean is None else self.data_misfit_mean / 2.0
        if self.prior_data_misfit_mean is None:
            self.prior_data_misfit_mean = value
        self.enX_old = self.ensemble.enX.copy()
        self.ensemble.enX = self.ensemble.enX + 1.0
        self.ensemble.forecast(self.ensemble.enX)
        return StepReport(accepted=True, state=self.ensemble.enX,
                          misfit=np.full(self.ensemble.enX.shape[1], value))


class NeverConvergingScheme(AssimilationScheme):
    """Scheme that always accepts but never satisfies a tolerance."""

    def update_step(self):
        self.prev_data_misfit_mean = self.data_misfit_mean
        value = 100.0 if self.data_misfit_mean is None else self.data_misfit_mean * 2.0
        if self.prior_data_misfit_mean is None:
            self.prior_data_misfit_mean = value
        self.enX_old = self.ensemble.enX.copy()
        self.ensemble.enX = self.ensemble.enX + 10.0
        return StepReport(accepted=True, state=self.ensemble.enX,
                          misfit=np.full(self.ensemble.enX.shape[1], value))


class StallingScheme(AssimilationScheme):
    """Accepts, but barely moves the state -- and does not snapshot enX_old.

    The shipped schemes are all like this: none of them assign ``enX_old``,
    so state convergence only works if the base loop takes the snapshot.
    """

    def update_step(self):
        self.prev_data_misfit_mean = self.data_misfit_mean
        value = 100.0 if self.data_misfit_mean is None else self.data_misfit_mean * 0.999
        if self.prior_data_misfit_mean is None:
            self.prior_data_misfit_mean = value
        self.ensemble.enX = self.ensemble.enX + 1e-12
        self.ensemble.forecast(self.ensemble.enX)
        return StepReport(accepted=True, state=self.ensemble.enX,
                          misfit=np.full(self.ensemble.enX.shape[1], value))


class AlwaysRejectingScheme(AssimilationScheme):
    """Scheme that reports it could not find an improving step.

    A scheme retries internally, so a rejected report means it has given up;
    the loop stops rather than asking again.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempts = 0

    def update_step(self):
        self.attempts += 1
        # Rejected: nothing moved, so report the misfit as it stands.
        value = 100.0 if self.data_misfit_mean is None else self.data_misfit_mean
        if self.prior_data_misfit_mean is None:
            self.prior_data_misfit_mean = value
        return StepReport(accepted=False, state=self.ensemble.enX,
                          misfit=np.full(self.ensemble.enX.shape[1], value))


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

def test_is_abstract():
    """The base class cannot be instantiated without update_step."""
    with pytest.raises(TypeError):
        AssimilationScheme(FakeEnsemble())


def test_defaults(in_tmp_dir):
    scheme = DecreasingMisfitScheme(FakeEnsemble())
    assert scheme.iteration == 0
    assert scheme.maxiter == 100
    assert scheme.misfit_tol == 0.01
    assert scheme.restart is False
    assert scheme.restart_file == "decreasingmisfitscheme_restart.pkl"


# ----------------------------------------------------------------------
# Loop behaviour
# ----------------------------------------------------------------------

def test_runs_prior_forecast_before_iterating(in_tmp_dir):
    ens = FakeEnsemble()
    DecreasingMisfitScheme(ens, maxiter=1).run_assimilation()
    # one prior forecast plus one per accepted iteration
    assert ens.forecast_calls == 2


def test_stops_at_maxiter(in_tmp_dir):
    scheme = NeverConvergingScheme(FakeEnsemble(), maxiter=4)
    res = scheme.run_assimilation()
    assert res.nit == 4
    assert res.success is False
    assert "Maximum number of iterations" in res.message


def test_converges_on_misfit_tolerance(in_tmp_dir):
    # misfit halves each step, so the relative change is 0.5 -- never below a
    # 0.01 tolerance, but comfortably below a 0.9 one.
    scheme = DecreasingMisfitScheme(FakeEnsemble(), maxiter=20, misfit_tol=0.9)
    res = scheme.run_assimilation()
    assert res.success is True
    assert res.nit < 20
    assert res.why_stop.get("misfit_tol") is True
    assert "Data misfit change" in res.message


def test_converges_on_state_tolerance(in_tmp_dir):
    # step_tol is huge, so the first state change counts as convergence.
    scheme = DecreasingMisfitScheme(FakeEnsemble(), maxiter=20, step_tol=1e9)
    res = scheme.run_assimilation()
    assert res.success is True
    assert res.why_stop.get("step_tol") is True


def test_state_convergence_works_without_the_scheme_snapshotting(in_tmp_dir):
    """The base loop takes the enX_old snapshot, so a scheme gets state
    convergence without doing any bookkeeping of its own -- which is the
    situation every shipped scheme is in.
    """
    scheme = StallingScheme(FakeEnsemble(), maxiter=20, step_tol=1e-6)
    res = scheme.run_assimilation()
    assert res.success is True
    assert res.why_stop.get("step_tol") is True
    assert "did not move" not in res.message  # names the criterion
    assert res.nit < 20                       # stopped early, not on maxiter


def test_state_convergence_ignores_rejected_steps(in_tmp_dir):
    """A rejected step leaves enX untouched, so the norm is exactly zero.

    Without the step_accepted guard that would read as instant convergence,
    when the truth is the scheme could not find an improvement.
    """
    scheme = AlwaysRejectingScheme(FakeEnsemble(), maxiter=5, step_tol=1e9)
    res = scheme.run_assimilation()
    assert res.why_stop.get("step_tol") is not True
    assert res.success is False


def test_no_snapshot_taken_when_the_criterion_is_off(in_tmp_dir):
    """enX can be large; the copy is skipped entirely when step_tol == 0."""
    scheme = NeverConvergingScheme(FakeEnsemble(), maxiter=3, step_tol=0.0)
    scheme.enX_old = None
    scheme.run_assimilation()
    # NeverConvergingScheme sets enX_old itself, so prove the *loop* did not:
    plain = AlwaysRejectingScheme(FakeEnsemble(), maxiter=2, step_tol=0.0)
    plain.run_assimilation()
    assert plain.enX_old is None


def test_subclass_convergence_hook(in_tmp_dir):
    class StopsAfterTwo(NeverConvergingScheme):
        def check_convergence(self):
            if self.iteration >= 2:
                self.conv_msg = "scheme-specific criterion"
                return True
            return False

    res = StopsAfterTwo(FakeEnsemble(), maxiter=50).run_assimilation()
    assert res.success is True
    assert res.nit == 2
    assert res.message == "scheme-specific criterion"


def test_a_rejected_step_stops_the_run(in_tmp_dir):
    """The scheme has already retried inside update_step; asking again would
    only repeat the step it just said it could not improve on."""
    scheme = AlwaysRejectingScheme(FakeEnsemble(), maxiter=5)
    res = scheme.run_assimilation()
    assert res.nit == 0
    assert scheme.attempts == 1
    assert res.success is False
    assert "No improving step" in res.message


def test_a_scheme_that_stops_keeps_its_own_message(in_tmp_dir):
    """A scheme explaining its own give-up is not overwritten by the loop."""

    class ExplainsItself(AlwaysRejectingScheme):
        def update_step(self):
            self.conv_msg = "ran out of damping attempts"
            return super().update_step()

    res = ExplainsItself(FakeEnsemble(), maxiter=5).run_assimilation()
    assert res.message == "ran out of damping attempts"


def test_the_loop_logs_one_row_per_accepted_iteration(in_tmp_dir):
    """Logging is the loop's job, so a scheme gets its rows without asking."""
    rows = []

    class Logging(DecreasingMisfitScheme):
        def log_update(self, success=None, prior_run=False):
            rows.append((self.iteration, prior_run))

    Logging(FakeEnsemble(), maxiter=3).run_assimilation()
    # No prior row: this scheme scores nothing, so there is no prior to report.
    assert rows == [(0, False), (1, False), (2, False)]


# ----------------------------------------------------------------------
# Result object
# ----------------------------------------------------------------------

def test_result_is_attribute_accessible(in_tmp_dir):
    res = DecreasingMisfitScheme(FakeEnsemble(), maxiter=2).run_assimilation()
    assert isinstance(res, AssimilationResult)
    assert res["nit"] == res.nit
    assert res.prior_data_misfit == 100.0


def test_assimilate_classmethod_matches_manual_run(in_tmp_dir):
    res = DecreasingMisfitScheme.assimilate(FakeEnsemble(), maxiter=3)
    manual = DecreasingMisfitScheme(FakeEnsemble(), maxiter=3).run_assimilation()
    assert res.nit == manual.nit
    assert res.data_misfit == manual.data_misfit


# ----------------------------------------------------------------------
# Restart
# ----------------------------------------------------------------------

def test_restart_roundtrip(in_tmp_dir):
    scheme = DecreasingMisfitScheme(
        FakeEnsemble(), maxiter=3, restartsave=True
    )
    scheme.run_assimilation()
    assert os.path.exists(scheme.restart_file)
    saved_iteration = scheme.iteration
    saved_misfit = scheme.data_misfit_mean

    resumed = DecreasingMisfitScheme(
        FakeEnsemble(), maxiter=3, restart=True
    )
    resumed.load_restart()
    assert resumed.iteration == saved_iteration
    assert resumed.data_misfit_mean == saved_misfit


def test_restart_file_rejects_foreign_scheme(in_tmp_dir):
    scheme = DecreasingMisfitScheme(
        FakeEnsemble(), maxiter=2, restartsave=True
    )
    scheme.run_assimilation()

    foreign = NeverConvergingScheme(FakeEnsemble(), restart=True)
    foreign.restart_file = scheme.restart_file
    with pytest.raises(RuntimeError, match="does not match"):
        foreign.load_restart()


def test_convergence_summary_says_what_happened():
    """It said 'Convergence was met.' after every run, including one that
    stopped on the iteration limit."""
    from pipt.update_schemes.esmda import ESMDA   # any concrete scheme; the base is abstract

    lines = []
    scheme = object.__new__(ESMDA)
    scheme.logger = lines.append
    scheme.prior_data_misfit_mean = 10.0
    scheme.data_misfit_mean = 4.0
    scheme.prev_data_misfit_mean = 5.0

    scheme._log_convergence_summary(False)
    assert "without convergence" in lines[-1] and "Convergence was met" not in lines[-1]
    scheme._log_convergence_summary(True)
    assert "Convergence was met" in lines[-1]
