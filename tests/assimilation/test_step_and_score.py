"""The scoring contract and the schemes' inner damping loops.

Two structural properties are covered here, both of which the numerical
characterisation tests would only catch indirectly:

- ``score()`` is the single definition of a scheme's data misfit, used for the
  prior and for every attempt inside a step. It replaced a per-scheme
  ``score_prior()`` hook that duplicated both the expression and the
  bookkeeping around it.
- The damping parameter is iterated *inside* ``update_step()``. One call is one
  iteration however many attempts it takes, mirroring popt's optimizers.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from misc.structures import PETDataFrame
from pipt import ESMDA, EnKF, GNEnRML, LMEnRML
from pipt.update_schemes.core import AssimilationScheme, StepReport


class FakeLogger:
    """Callable logger with the ``.info`` the schemes also use."""

    def __init__(self):
        self.rows = []

    def __call__(self, *args, **kwargs):
        self.rows.append(kwargs or args)

    def info(self, *args, **kwargs):
        self.rows.append(args)


class FakeEnsemble:
    """Minimal ensemble collaborator, as in test_scheme_base."""

    def __init__(self, nx=3, ne=4):
        self.enX = np.zeros((nx, ne))
        self.pred_data = None
        self.logger = None
        self.forecast_calls = 0
        self.keys_da = {}
        self.sim = SimpleNamespace(input_dict={})
        self._saving_enabled = False

    def forecast(self, enX):
        self.forecast_calls += 1
        self.pred_data = np.ones((5, self.enX.shape[1]))


class ScoringScheme(AssimilationScheme):
    """Scheme with observations bound, so the base ``score()`` applies."""

    def __init__(self, ensemble, **options):
        super().__init__(ensemble, **options)
        self.enObs = np.zeros((5, ensemble.enX.shape[1]))
        self.cov_data = np.ones(5)

    def update_step(self):
        misfit = np.asarray(self.score(), dtype=float)
        self.prev_data_misfit_mean = self.data_misfit_mean
        return StepReport(accepted=True, state=self.ensemble.enX, misfit=misfit)


# ----------------------------------------------------------------------
# score()
# ----------------------------------------------------------------------

class TestScore:

    def test_default_is_the_data_misfit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        scheme = ScoringScheme(FakeEnsemble())
        scheme.ensemble.forecast(scheme.ensemble.enX)

        # (1 - 0)^2 summed over 5 observations, per realisation.
        assert np.allclose(scheme.score(), 5.0)

    def test_scores_an_explicit_forecast(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        scheme = ScoringScheme(FakeEnsemble())

        assert np.allclose(scheme.score(np.full((5, 4), 2.0)), 20.0)

    def test_accepts_a_petdataframe(self, tmp_path, monkeypatch):
        """The ensemble hands over frames, not matrices."""
        monkeypatch.chdir(tmp_path)
        scheme = ScoringScheme(FakeEnsemble())
        frame = PETDataFrame(
            {"d": [np.full(4, 2.0) for _ in range(5)]}, index=range(5), is_ensemble=True
        )

        assert np.allclose(scheme.score(frame), 20.0)

    def test_returns_none_without_observations(self, tmp_path, monkeypatch):
        """A scheme that scores some other way opts out by having no enObs."""
        monkeypatch.chdir(tmp_path)
        scheme = ScoringScheme(FakeEnsemble())
        del scheme.enObs

        assert scheme.score() is None


class TestRecordPriorScore:

    def test_records_the_prior_from_score(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        scheme = ScoringScheme(FakeEnsemble())
        scheme.ensemble.forecast(scheme.ensemble.enX)

        scheme.record_prior_score()

        assert scheme.prior_data_misfit_mean == pytest.approx(5.0)
        assert scheme.data_misfit_mean == pytest.approx(5.0)
        assert scheme.data_misfit_std == pytest.approx(0.0)
        assert np.allclose(scheme.ensemble_misfit, 5.0)

    def test_prior_is_scored_before_the_first_step(self, tmp_path, monkeypatch):
        """The whole point of scoring the prior early: it is the *prior's*."""
        monkeypatch.chdir(tmp_path)
        scheme = ScoringScheme(FakeEnsemble(), maxiter=2)

        result = scheme.run_assimilation()

        assert result.prior_data_misfit == pytest.approx(5.0)

    def test_a_scheme_without_a_misfit_is_left_alone(self, tmp_path, monkeypatch):
        """score() returning None must not clobber the loop's bookkeeping."""
        monkeypatch.chdir(tmp_path)
        scheme = ScoringScheme(FakeEnsemble())
        del scheme.enObs

        scheme.record_prior_score()

        assert scheme.prior_data_misfit_mean is None
        assert scheme.data_misfit_mean is None


class TestSchemeOverrides:
    """ES-MDA scores its own way; the EnKF family scores the base's way."""

    @staticmethod
    def _bare(cls, **attrs):
        scheme = object.__new__(cls)
        for name, value in attrs.items():
            setattr(scheme, name, value)
        return scheme

    def test_esmda_scores_against_uninflated_observations(self):
        """`enObs` is redrawn inflated each step; `enObs_conv` is not."""
        scheme = self._bare(
            ESMDA,
            enObs=np.full((5, 4), 99.0),          # inflated: must not be used
            enObs_conv=np.zeros((5, 4)),
            ensemble=SimpleNamespace(pred_data=np.ones((5, 4))),
        )
        scheme.cov_data = np.ones(5)

        assert np.allclose(scheme.score(), 5.0)

    def test_enkf_scores_with_the_data_covariance(self):
        """It used to pass ``scale_data`` -- a square root -- where the
        objective expects a variance, so the misfit came out as r**2/sigma
        instead of r**2/sigma**2."""
        scheme = self._bare(
            EnKF,
            enObs=np.zeros((5, 4)),
            ensemble=SimpleNamespace(pred_data=np.ones((5, 4))),
        )
        scheme.scale_data = np.full(5, 99.0)     # must not be used
        scheme.cov_data = np.full(5, 4.0)        # variance: 5 data x 1 / 4

        assert np.allclose(scheme.score(), 5 * (1 / 4.0))


# ----------------------------------------------------------------------
# The damping loop inside update_step()
# ----------------------------------------------------------------------

class StubbedStep:
    """Drives a real ``update_step`` with a scripted sequence of misfits.

    Everything the step needs is set directly: the analysis, the forecast and
    the score are stubbed, so what is exercised is the loop the scheme wraps
    around them and nothing else.
    """

    def __init__(self, cls, misfits, **overrides):
        self.scheme = object.__new__(cls)
        self.misfits = list(misfits)
        self.analyses = []          # lambda/gamma at each analysis
        self.forecasts = 0

        scheme = self.scheme
        scheme.logger = FakeLogger()
        scheme.iteration = 0
        scheme.why_stop = {}
        scheme.conv_msg = ""
        scheme._converged = False
        scheme.step_accepted = True
        scheme.max_inner_iter = 10
        scheme.data_misfit_tol = 1e-6
        scheme.data_misfit_mean = 100.0
        scheme.data_misfit_std = 10.0
        scheme.prev_data_misfit_mean = 100.0
        scheme.prev_data_misfit_std = 10.0
        scheme.ensemble_misfit = np.full(4, 100.0)
        scheme.prior_data_misfit_mean = 100.0
        scheme.enX_proposal = np.zeros((3, 4))
        for name, value in overrides.items():
            setattr(scheme, name, value)

        scheme.calc_analysis = self._calc_analysis
        scheme.after_analysis = lambda: None
        scheme.run_forecast = self._run_forecast
        scheme.score = self._score

    @property
    def control(self):
        """The damping parameter under test, whichever this scheme uses."""
        return getattr(self.scheme, "lam", None) or self.scheme.gamma

    def _calc_analysis(self):
        self.analyses.append(self.control)

    def _run_forecast(self, state):
        self.forecasts += 1
        return state

    def _score(self, pred_data=None):
        mean, std = self.misfits.pop(0)
        return np.array([mean - std, mean, mean, mean + std], dtype=float)


def lm(misfits, **overrides):
    defaults = dict(lam=100.0, lam_max=1e10, lam_min=0.01, lam_factor=5.0)
    return StubbedStep(LMEnRML, misfits, **(defaults | overrides))


def gn(misfits, **overrides):
    defaults = dict(gamma=0.4, gamma_max=0.5, gamma_factor=2.0, lam=0.0)
    return StubbedStep(GNEnRML, misfits, **(defaults | overrides))


class TestInnerDampingLoop:

    def test_one_call_retries_until_it_improves(self):
        """A worse misfit re-damps and tries again inside the same call."""
        run = lm([(120.0, 12.0), (50.0, 5.0)])

        report = run.scheme.update_step()

        assert run.forecasts == 2                  # two attempts, one step
        assert run.analyses == [100.0, 500.0]      # λ grew before the retry
        assert report.accepted is True
        assert np.mean(report.misfit) == pytest.approx(50.0)

    def test_the_retry_starts_from_the_same_state(self):
        """Nothing is committed between attempts, so each re-solves the prior."""
        run = lm([(120.0, 12.0), (50.0, 5.0)])

        run.scheme.update_step()

        # prev_ is the last *accepted* misfit throughout, not the rejection's.
        assert run.scheme.prev_data_misfit_mean == pytest.approx(100.0)

    def test_accepting_first_time_takes_one_attempt(self):
        run = lm([(50.0, 5.0)])

        run.scheme.update_step()

        assert run.forecasts == 1
        assert run.analyses == [100.0]

    def test_a_converged_verdict_ends_the_loop(self):
        """λ_max is a stop, not another retry -- the loop must not spin on it."""
        run = lm([(120.0, 12.0)], lam=1e10, lam_max=1e10)

        report = run.scheme.update_step()

        assert run.forecasts == 1
        assert run.scheme._converged is True
        assert report.accepted is False

    def test_exhausting_the_attempts_stops_the_run(self):
        run = lm([(120.0, 12.0)] * 5, max_inner_iter=3)

        report = run.scheme.update_step()

        assert run.forecasts == 3
        assert report.accepted is False
        assert run.scheme._converged is True
        assert run.scheme.why_stop["inner_stop"] is True
        assert "damping attempts" in run.scheme.conv_msg

    def test_the_row_reports_the_damping_the_step_ran_with(self):
        """The loop logs after score_and_commit has already adjusted λ, so the
        column would otherwise report the *next* step's damping."""
        run = lm([(50.0, 5.0)])                     # accepted: λ 100 -> 20

        run.scheme.update_step()

        assert run.scheme.lam == pytest.approx(20.0)
        assert run.scheme.log_columns()["λ"] == pytest.approx(100.0)

    def test_gauss_newton_shortens_its_step_the_same_way(self):
        run = gn([(120.0, 12.0), (50.0, 5.0)])

        report = run.scheme.update_step()

        assert run.forecasts == 2
        assert run.analyses == [0.4, 0.2]          # γ halved before the retry
        assert report.accepted is True

    def test_gauss_newton_gives_up_after_max_inner_iter(self):
        """γ has no lower bound, so the attempt count is what stops it."""
        run = gn([(120.0, 12.0)] * 9, max_inner_iter=4)

        report = run.scheme.update_step()

        assert run.forecasts == 4
        assert report.accepted is False
        assert run.scheme.why_stop["inner_stop"] is True
        assert "step-length attempts" in run.scheme.conv_msg


class TestAutoLambda:
    """``lambda='auto'`` is sized by LM-EnRML's own ``score()`` override."""

    @staticmethod
    def _bare_lm(**attrs):
        scheme = object.__new__(LMEnRML)
        scheme.lam = "auto"
        scheme.enObs = np.zeros((10, 4))
        scheme.logger = FakeLogger()
        scheme.iteration = 0
        # 10 observations off by sqrt(5) each: a misfit of 50 per realisation.
        scheme.ensemble = SimpleNamespace(pred_data=np.full((10, 4), np.sqrt(5.0)))
        for name, value in attrs.items():
            setattr(scheme, name, value)
        scheme.cov_data = np.ones(10)
        return scheme

    def test_sized_from_the_first_score(self):
        scheme = self._bare_lm()

        misfit = scheme.score()

        assert np.allclose(misfit, 50.0)
        assert scheme.lam == pytest.approx(0.5 * 50.0 / 10)   # Φ / 2·nd

    def test_resolved_before_the_prior_row_is_logged(self):
        """The prior QA/QC pass computes with λ, so it cannot still be a str.

        The row logged for the prior reports it too, and both happen before
        the first update_step.
        """
        scheme = self._bare_lm()

        scheme.record_prior_score()

        assert scheme.lam == pytest.approx(2.5)
        logged = [row for row in scheme.logger.rows if isinstance(row, dict)]
        assert logged and logged[-1]["λ"] == pytest.approx(2.5)

    def test_resolved_once_and_left_alone(self):
        """Later scores must not re-size a λ the scheme has been adjusting."""
        scheme = self._bare_lm()
        scheme.score()
        scheme.lam = 0.4          # as an accepted step would have reduced it

        scheme.score()

        assert scheme.lam == pytest.approx(0.4)
