"""QA/QC diagnostics on the ensemble's frames.

Unit tests build small observation, variance and prediction frames by hand;
the end-to-end test enables ``qa`` and ``qc`` on the golden Van der Pol case
and checks the run completes and leaves the expected artefacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from misc.structures import PETDataFrame
from pipt.misc_tools.qaqc_tools import QAQC

NE = 12
POINTS = ["t0", "t1", "t2"]


class Recorder:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(str(message))

    def text(self):
        return "\n".join(self.lines)


def _frame(cells, is_ensemble=False):
    df = pd.DataFrame(cells, index=pd.Index(POINTS, name="steps"))
    return PETDataFrame.from_pandas(df, is_ensemble=is_ensemble)


def _case():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(NE)                                  # scalar parameter
    field = rng.standard_normal((5, NE))                         # field parameter
    ini_state = {"x": x[None, :].copy(), "field": field.copy()}
    # Data type "a" is 2x plus noise at every report point; "b" is missing at
    # t1; "sim2seis" is a 4-value vector at t0 and t2.
    a_pred = [2 * x + 0.1 * rng.standard_normal(NE) for _ in POINTS]
    b_pred = [rng.standard_normal(NE) for _ in POINTS]
    s_pred = [rng.standard_normal((4, NE)) for _ in POINTS]
    pred = _frame({"a": a_pred, "b": b_pred, "sim2seis": [s_pred[0], None, s_pred[2]]}, is_ensemble=True)
    obs = _frame({
        "a": [np.array([2 * x.mean() + 3.0]) for _ in POINTS],   # above the whole ensemble
        "b": [np.array([0.0]), None, np.array([0.1])],
        "sim2seis": [np.zeros(4), None, np.zeros(4)],
    })
    var = _frame({
        "a": [np.array([0.04]) for _ in POINTS],
        "b": [np.array([1.0]), None, np.array([1.0])],
        "sim2seis": [np.full(4, 0.5), None, np.full(4, 0.5)],
    })
    keys = {"assimindex": [[0, 1, 2]]}
    prior_info = {"x": {"nx": 1, "ny": 1, "nz": 1}, "field": {"nx": 5, "ny": 1, "nz": 1}}
    return keys, obs, var, pred, ini_state, prior_info


def _qaqc(tmp_path, lam=0.0, **kwargs):
    keys, obs, var, pred, ini_state, prior_info = _case()
    log = Recorder()
    qaqc = QAQC(keys, obs, var, logger=log, prior_info=prior_info, ini_state=ini_state,
                folder=tmp_path / "QAQC", **kwargs)
    qaqc.set(pred, {k: v.copy() for k, v in ini_state.items()}, lam)
    return qaqc, log


def test_frames_are_adapted_per_data_type(tmp_path):
    qaqc, _ = _qaqc(tmp_path)
    assert qaqc.ne == NE
    assert qaqc.en_fcst["a"].shape == (3, NE) and qaqc.en_obs["a"].shape == (3, 1)
    np.testing.assert_array_equal(qaqc.en_var["a"].ravel(), [0.04, 0.04, 0.04])   # variances, not observations
    np.testing.assert_array_equal(qaqc.en_var["b"].ravel(), [1.0, 1.0])
    np.testing.assert_array_equal(qaqc.en_var_vec["sim2seis"].ravel(), np.full(8, 0.5))
    assert qaqc.en_time["b"] == [0, 2]                          # the None at t1 is skipped
    assert qaqc.en_fcst["b"].shape == (2, NE)
    assert qaqc.en_obs_vec["sim2seis"].shape == (8, 1)          # two vintages of four values
    assert qaqc.en_fcst_vec["sim2seis"].shape == (8, NE)
    assert qaqc.en_fcst["sim2seis"].shape == (0, NE)            # no point data of that type


def test_multilevel_is_refused_explicitly(tmp_path):
    keys, obs, var, *_ = _case()
    with pytest.raises(NotImplementedError, match="multilevel"):
        QAQC({**keys, "multilevel": {}}, obs, var, folder=tmp_path)


def test_coverage_flags_observations_outside_the_ensemble(tmp_path):
    qaqc, log = _qaqc(tmp_path)
    qaqc.calc_coverage()
    assert "coverage a: 3 of 3 observations outside" in log.text()
    assert "coverage b: 0 of 2" in log.text()
    assert (tmp_path / "QAQC" / "a.png").exists() and (tmp_path / "QAQC" / "b.png").exists()
    assert "skipping the seismic maps" in log.text()          # no mask file, no field_dim


def test_update_statistics_report_movement_in_prior_standard_deviations(tmp_path):
    qaqc, log = _qaqc(tmp_path)
    moved = {k: v.copy() for k, v in qaqc.ini_state.items()}
    moved["x"] = moved["x"] + 5 * moved["x"].std()               # every x moved by 5 std
    qaqc.set(qaqc.pred_data, moved, 0.0)
    qaqc.calc_da_stat()
    text = log.text()
    assert "Group x:" in text and "100.0% / 100.0% / 100.0%" in text
    assert "Group field:" in text and "0.0% / 0.0% / 0.0%" in text


def test_mahalanobis_ranks_the_data_and_draws_crossplots(tmp_path):
    qaqc, log = _qaqc(tmp_path)
    qaqc.calc_mahalanobis((1, None, 2, None, 1, "time"))
    text = log.text()
    assert "Largest values are" in text and "Largest level-2 values" in text
    assert any(p.name.startswith("crossplot_") for p in (tmp_path / "QAQC").iterdir())
    # the observations of "a" sit far above the ensemble, so they score highest
    first = text.split("Largest values are:\n")[1].splitlines()[0]
    assert "'a'" in first


def test_kalman_gain_has_the_sign_of_the_residual_and_is_ranked(tmp_path):
    qaqc, log = _qaqc(tmp_path, lam=0.0)
    gain = qaqc._gain("x", qaqc.en_fcst["a"], qaqc.en_obs["a"], qaqc.en_var["a"], localize=False)
    assert gain.shape == (1,) and gain[0] > 0                    # data above forecast, positive correlation
    qaqc.calc_kg({"num_store": 3})
    text = log.text()
    assert "largest Kg mean values" in text and "largest Kg max values" in text
    assert "('a', 'field', None)" in text or "('sim2seis', 'field', None)" in text or "('b', 'field', None)" in text


def test_kalman_gain_per_report_point_plots_scalar_parameters(tmp_path):
    qaqc, _ = _qaqc(tmp_path)
    qaqc.calc_kg({"unique_time": True, "only_log": False, "plot_all_kg": True})
    assert (tmp_path / "QAQC" / "Kg_x_a.png").exists()


def test_diagnostics_refuse_to_run_before_set(tmp_path):
    keys, obs, var, pred, ini_state, prior_info = _case()
    qaqc = QAQC(keys, obs, var, logger=Recorder(), prior_info=prior_info, ini_state=ini_state, folder=tmp_path)
    with pytest.raises(ValueError, match="call set"):
        qaqc.calc_coverage()


# ----------------------------------------------------------------------
# End to end: qa and qc through a scheme on the golden case
# ----------------------------------------------------------------------

from input_output import read_config  # noqa: E402
from pipt import ESMDA, LMEnRML  # noqa: E402
from simulator.vanderpol import VanDerPolOscillator  # noqa: E402
from test_numerical_characterisation import GLOBAL_SEED, _write_config, _write_synthetic_case  # noqa: E402


@pytest.mark.parametrize("scheme_cls, name, analysis", [(ESMDA, "esmda", "approx"), (LMEnRML, "lmenrml", "approx")],
                         ids=["esmda", "lmenrml"])
def test_qa_and_qc_run_through_a_scheme(tmp_path, monkeypatch, caplog, scheme_cls, name, analysis):
    monkeypatch.chdir(tmp_path)
    caplog.set_level("INFO")
    report_points = _write_synthetic_case()
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("qaqc_case", name, analysis, report_points))
    cfg_da["qa"] = True
    cfg_da["qc"] = True
    np.random.seed(GLOBAL_SEED)

    result = scheme_cls.assimilate(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim), analysis=analysis)

    assert np.isfinite(result.x).all()
    produced = {p.name for p in Path("QAQC").iterdir()}
    assert "x1.png" in produced                                  # coverage of the one data type
    assert any(n.startswith("crossplot_") for n in produced)
    # The run logger propagates to the root logger, which pytest captures.
    assert "Statistics for updated parameters" in caplog.text
    assert "largest Kg mean values" in caplog.text
    assert "Mahalanobis" in caplog.text


def test_hls_conversion_round_trips():
    """The numpy HLS conversion replaced OpenCV's; it must invert itself."""
    from pipt.misc_tools.qaqc_tools import _hls_to_rgb, _rgb_to_hls

    rgb = np.random.default_rng(3).random((6, 7, 3))
    np.testing.assert_allclose(_hls_to_rgb(_rgb_to_hls(rgb)), rgb, atol=1e-12)
    grey = np.full((2, 2, 3), 0.4)
    np.testing.assert_allclose(_hls_to_rgb(_rgb_to_hls(grey)), grey, atol=1e-12)
