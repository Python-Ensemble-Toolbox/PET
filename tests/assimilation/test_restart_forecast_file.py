"""A hand-placed `restart_sim_results.pkl` stands in for the forecast only on a restart.

The file is how a user hands a crashed run the forecast it had already
finished. It used to be consumed on any run that found it in the working
directory, so a forgotten file silently replaced a fresh forecast.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from input_output import read_config
from pipt.ensembles import AssimilationEnsemble
from pipt.ensembles.forecast import ForecastMixin
from simulator.vanderpol import VanDerPolOscillator
from test_numerical_characterisation import _write_config, _write_synthetic_case

NE = 20


@pytest.fixture(params=["saving", "nosave"])
def ensemble_with_placed_file(request, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("restart_file", "esmda", "approx", report_points, ne=NE))
    if request.param == "saving":
        cfg_da.pop("nosave")
        cfg_da["savefolder"] = "run_results"
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    ensemble.forecast(ensemble.enX)
    placed = ensemble.sim_data
    with open(ForecastMixin.RESTART_RESULTS_FILE, "wb") as file:
        pickle.dump(placed, file)
    return ensemble, placed


def test_an_ordinary_run_ignores_the_file_and_forecasts(ensemble_with_placed_file, monkeypatch):
    ensemble, _ = ensemble_with_placed_file
    calls = []
    original = ensemble.calc_prediction
    monkeypatch.setattr(ensemble, "calc_prediction", lambda enX: calls.append(1) or original(enX))

    assert ensemble.restart is False
    ensemble.forecast(ensemble.enX)

    assert calls == [1]
    assert Path(ForecastMixin.RESTART_RESULTS_FILE).exists()


def test_a_restart_uses_the_file_once_and_files_it_with_the_results(ensemble_with_placed_file, monkeypatch):
    ensemble, placed = ensemble_with_placed_file

    def no_forecast(enX):
        raise AssertionError("the placed forecast should have been used instead of simulating")

    monkeypatch.setattr(ensemble, "calc_prediction", no_forecast)
    ensemble.restart = True
    ensemble.forecast(ensemble.enX)

    expected = ensemble._container_from_frame(ensemble.sim_to_pred_data(placed))
    np.testing.assert_array_equal(ensemble.pred_data.matrix, expected.matrix)
    assert not Path(ForecastMixin.RESTART_RESULTS_FILE).exists()
    filed_under = Path(ensemble.save_folder or ".") / ForecastMixin.SIM_RESULTS_FILE
    assert filed_under.exists()
