"""The forecast backends give the same forecast for the same state."""

import numpy as np
import pytest

from input_output import read_config
from pipt.ensembles import AssimilationEnsemble
from simulator.vanderpol import VanDerPolOscillator
from test_numerical_characterisation import _write_config, _write_synthetic_case

NE = 12


def _forecast(tmp_path, monkeypatch, parallel):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("backend", "esmda", "approx", report_points, ne=NE))
    cfg_sim["parallel"] = parallel
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    ensemble.forecast(ensemble.enX)
    return ensemble.pred_data.matrix


@pytest.mark.slow
def test_the_process_pool_forecast_matches_the_serial_one(tmp_path, monkeypatch):
    np.random.seed(5)
    serial = _forecast(tmp_path / "serial", monkeypatch, parallel=1)
    np.random.seed(5)
    pooled = _forecast(tmp_path / "pooled", monkeypatch, parallel=2)
    np.testing.assert_array_equal(pooled, serial)
