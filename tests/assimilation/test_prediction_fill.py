"""The directly filled prediction matrix is what the frame path produced."""

import numpy as np
import pytest

from input_output import read_config
from pipt.ensembles import AssimilationEnsemble
from simulator.vanderpol import VanDerPolOscillator
from test_numerical_characterisation import _write_config, _write_synthetic_case

NE = 10


@pytest.mark.parametrize("scale_data", [False, True])
def test_fill_matches_the_legacy_frame_flatten(tmp_path, monkeypatch, scale_data):
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("fill", "esmda", "approx", report_points, ne=NE))
    if scale_data:
        cfg_da["scale_data"] = True
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    ensemble.forecast(ensemble.enX)

    legacy = ensemble.sim_to_pred_data(ensemble.sim_data).to_matrix()
    np.testing.assert_array_equal(ensemble.pred_data.matrix, legacy)
    assert ensemble.pred_data.layout is ensemble.data_layout
    assert ensemble.pred_data.nd == ensemble.obs_vector.shape[0]


def test_a_missing_observation_no_longer_misaligns_predictions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("gap", "esmda", "approx", report_points, ne=NE))
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    # Blank one observation *before* the layout is built, as a data file with a gap would.
    label, column = ensemble.data_df.index[1], ensemble.data_df.columns[0]
    ensemble.data_df.at[label, column] = np.nan
    ensemble.data_layout = type(ensemble.data_layout).from_frame(ensemble.data_df)
    ensemble.obs_vector = ensemble.data_layout.vector(ensemble.data_df)

    ensemble.forecast(ensemble.enX)
    assert ensemble.pred_data.nd == ensemble.obs_vector.shape[0] == ensemble.data_layout.nd
