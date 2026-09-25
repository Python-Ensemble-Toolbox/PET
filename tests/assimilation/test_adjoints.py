"""Adjoints reach the analysis as an ``(nd, nx, ne)`` array aligned with the prediction rows."""

import numpy as np

from input_output import read_config
from misc.structures import PETDataFrame
from pipt.ensembles import AssimilationEnsemble
from simulator.vanderpol import VanDerPolOscillator
from test_numerical_characterisation import _write_config, _write_synthetic_case

NE = 8


def test_the_adjoint_array_is_what_the_frame_path_stacked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("adj", "esmda", "approx", report_points, ne=NE))
    cfg_sim["compute_adjoints"] = True
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    ensemble.forecast(ensemble.enX)

    # The legacy construction: merge the members' adjoint frames, keep the observed
    # data types, flatten as a Jacobian.
    legacy = PETDataFrame.merge_dataframes(ensemble.member_adjoints)[ensemble.data_df.columns]
    np.testing.assert_array_equal(ensemble.adjoints, legacy.to_matrix(is_jacobian=True))
    assert ensemble.adjoints.shape == (ensemble.pred_data.nd, ensemble.enX.shape[0], NE)


def test_without_adjoints_the_ensemble_carries_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("noadj", "esmda", "approx", report_points, ne=NE))
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    ensemble.forecast(ensemble.enX)
    assert ensemble.adjoints is None and ensemble.member_adjoints is None
