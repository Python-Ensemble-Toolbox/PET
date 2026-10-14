"""Adjoints reach the analysis as an ``(nd, nx, ne)`` array aligned with the prediction rows."""

import numpy as np
import pandas as pd

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


def _per_parameter(frame, parameters):
    """``frame`` with each cell's gradient split over ``(datatype, parameter)`` columns, as ``parameters`` slices it."""
    columns = pd.MultiIndex.from_tuples([(datatype, name) for datatype in frame.columns for name in parameters])
    out = pd.DataFrame(index=frame.index, columns=columns, dtype=object)
    for label in frame.index:
        for datatype in frame.columns:
            for name, part in parameters.items():
                out.at[label, (datatype, name)] = np.asarray(frame.at[label, datatype])[part]
    return out


def test_per_parameter_adjoint_columns_are_joined_into_one_gradient(tmp_path, monkeypatch):
    # A simulator may give one gradient per parameter, under (datatype, parameter)
    # columns. The array must be as if each cell held those gradients joined in
    # column order; the lookup used to hand numpy a Series of arrays and fail.
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("adjpar", "esmda", "approx", report_points, ne=NE))
    cfg_sim["compute_adjoints"] = True
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    ensemble.forecast(ensemble.enX)
    joined = ensemble.adjoints

    parameters = {"x": slice(0, 2), "mu": slice(2, None)}
    ensemble.member_adjoints = [_per_parameter(frame, parameters) for frame in ensemble.member_adjoints]
    np.testing.assert_array_equal(ensemble._adjoint_array(), joined)


def test_without_adjoints_the_ensemble_carries_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("noadj", "esmda", "approx", report_points, ne=NE))
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    ensemble.forecast(ensemble.enX)
    assert ensemble.adjoints is None and ensemble.member_adjoints is None
