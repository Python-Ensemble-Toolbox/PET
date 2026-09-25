"""
Unit tests for DataReader and PETDataFrame integration.

Covers:
- Reading data from CSV and pickle
- Handling absolute and relative variance definitions
- Support for external NPZ-referenced data
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from misc.structures import PETDataFrame
from misc.read_input_csv import DataReader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INDEX = ["idx1", "idx2"]
INDEX_NAME = "index"


@pytest.fixture
def base_data():
    """Simple 2x3 dataset."""
    df = pd.DataFrame(
        {
            "keyA": [1.0, 2.0],
            "keyB": [3.0, 4.0],
            "keyC": [5.0, 6.0],
        },
        index=INDEX,
    )
    df.index.name = INDEX_NAME
    return df


@pytest.fixture
def abs_variance(base_data):
    """Absolute variance definition."""
    df = pd.DataFrame(
        {
            col: [["abs", val] for val in values]
            for col, values in {
                "keyA": [0.1, 0.2],
                "keyB": [0.3, 0.4],
                "keyC": [0.5, 0.6],
            }.items()
        },
        index=base_data.index,
    )
    df.index.name = INDEX_NAME
    return df


@pytest.fixture
def rel_variance(base_data):
    """Relative variance definition."""
    def rel(val, obs):
        return float(np.sqrt(val) / (obs * 0.01))

    df = pd.DataFrame(
        {
            col: [
                ["rel", rel(var, base_data.loc[idx, col])]
                for idx, var in zip(INDEX, values)
            ]
            for col, values in {
                "keyA": [0.1, 0.2],
                "keyB": [0.3, 0.4],
                "keyC": [0.5, 0.6],
            }.items()
        },
        index=base_data.index,
    )
    df.index.name = INDEX_NAME
    return df


@pytest.fixture
def expected_variance():
    """Expected variance after processing."""
    df = PETDataFrame(
        {
            "keyA": [0.1, 0.2],
            "keyB": [0.3, 0.4],
            "keyC": [0.5, 0.6],
        },
        index=INDEX,
    )
    df.index.name = INDEX_NAME
    return df


@pytest.fixture
def data_with_npz(tmp_path):
    """DataFrame referencing an external NPZ file."""
    df = pd.DataFrame(
        {
            "keyA": [1.0, 2.0, 3.0],
            "keyB": [4.0, 5.0, 6.0],
            "keyC": [7.0, 8.0, 9.0],
            "keyNPZ": [None, None, None],
        },
        index=["idx1", "idx2", "idx3"],
    )
    df.index.name = INDEX_NAME

    # Create NPZ file
    npz_path = tmp_path / "data.npz"
    array = np.arange(10, 110, 10)
    np.savez(npz_path, array)

    df.loc["idx2", "keyNPZ"] = str(npz_path)

    return df, array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_data(path):
    reader = DataReader({"data": str(path), "datavar": ""})
    return reader.get_data()


def read_data_and_variance(data_path, var_path):
    reader = DataReader(
        {"data": str(data_path), "datavar": str(var_path)}
    )
    data = reader.get_data()
    variance = reader.get_variance(data)
    return data, variance


# ---------------------------------------------------------------------------
# Tests: Data Reading
# ---------------------------------------------------------------------------

class TestDataReading:

    def test_read_pickle(self, tmp_path, base_data):
        path = tmp_path / "data.pkl"
        base_data.to_pickle(path)

        result = read_data(path)

        assert isinstance(result, PETDataFrame)
        assert_frame_equal(result, base_data)

    def test_read_csv(self, tmp_path, base_data):
        path = tmp_path / "data.csv"
        base_data.to_csv(path)

        result = read_data(path)

        assert isinstance(result, PETDataFrame)
        assert_frame_equal(result, base_data)


# ---------------------------------------------------------------------------
# Tests: Variance Handling
# ---------------------------------------------------------------------------

class TestVarianceHandling:

    def test_absolute_variance_pickle(
        self, tmp_path, base_data, abs_variance, expected_variance
    ):
        data_path = tmp_path / "data.pkl"
        var_path = tmp_path / "var.pkl"

        base_data.to_pickle(data_path)
        abs_variance.to_pickle(var_path)

        _, result = read_data_and_variance(data_path, var_path)

        assert isinstance(result, PETDataFrame)
        assert_frame_equal(result, expected_variance, atol=1e-12, rtol=1e-12)

    def test_relative_variance_pickle(
        self, tmp_path, base_data, rel_variance, expected_variance
    ):
        data_path = tmp_path / "data.pkl"
        var_path = tmp_path / "var.pkl"

        base_data.to_pickle(data_path)
        rel_variance.to_pickle(var_path)

        _, result = read_data_and_variance(data_path, var_path)

        assert isinstance(result, PETDataFrame)
        assert_frame_equal(result, expected_variance, atol=1e-12, rtol=1e-12)

    def test_absolute_variance_csv(
        self, tmp_path, base_data, abs_variance, expected_variance
    ):
        data_path = tmp_path / "data.csv"
        var_path = tmp_path / "var.csv"

        base_data.to_csv(data_path)
        abs_variance.to_csv(var_path)

        _, result = read_data_and_variance(data_path, var_path)

        assert isinstance(result, PETDataFrame)
        assert_frame_equal(result, expected_variance, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tests: NPZ Integration
# ---------------------------------------------------------------------------

class TestNPZHandling:

    def test_npz_loading(self, tmp_path, data_with_npz):
        df, expected_array = data_with_npz

        path = tmp_path / "data.pkl"
        df.to_pickle(path)

        result = read_data(path)

        flattened_expected = np.concatenate([
            [1.0, 4.0, 7.0],
            [2.0, 5.0, 8.0],
            expected_array,
            [3.0, 6.0, 9.0],
        ])

        assert isinstance(result, PETDataFrame)

        # Check NPZ content
        np.testing.assert_array_equal(
            result.loc["idx2", "keyNPZ"],
            expected_array,
        )

        # Check flattened representation
        assert result.to_matrix().shape == (len(flattened_expected),)
        np.testing.assert_array_equal(result.to_matrix(), flattened_expected)






def test_the_ensemble_observation_vector_matches_the_frame_flatten(tmp_path, monkeypatch):
    """`obs_vector` replaces `data_df.to_matrix()` on every scheme; the two must agree."""
    from input_output import read_config
    from pipt.ensembles import AssimilationEnsemble
    from simulator.vanderpol import VanDerPolOscillator
    from test_numerical_characterisation import _write_config, _write_synthetic_case

    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=8)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("layout", "esmda", "approx", report_points, ne=8))
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    np.testing.assert_array_equal(ensemble.obs_vector, ensemble.data_df.to_matrix())
    assert ensemble.data_layout.nd == ensemble.obs_vector.shape[0]


def test_the_observation_variance_matches_the_frame_flatten_and_rejects_nan(tmp_path, monkeypatch):
    """`obs_variance` replaces construct_data_cov, whose NaN filter silently shortened the covariance."""
    from input_output import read_config
    from pipt.ensembles import AssimilationEnsemble
    from simulator.vanderpol import VanDerPolOscillator
    from test_numerical_characterisation import _write_config, _write_synthetic_case

    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=8)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("variance", "esmda", "approx", report_points, ne=8))
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    np.testing.assert_array_equal(ensemble.obs_variance, ensemble.data_var_df.to_matrix())
    assert ensemble.obs_variance.shape == ensemble.obs_vector.shape

    label, column = ensemble.data_var_df.index[2], ensemble.data_var_df.columns[0]
    ensemble.data_var_df.at[label, column] = np.nan
    with pytest.raises(ValueError, match="variance is NaN"):
        ensemble._observation_variance()
