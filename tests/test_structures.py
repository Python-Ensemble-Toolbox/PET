"""
Comprehensive tests for PETDataFrame and StateLayout.

This suite preserves:
- Exact numerical correctness
- Deterministic behavior
- Full operator coverage
- Field data edge cases
- Scaling consistency
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from misc.structures import StateLayout
from misc.structures.structures import PETDataFrame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NPARAMS = 3
NX = 8
NROWS = 2
NCOLS = 3
NY = NROWS * NCOLS
NE = 10

INDEX = ["idx1", "idx2"]
INDEX_NAME = "index"


# ---------------------------------------------------------------------------
# Deterministic MultiIndex Data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multicolumn_ensemble():
    """Generate deterministic ensemble of multi-column DataFrames."""
    np.random.seed(404)

    dfs = []
    for _ in range(NE):
        data = {}
        for key in ("keyA", "keyB", "keyC"):
            for param in ("param1", "param2", "param3"):
                data[(key, param)] = [
                    np.random.rand(NX) for _ in range(NROWS)
                ]

        df = pd.DataFrame(data, index=INDEX)
        df.columns = pd.MultiIndex.from_tuples(data.keys())
        df.index.name = INDEX_NAME
        dfs.append(df)

    return dfs


@pytest.fixture
def multicolumn_df(multicolumn_ensemble):
    return multicolumn_ensemble[0]


@pytest.fixture
def ensemble_singlelevel(multicolumn_ensemble):
    return [
        PETDataFrame._to_singlelevel_columns(df)
        for df in multicolumn_ensemble
    ]


# ---------------------------------------------------------------------------
# PETDataFrame: Basic
# ---------------------------------------------------------------------------

class TestPETDataFrameBasic:

    def setup_method(self):
        self.data = {
            "keyA": [1.0, 2.0],
            "keyB": [3.0, 4.0],
            "keyC": [5.0, 6.0],
        }

        self.df = pd.DataFrame(self.data, index=INDEX)
        self.df.index.name = INDEX_NAME
        self.df.attrs["units"] = {
            k: f"unit:{k}" for k in self.data
        }

    def test_from_pandas(self):
        pdf = PETDataFrame.from_pandas(self.df)

        expected = PETDataFrame(self.data, index=INDEX)
        expected.index.name = INDEX_NAME

        assert pdf.equals(expected)

    def test_attrs_preserved(self):
        pdf = PETDataFrame.from_pandas(self.df)
        assert pdf.attrs["units"] == self.df.attrs["units"]

    def test_to_matrix(self):
        pdf = PETDataFrame.from_pandas(self.df)

        vec = pdf.to_matrix(squeeze=False)
        vec_sq = pdf.to_matrix(squeeze=True)

        expected = np.array([1, 3, 5, 2, 4, 6], dtype=float)

        assert vec.shape == (NY, 1)
        assert np.array_equal(vec[:, 0], expected)

        assert vec_sq.shape == (NY,)
        assert np.array_equal(vec_sq, expected)

    def test_return_types(self):
        pdf = PETDataFrame.from_pandas(self.df)

        assert isinstance(pdf.copy(), PETDataFrame)
        assert isinstance(pdf.loc[["idx1"]], PETDataFrame)
        assert isinstance(pdf + 1, PETDataFrame)

# ---------------------------------------------------------------------------
# PETDataFrame: Filtering
# ---------------------------------------------------------------------------

class TestFilterDataFrame:

    def setup_method(self):
        self.data = {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
        }
        self.index = pd.Index(["x", "y", "z"], name="idx")
        self.df = PETDataFrame(self.data, index=self.index)

    def test_filter_columns(self):
        filtered = self.df.filter_dataframe(columns=["A", "C"])
        assert list(filtered.columns) == ["A", "C"]
        assert np.all(filtered["A"] == [1, 2, 3])
        assert np.all(filtered["C"] == [7, 8, 9])
        assert isinstance(filtered, PETDataFrame)

    def test_filter_index(self):
        filtered = self.df.filter_dataframe(index=["x", "z"])
        assert list(filtered.index) == ["x", "z"]
        assert np.all(filtered.loc["x"] == [1, 4, 7])
        assert np.all(filtered.loc["z"] == [3, 6, 9])
        assert isinstance(filtered, PETDataFrame)

    def test_filter_both(self):
        filtered = self.df.filter_dataframe(columns=["B"], index=["y"])
        assert list(filtered.columns) == ["B"]
        assert list(filtered.index) == ["y"]
        assert filtered.at["y", "B"] == 5
        assert isinstance(filtered, PETDataFrame)

    def test_filter_none(self):
        filtered = self.df.filter_dataframe()
        pd.testing.assert_frame_equal(filtered, self.df)
        assert isinstance(filtered, PETDataFrame)

    def test_filter_wrong_index_dtype(self):
        wrong_index = pd.Index([0, 1], dtype=int)
        with pytest.raises(ValueError):
            self.df.filter_dataframe(index=wrong_index)

    def test_filter_missing_label(self):
        with pytest.raises(ValueError):
            self.df.filter_dataframe(index=["x", "missing"])

    def test_filter_compatible_index_dtype(self):
        # datetime.date labels select fine against a DatetimeIndex even though
        # the dtypes differ (object vs datetime64[ns]).
        dates = pd.to_datetime(["2023-02-05", "2024-03-11", "2025-04-15"])
        df = PETDataFrame({"A": [1, 2, 3]}, index=dates)
        wanted = pd.Index([dt.date(2023, 2, 5), dt.date(2025, 4, 15)])

        filtered = df.filter_dataframe(index=wanted)

        assert list(filtered["A"]) == [1, 3]
        assert isinstance(filtered, PETDataFrame)

    def test_return_type(self):
        filtered = self.df.filter_dataframe(columns=["A"])
        assert isinstance(filtered, PETDataFrame)

# ---------------------------------------------------------------------------
# Jacobian (Multi-column)
# ---------------------------------------------------------------------------

class TestMultiColumnJacobian:

    def test_to_series(self, multicolumn_df):
        pdf = PETDataFrame.from_pandas(multicolumn_df)
        series = pdf.to_series()

        assert isinstance(series, pd.Series)
        assert series.shape == (NROWS * NCOLS * NPARAMS,)

    def test_to_matrix_exact(self, multicolumn_df):
        pdf = PETDataFrame.from_pandas(multicolumn_df)
        matrix = pdf.to_matrix(is_jacobian=True)

        expected_rows = []
        keys = ("keyA", "keyB", "keyC")
        params = ("param1", "param2", "param3")

        for r in range(NROWS):
            for key in keys:
                row = np.concatenate([
                    multicolumn_df[(key, param)].iloc[r]
                    for param in params
                ])
                expected_rows.append(row)

        expected = np.stack(expected_rows)

        assert matrix.shape == (NY, NX * NPARAMS)
        assert np.array_equal(matrix, expected)


# ---------------------------------------------------------------------------
# Ensemble handling
# ---------------------------------------------------------------------------

class TestEnsembleJacobian:

    def test_merge(self, ensemble_singlelevel):
        merged = PETDataFrame.merge_dataframes(ensemble_singlelevel)

        assert merged.iloc[0]["keyA"].shape == (NX * NPARAMS, NE)

    def test_matrix_shape(self, ensemble_singlelevel):
        merged = PETDataFrame.merge_dataframes(ensemble_singlelevel)
        matrix = merged.to_matrix(is_jacobian=True)

        assert matrix.shape == (NY, NX * NPARAMS, NE)

    def test_multi_vs_single_consistency(
        self, multicolumn_ensemble, ensemble_singlelevel
    ):
        merged_multi = PETDataFrame.merge_dataframes(multicolumn_ensemble)
        merged_single = PETDataFrame.merge_dataframes(ensemble_singlelevel)

        mat1 = PETDataFrame.to_matrix(merged_multi, is_jacobian=True)
        mat2 = PETDataFrame.to_matrix(merged_single, is_jacobian=True)

        assert np.array_equal(mat1, mat2)


# ---------------------------------------------------------------------------
# Field data
# ---------------------------------------------------------------------------

class TestFieldData:

    def setup_method(self):
        self.pdf1 = PETDataFrame(
            {
                "keyScalar1": [1, 2, 3],
                "keyScalar2": [4, 5, 6],
                "keyField": [None, np.array([7, 8, 9, 10]), None],
            },
            index=["idx1", "idx2", "idx3"],
        )

        self.pdf2 = PETDataFrame(
            {
                "keyScalar1": [10, 20, 30],
                "keyScalar2": [40, 50, 60],
                "keyField": [None, np.array([70, 80, 90, 100]), None],
            },
            index=["idx1", "idx2", "idx3"],
        )

    def test_filtered_unfiltered_vectors(self):
        vec_f = self.pdf1.to_matrix(filter=True, squeeze=True)
        vec_u = self.pdf1.to_matrix(filter=False, squeeze=True)

        expected_f = np.array([1,4,2,5,7,8,9,10,3,6], dtype=float)
        expected_u = np.array(
            [1,4,None,2,5,7,8,9,10,3,6,None], dtype=object
        )

        assert np.array_equal(vec_f, expected_f)
        assert np.array_equal(vec_u, expected_u)

    def test_ensemble_matrix(self):
        merged = PETDataFrame.merge_dataframes([self.pdf1, self.pdf2])

        mat_f = merged.to_matrix(filter=True)
        mat_u = merged.to_matrix(filter=False)

        expected_f = np.array([
            [1,10],[4,40],[2,20],[5,50],
            [7,70],[8,80],[9,90],[10,100],
            [3,30],[6,60]
        ])

        expected_u = np.array([
            [1,10],[4,40],[None,None],[2,20],[5,50],
            [7,70],[8,80],[9,90],[10,100],
            [3,30],[6,60],[None,None]
        ], dtype=object)

        assert np.array_equal(mat_f, expected_f)
        assert np.array_equal(mat_u, expected_u)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

class TestScaling:

    def setup_method(self):
        np.random.seed(404)

        self.data = PETDataFrame(
            {k: 10*np.random.rand(5) for k in ("keyA","keyB","keyC")}
        )
        self.var = PETDataFrame(
            {k: 0.1*np.random.rand(5) for k in ("keyA","keyB","keyC")}
        )
        self.jac = PETDataFrame(
            {
                (k,"param1"): [50*np.random.rand(NX,NE) for _ in range(5)]
                for k in ("keyA","keyB","keyC")
            }
        )

    def test_max_min(self):
        scaled = self.data.copy()
        scaled.scale(type="max-min")

        inv = scaled.copy()
        inv.invert_scale(type="max-min")

        assert scaled.is_scaled
        assert np.all((scaled >= 0) & (scaled <= 1))
        pd.testing.assert_frame_equal(inv, self.data)

    def test_variance(self):
        scaled = self.data.copy()
        scaled.scale(type="max-min")

        rng = scaled.scale_max - scaled.scale_min

        expected = self.var / (rng**2)

        var_scaled = self.var.copy()
        var_scaled.scale(type="max-min", minimum=0, maximum=rng**2)

        inv = var_scaled.copy()
        inv.invert_scale(type="max-min")

        pd.testing.assert_frame_equal(var_scaled, expected)
        pd.testing.assert_frame_equal(inv, self.var)

    def test_jacobian(self):
        scaled = self.data.copy()
        scaled.scale(type="max-min")

        rng = scaled.scale_max - scaled.scale_min

        expected = self.jac.div(rng, axis="columns", level=0)

        jac_scaled = self.jac.copy()
        jac_scaled.scale(type="max-min", minimum=0, maximum=rng)

        inv = jac_scaled.copy()
        inv.invert_scale(type="max-min")

        pd.testing.assert_frame_equal(jac_scaled, expected)
        pd.testing.assert_frame_equal(inv, self.jac)


# ---------------------------------------------------------------------------
# StateLayout
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    data = np.arange(1, NX * NPARAMS * NE + 1, dtype=float).reshape(NX * NPARAMS, NE)
    layout = StateLayout({f"key{i+1}": (i * NX, (i + 1) * NX) for i in range(NPARAMS)})
    return data, layout


class TestStateLayout:
    def test_shapes_and_variables(self, state):
        data, layout = state
        assert layout.nx == NX * NPARAMS and layout.variables == tuple(f"key{i+1}" for i in range(NPARAMS))
        assert layout.rows("key2") == slice(NX, 2 * NX)

    def test_dict_conversion_is_a_view_of_the_rows(self, state):
        data, layout = state
        d = layout.to_dict(data)
        assert all(v.shape == (NX, NE) for v in d.values())
        np.testing.assert_array_equal(d["key2"], data[NX:2 * NX])

    def test_member_dicts_round_trip_through_from_dict(self, state):
        data, layout = state
        members = layout.member_dicts(data)
        assert len(members) == NE and members[0]["key1"].shape == (NX,)
        rebuilt, rebuilt_layout = StateLayout.from_dict(
            {key: np.column_stack([m[key] for m in members]) for key in layout.variables})
        np.testing.assert_array_equal(rebuilt, data)
        assert rebuilt_layout == layout

    def test_from_dict_keeps_only_the_first_ne_columns_when_asked(self, state):
        data, layout = state
        matrix, _ = StateLayout.from_dict(layout.to_dict(data), ne=2)
        np.testing.assert_array_equal(matrix, data[:, :2])
        with pytest.raises(ValueError):
            StateLayout.from_dict({})

    def test_clip_by_variable_pair_and_list(self, state):
        data, layout = state
        by_variable = data.copy()
        layout.clip(by_variable, {"key1": (2.0, 4.0), "key2": (None, None)})
        np.testing.assert_array_equal(by_variable[:NX], np.clip(data[:NX], 2.0, 4.0))
        np.testing.assert_array_equal(by_variable[NX:], data[NX:])
        everywhere = data.copy()
        layout.clip(everywhere, (0.0, 3.0))
        assert everywhere.max() == 3.0
        as_list = data.copy()
        layout.clip(as_list, [(None, 1.0)] + [(None, None)] * (NPARAMS - 1))
        assert as_list[:NX].max() == 1.0 and np.array_equal(as_list[NX:], data[NX:])
        with pytest.raises(ValueError):
            layout.clip(data.copy(), "no")


# ---------------------------------------------------------------------------
# StateLayout: generation from prior info
# ---------------------------------------------------------------------------

class TestFromPriorInfo:
    """A prior with more than one variable used to raise ``KeyError``: the
    second variable's offset was read from an ``idX`` entry that did not exist
    yet, so no multi-variable prior could be generated at all."""

    @staticmethod
    def _scalar(mean, variance):
        # One cell, one layer: exercises the scalar path of gen_real and keeps
        # the field-covariance machinery out of the picture.
        return {"mean": [mean], "variance": [variance], "nx": 1, "ny": 1, "nz": 1}

    def test_variables_are_stacked_with_consecutive_indices(self):
        prior_info = {"a": self._scalar(1.0, 0.1), "b": self._scalar(2.0, 0.2), "c": self._scalar(3.0, 0.3)}
        np.random.seed(0)
        enX, layout = StateLayout.from_prior_info(prior_info, ne=NE, save=False)
        assert enX.shape == (3, NE)
        assert layout.indices == {"a": (0, 1), "b": (1, 2), "c": (2, 3)}

    def test_indices_address_the_rows_of_their_own_variable(self):
        prior_info = {"a": self._scalar(1.0, 1e-12), "b": self._scalar(2.0, 1e-12)}
        np.random.seed(0)
        enX, layout = StateLayout.from_prior_info(prior_info, ne=NE, save=False)
        as_dict = layout.to_dict(enX)
        np.testing.assert_allclose(as_dict["a"], 1.0, atol=1e-4)
        np.testing.assert_allclose(as_dict["b"], 2.0, atol=1e-4)
