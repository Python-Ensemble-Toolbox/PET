"""`DataLayout` is the one order every data array follows, derived from the observed frame."""

import numpy as np
import pandas as pd
import pytest

from misc.structures import DataLayout, PETDataFrame


def _frame(missing=False):
    df = pd.DataFrame(index=pd.Index([1, 2, 3], name="time"), columns=["WOPR", "SEIS"], dtype=object)
    df.at[1, "WOPR"], df.at[2, "WOPR"], df.at[3, "WOPR"] = 10.0, 20.0, 30.0
    df.at[1, "SEIS"] = np.array([1.0, 2.0, 3.0, 4.0])
    df.at[2, "SEIS"] = None if missing else np.array([5.0, 6.0, 7.0, 8.0])
    df.at[3, "SEIS"] = np.nan
    return PETDataFrame.from_pandas(df)


def test_rows_follow_the_frame_walk_and_skip_empty_cells():
    layout = DataLayout.from_frame(_frame(missing=True))
    assert [(r.label, r.datatype, r.start, r.stop) for r in layout.rows] == [
        (1, "WOPR", 0, 1), (1, "SEIS", 1, 5), (2, "WOPR", 5, 6), (3, "WOPR", 6, 7)]
    assert layout.nd == 7
    assert layout.row(1, "SEIS").size == 4
    with pytest.raises(KeyError):
        layout.row(2, "SEIS")
    assert list(layout.row_datatypes()) == ["WOPR", "SEIS", "SEIS", "SEIS", "SEIS", "WOPR", "WOPR"]


@pytest.mark.parametrize("missing", [False, True])
def test_the_vector_is_what_the_frame_flatten_produced(missing):
    frame = _frame(missing)
    np.testing.assert_array_equal(DataLayout.from_frame(frame).vector(frame), frame.to_matrix())


def test_an_ensemble_frame_reads_back_as_the_matrix_and_the_matrix_views_as_the_frame():
    layout = DataLayout.from_frame(_frame())
    ne = 3
    matrix = np.arange(layout.nd * ne, dtype=float).reshape(layout.nd, ne)

    view = layout.to_frame(matrix)
    assert view.is_ensemble
    assert view.at[1, "WOPR"].shape == (ne,) and view.at[1, "SEIS"].shape == (4, ne)
    assert view.at[3, "SEIS"] is None
    np.testing.assert_array_equal(view.to_matrix(), matrix)               # the legacy flatten agrees
    np.testing.assert_array_equal(layout.matrix(view, ne), matrix)        # and so does the layout read


def test_an_observation_vector_views_as_the_original_frame():
    frame = _frame()
    layout = DataLayout.from_frame(frame)
    view = layout.to_frame(layout.vector(frame))
    assert view.at[2, "WOPR"] == 20.0
    np.testing.assert_array_equal(view.at[2, "SEIS"], [5.0, 6.0, 7.0, 8.0])
    assert view.at[3, "SEIS"] is None
    assert view.index.name == "time" and list(view.columns) == ["WOPR", "SEIS"]
