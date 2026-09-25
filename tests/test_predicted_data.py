"""`PredictedData` is filled straight from the members' outputs, in the layout's row order."""

import numpy as np
import pandas as pd
import pytest

from misc.structures import DataLayout, PETDataFrame, PredictedData


def _observations():
    df = pd.DataFrame(index=pd.Index([10, 20, 30], name="time"), columns=["WOPR", "SEIS"], dtype=object)
    df.at[10, "WOPR"], df.at[20, "WOPR"], df.at[30, "WOPR"] = 1.0, 2.0, 3.0
    df.at[10, "SEIS"] = np.array([0.1, 0.2])
    df.at[20, "SEIS"] = None                      # not observed at this label
    df.at[30, "SEIS"] = np.array([0.3, 0.4])
    return PETDataFrame.from_pandas(df)


def _records(member):
    """What a simulator returns: one dict per report point, here at times 10, 20, 30 plus an extra one."""
    return [{"WOPR": 1.0 + member, "SEIS": np.array([0.1, 0.2]) + member, "EXTRA": 99.0},
            {"WOPR": 2.0 + member, "SEIS": np.array([0.5, 0.6]) + member, "EXTRA": 99.0},   # SEIS here is unobserved: ignored
            {"WOPR": 3.0 + member, "SEIS": np.array([0.3, 0.4]) + member, "EXTRA": 99.0},
            {"WOPR": 4.0 + member, "SEIS": np.array([0.7, 0.8]) + member, "EXTRA": 99.0}]   # a report point nobody observed


LAYOUT = DataLayout.from_frame(_observations())
POSITION = {10: 0, 20: 1, 30: 2}


def test_records_fill_the_layout_rows_and_nothing_else():
    pred = PredictedData.from_members(LAYOUT, [_records(0), _records(10)], position=POSITION)
    assert pred.matrix.shape == (LAYOUT.nd, 2) == (7, 2)
    np.testing.assert_array_equal(pred.matrix[:, 0], [1.0, 0.1, 0.2, 2.0, 3.0, 0.3, 0.4])
    np.testing.assert_array_equal(pred.matrix[:, 1], pred.matrix[:, 0] + 10)


def test_frames_per_member_fill_the_same_way():
    frames = [pd.DataFrame.from_records(_records(m), index=[10, 20, 30, 40]) for m in (0, 10)]
    from_frames = PredictedData.from_members(LAYOUT, frames)
    from_records = PredictedData.from_members(LAYOUT, [_records(0), _records(10)], position=POSITION)
    np.testing.assert_array_equal(from_frames.matrix, from_records.matrix)


def test_a_member_missing_an_observed_type_or_size_is_reported_not_dropped():
    broken = _records(0)
    del broken[2]["WOPR"]
    with pytest.raises(KeyError, match="no 'WOPR' at 30"):
        PredictedData.from_members(LAYOUT, [broken], position=POSITION)
    short = _records(0)
    short[0]["SEIS"] = np.array([0.1])
    with pytest.raises(ValueError, match="has 1 values; the observation has 2"):
        PredictedData.from_members(LAYOUT, [short], position=POSITION)


def test_scaling_matches_the_frame_scaling():
    # The frame's max-min scaling handles scalar cells (a minimum and maximum per data type), so compare on those.
    obs = PETDataFrame.from_pandas(pd.DataFrame({"WOPR": [1.0, 2.0, 3.0], "WWPR": [5.0, 7.0, 9.0]},
                                                index=pd.Index([10, 20, 30], name="time")))
    layout = DataLayout.from_frame(obs)
    obs.scale("max-min")
    records = [[{"WOPR": 1.0 + m, "WWPR": 5.0 + 2 * m}, {"WOPR": 2.0 + m, "WWPR": 7.0 + 2 * m}, {"WOPR": 3.0 + m, "WWPR": 9.0 + 2 * m}]
               for m in (0.0, 0.5)]
    pred = PredictedData.from_members(layout, records, position=POSITION, scale=(obs.scale_min, obs.scale_max))

    # The frame path: merge the members into cells, scale with the same min/max, flatten.
    frames = [pd.DataFrame.from_records(r, index=[10, 20, 30]) for r in records]
    merged = PETDataFrame.merge_dataframes(frames)
    merged.scale("max-min", minimum=obs.scale_min, maximum=obs.scale_max)
    np.testing.assert_array_equal(pred.matrix, layout.matrix(merged, 2))


def test_the_view_and_member_selection():
    pred = PredictedData.from_members(LAYOUT, [_records(m) for m in (0, 10, 20)], position=POSITION)
    view = pred.to_frame()
    assert view.at[10, "WOPR"].shape == (3,) and view.at[30, "SEIS"].shape == (2, 3) and view.at[20, "SEIS"] is None
    np.testing.assert_array_equal(view.to_matrix(), pred.matrix)
    picked = pred.take_members([2, 0])
    np.testing.assert_array_equal(picked.matrix[:, 0], pred.matrix[:, 2])
    assert pred.rows_of("SEIS") == [slice(1, 3), slice(5, 7)]
