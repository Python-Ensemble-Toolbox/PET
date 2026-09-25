"""``remove_outliers`` must move a replaced member's adjoint together with its
state and predictions. The adjoint filter used to sit after the ``return``
statement, so it never ran and a resampled member kept a stranger's gradient."""

import numpy as np
import pandas as pd

from misc.structures import DataLayout, PredictedData
from misc.structures.structures import PETDataFrame
from pipt.ensembles.forecast import OutlierMixin

# The 4-sigma rule can only flag a lone outlier when (ne - 1) / sqrt(ne) > 4.
NE = 25
NX = 3
STEPS = ["t1", "t2"]
TRUTH = {"t1": 1.0, "t2": 2.0}


def _frame(cells, is_ensemble):
    df = pd.DataFrame({"obs": [cells[s] for s in STEPS]}, index=pd.Index(STEPS, name="steps"))
    return PETDataFrame.from_pandas(df, is_ensemble=is_ensemble)


def _predictions(outlier_member=0):
    """Every member predicts the truth except one, which is far off."""
    cells = {}
    for step in STEPS:
        pred = np.full(NE, TRUTH[step])
        if outlier_member is not None:
            pred[outlier_member] = 100.0
        cells[step] = pred
    return cells


class Host(OutlierMixin):
    """The attributes remove_outliers reads, and nothing else."""

    def __init__(self, pred_cells, with_adjoints):
        self.ne = NE
        self.rng = np.random
        self.logger = lambda *args, **kwargs: None
        self.data_df = _frame(TRUTH, is_ensemble=False)
        self.data_var_df = _frame({s: 1.0 for s in STEPS}, is_ensemble=False)
        self.data_layout = DataLayout.from_frame(self.data_df)
        self.obs_vector = self.data_layout.vector(self.data_df)
        self.obs_variance = self.data_layout.vector(self.data_var_df)
        self.pred_data = PredictedData.from_frame(self.data_layout, _frame(pred_cells, is_ensemble=True), NE)
        # The full forecast as the members returned it: one list of records per member.
        self.member_outputs = [[[{"obs": pred_cells[s][j]} for s in STEPS] for j in range(NE)]]
        self._sim_data = None
        self.sim_data = None
        # Adjoint of member j is 10*j in every entry, so the member it came
        # from can be read straight off the array: (nd, nx, ne).
        self.adjoints = np.tile(10.0 * np.arange(NE), (len(STEPS), NX, 1)) if with_adjoints else None
        self.member_adjoints = None


def _state():
    # Column j holds j everywhere, so the member a column came from is readable.
    return np.tile(np.arange(NE, dtype=float), (NX, 1))


def test_adjoints_follow_the_resampled_member():
    pred_cells = _predictions(outlier_member=0)
    host = Host(pred_cells, with_adjoints=True)
    enX = _state()

    np.random.seed(1)
    new_enX = host.remove_outliers(enX)

    k = int(new_enX[0, 0])  # the member that replaced the outlier
    assert k != 0
    np.testing.assert_array_equal(new_enX[:, 0], enX[:, k])
    np.testing.assert_array_equal(new_enX[:, 1:], enX[:, 1:])
    np.testing.assert_array_equal(host.adjoints[:, :, 0], 10.0 * k)
    np.testing.assert_array_equal(host.adjoints[:, :, 1:], np.tile(10.0 * np.arange(1, NE), (len(STEPS), NX, 1)))
    for i, step in enumerate(STEPS):
        assert host.pred_data.to_frame().loc[step, "obs"][0] == pred_cells[step][k]
        assert host.member_outputs[0][0][i]["obs"] == pred_cells[step][k]      # the raw outputs follow too


def test_without_adjoints_the_state_and_predictions_are_still_resampled():
    pred_cells = _predictions(outlier_member=0)
    host = Host(pred_cells, with_adjoints=False)

    np.random.seed(1)
    new_enX = host.remove_outliers(_state())

    assert host.adjoints is None
    assert int(new_enX[0, 0]) != 0
    for step in STEPS:
        assert host.pred_data.to_frame().loc[step, "obs"][0] == TRUTH[step]


def test_no_outliers_returns_the_same_state_object():
    host = Host(_predictions(outlier_member=None), with_adjoints=True)
    enX = _state()
    assert host.remove_outliers(enX) is enX


def test_a_forecast_loaded_as_a_frame_is_resampled_cell_by_cell():
    """A forecast read from a restart file exists only as a frame; its empty
    cells are None and the filter used to call .ndim on them."""
    pred_cells = _predictions(outlier_member=0)
    host = Host(pred_cells, with_adjoints=False)
    host.member_outputs = None
    host.sim_data = _frame(pred_cells, is_ensemble=True)
    host.sim_data.loc["t2", "obs"] = None

    np.random.seed(1)
    new_enX = host.remove_outliers(_state())

    k = int(new_enX[0, 0])
    assert k != 0
    assert host.sim_data.loc["t1", "obs"][0] == pred_cells["t1"][k]
    assert host.sim_data.loc["t2", "obs"] is None
