"""A weighted adjoint reaches the scheme as an ``(nx, ne)`` gradient of the data misfit.

A simulator in weighted-adjoint mode is handed weights ``w`` and returns the
gradient of ``sum_i w_i d_i`` per parameter, one adjoint solve per member.
With the weights of the data misfit, ``w = C_d^-1 (d_obs - g(x))``, that is
``G^T C_d^-1 (d_obs - g(x))``.
"""

import numpy as np
import pandas as pd
import pytest

from misc.structures import DataLayout
from pipt.ensembles import AssimilationEnsemble, DataMisfitWeights, ForecastMixin

LABELS = [1, 2, 3, 4]
SIGMA = 0.5
NE = 5


# ---------------------------------------------------------------------- #
# The weights
# ---------------------------------------------------------------------- #
def _layout_and_obs():
    obs = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [0.5, np.nan, -1.0]}, index=pd.Index([10, 20, 30], name="t"))
    layout = DataLayout.from_frame(obs)
    return layout, layout.vector(obs)


def test_weights_vanish_at_the_observations():
    layout, obs = _layout_and_obs()
    weights = DataMisfitWeights(layout, obs, np.full(layout.nd, 0.1))
    member = pd.DataFrame(layout.to_frame(obs)).astype(float)
    np.testing.assert_allclose(weights.vector(member), 0.0)


@pytest.mark.parametrize("scaled", [False, True])
def test_weights_are_minus_half_the_misfit_gradient(scaled):
    """w = -1/2 dPhi/dd for Phi = r^T C^-1 r, also when the data were max-min scaled."""
    layout, obs = _layout_and_obs()
    rng = np.random.default_rng(6)
    A = rng.standard_normal((layout.nd, layout.nd))
    cov = A @ A.T + layout.nd * np.eye(layout.nd)
    scale = (pd.Series({"a": 0.5, "b": -2.0}), pd.Series({"a": 3.0, "b": 4.0})) if scaled else None
    weights = DataMisfitWeights(layout, obs, cov, scale=scale)

    raw = obs + rng.standard_normal(layout.nd)

    def misfit(values):
        r = weights.residual(pd.DataFrame(layout.to_frame(values)).astype(float))
        return r @ np.linalg.solve(cov, r)

    h = 1e-6
    gradient = np.array([(misfit(raw + h * e) - misfit(raw - h * e)) / (2 * h) for e in np.eye(layout.nd)])
    member = pd.DataFrame(layout.to_frame(raw)).astype(float)
    np.testing.assert_allclose(weights.vector(member), -0.5 * gradient, rtol=1e-6)


def test_weights_frame_holds_each_observed_cell_and_nan_elsewhere():
    layout, obs = _layout_and_obs()
    weights = DataMisfitWeights(layout, obs, np.ones(layout.nd))
    member = pd.DataFrame(layout.to_frame(obs - 1.0)).astype(float)
    frame = weights(member, 0)
    assert frame.loc[10, "a"] == pytest.approx(1.0) and frame.loc[30, "b"] == pytest.approx(1.0)
    assert np.isnan(frame.loc[30, "a"]) and np.isnan(frame.loc[20, "b"])


# ---------------------------------------------------------------------- #
# Weighted adjoints into the state
# ---------------------------------------------------------------------- #
def _ensemble_with_two_states():
    class Holder(ForecastMixin):
        idX = {"permx": (0, 3), "poro": (3, 5)}

    return Holder()


def test_weighted_gradients_are_placed_by_name_not_order():
    holder = _ensemble_with_two_states()
    member = pd.Series({"poro": np.array([4.0, 5.0]), "log_permx": np.array([1.0, 2.0, 3.0])}, dtype=object)
    np.testing.assert_array_equal(holder._weighted_gradient(member), [1, 2, 3, 4, 5])


@pytest.mark.parametrize("member", [
    pd.Series({"permx": np.zeros(3)}, dtype=object),                                         # one missing
    pd.Series({"permx": np.zeros(3), "permy": np.zeros(2)}, dtype=object),                  # wrong name
    pd.Series({"permx": np.zeros(3), "log_permx": np.zeros(3), "poro": np.zeros(2)}, dtype=object),  # twice
    pd.Series({"permx": np.zeros(2), "poro": np.zeros(3)}, dtype=object),                   # wrong sizes
])
def test_weighted_gradients_that_do_not_match_the_state_are_refused(member):
    with pytest.raises(ValueError, match="weighted adjoint"):
        _ensemble_with_two_states()._weighted_gradient(member)


# ---------------------------------------------------------------------- #
# From the simulator to the ensemble
# ---------------------------------------------------------------------- #
class LinearAdjointSim:
    """``d = H x``, returning the weighted adjoint ``H^T w`` for the weights it is handed."""

    def __init__(self, H):
        self.H = H
        self.input_dict = {"parallel": 1}
        self.compute_adjoints = True
        self.adjoint_weights = None

    def run_fwd_sim(self, state, member_index):
        output = pd.DataFrame({"value": self.H @ np.ravel(state["x"])}, index=pd.Index(LABELS, name="position"))
        weights = self.adjoint_weights(output.copy(), member_index)
        w = weights.reindex(index=output.index, columns=output.columns).fillna(0.0)["value"].to_numpy()
        return output, pd.Series({"x": self.H.T @ w}, dtype=object)


def test_the_ensemble_carries_the_gradient_of_the_data_misfit_for_every_member():
    rng = np.random.default_rng(3)
    H = np.array([[1.0, 0.5], [0.2, 1.0], [1.0, -1.0], [0.3, 0.3]])
    obs = H @ np.array([1.0, -0.5]) + SIGMA * rng.standard_normal(len(LABELS))
    pd.DataFrame({"value": obs}, index=pd.Index(LABELS, name="position")).to_pickle("true_data.pkl")
    pd.DataFrame({"value": [["abs", SIGMA ** 2]] * len(LABELS)},
                 index=pd.Index(LABELS, name="position")).to_pickle("var.pkl")
    np.savez("prior.npz", x=0.2 + 0.4 * rng.standard_normal((2, NE)))
    cfg_da = {"scheme": "esmda", "analysis": "approx", "obsname": "position", "data": "true_data.pkl",
              "datavar": "var.pkl", "nosave": True, "logit": False}
    cfg_en = {"ne": NE, "state": "x", "importstate": "prior.npz", "prior_x": {"var": 0.16}}

    sim = LinearAdjointSim(H)
    ensemble = AssimilationEnsemble(cfg_da, cfg_en, sim)
    sim.adjoint_weights = DataMisfitWeights(ensemble.data_layout, ensemble.obs_vector, ensemble.obs_variance,
                                            position=ensemble._record_positions())
    ensemble.forecast(ensemble.enX)

    expected = H.T @ ((obs[:, None] - H @ ensemble.enX) / SIGMA ** 2)
    assert ensemble.adjoints.shape == (2, NE)
    np.testing.assert_allclose(ensemble.adjoints, expected, rtol=1e-12)
