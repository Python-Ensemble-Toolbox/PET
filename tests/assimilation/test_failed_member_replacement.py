"""A crashed realisation is replaced by a successful one, state and prediction together."""

from types import SimpleNamespace

import numpy as np

from ensemble.ensemble import BaseEnsemble

NE, NX = 6, 3


def _host():
    log = []
    return SimpleNamespace(logger=SimpleNamespace(info=log.append), save=lambda: None, rng=np.random), log


def _members():
    # Column j of the state holds j; member j's output holds j too, so the
    # member that replaced a crash can be read off both.
    enX = np.tile(np.arange(NE, dtype=float), (NX, 1))
    outputs = [[{"d": np.array([float(j)])}] for j in range(NE)]
    return enX, outputs


def test_crashed_member_takes_state_and_output_of_the_same_successful_member():
    host, log = _host()
    enX, outputs = _members()
    outputs[2] = False                                   # member 2 crashed

    np.random.seed(0)
    new_out, new_enX, success = BaseEnsemble._replace_failed_simulations(host, outputs, enX)

    assert success
    k = int(new_out[2][0]["d"][0])
    assert k != 2
    np.testing.assert_array_equal(new_enX[:, 2], np.full(NX, float(k)))
    for j in range(NE):
        if j != 2:
            np.testing.assert_array_equal(new_enX[:, j], np.full(NX, float(j)))
    assert new_enX is enX                                # replaced in place, so the caller's state sees it
    assert any("member 2 failed" in m for m in log)


def test_nothing_changes_when_nothing_crashed():
    host, _ = _host()
    enX, outputs = _members()
    before = np.array(enX)

    new_out, new_enX, success = BaseEnsemble._replace_failed_simulations(host, outputs, enX)

    assert success and new_out is outputs
    np.testing.assert_array_equal(new_enX, before)


def test_more_crashes_than_successes_draw_with_replacement():
    host, _ = _host()
    enX, outputs = _members()
    for j in (0, 1, 2, 3):
        outputs[j] = False

    np.random.seed(1)
    new_out, new_enX, success = BaseEnsemble._replace_failed_simulations(host, outputs, enX)

    assert success
    for j in (0, 1, 2, 3):
        k = int(new_out[j][0]["d"][0])
        assert k in (4, 5)
        np.testing.assert_array_equal(new_enX[:, j], np.full(NX, float(k)))


# ----------------------------------------------------------------------
# Through the forecast itself: this is where the state matrix used to be
# passed as the list of member inputs, so the first crash raised.
# ----------------------------------------------------------------------

class CrashingSimulator:
    """Member 2 fails; every other member reports its own state value."""

    input_dict = {"parallel": 1}
    redund_sim = None
    true_order = ["steps", [1, 2]]
    datatype = ["d"]

    def run_fwd_sim(self, state, member_index):
        if member_index == 2:
            return False
        value = float(state["x"][0])
        return [{"d": np.array([value])}, {"d": np.array([value + 100.0])}]


def _bare_ensemble():
    ens = object.__new__(BaseEnsemble)
    ens.sim = CrashingSimulator()
    ens.multilevel = None
    ens.ne = NE
    ens.idX = {"x": (0, NX)}
    ens.aux_input = None
    ens.keys_en = {}
    ens.logger, _ = _host()
    ens.logger = ens.logger.logger
    ens.rng = np.random
    return ens


def test_forecast_survives_a_crashed_member_and_keeps_state_and_prediction_matched():
    ens = _bare_ensemble()
    enX, _ = _members()

    np.random.seed(0)
    ens.calc_prediction(enX)

    predicted = ens.sim_data.loc[1, "d"]                 # one value per member
    k = int(predicted[2])                                # member 2 now carries member k's prediction ...
    assert k != 2
    np.testing.assert_array_equal(enX[:, 2], np.full(NX, float(k)))   # ... and member k's state
    for j in range(NE):
        if j != 2:
            assert predicted[j] == float(j)
            np.testing.assert_array_equal(enX[:, j], np.full(NX, float(j)))
