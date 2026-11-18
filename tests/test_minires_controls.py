"""MiniRes with well controls: state variables that set the wells' rate schedules, for optimization."""

import logging

import numpy as np
import pytest

pytest.importorskip("minires", reason="pip install PET[minires]")

from simulator.minires import MiniRes  # noqa: E402

NX = NY = 8
DT = 0.05
NSTEPS = 6


def config(**kwargs):
    """Two injectors on rate and a producer on pressure; the rates of the injectors can be controls."""
    cfg = {
        "dt": DT,
        "reportpoint": [2, 4, 6],
        "reporttype": "steps",
        "datatype": ["FWIT", "FOPT", "WWIR:INJ1", "WWIR:INJ2"],
        "p0": 2.0,
        "model": {
            "Nx": NX, "Ny": NY, "por": 0.2, "ct": 0.1, "K": np.full((NX, NY), 2.0),
            "wells": [
                {"name": "INJ1", "xy": [0.05, 0.05], "rate": 1.0},
                {"name": "INJ2", "xy": [0.95, 0.05], "rate": 1.0},
                {"name": "PRD1", "xy": [0.5, 0.95], "bhp": 1.0, "rw": 0.01},
            ],
        },
    }
    cfg.update(kwargs)
    return cfg


CONTROLS = {"rate_inj1": {"well": "INJ1"}, "rate_inj2": {"well": "INJ2", "type": "rate"}}


def series(records, key):
    return np.array([row[key][0] for row in records])


def test_the_controls_are_the_wells_rates_held_constant_over_equal_intervals():
    sim = MiniRes(config(controls=CONTROLS))
    rates = {"rate_inj1": np.array([0.5, 1.0, 2.0]), "rate_inj2": np.array([1.5])}

    records = sim.run_fwd_sim(rates, 0)

    # Reported at the end of each of the three two-step intervals
    np.testing.assert_allclose(series(records, "WWIR:INJ1"), [0.5, 1.0, 2.0], rtol=1e-12)
    np.testing.assert_allclose(series(records, "WWIR:INJ2"), [1.5, 1.5, 1.5], rtol=1e-12)
    np.testing.assert_allclose(series(records, "FWIT"), np.cumsum([2.0, 2.5, 3.5]) * 2 * DT, rtol=1e-12)


def test_a_control_left_out_of_the_state_leaves_the_well_as_the_model_has_it():
    sim = MiniRes(config(controls=CONTROLS))
    records = sim.run_fwd_sim({"rate_inj1": np.full(3, 0.25)}, 0)
    np.testing.assert_allclose(series(records, "WWIR:INJ1"), 0.25, rtol=1e-12)
    np.testing.assert_allclose(series(records, "WWIR:INJ2"), 1.0, rtol=1e-12)


def test_without_a_permeability_in_the_state_the_models_is_used_and_with_one_the_states():
    sim = MiniRes(config(controls=CONTROLS))
    rates = {"rate_inj1": np.full(3, 1.0), "rate_inj2": np.full(3, 1.0)}

    on_model = sim.run_fwd_sim(rates, 0)
    same = sim.run_fwd_sim({**rates, "permx": np.full(NX * NY, np.log(2.0))}, 0)
    other = sim.run_fwd_sim({**rates, "permx": np.full(NX * NY, np.log(0.02))}, 0)

    np.testing.assert_allclose(series(on_model, "FOPT"), series(same, "FOPT"), rtol=1e-9)
    assert not np.allclose(series(on_model, "FOPT"), series(other, "FOPT"), rtol=1e-3)


def test_the_configuration_is_not_changed_by_a_run():
    """Each member gets its own copy of the well records; the next member starts from the config's."""
    cfg = config(controls=CONTROLS)
    sim = MiniRes(cfg)
    before = sim.run_fwd_sim({}, 0)
    sim.run_fwd_sim({"rate_inj1": np.full(3, 3.0)}, 0)
    after = sim.run_fwd_sim({}, 0)

    assert cfg["model"]["wells"][0]["rate"] == 1.0
    np.testing.assert_array_equal(series(before, "FWIT"), series(after, "FWIT"))


def test_a_sign_turns_a_magnitude_into_a_production_rate():
    cfg = config(controls={"prod": {"well": "PRD1", "type": "rate", "sign": -1}}, datatype=["WLPR:PRD1"])
    records = MiniRes(cfg).run_fwd_sim({"prod": np.array([0.5, 1.0])}, 0)
    np.testing.assert_allclose(series(records, "WLPR:PRD1"), [0.5, 1.0, 1.0], rtol=1e-12)


def test_aux_input_is_ignored_with_controls():
    sim = MiniRes(config(controls=CONTROLS))
    assert sim.run_fwd_sim({"rate_inj1": np.full(3, 1.0), "aux_input": 1}, 0) is not False


def test_a_schedule_that_does_not_divide_the_run_fails_the_member(caplog):
    sim = MiniRes(config(controls=CONTROLS))
    with caplog.at_level(logging.ERROR):
        assert sim.run_fwd_sim({"rate_inj1": np.ones(4)}, 0) is False
    assert "does not divide" in caplog.text


def test_a_control_must_name_a_well_and_a_known_type():
    with pytest.raises(ValueError, match="names the well 'INJ9'"):
        MiniRes(config(controls={"rate": {"well": "INJ9"}}))
    with pytest.raises(ValueError, match="must be 'rate' or 'bhp'"):
        MiniRes(config(controls={"rate": {"well": "INJ1", "type": "thp"}}))
    with pytest.raises(NotImplementedError, match="well controls"):
        MiniRes(config(controls=CONTROLS, compute_adjoints=True, datatype=["WOPR:PRD1"]))


def test_without_controls_nothing_is_different(caplog):
    """No permeability, a control's name or aux_input in the state: the member fails, as before."""
    sim = MiniRes(config())
    assert sim.controls == {}
    with caplog.at_level(logging.ERROR):
        assert sim.run_fwd_sim({}, 0) is False
        assert "has no 'permx'" in caplog.text
        caplog.clear()
        field = np.full(NX * NY, np.log(2.0))
        assert sim.run_fwd_sim({"permx": field, "rate_inj1": np.ones(3)}, 0) is False
        assert sim.run_fwd_sim({"permx": field, "aux_input": 0}, 0) is False
        assert caplog.text.count("no use for") == 2
    assert sim.run_fwd_sim({"permx": field}, 0) is not False
