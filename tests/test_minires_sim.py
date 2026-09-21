"""The MiniRes wrapper: the protocol, the state/report mappings, and the adjoint."""

import pickle

import numpy as np
import pytest

from ensemble import ForwardSimulator

pytest.importorskip("minires", reason="pip install PET[minires]")

from simulator.minires import MiniRes  # noqa: E402

NX = NY = 8


def config(**kwargs):
    cfg = {
        "dt": 0.05,
        "reportpoint": [2, 4, 6],
        "reporttype": "steps",
        "datatype": ["WWCT:PRD1", "WOPR:PRD1", "FWIR"],
        "model": {
            "Nx": NX, "Ny": NY, "por": 0.2,
            "wells": [
                {"name": "INJ1", "xy": [0.05, 0.05], "rate": 1.0},
                {"name": "PRD1", "xy": [0.95, 0.95], "rate": -1.0},
            ],
        },
    }
    cfg.update(kwargs)
    return cfg


def a_field(seed=1, scale=0.5):
    return scale * np.random.default_rng(seed).standard_normal(NX * NY)


def test_it_satisfies_the_forward_simulator_protocol():
    assert isinstance(MiniRes(config()), ForwardSimulator)


def test_it_pickles_as_the_parallel_forecast_requires():
    assert isinstance(pickle.loads(pickle.dumps(MiniRes(config()))), MiniRes)


def test_the_records_are_one_dict_per_report_point_keyed_by_data_type():
    sim = MiniRes(config())
    records = sim.run_fwd_sim({"permx": a_field()}, 0)
    assert len(records) == len(sim.report)
    for row in records:
        assert set(row) == set(sim.datatype)
        assert all(np.shape(v) == (1,) for v in row.values())


def test_the_same_state_gives_the_same_records():
    sim = MiniRes(config())
    state = {"permx": a_field()}
    first, second = sim.run_fwd_sim(state, 0), sim.run_fwd_sim(state, 1)
    for a, b in zip(first, second):
        for key in a:
            assert a[key] == b[key]


def test_a_member_the_simulator_cannot_run_comes_back_as_False():
    """`False`, not an exception: the ensemble replaces the member and carries on."""
    sim = MiniRes(config())
    assert sim.run_fwd_sim({"permx": np.zeros(NX * NY + 1)}, 0) is False
    assert sim.run_fwd_sim({"not_the_state": np.zeros(NX * NY)}, 0) is False


def test_the_state_is_the_log_permeability_in_the_declared_grid_ordering():
    field = a_field()
    for order in ("C", "F"):
        sim = MiniRes(config(field_order=order))
        model = sim.model
        sim.set_permeability(model, {"permx": field})
        np.testing.assert_allclose(model.K[0], np.exp(field.reshape(model.shape, order=order)))
        np.testing.assert_allclose(model.K[1], model.K[0])  # isotropic

    sim = MiniRes(config(log_perm=False))
    sim.set_permeability(sim.model, {"permx": np.abs(field)})
    np.testing.assert_allclose(sim.model.K[0], np.abs(field).reshape(sim.model.shape))


def test_the_water_cut_is_the_produced_water_over_the_produced_liquid():
    sim = MiniRes(config())
    records = sim.run_fwd_sim({"permx": a_field()}, 0)
    for row in records:
        assert 0 <= row["WWCT:PRD1"][0] <= 1
        # The producer's liquid rate is its rate spec, so the cut and the oil rate agree
        assert row["WOPR:PRD1"][0] == pytest.approx(1.0 - row["WWCT:PRD1"][0])
    assert records[0]["FWIR"][0] == pytest.approx(1.0)  # the injector, unsplit


def test_an_unknown_data_type_is_refused_when_the_simulator_is_built():
    with pytest.raises(ValueError, match="Unknown well quantity"):
        MiniRes(config(datatype=["WXYZ:PRD1"]))
    with pytest.raises(ValueError, match="names no well"):
        MiniRes(config(datatype=["WWCT:NOSUCH"]))


def test_a_fidelity_level_reconfigures_the_model():
    sim = MiniRes(config(levels=[{"dt": 0.05}, {"dt": 0.01}]))
    sim.setup_fwd_run(level=1)
    assert sim.dt == 0.01
    sim.setup_fwd_run(level=0)
    assert sim.dt == 0.05


class TestAdjoints:
    """`compute_adjoints` gives each datum's sensitivity to the state, checked against a difference."""

    def test_the_frame_is_indexed_by_report_point_and_holds_one_row_per_state(self):
        sim = MiniRes(config(compute_adjoints=True, datatype=["WWCT:PRD1"]))
        _records, adjoint = sim.run_fwd_sim({"permx": a_field()}, 0)
        assert adjoint.index.name == sim.report_type
        assert list(adjoint.index) == sim.report
        assert all(np.shape(cell) == (NX * NY,) for cell in adjoint["WWCT:PRD1"])

    @pytest.mark.parametrize("datatype", ["WWCT:PRD1", "WWPR:PRD1", "WOPR:PRD1"])
    def test_it_agrees_with_a_central_difference(self, datatype):
        cfg = config(compute_adjoints=True, datatype=[datatype], reportpoint=[4, 6])
        cfg["model"]["cached_precond"] = False  # spare the difference the iteration's noise
        sim = MiniRes(cfg)
        rng = np.random.default_rng(0)
        x, v, eps = a_field(), rng.standard_normal(NX * NY), 1e-6

        _records, adjoint = sim.run_fwd_sim({"permx": x}, 0)
        plus = sim.run_fwd_sim({"permx": x + eps * v}, 0)[0]
        minus = sim.run_fwd_sim({"permx": x - eps * v}, 0)[0]

        for i, step in enumerate(sim.report):
            difference = (plus[i][datatype][0] - minus[i][datatype][0]) / (2 * eps)
            assert adjoint.loc[step, datatype] @ v == pytest.approx(difference, rel=1e-4)

    def test_a_quantity_it_cannot_differentiate_is_refused_rather_than_dropped(self):
        with pytest.raises(NotImplementedError, match="not among the differentiated"):
            MiniRes(config(compute_adjoints=True, datatype=["FWIR"]))
