"""The MiniRes wrapper: the protocol, the state/report mappings, and the adjoint."""

import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
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



def bhp_config(**kwargs):
    """A producer on BHP control, so its rate follows from its well index, and a compressible model."""
    cfg = config(**{"datatype": ["WOPR:PRD1", "WWCT:PRD1", "WBHP:INJ1"], "p0": 2.0, **kwargs})
    cfg["model"]["ct"] = 0.1
    cfg["model"]["wells"] = [
        {"name": "INJ1", "xy": [0.05, 0.05], "rate": 1.0, "rw": 0.01},
        {"name": "PRD1", "xy": [0.95, 0.95], "bhp": 0.0, "rw": 0.01},
    ]
    return cfg


def test_a_member_runs_as_the_model_built_with_its_permeability_would():
    """The well indices are derived from K, so they must be the member's, not the config's stand-in's."""
    from minires import ResSim

    cfg = bhp_config()
    sim = MiniRes(cfg)
    for field in (a_field(seed=1), a_field(seed=2) + 2):
        records = sim.run_fwd_sim({"permx": field}, 0)

        spec = {k: v for k, v in cfg["model"].items() if k != "por"}
        model = ResSim(**spec, K=np.exp(field.reshape(NX, NY)), por=np.full((NX, NY), 0.2))
        SS, _PP = model.sim(sim.dt, sim.nSteps, sim.S0, np.full(NX * NY, cfg["p0"]), pbar=False)
        expected = sim.records(model, SS)

        for row, want in zip(records, expected):
            for key in row:
                assert row[key][0] == pytest.approx(want[key][0], rel=1e-10)


def test_the_initial_pressure_counts_when_the_model_is_compressible():
    low, high = (MiniRes(bhp_config(p0=p0)).run_fwd_sim({"permx": a_field()}, 0) for p0 in (1.0, 3.0))
    assert high[0]["WOPR:PRD1"][0] > low[0]["WOPR:PRD1"][0]  # more pressure to deplete


def test_a_fluid_parameter_in_the_state_overrides_the_configs():
    in_config = config()
    in_config["model"]["fluid"] = {"sor": 0.2, "vo": 3.0}
    expected = MiniRes(in_config).run_fwd_sim({"permx": a_field()}, 0)

    sim = MiniRes(config())
    records = sim.run_fwd_sim({"permx": a_field(), "sor": np.array([0.2]), "vo": np.array([3.0])}, 0)
    assert records == expected
    assert sim.model.fluid.sor == 0  # the member's copy took it, not the template
    assert records != sim.run_fwd_sim({"permx": a_field()}, 0)


def test_a_porosity_in_the_state_is_a_field_in_the_declared_grid_ordering():
    poro = 0.1 + 0.2 * np.random.default_rng(3).random(NX * NY)
    for order in ("C", "F"):
        sim = MiniRes(config(field_order=order))
        model = sim.model
        sim.set_state(model, {"permx": a_field(), "por": poro})
        np.testing.assert_array_equal(model.por, poro.reshape(model.shape, order=order))


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"banana": np.array([1.0])}, "no use for"),
        ({"sor": np.array([0.1, 0.2])}, "a scalar"),
    ],
)
def test_a_state_it_cannot_place_fails_the_member_and_says_why(extra, message, caplog):
    sim = MiniRes(config())
    assert sim.run_fwd_sim({"permx": a_field(), **extra}, 0) is False
    assert message in caplog.text

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

    @pytest.mark.parametrize("datatype", ["WOPR:PRD1", "WWPR:PRD1", "WWCT:PRD1", "WBHP:INJ1"])
    def test_it_covers_the_well_model_and_the_well_indices_that_follow_the_permeability(self, datatype):
        # A producer on BHP control, and the pressure of an injector on rate: both go through
        # well indices that the wrapper derives from the member's permeability.
        cfg = bhp_config(compute_adjoints=True, datatype=[datatype], reportpoint=[4, 6])
        cfg["model"]["cached_precond"] = False
        sim = MiniRes(cfg)
        rng = np.random.default_rng(0)
        x, v, eps = a_field(), rng.standard_normal(NX * NY), 1e-6

        _records, adjoint = sim.run_fwd_sim({"permx": x}, 0)
        plus = sim.run_fwd_sim({"permx": x + eps * v}, 0)[0]
        minus = sim.run_fwd_sim({"permx": x - eps * v}, 0)[0]

        for i, step in enumerate(sim.report):
            difference = (plus[i][datatype][0] - minus[i][datatype][0]) / (2 * eps)
            assert adjoint.loc[step, datatype] @ v == pytest.approx(difference, rel=1e-4)

    def test_a_state_beyond_the_permeability_is_refused_rather_than_left_out(self, caplog):
        sim = MiniRes(config(compute_adjoints=True, datatype=["WWCT:PRD1"]))
        assert sim.run_fwd_sim({"permx": a_field(), "sor": np.array([0.1])}, 0) is False
        assert "only with respect to the permeability" in caplog.text

    def test_rate_controlled_path_is_refused_by_both_adjoint_entry_points(self):
        cfg = bhp_config(datatype=["WBHP:INJ1"], reportpoint=[4])
        injector = cfg["model"]["wells"][0]
        injector.pop("xy")
        injector["path"] = [[.05, .05], [.45, .05]]
        state = {"permx": a_field()}
        sim = MiniRes(cfg)
        assert np.isfinite(sim.run_fwd_sim(state, 0)[0]["WBHP:INJ1"]).all()
        with pytest.raises(NotImplementedError, match="rate allocation.*INJ1"):
            sim.run_vjp(state, pd.DataFrame({"WBHP:INJ1": [np.ones(1)]}, index=[4]))
        with pytest.raises(NotImplementedError, match="rate allocation.*INJ1"):
            MiniRes({**cfg, "compute_adjoints": True}).run_fwd_sim(state, 0)

    def test_a_quantity_it_cannot_differentiate_is_refused_rather_than_dropped(self):
        with pytest.raises(NotImplementedError, match="not among the differentiated"):
            MiniRes(config(compute_adjoints=True, datatype=["FWIR"]))

    def test_a_weighted_sum_of_the_data_costs_one_sweep_and_agrees_with_the_frame(self):
        datatypes = ["WWCT:PRD1", "WOPR:PRD1"]
        sim = MiniRes(config(compute_adjoints=True, datatype=datatypes))
        state = {"permx": a_field()}
        _records, adjoint = sim.run_fwd_sim(state, 0)

        # Two sums; the last report point of the second data type is in neither
        weights = pd.DataFrame(index=adjoint.index, columns=datatypes, dtype=object)
        rng = np.random.default_rng(0)
        expected = np.zeros((2, NX * NY))
        for step in sim.report:
            for datatype in datatypes:
                if (step, datatype) == (sim.report[-1], datatypes[1]):
                    weights.at[step, datatype] = None
                    continue
                weights.at[step, datatype] = rng.standard_normal(2)
                expected += np.outer(weights.at[step, datatype], adjoint.at[step, datatype])

        with patch.object(sim, "_simulate", wraps=sim._simulate) as forward:
            products = sim.run_vjp(state, weights)
        forward.assert_called_once_with(state)
        np.testing.assert_allclose(products["permx"], expected, rtol=1e-8, atol=1e-12)

    def test_a_weighted_sum_is_also_differentiated_with_respect_to_a_scalar_of_the_fluid(self):
        cfg = bhp_config(reportpoint=[4, 6])
        cfg["model"]["cached_precond"] = False
        sim = MiniRes(cfg)
        rng = np.random.default_rng(0)
        weights = pd.DataFrame(
            [[rng.standard_normal(1) for _ in sim.datatype] for _ in sim.report],
            index=sim.report, columns=sim.datatype, dtype=object,
        )

        def weighted_sum(state):
            records = sim.run_fwd_sim(state, 0)
            return sum(weights.at[step, d][0] * row[d][0] for step, row in zip(sim.report, records) for d in sim.datatype)

        state = {"permx": a_field(), "sor": np.array([0.2])}
        products = sim.run_vjp(state, weights)
        assert products["permx"].shape == (1, NX * NY) and products["sor"].shape == (1, 1)

        v, eps = rng.standard_normal(NX * NY), 1e-6
        along_field = (weighted_sum({**state, "permx": state["permx"] + eps * v})
                       - weighted_sum({**state, "permx": state["permx"] - eps * v})) / (2 * eps)
        assert products["permx"][0] @ v == pytest.approx(along_field, rel=1e-4)
        # Only roughly: the response to the fluid has small jumps, so its slope depends on the step
        along_sor = (weighted_sum({**state, "sor": np.array([0.22])}) - weighted_sum({**state, "sor": np.array([0.18])})) / 0.04
        assert products["sor"][0, 0] == pytest.approx(along_sor, rel=0.2)

    def test_a_field_beyond_the_permeability_is_refused_by_the_weighted_sum(self):
        sim = MiniRes(config(datatype=["WWCT:PRD1"]))
        weights = pd.DataFrame({"WWCT:PRD1": [np.ones(1)]}, index=[2], dtype=object)
        with pytest.raises(NotImplementedError, match="scalars of the fluid"):
            sim.run_vjp({"permx": a_field(), "por": np.full(NX * NY, 0.2)}, weights)

    def test_a_weighted_datum_it_cannot_differentiate_is_refused(self):
        sim = MiniRes(config(datatype=["FWIR"]))
        weights = pd.DataFrame({"FWIR": [np.ones(1)]}, index=[2], dtype=object)
        with pytest.raises(NotImplementedError, match="not among the differentiated"):
            sim.run_vjp({"permx": a_field()}, weights)
