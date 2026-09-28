"""Every bundled simulator satisfies the contract the base ensemble drives it through."""

import pytest

from ensemble import ForwardSimulator
from simulator.simple_models import lin_1d, noSimulation, nonlin_onedimmodel
from simulator.vanderpol import VanDerPolOscillator

SIM_CONFIG = {"reporttype": "steps", "reportpoint": [1, 2, 3], "datatype": ["x"]}

MINIRES_CONFIG = {
    "reporttype": "steps",
    "reportpoint": [1, 2, 3],
    "dt": 0.1,
    "datatype": ["FWIR"],
    "model": {"Nx": 4, "Ny": 4, "wells": [{"name": "W", "xy": [0.5, 0.5], "bhp": 1.0}]},
}


def make_minires():
    """MiniRes is an optional dependency (``pip install PET[minires]``)."""
    pytest.importorskip("minires")
    from simulator.minires import MiniRes

    return MiniRes(MINIRES_CONFIG)


@pytest.mark.parametrize(
    "make",
    [
        lambda: lin_1d(SIM_CONFIG),
        lambda: nonlin_onedimmodel(SIM_CONFIG),
        lambda: noSimulation(SIM_CONFIG),
        lambda: VanDerPolOscillator({}),
        make_minires,
    ],
    ids=["lin_1d", "nonlin_onedimmodel", "noSimulation", "VanDerPolOscillator", "MiniRes"],
)
def test_bundled_simulators_satisfy_the_protocol(make):
    assert isinstance(make(), ForwardSimulator)


def test_an_object_without_run_fwd_sim_does_not():
    class Half:
        input_dict = {}

    assert not isinstance(Half(), ForwardSimulator)
