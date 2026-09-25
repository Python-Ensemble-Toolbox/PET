"""Every bundled simulator satisfies the contract the base ensemble drives it through."""

import pytest

from ensemble import ForwardSimulator
from simulator.simple_models import lin_1d, noSimulation, nonlin_onedimmodel
from simulator.vanderpol import VanDerPolOscillator

SIM_CONFIG = {"reporttype": "steps", "reportpoint": [1, 2, 3], "datatype": ["x"]}


@pytest.mark.parametrize(
    "make",
    [
        lambda: lin_1d(SIM_CONFIG),
        lambda: nonlin_onedimmodel(SIM_CONFIG),
        lambda: noSimulation(SIM_CONFIG),
        lambda: VanDerPolOscillator({}),
    ],
    ids=["lin_1d", "nonlin_onedimmodel", "noSimulation", "VanDerPolOscillator"],
)
def test_bundled_simulators_satisfy_the_protocol(make):
    assert isinstance(make(), ForwardSimulator)


def test_an_object_without_run_fwd_sim_does_not():
    class Half:
        input_dict = {}

    assert not isinstance(Half(), ForwardSimulator)
