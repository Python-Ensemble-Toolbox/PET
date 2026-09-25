"""Regression tests for verified bugs in the small infrastructure modules."""

import os

import numpy as np
import pytest

from misc.system_tools.environ_var import OpenBlasSingleThread
from pipt.localization.factory import build_localization_instance
from simulator.simple_models import lin_1d, nonlin_onedimmodel


def test_single_thread_context_exits_cleanly_when_the_variable_was_unset(monkeypatch):
    """`__exit__` used to call the nonexistent os.environ.unsetenv."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with OpenBlasSingleThread():
        assert os.environ["OMP_NUM_THREADS"] == "1"
    assert "OMP_NUM_THREADS" not in os.environ


def test_single_thread_context_restores_a_previous_value(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    with OpenBlasSingleThread():
        assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "7"


@pytest.mark.parametrize("model", [lin_1d, nonlin_onedimmodel])
def test_simple_models_return_a_fresh_output_per_member(model):
    """They returned the shared attribute, so every member aliased the last one."""
    sim = model({"reporttype": "steps", "reportpoint": [0, 1], "datatype": ["x"]})
    sim.setup_fwd_run()
    first = sim.run_fwd_sim({"p": np.array([1.0, 2.0])}, 0)
    second = sim.run_fwd_sim({"p": np.array([10.0, 20.0])}, 1)
    assert first is not second
    assert not np.array_equal(first[0]["x"], second[0]["x"])


def test_unknown_localization_name_raises_instead_of_returning_none():
    with pytest.raises(ValueError, match="Unknown localization type 'banana'"):
        build_localization_instance({"name": "banana"}, None, None, None, 10)


def test_a_block_naming_no_mode_is_inferred_then_refused_by_that_mode():
    """A nameless block used to be rejected for having no 'name' -- a key its author had
    never written. The mode is now inferred the way it always was selected; a block that
    names nothing meant the parallel update, so that is what it is refused as."""
    with pytest.raises(ValueError, match="parallel update is not supported"):
        build_localization_instance({}, None, None, None, 10)


def test_an_explicitly_empty_localization_name_still_raises():
    with pytest.raises(ValueError, match="no 'name'"):
        build_localization_instance({"name": None}, None, None, None, 10)

