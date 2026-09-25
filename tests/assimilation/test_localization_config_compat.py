"""A LOCALIZATION block written before the strategies were named still runs.

Localization used to be selected by which keyword appeared in the block rather than by
a ``name``, and the auto-adaptive cutoff was carried as the value of the ``autoadaloc``
keyword itself. Reading neither meant an existing config either died at startup naming
three strings its author had never seen, or -- worse -- ran with a different taper and
said nothing.
"""

import numpy as np
import pytest

from input_output.config import ConfigError
from pipt.localization import build_localization_instance
from pipt.localization.auto_ada_loc import AutoAdaptiveLocalization
from pipt.localization.common import infer_name
from pipt.localization.distance_localization import DistanceLocalization

FIELD = [1, 10, 10]


def _build(config):
    return build_localization_instance(
        config, data_indices=[0], data_types=["d"], parameters=["p"], ensemble_size=10
    )


# --------------------------------------------------------------------------
# The mode is inferred from the keyword that used to select it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config, expected", [
    ({"autoadaloc": 2},                    "autoadaloc"),
    ({"localanalysis": True},              "localanalysis"),
    ({"dist_loc": True},                   "distance_loc"),   # as a key
    ({"mode": "dist_loc"},                 "distance_loc"),   # ... or as a bare value
    ({"anything": "masks.p"},              "distance_loc"),   # a pickled mask file
    ({"anything": "masks.pkl"},            "distance_loc"),
    ({},                                   "parallel_update"),  # the old fallback
])
def test_the_mode_is_inferred_from_the_keyword_that_selected_it(config, expected):
    assert infer_name(config) == expected


def test_an_explicit_name_wins_over_inference():
    assert _build({"name": "autoadaloc", "field": FIELD, "dist_loc": True}).name == "autoadaloc"


@pytest.mark.parametrize("config", [
    {"field": FIELD, "autoadaloc": 2},
    {"field": FIELD, "dist_loc": True},
])
def test_a_config_without_a_name_still_builds(config):
    assert _build(config) is not None


# --------------------------------------------------------------------------
# autoadaloc's value is the cutoff, and it is no longer discarded
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config, expected", [
    ({"autoadaloc": 2},                          2.0),    # the old spelling
    ({"autoadaloc": 2, "cutoff": 0.5},           0.5),    # cutoff wins if both are given
    ({"name": "autoadaloc", "nstd": 1.5},        1.5),    # the name it had inside the code
    ({"name": "autoadaloc", "cutoff": 0.5},      0.5),
    ({"autoadaloc": True},                       0.3),    # a bare flag is not a value
    ({"name": "autoadaloc"},                     0.3),    # nothing given at all
])
def test_the_cutoff_comes_from_whichever_spelling_the_config_used(config, expected):
    loc = _build({"field": FIELD, **config})

    assert isinstance(loc, AutoAdaptiveLocalization)
    assert loc.cutoff == expected


def test_the_cutoff_actually_reaches_the_taper():
    """The value has to change the threshold, not just land on the instance."""
    rng = np.random.default_rng(0)
    corr = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    shuffled = rng.normal(scale=0.1, size=(50, 1))

    strict = _build({"field": FIELD, "autoadaloc": 3}).tapering_function(corr, shuffled)
    lenient = _build({"field": FIELD, "autoadaloc": 1}).tapering_function(corr, shuffled)

    assert strict.sum() < lenient.sum()          # a higher cutoff keeps fewer correlations


# --------------------------------------------------------------------------
# The two modes that cannot run say so while the config is being read
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config, expected", [
    ({"localanalysis": True, "type": "gc", "range": 5}, "Local analysis is not supported"),
    ({},                                                "The parallel update is not supported"),
])
def test_an_unsupported_mode_is_refused_at_config_time(config, expected):
    with pytest.raises(ConfigError, match=expected):
        _build({"field": FIELD, **config})


def test_the_refusal_names_something_the_user_can_do_instead():
    with pytest.raises(ConfigError, match="distance_loc"):
        _build({"field": FIELD, "localanalysis": True})


# --------------------------------------------------------------------------
# A list is still accepted where a dict is
# --------------------------------------------------------------------------

def test_a_list_shaped_block_is_normalized_and_named():
    loc = _build([["field", *FIELD], ["autoadaloc", 2]])

    assert isinstance(loc, (AutoAdaptiveLocalization, DistanceLocalization))
    assert loc.name == "autoadaloc"
