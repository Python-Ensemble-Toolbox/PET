"""Localization strategies are selected from a table, and the table is open."""

import pytest

from input_output.config import ConfigError
from pipt.localization import (
    LOCALIZATIONS,
    available_localizations,
    build_localization_instance,
    register_localization,
)


class Custom:
    name = "custom"

    def __init__(self, info, ensemble_size):
        self.info = info
        self.ensemble_size = ensemble_size


def _build_custom(*, info, ensemble_size, **_):
    return Custom(info, ensemble_size)


def test_the_shipped_strategies_are_registered():
    assert available_localizations() == ["autoadaloc", "distance_loc"]


@pytest.mark.parametrize("name, expected", [
    ("localanalysis", "Local analysis is not supported"),
    ("parallel_update", "The parallel update is not supported"),
])
def test_an_unsupported_mode_says_so_and_names_the_alternatives(name, expected):
    """Both ran before the schemes were restructured. Refusing while the config is read
    beats failing part way through the first update -- or, as local analysis used to,
    reporting a misfit for a posterior that is still the prior."""
    with pytest.raises(ConfigError, match=expected):
        build_localization_instance({"name": name, "field": [1, 10, 10]}, None, None, None, 10)

    assert name not in available_localizations()


def test_a_registered_strategy_is_built_from_its_name(monkeypatch):
    monkeypatch.setitem(LOCALIZATIONS, "custom", _build_custom)

    loc = build_localization_instance({"name": "custom", "radius": 3}, None, None, None, 17)

    assert isinstance(loc, Custom)
    assert loc.info == {"radius": 3} and loc.ensemble_size == 17
    assert "custom" in available_localizations()


def test_registering_an_existing_name_needs_overwrite(monkeypatch):
    monkeypatch.setitem(LOCALIZATIONS, "custom", _build_custom)
    with pytest.raises(ValueError, match="already registered"):
        register_localization("custom", _build_custom)
    register_localization("custom", _build_custom, overwrite=True)


def test_register_localization_adds_to_the_table(monkeypatch):
    monkeypatch.delitem(LOCALIZATIONS, "brand_new", raising=False)
    register_localization("brand_new", _build_custom)
    try:
        assert "brand_new" in LOCALIZATIONS
    finally:
        LOCALIZATIONS.pop("brand_new", None)


def test_unknown_name_lists_what_is_available():
    with pytest.raises(ValueError, match="autoadaloc"):
        build_localization_instance({"name": "nope"}, None, None, None, 1)
