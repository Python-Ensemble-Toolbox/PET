"""Tests for the explicit scheme registry and init_da dispatch."""

import pytest

from pipt import pipt_init
from pipt.update_schemes import registry


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

def test_algorithms_cover_the_public_classes_and_the_historical_names():
    from pipt.update_schemes.enkf import EnKF
    from pipt.update_schemes.enrml import GNEnRML, LMEnRML, co_lm_enrml, gn_enrml
    from pipt.update_schemes.es import ES
    from pipt.update_schemes.esmda import ESMDA

    assert set(registry.ALGORITHMS.values()) == {
        EnKF, ES, ESMDA, LMEnRML, GNEnRML, co_lm_enrml, gn_enrml,
    }


def test_hybrid_is_a_special_scheme_not_a_registered_flavour():
    """``hybrid`` is not a globally registered analysis flavour.

    ``esmda_hybrid`` is a real, distinct implementation (multilevel ES-MDA)
    that happens to share the ``esmda`` name, not an alias -- so it resolves
    only through ``SPECIAL_SCHEMES``, never through ``ALGORITHMS`` + a bound
    strategy.
    """
    from pipt.update_schemes.analysis.registry import available_analyses

    assert "hybrid" not in available_analyses()
    assert ("esmda", "hybrid") in registry.SPECIAL_SCHEMES


def test_margis_is_a_gnenrml_specific_flavour_not_a_special_scheme():
    """``margis`` binds normally on ``GNEnRML``, unlike ``hybrid``.

    It is not a *globally* registered flavour (only ``GNEnRML`` offers it,
    not every algorithm), but it is an ordinary ``COMPATIBLE_ANALYSES`` entry
    on that one class -- resolved through ``ALGORITHMS`` + a bound analysis,
    not through ``SPECIAL_SCHEMES`` the way ``hybrid`` is.
    """
    from pipt.update_schemes.analysis.registry import available_analyses
    from pipt.update_schemes.analysis.margis import margIS_update
    from pipt.update_schemes.enrml import GNEnRML

    assert "margis" not in available_analyses()
    assert ("gnenrml", "margis") not in registry.SPECIAL_SCHEMES
    assert GNEnRML.COMPATIBLE_ANALYSES["margis"] is margIS_update
    ctor = registry.get_scheme("gnenrml", "margis")
    assert ctor.func is GNEnRML
    assert ctor.keywords == {"analysis": "margis"}


@pytest.mark.parametrize(
    "name, parent_name, flavour, other",
    [("co_lm_enrml", "lmenrml", "approx", "full"),
     ("gn_enrml", "gnenrml", "subspace", "approx")],
)
def test_historical_names_pin_one_flavour_of_a_live_algorithm(name, parent_name, flavour, other):
    """``co_lm_enrml`` and ``gn_enrml`` resolve like any scheme, to a subclass
    of the algorithm they always were, and offer exactly the flavour the
    name meant -- so a config asking for another flavour gets the usual
    "no such flavour" error rather than silently running something else."""
    cls = registry.ALGORITHMS[name]
    assert issubclass(cls, registry.ALGORITHMS[parent_name])
    assert cls.COMPATIBLE_ANALYSES == {flavour: registry.ALGORITHMS[parent_name].COMPATIBLE_ANALYSES[flavour]}
    assert (name, flavour) in registry.available_schemes()
    assert registry.get_scheme(name, flavour).func is cls
    with pytest.raises(KeyError, match=f"no '{other}' analysis flavour"):
        registry.get_scheme(name, other)


def test_get_scheme_binds_the_algorithm_and_flavour():
    from pipt.update_schemes.esmda import ESMDA

    ctor = registry.get_scheme("esmda", "approx")
    assert ctor.func is ESMDA
    assert ctor.keywords == {"analysis": "approx"}


def test_get_scheme_is_case_insensitive():
    from pipt.update_schemes.esmda import ESMDA

    assert registry.get_scheme("ESMDA", "Approx").func is ESMDA


def test_get_scheme_resolves_special_schemes_directly():
    from pipt.update_schemes.multilevel import esmda_hybrid

    assert registry.get_scheme("esmda", "hybrid") is esmda_hybrid


def test_available_schemes_is_sorted_and_covers_specials():
    combos = registry.available_schemes()
    assert combos == sorted(combos)
    assert ("esmda", "hybrid") in combos
    assert ("gnenrml", "margis") in combos
    assert ("esmda", "geo") not in combos, "esmda_geo was dead code and has been removed"


#: Flavour/algorithm pairs deliberately absent, and why. `subspace2` solves for an
#: ne x ne transform, which the sequential schemes cannot apply one datum at a time;
#: `subspace` is registered on them and already fails on the characterisation case
#: (see the note on CASES in test_numerical_characterisation), so advertising a second
#: weight-space flavour there would only widen a known fault.
DELIBERATELY_UNREGISTERED = {("enkf", "subspace2"), ("es", "subspace2")}


def test_every_algorithm_gets_every_registered_flavour():
    from pipt.update_schemes.analysis.registry import available_analyses

    combos = set(registry.available_schemes())
    for algo in registry.ALGORITHMS:
        if algo in ("co_lm_enrml", "gn_enrml"):
            continue  # historical names pin one flavour by design
        for flavour in available_analyses():
            if (algo, flavour) in DELIBERATELY_UNREGISTERED:
                assert (algo, flavour) not in combos, "remove it from the exception set"
                continue
            assert (algo, flavour) in combos


def test_unknown_scheme_error_lists_alternatives():
    with pytest.raises(KeyError, match="Unknown assimilation scheme") as err:
        registry.get_scheme("esmdaa", "approx")
    assert "esmda" in str(err.value)


def test_unknown_flavour_error_is_distinct_and_lists_flavours():
    with pytest.raises(KeyError, match="no 'banana' analysis flavour") as err:
        registry.get_scheme("esmda", "banana")
    message = str(err.value)
    assert "hybrid" in message and "approx" in message


def test_register_scheme_roundtrip():
    class Dummy:
        pass

    registry.register_scheme("dummy", "approx", Dummy)
    try:
        assert registry.get_scheme("dummy", "approx") is Dummy
        with pytest.raises(ValueError, match="already registered"):
            registry.register_scheme("dummy", "approx", Dummy)
        registry.register_scheme("dummy", "approx", Dummy, overwrite=True)
    finally:
        registry.SPECIAL_SCHEMES.pop(("dummy", "approx"), None)


def test_register_scheme_rejects_clashing_with_a_generic_combo():
    """A generic algorithm+flavour combo counts as "already registered" too."""
    class Dummy:
        pass

    with pytest.raises(ValueError, match="already registered"):
        registry.register_scheme("esmda", "approx", Dummy)


# ----------------------------------------------------------------------
# init_da validation
# ----------------------------------------------------------------------

def test_init_da_missing_scheme():
    with pytest.raises(ValueError, match="SCHEME is missing"):
        pipt_init.init_da({}, {}, None)


def test_init_da_legacy_daalg_points_at_migrate():
    """Clean break: the old key is rejected, but with a pointer to the tool."""
    with pytest.raises(ValueError, match="pet migrate"):
        pipt_init.init_da({"daalg": ["esmda", "esmda"], "analysis": "approx"}, {}, None)


def test_init_da_missing_analysis():
    with pytest.raises(ValueError, match="ANALYSIS is missing"):
        pipt_init.init_da({"scheme": "esmda"}, {}, None)


def test_init_da_unknown_scheme_reports_clearly():
    """The old importlib path raised a bare ModuleNotFoundError here."""
    with pytest.raises(KeyError, match="Unknown assimilation scheme"):
        pipt_init.init_da({"scheme": "nope", "analysis": "approx"}, {}, None)
