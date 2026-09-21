"""Tests for the friendly scheme constructors.

The flavour is a parameter of the algorithm, not a different algorithm, so
``ESMDA(..., analysis="full")`` and ``registry.get_scheme("esmda", "full")``
must resolve to the same behaviour: the ``ESMDA`` class with ``analysis``
pre-bound.
"""

import pytest

import pipt
from pipt.update_schemes import registry


ALGORITHMS = {
    "EnKF": ("enkf", ["approx", "full", "subspace"]),
    "ES": ("es", ["approx", "full", "subspace"]),
    "ESMDA": ("esmda", ["approx", "full", "subspace", "subspace2", "hybrid"]),
    "LMEnRML": ("lmenrml", ["approx", "full", "subspace", "subspace2"]),
    "GNEnRML": ("gnenrml", ["approx", "full", "subspace", "subspace2", "margis"]),
}


def test_top_level_exports():
    for name in ALGORITHMS:
        assert hasattr(pipt, name), f"pipt.{name} should be importable"
    assert hasattr(pipt, "build_scheme")


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_constructor_is_named_readably(name):
    assert getattr(pipt, name).__name__ == name


@pytest.mark.parametrize(
    "name,scheme,flavour",
    [(n, s, f) for n, (s, fs) in ALGORITHMS.items() for f in fs],
)
def test_every_flavour_documented_is_registered(name, scheme, flavour):
    """Every advertised (scheme, flavour) pair must still resolve."""
    assert registry.get_scheme(scheme, flavour) is not None


def test_five_algorithms_cover_every_registered_combination():
    """The five algorithm classes between them reach every registered combo."""
    for name, (scheme, _) in ALGORITHMS.items():
        flavours = [f for s, f in registry.available_schemes() if s == scheme]
        assert flavours, f"{scheme} has no registered flavours"


def test_registry_size_matches_the_algorithms_specials_and_historical_names():
    """Down from eighteen hand-written classes: 5 algorithms x 3 flavours, ``subspace2``
    on the three schemes that can apply an ensemble transform, the two combinations
    backed by a distinct implementation, and the two historical names (co_lm_enrml,
    gn_enrml) that each pin a single flavour."""
    assert len(registry.available_schemes()) == 5 * 3 + 3 + 2 + 2
    assert len(ALGORITHMS) == 5               # the public constructors above
    assert len(registry.ALGORITHMS) == 5 + 2  # plus the two historical names


def test_build_scheme_still_dispatches_through_the_registry(monkeypatch):
    """`build_scheme` resolves by name; the classes no longer need to."""
    captured = {}

    class Spy:
        def __init__(self, da, en, sim):
            captured["args"] = (da, en, sim)

    monkeypatch.setitem(registry.SPECIAL_SCHEMES, ("esmda", "approx"), Spy)

    b = pipt.build_scheme("esmda", {"d": 1}, {"e": 2}, "sim", analysis="approx")
    assert isinstance(b, Spy)
    assert captured["args"] == ({"d": 1}, {"e": 2}, "sim")


def test_full_coincides_with_approx_for_single_step_schemes():
    """EnKF/ES never revisit a data group, so `full` resolves to `approx`.

    This used to be encoded as `enkf_full`/`es_full` pinning `FLAVOUR =
    "approx"`. It now lives directly in `EnKF.COMPATIBLE_ANALYSES`, inherited
    unchanged by `ES`: `"full"` and `"approx"` point at the same class, so
    binding either builds the identical strategy regardless of entry point.
    """
    from pipt.update_schemes.analysis.approx import approx_update
    from pipt.update_schemes.analysis.subspace import subspace_update
    from pipt.update_schemes.enkf import EnKF
    from pipt.update_schemes.es import ES

    assert EnKF.COMPATIBLE_ANALYSES["full"] is EnKF.COMPATIBLE_ANALYSES["approx"] is approx_update
    # Other flavours are unaffected.
    assert EnKF.COMPATIBLE_ANALYSES["subspace"] is subspace_update

    assert ES.COMPATIBLE_ANALYSES is EnKF.COMPATIBLE_ANALYSES


def test_esmda_and_enrml_do_not_fold_full_into_approx():
    """The fold is specific to EnKF/ES; the iterative schemes keep `full` as is."""
    from pipt.update_schemes.analysis.approx import approx_update
    from pipt.update_schemes.analysis.full import full_update
    from pipt.update_schemes.enrml import GNEnRML, LMEnRML
    from pipt.update_schemes.esmda import ESMDA

    for algorithm in (ESMDA, LMEnRML, GNEnRML):
        assert algorithm.COMPATIBLE_ANALYSES["full"] is full_update
        assert algorithm.COMPATIBLE_ANALYSES["full"] is not approx_update


def test_geo_is_gone_and_hybrid_stays_a_separate_class():
    """`geo` was dead code (a broken, untested `__init__`) and has been removed.

    `hybrid` is a distinct algorithm sharing the ESMDA name, not an analysis,
    so it remains its own class reachable through the registry rather than
    through `ESMDA(analysis=...)`.
    """
    from pipt.update_schemes.analysis.registry import available_analyses

    assert "geo" not in available_analyses()
    assert "hybrid" not in available_analyses()
    with pytest.raises(KeyError):
        registry.get_scheme("esmda", "geo")
    assert registry.get_scheme("esmda", "hybrid") is not None


def test_default_analysis_is_approx(monkeypatch):
    class Spy:
        def __init__(self, da, en, sim):
            pass

    monkeypatch.setitem(registry.SPECIAL_SCHEMES, ("esmda", "approx"), Spy)
    assert isinstance(pipt.build_scheme("esmda", {}, {}, None), Spy)


def test_bad_flavour_reports_valid_ones():
    with pytest.raises(KeyError, match="no 'nope' analysis flavour"):
        pipt.build_scheme("esmda", {}, {}, None, analysis="nope")


def test_per_flavour_class_names_no_longer_exist():
    """The eighteen deprecated names (`esmda_approx`, `lmenrml_full`, ...) were
    a documented backward-compatibility promise; it has been deliberately
    retracted in favour of `ESMDA(..., analysis=...)` and friends."""
    import pipt.update_schemes as us

    for name in (
        "esmda_approx", "esmda_full", "esmda_subspace", "esmda_geo",
        "es_approx", "es_full", "es_subspace",
        "enkf_approx", "enkf_full", "enkf_subspace",
        "lmenrml_approx", "lmenrml_full", "lmenrml_subspace",
        "gnenrml_approx", "gnenrml_full", "gnenrml_subspace",
    ):
        assert not hasattr(us, name), f"{name} should have been removed"


def test_factory_honours_config_analysis():
    """The factory must not silently disagree with init_da.

    `analysis` used to default to "approx" in the factory while init_da read it
    from the config, so a config asking for "subspace" built esmda_approx
    through one entry point and esmda_subspace through the other.
    """
    import inspect

    from pipt import ESMDA, build_scheme

    # The defaults are what caused the disagreement: "approx" here vs the
    # config's value in init_da.
    assert inspect.signature(ESMDA).parameters["analysis"].default is None
    assert inspect.signature(build_scheme).parameters["analysis"].default is None


def test_factory_resolves_each_flavour_from_config():
    from pipt.update_schemes.registry import get_scheme

    for flavour in ("approx", "full", "subspace"):
        cfg_da = {"scheme": "esmda", "analysis": flavour}
        assert get_scheme(cfg_da["scheme"], cfg_da["analysis"]) is get_scheme(
            "esmda", flavour
        )


def test_config_analysis_beats_the_fallback(monkeypatch):
    """A config asking for a flavour must not be overridden by the default."""
    class Spy:
        def __init__(self, da, en, sim):
            pass

    monkeypatch.setitem(registry.SPECIAL_SCHEMES, ("esmda", "subspace"), Spy)
    cfg = {"scheme": "esmda", "analysis": "subspace"}
    assert isinstance(pipt.build_scheme("esmda", cfg, {}, None), Spy)


def test_explicit_analysis_beats_the_config(monkeypatch):
    class Spy:
        def __init__(self, da, en, sim):
            pass

    monkeypatch.setitem(registry.SPECIAL_SCHEMES, ("esmda", "full"), Spy)
    cfg = {"scheme": "esmda", "analysis": "subspace"}
    assert isinstance(pipt.build_scheme("esmda", cfg, {}, None, analysis="full"), Spy)


def test_class_resolves_flavour_by_the_same_precedence():
    """The classes apply explicit -> config -> approx, as build_scheme does."""
    from pipt.update_schemes.esmda import ESMDA

    resolve = ESMDA.resolve_analysis
    assert resolve(ESMDA, "full", {"analysis": "subspace"}) == "full"
    assert resolve(ESMDA, None, {"analysis": "subspace"}) == "subspace"
    assert resolve(ESMDA, None, {}) == "approx"
