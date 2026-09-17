"""Binding an analysis to a scheme instead of mixing it in.

Groundwork for making ``analysis`` a parameter of one scheme class rather than
the thing that selects which of eighteen classes you get. Strategy code reads
its context explicitly off ``self.scheme`` -- always that one object, never
``scheme.ensemble``. Some of those names are the scheme's own (``lam``,
``trunc_energy``, ``iteration``) and some belong to its ensemble
(``localization``, ``keys_da``, ``proj``, ``prior_enX``, ``state_scaling``),
but the scheme exposes both as properties, so an analysis never has to know
which -- see :class:`~pipt.update_schemes.core.AssimilationScheme`.

The load-bearing test is
:func:`test_bound_strategy_matches_mixed_in_result`: bound and mixed-in must
produce bit-identical steps, or the collapse would silently change every
scheme's numbers.
"""

import numpy as np
import pytest

from pipt.update_schemes.analysis import AnalysisBase
from pipt.update_schemes.analysis.registry import (
    available_analyses,
    get_analysis,
    register_analysis,
)
from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.analysis.subspace import subspace_update


class FakeLocalization:
    name = None


class FakeScheme:
    """The context an analysis reads, and nothing else.

    Flat on purpose: a real scheme exposes ensemble-owned state (``proj``,
    ``prior_enX``, ``keys_da``, ...) as properties of its own, so an analysis
    only ever reads ``scheme.<name>``. A double just needs those names
    present -- it does not have to reproduce the scheme/ensemble split.

    Worth recording: the context is wider than what any one flavour needs on
    its own. ``full_update`` also reads ``prior_enX``, ``Am``, ``ext_Am``
    and ``state_scaling``. Anything binding analyses has to supply these.
    """

    def __init__(self, ne=8, nx=5, lam=0.0, trunc_energy=0.99):
        self.lam = lam
        self.trunc_energy = trunc_energy
        self.keys_da = {}
        self.localization = FakeLocalization()
        self.proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)
        self.prior_enX = np.random.default_rng(7).standard_normal((nx, ne))
        self.Am = None
        self.state_scaling = np.ones(nx)


def _case(seed=0, nx=5, ny=4, ne=8):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((nx, ne)),
        rng.standard_normal((ny, ne)),
        rng.standard_normal((ny, ne)),
    )


# ----------------------------------------------------------------------
# Delegation
# ----------------------------------------------------------------------
def test_scheme_property_returns_the_bound_scheme():
    """``self.scheme`` is what strategy code reads context off of and writes
    results onto -- explicitly, at every use, not synced or resolved lazily.
    """
    scheme = FakeScheme(lam=3.5, trunc_energy=0.77)
    strategy = approx_update(scheme)

    assert strategy.scheme is scheme
    assert strategy.scheme.lam == 3.5
    assert strategy.scheme.trunc_energy == 0.77
    assert strategy.scheme.localization.name is None


def test_unbound_strategy_scheme_falls_back_to_self():
    """An unbound analysis's ``self.scheme`` is itself, so a context read
    goes looking on the analysis -- which does not have it -- and raises
    a plain ``AttributeError`` rather than finding a half-initialised scheme.
    """
    strategy = approx_update()

    assert strategy.scheme is strategy
    with pytest.raises(AttributeError):
        strategy.scheme.lam


def test_writes_land_wherever_the_strategy_writes_them():
    """No __setattr__ magic any more: state an analysis keeps between
    iterations (``full_update``'s ``Am``, the weight-space flavours'
    ``current_W``) lands exactly where it writes it, ``self.scheme``.
    """
    scheme = FakeScheme(lam=1.0)
    strategy = approx_update(scheme)

    strategy.scheme.lam = 99.0
    assert scheme.lam == 99.0

    # An analysis that (wrongly) wrote to itself instead of self.scheme would
    # not be visible to the scheme -- there is nothing to catch that mistake
    # any more, which is the tradeoff for there being no magic to misfire.
    strategy.lam = -1.0
    assert scheme.lam == 99.0


# ----------------------------------------------------------------------
# Equivalence with the mixin path -- the one that matters
# ----------------------------------------------------------------------
@pytest.mark.parametrize("flavour", ["approx", "full", "subspace"])
def test_bound_strategy_matches_mixed_in_result(flavour):
    """Bound and mixed-in must agree bit-for-bit.

    This is what makes collapsing the eighteen classes safe: if the two paths
    diverged, every scheme's numbers would move with no test to catch it.
    """
    strategy_cls = get_analysis(flavour)
    enX, enY, enE = _case()

    # Mixed in: `self` is the scheme, context resolves by inheritance.
    class MixedIn(FakeScheme, strategy_cls):
        pass

    mixed = MixedIn()
    mixed.iteration = 0
    mixed_result = mixed.update(enX=enX, enY=enY, enE=enE)

    # Bound: context resolves by delegation.
    scheme = FakeScheme()
    scheme.iteration = 0
    bound_result = strategy_cls(scheme).update(enX=enX, enY=enY, enE=enE)

    # Whichever kind of step the flavour returns, the two paths must agree.
    for field in ("step", "w_step", "W_step"):
        mixed_value, bound_value = getattr(mixed_result, field), getattr(bound_result, field)
        assert (mixed_value is None) == (bound_value is None), f"{flavour}: paths return different kinds of step"
        if mixed_value is not None:
            np.testing.assert_array_equal(
                np.asarray(bound_value, dtype=float),
                np.asarray(mixed_value, dtype=float),
                err_msg=(
                    f"{flavour}: bound and mixed-in {field} disagree, so "
                    f"collapsing the per-flavour classes would change the numerics."
                ),
            )

    # State kept on the scheme between iterations (full_update caches Am) must
    # land the same way. (Mixed in, self.scheme is self, so both land on `mixed`.)
    for attr in ("Am",):
        assert hasattr(scheme, attr) == hasattr(mixed, attr), (
            f"{flavour}: bound path {'set' if hasattr(scheme, attr) else 'did not set'} "
            f"{attr} but mixed-in path did the opposite"
        )
        if hasattr(mixed, attr) and getattr(mixed, attr) is not None:
            np.testing.assert_array_equal(
                np.asarray(getattr(scheme, attr), dtype=float),
                np.asarray(getattr(mixed, attr), dtype=float),
                err_msg=f"{flavour}: bound and mixed-in disagree on {attr}",
            )


def test_mixin_path_is_untouched_by_the_new_init():
    """Adding __init__ to AnalysisBase must not perturb the mixin MRO.

    Nothing in the scheme's __init__ chain calls super().__init__(), so
    AnalysisBase.__init__ is never invoked there and `_scheme` is never
    set -- which is exactly why mixed-in lookup is unaffected.
    """
    class MixedIn(FakeScheme, approx_update):
        pass

    mixed = MixedIn(lam=2.0)
    assert "_scheme" not in vars(mixed)
    assert mixed.lam == 2.0


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
def test_registry_resolves_the_shipped_flavours():
    assert get_analysis("approx") is approx_update
    assert get_analysis("subspace") is subspace_update
    assert available_analyses() == ["approx", "full", "subspace", "subspace2"]


def test_registry_is_case_insensitive():
    assert get_analysis("APPROX") is approx_update


def test_unknown_flavour_lists_the_valid_ones():
    with pytest.raises(KeyError, match="Unknown analysis flavour 'nope'"):
        get_analysis("nope")


def test_registering_a_duplicate_needs_overwrite():
    class Extra(AnalysisBase):
        def update(self, enX, enY, enE, **kwargs):
            return None

    with pytest.raises(ValueError, match="already registered"):
        register_analysis("approx", Extra)


def test_register_and_resolve_an_out_of_tree_flavour():
    from pipt.update_schemes.analysis import registry

    class Extra(AnalysisBase):
        def update(self, enX, enY, enE, **kwargs):
            return None

    register_analysis("extra_flavour", Extra)
    try:
        assert get_analysis("extra_flavour") is Extra
    finally:
        del registry.ANALYSES["extra_flavour"]
