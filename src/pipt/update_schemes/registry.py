"""Explicit registry of selectable assimilation schemes.

PIPT historically resolved a scheme by string surgery on the config::

    getattr(import_module('pipt.update_schemes.' + daalg[0]),
            f'{daalg[1]}_{analysis}')

That failed badly: a typo in ``daalg`` surfaced as a bare
``ModuleNotFoundError`` or ``AttributeError`` naming a symbol the user never
wrote, there was no way to ask what the valid combinations are, and any tool
wanting to list the available schemes had to guess at module contents.

A later refactor replaced the string surgery with an explicit table, but built
it from eighteen hand-written classes -- one per ``(scheme, analysis)``
combination -- because the analysis flavour used to be baked into the class
through mixin composition. It no longer is: every algorithm class declares its
own ``COMPATIBLE_ANALYSES`` (flavour name -> analysis class) and takes
``analysis`` as a constructor argument that picks from it (see
``AnalysisBindingMixin`` for how). The per-combination classes had become pure
duplication -- ``esmda_approx`` was nothing but ``class esmda_approx(ESMDA):
FLAVOUR = "approx"`` -- so this module now derives the regular combinations
from two small tables instead of storing eighteen classes:

``ALGORITHMS``
    One entry per algorithm, e.g. ``"esmda" -> ESMDA``.
``SPECIAL_SCHEMES``
    Combinations backed by a real, distinct implementation rather than a
    registered analysis flavour -- ``esmda_hybrid`` (multilevel ES-MDA) is
    an algorithm in its own right that happens to share the ``esmda`` name,
    not an alias.

Extending the registry
----------------------
Schemes living outside this repository can register themselves without
editing this file::

    from pipt.update_schemes.registry import register_scheme
    register_scheme("myscheme", "approx", MyScheme)
"""

from functools import partial

from pipt.update_schemes.enkf import EnKF
from pipt.update_schemes.enrml import GNEnRML, LMEnRML, co_lm_enrml, gn_enrml
from pipt.update_schemes.es import ES
from pipt.update_schemes.esmda import ESMDA
# esmda_hybrid is a multilevel variant and lives with the multilevel machinery.
from pipt.update_schemes.multilevel import esmda_hybrid

__all__ = [
    "ALGORITHMS",
    "SPECIAL_SCHEMES",
    "available_schemes",
    "get_scheme",
    "register_scheme",
]


#: One class per algorithm. The analysis flavour is a constructor argument,
#: not part of this mapping.
ALGORITHMS: dict[str, type] = {
    "enkf": EnKF,
    "es": ES,
    "esmda": ESMDA,
    "lmenrml": LMEnRML,
    "gnenrml": GNEnRML,
    # Historical names still found in configs. Each is a thin subclass whose
    # COMPATIBLE_ANALYSES holds the one flavour the name always meant, so
    # asking it for another flavour fails the same way as any other scheme.
    "co_lm_enrml": co_lm_enrml,
    "gn_enrml": gn_enrml,
}

#: Combinations backed by a distinct implementation rather than a registered
#: analysis flavour. Checked before the generic algorithm+flavour resolution,
#: so also the way to override or add a genuinely different scheme.
SPECIAL_SCHEMES: dict[tuple[str, str], type] = {
    ("esmda", "hybrid"): esmda_hybrid,
}


def register_scheme(scheme: str, analysis: str, cls: type, *, overwrite: bool = False) -> None:
    """Add a scheme to the registry.

    Parameters
    ----------
    scheme : str
        Scheme name, as it appears in the config's ``scheme`` key.
    analysis : str
        Analysis flavour, as it appears in the config's ``analysis`` key.
    cls : type
        Class implementing the combination.
    overwrite : bool, optional
        Allow replacing an existing entry. Defaults to ``False`` so that two
        packages silently claiming the same key is an error rather than a
        load-order lottery.
    """
    key = (str(scheme).lower(), str(analysis).lower())
    if not overwrite:
        existing = _resolve(key)
        if existing is not None:
            name = getattr(existing, "func", existing).__name__
            raise ValueError(
                f"Scheme {key} is already registered to {name}; "
                f"pass overwrite=True to replace it."
            )
    SPECIAL_SCHEMES[key] = cls


def available_schemes() -> list[tuple[str, str]]:
    """Return the registered ``(scheme, analysis)`` combinations, sorted."""
    combos = {
        (name, flavour)
        for name, cls in ALGORITHMS.items()
        for flavour in cls.COMPATIBLE_ANALYSES
    }
    combos |= set(SPECIAL_SCHEMES)
    return sorted(combos)


#: Generic algorithm+flavour combinations, built lazily and cached so that
#: repeated lookups of the same combination return the same object -- as they
#: did when this was a flat dict of classes.
_generic_cache: dict[tuple[str, str], partial] = {}


def _resolve(key: tuple[str, str]):
    """Look up ``key`` without raising. ``None`` if it is not registered."""
    if key in SPECIAL_SCHEMES:
        return SPECIAL_SCHEMES[key]
    algo, flavour = key
    if algo in ALGORITHMS and flavour in ALGORITHMS[algo].COMPATIBLE_ANALYSES:
        if key not in _generic_cache:
            _generic_cache[key] = partial(ALGORITHMS[algo], analysis=flavour)
        return _generic_cache[key]
    return None


def get_scheme(scheme: str, analysis: str):
    """Look up the constructor for a ``(scheme, analysis)`` combination.

    Returns
    -------
    callable
        Either the class directly (for a :data:`SPECIAL_SCHEMES` entry) or the
        algorithm class with ``analysis`` pre-bound via :func:`functools.partial`.
        Either way, call it as ``result(da_input, en_input, sim)``.

    Raises
    ------
    KeyError
        If the combination is not registered. The message distinguishes an
        unknown scheme from a known scheme with an unsupported analysis
        flavour, and lists the valid options in both cases.
    """
    key = (str(scheme).lower(), str(analysis).lower())
    resolved = _resolve(key)
    if resolved is not None:
        return resolved

    if key[0] not in ALGORITHMS:
        raise KeyError(
            f"Unknown assimilation scheme '{scheme}'. "
            f"Available schemes: {', '.join(sorted(ALGORITHMS))}."
        )

    flavours = sorted(flavour for name, flavour in available_schemes() if name == key[0])
    raise KeyError(
        f"Scheme '{scheme}' has no '{analysis}' analysis flavour. "
        f"Available flavours for '{scheme}': {', '.join(flavours)}."
    )
