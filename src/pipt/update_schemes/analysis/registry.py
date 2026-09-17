"""Canonical name-to-class lookup for the shipped analysis flavours.

A convenience for introspection (``available_analyses()``) and for anyone
building a scheme's own ``COMPATIBLE_ANALYSES`` dict (see
:class:`~pipt.update_schemes.core.analysis_binding.AnalysisBindingMixin`) without importing
``approx_update``/``full_update``/``subspace_update`` individually.

Registering a flavour here (:func:`register_analysis`) does **not** by itself
make it selectable on any existing scheme: each algorithm class (``ESMDA``,
``EnKF``, ...) declares its own ``COMPATIBLE_ANALYSES``, read directly off the
class rather than computed from this registry, so that reading one scheme's
source tells you everything it supports. Wiring a newly registered flavour
into a scheme means adding it to that scheme's ``COMPATIBLE_ANALYSES`` --
or, for a wholly out-of-tree scheme, registering the combination directly via
:func:`pipt.update_schemes.registry.register_scheme`.

Kept in its own module rather than in :mod:`pipt.update_schemes.analysis.base`:
the concrete flavours import the base, so a registry living there would import
its own importers. ``tests/test_import_hygiene.py`` guards the layering.
"""

from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.analysis.full import full_update
from pipt.update_schemes.analysis.subspace import subspace_update
from pipt.update_schemes.analysis.subspace2 import subspace2_update

__all__ = ["ANALYSES", "available_analyses", "get_analysis", "register_analysis"]


#: Maps an ``analysis`` flavour to the analysis class implementing it.
ANALYSES: dict[str, type] = {
    "approx": approx_update,
    "full": full_update,
    "subspace": subspace_update,
    "subspace2": subspace2_update,
}


def register_analysis(analysis: str, cls: type, *, overwrite: bool = False) -> None:
    """Add an analysis under a flavour name, for later lookup by that name.

    This alone does not make ``cls`` selectable on any existing scheme -- see
    the module docstring for how to actually wire a new flavour in.

    Parameters
    ----------
    analysis : str
        Flavour name to register it under.
    cls : type
        Analysis class implementing it.
    overwrite : bool, optional
        Allow replacing an existing entry. Defaults to ``False``, so two
        packages claiming one name is an error rather than a load-order
        lottery -- matching ``registry.register_scheme``.
    """
    key = str(analysis).lower()
    if key in ANALYSES and not overwrite:
        raise ValueError(
            f"Analysis flavour '{key}' is already registered to "
            f"{ANALYSES[key].__name__}; pass overwrite=True to replace it."
        )
    ANALYSES[key] = cls


def available_analyses() -> list[str]:
    """Return the registered flavour names, sorted."""
    return sorted(ANALYSES)


def get_analysis(analysis: str) -> type:
    """Look up the analysis class for a flavour.

    Raises
    ------
    KeyError
        If the flavour is not registered. The message lists the valid ones.
    """
    key = str(analysis).lower()
    if key in ANALYSES:
        return ANALYSES[key]
    raise KeyError(
        f"Unknown analysis flavour '{analysis}'. "
        f"Available flavours: {', '.join(available_analyses())}."
    )
