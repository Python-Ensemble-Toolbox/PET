"""Analysis-step analyses.

An *analysis* computes the state update for one assimilation
iteration. The flavours differ only in how the ensemble-approximated
sensitivity is inverted; they share a calling convention and their
linear-algebra helpers.

The analysis is a *parameter* of a scheme, not part of its identity::

    ESMDA(keys_da, keys_en, sim, analysis="subspace")

Layout
------
``base``
    :class:`AnalysisBase` -- the shared contract and helpers.
``approx``, ``full``, ``subspace``, ``subspace2``
    The four registered flavours.
``hybrid``, ``margis``
    Flavours consumed as mixins rather than through the registry: ``hybrid``
    belongs to the multilevel scheme and ``margis`` is backed by a private
    package when installed.
``registry``
    Name-to-class lookup, plus :func:`register_analysis` for out-of-tree
    flavours.

These previously lived in ``update_schemes.update_methods_ns`` while this
package held only the base class, because the flavours were consumed as mixins
and re-exporting them here would have formed an import cycle. Now that schemes
hold an analysis rather than inheriting one, they live together.
"""

from .base import AnalysisBase, AnalysisResult
from .approx import approx_update
from .full import full_update
from .hybrid import hybrid_update
from .subspace import subspace_update
from .subspace2 import subspace2_update
from .registry import (
    ANALYSES,
    available_analyses,
    get_analysis,
    register_analysis,
)

__all__ = [
    "AnalysisBase",
    "AnalysisResult",
    "approx_update",
    "full_update",
    "subspace_update",
    "subspace2_update",
    "hybrid_update",
    "ANALYSES",
    "available_analyses",
    "get_analysis",
    "register_analysis",
]
