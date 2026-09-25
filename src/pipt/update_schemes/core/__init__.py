"""Machinery every assimilation scheme is built from.

Separated from the algorithms themselves so that ``pipt.update_schemes`` reads
as a list of schemes rather than a mixture of schemes and the scaffolding they
stand on. Two pieces::

    class ESMDA(AssimilationScheme)

:class:`AssimilationScheme`
    The iteration loop, convergence bookkeeping, restart handling, the run
    table, the result object, and the diagnostics and artifact saving that
    surround a run. Subclasses supply :meth:`~AssimilationScheme.update_step`.
:class:`AnalysisBindingMixin`
    Resolves the ``analysis`` flavour to an analysis object and delegates
    ``update()`` to it, so the flavour is a parameter rather than part of the
    class name.
"""

from .scheme_base import AssimilationResult, AssimilationScheme, StepReport, restart_options
from .analysis_binding import AnalysisBindingMixin

__all__ = [
    "AssimilationScheme",
    "AssimilationResult",
    "StepReport",
    "restart_options",
    "AnalysisBindingMixin",
]
