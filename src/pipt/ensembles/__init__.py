"""Ensemble containers for data assimilation.

Mirrors the layout of :mod:`popt.ensembles`.
"""

from .ensemble_base import AssimilationEnsemble
from .forecast import ForecastMixin, OutlierMixin
from .local_analysis import LocalAnalysisMixin

#: Historical name, kept so existing code and subclasses keep working.
Ensemble = AssimilationEnsemble

__all__ = [
    "AssimilationEnsemble",
    "Ensemble",
    "ForecastMixin",
    "OutlierMixin",
    "LocalAnalysisMixin",
]
