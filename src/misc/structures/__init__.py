"""PET's data containers.

``PETDataFrame`` is the ragged table observed and predicted data arrive in
and are saved as; on the analysis path the data live in matrices ordered by
a ``DataLayout`` (``PredictedData`` for the forecast). The state is a plain
``(nx, ne)`` array whose variable layout is a ``StateLayout``.
"""
from .structures import PETDataFrame
from misc.structures.layout import DataLayout, LayoutRow, StateLayout
from misc.structures.predicted import PredictedData

__all__ = ["PETDataFrame", "DataLayout", "LayoutRow", "PredictedData", "StateLayout"]
