"""The predicted-data ensemble as the analyses use it: an ``(nd, ne)`` matrix in a layout's row order."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from misc.structures.layout import DataLayout

__all__ = ["PredictedData", "member_cell"]


def member_cell(member, row, position):
    """One member's value for one observed cell, from its records or its frame."""
    if isinstance(member, pd.DataFrame):
        return member.loc[row.label, row.datatype]
    where = row.label if position is None else position[row.label]
    try:
        record = member[where]
    except (IndexError, TypeError) as exc:
        raise ValueError(
            f"no record at position {where!r} for label {row.label!r}: the simulator must report every "
            f"observed label, and name the labels in `true_order` when they are not positions."
        ) from exc
    try:
        return record[row.datatype]
    except KeyError as exc:
        raise KeyError(f"member output has no {row.datatype!r} at {row.label!r}") from exc


@dataclass
class PredictedData:
    """Predictions for every observed cell, one column per member.

    Built straight from what each member's simulation returned, so its rows
    are the layout's rows: the same rows the observation vector and its
    variance have. The frame the older code passed around is available as a
    view (:meth:`to_frame`) for saving and inspection.
    """

    matrix: np.ndarray
    layout: DataLayout

    @property
    def nd(self) -> int:
        return self.matrix.shape[0]

    @property
    def ne(self) -> int:
        return self.matrix.shape[1]

    @classmethod
    def from_members(cls, layout, members, position=None, scale=None, transform=None) -> "PredictedData":
        """Fill the matrix from one output per member.

        Parameters
        ----------
        members : sequence
            One output per member: a list of records (one dict per report
            point, keyed by data type) or a DataFrame indexed by label.
        position : dict, optional
            Where each observed label sits in a member's records. Omit when
            the labels are the positions.
        scale : (minimum, maximum), optional
            Per-data-type max-min scaling to apply, as the observations were
            scaled: ``(value - minimum) / (maximum - minimum)``.
        transform : callable, optional
            ``transform(row, values) -> values``, applied to a member's
            (scaled) raw values before they enter the matrix -- how a
            simulated seismic vintage becomes the wavelet coefficients the
            observed one was reduced to. Its output must have ``row.size``
            values; the raw values need not.
        """
        matrix = np.empty((layout.nd, len(members)))
        minimum, maximum = scale if scale is not None else (None, None)
        for j, member in enumerate(members):
            for row in layout.rows:
                values = np.ravel(np.asarray(member_cell(member, row, position), dtype=float))
                if scale is not None:
                    low = minimum[row.datatype]
                    values = (values - low) / (maximum[row.datatype] - low)
                if transform is not None:
                    values = np.ravel(np.asarray(transform(row, values), dtype=float))
                if values.size != row.size:
                    raise ValueError(
                        f"member {j}: {row.datatype!r} at {row.label!r} has {values.size} values; "
                        f"the observation has {row.size}"
                    )
                matrix[row.rows, j] = values
        return cls(matrix, layout)

    @classmethod
    def from_frame(cls, layout, frame, ne) -> "PredictedData":
        """From a prediction frame whose cells are ``(ne,)`` or ``(size, ne)`` arrays."""
        return cls(layout.matrix(frame, ne), layout)

    def to_frame(self, name=None):
        """The frame view: one cell per observed label and data type."""
        return self.layout.to_frame(self.matrix, name=name)

    def take_members(self, index) -> "PredictedData":
        """The predictions of the members ``index`` names, in that order."""
        return PredictedData(self.matrix[:, index], self.layout)

    def rows_of(self, datatype):
        """The row slices holding ``datatype``, in layout order."""
        return [row.rows for row in self.layout.rows if row.datatype == datatype]
