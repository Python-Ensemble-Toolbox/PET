"""Weights for a weighted adjoint: the gradient of the data misfit from one adjoint solve per member.

A simulator that computes adjoints returns, by default, the sensitivity of
every observed value to the state: the ``(nd, nx, ne)`` array of
:meth:`pipt.ensembles.ForecastMixin._adjoint_array`, at the cost of one
adjoint solve per observed value. A method that only needs the gradient of a
scalar does not need that Jacobian. In weighted-adjoint mode the simulator is
handed weights :math:`w` and returns the gradient of :math:`\\sum_i w_i d_i`
with respect to each parameter, :math:`G^\\mathsf{T} w`, from a single solve.

:class:`DataMisfitWeights` gives the weights of the data misfit,
:math:`w = C_d^{-1} (d_{obs} - g(x))`. The simulator then returns
:math:`G^\\mathsf{T} C_d^{-1} (d_{obs} - g(x))`, the gradient of the
log-likelihood, which reaches the scheme as the ``(nx, ne)`` array
``ensemble.adjoints``.
"""

import numpy as np
import pandas as pd
from misc.structures import PredictedData
from scipy import linalg

__all__ = ["DataMisfitWeights"]


def stack_vjps(products, indices, count):
    """Place named (direction, parameter) blocks into PET state order."""
    if set(products) != set(indices):
        raise ValueError(f"weighted adjoint names {list(products)} do not match state variables {list(indices)}")
    rows = np.empty((count, max(stop for _, stop in indices.values())))
    for name, (start, stop) in indices.items():
        block = np.asarray(products[name], dtype=float)
        expected = (count, stop - start)
        if block.shape != expected:
            raise ValueError(f"weighted adjoint for {name!r} has shape {block.shape}; expected {expected}")
        rows[:, start:stop] = block
    return rows


class DataMisfitWeights:
    """The adjoint weights :math:`w = C_d^{-1} (d_{obs} - g(x))` of one member, from its forward output.

    Handed to the simulator as its ``adjoint_weights`` and called there with
    each member's output, so it is a plain picklable object rather than a
    closure: parallel runs ship it to their workers. The predictions are read
    off the output exactly as the ensemble reads them into ``pred_data``,
    which keeps the weights row for row with the observations.

    Parameters
    ----------
    layout : misc.structures.DataLayout
        The order of the data vector.
    obs : np.ndarray
        Observations, ``(nd,)``, as the analyses compare against them.
    cov : np.ndarray
        Data covariance ``(nd, nd)`` or variance ``(nd,)``.
    position : dict, optional
        Where each observed label sits in a member's records.
    scale : (minimum, maximum), optional
        Per-data-type max-min scaling the observations went through. The
        weights are then those of the scaled misfit, carried back to the raw
        outputs the simulator differentiates.
    """

    def __init__(self, layout, obs, cov, position=None, scale=None):
        vectors = [row for row in layout.rows if row.size != 1]
        if vectors:
            raise ValueError(
                f"adjoint weights need one value per observed cell; "
                f"{vectors[0].datatype!r} at {vectors[0].label!r} has {vectors[0].size}"
            )
        self.layout = layout
        self.obs = np.asarray(obs, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.position = position
        self.scale = scale

    def residual(self, member) -> np.ndarray:
        """:math:`d_{obs} - g(x)` for one member's output, ``(nd,)``."""
        pred = PredictedData.from_members(self.layout, [member], position=self.position, scale=self.scale)
        return self.obs - pred.matrix[:, 0]

    def vector(self, member) -> np.ndarray:
        """The weights in layout order, ``(nd,)``, as derivatives with respect to the raw outputs."""
        r = self.residual(member)
        w = r / self.cov if self.cov.ndim == 1 else linalg.solve(self.cov, r, assume_a='pos')
        if self.scale is not None:
            minimum, maximum = self.scale
            w = w / np.array([maximum[row.datatype] - minimum[row.datatype] for row in self.layout.rows])
        return w

    def __call__(self, member, member_index=None) -> pd.DataFrame:
        """The weights as a frame of the observed labels and data types; unobserved cells are NaN."""
        return self.layout.to_frame(self.vector(member)).astype(float)
