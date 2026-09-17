"""Ensemble-transform IES: Gauss-Newton on the ne x ne transform matrix."""

import numpy as np

from pipt.update_schemes.analysis.base import AnalysisBase, AnalysisResult


class subspace2_update(AnalysisBase):
    """
    Ensemble-transform subspace update (matrix-formulation IES).

    Solves directly for the ensemble transform ``W`` (shape ne x ne), starting from
    ``W = I``, minimising

        J(W) = 0.5 (ne-1) ||W - I||_F^2 + 0.5 ||D - g(xbar + Xp W)||^2_{Cd^-1}

    by Gauss-Newton. Unlike :class:`subspace_update` it uses the analytic data
    covariance throughout -- via ``scale_data`` -- rather than the ensemble
    representation ``E E.T``, so there is no SVD and ``energy``/``trunc_energy`` is
    not consulted. The trial state is reconstructed by ``propose_state`` as
    ``mean(prior_enX) + prior_anomalies * sqrt(ne - 1) @ W``.

    This is exactly :class:`margIS_update` with ``Ratio`` fixed at 1: the data error
    scale is taken as known instead of being marginalised over an inverse-chi2 prior.
    ``tests/assimilation/test_subspace2.py`` pins that identity.

    References
    ----------
    Raanes, P. N., Stordal, A. S., & Evensen, G. (2019).
    Revising the stochastic iterative ensemble smoother.
    Nonlinear Processes in Geophysics, 26(3), 325-338.
    https://doi.org/10.5194/npg-26-325-2019
    """

    def update(self, enX, enY, enE, **kwargs):
        """
        Perform one Gauss-Newton step on the ensemble transform.

        Parameters
        ----------
        enX : np.ndarray, shape (nx, ne)
            State ensemble matrix (unused; the reconstruction works from the prior).
        enY : np.ndarray, shape (nd, ne)
            Predicted data ensemble matrix.
        enE : np.ndarray, shape (nd, ne)
            Perturbed observations.

        Returns
        -------
        AnalysisResult
            The transform step ``W_step`` of shape (ne, ne).
        """
        scheme = self.scheme
        ne = enY.shape[1]

        if scheme.iteration == 0:
            scheme.current_W = np.eye(ne)

        # Whiten both the observations and the predictions with the *current*
        # scale. ES-MDA redraws enE and scale_data at every assimilation step
        # (with alpha[iteration] * cov_data), so caching D on the first call --
        # as the reference implementation does -- would drive later steps with
        # the first step's observations whitened by the first step's factor,
        # while sY used the current one. The two would be in different units and
        # nothing would report it: the run still completes and the misfit still
        # falls. It is one solve, so there is nothing to gain by keeping it.
        D  = self.solve(scheme.scale_data, enE)              # shape: (nd, ne)
        sY = self.solve(scheme.scale_data, enY)              # shape: (nd, ne)

        # Predicted anomalies seen through the current transform.
        Y = np.linalg.solve(scheme.current_W.T, sY.T).T      # shape: (nd, ne)
        Y = Y @ scheme.proj * np.sqrt(ne - 1)                # shape: (nd, ne)

        # Gradients: data misfit and the prior pull back towards W = I.
        deltaD = Y.T @ (D - sY)                              # shape: (ne, ne)
        deltaM = (ne - 1) * (np.eye(ne) - scheme.current_W)  # shape: (ne, ne)

        # Gauss-Newton Hessian.
        S = Y.T @ Y + np.eye(ne) * (ne - 1)                  # shape: (ne, ne)

        W_step = np.linalg.solve(S, deltaM + deltaD) / (1 + scheme.lam)
        return AnalysisResult(W_step=W_step)
