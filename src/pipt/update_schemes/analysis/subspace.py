"""Stochastic iterative ensemble smoother (IES) with subspace implementation."""

import numpy as np

from pipt.update_schemes.analysis.base import AnalysisBase, AnalysisResult
import pipt.misc_tools.analysis_tools as at


class subspace_update(AnalysisBase):
    """
    Ensemble subspace update (weight-space IES).

    The update is formulated in the ensemble weight space W (shape ne × ne)
    rather than model space, making it efficient when ne ≪ nx. The caller
    checks ``self.w_step`` (not ``self.step``) to apply the update.

    References
    ----------
    Raanes, P. N., Stordal, A. S., & Evensen, G. (2019).
    Revising the stochastic iterative ensemble smoother.
    Nonlinear Processes in Geophysics, 26(3), 325-338.
    https://doi.org/10.5194/npg-26-325-2019

    Evensen, G., Raanes, P. N., Stordal, A. S., & Hove, J. (2019).
    Efficient implementation of an iterative ensemble smoother for data
    assimilation and reservoir history matching.
    Frontiers in Applied Mathematics and Statistics, 5, 47.
    https://doi.org/10.3389/fams.2019.00047
    """

    def update(self, enX, enY, enE, **kwargs):
        """
        Perform the subspace (weight-space) LM update.

        Sets ``self.scheme.w_step`` (shape ne × ne) and returns ``None`` --
        the caller applies the weight update, not a state-space step.

        Parameters
        ----------
        enX : np.ndarray, shape (nx, ne)
            State ensemble matrix (unused directly; included for interface parity).
        enY : np.ndarray, shape (nd, ne)
            Predicted data ensemble matrix.
        enE : np.ndarray, shape (nd, ne)
            Perturbed observations ensemble.

        Returns
        -------
        AnalysisResult
            The weight-space step ``w_step`` (ne, ne).
        """
        scheme = self.scheme
        ny, ne = enY.shape

        # Fallbacks are built only when the scheme lacks the attribute; see
        # approx_update for why.
        scy = scheme.scale_data if hasattr(scheme, 'scale_data') else np.ones(ny)
        PI  = (scheme.proj if hasattr(scheme, 'proj')
               else (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1))

        # Initialise weight matrix and projected observation perturbations once
        if scheme.iteration == 0:
            scheme.current_W = np.zeros((ne, ne))
            scheme.E = enE @ PI                              # shape: (nd, ne)

        # Whitened predicted-data anomalies. The SVD below, the residual and
        # the observation perturbations must all live in the same (data-scaled)
        # space, otherwise the weights depend on the units of the data.
        Y = self.solve(scy, enY @ PI)                        # shape: (nd, ne)

        # S = Y @ Omega^{-1},  Omega = I + W @ PI
        Omega = np.eye(ne) + scheme.current_W @ PI           # shape: (ne, ne)
        S = np.linalg.solve(Omega.T, Y.T).T                  # shape: (nd, ne)

        # Scaled observation residuals
        enRes = self.solve(scy, enY - enE)                   # shape: (nd, ne)

        # Truncated SVD of S
        Us, Ss, VsT = at.truncSVD(S, energy=scheme.trunc_energy)  # (nd,nr), (nr,), (nr,ne)
        Sinv = (1 / Ss)[:, None]                             # shape: (nr, 1)

        # Projected observation perturbations in reduced space
        X  = Sinv * (Us.T @ self.solve(scy, scheme.E))       # shape: (nr, ne)
        eigval, eigvec = np.linalg.eigh(X @ X.T)            # shape: (nr,), (nr, nr); symmetric, so eigh
        X2 = (Us * Sinv.T) @ eigvec                          # shape: (nd, nr)
        X3 = S.T @ X2                                        # shape: (ne, nr)

        lam_term = np.eye(len(eigval)) + (1 + scheme.lam) * np.diag(eigval)  # shape: (nr, nr)
        deltaM = X3 @ self.solve(lam_term, X3.T @ scheme.current_W)  # shape: (ne, ne)
        deltaD = X3 @ self.solve(lam_term, X2.T @ enRes)           # shape: (ne, ne)

        w_step = (
            -scheme.current_W / (1 + scheme.lam)
            - (deltaD - deltaM) / (1 + scheme.lam)
        )
        return AnalysisResult(w_step=w_step)
