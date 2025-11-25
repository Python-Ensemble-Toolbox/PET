"""Stochastic iterative ensemble smoother (IES, i.e. EnRML) with *subspace* implementation."""

import numpy as np
from scipy.linalg import solve, lu_solve, lu_factor
import pipt.misc_tools.analysis_tools as at


class subspace_update():
    """
    Ensemble subspace update, as described in  Raanes, P. N., Stordal, A. S., &
    Evensen, G. (2019). Revising the stochastic iterative ensemble smoother.
    Nonlinear Processes in Geophysics, 26(3), 325–338. https://doi.org/10.5194/npg-26-325-2019
    More information about the method is found in Evensen, G., Raanes, P. N., Stordal, A. S., & Hove, J. (2019).
    Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching.
    Frontiers in Applied Mathematics and Statistics, 5(October), 114. https://doi.org/10.3389/fams.2019.00047
    """

    def update(self, enX, enY, enE, **kwargs):

        if self.iteration == 1:  # method requires some initiallization
            self.current_W = np.zeros((self.ne, self.ne))
            self.E = np.dot(enE, self.proj)
        
        # Center ensemble matrices
        Y = np.dot(enY, self.proj)

        omega = np.eye(self.ne) + np.dot(self.current_W, self.proj)
        S = lu_solve(lu_factor(omega.T), Y.T).T

        # Compute scaled misfit (residual between predicted and observed data)
        enRes = self.scale(enY - enE, self.scale_data)

        # Truncate SVD of S
        Us, Ss, VsT = at.truncSVD(S, energy=self.trunc_energy)
        Sinv = np.diag(1/Ss)

        # Compute update step
        X = Sinv @ Us.T @ self.scale(self.E, self.scale_data)
        eigval, eigvec = np.linalg.eig(X @ X.T)
        X2 = Us @ Sinv.T @ eigvec
        X3 = S.T @ X2

        lam_term = np.eye(len(eigval)) + (1+self.lam) * np.diag(eigval)
        deltaM = X3 @ solve(lam_term, X3.T @ self.current_W)
        deltaD = X3 @ solve(lam_term, X2.T @ enRes)
        self.w_step = -self.current_W/(1 + self.lam) - (deltaD - deltaM)/(1 + self.lam)
        

    def scale(self, data, scaling):
        """
        Scale the data perturbations by the data error standard deviation.

        Args:
            data (np.ndarray): data perturbations
            scaling (np.ndarray): data error standard deviation

        Returns:
            np.ndarray: scaled data perturbations
        """

        if len(scaling.shape) == 1:
            return (scaling ** (-1))[:, None] * data
        else:
            return solve(scaling, data)
