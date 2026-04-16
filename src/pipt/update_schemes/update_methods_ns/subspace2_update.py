"""Stochastic iterative ensemble smoother (IES, i.e. EnRML) with *subspace* implementation."""

import numpy as np
from scipy.linalg import solve, lu_solve, lu_factor
import pipt.misc_tools.analysis_tools as at





class subspace2_update():
    """
    Ensemble subspace update, as described in  Raanes, P. N., Stordal, A. S., &
    Evensen, G. (2019). Revising the stochastic iterative ensemble smoother.
    Nonlinear Processes in Geophysics, 26(3), 325–338. https://doi.org/10.5194/npg-26-325-2019

    """

    def update(self, enX, enY, enE, **kwargs):

        if self.iteration == 1:  # method requires some initiallization
            self.current_W = np.eye(self.ne)
            self.D = self.scale(enE, self.scale_data)
        # Scale everything so that data uncertainty is I
        sY = self.scale(enY, self.scale_data)
        Y = np.linalg.solve(self.current_W.T,sY.T).T #Raanes
        Y = np.dot(Y, self.proj) * np.sqrt(self.ne - 1) #Raanes


        #Gradients

        deltaD = Y.T @ (self.D - sY)
        deltaM = (self.ne-1)*(np.eye(self.ne)-self.current_W)

        #Hessian
        S = Y.T @ Y + np.eye(self.ne)*(self.ne-1)

        self.W_step =   np.linalg.solve(S , (deltaM + deltaD))/(1 + self.lam)


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