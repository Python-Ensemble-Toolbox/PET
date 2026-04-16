"""Stochastic iterative ensemble smoother (IES, i.e. EnRML) with *subspace* implementation."""

import numpy as np
from scipy.linalg import solve, lu_solve, lu_factor, cho_solve

import pipt.misc_tools.analysis_tools as at


class margIS_update():
    """
    MargIES update from Stordal et.al.
    This is now implemented with perturbed observations, which means that we set a prior belief on the data uncertainty.
    Thus, the prior is an invers chi2 distriubtuinm and after scaling the mean varians is 1.
    """

    def update(self, enX, enY, enE, **kwargs):

        if self.iteration == 1:  # method requires some initiallization
            self.current_W = np.eye(self.ne)
            self.current_w = np.zeros(self.ne)
            self.D = self.scale(enE, self.scale_data)
            # Scale everything so that data uncertainty is I

        sY = self.scale(enY, self.scale_data) #Scaling is same as with 'known' uncertainty, hence makes sense to set s = 1
        self.S = 0

        deltaD = 0
        deltaD_sqrt = 0

        Y = np.linalg.solve(self.current_W.T, sY.T).T
        Y = Y @ self.proj * np.sqrt(self.ne - 1)
        index = np.arange(0, 70, 70)  # Has to be specified via data types (or select each data)...
        M = 1 #Numbers of data per type. Computed from index
        s = 1 #should be default option with possibility to change in setup
        nu = self.ne-1 #should be default option with possibility to change in setup
        for j in range(70):

            delta = self.D[index,:]-sY[index,:]
            Chi = np.sum(delta * delta, axis = 0)
            Chi = np.mean(Chi)
            Ratio = (M + nu) / (Chi + nu*s*s)
            #Ratio = 1
            #Gradient
            deltaD = deltaD + (Y[index,:] * Ratio).T @ delta
            deltaD_sqrt = deltaD_sqrt + np.mean((Y[index, :] * Ratio).T @ delta ,axis=1)
            # Hessian
            self.S = self.S + (Y[index,:] * Ratio).T @ Y[index,:]
            index += 1

        deltaM = (self.ne-1)*(np.eye(self.ne)-self.current_W)
        deltaM_sqrt = (self.ne-1)*self.current_w
        self.S = self.S + np.eye(self.ne) * (self.ne - 1)
        Delta = deltaM + deltaD
        Delta_sqrt = deltaM_sqrt + deltaD_sqrt


        self.W_step =   np.linalg.solve(self.S, Delta) / (1 + self.lam)
       # self.sqrt_w_step = np.linalg.solve(self.S, Delta_sqrt) / (1 + self.lam)

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


























