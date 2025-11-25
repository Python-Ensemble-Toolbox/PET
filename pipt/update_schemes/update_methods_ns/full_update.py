"""EnRML (IES) as in 2013."""

import numpy as np
from copy import deepcopy
import copy as cp
from scipy.linalg import solve, solve_banded, cholesky, lu_solve, lu_factor, inv
import pickle
import pipt.misc_tools.analysis_tools as at


class full_update():
    """
    Full LM Update scheme as defined in "Chen, Y., & Oliver, D. S. (2013). Levenberg–Marquardt forms of the iterative ensemble
    smoother for efficient history matching and uncertainty quantification. Computational Geosciences, 17(4), 689–703.
    https://doi.org/10.1007/s10596-013-9351-5". Note that for a EnKF or ES update, or for update within GN scheme, lambda = 0.

    !!! note
        no localization is implemented for this method yet.
    """

    def update(self, enX, enY, enE, **kwargs):

        # Get prior ensemble if provided
        priorX = kwargs.get('prior', self.prior_enX)

        if self.Am is None:
            self.ext_Am() # do this only once 

        # Scale and center the ensemble matrecies
        enYcentered = self.scale(np.dot(enY, self.proj), self.scale_data) 
        enXcentered = self.scale(np.dot(enX, self.proj), self.state_scaling)

        # Perform tuncated SVD
        u_d, s_d, v_d = at.truncSVD(enYcentered, energy=self.trunc_energy)

        # Compute the update step
        x_1 = np.dot(u_d.T, self.scale(enE - enY, self.scale_data))
        x_2 = solve(((self.lam + 1) * np.eye(len(s_d)) + np.diag(s_d ** 2)), x_1)
        x_3 = np.dot(np.dot(v_d.T, np.diag(s_d)), x_2)
        delta_m1 = np.dot((self.state_scaling[:, None]*enXcentered), x_3)

        x_4 = np.dot(self.Am.T, (self.state_scaling**(-1))[:, None]*(enX - priorX))
        x_5 = np.dot(self.Am, x_4)
        x_6 = np.dot(enXcentered.T, x_5)
        x_7 = np.dot(v_d.T, solve(((self.lam + 1) * np.eye(len(s_d)) + np.diag(s_d ** 2)), np.dot(v_d, x_6)))
        delta_m2 = -np.dot((self.state_scaling[:, None]*enXcentered), x_7)

        self.step = delta_m1 + delta_m2
    

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
    
    def ext_Am(self, *args, **kwargs):
        """
        The class is initialized by calculating the required Am matrix.
        """

        delta_scaled_prior = self.state_scaling[:, None] * np.dot(self.prior_enX, self.proj)
        u_d, s_d, v_d = np.linalg.svd(delta_scaled_prior, full_matrices=False)

        # remove the last singular value/vector. This is because numpy returns all ne values, while the last is actually
        # zero. This part is a good place to include eventual additional truncation.
        energy = 0
        trunc_index = len(s_d) - 1  # inititallize
        for c, elem in enumerate(s_d):
            energy += elem
            if energy / sum(s_d) >= self.trunc_energy:
                trunc_index = c  # take the index where all energy is preserved
                break
        u_d, s_d, v_d = u_d[:, :trunc_index +
                                1], s_d[:trunc_index + 1], v_d[:trunc_index + 1, :]
        self.Am = np.dot(u_d, np.eye(trunc_index + 1) *
                         ((s_d ** (-1))[:, None]))  # notation from paper
