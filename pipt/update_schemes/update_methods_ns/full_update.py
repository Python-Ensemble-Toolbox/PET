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

    def update(self):
        aug_state = at.aug_state(self.current_state, self.list_states)
        aug_prior_state = at.aug_state(self.prior_state, self.list_states)

        delta_state = (self.state_scaling**(-1))[:, None]*np.dot(aug_state, self.proj)

        u_d, s_d, v_d = np.linalg.svd(self.pert_preddata, full_matrices=False)
        if self.trunc_energy < 1:
            ti = (np.cumsum(s_d) / sum(s_d)) <= self.trunc_energy
            u_d, s_d, v_d = u_d[:, ti].copy(), s_d[ti].copy(), v_d[ti, :].copy()

        if len(self.scale_data.shape) == 1:
            x_1 = np.dot(u_d.T, np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) *
                         (self.real_obs_data - self.aug_pred_data))
        else:
            x_1 = np.dot(u_d.T, solve(self.scale_data,
                         (self.real_obs_data - self.aug_pred_data)))
        x_2 = solve(((self.lam + 1) * np.eye(len(s_d)) + np.diag(s_d ** 2)), x_1)
        x_3 = np.dot(np.dot(v_d.T, np.diag(s_d)), x_2)
        delta_m1 = np.dot((self.state_scaling[:, None]*delta_state), x_3)

        x_4 = np.dot(self.Am.T, (self.state_scaling**(-1))
                     [:, None]*(aug_state - aug_prior_state))
        x_5 = np.dot(self.Am, x_4)
        x_6 = np.dot(delta_state.T, x_5)
        x_7 = np.dot(v_d.T, solve(
            ((self.lam + 1) * np.eye(len(s_d)) + np.diag(s_d ** 2)), np.dot(v_d, x_6)))
        delta_m2 = -np.dot((self.state_scaling[:, None]*delta_state), x_7)

        self.step = delta_m1 + delta_m2
