import numpy as np
from scipy.linalg import solve
import copy as cp
from pipt.misc_tools import analysis_tools as at

class margIS_update():

    """
    Placeholder for private margIS method
    """
    def update(self):
        if self.iteration == 1: # method requires some initiallization
            self.aug_prior = cp.deepcopy(at.aug_state(self.prior_state, self.list_states))
            self.mean_prior = self.aug_prior.mean(axis=1)
            self.X = (self.aug_prior - np.dot(np.resize(self.mean_prior, (len(self.mean_prior), 1)),
                                              np.ones((1, self.ne))))
            self.W = np.eye(self.ne)
            self.current_w = np.zeros((self.ne,))
            self.E = np.dot(self.real_obs_data, self.proj)

        M = len(self.real_obs_data)
        Ytmp = solve(self.W, self.proj)
        if len(self.scale_data.shape) == 1:
            Y = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) * \
                np.dot(self.aug_pred_data, Ytmp)
        else:
            Y = solve(self.scale_data, np.dot(self.aug_pred_data, Ytmp))

        pred_data_mean = np.mean(self.aug_pred_data, 1)
        delta_d = (self.obs_data_vector - pred_data_mean)

        if len(self.cov_data.shape) == 1:
            S = np.dot(delta_d, (self.cov_data**(-1)) * delta_d)
            Ratio = M / S
            grad_lklhd = np.dot(Y.T * Ratio, (self.cov_data**(-1)) * delta_d)
            grad_prior = (self.ne - 1) * self.current_w
            self.C_w = (np.dot(Ratio * Y.T, np.dot(np.diag(self.cov_data ** (-1)), Y)) + (self.ne - 1) * np.eye(self.ne))
        else:
            S = np.dot(delta_d, solve(self.cov_data, delta_d))
            Ratio = M / S
            grad_lklhd = np.dot(Y.T * Ratio, solve(self.cov_data, delta_d))
            grad_prior = (self.ne - 1) * self.current_w
            self.C_w = (np.dot(Ratio * Y.T, solve(self.cov_data, Y)) + (self.ne - 1) * np.eye(self.ne))

        self.sqrt_w_step = solve(self.C_w, grad_prior + grad_lklhd)
