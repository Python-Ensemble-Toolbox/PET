"""Stochastic iterative ensemble smoother (IES, i.e. EnRML) with *subspace* implementation."""

import numpy as np
from copy import deepcopy
import copy as cp
from scipy.linalg import solve, solve_banded, cholesky, lu_solve, lu_factor, inv
import pickle
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

    def update(self):
        if self.iteration == 1:  # method requires some initiallization
            self.current_W = np.zeros((self.ne, self.ne))
            self.E = np.dot(self.real_obs_data, self.proj)
        Y = np.dot(self.aug_pred_data, self.proj)
        # Y = self.pert_preddata

        omega = np.eye(self.ne) + np.dot(self.current_W, self.proj)
        LU = lu_factor(omega.T)
        S = lu_solve(LU, Y.T).T

        # scaled_misfit = (self.aug_pred_data - self.real_obs_data)
        if len(self.scale_data.shape) == 1:
            scaled_misfit = (self.scale_data ** (-1)
                             )[:, None] * (self.aug_pred_data - self.real_obs_data)
        else:
            scaled_misfit = solve(
                self.scale_data, (self.aug_pred_data - self.real_obs_data))

        u, s, v = np.linalg.svd(S, full_matrices=False)
        if self.trunc_energy < 1:
            ti = (np.cumsum(s) / sum(s)) <= self.trunc_energy
            if sum(ti) == 0:
                # the first singular value contains more than the prescibed trucation energy.
                ti[0] = True
            u, s, v = u[:, ti].copy(), s[ti].copy(), v[ti, :].copy()

        ps_inv = np.diag([el_s ** (-1) for el_s in s])
        # if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
        X = np.dot(ps_inv, np.dot(u.T, self.E))
        if len(self.scale_data.shape) == 1:
            X = np.dot(ps_inv, np.dot(u.T, (self.scale_data ** (-1))[:, None]*self.E))
        else:
            X = np.dot(ps_inv, np.dot(u.T, solve(self.scale_data, self.E)))
        Lam, z = np.linalg.eig(np.dot(X, X.T))
        # else:
        #     X =  np.dot(np.dot(ps_inv, np.dot(u.T, np.diag(self.cov_data))),np.dot(u,ps_inv))
        #     Lam, z = np.linalg.eig(X)
        # Lam = s**2
        # z = np.eye(len(s))

        X2 = np.dot(u, np.dot(ps_inv.T, z))
        X3 = np.dot(S.T, X2)

        # X3_old = np.dot(X2, np.linalg.solve(np.eye(len(Lam)) + np.diag(Lam), X2.T))
        step_m = np.dot(X3, solve(np.eye(len(Lam)) + (1+self.lam) *
                        np.diag(Lam), np.dot(X3.T, self.current_W)))

        step_d = np.dot(X3, solve(np.eye(len(Lam)) + (1+self.lam) *
                        np.diag(Lam), np.dot(X2.T, scaled_misfit)))

        # step_d = np.dot(np.linalg.inv(omega).T, np.dot(np.dot(Y.T, X2),
        #                                                solve((np.eye(len(Lam)) + (self.lam+1)*np.diag(Lam)),
        #                                                       np.dot(X2.T, scaled_misfit))))
        self.w_step = -self.current_W/(1+self.lam) - (step_d - step_m/(1+self.lam))
