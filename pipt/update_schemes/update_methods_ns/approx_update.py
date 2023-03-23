import numpy as np
from copy import deepcopy
import copy as cp
from scipy.linalg import solve, solve_banded, cholesky, lu_solve, lu_factor, inv
import pickle
import pipt.misc_tools.analysis_tools as at

class approx_update():
    """
    Approximate LM Update scheme as defined in "Chen, Y., & Oliver, D. S. (2013). Levenberg–Marquardt forms of the iterative ensemble
    smoother for efficient history matching and uncertainty quantification. Computational Geosciences, 17(4), 689–703.
    https://doi.org/10.1007/s10596-013-9351-5". Note that for a EnKF or ES update, or for update within GN scheme, lambda = 0.
    """
    def update(self):
        # calc the svd of the scaled data pertubation matrix
        u_d, s_d, v_d = np.linalg.svd(self.pert_preddata, full_matrices=False)
        aug_state = at.aug_state(self.current_state, self.list_states,self.cell_index)

        # remove the last singular value/vector. This is because numpy returns all ne values, while the last is actually
        # zero. This part is a good place to include eventual additional truncation.
        if self.trunc_energy < 1:
            ti = (np.cumsum(s_d) / sum(s_d)) <= self.trunc_energy
            u_d, s_d, v_d = u_d[:, ti].copy(), s_d[ti].copy(), v_d[ti, :].copy()
        if 'localization' in self.keys_da:
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                if len(self.scale_data.shape) == 1:
                    E_hat = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) * self.E
                    x_0 = np.dot(np.diag(s_d[:] ** (-1)), np.dot(u_d[:, :].T, E_hat))
                    Lam, z = np.linalg.eig(np.dot(x_0, x_0.T))
                else:
                    E_hat = solve(self.scale_data, self.E)
                    x_0 = np.dot(np.diag(s_d[:] ** (-1)), np.dot(u_d[:, :].T, E_hat))
                    Lam, z = np.linalg.eig(np.dot(x_0, x_0.T))

                X = np.dot(np.dot(v_d.T, z), solve((self.lam + 1) * np.diag(Lam) + np.eye(len(Lam)),
                                                   np.dot(u_d[:, :], np.dot(np.diag(s_d[:] ** (-1)).T, z)).T))

            else:
                X = np.dot(np.dot(v_d.T, np.diag(s_d)),
                           solve(((self.lam + 1) * np.eye(len(s_d)) + np.diag(s_d ** 2)), u_d.T))

            # we must perform localization
            # store the size of all data
            data_size = [[self.obs_data[int(time)][data].size if self.obs_data[int(time)][data] is not None else 0
                          for data in self.list_datatypes] for time in self.assim_index[1]]

            f = self.keys_da['localization']

            if f[1][0] == 'autoadaloc':

                # Mean state and perturbation matrix
                mean_state = np.mean(aug_state, 1)
                if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                    pert_state = (self.state_scaling**(-1))[:,None] * (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                                                 np.ones((1, self.ne))))
                else:
                    pert_state = (self.state_scaling**(-1))[:,None] * np.dot(aug_state,self.proj)
                if len(self.scale_data.shape) == 1:
                    scaled_delta_data = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                               np.ones((1, pert_state.shape[1]))) * (
                                                self.real_obs_data - self.aug_pred_data)
                else:
                    scaled_delta_data = solve(self.scale_data, (self.real_obs_data - self.aug_pred_data))

                self.step = self.localization.auto_ada_loc(self.state_scaling[:,None] * pert_state, np.dot(X, scaled_delta_data),
                                                           self.list_states,
                                                           **{'prior_info': self.prior_info})
            elif sum(['dist_loc' in el for el in f]) >= 1:

                local_mask = self.localization.localize(self.list_datatypes, [self.keys_da['truedataindex'][int(elem)]
                                                                         for elem in self.assim_index[1]],
                                                        self.list_states, self.ne, self.prior_info, data_size)
                mean_state = np.mean(aug_state, 1)
                if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                    pert_state = (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                                     np.ones((1, self.ne))))
                else:
                    pert_state = (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                                     np.ones((1, self.ne)))) / (np.sqrt(self.ne - 1))

                if len(self.scale_data.shape) == 1:
                    scaled_delta_data = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                               np.ones((1, pert_state.shape[1]))) * (
                                                self.read_obs_data - self.aug_pred_data)
                else:
                    scaled_delta_data = solve(self.scale_data, (self.real_obs_data - self.aug_pred_data))

                self.step = local_mask.multiply(np.dot(pert_state, X)).dot(scaled_delta_data)

            else:
                act_data_list = {}
                count = 0
                for i in self.assim_index[1]:
                    for el in self.list_datatypes:
                        if self.real_obs_data[int(i)][el] is not None:
                            act_data_list[(el, float(self.keys_da['truedataindex'][int(i)]))] = count
                            count += 1

                well = [w for w in
                        set([el[0] for el in self.localization.loc_info.keys() if type(el) == tuple])]
                times = [t for t in set([el[1] for el in self.localization.loc_info.keys() if type(el) == tuple])]
                tot_dat_index = {}
                for uniq_well in well:
                    tmp_index = []
                    for t in times:
                        if (uniq_well, t) in act_data_list:
                            tmp_index.append(act_data_list[(uniq_well, t)])
                    tot_dat_index[uniq_well] = tmp_index
                if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                    emp_cov = True
                else:
                    emp_cov = False

                self.step = at.parallel_upd(self.list_states, self.prior_info, self.current_state, X,
                                            self.localization.loc_info, self.real_obs_data, self.aug_pred_data,
                                            int(self.keys_fwd['parallel']),
                                            actnum=self.localization.loc_info['actnum'],
                                            field_dim=self.localization.loc_info['field'],
                                            act_data_list=tot_dat_index,
                                            scale_data=self.scale_data,
                                            num_states=len([el for el in self.list_states]),
                                            emp_d_cov=emp_cov)
                self.step = at.aug_state(self.step, self.list_states)

        else:
            # Mean state and perturbation matrix
            mean_state = np.mean(aug_state, 1)
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                pert_state = (self.state_scaling**(-1))[:,None] * (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                                             np.ones((1, self.ne))))
            else:
                pert_state = (self.state_scaling**(-1))[:,None] * np.dot(aug_state,self.proj)
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                if len(self.scale_data.shape) == 1:
                    E_hat = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) * self.E
                    x_0 = np.dot(np.diag(s_d[:] ** (-1)), np.dot(u_d[:, :].T, E_hat))
                    Lam, z = np.linalg.eig(np.dot(x_0, x_0.T))
                    x_1 = np.dot(np.dot(u_d[:, :], np.dot(np.diag(s_d[:] ** (-1)).T, z)).T,
                                 np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) *
                                 (self.real_obs_data - self.aug_pred_data))
                else:
                    E_hat = solve(self.scale_data, self.E)
                    x_0 = np.dot(np.diag(s_d[:] ** (-1)), np.dot(u_d[:, :].T, E_hat))
                    Lam, z = np.linalg.eig(np.dot(x_0, x_0.T))
                    x_1 = np.dot(np.dot(u_d[:, :], np.dot(np.diag(s_d[:] ** (-1)).T, z)).T,
                                 solve(self.scale_data, (self.real_obs_data - self.aug_pred_data)))

                x_2 = solve((self.lam + 1) * np.diag(Lam) + np.eye(len(Lam)), x_1)
                x_3 = np.dot(np.dot(v_d.T, z), x_2)
                delta_1 = np.dot(self.state_scaling[:,None] * pert_state, x_3)
                self.step = delta_1
            else:
                # Compute the approximate update (follow notation in paper)
                if len(self.scale_data.shape) == 1:
                    x_1 = np.dot(u_d.T, np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) *
                                 (self.real_obs_data - self.aug_pred_data))
                else:
                    x_1 = np.dot(u_d.T, solve(self.scale_data, (self.real_obs_data - self.aug_pred_data)))
                x_2 = solve(((self.lam + 1) * np.eye(len(s_d)) + np.diag(s_d ** 2)), x_1)
                x_3 = np.dot(np.dot(v_d.T, np.diag(s_d)), x_2)
                self.step = np.dot(self.state_scaling[:,None] * pert_state, x_3)