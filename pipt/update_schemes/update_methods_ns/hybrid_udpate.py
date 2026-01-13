"""
ES, and Iterative ES updates with hybrid update matrix calculated from multi-fidelity runs.
"""

import numpy as np
from scipy.linalg import solve
from pipt.misc_tools import analysis_tools as at

class hybrid_update:
    '''
    Class for hybrid update schemes as described in: Fossum, K., Mannseth, T., & Stordal, A. S. (2020). Assessment of
    multilevel ensemble-based data assimilation for reservoir history matching. Computational Geosciences, 24(1),
    217–239. https://doi.org/10.1007/s10596-019-09911-x

    Note that the scheme is slightly modified to be inline with the standard (I)ES approximate update scheme. This
    enables the scheme to efficiently be coupled with multiple updating strategies via class MixIn
    '''

    def update(self):
        x_3 = []
        pert_state = []
        for l in range(self.tot_level):
            aug_state = at.aug_state(self.current_state[l], self.list_states, self.cell_index)
            mean_state = np.mean(aug_state, 1)
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                pert_state.append((self.state_scaling**(-1))[:, None] * (aug_state - np.dot(np.resize(mean_state, (len(mean_state), 1)),
                                                                                       np.ones((1, self.ml_ne[l])))))
            else:
                pert_state.append((self.state_scaling**(-1)
                              )[:, None] * np.dot(aug_state, self.proj[l]))

            u_d, s_d, v_d = np.linalg.svd(self.pert_preddata[l], full_matrices=False)
            if self.trunc_energy < 1:
                ti = (np.cumsum(s_d) / sum(s_d)) <= self.trunc_energy
                u_d, s_d, v_d = u_d[:, ti].copy(), s_d[ti].copy(), v_d[ti, :].copy()

            # x_1 = np.dot(u_d.T, solve(self.scale_data[l],
            #                  (self.real_obs_data[l] - self.aug_pred_data[l])))

            x_2 = solve(((self.lam + 1) * np.eye(len(s_d)) + np.diag(s_d ** 2)), u_d.T)
            x_3.append(np.dot(np.dot(v_d.T, np.diag(s_d)), x_2))

        # Calculate each row of self.step individually to avoid memory issues.
        self.step = [np.empty(pert_state[l].shape) for l in range(self.tot_level)]

        # do maximum 1000 rows at a time.
        step_size = min(1000, int(self.state_scaling.shape[0]/2))
        row_step = [np.arange(start, start+step_size) for start in
         np.arange(0, self.state_scaling.shape[0]-step_size, step_size)]
        #add the last rows
        row_step.append(np.arange(row_step[-1][-1]+1, self.state_scaling.shape[0]))

        for row in row_step:
            kg = sum([self.cov_wgt[indx_l]*np.dot(pert_state[indx_l][row, :], x_3[indx_l]) for indx_l in
                          range(self.tot_level)])
            for l in range(self.tot_level):
                if len(self.scale_data[l].shape) == 1:
                    self.step[l][row, :] = np.dot(self.state_scaling[row, None] * kg,
                                                  np.dot(np.expand_dims(self.scale_data[l] ** (-1), axis=1),
                                                                      np.ones((1, self.ml_ne[l]))) *
                                 (self.real_obs_data[l] - self.aug_pred_data[l]))
                else:
                    self.step[l][row, :] = np.dot(self.state_scaling[row, None] * kg, solve(self.scale_data[l],
                                                                                            (self.real_obs_data[l] -
                                                                                             self.aug_pred_data[l])))