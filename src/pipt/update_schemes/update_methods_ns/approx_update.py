"""EnRML (IES) without the prior increment term."""

import numpy as np
from copy import deepcopy
import copy as cp
from scipy.linalg import solve, solve_banded, cholesky, lu_solve, lu_factor, inv
import pickle

import pipt.misc_tools.ensemble_tools as entools
import pipt.misc_tools.analysis_tools as at

from pipt.misc_tools.cov_regularization import _calc_loc


class approx_update():
    """
    Approximate LM Update scheme as defined in "Chen, Y., & Oliver, D. S. (2013). Levenberg–Marquardt forms of the iterative ensemble
    smoother for efficient history matching and uncertainty quantification. Computational Geosciences, 17(4), 689–703.
    https://doi.org/10.1007/s10596-013-9351-5". Note that for a EnKF or ES update, or for update within GN scheme, lambda = 0.
    """

    def update(self, enX, enY, enE, **kwargs):
        ''' 
        Perform the approximate LM update.

        Parameters:
        ----------
            enX : np.ndarray 
                State ensemble matrix (nx, ne)
            
            enY : np.ndarray
                Predicted data ensemble matrix (nd, ne)
            
            enE : np.ndarray
                Ensemble of perturbed observations (nd, ne)
        '''

        # Scale and center the ensemble matrecies
        enYcentered = self.scale(np.dot(enY, self.proj), self.scale_data)

        # Perform truncated SVD
        Ud, Sd, VTd = at.truncSVD(enYcentered, energy=self.trunc_energy)

        # Check for localization methods
        if 'localization' in self.keys_da:
            loc_info = self.localization.loc_info

            # Calculate the localization projection matrix
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                # Scale and center the data ensemble matrix
                enEcentered = self.scale(np.dot(enE, self.proj), self.scale_data)

                # Calculate intermediate matrix
                Sinv = np.diag(1/Sd)
                X0 = Sinv @ Ud.T @ enEcentered

                # Eigen decomposition of X0 X0^T
                eigval, eigvec = np.linalg.eig(X0 @ X0.T)
                reg_term = (self.lam + 1) * np.diag(eigval) + np.eye(len(eigval))
                X = (VTd.T @ eigvec) @ solve(reg_term, (Ud.T @ (Sinv @ eigvec)).T)
                
            else:
                reg_term = (self.lam + 1)*np.eye(Sd.size) + np.diag(Sd**2)
                X = VTd.T @ np.diag(Sd) @ solve(reg_term, Ud.T)

            
            # Check for adaptive localization
            if 'autoadaloc' in loc_info:

                # Scale and center the state ensemble matrix, enX
                if ('emp_cov' in self.keys_da) and (self.keys_da['emp_cov'] == 'yes'):
                    enXcentered = self.scale(enX - np.mean(enX, 1)[:,None], self.state_scaling)
                else:
                    enXcentered = self.scale(np.dot(enX, self.proj), self.state_scaling)

                # Calculate and scale difference between observations and predictions (residuals)
                enRes = self.scale(enE - enY, self.scale_data)

                # Compute the update step with auto-adaptive localization
                self.step = self.localization.auto_ada_loc(
                    pert_state     = self.state_scaling[:, None]*enXcentered, 
                    proj_pred_data = np.dot(X, enRes),
                    curr_param     = self.list_states,
                    prior_info     = self.prior_info
                )


            # Check for local analysis 
            elif ('localanalysis' in loc_info) and (loc_info['localanalysis']):
                
                # Calculate weights
                if 'distance' in loc_info:
                    weight = _calc_loc(
                        max_dist   = loc_info['range'], 
                        distance   = loc_info['distance'],
                        prior_info = self.prior_info[self.list_states[0]], 
                        loc_type   = loc_info['type'], 
                        ne = self.ne
                    )
                else: # if no distance, do full update
                    weight = np.ones((enX.shape[0], X.shape[1]))

                # Center ensemble matrix
                enXcentered = enX - np.mean(enX, axis=1, keepdims=True)

                if (not ('emp_cov' in self.keys_da) and (self.keys_da['emp_cov'] == 'yes')):
                    enXcentered /= np.sqrt(self.ne - 1)

                # Calculate and scale difference between observations and predictions (residuals)
                enRes = self.scale(enE - enY, self.scale_data)

                # Compute the update step with local analysis
                try:
                    self.step = weight.multiply(np.dot(enXcentered, X)).dot(enRes)
                except:
                    self.step = (weight*(np.dot(enXcentered, X))).dot(enRes)


            # Check for distance based localization
            elif ('dist_loc' in self.keys_da['localization'].keys()) or ('dist_loc' in self.keys_da['localization'].values()):

                # Setup localization mask
                mask = self.localization.localize(
                    self.list_datatypes, 
                    [self.keys_da['truedataindex'][int(elem)] for elem in self.assim_index[1]],
                    self.list_states, 
                    self.ne, 
                    self.prior_info, 
                    at.get_obs_size(self.obs_data, self.assim_index[1], self.list_datatypes)
                )

                # Center ensemble matrix
                enXcentered = enX - np.mean(enX, axis=1, keepdims=True)

                if not ('emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes'):
                    enXcentered /= np.sqrt(self.ne - 1)

                # Calculate and scale difference between observations and predictions (residuals)
                enRes = self.scale(enE - enY, self.scale_data)

                # Compute the update step with distance-based localization
                self.step = mask.multiply(np.dot(enXcentered, X)).dot(enRes)



            # Else do parallel update (NOT TESTED AFTER UPDATES)
            else:
                act_data_list = {}
                count = 0
                for i in self.assim_index[1]:
                    for el in list(self.idX.keys()):
                        if self.real_obs_data[int(i)][el] is not None:
                            act_data_list[(el, float(self.keys_da['truedataindex'][int(i)]))] = count
                            count += 1

                well  = [w for w in set([el[0] for el in loc_info.keys() if type(el) == tuple])]
                times = [t for t in set([el[1] for el in loc_info.keys() if type(el) == tuple])]

                tot_dat_index = {}
                for uniq_well in well:
                    tmp_index = []
                    for t in times:
                        if (uniq_well, t) in act_data_list:
                            tmp_index.append(act_data_list[(uniq_well, t)])
                    tot_dat_index[uniq_well] = tmp_index

                if ('emp_cov' in self.keys_da) and (self.keys_da['emp_cov'] == 'yes'):
                    emp_cov = True
                else:
                    emp_cov = False

                self.step = at.parallel_upd(
                    list(self.idX.keys()), 
                    self.prior_info, 
                    entools.matrix_to_dict(enX, self.idX),
                    X,
                    loc_info, 
                    enE, 
                    enY,
                    int(self.keys_fwd['parallel']),
                    actnum=loc_info['actnum'],
                    field_dim=loc_info['field'],
                    act_data_list=tot_dat_index,
                    scale_data=self.scale_data,
                    num_states=len([el for el in list(self.idX.keys())]),
                    emp_d_cov=emp_cov
                )
                self.step = at.aug_state(self.step, list(self.idX.keys()))

        else:

            # if ('emp_cov' in self.keys_da) and (self.keys_da['emp_cov'] == 'yes'):
                
            #     # Scale and center the ensemble matrecies: enX and enE
            #     enXcentered = self.scale(enX - np.mean(enX, 1)[:,None], self.state_scaling)
            #     enEcentered = self.scale(enE - np.mean(enE, 1)[:,None], self.scale_data)

            #     Sinv = np.diag(1/Sd)
            #     X0 = Sinv @ Ud.T @ enEcentered
            #     eigval, eigvec = np.linalg.eig(X0 @ X0.T)

            #     # Calculate and scale difference between observations and predictions (residuals)
            #     enRes = self.scale(enE - enY, self.scale_data)

            #     # Compute the update step
            #     X1 = (Ud @ Sinv @ eigvec).T @ enRes
            #     X2 = solve((self.lam + 1) * np.diag(eigval) + np.eye(len(eigval)), X1)
            #     X3 = np.dot(VTd.T, eigvec) @ X2
            #     self.step = np.dot(self.state_scaling[:, None]*enXcentered, X3)
 
            # else:
            enXcentered = self.scale(np.dot(enX, self.proj), self.state_scaling)
            enRes = self.scale(enE - enY, self.scale_data)
            
            # Compute the update step
            X1 = Ud.T @ enRes
            X2 = solve((self.lam + 1)*np.eye(Sd.size) + np.diag(Sd**2), X1)
            X3 = VTd.T @ np.diag(Sd) @ X2
            self.step = np.dot(self.state_scaling[:, None] * enXcentered, X3)


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
