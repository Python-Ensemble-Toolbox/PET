"""Local-analysis update for assimilation ensembles.

This is analysis mathematics rather than ensemble state, and sits here only
because it needs the ensemble's data and localization objects. It is mixed into
:class:`pipt.ensembles.AssimilationEnsemble` so the schemes can keep calling
``self.local_analysis_update()``.

Longer term this belongs with the analysis strategies in
:mod:`pipt.update_schemes.analysis`; keeping it as its own mixin is the first
step of that separation.
"""

import numpy as np
from copy import deepcopy
from scipy.linalg import solve

import pipt.misc_tools.analysis_tools as at
from pipt.localization import _calc_distance

__all__ = ["LocalAnalysisMixin"]


class LocalAnalysisMixin:
    """Localized (per-parameter-neighbourhood) analysis update."""

    def local_analysis_update(self):
        '''
        Function for updates that can be used by all algorithms. Do this once to avoid duplicate code for local
        analysis.
        '''
        # Copy original info to restore after local updates
        orig_list_data = deepcopy(self.list_datatypes)
        orig_list_state = deepcopy(self.list_states)
        orig_cd = deepcopy(self.cov_data)
        orig_real_obs_data = deepcopy(self.real_obs_data)
        orig_data_vector = deepcopy(self.obs_data_vector)

        # loop over the states that we want to update. Assume that the state and data combinations have been
        # determined by the initialization.
        # TODO: augment parameters with identical mask.

        # REGION PARAMETERS
        ############################################################################################################
        for state in self.local_analysis['region_parameter']:
            self.list_datatypes = [
                elem for elem in self.list_datatypes if
                elem in self.local_analysis['update_mask'][state]
            ]
            self.list_states = [deepcopy(state)]

            self._ext_scaling()  # scaling for this state
            if 'localization' in self.keys_da:
                self.localization.loc_info['field'] = self.state_scaling.shape
            del self.cov_data

            # reset the random state for consistency
            np.random.set_state(self.data_random_state)
            self.vecObs, self.enObs = self.set_observations()
            _, self.enPred = at.aug_obs_pred_data(
                self.obs_data,
                self.pred_data,
                self.assim_index,
                self.list_datatypes
            )

            # Get state ensemble for list_states
            enX = []
            idX = {}
            for idx in self.list_states:
                start, end = self.idX[idx]
                tempX = self.enX[start:end, :]
                enX.append(tempX)
                idX[idx] = (enX.shape[0] - tempX.shape[0], enX.shape[0])

            # Compute the analysis update
            self.update(
                enX = np.vstack(enX),
                enY = self.enPred,
                enE = self.enObs,
            )

            # Update the state
            if hasattr(self, 'step'):
                self.enX_temp = self.enX + self.step
        ############################################################################################################

        # VECTOR REGION PARAMETERS
        ############################################################################################################
        for state in self.local_analysis['vector_region_parameter']:
            current_list_datatypes = deepcopy(self.list_datatypes)
            for state_indx in range(self.state[state].shape[0]): # loop over the elements in the region
                self.list_datatypes = [elem for elem in self.list_datatypes if
                                       elem in self.local_analysis['update_mask'][state][state_indx]]
                if len(self.list_datatypes):
                    self.list_states = [deepcopy(state)]
                    self._ext_scaling()  # scaling for this state
                    if 'localization' in self.keys_da:
                        self.localization.loc_info['field'] = self.state_scaling.shape
                    del self.cov_data
                    # reset the random state for consistency
                    np.random.set_state(self.data_random_state)
                    self._ext_obs()  # get the data that's in the list of data.
                    _, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                                 self.list_datatypes)
                    # Mean pred_data and perturbation matrix with scaling
                    if len(self.scale_data.shape) == 1:
                        self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                                    np.ones((1, self.ne))) * np.dot(self.aug_pred_data, self.proj)
                    else:
                        self.pert_preddata = solve(
                            self.scale_data, np.dot(self.aug_pred_data, self.proj))

                    aug_state = at.aug_state(self.current_state, self.list_states)[state_indx,:]
                    self.update()
                    if hasattr(self, 'step'):
                        aug_state_upd = aug_state + self.step[state_indx,:]
                    self.state[state][state_indx,:] = aug_state_upd

                self.list_datatypes = deepcopy(current_list_datatypes)
        ############################################################################################################


        for state in self.local_analysis['cell_parameter']:
            self.list_states = [deepcopy(state)]
            self._ext_scaling()  # scaling for this state
            orig_state_scaling = deepcopy(self.state_scaling)
            param_position = self.local_analysis['parameter_position'][state]
            field_size = param_position.shape
            for k in range(field_size[0]):
                for j in range(field_size[1]):
                    for i in range(field_size[2]):
                        current_data_list = list(
                            self.local_analysis['update_mask'][state][k][j][i])
                        current_data_list.sort()  # ensure consistent ordering of data
                        if len(current_data_list):
                            # if non-unique data for assimilation index, get the relevant data.
                            if self.local_analysis['unique'] is False:
                                orig_assim_index = deepcopy(self.assim_index)
                                assim_index_data_list = set(
                                    [el.split('_')[0] for el in current_data_list])
                                current_assim_index = [
                                    int(el.split('_')[1]) for el in current_data_list]
                                current_data_list = list(assim_index_data_list)
                                self.assim_index[1] = current_assim_index
                            self.list_datatypes = deepcopy(current_data_list)
                            del self.cov_data
                            # reset the random state for consistency
                            np.random.set_state(self.data_random_state)
                            self._ext_obs()
                            _, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data,
                                                                         self.assim_index,
                                                                         self.list_datatypes)
                            # get parameter indexes
                            full_cell_index = np.ravel_multi_index(
                                np.array([[k], [j], [i]]), tuple(field_size))
                            # count active values
                            self.cell_index = [sum(param_position.flatten()[:el])
                                               for el in full_cell_index]
                            if 'localization' in self.keys_da:
                                self.localization.loc_info['field'] = (
                                    len(self.cell_index),)
                                self.localization.loc_info['distance'] = _calc_distance(
                                    self.local_analysis['data_position'],
                                    self.local_analysis['unique'],
                                    current_data_list, self.assim_index,
                                    self.obs_data, self.pred_data, [(k, j, i)])
                            # Set relevant state scaling
                            self.state_scaling = orig_state_scaling[self.cell_index]

                            # Mean pred_data and perturbation matrix with scaling
                            if len(self.scale_data.shape) == 1:
                                self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                                            np.ones((1, self.ne))) * np.dot(self.aug_pred_data,
                                                                                            self.proj)
                            else:
                                self.pert_preddata = solve(
                                    self.scale_data, np.dot(self.aug_pred_data, self.proj))

                            aug_state = at.aug_state(
                                self.current_state, self.list_states, self.cell_index)
                            self.update()
                            if hasattr(self, 'step'):
                                aug_state_upd = aug_state + self.step
                            self.state = at.update_state(
                                aug_state_upd, self.state, self.list_states, self.cell_index)

                            if self.local_analysis['unique'] is False:
                                # reset assim index
                                self.assim_index = deepcopy(orig_assim_index)
                            if hasattr(self, 'localization') and 'distance' in self.localization.loc_info:  # reset
                                del self.localization.loc_info['distance']

        self.list_datatypes = deepcopy(orig_list_data)  # reset to original list
        self.list_states = deepcopy(orig_list_state)
        self.cov_data = deepcopy(orig_cd)
        self.real_obs_data = deepcopy(orig_real_obs_data)
        self.obs_data_vector = deepcopy(orig_data_vector)
        self.cell_index = None
