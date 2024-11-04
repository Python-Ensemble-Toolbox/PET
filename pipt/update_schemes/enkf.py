"""
EnKF type schemes
"""
# External imports
import numpy as np
from scipy.linalg import solve
from copy import deepcopy
from geostat.decomp import Cholesky                     # Making realizations

# Internal imports
from pipt.loop.ensemble import Ensemble
# Misc. tools used in analysis schemes
from pipt.misc_tools import analysis_tools as at

from pipt.update_schemes.update_methods_ns.approx_update import approx_update
from pipt.update_schemes.update_methods_ns.full_update import full_update
from pipt.update_schemes.update_methods_ns.subspace_update import subspace_update


class enkfMixIn(Ensemble):
    """
    Straightforward EnKF analysis scheme implementation. The sequential updating can be done with general grouping and
    ordering of data. If only one-step EnKF is to be done, use `es` instead.
    """

    def __init__(self, keys_da, keys_fwd, sim):
        """
        The class is initialized by passing the PIPT init. file upwards in the hierarchy to be read and parsed in
        `pipt.input_output.pipt_init.ReadInitFile`.
        """
        # Pass the init_file upwards in the hierarchy
        super().__init__(keys_da, keys_fwd, sim)

        self.prev_data_misfit = None

        if self.restart is False:
            self.prior_state = deepcopy(self.state)
            self.list_states = list(self.state.keys())
            # At the moment, the iterative loop is threated as an iterative smoother an thus we check if assim. indices
            # are given as in the Simultaneous loop.
            self.check_assimindex_sequential()

            # Extract no. assimilation steps from MDA keyword in DATAASSIM part of init. file and set this equal to
            # the number of iterations pluss one. Need one additional because the iter=0 is the prior run.
            self.max_iter = len(self.keys_da['assimindex'])+1
            self.iteration = 0
            self.lam = 0  # set LM lamda to zero as we are doing one full update.
            if 'energy' in self.keys_da:
                # initial energy (Remember to extract this)
                self.trunc_energy = self.keys_da['energy']
                if self.trunc_energy > 1:  # ensure that it is given as percentage
                    self.trunc_energy /= 100.
            else:
                self.trunc_energy = 0.98
            self.current_state = deepcopy(self.state)

            self.state_scaling = at.calc_scaling(
                self.prior_state, self.list_states, self.prior_info)

    def calc_analysis(self):
        """
        Calculate the analysis step of the EnKF procedure. The updating is done using the Kalman filter equations, using
        svd for numerical stability. Localization is available.
        """
        # If this is initial analysis we calculate the objective function for all data. In the final convergence check
        # we calculate the posterior objective function for all data
        if not hasattr(self, 'prior_data_misfit'):
            assim_index = [self.keys_da['obsname'], list(
                np.concatenate(self.keys_da['assimindex']))]
            list_datatypes, list_active_dataypes = at.get_list_data_types(
                self.obs_data, assim_index)
            if not hasattr(self, 'cov_data'):
                self.full_cov_data = at.gen_covdata(
                    self.datavar, assim_index, list_datatypes)
            else:
                self.full_cov_data = self.cov_data
            obs_data_vector, pred_data = at.aug_obs_pred_data(
                self.obs_data, self.pred_data, assim_index, list_datatypes)
            # Generate realizations of the observed data
            init_en = Cholesky()  # Initialize GeoStat class for generating realizations
            self.full_real_obs_data = init_en.gen_real(
                obs_data_vector, self.full_cov_data, self.ne)

            # Calc. misfit for the initial iteration
            data_misfit = at.calc_objectivefun(
                self.full_real_obs_data, pred_data, self.full_cov_data)

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = np.mean(data_misfit)
            self.prior_data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            self.logger.info(
                f'Prior run complete with data misfit: {self.prior_data_misfit:0.1f}.')

        # Get assimilation order as a list
        # must subtract one to be inline
        self.assim_index = [self.keys_da['obsname'],
                            self.keys_da['assimindex'][self.iteration-1]]

        # Get list of data types to be assimilated and of the free states. Do this once, because listing keys from a
        # Python dictionary just when needed (in different places) may not yield the same list!
        self.list_datatypes, list_active_dataypes = at.get_list_data_types(
            self.obs_data, self.assim_index)

        # Augment observed and predicted data
        self.obs_data_vector, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                                        self.list_datatypes)
        self.cov_data = at.gen_covdata(
            self.datavar, self.assim_index, self.list_datatypes)

        init_en = Cholesky()  # Initialize GeoStat class for generating realizations
        self.data_random_state = deepcopy(np.random.get_state())
        self.real_obs_data, self.scale_data = init_en.gen_real(self.obs_data_vector, self.cov_data, self.ne,
                                                               return_chol=True)

        self.E = np.dot(self.real_obs_data, self.proj)

        if 'localanalysis' in self.keys_da:
            self.local_analysis_update()
        else:
            # Mean pred_data and perturbation matrix with scaling
            if len(self.scale_data.shape) == 1:
                self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                            np.ones((1, self.ne))) * np.dot(self.aug_pred_data, self.proj)
            else:
                self.pert_preddata = solve(
                    self.scale_data, np.dot(self.aug_pred_data, self.proj))

            aug_state = at.aug_state(self.current_state, self.list_states)
            self.update()
            if hasattr(self, 'step'):
                aug_state_upd = aug_state + self.step
            if hasattr(self, 'w_step'):
                self.W = self.current_W + self.w_step
                aug_prior_state = at.aug_state(self.prior_state, self.list_states)
                aug_state_upd = np.dot(aug_prior_state, (np.eye(
                    self.ne) + self.W / np.sqrt(self.ne - 1)))
            # Extract updated state variables from aug_update
            self.state = at.update_state(aug_state_upd, self.state, self.list_states)
            self.state = at.limits(self.state, self.prior_info)

    def check_convergence(self):
        """
        Calculate the "convergence" of the method. Important to
        """
        self.prev_data_misfit = self.prior_data_misfit
        # only calulate for the final (posterior) estimate
        if self.iteration == len(self.keys_da['assimindex']):
            assim_index = [self.keys_da['obsname'], list(
                np.concatenate(self.keys_da['assimindex']))]
            list_datatypes = self.list_datatypes
            obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, assim_index,
                                                              list_datatypes)

            data_misfit = at.calc_objectivefun(
                self.full_real_obs_data, pred_data, self.full_cov_data)
            self.data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

        else:  # sequential updates not finished. Misfit is not relevant
            self.data_misfit = self.prior_data_misfit

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit / self.prev_data_misfit),
                    'data_misfit': self.data_misfit,
                    'prev_data_misfit': self.prev_data_misfit}

        self.current_state = deepcopy(self.state)
        if self.data_misfit == self.prev_data_misfit:
            self.logger.info(
                f'EnKF update {self.iteration} complete!')
        else:
            if self.data_misfit < self.prior_data_misfit:
                self.logger.info(
                    f'EnKF update complete! Objective function decreased from {self.prior_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
            else:
                self.logger.info(
                    f'EnKF update complete! Objective function increased from {self.prior_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        # Return conv = False, why_stop var.
        return False, True, why_stop


class enkf_approx(enkfMixIn, approx_update):
    """
    MixIn the main EnKF update class with the standard analysis scheme.
    """
    pass


class enkf_full(enkfMixIn, approx_update):
    """
    MixIn the main EnKF update class with the standard analysis scheme. Note that this class is only included for
    completness. The EnKF does not iterate, and the standard scheme is therefor always applied.
    """
    pass


class enkf_subspace(enkfMixIn, subspace_update):
    """
    MixIn the main EnKF update class with the subspace analysis scheme.
    """
    pass
