"""
EnRML type schemes
"""
# External imports
import pipt.misc_tools.analysis_tools as at
from geostat.decomp import Cholesky
from pipt.loop.ensemble import Ensemble
from pipt.update_schemes.update_methods_ns.subspace_update import subspace_update
from pipt.update_schemes.update_methods_ns.full_update import full_update
from pipt.update_schemes.update_methods_ns.approx_update import approx_update
import sys
import pkgutil
import inspect
import numpy as np
import copy as cp
from scipy.linalg import cholesky, solve

import importlib.util

# List all available packages in the namespace package
# Import those that are present
import pipt.update_schemes.update_methods_ns as ns_pkg
tot_ns_pkg = []
# extract all class methods from namespace
for finder, name, ispkg in pkgutil.walk_packages(ns_pkg.__path__):
    spec = finder.find_spec(name)
    _module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_module)
    tot_ns_pkg.extend(inspect.getmembers(_module, inspect.isclass))

# import standard libraries

# Check and import (if present) from other namespace packages
if 'margIS_update' in [el[0] for el in tot_ns_pkg]:  # only compare package name
    from pipt.update_schemes.update_methods_ns.margIS_update import margIS_update
else:
    class margIS_update:
        pass

# Internal imports


class lmenrmlMixIn(Ensemble):
    """
    This is an implementation of EnRML using Levenberg-Marquardt. The update scheme is selected by a MixIn with multiple
    update_methods_ns. This class must therefore facititate many different update schemes.
    """

    def __init__(self, keys_da, keys_fwd, sim):
        """
        The class is initialized by passing the PIPT init. file upwards in the hierarchy to be read and parsed in
        `pipt.input_output.pipt_init.ReadInitFile`.
        """
        # Pass the init_file upwards in the hierarchy
        super().__init__(keys_da, keys_fwd, sim)

        if self.restart is False:
            # Save prior state in separate variable
            self.prior_state = cp.deepcopy(self.state)

            # Extract parameters like conv. tol. and damping param. from ITERATION keyword in DATAASSIM
            self._ext_iter_param()

            # Within variables
            self.prev_data_misfit = None  # Data misfit at previous iteration
            if 'actnum' in self.keys_da.keys():
                try:
                    self.actnum = np.load(self.keys_da['actnum'])['actnum']
                except:
                    print('ACTNUM file cannot be loaded!')
            else:
                self.actnum = None
            # At the moment, the iterative loop is threated as an iterative smoother and thus we check if assim. indices
            # are given as in the Simultaneous loop.
            self.check_assimindex_simultaneous()
            # define the assimilation index
            self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            # define the list of states
            self.list_states = list(self.state.keys())
            # define the list of datatypes
            self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(
                self.obs_data, self.assim_index)
            # Get the perturbed observations and observation scaling
            self.data_random_state = cp.deepcopy(np.random.get_state())
            self._ext_obs()
            # Get state scaling and svd of scaled prior
            self._ext_state()
            self.current_state = cp.deepcopy(self.state)

    def calc_analysis(self):
        """
        Calculate the update step in LM-EnRML, which is just the Levenberg-Marquardt update algorithm with
        the sensitivity matrix approximated by the ensemble.
        """

        # reformat predicted data
        _, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                     self.list_datatypes)

        if self.iteration == 1:  # first iteration
            data_misfit = at.calc_objectivefun(
                self.real_obs_data, self.aug_pred_data, self.cov_data)

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = np.mean(data_misfit)
            self.prior_data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            if self.lam == 'auto':
                self.lam = (0.5 * self.prior_data_misfit)/self.aug_pred_data.shape[0]

            self.logger.info(
                f'Prior run complete with data misfit: {self.prior_data_misfit:0.1f}. Lambda for initial analysis: {self.lam}')

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
            self.update()  # run ordinary analysis
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
        Check if LM-EnRML have converged based on evaluation of change sizes of objective function, state and damping
        parameter. 

        Returns
        -------
        conv: bool
            Logic variable telling if algorithm has converged
        why_stop: dict
            Dict. with keys corresponding to conv. criteria, with logical variable telling which of them that has been
            met
        """

        _, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                            self.list_datatypes)
        # Initialize the initial success value
        success = False

        # if inital conv. check, there are no prev_data_misfit
        self.prev_data_misfit = self.data_misfit
        self.prev_data_misfit_std = self.data_misfit_std

        # Calc. std dev of data misfit (used to update lamda)
        # mat_obs = np.dot(obs_data_vector.reshape((len(obs_data_vector),1)), np.ones((1, self.ne))) # use the perturbed
        # data instead.

        data_misfit = at.calc_objectivefun(self.real_obs_data, pred_data, self.cov_data)

        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # # Calc. mean data misfit for convergence check, using the updated state variable
        # self.data_misfit = np.dot((mean_preddata - obs_data_vector).T,
        #                      solve(cov_data, (mean_preddata - obs_data_vector)))

        # Convergence check: Relative step size of data misfit or state change less than tolerance
        if abs(1 - (self.data_misfit / self.prev_data_misfit)) < self.data_misfit_tol \
                or self.lam >= self.lam_max:
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit / self.prev_data_misfit) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit,
                        'prev_data_misfit': self.prev_data_misfit,
                        'lambda': self.lam,
                        'lambda_stop': self.lam >= self.lam_max}

            if self.data_misfit >= self.prev_data_misfit:
                success = False
                self.logger.info(
                    f'Iterations have converged after {self.iteration} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit:0.1f} to {self.prev_data_misfit:0.1f}')
            else:
                self.logger.info(
                    f'Iterations have converged after {self.iteration} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit:0.1f} to {self.data_misfit:0.1f}')
            # Return conv = True, why_stop var.
            return True, success, why_stop

        else:  # conv. not met
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit / self.prev_data_misfit) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit,
                        'prev_data_misfit': self.prev_data_misfit,
                        'lambda': self.lam,
                        'lambda_stop': self.lam >= self.lam_max}

            ###############################################
            ##### update Lambda step-size values ##########
            ###############################################
            # If reduction in mean data misfit, reduce damping param
            if self.data_misfit < self.prev_data_misfit and self.data_misfit_std < self.prev_data_misfit_std:
                # Reduce damping parameter (divide calculations for ANALYSISDEBUG purpose)
                if self.lam > self.lam_min:
                    self.lam = self.lam / self.gamma
                success = True
                self.current_state = cp.deepcopy(self.state)
                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)
            elif self.data_misfit < self.prev_data_misfit and self.data_misfit_std >= self.prev_data_misfit_std:
                # accept itaration, but keep lam the same
                success = True
                self.current_state = cp.deepcopy(self.state)
                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)

            else:  # Reject iteration, and increase lam
                # Increase damping parameter (divide calculations for ANALYSISDEBUG purpose)
                self.lam = self.lam * self.gamma
                success = False

            if success:
                self.logger.info(f'Successfull iteration number {self.iteration}! Objective function reduced from '
                                 f'{self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}. New Lamba for next analysis: '
                                 f'{self.lam}')
                # self.prev_data_misfit = self.data_misfit
                # self.prev_data_misfit_std = self.data_misfit_std
            else:
                self.logger.info(f'Failed iteration number {self.iteration}! Objective function increased from '
                                 f'{self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}. New Lamba for repeated analysis: '
                                 f'{self.lam}')
                # Reset the objective function after report
                self.data_misfit = self.prev_data_misfit
                self.data_misfit_std = self.prev_data_misfit_std

            # Return conv = False, why_stop var.
            return False, success, why_stop

    def _ext_iter_param(self):
        """
        Extract parameters needed in LM-EnRML from the ITERATION keyword given in the DATAASSIM part of PIPT init.
        file. These parameters include convergence tolerances and parameters for the damping parameter. Default
        values for these parameters have been given here, if they are not provided in ITERATION.
        """

        # Predefine all the default values
        self.data_misfit_tol = 0.01
        self.step_tol = 0.01
        self.lam = 100
        self.lam_max = 1e10
        self.lam_min = 0.01
        self.gamma = 5
        self.trunc_energy = 0.95
        self.iteration = 0

        # Loop over options in ITERATION and extract the parameters we want
        for i, opt in enumerate(list(zip(*self.keys_da['iteration']))[0]):
            if opt == 'data_misfit_tol':
                self.data_misfit_tol = self.keys_da['iteration'][i][1]
            if opt == 'step_tol':
                self.step_tol = self.keys_da['iteration'][i][1]
            if opt == 'lambda':
                self.lam = self.keys_da['iteration'][i][1]
            if opt == 'lambda_max':
                self.lam_max = self.keys_da['iteration'][i][1]
            if opt == 'lambda_min':
                self.lam_min = self.keys_da['iteration'][i][1]
            if opt == 'lambda_factor':
                self.gamma = self.keys_da['iteration'][i][1]

        if 'energy' in self.keys_da:
            # initial energy (Remember to extract this)
            self.trunc_energy = self.keys_da['energy']
            if self.trunc_energy > 1:  # ensure that it is given as percentage
                self.trunc_energy /= 100.


class lmenrml_approx(lmenrmlMixIn, approx_update):
    pass


class lmenrml_full(lmenrmlMixIn, full_update):
    pass


class lmenrml_subspace(lmenrmlMixIn, subspace_update):
    pass


class gnenrmlMixIn(Ensemble):
    """
    This is an implementation of EnRML using the Gauss-Newton approach. The update scheme is selected by a MixIn with multiple
    update_methods_ns. This class must therefore facititate many different update schemes.
    """

    def __init__(self, keys_da, keys_fwd, sim):
        """
        The class is initialized by passing the PIPT init. file upwards in the hierarchy to be read and parsed in
        `pipt.input_output.pipt_init.ReadInitFile`.
        """
        # Pass the init_file upwards in the hierarchy
        super().__init__(keys_da, keys_fwd, sim)

        if self.restart is False:
            # Save prior state in separate variable
            self.prior_state = cp.deepcopy(self.state)

            # extract and save state scaling

            # Extract parameters like conv. tol. and damping param. from ITERATION keyword in DATAASSIM
            self._ext_iter_param()

            # Within variables
            self.prev_data_misfit = None  # Data misfit at previous iteration
            if 'actnum' in self.keys_da.keys():
                try:
                    self.actnum = np.load(self.keys_da['actnum'])['actnum']
                except:
                    print('ACTNUM file cannot be loaded!')
            else:
                self.actnum = None
            # At the moment, the iterative loop is threated as an iterative smoother and thus we check if assim. indices
            # are given as in the Simultaneous loop.
            self.check_assimindex_simultaneous()
            # define the assimilation index
            self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            # define the list of states
            self.list_states = list(self.state.keys())
            # define the list of datatypes
            self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(
                self.obs_data, self.assim_index)
            # Get the perturbed observations and observation scaling
            self._ext_obs()
            # Get state scaling and svd of scaled prior
            self._ext_state()
            self.current_state = cp.deepcopy(self.state)
            # ensure that the updates does not invoke the LM inflation of the Hessian.
            self.lam = 0

    def _ext_obs(self):

        self.obs_data_vector, _ = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                       self.list_datatypes)

        # Generate the data auto-covariance matrix
        if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
            if hasattr(self, 'cov_data'):  # cd matrix has been imported
                tmp_E = np.dot(cholesky(self.cov_data).T,
                               np.random.randn(self.cov_data.shape[0], self.ne))
            else:
                tmp_E = at.extract_tot_empirical_cov(
                    self.datavar, self.assim_index, self.list_datatypes, self.ne)
            # self.E = (tmp_E - tmp_E.mean(1)[:,np.newaxis])/np.sqrt(self.ne - 1)/
            if 'screendata' in self.keys_da and self.keys_da['screendata'] == 'yes':
                tmp_E = at.screen_data(tmp_E, self.aug_pred_data,
                                       self.obs_data_vector, self.iteration)
            self.E = tmp_E
            self.real_obs_data = self.obs_data_vector[:, np.newaxis] - tmp_E

            self.cov_data = np.var(self.E, ddof=1,
                                   axis=1)  # calculate the variance, to be used for e.g. data misfit calc
            # self.cov_data = ((self.E * self.E)/(self.ne-1)).sum(axis=1) # calculate the variance, to be used for e.g. data misfit calc
            self.scale_data = np.sqrt(self.cov_data)
        else:
            if not hasattr(self, 'cov_data'):  # if cd is not loaded
                self.cov_data = at.gen_covdata(
                    self.datavar, self.assim_index, self.list_datatypes)
            # data screening
            if 'screendata' in self.keys_da and self.keys_da['screendata'] == 'yes':
                self.cov_data = at.screen_data(
                    self.cov_data, self.aug_pred_data, self.obs_data_vector, self.iteration)

            init_en = Cholesky()  # Initialize GeoStat class for generating realizations
            self.real_obs_data, self.scale_data = init_en.gen_real(self.obs_data_vector, self.cov_data, self.ne,
                                                                   return_chol=True)

    def _ext_state(self):
        # get vector of scaling
        self.state_scaling = at.calc_scaling(
            self.prior_state, self.list_states, self.prior_info)

        delta_scaled_prior = self.state_scaling[:, None] * \
            np.dot(at.aug_state(self.prior_state, self.list_states), self.proj)

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
        self.Am = np.dot(u_d, np.eye(trunc_index+1) *
                         ((s_d**(-1))[:, None]))  # notation from paper

    def calc_analysis(self):
        """
        Calculate the update step in LM-EnRML, which is just the Levenberg-Marquardt update algorithm with
        the sensitivity matrix approximated by the ensemble.

        """

        # reformat predicted data
        _, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                     self.list_datatypes)

        if self.iteration == 1:  # first iteration
            data_misfit = at.calc_objectivefun(
                self.real_obs_data, self.aug_pred_data, self.cov_data)

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = np.mean(data_misfit)
            self.prior_data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            if self.gamma == 'auto':
                self.gamma = 0.1

        # Mean pred_data and perturbation matrix with scaling
        if len(self.scale_data.shape) == 1:
            self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                        np.ones((1, self.ne))) * np.dot(self.aug_pred_data, self.proj)
        else:
            self.pert_preddata = solve(
                self.scale_data, np.dot(self.aug_pred_data, self.proj))

        aug_state = at.aug_state(self.current_state, self.list_states)

        self.update()  # run analysis
        if hasattr(self, 'step'):
            aug_state_upd = aug_state + self.gamma*self.step
        if hasattr(self, 'w_step'):
            self.W = self.current_W + self.gamma*self.w_step
            aug_prior_state = at.aug_state(self.prior_state, self.list_states)
            aug_state_upd = np.dot(aug_prior_state, (np.eye(
                self.ne) + self.W / np.sqrt(self.ne - 1)))
        if hasattr(self, 'sqrt_w_step'):  # if we do a sqrt update
            self.w = self.current_w + self.gamma*self.sqrt_w_step
            new_mean_state = self.mean_prior + np.dot(self.X, self.w)
            u, sigma, v = np.linalg.svd(self.C_w, full_matrices=True)
            sigma_inv_sqrt = np.diag([el_s ** (-1 / 2) for el_s in sigma])
            C_w_inv_sqrt = np.dot(np.dot(u, sigma_inv_sqrt), v.T)
            self.W = C_w_inv_sqrt * np.sqrt(self.ne - 1)
            aug_state_upd = np.tile(new_mean_state, (self.ne, 1)
                                    ).T + np.dot(self.X, self.W)

        # Extract updated state variables from aug_update
        self.state = at.update_state(aug_state_upd, self.state, self.list_states)
        self.state = at.limits(self.state, self.prior_info)

    def check_convergence(self):
        """
        Check if LM-EnRML have converged based on evaluation of change sizes of objective function, state and damping
        parameter.

        Returns
        -------
        conv: bool
            Logic variable telling if algorithm has converged
        why_stop: dict
            Dict. with keys corresponding to conv. criteria, with logical variable telling which of them that has been
            met
        """
        # Prelude to calc. conv. check (everything done below is from calc_analysis)
        if hasattr(self, 'list_datatypes'):
            assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            list_datatypes = self.list_datatypes
            cov_data = self.cov_data
            obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, assim_index,
                                                              list_datatypes)
            mean_preddata = np.mean(pred_data, 1)
        else:
            assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            list_datatypes, _ = at.get_list_data_types(self.obs_data, assim_index)
            # cov_data = at.gen_covdata(self.datavar, assim_index, list_datatypes)
            obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, assim_index,
                                                              list_datatypes)
            # mean_preddata = np.mean(pred_data, 1)

        # Initialize the initial success value
        success = False

        # if inital conv. check, there are no prev_data_misfit
        if self.prev_data_misfit is None:
            self.data_misfit = np.mean(self.data_misfit)
            self.prev_data_misfit = self.data_misfit
            self.prev_data_misfit_std = self.data_misfit_std
            success = True

        # update the last mismatch, only if this was a reduction of the misfit
        if self.data_misfit < self.prev_data_misfit:
            self.prev_data_misfit = self.data_misfit
            self.prev_data_misfit_std = self.data_misfit_std
            success = True
        # if there was no reduction of the misfit, retain the old "valid" data misfit.

        # Calc. std dev of data misfit (used to update lamda)
        # mat_obs = np.dot(obs_data_vector.reshape((len(obs_data_vector),1)), np.ones((1, self.ne))) # use the perturbed
        # data instead.
        mat_obs = self.real_obs_data
        data_misfit = at.calc_objectivefun(mat_obs, pred_data, self.cov_data)

        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # # Calc. mean data misfit for convergence check, using the updated state variable
        # self.data_misfit = np.dot((mean_preddata - obs_data_vector).T,
        #                      solve(cov_data, (mean_preddata - obs_data_vector)))

        # Convergence check: Relative step size of data misfit or state change less than tolerance
        if abs(1 - (self.data_misfit / self.prev_data_misfit)) < self.data_misfit_tol:
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit / self.prev_data_misfit) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit,
                        'prev_data_misfit': self.prev_data_misfit,
                        'gamma': self.gamma,
                        }

            if self.data_misfit >= self.prev_data_misfit:
                success = False
                self.logger.info(
                    f'Iterations have converged after {self.iteration} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit:0.1f} to {self.prev_data_misfit:0.1f}')
            else:
                self.logger.info(
                    f'Iterations have converged after {self.iteration} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit:0.1f} to {self.data_misfit:0.1f}')
            # Return conv = True, why_stop var.
            return True, success, why_stop

        else:  # conv. not met
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit / self.prev_data_misfit) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit,
                        'prev_data_misfit': self.prev_data_misfit,
                        'gamma': self.gamma}

            ###############################################
            ##### update Lambda step-size values ##########
            ###############################################
            # If reduction in mean data misfit, reduce damping param
            if self.data_misfit < self.prev_data_misfit and self.data_misfit_std < self.prev_data_misfit_std:
                # Reduce damping parameter (divide calculations for ANALYSISDEBUG purpose)
                self.gamma = self.gamma + (self.gamma_max - self.gamma) * 2 ** (
                    -(self.iteration) / (self.gamma_factor - 1))
                success = True
                self.current_state = cp.deepcopy(self.state)
                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)

            elif self.data_misfit < self.prev_data_misfit and self.data_misfit_std >= self.prev_data_misfit_std:
                # accept itaration, but keep lam the same
                success = True
                self.current_state = cp.deepcopy(self.state)
                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)

            else:  # Reject iteration, and increase lam
                # Increase damping parameter (divide calculations for ANALYSISDEBUG purpose)
                err_str = f"Misfit increased. Set new start step length and try again. Final ojective function value is {self.data_misfit:0.1f}"
                self.logger.info(err_str)
                sys.exit(err_str)
                success = False

            if success:
                self.logger.info(f'Successfull iteration number {self.iteration}! Objective function reduced from '
                                 f'{self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}. New Gamma for next analysis: '
                                 f'{self.gamma}')
            else:
                self.logger.info(f'Failed iteration number {self.iteration}! Objective function increased from '
                                 f'{self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}. New Gamma for repeated analysis: '
                                 f'{self.gamma}')

            # Return conv = False, why_stop var.
            return False, success, why_stop

    def _ext_iter_param(self):
        """
        Extract parameters needed in LM-EnRML from the ITERATION keyword given in the DATAASSIM part of PIPT init.
        file. These parameters include convergence tolerances and parameters for the damping parameter. Default
        values for these parameters have been given here, if they are not provided in ITERATION.
        """

        # Predefine all the default values
        self.data_misfit_tol = 0.01
        self.step_tol = 0.01
        self.gamma = 0.2
        self.gamma_max = 0.5
        self.gamma_factor = 2.5
        self.trunc_energy = 0.95
        self.iteration = 0

        # Loop over options in ITERATION and extract the parameters we want
        for i, opt in enumerate(list(zip(*self.keys_da['iteration']))[0]):
            if opt == 'data_misfit_tol':
                self.data_misfit_tol = self.keys_da['iteration'][i][1]
            if opt == 'step_tol':
                self.step_tol = self.keys_da['iteration'][i][1]
            if opt == 'gamma':
                self.gamma = self.keys_da['iteration'][i][1]
            if opt == 'gamma_max':
                self.gamma_max = self.keys_da['iteration'][i][1]
            if opt == 'gamma_factor':
                self.gamma_factor = self.keys_da['iteration'][i][1]

        if 'energy' in self.keys_da:
            # initial energy (Remember to extract this)
            self.trunc_energy = self.keys_da['energy']
            if self.trunc_energy > 1:  # ensure that it is given as percentage
                self.trunc_energy /= 100.


class gnenrml_approx(gnenrmlMixIn, approx_update):
    pass


class gnenrml_full(gnenrmlMixIn, full_update):
    pass


class gnenrml_subspace(gnenrmlMixIn, subspace_update):
    pass


class gnenrml_margis(gnenrmlMixIn, margIS_update):
    '''
    The marg-IS scheme is currently not available in this version of PIPT. To utilize the scheme you have to import the
    *margIS_update* class from a standalone repository.
    '''
    pass


class co_lm_enrml(lmenrmlMixIn, approx_update):
    """
    This is the implementation of the approximative LM-EnRML algorithm as described in [`chen2013`][].

    This algorithm is quite similar to the lm_enrml as provided above, and will therefore inherit most of its methods.
    We only change the calc_analysis part...

    % Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
    """

    def __init__(self, keys_da):
        """
        The class is initialized by passing the PIPT init. file upwards in the hierarchy to be read and parsed in
        `pipt.input_output.pipt_init.ReadInitFile`.
        """
        # Call __init__ in parent class
        super().__init__(keys_da)

    def calc_analysis(self):
        """
        Calculate the update step in approximate LM-EnRML code.

        Attributes
        ----------
        iteration : int
            Iteration number

        Returns
        -------
        success : bool
            True if data mismatch is decreasing, False if increasing
        """
        # Get assimilation order as a list
        self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]

        # When handling large cases, it may be very costly to assemble the data covariance and localizaton matrix.
        # To alleviate this in the simultuaneus-iterative scheme we store these matrices, the list of states and
        # the list of data types after the first iteration.

        if not hasattr(self, 'list_datatypes'):
            # Get list of data types to be assimilated and of the free states. Do this once, because listing keys from a
            # Python dictionary just when needed (in different places) may not yield the same list!
            self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(
                self.obs_data, self.assim_index)
            list_datatypes = self.list_datatypes
            self.list_states = list(self.state.keys())
            list_states = self.list_states
            list_act_datatypes = self.list_act_datatypes

            # self.cov_data = np.load('CD.npz')['arr_0']
            # Generate the realizations of the observed data once
            # Augment observed and predicted data
            self.obs_data_vector, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                                            self.list_datatypes)
            obs_data_vector = self.obs_data_vector

            # Generate the data auto-covariance matrix
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                if hasattr(self, 'cov_data'):  # cd matrix has been imported
                    tmp_E = np.dot(cholesky(self.cov_data).T,
                                   np.random.randn(self.cov_data.shape[0], self.ne))
                else:
                    tmp_E = at.extract_tot_empirical_cov(
                        self.datavar, self.assim_index, self.list_datatypes, self.ne)
                # self.E = (tmp_E - tmp_E.mean(1)[:,np.newaxis])/np.sqrt(self.ne - 1)/
                if 'screendata' in self.keys_da and self.keys_da['screendata'] == 'yes':
                    tmp_E = at.screen_data(tmp_E, self.aug_pred_data,
                                           obs_data_vector, self.iteration)
                self.E = tmp_E
                self.real_obs_data = obs_data_vector[:, np.newaxis] - tmp_E

                self.cov_data = np.var(self.E, ddof=1,
                                       axis=1)  # calculate the variance, to be used for e.g. data misfit calc
                # self.cov_data = ((self.E * self.E)/(self.ne-1)).sum(axis=1) # calculate the variance, to be used for e.g. data misfit calc
                self.scale_data = np.sqrt(self.cov_data)
            else:
                if not hasattr(self, 'cov_data'):  # if cd is not loaded
                    self.cov_data = at.gen_covdata(
                        self.datavar, self.assim_index, self.list_datatypes)
                # data screening
                if 'screendata' in self.keys_da and self.keys_da['screendata'] == 'yes':
                    self.cov_data = at.screen_data(
                        self.cov_data, self.aug_pred_data, obs_data_vector, self.iteration)

                init_en = Cholesky()  # Initialize GeoStat class for generating realizations
                self.real_obs_data, self.scale_data = init_en.gen_real(self.obs_data_vector, self.cov_data, self.ne,
                                                                       return_chol=True)

            self.datavar = at.update_datavar(
                self.cov_data, self.datavar, self.assim_index, self.list_datatypes)
            self.current_state = cp.deepcopy(self.state)

            # Calc. misfit for the initial iteration
            data_misfit = at.calc_objectivefun(
                self.real_obs_data, self.aug_pred_data, self.cov_data)
            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = np.mean(data_misfit)
            self.prior_data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            if self.lam == 'auto':
                self.lam = 0.5 * self.prior_data_misfit

        else:
            _, self.aug_pred_data = at.aug_obs_pred_data(
                self.obs_data, self.pred_data, assim_index, self.list_datatypes)

        # Mean pred_data and perturbation matrix with scaling
        mean_preddata = np.mean(self.aug_pred_data, 1)
        if len(self.scale_data.shape) == 1:
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) * (
                    self.aug_pred_data - np.dot(mean_preddata[:, None], np.ones((1, self.ne))))
            else:
                pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) * (
                    self.aug_pred_data - np.dot(mean_preddata[:, None], np.ones((1, self.ne)))) / \
                    (np.sqrt(self.ne - 1))
        else:
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                pert_preddata = solve(self.scale_data, self.aug_pred_data -
                                      np.dot(mean_preddata[:, None], np.ones((1, self.ne))))
            else:
                pert_preddata = solve(self.scale_data, self.aug_pred_data - np.dot(mean_preddata[:, None], np.ones((1, self.ne)))) / \
                    (np.sqrt(self.ne - 1))
        self.pert_preddata = pert_preddata

        self.update()
        if hasattr(self, 'step'):
            aug_state_upd = aug_state + self.step
        if hasattr(self, 'w_step'):
            self.W = self.current_W - self.w_step
            aug_prior_state = at.aug_state(self.prior_state, self.list_states)
            aug_state_upd = np.dot(aug_prior_state, (np.eye(
                self.ne) + self.W / np.sqrt(self.ne - 1)))

        # Extract updated state variables from aug_update
        self.state = at.update_state(aug_state_upd, self.state, self.list_states)
        self.state = at.limits(self.state, self.prior_info)


class gn_enrml(lmenrmlMixIn):
    """
    This is the implementation of the stochastig IES as  described in [`raanes2019`][].

    More information about the method is found in [`evensen2019`][].
    This implementation is the Gauss-Newton version.

    This algorithm is quite similar to the `lm_enrml` as provided above, and will therefore inherit most of its methods.
    We only change the calc_analysis part...
    """

    def __init__(self, keys_da):
        """
        The class is initialized by passing the PIPT init. file upwards in the hierarchy to be read and parsed in
        `pipt.input_output.pipt_init.ReadInitFile`.
        """
        # Call __init__ in parent class
        super().__init__(keys_da)

    def calc_analysis(self):
        """
        Changelog
        ---------
        - KF 25/2-20
        """
        # Get assimilation order as a list
        assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]

        # When handling large cases, it may be very costly to assemble the data covariance and localizaton matrix.
        # To alleviate this in the simultuaneus-iterative scheme we store these matrices, the list of states and
        # the list of data types after the first iteration.

        if not hasattr(self, 'list_datatypes'):
            # Get list of data types to be assimilated and of the free states. Do this once, because listing keys from a
            # Python dictionary just when needed (in different places) may not yield the same list!
            self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(
                self.obs_data, assim_index)
            list_datatypes = self.list_datatypes
            self.list_states = list(self.state.keys())
            list_states = self.list_states
            list_act_datatypes = self.list_act_datatypes

            # Generate the realizations of the observed data once
            # Augment observed and predicted data
            self.obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, assim_index,
                                                                   self.list_datatypes)
            obs_data_vector = self.obs_data_vector

            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                if hasattr(self, 'cov_data'):  # cd matrix has been imported
                    tmp_E = np.dot(cholesky(self.cov_data).T, np.random.randn(
                        self.cov_data.shape[0], self.ne))
                else:
                    tmp_E = at.extract_tot_empirical_cov(
                        self.datavar, assim_index, self.list_datatypes, self.ne)
                # self.E = (tmp_E - tmp_E.mean(1)[:,np.newaxis])/np.sqrt(self.ne - 1)/
                self.real_obs_data = obs_data_vector[:, np.newaxis] - tmp_E

                self.cov_data = np.var(tmp_E, ddof=1,
                                       axis=1)  # calculate the variance, to be used for e.g. data misfit calc
                # self.cov_data = ((self.E * self.E)/(self.ne-1)).sum(axis=1) # calculate the variance, to be used for e.g. data misfit calc
                self.scale_data = np.sqrt(self.cov_data)
            else:
                if not hasattr(self, 'cov_data'):  # if cd is not loaded
                    self.cov_data = at.gen_covdata(
                        self.datavar, assim_index, self.list_datatypes)
                # data screening
                if 'screendata' in self.keys_da and self.keys_da['screendata'] == 'yes':
                    self.cov_data = at.screen_data(
                        self.cov_data, pred_data, obs_data_vector, self.iteration)

                init_en = Cholesky()  # Initialize GeoStat class for generating realizations
                self.real_obs_data, self.scale_data = init_en.gen_real(self.obs_data_vector, self.cov_data, self.ne,
                                                                       return_chol=True)

            self.datavar = at.update_datavar(
                self.cov_data, self.datavar, assim_index, self.list_datatypes)
            cov_data = self.cov_data
            obs_data = self.real_obs_data
            #
            self.current_state = cp.deepcopy(self.state)
            #
            self.aug_prior = cp.deepcopy(at.aug_state(
                self.current_state, self.list_states))
            # self.mean_prior = aug_prior.mean(axis=1)
            # self.X = (aug_prior - np.dot(np.resize(self.mean_prior, (len(self.mean_prior), 1)),
            #                                                  np.ones((1, self.ne))))
            self.W = np.zeros((self.ne, self.ne))

            self.proj = (np.eye(self.ne) - (1 / self.ne) *
                         np.ones((self.ne, self.ne))) / np.sqrt(self.ne - 1)
            self.E = np.dot(obs_data, self.proj)

            # Calc. misfit for the initial iteration
            if len(cov_data.shape) == 1:
                tmp_data_misfit = np.diag(np.dot((pred_data - obs_data).T,
                                                 np.dot(np.expand_dims(self.cov_data ** (-1), axis=1),
                                                        np.ones((1, self.ne))) * (pred_data - obs_data)))
            else:
                tmp_data_misfit = np.diag(
                    np.dot((pred_data - obs_data).T, solve(self.cov_data, (pred_data - obs_data))))
            mean_data_misfit = np.mean(tmp_data_misfit)
            # mean_data_misfit = np.median(tmp_data_misfit)
            std_data_misfit = np.std(tmp_data_misfit)

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = mean_data_misfit
            self.prior_data_misfit = mean_data_misfit
            self.data_misfit_std = std_data_misfit

        else:
            # for analysis debug...
            list_datatypes = self.list_datatypes
            list_act_datatypes = self.list_act_datatypes
            list_states = self.list_states
            cov_data = self.cov_data
            obs_data_vector = self.obs_data_vector
            _, pred_data = at.aug_obs_pred_data(
                self.obs_data, self.pred_data, assim_index, self.list_datatypes)
            obs_data = self.real_obs_data

        if len(self.scale_data.shape) == 1:
            Y = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) * \
                np.dot(pred_data, self.proj)
        else:
            Y = solve(self.scale_data, np.dot(pred_data, self.proj))
        omega = np.eye(self.ne) + np.dot(self.W, self.proj)
        LU = lu_factor(omega.T)
        S = lu_solve(LU, Y.T).T
        if len(self.scale_data.shape) == 1:
            scaled_misfit = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                   np.ones((1, self.ne))) * (obs_data - pred_data)
        else:
            scaled_misfit = solve(self.scale_data, (obs_data - pred_data))

        u, s, v = np.linalg.svd(S, full_matrices=False)
        if self.trunc_energy < 1:
            ti = (np.cumsum(s) / sum(s)) <= self.trunc_energy
            u, s, v = u[:, ti].copy(), s[ti].copy(), v[ti, :].copy()

        ps_inv = np.diag([el_s ** (-1) for el_s in s])
        if len(self.scale_data.shape) == 1:
            X = np.dot(ps_inv, np.dot(u.T, np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                                  np.ones((1, self.ne))) * self.E))
        else:
            X = np.dot(ps_inv, np.dot(u.T, solve(self.scale_data, self.E)))
        Lam, z = np.linalg.eig(np.dot(X, X.T))

        X2 = np.dot(u, np.dot(ps_inv.T, z))

        X3_m = np.dot(S.T, X2)
        # X3_old = np.dot(X2, np.linalg.solve(np.eye(len(Lam)) + np.diag(Lam), X2.T))
        step_m = np.dot(np.dot(X3_m, inv(np.eye(len(Lam)) + np.diag(Lam))),
                        np.dot(X3_m.T, self.W))

        if 'localization' in self.keys_da:
            if self.keys_da['localization'][1][0] == 'autoadaloc':
                loc_step_d = np.dot(np.linalg.pinv(self.aug_prior), self.localization.auto_ada_loc(self.aug_prior,
                                                                                                   np.dot(np.dot(S.T, X2),
                                                                                                          np.dot(inv(
                                                                                                              np.eye(len(Lam)) + np.diag(Lam)),
                                                                                                       np.dot(X2.T, scaled_misfit))),
                                                                                                   self.list_states,
                                                                                                   **{'prior_info': self.prior_info}))
                self.step = self.lam * (self.W - (step_m + loc_step_d))
        else:
            step_d = np.dot(np.linalg.inv(omega).T,  np.dot(np.dot(Y.T, X2),
                                                            np.dot(inv(np.eye(len(Lam)) + np.diag(Lam)),
                                                                   np.dot(X2.T, scaled_misfit))))
            self.step = self.lam * (self.W - (step_m + step_d))

        self.W -= self.step

        aug_state_upd = np.dot(self.aug_prior, (np.eye(
            self.ne) + self.W / np.sqrt(self.ne - 1)))

        # Extract updated state variables from aug_update
        self.state = at.update_state(aug_state_upd, self.state, self.list_states)

        self.state = at.limits(self.state, self.prior_info)

    def check_convergence(self):
        """
        Check if GN-EnRML have converged based on evaluation of change sizes of objective function, state and damping
        parameter. Very similar to original function, but exit if there is no reduction in obj. function.

        Returns
        -------
        conv : bool
            Logic variable indicating if the algorithm has converged.

        status : bool
            Indicates whether the objective function has reduced.

        why_stop : dict
            Dictionary with keys corresponding to convergence criteria, with logical variables indicating
            which of them has been met.

        Changelog
        ---------
        - ST 3/6-16
        - ST 6/6-16: Added LM damping param. check
        - KF 16/11-20: Modified for GN-EnRML
        - KF 10/3-21: Output whether the method reduced the objective function
        """
        # Prelude to calc. conv. check (everything done below is from calc_analysis)
        if hasattr(self, 'list_datatypes'):
            assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            list_datatypes = self.list_datatypes
            cov_data = self.cov_data
            obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, assim_index,
                                                              list_datatypes)
            mean_preddata = np.mean(pred_data, 1)
        else:
            assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            list_datatypes, _ = at.get_list_data_types(self.obs_data, assim_index)
            # cov_data = at.gen_covdata(self.datavar, assim_index, list_datatypes)
            obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, assim_index,
                                                              list_datatypes)
            # mean_preddata = np.mean(pred_data, 1)

        success = False

        # if inital conv. check, there are no prev_data_misfit
        if self.prev_data_misfit is None:
            self.data_misfit = np.mean(self.data_misfit)
            self.prev_data_misfit = self.data_misfit
            self.prev_data_misfit_std = self.data_misfit_std
            success = True
        # update the last mismatch, only if this was a reduction of the misfit
        if self.data_misfit < self.prev_data_misfit:
            self.prev_data_misfit = self.data_misfit
            self.prev_data_misfit_std = self.data_misfit_std
            success = True
        # if there was no reduction of the misfit, retain the old "valid" data misfit.

        # Calc. std dev of data misfit (used to update lamda)
        # mat_obs = np.dot(obs_data_vector.reshape((len(obs_data_vector), 1)), np.ones((1, self.ne)))  # use the perturbed
        # data instead.
        mat_obs = self.real_obs_data
        if len(cov_data.shape) == 1:
            data_misfit = np.diag(np.dot((pred_data - mat_obs).T,
                                         np.dot(np.expand_dims(self.cov_data ** (-1), axis=1),
                                                np.ones((1, self.ne))) * (pred_data - mat_obs)))
        else:
            data_misfit = np.diag(np.dot((pred_data - mat_obs).T,
                                  solve(self.cov_data, (pred_data - mat_obs))))
        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # # Calc. mean data misfit for convergence check, using the updated state variable
        # self.data_misfit = np.dot((mean_preddata - obs_data_vector).T,
        #                      solve(cov_data, (mean_preddata - obs_data_vector)))
        # if self.data_misfit > self.prev_data_misfit:
        #    print(f'\n\nMisfit increased from {self.prev_data_misfit:.1f} to {self.data_misfit:.1f}. Exiting')
        #    self.logger.info(f'\n\nMisfit increased from {self.prev_data_misfit:.1f} to {self.data_misfit:.1f}. Exiting')

        # Convergence check: Relative step size of data misfit or state change less than tolerance
        if abs(1 - (self.data_misfit / self.prev_data_misfit)) < self.data_misfit_tol \
                or np.any(abs(np.mean(self.step, 1)) < self.step_tol) \
                or self.lam >= self.lam_max:
            # or self.data_misfit > self.prev_data_misfit:
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit / self.prev_data_misfit) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit,
                        'prev_data_misfit': self.prev_data_misfit,
                        'step_size_stop': np.any(abs(np.mean(self.step, 1)) < self.step_tol),
                        'step_size': self.step,
                        'lambda': self.lam,
                        'lambda_stop': self.lam >= self.lam_max}

            if self.data_misfit >= self.prev_data_misfit:
                success = False
                self.logger.info(f'Iterations have converged after {self.iteration} iterations. Objective function reduced '
                                 f'from {self.prior_data_misfit:0.1f} to {self.prev_data_misfit:0.1f}')
            else:
                self.logger.info(f'Iterations have converged after {self.iteration} iterations. Objective function reduced '
                                 f'from {self.prior_data_misfit:0.1f} to {self.data_misfit:0.1f}')

            # Return conv = True, why_stop var.
            return True, success, why_stop

        else:  # conv. not met
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit / self.prev_data_misfit) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit,
                        'prev_data_misfit': self.prev_data_misfit,
                        'step_size': self.step,
                        'step_size_stop': np.any(abs(np.mean(self.step, 1)) < self.step_tol),
                        'lambda': self.lam,
                        'lambda_stop': self.lam >= self.lam_max}

            ###############################################
            ##### update Lambda step-size values ##########
            ###############################################
            if self.data_misfit < self.prev_data_misfit and self.data_misfit_std < self.prev_data_misfit_std:
                # If reduction in mean data misfit, increase step length
                self.lam = self.lam + (self.lam_max - self.lam) * \
                    2 ** (-(self.iteration) / (self.gamma - 1))
                success = True
                self.current_state = deepcopy(self.state)
            elif self.data_misfit < self.prev_data_misfit and self.data_misfit_std >= self.prev_data_misfit_std:
                # Accept itaration, but keep lam the same
                success = True
                self.current_state = deepcopy(self.state)
            else:  # Reject iteration, and decrease step length
                self.lam = self.lam / self.gamma
                success = False

            if success:
                self.logger.info(f'Successfull iteration number {self.iteration}! Objective function reduced from '
                                 f'{self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}. New Lamba for next analysis: '
                                 f'{self.lam}')
            else:
                self.logger.info(f'Failed iteration number {self.iteration}! Objective function increased from '
                                 f'{self.prev_data_misfit:0.1f} to {self.data_misfit:0.1f}. New Lamba for repeated analysis: '
                                 f'{self.lam}')
                # Reset data misfit to prev_data_misfit (because the current state is neglected)
                self.data_misfit = self.prev_data_misfit
                self.data_misfit_std = self.prev_data_misfit_std

            return False, success, why_stop
