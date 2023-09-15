"""Ensemble optimisation (steepest descent with ensemble gradient)."""
# External imports
import numpy as np
import scipy as scipy
import os

from numpy import linalg as la
from copy import deepcopy
import logging

from scipy.special import polygamma, digamma
from scipy import stats

# Internal imports
from popt.misc_tools import optim_tools as ot, basic_tools as bt
from pipt.misc_tools import analysis_tools as at
from ensemble.ensemble import Ensemble as PETEnsemble
import popt.update_schemes.optimizers as opt


class EnOpt(PETEnsemble):
    """
    This is an implementation of the steepest ascent ensemble optimization algorithm given in, e.g., Chen et al.,
    2009, 'Efficient Ensemble-Based Closed-Loop Production Optimization', SPE Journal, 14 (4): 634-645.
    The update of the control variable is done with the simple steepest (or gradient) ascent algorithm:

        x_l = (1 / alpha) * R * S + x_(l-1)

    where x is the control variable, l and l-1 is the current and previous iteration, alpha is a step limiter,
    R is a smoothing matrix (e.g., covariance matrix for x), and S is the ensemble gradient (or sensitivity).
    """

    def __init__(self, keys_opt, keys_en, sim, obj_func, optimizer='GA'):

        # init PETEnsemble
        super(EnOpt, self).__init__(keys_en, sim)

        # set logger
        self.logger = logging.getLogger('PET.POPT')

        # Optimization keys
        self.keys_opt = keys_opt

        # Initialize EnOPT parameters
        self.upper_bound = []
        self.lower_bound = []
        self.step = 0  # state step
        self.cov = np.array([])  # ensemble covariance
        self.cov_step = 0  # covariance step
        self.num_samples = self.ne
        self.alpha_iter = 0  # number of backtracking steps
        self.resamp_iter = 0
        self.num_func_eval = 0  # Total number of function evaluations
        self._ext_enopt_param()

        # Get objective function
        self.obj_func = obj_func

        # Calculate objective function of startpoint
        self.ne = self.num_models
        for key in self.state.keys():
            # make sure state is a numpy array
            self.state[key] = np.array(self.state[key])
        if self.num_models > 1:
            self.aux_input = list(np.arange(self.num_models))
        np.savez('ini_state.npz', **self.state)
        self.calc_prediction()
        self.num_func_eval += self.ne
        self.obj_func_values = self.obj_func(self.pred_data, self.keys_opt, self.sim.true_order)
        self.save_analysis_debug(0)

        # Initialize function values and covariance
        self.sens_matrix = None
        self.cov_sens_matrix = None

        # Initialize optimizer
        if optimizer == 'GA':
            self.optimizer = opt.GradientAscent(self.alpha, self.beta)
        elif optimizer == 'Adam':
            self.optimizer = opt.Adam(self.alpha)

    def calc_update(self, iteration, logger=None):
        """
        Update using steepest ascent method with ensemble gradients
        """

        # Augment the state dictionary to an array
        list_states = list(self.state.keys())

        # Current state vector
        self._scale_state()
        current_state = self.state

        # Calc sensitivity
        self.calc_ensemble_sensitivity()

        improvement = False
        success = False
        self.alpha_iter = 0
        beta = self.beta
        self.resamp_iter = 0
         
        while improvement is False:

            # Augment state
            aug_state = ot.aug_optim_state(current_state, list_states)

            # Compute the steepest ascent step. Scale the gradient with 2-norm (or inf-norm: np.inf)
            normalize = np.maximum(la.norm(self.sens_matrix, np.inf), 1e-12)            
	    search_direction = self.sens_matrix / normalize
	    H = 1
            if self.hessian:
                H = 1 / np.diag(self.cov_sens_matrix)
	    search_direction *= H
            aug_state_upd     = self.optimizer.apply_update(aug_state, search_direction, iter=iteration)

            # Make sure update is within bounds
            if self.upper_bound and self.lower_bound:
                np.clip(aug_state_upd, 0, 1, out=aug_state_upd)

            # Calculate new objective function
            self.state = ot.update_optim_state(aug_state_upd, self.state, list_states)
            self.ne = self.num_models
            self._invert_scale_state()
            if self.num_models > 1:
                self.aux_input = list(np.arange(self.num_models))
            run_success = self.calc_prediction()
            self.num_func_eval += self.ne
            new_func_values = 0
            if run_success:
                new_func_values = self.obj_func(self.pred_data, self.keys_opt, self.sim.true_order)

            if np.mean(new_func_values) - np.mean(self.obj_func_values) > self.obj_func_tol:

                # Iteration was a success
                improvement = True
                success = True
                self.optimizer.restore_parameters()

                # Update objective function values and step
                self.obj_func_values = new_func_values
           
                # Update covariance (currently we don't apply backtracking for alpha_cov)
                normalize = np.maximum(la.norm(self.cov_sens_matrix, np.inf), 1e-12)
                self.cov_step = self.alpha_cov * self.cov_sens_matrix / normalize + beta * self.cov_step
                self.cov = self.cov + self.cov_step
                self.cov = self.get_sym_pos_semidef(self.cov)

                # Write logging info
                if logger is not None:
                    info_str_iter = '{:<10} {:<10} {:<10.2f} {:<10.2e} {:<10.2e}'.\
                        format(iteration, self.alpha_iter, np.mean(self.obj_func_values), self.optimizer._step_size, self.cov[0, 0])
                    logger.info(info_str_iter)

                # Update self.alpha in the one-dimensional case
                if len(aug_state) == 1:
                    self.optimizer.step_size /= 2

            else:

                # If we do not have a reduction in the objective function, we reduce the step limiter
                if self.alpha_iter < self.alpha_iter_max:
                    # Decrease alpha
                    self.optimizer.apply_backtracking()
                    self.alpha_iter += 1
                elif self.resamp_iter < self.resamplings and not np.mean(new_func_values) - np.mean(self.obj_func_values) > 0:
                    self.resamp_iter += 1
                    self.alpha_iter = 0
                    self.calc_ensemble_sensitivity()
                else:
                    success = False
                    break

        # Save variables defined in ANALYSISDEBUG keyword.
        if success:
            self.save_analysis_debug(iteration)

        return success

    # force matrix to positive semidefinite
    @staticmethod
    def get_sym_pos_semidef(a):

        rtol = 1e-05
        if a.ndim > 1:
            S, U = np.linalg.eigh(a)
            S = np.clip(S, 0, None) + rtol
            a = (U*S)@U.T
        else:
            a = np.maximum(a, rtol)
        return a

    def _ext_enopt_param(self):
        """
        Extract ENOPT parameters in OPTIM part if inputted.
        """

        # Default value for max. iterations
        default_obj_func_tol = 1e-6
        default_step_tol = 1e-6
        default_alpha = 0.1
        default_alpha_cov = 0.001
        default_beta = 0.0
        default_alpha_iter_max = 5
        default_num_models = 1

        default_resamplings = 0

        # Check if ENOPT has been given in OPTIM. If it is not present, we assign a default value
        if 'enopt' not in self.keys_opt:
            # Default value
            print('ENOPT keyword not found. Assigning default value to EnOpt parameters!')
        else:
            # Make ENOPT a 2D list
            if not isinstance(self.keys_opt['enopt'][0], list):
                enopt = [self.keys_opt['enopt']]
            else:
                enopt = self.keys_opt['enopt']
            
            # Assign resamplings, if not exit, assign default value
            ind_resampling = bt.index2d(enopt, 'resamplings')
            # obj_func_tol does not exist
            if ind_resampling is None:
                self.resamplings = default_resamplings
            else:
                self.resamplings = enopt[ind_resampling[0]][ind_resampling[1] + 1]
            
            # Assign obj_func_tol, if not exit, assign default value
            ind_obj_func_tol = bt.index2d(enopt, 'obj_func_tol')
            # obj_func_tol does not exist
            if ind_obj_func_tol is None:
                self.obj_func_tol = default_obj_func_tol
            # obj_func_tol present; assign value
            else:
                self.obj_func_tol = enopt[ind_obj_func_tol[0]][ind_obj_func_tol[1] + 1]

            # Assign step_tol, if not exit, assign default value
            ind_step_tol = bt.index2d(enopt, 'step_tol')
            # step_tol does not exist
            if ind_step_tol is None:
                self.step_tol = default_step_tol
            # step_tol present; assign value
            else:
                self.step_tol = enopt[ind_step_tol[0]][ind_step_tol[1] + 1]

            # Assign alpha, if not exit, assign default value
            ind_alpha = bt.index2d(enopt, 'alpha')
            # step_tol does not exist
            if ind_alpha is None:
                self.alpha = default_alpha
            # step_tol present; assign value
            else:
                self.alpha = enopt[ind_alpha[0]][ind_alpha[1] + 1]
            # Assign alpha_cov, if not exit, assign default value
            ind_alpha_cov = bt.index2d(enopt, 'alpha_cov')
            # step_tol does not exist
            if ind_alpha_cov is None:
                self.alpha_cov = default_alpha_cov
            # step_tol present; assign value
            else:
                self.alpha_cov = enopt[ind_alpha_cov[0]][ind_alpha_cov[1] + 1]
            # Assign beta, if not exit, assign default value
            ind_beta = bt.index2d(enopt, 'beta')
                # step_tol does not exist
            if ind_beta is None:
                self.beta = default_beta
            # step_tol present; assign value
            else:
                self.beta = enopt[ind_beta[0]][ind_beta[1] + 1]

            # Assign alpha_iter_max, if not exit, assign default value
            ind_alpha_iter_max = bt.index2d(enopt, 'alpha_iter_max')
            # alpha_iter_max does not exist
            if ind_alpha_iter_max is None:
                self.alpha_iter_max = default_alpha_iter_max
            # alpha_iter present; assign value
            else:
                self.alpha_iter_max = enopt[ind_alpha_iter_max[0]][ind_alpha_iter_max[1] + 1]

            # Assign number of models, if not exit, assign default value
            ind_num_models = bt.index2d(enopt, 'num_models')
            if ind_num_models is None:  # num_models does not exist
                self.num_models = default_num_models
            else:  # num_models present; assign value
                self.num_models = int(enopt[ind_num_models[0]][ind_num_models[1] + 1])
            if self.num_models > 1:
                self.aux_input = list(np.arange(self.num_models))

            # Check if Hessian should be used
            ind_hessian = bt.index2d(enopt, 'hessian')
            if ind_hessian is None:  # do not use Hessian
                self.hessian = None
            else:  # use Hessian
                self.hessian = True

            value_cov = np.array([])
            for name in self.prior_info.keys():
                self.state[name] = self.prior_info[name]['mean']
                value_cov = np.append(value_cov, self.prior_info[name]['variance'] * np.ones((len(self.state[name]),)))
                if 'limits' in self.prior_info[name].keys():
                    self.lower_bound.append(self.prior_info[name]['limits'][0])
                    self.upper_bound.append(self.prior_info[name]['limits'][1])

            # Augment state
            list_state = list(self.state.keys())
            aug_state = ot.aug_optim_state(self.state, list_state)
            self.cov = value_cov * np.eye(len(aug_state))

    def calc_ensemble_sensitivity(self):
        """
        Calculate the sensitivity matrix normally associated with ensemble optimization algorithms, usually defined as:

            S ~= C_x * G.T

        where '~=' means 'approximately equal', C_x is the state covariance matrix, and G is the standard
        sensitivity matrix. The ensemble sensitivity matrix is calculated as:

            S = (1/ (ne - 1)) * U * J.T

        where U and J are ensemble matrices of state (or control variables) and objective function perturbed by their
        respective means. In practice (and in this method), S is calculated by perturbing the current state (control
        variable) with Gaussian random numbers from N(0, C_x) (giving U), running the generated state ensemble (U)
        through the simulator to give an ensemble of objective function values (J), and in the end calculate S. Note
        that S is an Ns x 1 vector, where Ns is length of the state vector (the objective function is just a scalar!)
        """

        # Generate ensemble of states
        self.ne = self.num_samples
        nr = 1
        if self.num_models > 1:
            if np.remainder(self.num_samples, self.num_models) == 0:
                nr = self.num_samples / self.num_models
                self.aux_input = list(np.repeat(np.arange(self.num_models), nr))
            else:
                print('num_samples must be a multiplum of num_models!')
                sys.exit(0)
        self.state = self._gen_state_ensemble()

        self._invert_scale_state()
        self.calc_prediction()
        self._scale_state()
        self.num_func_eval += self.ne
        obj_func_values = self.obj_func(self.pred_data, self.keys_opt, self.sim.true_order)
        obj_func_values = np.array(obj_func_values)

        # Finally, we calculate the ensemble sensitivity matrix.
        # First we need to perturb state and obj. func. ensemble with their mean. Note that, obj_func has shape (
        # ne,)!
        list_states = list(self.state.keys())
        aug_state = at.aug_state(self.state, list_states)
        pert_state = aug_state - np.dot(aug_state.mean(1)[:, None], np.ones((1, self.ne)))
        pert_obj_func = obj_func_values - np.array(np.repeat(self.obj_func_values, nr))

        # Calculate cross-covariance between state and obj. func. which is the ensemble sensitivity matrix
        # self.sens_matrix = at.calc_crosscov(aug_state, obj_func_values)

        # Calculate the gradient for mean and covariance matrix
        g_c = np.zeros(self.cov.shape)
        g_m = np.zeros(aug_state.shape[0])
        for i in np.arange(self.ne):
            g_m = g_m + pert_obj_func[i] * pert_state[:, i]
            g_c = g_c + pert_obj_func[i] * (np.outer(pert_state[:, i], pert_state[:, i]) - self.cov)

        self.cov_sens_matrix = g_c / (self.ne - 1)
        self.sens_matrix = g_m / (self.ne - 1)

    def _gen_state_ensemble(self):

        # Generate ensemble with the current state (control variable) as the mean and using the covariance matrix
        state_en = {}
        cov_blocks = ot.corr2BlockDiagonal(self.state, self.cov)
        start = 0
        for i, statename in enumerate(self.state.keys()):
            mean = self.state[statename]
            cov = cov_blocks[i]
            if ['nesterov'] in self.keys_opt['enopt']:
                step = self.optimizer.get_momentum_for_nesterov()
                if not isinstance(step, (int, float)):
                    stop  = start + len(self.state[statename])
                    mean  = mean + step[start:stop]
                    cov   = cov + self.beta * self.cov_step[start:stop, start:stop]
                    cov   = self.get_sym_pos_semidef(cov)
                    start = start + len(self.state[statename]) 

            temp_state_en = np.random.multivariate_normal(mean, cov, self.ne).transpose()
            if self.upper_bound and self.lower_bound:
                np.clip(temp_state_en, 0, 1, out=temp_state_en)

            state_en[statename] = np.array([mean]).T + temp_state_en - np.array([np.mean(temp_state_en,1)]).T

        return state_en

    def _scale_state(self):
        if self.upper_bound and self.lower_bound:
            for i, key in enumerate(self.state):
                self.state[key] = (self.state[key] - self.lower_bound[i]) / (self.upper_bound[i] - self.lower_bound[i])
                np.clip(self.state[key], 0, 1, out=self.state[key])

    def _invert_scale_state(self):
        if self.upper_bound and self.lower_bound:
            for i, key in enumerate(self.state):
                self.state[key] = self.lower_bound[i] + self.state[key] * (self.upper_bound[i] - self.lower_bound[i])

    def save_analysis_debug(self, iteration):
        if 'analysisdebug' in self.keys_opt:

            # Init dict. of variables to save
            save_dict = {}

            # Make sure "ANALYSISDEBUG" gives a list
            if isinstance(self.keys_opt['analysisdebug'], list):
                analysisdebug = self.keys_opt['analysisdebug']
            else:
                analysisdebug = [self.keys_opt['analysisdebug']]

            # Loop over variables to store in save list
            for save_typ in analysisdebug:
                if save_typ in locals():
                    save_dict[save_typ] = eval('{}'.format(save_typ))
                elif hasattr(self, save_typ):
                    save_dict[save_typ] = eval('self.{}'.format(save_typ))
                else:
                    print(f'Cannot save {save_typ}!\n\n')

            # Save the variables
            if 'debug_save_folder' in self.keys_opt:
                folder = self.keys_opt['debug_save_folder']
            else:
                folder = './'
            
            #Make folder (if it does not exist) 
            if not os.path.exists(folder):
                os.mkdir(folder)

