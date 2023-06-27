"""Ensemble optimisation (steepest descent with ensemble gradient)."""
# External imports
import numpy as np
from numpy import linalg as la
from copy import deepcopy
import logging

# Internal imports
from popt.misc_tools import optim_tools as ot, basic_tools as bt
from pipt.misc_tools import analysis_tools as at
from ensemble.ensemble import Ensemble as PETEnsemble


class EnOpt(PETEnsemble):
    """
    This is an implementation of the steepest ascent ensemble optimization algorithm given in, e.g., Chen et al.,
    2009, 'Efficient Ensemble-Based Closed-Loop Production Optimization', SPE Journal, 14 (4): 634-645.
    The update of the control variable is done with the simple steepest (or gradient) ascent algorithm:

        x_l = (1 / alpha) * R * S + x_(l-1)

    where x is the control variable, l and l-1 is the current and previous iteration, alpha is a step limiter,
    R is a smoothing matrix (e.g., covariance matrix for x), and S is the ensemble gradient (or sensitivity).
    """

    def __init__(self, keys_opt, keys_en, sim, obj_func):

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
        self.num_func_eval = 0  # Total number of function evaluations
        self._ext_enopt_param()

        # Get objective function
        self.obj_func = obj_func

        # Calculate objective function of startpoint
        self.ne = self.num_models
        for key in self.state.keys():
            self.state[key] = np.array(self.state[key])  # make sure state is a numpy array
        np.savez('ini_state.npz', **self.state)
        self.calc_prediction()
        self.num_func_eval += self.ne
        self.obj_func_values = self.obj_func(self.pred_data, self.keys_opt, self.sim.true_order)
        self.save_analysis_debug(0)

        # Initialize function values and covariance
        self.sens_matrix = None
        self.cov_sens_matrix = None

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
        alpha = self.alpha
        beta = self.beta
        while improvement is False:

            # Augment state
            aug_state = ot.aug_optim_state(current_state, list_states)

            # Compute the steepest ascent step. Scale the gradient with 2-norm (or inf-norm: np.inf)
            normalize = np.maximum(la.norm(self.sens_matrix, np.inf), 1e-6)
            new_step = alpha * self.sens_matrix / normalize + beta * self.step

            # Calculate updated state
            aug_state_upd = aug_state + np.squeeze(new_step)

            # Make sure update is within bounds
            if self.upper_bound and self.lower_bound:
                np.clip(aug_state_upd, 0, 1, out=aug_state_upd)

            # Calculate new objective function
            self.state = ot.update_optim_state(aug_state_upd, self.state, list_states)
            self.ne = self.num_models
            self._invert_scale_state()
            run_success = self.calc_prediction()
            self.num_func_eval += self.ne
            new_func_values = 0
            if run_success:
                new_func_values = self.obj_func(self.pred_data, self.keys_opt, self.sim.true_order)

            if np.mean(new_func_values) - np.mean(self.obj_func_values) > self.obj_func_tol:

                # Iteration was a success
                improvement = True
                success = True

                # Update objective function values and step
                self.obj_func_values = new_func_values
                self.step = new_step

                # Update covariance (currently we don't apply backtracking for alpha_cov)
                normalize = np.maximum(la.norm(self.cov_sens_matrix, np.inf), 1e-6)
                self.cov_step = self.alpha_cov * self.cov_sens_matrix / normalize + beta * self.cov_step
                self.cov = self.cov + self.cov_step
                self.cov = self.get_sym_pos_semidef(self.cov)

                # Write logging info
                if logger is not None:
                    info_str_iter = '{:<10} {:<10} {:<10.2f} {:<10.2e} {:<10.2e}'.\
                        format(iteration, self.alpha_iter, np.mean(self.obj_func_values), alpha, self.cov[0, 0])
                    logger.info(info_str_iter)

                # Update self.alpha in the one-dimensional case
                if len(aug_state) == 1:
                    self.alpha /= 2

            else:

                # If we do not have a reduction in the objective function, we reduce the step limiter
                if self.alpha_iter < self.alpha_iter_max:
                    # Decrease alpha
                    alpha /= 2
                    beta /= 2
                    self.alpha_iter += 1
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
        self.state = self._gen_state_ensemble()
        self._invert_scale_state()
        self.calc_prediction()
        self.num_func_eval += self.ne
        obj_func_values = self.obj_func(self.pred_data, self.keys_opt, self.sim.true_order)
        obj_func_values = np.array(obj_func_values)

        # Finally, we calculate the ensemble sensitivity matrix.
        # First we need to perturb state and obj. func. ensemble with their mean. Note that, obj_func has shape (
        # ne,)!
        list_states = list(self.state.keys())
        aug_state = at.aug_state(self.state, list_states)
        pert_state = aug_state - np.dot(aug_state.mean(1)[:, None], np.ones((1, self.ne)))
        pert_obj_func = obj_func_values - np.array(self.obj_func_values)

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
            # state_en[statename] = chol.gen_real(self.state[statename], self.cov[i, i], self.ne)
            mean = self.state[statename]
            cov = cov_blocks[i]
            if ['nesterov'] in self.keys_opt['enopt']:
                if not isinstance(self.step, int):
                    stop = start + len(self.state[statename])
                    mean += self.beta * self.step[start:stop]
                    cov += self.beta * self.cov_step[start:stop, start:stop]
                    cov = self.get_sym_pos_semidef(cov)
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
            np.savez('debug_analysis_step_{0}'.format(str(iteration)), **save_dict)

