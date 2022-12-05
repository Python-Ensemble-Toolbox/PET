# External imports
import numpy as np
from numpy import linalg as la

# Internal imports
from popt.misc_tools import optim_tools as ot, basic_tools as bt
from pipt.misc_tools import analysis_tools as at
from popt.loop.ensemble import Ensemble


class EnOpt(Ensemble):
    """
    This is an implementation of the steepest ascent ensemble optimization algorithm given in, e.g., Chen et al.,
    2009, 'Efficient Ensemble-Based Closed-Loop Production Optimization', SPE Journal, 14 (4): 634-645.
    The update of the control variable is done with the simple steepest (or gradient) ascent algorithm:

        x_l = (1 / alpha) * R * S + x_(l-1)

    where x is the control variable, l and l-1 is the current and previous iteration, alpha is a step limiter,
    R is a smoothing matrix (e.g., covariance matrix for x), and S is the ensemble gradient (or sensitivity).
    """

    def __init__(self, keys_opt, sim, obj_func):

        # Pass the init_file upwards in the hierarchy
        super().__init__(keys_opt, sim)

        # EnOPT parameters
        self._ext_enopt_param()
        self.step = 0  # Iteration step
        self.cov_step = 0

        # Load initial state (control variable)
        self._load_state()

        # Scale the initial state to [0, 1]
        self._scale_state()

        # Load information needed to calculate the sensitivity matrix
        self.cov = None
        self._load_sensitivity_info()

        # Get objective function
        self.obj_func = obj_func

        # Calculate objective function of startpoint
        self.run_ensemble()
        self.obj_func_values = self.obj_func(self.pred_data, self.keys_opt, self.sim.report)

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
        current_state = self.state

        # Calc sensitivity
        self.calc_ensemble_sensitivity()

        improvement = False
        success = False
        alpha_iter = 0
        alpha = self.alpha
        while improvement is False:

            # Augment state
            aug_state = ot.aug_optim_state(current_state, list_states)

            # Compute the steepest ascent step
            # Scale the gradient with 2-norm (or inf-norm: np.inf)

            new_step = alpha * self.sens_matrix / la.norm(self.sens_matrix, 2) + self.beta * self.step

            # Can set different stepsize for covariance update, e.g. change self.alpha to self.beta.
            # todo: add stepsize of covariance to the init.txt file, here we put 10 for now.
            self.cov_step = self.alpha_cov * self.cov_sens_matrix / la.norm(self.cov_sens_matrix, 2) + self.beta * self.cov_step

            # Calculate updated state
            aug_state_upd = aug_state + np.squeeze(new_step)

            self.cov = np.squeeze(self.cov + self.cov_step)
            self.cov = self.get_sym_pos_semidef(self.cov)

            # Make sure update is within bounds
            lowerBound, upperBound = self.keys_opt['scaling'][0], self.keys_opt['scaling'][1]
            np.clip(aug_state_upd, lowerBound, upperBound, out=aug_state_upd)

            # Calculate new objective function
            self.state = ot.update_optim_state(aug_state_upd, self.state, list_states)
            self.run_ensemble()
            new_func_values = self.obj_func(self.pred_data, self.keys_opt, self.sim.report)

            if np.mean(new_func_values) - np.mean(self.obj_func_values) > self.obj_func_tol:

                # Iteration was a success
                improvement = True
                success = True

                # Update objective function values and step
                self.obj_func_values = new_func_values
                self.step = new_step

                # Write logging info
                if logger is not None:
                    info_str_iter = '{:<10} {:<10} {:<10.2f} {:<10.2e} {:<10.2e}'.\
                        format(iteration, alpha_iter, np.mean(self.obj_func_values), alpha, self.cov[0, 0])
                    logger.info(info_str_iter)

            else:

                # If we do not have a reduction in the objective function, we reduce the step limiter
                if alpha_iter < self.alpha_iter_max:
                    # Decrease alpha
                    self.alpha /= 2
                    alpha_iter += 1
                else:
                    success = False
                    break

        # Save variables defined in ANALYSISDEBUG keyword.
        if 'analysisdebug' in self.keys_opt and success:

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

        return success

    # force matrix to positive semidefinite
    @staticmethod
    def get_sym_pos_semidef(a):

        rtol = 1e-05
        S,U = np.linalg.eigh(a)
        S=np.clip(S,0,None) + rtol
        a = (U*S)@U.T
        return a

    def _ext_enopt_param(self):
        """
        Extract ENOPT parameters in OPTIM part if inputted.

        YC 24/9-19
        """
        # Default value for max. iterations
        default_obj_func_tol = 1e-6
        default_step_tol = 1e-6
        default_alpha = 0.1
        default_alpha_cov = 0.001
        default_beta = 0.9
        default_alpha_iter_max = 5
        default_num_models = 1

        # Todo: check self.keys_opt, self.orig_lb & self.orig_ub are not used for now
        if 'origbounds' in self.keys_opt:
            if isinstance(self.keys_opt['origbounds'][0], list):
                origbounds = np.array(self.keys_opt['origbounds'])
            else:
                origbounds = np.array([self.keys_opt['origbounds']])
            self.orig_lb = origbounds[:, 0]
            self.orig_ub = origbounds[:, 1]

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
                self.num_models = enopt[ind_num_models[0]][ind_num_models[1] + 1]
            if self.num_models > 1:
                self.aux_input = list(np.arange(self.num_models))

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

        ST 3/5-18: First implementation. Much of the code here is taken directly from fwd_sim.ensemble Ensemble class.
        YC 2/10-19: Added calcuating gradient of covariance.
        """
        # Generate ensemble of states
        self.state = self._gen_state_ensemble()

        self.run_ensemble()
        obj_func_values = self.obj_func(self.pred_data, self.keys_opt, self.sim.report)
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
            g_m = g_m + pert_obj_func[i]*pert_state[:, i]
            g_c = g_c + pert_obj_func[i] * (np.outer(pert_state[:, i], pert_state[:, i]))

        self.cov_sens_matrix = g_c / (self.ne - 1)
        self.sens_matrix = g_m / (self.ne - 1)

    def _load_sensitivity_info(self):
        """
        Load  information on how to calculate the sensitvity matrix.

        ST 4/5-18
        """
        # METHOD - Check which method has been choosen.
        # If SENSITIVITY contains more than one line, we need to search for METHOD
        if isinstance(self.keys_opt['sensitivity'][0], list):
            # Get indices
            ind = bt.index2d(self.keys_opt['sensitivity'], 'method')

            # Check if METHOD has been inputted
            assert None not in ind, 'METHOD not found in keyword SENSITIVITY!'

            # Assign method name
            self.sens_method = self.keys_opt['sensitivity'][ind[0]][ind[1] + 1]

        # If SENSITIVITY contains only one line, this should be METHOD
        elif isinstance(self.keys_opt['sensitivity'], list):
            # Really check if METHOD has been inputted
            assert self.keys_opt['sensitivity'] == 'method', 'METHOD not found in keyword SENSITIVITY!'

            # Assign method name
            self.sens_method = self.keys_opt['sensitivity'][1]

        # If we have ensemble sensitivity matrix, we need to load the covariance matrix from which realizations of
        # the state is made
        if self.sens_method == 'ensemble':
            # Search for COVARIANCE in SENSITIVTY
            ind_cov = bt.index2d(self.keys_opt['sensitivity'], 'cov')

            # Check if COVARIANCE has been inputted
            assert None not in ind_cov, 'You have chosen ENSEMBLE as sensitivity method but COV has not been found!'

            value_cov = self.keys_opt['sensitivity'][ind_cov[0]][ind_cov[1] + 1]
            if isinstance(value_cov, str):
                # Load covariance matrix
                load_file = np.load(value_cov)
                self.cov = load_file[load_file.files[0]]
            else:
                # Augment state
                list_state = list(self.state.keys())
                aug_state = ot.aug_optim_state(self.state, list_state)
                self.cov = value_cov * np.eye(len(aug_state))
                # Loop over the statename to assign covariance values.
                # for i, statename in enumerate(self.state.keys()):
                #     self.cov[statename] = value_cov * np.eye(self.state[statename].shape[0])

                # Search for NE (no. of ensemble members)
            ind_ne = bt.index2d(self.keys_opt['sensitivity'], 'ne')

            # Check if NE has been inputted
            assert None not in ind_ne, 'You have chosen ENSEMBLE as sensitivity method but NE has not been found!'

            # Assign NE
            self.ne = int(self.keys_opt['sensitivity'][ind_ne[0]][ind_ne[1] + 1])

