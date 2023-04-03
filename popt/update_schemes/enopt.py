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
        self.calc_prediction()
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
            new_step = alpha * self.sens_matrix / la.norm(self.sens_matrix, np.inf) + beta * self.step

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
                self.cov_step = self.alpha_cov * self.cov_sens_matrix / la.norm(self.cov_sens_matrix, np.inf) + \
                    beta * self.cov_step
                self.cov = np.squeeze(self.cov + self.cov_step)
                self.cov = self.get_sym_pos_semidef(self.cov)

                # Write logging info
                if logger is not None:
                    info_str_iter = '{:<10} {:<10} {:<10.2f} {:<10.2e} {:<10.2e}'.\
                        format(iteration, self.alpha_iter, np.mean(self.obj_func_values), alpha, self.cov[0, 0])
                    logger.info(info_str_iter)

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
        S, U = np.linalg.eigh(a)
        S = np.clip(S, 0, None) + rtol
        a = (U*S)@U.T
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
                self.num_models = enopt[ind_num_models[0]][ind_num_models[1] + 1]
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

        ST 3/5-18: First implementation. Much of the code here is taken directly from fwd_sim.ensemble Ensemble class.
        YC 2/10-19: Added calcuating gradient of covariance.
        """
        # Generate ensemble of states
        self.ne = self.num_samples
        self.state = self._gen_state_ensemble()
        self._invert_scale_state()
        
        self.calc_prediction()
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
            g_c = g_c + pert_obj_func[i] * (np.outer(pert_state[:, i], pert_state[:, i]))

        self.cov_sens_matrix = g_c / (self.ne - 1)
        self.sens_matrix = g_m / (self.ne - 1)

    def _gen_state_ensemble(self):
        """
        Generate an ensemble of states (control variables) to run in calc_ensemble_sensitivity. It is assumed that
        the covariance function needed to generate realizations has been inputted via the SENSITIVITY keyword (with
        METHOD option ENSEMBLE).

        ST 4/5-18
        """
        # TODO: Gen. realizations for control variables at separate time steps (cov is a block diagonal matrix),
        # and for more than one STATENAME

        # # Initialize Cholesky class
        # chol = decomp.Cholesky()
        #
        # # Augment state
        # list_state = list(self.state.keys())
        # aug_state = ot.aug_optim_state(self.state, list_state)

        # Generate ensemble with the current state (control variable) as the mean and using the imported covariance
        # matrix
        state_en = {}
        for i, statename in enumerate(self.state.keys()):
            # state_en[statename] = chol.gen_real(self.state[statename], self.cov[i, i], self.ne)
            mean = self.state[statename]
            len_state = len(self.state[statename])
            cov = self.cov[len_state * i:len_state * (i + 1), len_state * i:len_state * (i + 1)]
            if len(cov) != len(mean):  # make sure cov is diagonal matrix
                print('\033[1;31mERROR: Covariance must be diagonal matrix!\033[1;31m')
            #     cov = cov*np.identity(len(mean))
            if ['nesterov'] in self.keys_opt['enopt']:
                if not isinstance(self.step, int):
                    mean += self.beta * self.step[len_state * i:len_state * (i + 1)]
                    cov += self.beta * self.cov_step[len_state * i:len_state * (i + 1), len_state * i:len_state * (i + 1)]
                    cov = self.get_sym_pos_semidef(cov)
            temp_state_en = np.random.multivariate_normal(mean, cov, self.ne).transpose()
            if self.upper_bound and self.lower_bound:
                np.clip(temp_state_en, 0, 1, out=temp_state_en)

            state_en[statename] = np.array([mean]).T + temp_state_en - np.array([np.mean(temp_state_en,1)]).T

        return state_en

    def _scale_state(self):
        if self.upper_bound and self.lower_bound:
            for i, key in enumerate(self.state):
                self.state[key] = (self.state[key] - self.lower_bound[i]) / (self.upper_bound[i] - self.lower_bound[i])
                if np.min(self.state[key]) < 0:
                    self.state[key] = 0
                elif np.max(self.state[key]) > 1:
                    self.state[key] = 1

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

            # Save the variables
            np.savez(folder + '/debug_analysis_step_{0}'.format(str(iteration)), **save_dict)


#Some extra functions for GenOpt
#--------------------------------------------------------------------------------------
def sample_GaussianCopula(n, corr, marginals, return_Gaussian=False):
    '''
    Draws n i.i.d. samples from a Gaussian copula with given marginals.

    Parameters
    ----------------------------------------------------------
        n : int
            Number of samples to be drawn.
        
        corr : 2D-array_like, of shape (d, d)
            Correlation matrix of Gaussian Copula
        
        marginals : list[rv_continuous_frozen]
            List of marginal distributions 
            (rv_continuous_frozen: object from scipy.stats).
        
        return_Gaussian : bool
            If True, the Gaussian ensemble is returned also.
            Default is False
    
    Returns
    ----------------------------------------------------------
        out : ndarray, of shape (n, d)
            The ensemble drawn from the Copula
        
        out : ndarray, of shape (n, d)
            The Gaussian ensemble. Only returned if
            if return_Gaussian is True 
    '''
    d = len(marginals)
    Z = np.random.multivariate_normal(mean=np.zeros(d), cov=corr, size=n)
    U = np.zeros_like(Z)
    X = np.zeros_like(Z)

    for i in range(n): 
        U[i] = stats.norm.cdf(Z[i])
        X[i] = np.array([marginals[j].ppf(U[i, j]) for j in range(d)])
    
    if return_Gaussian: return X, Z
    else: return X

def fisher_beta(alpha, beta):
    '''
    Calculates the Fisher matrix of a Beta distribution.
    
    Parameters
    ----------------------------------------------
        alpha : float
            alpha parameter in Beta distribution 

        beta : float
            beta parameter in Beta distribution

    Returns
    ----------------------------------------------
        out : 2-D array_like, of shape (2, 2)
            Fisher matrix 
    '''
    a = alpha
    b = beta
    
    upper_row = [polygamma(1, a) - polygamma(1, a+b), -polygamma(1, a + b)]
    lower_row = [-polygamma(1, a + b), polygamma(1, b) - polygamma(1, a+b)]

    fisher = np.array([upper_row, lower_row])
    return fisher

def delA(a, b):
    '''
    Calculates the expression psi(a) - psi(a+b),
    where psi() is the digamma function.

    Parameters
    --------------------------------------------
        a : float
        b : float
    
    Returns
    --------------------------------------------
        out : float
    '''
    return digamma(a)-digamma(a+b)

#--------------------------------------------------------------------------------------

class CMA:

    def __init__(self, ne, dim, alpha_mu=None, n_mu=None, alpha_1=None, alpha_c=None, corr_update=False, equal_weights=True):
        '''
        This is a rather simple simple CMA class

        Parameters
        ----------------------------------------------------------------------------------------------------------
            ne : int
                Ensemble size
            
            dim : int
                Dimensions of control vector
            
            alpha_mu : float
                Learning rate for rank-mu update. If None, value proposed in [1] is used.
            
            n_mu : int, `n_mu < ne`
                Number of best samples of ne, to be used for rank-mu update.
                Default is int(ne/2).
            
            alpha_1 : float
                Learning rate fro rank-one update. If None, value proposed in [1] is used.
            
            alpha_c : float
                Parameter (inverse if backwards time horizen)for evolution path update 
                in the rank-one update. See [1] for more info. If None, value proposed in [1] is used.

            corr_update : bool
                If True, CMA is used to update a correlation matrix. Default is False.
            
            equal_weights : bool
                If True, all n_mu members are assign equal weighting, `w_i = 1/n_mu`.
                If False, the weighting scheme proposed in [1], where `w_i = log(n_mu + 1)-log(i)`,
                and normalized such that they sum to one. Defualt is True.

        References
        ----------------------------------------------------------------------------------------------------------
            [1] Hansen, N. (2006). The CMA evolution strategy: a comparing review. 
                In J. Lozano, P. Larranaga, I. Inza & E. Bengoetxea (ed.), Towards a new evolutionary computation. 
                Advances on estimation of distribution algorithms (pp. 75--102) . Springer .     
        '''
        self.alpha_mu       = alpha_mu
        self.n_mu           = n_mu
        self.alpha_1        = alpha_1
        self.alpha_c        = alpha_c
        self.ne             = ne
        self.dim            = dim
        self.evo_path       = 0
        self.corr_update    = corr_update

        #If None is given, default values are used
        if self.n_mu is None:
            self.n_mu = int(self.ne/2)
        
        if equal_weights:
            self.weights = np.ones(self.n_mu)/self.n_mu
        else:
            self.weights = np.array([np.log(self.n_mu + 1)-np.log(i+1) for i in range(self.n_mu)])
            self.weights = self.weights/np.sum(self.weights)

        self.mu_eff = 1/np.sum(self.weights**2)
        self.c_cov  = 1/self.mu_eff * 2/(dim+2**0.5)**2 +\
                    (1-1/self.mu_eff)*min(1, (2*self.mu_eff-1)/((dim+2)**2+self.mu_eff))

        if self.alpha_1 is None:
            self.alpha_1  = self.c_cov/self.mu_eff
        if self.alpha_mu is None:
            self.alpha_mu = self.c_cov*(1-1/self.mu_eff)
        if self.alpha_c is None:
            self.alpha_c  = 4/(dim+4)
        
    def _rank_mu_update(self, X, J):
        '''
        Performs the rank-mu update of the CMA-ES.
        '''
        index  = J.argsort()[::-1]
        Xs = (X[index[:self.n_mu]] - np.mean(X, axis=0)).T
        C_ = (Xs*self.weights)@Xs.T
        if self.corr_update: C_ = ot.cov2corr(C_)
        
        return C_

    def _rank_one_update(self, step):
        '''
        Perfors the rank-one update for the CMA-ES.
        '''
        s = self.alpha_c
        self.evo_path = (1-s)*self.evo_path + np.sqrt(s*(2-s)*self.mu_eff)*step
        C_ = np.outer(self.evo_path, self.evo_path)
        if self.corr_update: C_ = ot.cov2corr(C_)

        return C_
    
    def __call__(self, cov, step, X, J):
        '''
        Performs the CMA update.

        Parameters
        --------------------------------------------------
            cov : 2-D array_like, of shape (d, d)
                Current covariance or correlation matrix.
            
            step : 1-D array_like, of shape (d,) 
                New step of control vector.
                Used to update the evolution path.

            X : 2-D array_like, of shape (n, d)
                Control ensemble of size n.
            
            J : 2-D-array_like, of shape (n,)
                Objective ensemble of size n.
        
        Returns
        --------------------------------------------------
            out : 2-D array_like, of shape (d, d)
                CMA updated covariance (correlation) matrix.
        '''
        a_mu  = self.alpha_mu
        a_one = self.alpha_1 
        C_mu  = self._rank_mu_update(X, J)
        C_one = self._rank_one_update(step)
        cov   =  (1 - a_one - a_mu)*cov + a_one*C_one + a_mu*C_mu       
        return cov

class GenOpt(PETEnsemble):
    """
    This is an implementation of the steepest ascent ensemble optimization algorithm with beta
    distributions as marginals and a Gaussian Copula.

    Parameters in mako-file
    ------------------------------------------------------------------------------------
        epsilon : float
            Samling radius. Samples from Beta distribution lie within 0 to 1.
            These samples are then transfrormed such that Y = mu + 2*epsilon*(X-0.5),
            where Y is the new ensemble. Default value is 0.4.
        
        alpha_parameter : float
            Initial alpha parameter in Beta distribution. Default value is 20.
        
        beta_parameter : float
            Initial beta parameter in Beta distribution. Default value is 20.
        
        alpha_mu : float
            Learning rate for CMA-ES rank-mu update for correlation matrix.
        
        n_mu : int
            Mu number of samples used in the CMA-ES rank-mu update. 
            Default value is int(self.num_samples/2).
        
        alpha_one : float
            Learning rate for CMA-ES rank-one update for correlation matrix
    ------------------------------------------------------------------------------------
    """

    def __init__(self, keys_opt, keys_en, sim, obj_func):

        # init PETEnsemble
        super(GenOpt, self).__init__(keys_en, sim)
        
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

        self.corr    = np.identity(self.dim) #correlation matrix
        self.use_CMA = False

        if self.alpha_mu is not None:
            self.use_CMA = True
            self.cma_update = CMA(ne=self.ne,
                                dim=self.dim, 
                                alpha_mu=self.alpha_mu,
                                n_mu=self.n_mu, 
                                alpha_1=self.alpha_one, 
                                alpha_c=self.alpha_c, 
                                corr_update=True)

        # Get objective function
        self.obj_func = obj_func

        # Calculate objective function of startpoint
        self.ne = self.num_models
        self.calc_prediction()
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
            new_step = alpha * self.sens_matrix / la.norm(self.sens_matrix, np.inf) + beta * self.step
    
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

                #Update the distribution
                self.theta += self.alpha_theta*self.theta_sens_matrix#/np.linalg.norm(self.theta_sens_matrix, np.inf)
                self.marginals = [stats.beta(*self.theta[l]) for l in range(self.dim)]

                #Update correlation
                if self.use_CMA:
                    #CMA-ES Update
                    self.corr = self.cma_update(self.corr, new_step/alpha, self.Ze, self.Je)
                else:
                    #Mutation
                    self.calc_corr_sensitivity()
                    self.corr += self.alpha_corr*self.corr_sens_matrix/la.norm(self.corr_sens_matrix, np.inf)
        
                print('Max corr: ', np.max(self.corr-np.identity(self.dim)))
                print('Min corr: ', np.min(self.corr))
                
                # Write logging info
                if logger is not None:
                    info_str_iter = '{:<10} {:<10} {:<10.2f} {:<10.2e} {:<10.2f}, {:<10.2f}'.\
                        format(iteration, self.alpha_iter, np.mean(self.obj_func_values), alpha, self.theta[0,0], self.theta[0,1])
                    logger.info(info_str_iter)

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
        S, U = np.linalg.eigh(a)
        S = np.clip(S, 0, None) + rtol
        a = (U*S)@U.T
        return a
    
    @staticmethod
    def fisher_inv(theta):
        theta = np.squeeze(theta)
        alpha = theta[0]
        beta  = theta[1]

        f_inv = la.inv(fisher_beta(alpha, beta))
        return f_inv
    
    @staticmethod
    def delA(x):
        a = x[0]
        b = x[1]
        return digamma(a)-digamma(a+b)

    def _ext_enopt_param(self):
        """
        Extract ENOPT parameters in OPTIM part if inputted.
        """

        #Default values 
        default_obj_func_tol    = 1e-6
        default_step_tol        = 1e-6
        default_alpha           = 0.1
        default_alpha_theta     = 0.001
        default_alpha_cor       = 0.2
        default_beta            = 0.0
        default_alpha_iter_max  = 5
        default_num_models      = 1
        default_epsilon         = 0.4
        default_alpha_parameter = 20.0
        default_beta_parameter  = 20.0
        default_alpha_mu        = None
        default_alpha_one       = None
        default_alpha_c         = None
        default_n_mu            = None

        parameters = {  'obj_func_tol'      :   default_obj_func_tol,
                        'step_tol'          :   default_step_tol,
                        'alpha'             :   default_alpha,
                        'alpha_theta'       :   default_alpha_theta,
                        'alpha_corr'        :   default_alpha_cor,
                        'beta'              :   default_beta,
                        'alpha_iter_max'    :   default_alpha_iter_max,
                        'epsilon'           :   default_epsilon,
                        'alpha_parameter'   :   default_alpha_parameter,
                        'beta_parameter'    :   default_beta_parameter,
                        'alpha_mu'          :   default_alpha_mu,
                        'alpha_one'         :   default_alpha_one,
                        'alpah_c'           :   default_alpha_c,
                        'n_mu'              :   default_n_mu,
                        'num_models'        :   default_num_models      }

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

            #Extract EnOpt parameters
            for key in parameters.keys():
                value_index = bt.index2d(enopt, key)
                if not value_index is None:
                    parameters[key] = enopt[value_index[0]][value_index[1] + 1]

            #Set the parameters as class attributes
            self.obj_func_tol, self.step_tol, self.alpha,\
            self.alpha_theta, self.alpha_corr, self.beta, self.alpha_iter_max,\
            self.epsilon, self.alpha_parameter, self.beta_parameter,\
            self.alpha_mu, self.alpha_one, self.alpha_c, self.n_mu, \
            self.num_models = tuple(parameters.values())

            if self.n_mu is not None:
                self.n_mu = int(self.n_mu) #Extracted as float for some reason

            #Extract prior information 
            value_cov = np.array([])
            for name in self.prior_info.keys():
                self.state[name] = self.prior_info[name]['mean']
                value_cov = np.append(value_cov, self.prior_info[name]['variance'] * np.ones((len(self.state[name]),)))
                if 'limits' in self.prior_info[name].keys():
                    self.lower_bound.append(self.prior_info[name]['limits'][0])
                    self.upper_bound.append(self.prior_info[name]['limits'][1])
            
            #Augment state
            list_state = list(self.state.keys())
            aug_state = ot.aug_optim_state(self.state, list_state)
            self.cov = value_cov * np.eye(len(aug_state))

            #Multivariate distribution
            theta0          = self.alpha_parameter, self.beta_parameter
            self.dim        = sum([len(val) for val in self.state.values()])    #Dimension of the control
            self.theta      = np.row_stack([theta0 for _ in range(self.dim)])   #Inital thetas
            self.marginals  = [stats.beta(*theta0) for _ in range(self.dim)]    #Inital marginals

    def calc_ensemble_sensitivity(self):
        """
        Calculates the gradient of state and of theta using the Stein's lemma.
        """
        # Generate ensemble of states
        self.ne = self.num_samples

        #Vanilla_state is the un-transformed ensemble
        self.state, vanilla_state = self._gen_state_ensemble()
        
        #Invert scaling of self.state
        self._invert_scale_state()
        self.calc_prediction()

        #Get objective ensemble
        obj_func_values = self.obj_func(self.pred_data, self.keys_opt, self.sim.true_order)
        obj_func_values = np.array(obj_func_values)

        #Finally, we calculate the ensemble sensitivity matrix.
        #First we need to perturb the obj. func. Note that, obj_func has shape (ne,)!
        list_states   = list(self.state.keys())
        #aug_state     = at.aug_state(vanilla_state, list_states)
        pert_obj_func = obj_func_values - np.array(self.obj_func_values)
        
        #Save ensembles for corrupdate update
        self.Xe = at.aug_state(self.state, list_states).T
        self.Je = obj_func_values
        self.Je_pert = pert_obj_func

        #Re-defines some Ensemble matrecies
        JX    = pert_obj_func[:, None]  #shape (Ne,1)
        X     = vanilla_state.T         #shape (d, Ne)
        Z     = self.Ze.T               #shape (d, Ne)
        theta = self.theta              #shape (d, 2) 
        F_inv = np.apply_along_axis(self.fisher_inv, axis=1, arr=self.theta) #shape (d, 2, 2) 

        #Finally, we calculate the ensemble sensitivity matrix
        #Gradient of X----------------------------------------------------------------------------
        alphas = (theta[:,0])[:,None]
        betas  = (theta[:,1])[:,None]

        H = la.inv(self.corr) - np.identity(self.dim)                           #shape (d, d)
        M = np.array([self.marginals[d].pdf(X[d]) for d in range(self.dim)])    #shape (d, Ne)
        N = np.apply_along_axis(stats.norm.pdf, axis=1, arr=Z)                  #shape (d, Ne)

        ensembleX = (alphas-1)/X - (betas-1)/(1-X) - (H@Z)*M/N                  #shape (d, Ne)
        gradX     = np.squeeze(-ensembleX@JX)/((self.ne-1)*2*self.epsilon)      #shape (d,)
        #-----------------------------------------------------------------------------------------

        #Gradeint of Theta------------------------------------------------------------------------
        log_term = np.array([np.log(X), np.log(1-X)])
        psi_term = np.array([np.apply_along_axis(self.delA, axis=1, arr=theta),
                             np.apply_along_axis(self.delA, axis=1, arr=np.flip(theta, axis=1))])

        ensembleTheta = log_term-psi_term[:,:,None]
        gradTheta     = np.sum(ensembleTheta*JX.T, axis=-1)
        gradTheta     = np.einsum('kij,jk->ki', F_inv, gradTheta, optimize=True)/(self.ne-1)
        #-----------------------------------------------------------------------------------------
    
        self.theta_sens_matrix  = gradTheta
        self.sens_matrix        = gradX
  
    def calc_corr_sensitivity(self):
        ne = self.num_samples
        Z  = self.Ze
        J  = self.Je_pert
        gradCorr = np.zeros_like(self.corr)

        for n in range(ne):
            gradCorr += J[n]*(np.outer(Z[n], Z[n]) - self.corr)
        
        np.fill_diagonal(gradCorr, 0)
        self.corr_sens_matrix = gradCorr/(ne-1)

    def _gen_state_ensemble(self):
        """
        Generate an ensemble of states (control variables) to run in calc_ensemble_sensitivity.
        This is done using the Coupla object. A transfromation is done on the ensemble such that samples
        lie within [state - epsilon, state + epsilon]. 

        Returns
        ---------------------------------------
            staten_en : dict
                Transformed state ensemble
            
            vanilla_en : 2D-array_like, of shape (ne, d)
                Un-transformed state ensemble
        ---------------------------------------
        """
        state_en   = {}     #Transfromed ensemble
        eps = self.epsilon
        
        #Xe is the ensemble and self.Ze is the Gaussian ensemble
        self.corr = scipy.linalg.block_diag(*ot.corr2BlockDiagonal(self.state, self.corr))
        Xe, self.Ze = sample_GaussianCopula(self.ne, self.corr, self.marginals, return_Gaussian=True)
        
        d = 0
        for statename in list(self.state.keys()):

            statename_en = []
            for x in self.state[statename]:                     #Loop over components   

                a = (x-eps) - ( (x-eps)*(x-eps < 0) ) \
                            - ( (x+eps-1)*(x+eps > 1) ) \
                            + (x+eps-1)*(x-eps < 0)*(x+eps > 1) #Lower bound of ensemble
                
                b = (x+eps) - ( (x-eps)*(x-eps < 0) ) \
                            - ( (x+eps-1)*(x+eps > 1) ) \
                            + (x-eps)*(x-eps < 0)*(x+eps > 1)   #Upper bound of ensemble
        
                statename_en.append(a + Xe[:, d]*(b-a))         #Component-wise trafo.
                d += 1

            state_en[statename] = np.array(statename_en)

        return state_en, Xe

    def _scale_state(self):
        if self.upper_bound and self.lower_bound:
            for i, key in enumerate(self.state):
                self.state[key] = (self.state[key] - self.lower_bound[i]) / (self.upper_bound[i] - self.lower_bound[i])
                if np.min(self.state[key]) < 0:
                    self.state[key] = 0
                elif np.max(self.state[key]) > 1:
                    self.state[key] = 1

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
                if not os.path.exists(folder):
                    os.mkdir(folder)
            else:
                folder = './'

            np.savez(folder + '/debug_analysis_step_{0}'.format(str(iteration)), **save_dict)
    
        
