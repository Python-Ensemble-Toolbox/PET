# External imports
import numpy as np
import sys
import warnings

from copy import deepcopy

# Internal imports
from popt.misc_tools import optim_tools as ot
from pipt.misc_tools import analysis_tools as at
from popt.loop.ensemble_base import EnsembleOptimizationBaseClass

__all__ = ['GaussianEnsemble']

class GaussianEnsemble(EnsembleOptimizationBaseClass):
    """
    Class to store control states and evaluate objective functions.

    Methods
    -------
    get_state()
        Returns control vector as ndarray

    get_final_state(return_dict)
        Returns final control vector between [lb,ub]

    get_cov()
        Returns the ensemble covariance matrix

    function(x,*args)
        Objective function called during optimization

    gradient(x,*args)
        Ensemble gradient

    hessian(x,*args)
        Ensemble hessian

    calc_ensemble_weights(self,x,*args):
        Calculate weights used in sequential monte carlo optimization

    """

    def __init__(self, options, simulator, objective):
        """
        Parameters
        ----------
        keys_en : dict
            Options for the ensemble class

            - disable_tqdm: supress tqdm progress bar for clean output in the notebook
            - ne: number of perturbations used to compute the gradient
            - state: name of state variables passed to the .mako file
            - prior_<name>: the prior information the state variables, including mean, variance and variable limits
            - num_models: number of models (if robust optimization) (default 1)
            - transform: transform variables to [0,1] if true (default true)

        sim : callable
            The forward simulator (e.g. flow)

        obj_func : callable
            The objective function (e.g. npv)
        """

        # Initialize PETEnsemble
        super().__init__(options, simulator, objective)

        # Inflation factor used in SmcOpt
        self.inflation_factor = None
        self.survival_factor = None
        self.particles = []  # list in case of multilevel
        self.particle_values = []  # list in case of multilevel
        self.resample_index = None

        # Initialize variables for bias correction
        if 'bias_file' in self.sim.input_dict:  # use bias correction
            self.bias_file = self.sim.input_dict['bias_file'].upper()  # mako file for simulations
        else:
            self.bias_file = None
        self.bias_adaptive = None  # flag to adaptively update the bias correction (not implemented yet)
        self.bias_factors = None  # this is J(x_j,m_j)/J(x_j,m)
        self.bias_weights = np.ones(self.num_samples) / self.num_samples  # initialize with equal weights
        self.bias_points = None  # this is the points used to estimate the bias correction
    
    def get_final_state(self, return_dict=False):
        """
        Parameters
        ----------
        return_dict : bool
            Retrun dictionary if true

        Returns
        -------
        x : numpy.ndarray
            Control vector as ndarray, shape (number of controls, number of perturbations)
        """

        self._invert_scale_state()
        if return_dict:
            x = self.state
        else:
            x = self.get_state()
        return x
    
    def gradient(self, x, *args, **kwargs):
        '''
        Ensemble-based Gradient (EnOpt)

        Parameters
        ----------
        x : ndarray
            Control vector, shape (number of controls, )
        
        args : tuple
            Covarice ($C_x$), shape (number of controls, number of controls)
        
        Returns
        -------
        gradient : ndarray
            Ensemble gradient, shape (number of controls, )
        '''
        # Update state vector
        self.stateX = x

        # Set covariance equal to the input
        self.covX = args[0]

        # Generate state ensemble
        self.ne = self.num_samples
        nr = self._aux_input()   
        self.enX = np.random.multivariate_normal(self.stateX, self.covX, self.ne).T

        # Shift ensemble to have correct mean
        self.enX = self.enX - self.enX.mean(axis=1, keepdims=True) + self.stateX[:,None]

        # Truncate to bounds
        if (self.lb is not None) and (self.ub is not None):
            self.enX = np.clip(self.enX, self.lb[:, None], self.ub[:, None])

        # Evaluate objective function for ensemble
        self.enF = self.function(self.enX, *args, **kwargs)

        # Make function ensemble to a list (for Multilevel) 
        if not isinstance(self.enF, list):
            self.enF = [self.enF]

        # Define some variables for gradient calculation
        index = 0       
        nlevels = len(self.enF)
        grad_ml = np.zeros((nlevels, self.dimX))

        # Loop over levels (only one level if not multilevel)
        for id_level in range(nlevels):
            dF = self.enF[id_level] - np.repeat(self.stateF, nr)
            ne = self.enF[id_level].shape[0] 

            # Calculate ensemble gradient for level
            g = np.zeros(self.dimX)
            for n in range(ne):
                g = g + dF[n] * (self.enX[:, index+n] - self.stateX)

            grad_ml[id_level] = g/ne
            index += ne

        if 'multilevel' in self.keys_en:
            weight = ot.get_list_element(self.keys_en['multilevel'], 'cov_wgt')
            weight = np.array(weight)
            if not np.sum(weight) == 1.0:
                weight = weight / np.sum(weight)  
            grad = np.dot(grad_ml, weight)
        else:
            grad = grad_ml[0]

        # Check if natural or averaged gradient (default is natural)
        if not self.keys_en.get('natural_gradient', True):
            cov_inv = np.linalg.inv(self.covX)
            grad = np.matmul(cov_inv, grad)

        return grad

    def hessian(self, x=None, *args):
        r"""
        Calculate the hessian matrix associated with ensemble, defined as:

        $$ H = J(XX^T - \Sigma)/ (N_e-1) $$

        where $X$ and $J$ are ensemble matrices of $x$ (or control variables) and objective function
        perturbed by their respective means.

        !!! note
            state and ens_func_values are assumed to already exist from computation of the gradient.
            Save time by not running them again.

        Parameters
        ----------
        x : ndarray
            Control vector, shape (number of controls, number of perturbations)

        Returns
        -------
        hessian: numpy.ndarray
            The hessian evaluated at x, shape (number of controls, number of controls)

        References
        ----------
        Zhang, Y., Stordal, A.S. & Lorentzen, R.J. A natural Hessian approximation for ensemble based optimization.
        Comput Geosci 27, 355–364 (2023). https://doi.org/10.1007/s10596-022-10185-z
        """

        # Perturb state and function values with their mean
        state_ens = at.aug_state(self.state, list(self.state.keys()))
        pert_state = state_ens - np.dot(state_ens.mean(1)[:, None], np.ones((1, self.ne)))
        nr = self._aux_input()

        if not isinstance(self.ens_func_values,list):
            self.ens_func_values = [self.ens_func_values]
        start_index = 0
        level_hessian = []
        L = len(self.ens_func_values)
        hessian = np.zeros(self.cov.shape)
        for l in range(L):
            pert_obj_func = self.ens_func_values[l] - np.array(np.repeat(self.state_func_values, nr))
            ml_ne = self.ens_func_values[l].size
            
            # Calculate the gradient for mean and covariance matrix
            g_c = np.zeros(self.cov.shape)
            for i in np.arange(ml_ne):
                g_c = g_c + pert_obj_func[i] * (np.outer(pert_state[:, start_index + i], pert_state[:, start_index + i]) - self.cov)

            start_index += ml_ne
            level_hessian.append(g_c / (ml_ne - 1))

        if 'multilevel' in self.keys_en.keys():
            cov_wgt = ot.get_list_element(self.keys_en['multilevel'], 'cov_wgt')
            for l in range(L):
                hessian += level_hessian[l]*cov_wgt[l]
            hessian /= self.ne
        else:
            hessian = level_hessian[0]
            
        return hessian

    def calc_ensemble_weights(self, x, *args, **kwargs):
        r"""
        Calculate weights used in sequential monte carlo optimization.

        Parameters
        ----------
        x : ndarray
            Control vector, shape (number of controls, )

        args : tuple
            Inflation factor, covariance ($C_x$, shape (number of controls, number of controls)) and survival factor

        Returns
        -------
        sens_matrix, best_ens, best_func : tuple
                The weighted ensemble, the best ensemble member, and the best objective function value
        """

        # Set the ensemble state equal to the input control vector x
        self.state = ot.update_optim_state(x, self.state, list(self.state.keys()))

        # Set the inflation factor and covariance equal to the input
        self.inflation_factor = args[0]
        self.cov = args[1]
        self.survival_factor = args[2]

        # If bias correction is used we need to temporarily store the initial state
        initial_state = None
        if self.bias_file is not None and self.bias_factors is None:  # first iteration
            initial_state = deepcopy(self.state)  # store this to update current objective values

        # Generate ensemble of states
        if self.resample_index is None:
            self.ne = self.num_samples
        else:
            self.ne = int(np.round(self.num_samples*self.survival_factor))
        self._aux_input()
        self.state = self._gen_state_ensemble()

        state_ens = at.aug_state(self.state, list(self.state.keys()))
        self.function(state_ens, **kwargs)

        if not isinstance(self.ens_func_values, list):
            self.ens_func_values = [self.ens_func_values]
        L = len(self.ens_func_values)
        if self.resample_index is None:
            self.resample_index = [None]*L

        # If bias correction is used we need to calculate the bias factors, J(u_j,m_j)/J(u_j,m)
        if self.bias_file is not None:  # use bias corrections
            self._bias_factors(self.ens_func_values, initial_state)

        warnings.filterwarnings('ignore')  # suppress warnings
        start_index = 0
        level_sens = []
        sens_matrix = np.zeros(state_ens.shape[0])
        best_ens = 0
        best_func = 0
        ml_ne_new_total = 0
        if 'multilevel' in self.keys_en.keys():
            en_size = ot.get_list_element(self.keys_en['multilevel'], 'en_size')
        else:
            en_size = [self.num_samples]
        for l in range(L):

            ml_ne = en_size[l] 
            if L > 1 and l == L-1:
                ml_ne_new = int(np.round(self.num_samples*self.survival_factor)) - ml_ne_new_total
            else:
                ml_ne_new = int(np.round(ml_ne*self.survival_factor))  # new samples
                ml_ne_new_total += ml_ne_new
            ml_ne_surv = ml_ne - ml_ne_new  # surviving samples

            if self.resample_index[l] is None:
                self.particles.append(deepcopy(state_ens[:, start_index:start_index + ml_ne]))
                self.particle_values.append(deepcopy(self.ens_func_values[l]))
            else:
                self.particles[l][:, :ml_ne_surv] = self.particles[l][:, self.resample_index[l]]
                self.particles[l][:, ml_ne_surv:] = deepcopy(state_ens[:, start_index:start_index + ml_ne_new])
                self.particle_values[l][:ml_ne_surv] = self.particle_values[l][self.resample_index[l]]
                self.particle_values[l][ml_ne_surv:] = deepcopy(self.ens_func_values[l])

            # Calculate the weights and ensemble sensitivity matrix
            weights = np.zeros(ml_ne)
            for i in np.arange(ml_ne):
                weights[i] = np.exp(np.clip(-(self.particle_values[l][i] - np.min(
                    self.particle_values[l])) * self.inflation_factor, None, 10))

            weights = weights + 0.000001
            weights = weights/np.sum(weights)  # TODO: Sjekke at disse er riktig

            level_sens.append(self.particles[l] @ weights)
            if l == L-1:  # keep the best from the finest level
                index = np.argmin(self.particle_values[l])
                best_ens = self.particles[l][:, index]
                best_func = self.particle_values[l][index]
            self.resample_index[l] = np.random.choice(ml_ne,ml_ne_surv,replace=True,p=weights)

            start_index += ml_ne_new

        if 'multilevel' in self.keys_en.keys():
            cov_wgt = ot.get_list_element(self.keys_en['multilevel'], 'cov_wgt')
            for l in range(L):
                sens_matrix += level_sens[l]*cov_wgt[l]
            sens_matrix /= self.num_samples
        else:
            sens_matrix = level_sens[0]

        return sens_matrix, best_ens, best_func

    def _gen_state_ensemble(self):
        """
        Generate ensemble with the current state (control variable) as the mean and using the covariance matrix
        """

        state_en = {}
        cov_blocks = ot.corr2BlockDiagonal(self.state, self.cov)
        for i, statename in enumerate(self.state.keys()):
            mean = self.state[statename]
            cov = cov_blocks[i]
            temp_state_en = np.random.multivariate_normal(mean, cov, self.ne).transpose()
            shifted_ensemble = np.array([mean]).T + temp_state_en - np.array([np.mean(temp_state_en, 1)]).T
            if self.lb and self.ub:
                if self.transform:
                    np.clip(shifted_ensemble, 0, 1, out=shifted_ensemble)
                else:
                    np.clip(shifted_ensemble, self.lb[i], self.ub[i], out=shifted_ensemble)
            state_en[statename] = shifted_ensemble

        return state_en

    def _bias_correction(self, state):
        """
        Calculate bias correction. Currently, the bias correction is a constant independent of the state
        """
        if self.bias_factors is not None:
            return np.sum(self.bias_weights * self.bias_factors)
        else:
            return 1

    def _bias_factors(self, obj_func_values, initial_state):
        """
        Function for computing the bias factors
        """

        if self.bias_factors is None:  # first iteration
            currentfile = self.sim.file
            self.sim.file = self.bias_file
            self.ne = self.num_samples
            self.aux_input = list(np.arange(self.ne))
            self.calc_prediction()
            self.sim.file = currentfile
            bias_func_values = self.obj_func(self.pred_data, self.sim.input_dict, self.sim.true_order)
            bias_func_values = np.array(bias_func_values)
            self.bias_factors = bias_func_values / obj_func_values
            self.bias_points = deepcopy(self.state)
            self.state_func_values *= self._bias_correction(initial_state)
        elif self.bias_adaptive is not None and self.bias_adaptive > 0:  # update factors to account for new information
            pass  # not implemented yet




        
