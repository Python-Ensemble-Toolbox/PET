# External imports
import numpy as np
import sys
import warnings

from copy import deepcopy

# Internal imports
from popt.misc_tools import optim_tools as ot
from pipt.misc_tools import analysis_tools as at
from ensemble.ensemble import Ensemble as PETEnsemble

class EnsembleOptimizationBase(PETEnsemble):
    '''
    Base class for the popt ensemble
    '''
    def __init__(self, kwargs_ens, sim, obj_func):
        '''
        Parameters
        ----------
        kwargs_ens : dict
            Options for the ensemble class
        
        sim : callable
            The forward simulator (e.g. flow)
        
        obj_func : callable
            The objective function (e.g. npv)
        '''

        # Initialize PETEnsemble
        super().__init__(kwargs_ens, sim)

        self.save_prediction = kwargs_ens.get('save_prediction', None)
        self.num_models  = kwargs_ens.get('num_models', 1)
        self.transform   = kwargs_ens.get('transform', True)
        self.num_samples = self.ne

        # Get bounds and varaince
        self.upper_bound = []
        self.lower_bound = []
        self.bounds = []
        self.cov = np.array([])
        for name in self.prior_info.keys():
            self.state[name] = np.asarray(self.prior_info[name]['mean'])
            num_state_var = len(self.state[name])
            value_cov = self.prior_info[name]['variance'] * np.ones((num_state_var,))
            if 'limits' in self.prior_info[name].keys():
                lb = self.prior_info[name]['limits'][0]
                ub = self.prior_info[name]['limits'][1]
                self.lower_bound.append(lb)
                self.upper_bound.append(ub)
                if self.transform:
                    value_cov = value_cov / (ub - lb)**2
                    np.clip(value_cov, 0, 1, out=value_cov)
                    self.bounds += num_state_var*[(0, 1)]
                else:
                    self.bounds += num_state_var*[(lb, ub)]
                self.cov = np.append(self.cov, value_cov)
            else:
                self.bounds += num_state_var*[(None, None)]
            
        
        self._scale_state()
        self.cov = np.diag(self.cov)

        # Set objective function (callable)
        self.obj_func = obj_func

        # Objective function values
        self.state_func_values = None
        self.ens_func_values = None
    
    def get_state(self):
        """
        Returns
        -------
        x : numpy.ndarray
            Control vector as ndarray, shape (number of controls, number of perturbations)
        """
        x = ot.aug_optim_state(self.state, list(self.state.keys()))
        return x

    def get_bounds(self):
        """
        Returns
        -------
        bounds : list
            (min, max) pairs for each element in x. None is used to specify no bound.
        """

        return self.bounds
    
    def function(self, x, *args):
        """
        This is the main function called during optimization.

        Parameters
        ----------
        x : ndarray
            Control vector, shape (number of controls, number of perturbations)

        Returns
        -------
        obj_func_values : numpy.ndarray
            Objective function values, shape (number of perturbations, )
        """
        self._aux_input()

        if len(x.shape) == 1:
            self.ne = self.num_models
        else:
            self.ne = x.shape[1]

        self.state = ot.update_optim_state(x, self.state, list(self.state.keys()))  # go from nparray to dict
        self._invert_scale_state()  # ensure that state is in [lb,ub]
        run_success = self.calc_prediction(save_prediction=self.save_prediction)  # calculate flow data
        self._scale_state()  # scale back to [0, 1]
        if run_success:
            func_values = self.obj_func(self.pred_data, self.sim.input_dict, self.sim.true_order)
        else:
            func_values = np.inf  # the simulations have crashed

        if len(x.shape) == 1:
            self.state_func_values = func_values
        else:
            self.ens_func_values = func_values
        
        return func_values

    def _aux_input(self):
        """
        Set the auxiliary input used for multiple geological realizations
        """

        nr = 1  # nr is the ratio of samples over models
        if self.num_models > 1:
            if np.remainder(self.num_samples, self.num_models) == 0:
                nr = int(self.num_samples / self.num_models)
                self.aux_input = list(np.repeat(np.arange(self.num_models), nr))
            else:
                print('num_samples must be a multiplum of num_models!')
                sys.exit(0)
        return nr

    def _scale_state(self):
        """
        Transform the internal state from [lb, ub] to [0, 1]
        """
        if self.transform and (self.upper_bound and self.lower_bound):
            for i, key in enumerate(self.state):
                self.state[key] = (self.state[key] - self.lower_bound[i])/(self.upper_bound[i] - self.lower_bound[i])
                np.clip(self.state[key], 0, 1, out=self.state[key])

    def _invert_scale_state(self):
        """
        Transform the internal state from [0, 1] to [lb, ub]
        """
        if self.transform and (self.upper_bound and self.lower_bound):
            for i, key in enumerate(self.state):
                if self.transform:
                    self.state[key] = self.lower_bound[i] + self.state[key]*(self.upper_bound[i] - self.lower_bound[i])
                np.clip(self.state[key], self.lower_bound[i], self.upper_bound[i], out=self.state[key])