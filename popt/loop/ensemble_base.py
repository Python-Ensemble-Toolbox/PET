# External imports
import numpy as np
import sys
import warnings

from copy import deepcopy

# Internal imports
from popt.misc_tools import optim_tools as ot
from pipt.misc_tools import analysis_tools as at
from ensemble.ensemble import Ensemble as SupEnsemble
from simulator.simple_models import noSimulation

class EnsembleOptimizationBaseClass(SupEnsemble):
    '''
    Base class for the popt ensemble
    '''
    def __init__(self, options, simulator, objective):
        '''
        Parameters
        ----------
        options : dict
            Options for the ensemble class
        
        simulator : callable
            The forward simulator (e.g. flow). If None, no simulation is performed.
        
        objective : callable
            The objective function (e.g. npv)
        '''
        if simulator is None:
            sim = noSimulation()
        else:
            sim = simulator

        # Initialize the PET Ensemble
        super().__init__(options, sim)

        # Unpack some options
        self.save_prediction = options.get('save_prediction', None)
        self.num_models      = options.get('num_models', 1)
        self.transform       = options.get('transform', False)
        self.num_samples     = self.ne

        # Set objective function (callable)
        self.obj_func = objective
        self.state_func_values = None
        self.ens_func_values = None

        # Initialize prior
        self._initialize_state_info() # Initialize cov, bounds, and state
        self._scale_state() # Scale self.state to [0, 1] if transform is True

    def _initialize_state_info(self):
        '''
        Initialize covariance and bounds based on prior information.
        '''
        self.cov = np.array([])
        self.lb = []
        self.ub = []
        self.bounds = []
        
        for key in self.prior_info.keys():
            variable = self.prior_info[key]
            
            # mean
            self.state[key] = np.asarray(variable['mean'])

            # Covariance
            dim = self.state[key].size
            var = variable['variance']*np.ones(dim)
        
            if 'limits' in variable.keys():
                lb, ub = variable['limits']
                self.lb.append(lb)
                self.ub.append(ub)
        
                # transform var to [0, 1] if transform is True
                if self.transform:
                    var = var/(ub - lb)**2
                    var = np.clip(var, 0, 1, out=var)
                    self.bounds += dim*[(0, 1)]
                else:
                    self.bounds += dim*[(lb, ub)]
            else:
                self.bounds += dim*[(None, None)]

            # Add to covariance
            self.cov = np.append(self.cov, var)
            self.dim = self.cov.shape[0]

        # Make cov full covariance matrix
        self.cov = np.diag(self.cov)
    
    def get_state(self):
        """
        Returns
        -------
        x : numpy.ndarray
            Control vector as ndarray, shape (number of controls, number of perturbations)
        """
        return ot.aug_optim_state(self.state, list(self.state.keys()))
    
    def get_cov(self):
        """
        Returns
        -------
        cov : numpy.ndarray
            Covariance matrix, shape (number of controls, number of controls)
        """
        return self.cov
    
    def vec_to_state(self, x):
        """
        Converts a control vector to the internal state representation.
        """
        return ot.update_optim_state(x, self.state, list(self.state.keys()))

    def get_bounds(self):
        """
        Returns
        -------
        bounds : list
            (min, max) pairs for each element in x. None is used to specify no bound.
        """

        return self.bounds
    
    def function(self, x, *args, **kwargs):
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

        # check for ensmble
        if len(x.shape) == 1: self.ne = self.num_models
        else: self.ne = x.shape[1]

        # convert x (nparray) to state (dict)
        self.state = self.vec_to_state(x)

        # run the simulation
        self._invert_scale_state()  # ensure that state is in [lb,ub]
        self._set_multilevel_state(self.state, x)  # set multilevel state if applicable
        run_success = self.calc_prediction(save_prediction=self.save_prediction)  # calculate flow data
        self._set_multilevel_state(self.state, x) # For some reason this has to be done again after calc_prediction
        self._scale_state()  # scale back to [0, 1]

        # Evaluate the objective function
        if run_success:
            func_values = self.obj_func(
                self.pred_data, 
                input_dict=self.sim.input_dict,
                true_order=self.sim.true_order, 
                **kwargs
            )
        else:
            func_values = np.inf  # the simulations have crashed

        if len(x.shape) == 1: self.state_func_values = func_values
        else: self.ens_func_values = func_values
        
        return func_values
    
    def _set_multilevel_state(self, state, x):
        if 'multilevel' in self.keys_en.keys() and len(x.shape) > 1:  
            en_size = ot.get_list_element(self.keys_en['multilevel'], 'en_size')
            self.state = ot.toggle_ml_state(self.state, en_size)


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
        if self.transform and (self.lb and self.ub):
            for i, key in enumerate(self.state):
                self.state[key] = (self.state[key] - self.lb[i])/(self.ub[i] - self.lb[i])
                np.clip(self.state[key], 0, 1, out=self.state[key])

    def _invert_scale_state(self):
        """
        Transform the internal state from [0, 1] to [lb, ub]
        """
        if self.transform and (self.lb and self.ub):
            for i, key in enumerate(self.state):
                if self.transform:
                    self.state[key] = self.lb[i] + self.state[key]*(self.ub[i] - self.lb[i])
                np.clip(self.state[key], self.lb[i], self.ub[i], out=self.state[key])