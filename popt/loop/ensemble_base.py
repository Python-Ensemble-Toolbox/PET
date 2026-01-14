# External imports
import numpy as np
import pandas as pd
import sys

# Internal imports
from popt.misc_tools import optim_tools as ot
from ensemble.ensemble import Ensemble as SupEnsemble
from simulator.simple_models import noSimulation
from pipt.misc_tools.ensemble_tools import matrix_to_dict

__all__ = ['EnsembleOptimizationBaseClass']

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

        # Initialize state-related attributes
        self.stateX = np.array([]) # Current state vector, (nx,)
        self.stateF = None         # Function value(s) of current state
        self.bounds = []           # Bounds (untransformed) for each variable in stateX
        self.varX   = np.array([]) # Variance for state vector
        self.covX   = None         # Covariance matrix for state vector
        self.enX    = None         # Ensemble of state vectors ,(nx, ne)
        self.enF    = None         # Ensemble of function values, (ne, )
        self.lb     = np.array([]) # Lower bounds (transformed) for state vector, (nx,)
        self.ub     = np.array([]) # Upper bounds (transformed) for state vector, (nx,)

        # Intialize state information
        for key in self.prior_info.keys():

            # Extract prior information for this variable
            mean   = np.asarray(self.prior_info[key]['mean'])
            var    = self.prior_info[key]['variance']*np.ones(mean.size)
            lb, ub = self.prior_info[key].get('limits', (None, None))

            # Fill in state vector and index information    
            self.stateX = np.append(self.stateX, mean)
            self.idX[key] = (self.stateX.size - mean.size, self.stateX.size)

            # Set bounds and transform variance if applicable
            if self.transform and (lb is not None) and (ub is not None):
                var = var/(ub - lb)**2
                var = np.clip(var, 0, 1, out=var)
                self.bounds += mean.size*[(0, 1)]
            else:
                self.bounds.append((lb, ub))

            # Fill in lb and ub vectors
            self.lb = np.append(self.lb, lb*np.ones(mean.size))
            self.ub = np.append(self.ub, ub*np.ones(mean.size))

            # Fill in variance vector
            self.varX = np.append(self.varX, var)
            
        self.covX = np.diag(self.varX)  # Covariance matrix
        self.dimX = self.stateX.size    # Dimension of state vector
        
        # Scale state if applicable
        self.stateX = self.scale_state(self.stateX)

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
        if len(x.shape) == 1: 
            x = x[:,np.newaxis]
            self.ne = self.num_models
        else: self.ne = x.shape[1]

        # Run simulation
        x = self.invert_scale_state(x)
        x = self._reorganize_multilevel_ensemble(x)
        run_success = self.calc_prediction(enX=x, save_prediction=self.save_prediction)
        x = self._reorganize_multilevel_ensemble(x)
        x = self.scale_state(x).squeeze()

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

        if len(x.shape) == 1: 
            self.stateF = func_values
        else:
            self.enF = func_values 
        
        return func_values

    def get_state(self):
        """
        Returns
        -------
        x : numpy.ndarray
            Control vector as ndarray, shape (number of controls, number of perturbations)
        """
        return self.stateX
    
    def get_cov(self):
        """
        Returns
        -------
        cov : numpy.ndarray
            Covariance matrix, shape (number of controls, number of controls)
        """
        return self.covX

    def get_bounds(self):
        """
        Returns
        -------
        bounds : list
            (min, max) pairs for each element in x. None is used to specify no bound.
        """

        return self.bounds

    def scale_state(self, x):
        """
        Transform the internal state from [lb, ub] to [0, 1]

        Parameters
        ----------
        x : array_like
            The input state

        Returns
        -------
        x : array_like
            The scaled state
        """
        x = np.asarray(x)
        scaled_x = np.zeros_like(x)
        
        if self.transform is False:
            return x
        
        for i in range(len(x)):
            if (self.lb[i] is not None) and (self.ub[i] is not None):
                scaled_x[i] = (x[i] - self.lb[i]) / (self.ub[i] - self.lb[i])
            else:
                scaled_x[i] = x[i]  # No scaling if bounds are None
                
        return scaled_x

    def invert_scale_state(self, u):
        """
        Transform the internal state from [0, 1] to [lb, ub]

        Parameters
        ----------
        u : array_like
            The scaled state

        Returns
        -------
        x : array_like
            The unscaled state
        """
        u = np.asarray(u)
        x = np.zeros_like(u)

        if self.transform is False:
            return u

        for i in range(len(u)):
            if (self.lb[i] is not None) and (self.ub[i] is not None):
                x[i] = self.lb[i] + u[i] * (self.ub[i] - self.lb[i])
            else:
                x[i] = u[i]  # No scaling if bounds are None
                
        return x

    def save_stateX(self, path='./', filetype='npz'):
        '''
        Save the state vector.

        Parameters
        ----------
        path : str
            Path to save the state vector. Default is current directory.
        
        filetype : str
            File type to save the state vector. Options are 'csv', 'npz' or 'npy'. Default is 'npz'.
        '''
        if self.transform:
            stateX = self.invert_scale_state(self.stateX)
        else:
            stateX = self.stateX

        if filetype == 'csv':
            state_dict = matrix_to_dict(stateX, self.idX)
            state_df = pd.DataFrame(data=state_dict)
            state_df.to_csv(path + 'stateX.csv', index=False)
        elif filetype == 'npz':
            state_dict = matrix_to_dict(stateX, self.idX)
            np.savez_compressed(path + 'stateX.npz', **state_dict)
        elif filetype == 'npy':
            np.save(path + 'stateX.npy', stateX)
    
    def _reorganize_multilevel_ensemble(self, x):
        if ('multilevel' in self.keys_en) and (len(x.shape) > 1):  
            ml_ne = self.keys_en['multilevel']['ml_ne']
            x = ot.toggle_ml_state(x, ml_ne)
            return x


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