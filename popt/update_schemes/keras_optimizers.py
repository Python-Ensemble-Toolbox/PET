# turn off all tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import shutil
import numpy as np
import keras.optimizers

from tensorflow import Variable, clip_by_value

__all__ = ['OptimizerKeras']

def first_wolfie_condition(f_old, f_new, du, gradient, c):
        if f_new <= f_old + c*np.dot(du, gradient):
            return True
        else:
            return False
        
class OptimizerKeras:
    
    def __init__(self, optimizer='SGD', step_size=0.1, **kwargs):
        '''
        An optimization loop built around the optimizers from Keras.
        Backtrakcing with the first Wolfie condition is built in, and can be turned off

        Parameters
        ----------
        optimizer : str
            Name of the Keras optimizer to use. Defaults to SGD (Stochastic Gradient Descent)
        
        step_size : float
            Step-size passed to the optimizer (called learning rate in Keras).
            Defaults to 0.1.
        
        kwargs : 
            Keyword arguments passed to the opimizer. (Consult the Keras documentation)

        '''
        self.optimizer = getattr(keras.optimizers, optimizer)(learning_rate=step_size, **kwargs)
        self.step_size = step_size

        self.n_fev = 0
        self.n_gev = 0
    
    def run(self, u0, func, grad, args=(), max_iter=20, max_cuts=4, tol=0.0001, bounds=None, **kwargs):
        '''
        Performes the iterativ optimization.

        Parameters
        ----------
        u0 : ndarray
            Initial control vector
        
        func : callable
            Objective function
        
        grad : callable
            Gradient function
        
        args : tuple
            Args passed to func and grad
        
        max_iter : int
            Maximum number of iterations
        
        max_cuts : int
            Maximum number of step-size cuts for the backtracking. Defaults to 4.
            If max_cuts=0, optimization is run until max_iter is reached.

        tol : float
            Tolerance constant passed to first Wolfie condition
        
        bounds : list[tuple]
            (u_min, u_max) pairs for each element in control vector, u. None is used to specify no bound.
        
        kwargs : 
            Keyword arguments
                - save_name : Folder where info from ecah iterations is stored. Defaults to './iterations'
                - normalize : If True, the gradient is normalized with the inf-norm. Defaults to False
        
        Returns
        -------
        ndarray : Final control vector
        '''
        
        # Read **kwargs
        save_name = kwargs.get('save_name', './iterations')
        normalize = kwargs.get('normalize', False)

        # Check if line-seacrh (backtracking should be applied)
        if max_cuts == 0:
            line_search = False
        else: 
            line_search = True

        if bounds is None:
            bounds = [(-np.inf, np.inf) for _ in u0]

        # Overwrite existing save_folder
        if os.path.exists(save_name):
            shutil.rmtree(save_name)
        os.makedirs(save_name)
        
        # Some dummy functions
        #--------------------------------------------------------------------------------
        def _func(x):
            self.n_fev += 1
            return func(x, *args)

        def _grad(x):
            self.n_gev += 1
            g = grad(x, *args)
            g = g/np.linalg.norm(g, np.inf) if normalize else g
            return g
        
        def _truncate(x):           # lower bounds         # upper bounds
            return clip_by_value(x, np.array(bounds)[:,0], np.array(bounds)[:,1])
        #--------------------------------------------------------------------------------

        # Initial values
        u = Variable(u0, constraint=_truncate)  # tensorflow object
        f_current = _func(u0)                   # initial objective value

        # Save initial state
        np.savez(f'{save_name}/iter_0', 
                 u=u0, 
                 func=f_current, 
                 n_fev=self.n_fev, 
                 n_gev=self.n_gev)

        self.log_info(iter=0, func_val=f_current)

        # start loop
        for t in range(1, max_iter+1):
            u_current = u.numpy() # current iterate 
            
            # Calculate gradient and make step
            gradient = _grad(u_current)
            self.optimizer.apply_gradients(zip([gradient],[u]))

            # Get new values
            u_trial = u.numpy()
            u_delta = u_trial - u_current
            f_trial = _func(u_trial)

            # Apply backtracking (if applicable)
            n_cuts = 0
            sufficient_decrease = first_wolfie_condition(f_old=f_current, 
                                                         f_new=f_trial,
                                                         du=u_delta,
                                                         gradient=gradient, 
                                                         c=tol)
           
            while line_search and (not sufficient_decrease) and (n_cuts <= max_cuts):
                n_cuts += 1
                u_delta = u_delta/2.0
                u_trial = u_current + u_delta
                f_trial = _func(u_trial)

                # re-check sufficient decrease
                sufficient_decrease = first_wolfie_condition(f_old=f_current, 
                                                             f_new=f_trial,
                                                             du=u_delta,
                                                             gradient=gradient, 
                                                             c=tol)
            
            if n_cuts > max_cuts: # bactracking not succsessful
                u.assign(u_current)
                print('Optimization terminated: Backtracking failed')
                break
            else:
                f_current = f_trial
                u.assign(u_trial)

                # save stuff
                np.savez(f'{save_name}/iter_{t}', 
                         u=u.numpy(),       # control
                         func=f_current,    # function value 
                         n_fev=self.n_fev,  # number of function evaluations
                         n_gev=self.n_gev)  # number of gradient computations
                
                self.log_info(iter=t, func_val=f_current)
        
        return u.numpy() # return final control
    

    def log_info(self, iter, func_val):
        print('\n')
        print('Iteration: ', iter)
        print('Function value: ', np.round(func_val, 4))
        print('Function evaluations: ', self.n_fev)
        print('Gradient calculations: ', self.n_gev)
        print('\n')
