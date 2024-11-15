"""Line Search."""
# External imports
import numpy as np
import time
import pprint
import warnings

from numpy import linalg as la
from scipy.optimize import line_search

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.loop.optimize import Optimize

# ignore line_search did not converge message
warnings.filterwarnings('ignore', message='The line search algorithm did not converge')


class LineSearch(Optimize):

    def __init__(self, fun, x, args, jac, hess, bounds=None, **options):

        # init PETEnsemble
        super(LineSearch, self).__init__(**options)

        def __set__variable(var_name=None, defalut=None):
            if var_name in options:
                return options[var_name]
            else:
                return defalut

        # Set input as class variables
        self.options    = options   # options
        self.fun        = fun       # objective function
        self.cov        = args[0]   # initial covariance
        self.jac        = jac       # gradient function
        self.hess       = hess      # hessian function
        self.bounds     = bounds    # parameter bounds
        self.mean_state = x         # initial mean state
        self.pk_from_ls = None
        
        # Set other optimization parameters
        self.alpha_iter_max = __set__variable('alpha_maxiter', 5)
        self.alpha_cov      = __set__variable('alpha_cov', 0.001)
        self.normalize      = __set__variable('normalize', True)
        self.max_resample   = __set__variable('resample', 0)
        self.normalize      = __set__variable('normalize', True)
        self.cov_factor     = __set__variable('cov_factor', 0.5)
        self.alpha          = 0.0

        # Initialize line-search parameters (scipy defaults for c1, and c2)
        self.alpha_max  = __set__variable('alpha_max', 1.0)
        self.ls_options = {'c1': __set__variable('c1', 0.0001),
                           'c2': __set__variable('c2', 0.9)}
        
        
        # Calculate objective function of startpoint
        if not self.restart:
            self.start_time = time.perf_counter()
            self.obj_func_values = self.fun(self.mean_state)
            self.nfev += 1
            self.optimize_result = ot.get_optimize_result(self)
            ot.save_optimize_results(self.optimize_result)
            if self.logger is not None:
                self.logger.info('       ====== Running optimization - EnOpt ======')
                self.logger.info('\n'+pprint.pformat(self.options))
                info_str = '       {:<10} {:<10} {:<15} {:<15} {:<15} '.format('iter', 'alpha_iter',
                                                                        'obj_func', 'step-size', 'cov[0,0]')
                self.logger.info(info_str)
                self.logger.info('       {:<21} {:<15.4e}'.format(self.iteration, np.mean(self.obj_func_values)))

        
        self.run_loop() 
    
    def set_amax(self, xk, dk):
        '''not used currently'''
        amax = np.zeros_like(xk)
        for i, xi in enumerate(xk):
            lower, upper = self.bounds[i]
            if np.sign(dk[i]) == 1:
                amax[i] = (upper-xi)/dk[i]
            else:
                amax[i] = (lower-xi)/dk[i]
        return np.min(amax)
                
    def calc_update(self):

        # Initialize variables for this step
        success = False

        # define dummy functions for scipy.line_search
        def _jac(x):
            self.njev += 1
            x = ot.clip_state(x, self.bounds) # ensure bounds are respected
            g = self.jac(x, self.cov)
            g = g/la.norm(g, np.inf) if self.normalize else g 
            return g
        
        def _fun(x):
            self.nfev += 1
            x = ot.clip_state(x, self.bounds) # ensure bounds are respected
            f = self.fun(x, self.cov).mean()
            return f

        #compute gradient. If a line_search is already done, the new grdient is alread returned as slope by the function
        if self.pk_from_ls is None:
            pk = _jac(self.mean_state)
        else:
            pk = self.pk_from_ls
    
        # Compute the hessian
        hessian = self.hess()
        if self.normalize:
            hessian /= np.maximum(la.norm(hessian, np.inf), 1e-12)  # scale the hessian with inf-norm
            

        # perform line search
        self.logger.info('Performing line search...')  
        ls_results = line_search(f=_fun, 
                                 myfprime=_jac, 
                                 xk=self.mean_state, 
                                 pk=-pk, 
                                 gfk=pk,
                                 old_fval=self.obj_func_values.mean(),
                                 c1=self.ls_options['c1'],
                                 c2=self.ls_options['c2'],
                                 amax=self.alpha_max,
                                 maxiter=self.alpha_iter_max)
        
        step_size, nfev, njev, fnew, fold, slope = ls_results
        
        if isinstance(step_size, float):
            self.logger.info('Strong Wolfie conditions satisfied')

            # update state
            self.mean_state      = ot.clip_state(self.mean_state - step_size*pk, self.bounds)
            self.obj_func_values = fnew
            self.alpha           = step_size
            self.pk_from_ls      = slope

            # Update covariance
            self.cov = self.cov - self.alpha_cov * hessian
            self.cov = ot.get_sym_pos_semidef(self.cov)

            # update status
            success = True
            self.optimize_result = ot.get_optimize_result(self)
            ot.save_optimize_results(self.optimize_result)

            # Write logging info
            if self.logger is not None:
                info_str_iter = '       {:<10} {:<10} {:<15.4e} {:<15.2e} {:<15.2e}'.\
                    format(self.iteration, 0, np.mean(self.obj_func_values),
                            self.alpha, self.cov[0, 0])
                self.logger.info(info_str_iter)

            # update iteration
            self.iteration += 1
        
        else:
            self.logger.info('Strong Wolfie conditions not satisfied!')
            success = False

        return success


