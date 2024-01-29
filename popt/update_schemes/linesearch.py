# External imports
import numpy as np
import time
import pprint

from numpy import linalg as la
from scipy.optimize import line_search

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.loop.optimize import Optimize
import popt.update_schemes.optimizers as opt


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
        
        # Set other optimization parameters
        self.alpha          = __set__variable('alpha', 0.25)
        self.alpha_iter_max = __set__variable('alpha_maxiter', 5)
        self.alpha_cov      = __set__variable('alpha_cov', 0.001)
        self.normalize      = __set__variable('normalize', True)
        self.max_resample   = __set__variable('resample', 0)
        self.normalize      = __set__variable('normalize', True)
        self.cov_factor     = __set__variable('cov_factor', 0.5)

        # Initialize line-search parameters (scipy defaults)
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

        # The EnOpt class self-ignites, and it is possible to send the EnOpt class as a callale method to scipy.minimize
        self.run_loop()  # run_loop resides in the Optimization class (super)
    
    def set_amax(self, xk, dk):
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
        improvement = False
        success = False
        resampling_iter = 0


        # Shrink covariance each time we try resampling
        shrink = self.cov_factor ** resampling_iter

        #compute gradient
        def _jac(x):
            g = self.jac(x, shrink*self.cov)
            return g/la.norm(g, np.inf)
        
        pk = _jac(self.mean_state) 
    
        # Compute the hessian
        hessian = self.hess()
        amax    = self.alpha
        if self.normalize:
            hessian /= np.maximum(la.norm(hessian, np.inf), 1e-12)  # scale the hessian with inf-norm
            

        # perform line search
        ls_results = line_search(f=lambda x: self.fun(x, shrink*self.cov).mean(), 
                                 myfprime=_jac, 
                                 xk=self.mean_state, 
                                 pk=-pk, 
                                 gfk=pk,
                                 old_fval=self.obj_func_values.mean(),
                                 c1=self.ls_options['c1'],
                                 c2=self.ls_options['c2'],
                                 amax=0.9*self.set_amax(self.mean_state, -pk),
                                 maxiter=self.alpha_iter_max)
        
        step_size, nfev, njev, fnew, fold, slope = ls_results
        
        if not step_size == None:

            # update state
            self.mean_state = self.mean_state - step_size*pk
            self.obj_func_values = fnew
            self.alpha = step_size

            # update evaluations
            self.nfev += nfev
            self.njev += njev

            # Update covariance
            self.cov = self.cov - self.alpha_cov * hessian
            self.cov = ot.get_sym_pos_semidef(self.cov)

            # update status
            improvement = True
            success = True
            self.optimize_result = ot.get_optimize_result(self)
            ot.save_optimize_results(self.optimize_result)
            self.iteration += 1

            # Write logging info
            if self.logger is not None:
                info_str_iter = '       {:<10} {:<10} {:<15.4e} {:<15.2e} {:<15.2e}'.\
                    format(self.iteration, 0, np.mean(self.obj_func_values),
                            self.alpha, self.cov[0, 0])
                self.logger.info(info_str_iter)

        return success


