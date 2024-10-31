# External imports
import numpy as np
import time
import pprint
import warnings

from numpy import linalg as la
#from scipy.optimize import line_search

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.loop.optimize import Optimize

# ignore line_search did not converge message
warnings.filterwarnings('ignore', message='The line search algorithm did not converge')


class LineSearch(Optimize):

    def __init__(self, fun, x, args, jac, hess, bounds=None, **options):

        # init PETEnsemble
        super(LineSearch, self).__init__(**options)

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
        self.alpha_iter_max  = options.get('alpha_maxiter', 5)
        self.alpha_cov       = options.get('alpha_cov', 0.01)
        self.normalize       = options.get('normalize', True)
        self.iter_resamp_max = options.get('resample', 0)
        self.shrink_factor   = options.get('shrink_factor', 0.25)
        self.alpha           = 0.0

        # Initialize line-search parameters (scipy defaults for c1, and c2)
        self.alpha_max  = options.get('alpha_max', 1.0)
        self.ls_options = {'c1': options.get('c1', 0.0001),
                           'c2': options.get('c2', 0.9)}
        
        
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
                
    def calc_update(self, iter_resamp=0):

        # Initialize variables for this step
        success = False

        # Dummy functions for scipy.line_search
        def _jac(x):
            self.njev += 1
            x = ot.clip_state(x, self.bounds) # ensure bounds are respected
            g = self.jac(x, self.cov)
            return g
        
        def _fun(x):
            self.nfev += 1
            x = ot.clip_state(x, self.bounds) # ensure bounds are respected
            f = self.fun(x, self.cov).mean()
            return f

        #Compute gradient. If a line_search is already done, the new grdient is alread returned as slope by the function
        if self.pk_from_ls is None:
            pk = _jac(self.mean_state)
        else:
            pk = self.pk_from_ls
    
        # Compute the hessian (Not Used Currently)
        hessian = self.hess()

        if self.normalize:
            hessian /= np.maximum(la.norm(hessian, np.inf), 1e-12)  # scale the hessian with inf-norm
            pk_norm = la.norm(pk, np.inf)
        else:
            pk_norm = 1
            
        # Perform Line Search
        self.logger.info('Performing line search...')
        ls_kw = {'fun': _fun,
                 'grad': _jac,
                 'amax': self.alpha_max,
                 'xk': self.mean_state,
                 'pk': -pk/pk_norm,
                 'fk': self.obj_func_values.mean(),
                 'gk': pk,
                 'c1': self.ls_options['c1'],
                 'c2': self.ls_options['c2'],
                 'maxiter': self.alpha_iter_max,
                 'logger': self.logger}
        
        step_size, fnew, gnew, alpha_iter = line_search_step(**ls_kw)
        
        if isinstance(step_size, float):
            self.logger.info('Strong Wolfie conditions satisfied')

            # Update state
            self.mean_state      = ot.clip_state(self.mean_state - step_size*pk/pk_norm, self.bounds)
            self.obj_func_values = fnew
            self.alpha           = step_size
            self.pk_from_ls      = gnew

            # Update covariance 
            #TODO: This sould be mande into an callback function for generality 
            #      (in case of non ensemble gradients or GenOpt gradient)
            self.cov = self.cov - self.alpha_cov * hessian
            self.cov = ot.get_sym_pos_semidef(self.cov)

            # Update status
            success = True
            self.optimize_result = ot.get_optimize_result(self)
            ot.save_optimize_results(self.optimize_result)

            # Write logging info
            if self.logger is not None:
                info_str_iter = '       {:<10} {:<10} {:<15.4e} {:<15.2e} {:<15.2e}'.\
                    format(self.iteration, alpha_iter, np.mean(self.obj_func_values),
                            self.alpha, self.cov[0, 0])
                self.logger.info(info_str_iter)

            # Update iteration
            self.iteration += 1
        
        else:
            self.logger.info('Strong Wolfie conditions not satisfied')

            if iter_resamp < self.iter_resamp_max:

                self.logger.info('Resampling Gradient')
                iter_resamp += 1
                self.pk_from_ls = None

                # Shrink cov matrix
                self.cov = self.cov*self.shrink_factor

                # Recursivly call function
                success = self.calc_update(iter_resamp=iter_resamp)

            else:
                success = False
    
        return success


def line_search_step(fun, grad, amax, xk, pk, fk=None, gk=None, args=(), c1=0.0001, c2=0.9, maxiter=5, **kwargs):
    '''
    Perform a line search to find an acceptable step size that satisfies the Armijo and curvature conditions.

    Parameters
    -----------------------------------------------------------------------------------------------------------------------------
    fun : callable
        The objective function to be minimized.
    grad : callable
        The gradient of the objective function.
    amax : float
        The maximum step size.
    xk : ndarray
        The current point.
    pk : ndarray
        The search direction.
    fk : float, optional
        The value of the objective function at xk. If None, it will be computed.
    gk : ndarray, optional
        The gradient of the objective function at xk. If None, it will be computed.
    args : tuple, optional
        Additional arguments passed to the objective function and its gradient.
    c1 : float, optional
        The Armijo condition constant. Default is 0.0001.
    c2 : float, optional
        The curvature condition constant. Default is 0.9. If None, the curvature condition is ignored.
    maxiter : int, optional
        The maximum number of iterations. Default is 5.
    **kwargs : dict, optional
        Additional keyword arguments. 'rho' can be used to specify the step size reduction factor. Default is 0.5.

    Returns
    -----------------------------------------------------------------------------------------------------------------------------
    step_size : float
        The step size that satisfies the conditions.
    fnew : float
        The value of the objective function at the new point.
    gnew : ndarray
        The gradient of the objective function at the new point.
    i : int
        The number of iterations performed.

    Notes
    -----------------------------------------------------------------------------------------------------------------------------
    The function performs a backtracking line search to find a step size that satisfies both the Armijo and curvature conditions.
    If the conditions are not satisfied within the maximum number of iterations, the function returns None for the step size.
    If c2 is None, the curvature condition is ignored, and only the Armijo condition is checked.
    '''
    # Compute the function value and gradient at the current point if not provided
    if fk is None:
        fk = fun(xk, *args)
    if gk is None:
        gk = grad(xk, *args)

    # Assert that the constants c1 and c2 are within valid ranges
    assert 0.0 <= c1, '0 <= c1'
    if c2 is not None: 
        assert 0.0 <= c2 < 1.0, '0 <= c2 < 1'

    # Check for logger
    logger = kwargs.get('logger', None)

    # Initialize conditions and step size
    step_size = amax
    fnew = None
    gnew = None
    rho  = kwargs.get('rho', 0.5)  # Step size reduction factor

    # Perform the line search
    for i in range(maxiter+1):   
        armijo = False
        curvature = False

        # Compute the function value at the new point
        fnew = fun(xk + step_size*pk)

        # Check the Armijo condition
        if fnew <= fk + c1*step_size*np.dot(pk, gk):
            armijo = True
            _log(logger, message='Armijo condition: Satisfied')

            # Check the Curvature condition if c2 is not None
            if c2 is None:
                curvature = True
            else:
                gnew = grad(xk + step_size*pk)  
                if np.abs(np.dot(pk, gnew)) <= c2*np.abs(np.dot(pk, gk)):
                    curvature = True
                    _log(logger, message='Curvature condition: Satisfied')
        
        # If both conditions are satisfied, return the step size and new values
        if armijo and curvature:
            return step_size, fnew, gnew, i
        else:
            if not armijo:
                _log(logger, message='Armijo condition: Failed')
            if not curvature and c2 is not None:
                _log(logger, message='Curvature condition: Failed')

            # Reduce the step size
            step_size = rho*step_size

    # If the line search did not converge, print a message and return NaN
    _log(logger, message='Line Search did not converge')
    
    return None, fnew, gnew, i

def _log(logger, message):
    if logger is None:
        print(message)
    else:
        logger.info(message)







    
