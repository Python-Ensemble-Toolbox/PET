# External imports
import numpy as np
import time
import pprint
import warnings

from numpy import linalg as la
from functools import cache
from scipy.optimize import OptimizeResult
from scipy.optimize._dcsrch import DCSRCH
from scipy.optimize._linesearch import _zoom

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.loop.optimize import Optimize
from popt.update_schemes.line_search_step import line_search, line_search_backtracking

# some symbols for logger
subk = '\u2096'
sup2 = '\u00b2'
jac_inf_symbol = f'‖jac(x{subk})‖\u221E'
fun_xk_symbol  = f'fun(x{subk})'
nabla_symbol = "\u2207"


def LineSearch(fun, x, jac, method='GD', hess=None, args=(), bounds=None, callback=None, **options):
    '''
    A Line Search Optimizer.

    Parameters
    ----------
    fun: callable
        Objective function, fun(x, *args).
    
    x: ndarray
        Initial control vector.

    jac: callable
        Jacobian/Gradient function, jac(x, *args)
    
    method: str
        Which optimization method to use. Default is 'GD' for 'Gradient Descent'.
        Other options are 'BFGS' for the 'Broyden–Fletcher–Goldfarb–Shanno' method,
        and 'Newton-CG'.  
    
    hess: callable, optional
        Hessian function, hess(x, *args). Default is None. 
    
    args: tuple, optional
        Args passed to fun, jac and hess.

    bounds: list, optional
        (min, max) pairs for each element in x. None is used to specify no bound.
    
    callback: callable, optional
        A callable called after each successful iteration. The class instance of LineSearch 
        is passed as the only argument to the callback function: callback(self) 
    
    **options: 
        keyword arguments, optional

    
    LineSearch Options (**options)
    ------------------------------
    - maxiter: int,
        Maximum number of iterations. Default is 20.
    
    - lsmaxiter: int,
        Maximum number of iterations for the line search. Default is 10.
    
    - step_size: float,
        Step-size for optimizer. Default is 0.25/inf-norm(jac(x0)).

    - step_size_max: float,
        Maximum step-size. Default is 1e5. If bounds are specified, 
        the maximum step-size is set to the maximum step-size allowed by the bounds.

    - step_size_adapt: int,
        Set method for choosing initial step-size for each iteration. If 0, step_size value is used.
        If 1, Equation (3.6) from "Numercal Optimization" [1] is used. If 2, the equation above Equation (3.6) is used. 
        Default is 0.
                    
    - c1: float,
        Tolerance parameter for the Armijo condition. Default is 1e-4.

    - c2: float,
        Tolerance parameter for the Curvature condition. Default is 0.9.

    - xtol: float,
        Optimization stop whenever |dx|<xtol. Default is 1e-8. 
    
    - ftol: float,
        Optimization stop whenever |f_new - f_old| < ftol * |f_old|. Default is 1e-4.
    
    - gtol: float,
        Optimization stop whenever ||jac||_inf < gtol. Default is 1e-5.

    - lsmethod: int,
        Sets method for proposing new step-size in the line search.  Default is 1.
        If lsmethod=0: backtracking is used (step-size is cut in half).
        If lsmethod=1: line search Algorithm (3.5) from [1] with polynomial interpolation is used.
    
    - convergence_criteria: callable,
        A callable that takes the current optimization object as an argument and returns True if the optimization should stop.
        It can be used to implement custom convergence criteria. Default is None.

    - saveit: bool,
        If True, the results from each iteration is saved. Default is True.

    - save_folder: str,
        Name of folder to save the results to. Defaul is ./ (the current directory).

    - fun0: float,
        Function value of the intial control.

    - jac0: ndarray,
        Jacobian of the initial control.
    
    - hess0: ndarray,
        Hessian value of the initial control.

    - hess0_inv: ndarray, 
        Initial inverse of hessian (only if method = 'BFGS').

    - resample: int,
        Number of jacobian re-computations allowed if a line search fails. Default is 0.
        (useful if jacobian is stochastic)

    - savedata: list[str],
        Further specification of which class variables to save to the result files.

    - restart: bool,
        Restart optimization from a restart file. Default is False

    - restartsave: bool,
        Save a restart file after each successful iteration. Default is False
    
                
    Returns
    -------
    res: OptimizeResult
        Important attributes:
        - x: optimized control
        - fun: objective function value
        - nfev: number of function evaluations
        - njev: number of jacobian evaluations

            
    References
    ----------
    [1] Nocedal and Wright: "Numerical Optimization", pp. 30–65. Springer, New York
        (2006). https://doi.org/10.1007/978-0-387-40065-5_3 

    
    Example use
    -----------
    >>> import numpy as np
    >>> from scipy.optimize import rosen, rosen_der
    >>> from popt.update_schemes.linesearch import LineSearch
    >>> x0 = np.random.uniform(-3, 3, 2)
    >>> kwargs = {'maxiter': 100,
                  'lsmaxiter': 10,
                  'step_size_adapt': 1,
                  'saveit': False}
    >>> res = LineSearch(fun=rosen, x=x0, jac=rosen_der, method='BFGS', **kwargs)
    >>> print(res)
    '''
    ls_obj = LineSearchClass(
        fun, 
        x, 
        jac, 
        method, 
        hess, 
        args, 
        bounds, 
        callback, 
        **options
    )
    return ls_obj.optimize_result


class LineSearchClass(Optimize):

    def  __init__(self, fun, x, jac, method='GD', hess=None, args=(), bounds=None, callback=None, **options):

        # init PETEnsemble
        super(LineSearchClass, self).__init__(**options)

        # Set input as class variables
        self.function = fun      
        self.xk       = x      
        self.jacobian = jac     
        self.method   = method  
        self.hessian  = hess     
        self.args     = args
        self.bounds   = bounds  
        self.options  = options 

        # Check for Callback function
        if callable(callback):
            self.callback = callback
        else:
            self.callback = None

        # Custom convergence criteria (callable)
        convergence_criteria = options.get('convergence_criteria', None)
        if callable(convergence_criteria):
            self.convergence_criteria = self.convergence_criteria
        else:
            self.convergence_criteria = None
        
        # Set options for step-size
        self.step_size       = options.get('step_size', None)
        self.step_size_max   = options.get('step_size_max', 1e5)
        self.step_size_adapt = options.get('step_size_adapt', 0)

        # Set options for line-search 
        self.lskwargs = {
            'c1': options.get('c1', 1e-4),
            'c2': options.get('c2', 0.9),
            'rho': options.get('rho', 0.5),
            'amax': self.step_size_max,
            'maxiter': options.get('lsmaxiter', 10),
            'method' : options.get('lsmethod', 1)
        }

        # Set other options
        self.normalize = options.get('normalize', False)
        self.resample  = options.get('resample', 0)
        self.saveit    = options.get('saveit', True)

        # set tolerance for convergence
        self.xtol = options.get('xtol', 1e-8) # tolerance for control vector
        self.ftol = options.get('ftol', 1e-4) # relative tolerance for function value
        self.gtol = options.get('gtol', 1e-5) # tolerance for inf-norm of jacobian

        # Check method
        valid_methods = ['GD', 'BFGS', 'Newton-CG']
        if not self.method in valid_methods:
            raise ValueError(f"'{self.method}' is not a valid method. Valid methods are: {valid_methods}")
        
        if (self.method == 'Newton-CG') and (self.hessian is None):
            print(f'Warning: No hessian function provided. Finite difference approximation is used: {nabla_symbol}{sup2}f(x{subk})d ≈ ({nabla_symbol}f(x{subk}+hd)-{nabla_symbol}f(x{subk}))/h')

        # Calculate objective function of startpoint
        if not self.restart:
            self.start_time = time.perf_counter()

            # Check for initial callable values
            self.fk = options.get('fun0', None)
            self.jk = options.get('jac0', None)
            self.Hk = options.get('hess0', None)

            if self.fk is None: self.fk = self._fun(self.xk)
            if self.jk is None: self.jk = self._jac(self.xk)
            if self.Hk is None: self.Hk = self._hess(self.xk)

            # Check for initial inverse hessian for the BFGS method
            if self.method == 'BFGS':
                self.Hk_inv = options.get('hess0_inv', np.eye(x.size))
            else:
                self.Hk_inv = None

            # Initialize some variables
            self.f_old = None
            self.j_old = None
            self.p_old = None
        
            # Initial results
            self.optimize_result = self.get_intermediate_results()
            if self.saveit:
                ot.save_optimize_results(self.optimize_result)
            if self.logger is not None:
                self.logger.info(f'       ====== Running optimization - Line search ({method}) ======')
                self.logger.info('\nSPECIFIED OPTIONS:\n'+pprint.pformat(OptimizeResult(self.options)))
                self.logger.info('')
                self.logger.info(f'       {"iter.":<10} {fun_xk_symbol:<15} {jac_inf_symbol:<15} {"step-size":<15}')
                self.logger.info(f'       {self.iteration:<10} {self.fk:<15.4e} {la.norm(self.jk, np.inf):<15.4e} {0:<15.4e}')
                self.logger.info('')

        self.run_loop() 


    def _fun(self, x):
        self.nfev += 1
        x = ot.clip_state(x, self.bounds) # ensure bounds are respected
        if self.args is None:
            f = np.mean(self.function(x))
        else:
            f = np.mean(self.function(x, *self.args))
        return f

    def _jac(self, x):
        self.njev += 1
        x = ot.clip_state(x, self.bounds) # ensure bounds are respected
        if self.args is None:
            g = self.jacobian(x)
        else:
            g = self.jacobian(x, *self.args)

        # project gradient onto the feasible set
        if self.bounds is not None:
            g = - self._project_pk(-g, x)

        return g
    
    def _hess(self, x):
        if self.hessian is None:
            return None
    
        x = ot.clip_state(x, self.bounds) # ensure bounds are respected
        if self.args is None:
            h = self.hessian(x)
        else:
            h = self.hessian(x, *self.args)
        return h
    
    
    def calc_update(self, iter_resamp=0):

        # Initialize variables for this step
        success = False

        # If in resampling mode, compute jacobian
        # Else, jacobian from in __init__ or from latest line_search is used
        if self.jk is None:
            self.jk = self._jac(self.xk)

        # Compute hessian
        if (self.iteration != 1) or (iter_resamp > 0):
            self.Hk = self._hess(self.xk)

        # Check normalization
        if self.normalize:
            self.jk = self.jk/la.norm(self.jk, np.inf)
            if not self.Hk is None:
                self.Hk = self.Hk/np.maximum(la.norm(self.Hk, np.inf), 1e-12)

        # Calculate search direction (pk)
        if self.method == 'GD':
            pk = - self.jk
        if self.method == 'BFGS':
            pk = - np.matmul(self.Hk_inv, self.jk)
        if self.method == 'Newton-CG':
            pk = newton_cg(self.jk, Hk=self.Hk, xk=self.xk, jac=self._jac, eps=1e-4)

        # porject search direction onto the feasible set
        if self.bounds is not None:
            pk = self._project_pk(pk, self.xk)
        
        # Set step_size
        if self.bounds is not None:
            self.step_size_max = self._set_max_step_size(pk, self.xk)
            self.lskwargs['amax'] = self.step_size_max
        step_size = self._set_step_size(pk, self.step_size_max)

        # Perform line-search 
        self.logger.info('Performing line search...')
        if self.lskwargs['method'] == 0:
            ls_res = line_search_backtracking(
                step_size=step_size,
                xk=self.xk,
                pk=pk,
                fun=self._fun,
                jac=self._jac,
                fk=self.fk,
                jk=self.jk,
                **self.lskwargs
            )
        else:
            ls_res = line_search(
                step_size=step_size,
                xk=self.xk,
                pk=pk,
                fun=self._fun,
                jac=self._jac,
                fk=self.fk,
                jk=self.jk,
                **self.lskwargs
            )
        step_size, f_new, j_new, _, _ = ls_res
    
        if not (step_size is None):
        
            # Save old values
            x_old = self.xk
            j_old = self.jk
            f_old = self.fk

            # Update control
            x_new = ot.clip_state(x_old + step_size*pk, self.bounds)

            # Update state
            self.xk = x_new
            self.fk = f_new
            self.jk = j_new
            
            # Update old fun, jac and pk values
            self.j_old = j_old
            self.f_old = f_old
            self.p_old = pk
            sk = x_new - x_old

            # Call the callback function
            if callable(self.callback):
                self.callback(self) 

            # Update BFGS
            if self.method == 'BFGS':
                yk = j_new - j_old
                if self.iteration == 1: self.Hk_inv = np.dot(yk,sk)/np.dot(yk,yk) * np.eye(sk.size)
                self.Hk_inv = bfgs_update(self.Hk_inv, sk, yk)

            # Update status
            success = True

            # Save Results
            self.optimize_result = self.get_intermediate_results()
            if self.saveit:
                ot.save_optimize_results(self.optimize_result)

            # Write logging info
            if self.logger is not None:
                self.logger.info('')
                self.logger.info(f'       {"iter.":<10} {fun_xk_symbol:<15} {jac_inf_symbol:<15} {"step-size":<15}')
                self.logger.info(f'       {self.iteration:<10} {self.fk:<15.4e} {la.norm(self.jk, np.inf):<15.4e} {step_size:<15.4e}')
                self.logger.info('')
            
            # Check for convergence
            if (la.norm(sk, np.inf) < self.xtol):
                self.msg = 'Convergence criteria met: |dx| < xtol'
                self.logger.info(self.msg)
                success = False
                return success
            if (np.abs(self.fk - f_old) < self.ftol * np.abs(f_old)):
                self.msg = 'Convergence criteria met: |f(x+dx) - f(x)| < ftol * |f(x)|'
                self.logger.info(self.msg)
                success = False
                return success
            if (la.norm(self.jk, np.inf) < self.gtol):
                self.msg = f'Convergence criteria met: {jac_inf_symbol} < gtol'
                self.logger.info(self.msg)
                success = False
                return success

            # Check for custom convergence
            if callable(self.convergence_criteria):
                if self.convergence_criteria(self):
                    self.logger.info('Custom convergence criteria met. Stopping optimization.')
                    success = False
                    return success

            if self.step_size_adapt == 2:
                self.step_size = step_size

            # Update iteration
            self.iteration += 1
        
        else:
            if iter_resamp < self.resample:

                self.logger.info('Resampling Gradient')
                iter_resamp += 1
                self.jk = None

                # Recursivly call function
                success = self.calc_update(iter_resamp=iter_resamp)

            else:
                success = False
    
        return success
    
    def get_intermediate_results(self):

        # Define default results
        results = {
            'fun': self.fk, 
            'x': self.xk, 
            'jac': self.jk,
            'nfev': self.nfev,
            'njev': self.njev,
            'nit': self.iteration,
            'method': self.method,
            'save_folder': self.options.get('save_folder', './')
        }

        if 'savedata' in self.options:
            # Make sure "SAVEDATA" gives a list
            if isinstance(self.options['savedata'], list):
                savedata = self.options['savedata']
            else:
                savedata = [self.options['savedata']]

            if 'args' in savedata:
                for a, arg in enumerate(self.args):
                    results[f'args[{a}]'] = arg

            # Loop over variables to store in save list
            for variable in savedata:
                if variable in locals():
                    results[variable] = eval('{}'.format(variable))
                elif hasattr(self, variable):
                    results[variable] = eval('self.{}'.format(variable))
                else:
                    print(f'Cannot save {variable}!\n\n')

        return OptimizeResult(results)
    
    def _set_step_size(self, pk, amax):
        ''' Sets the step-size '''

        # If first iteration
        if (self.iteration == 1):
            if (self.step_size is None):
                self.step_size = 0.25/la.norm(pk, np.inf)
                alpha = self.step_size
            else:
                alpha = self.step_size

        else:
            if (self.step_size_adapt == 1) and (np.dot(pk, self.jk) != 0):
                alpha = 2*(self.fk - self.f_old)/np.dot(pk, self.jk)
            elif (self.step_size_adapt == 2) and (np.dot(pk, self.jk) == 0):
                slope_old = np.dot(self.p_old, self.j_old)
                slope_new = np.dot(pk, self.jk)
                alpha = self.step_size*slope_old/slope_new
            else:
                alpha = self.step_size

        if alpha < 0: 
            alpha = abs(alpha)

        if alpha >= amax:
            alpha = 0.75*amax
        
        return alpha
    
    def _project_pk(self, pk, xk):
        ''' Projects the jacobian onto the feasible set defined by bounds '''
        lb = np.array(self.bounds)[:, 0]
        ub = np.array(self.bounds)[:, 1]
        for i, pk_val in enumerate(pk):
            if (xk[i] <= lb[i] and pk_val < 0) or (xk[i] >= ub[i] and pk_val > 0):
                pk[i] = 0
        return pk
    
    def _set_max_step_size(self, pk, xk):
        lb = np.array(self.bounds)[:, 0]
        ub = np.array(self.bounds)[:, 1]

        amax = []
        for i, pk_val in enumerate(pk):
            if pk_val < 0:
                amax.append((lb[i] - xk[i])/pk_val)
            elif pk_val > 0:
                amax.append((ub[i] - xk[i])/pk_val)
            else:
                amax.append(np.inf)
        amax = min(amax)
        return amax



def bfgs_update(Hk, sk, yk):
    """
    Perform the BFGS update of the inverse Hessian approximation.

    Parameters:
    - Hk: np.ndarray, current inverse Hessian approximation (n x n)
    - sk: np.ndarray, step vector (x_{k+1} - x_k), shape (n,)
    - yk: np.ndarray, gradient difference (grad_{k+1} - grad_k), shape (n,)

    Returns:
    - Hk_new: np.ndarray, updated inverse Hessian approximation
    """
    sk = sk.reshape(-1, 1)
    yk = yk.reshape(-1, 1)
    rho = 1.0 / (yk.T @ sk)

    if rho <= 0:
        raise ValueError("Non-positive curvature detected. BFGS update skipped.")

    I = np.eye(Hk.shape[0])
    Vk = I - rho * sk @ yk.T
    Hk_new = Vk @ Hk @ Vk.T + rho * sk @ sk.T

    return Hk_new

def newton_cg(gk, Hk=None, maxiter=None, **kwargs):
    print('\nRunning Newton-CG subroutine...')

    if Hk is None:
        jac = kwargs.get('jac')
        eps = kwargs.get('eps', 1e-4)
        xk  = kwargs.get('xk')

        # define a finite difference approximation of the Hessian times a vector
        def Hessd(d):
            return (jac(xk + eps*d) - gk)/eps

    if maxiter is None:
        maxiter = 20*gk.size # Same dfault as in scipy

    tol = min(0.5, np.sqrt(la.norm(gk)))*la.norm(gk)
    z = 0
    r = gk
    d = -r

    for j in range(maxiter):
        print('iteration: ', j)
        if Hk is None:
            Hd = Hessd(d)
        else:
            Hd = np.matmul(Hk, d)

        dTHd = np.dot(d, Hd)

        if dTHd <= 0:
            print('Negative curvature detected, terminating subroutine')
            print('\n')
            if j == 0:
                return -gk
            else:
                return z
            
        rold = r
        a = np.dot(r,r)/dTHd
        z = z + a*d
        r = r + a*Hd

        if la.norm(r) < tol:
            print('Subroutine converged')
            print('\n')
            return z

        b = np.dot(r, r)/np.dot(rold, rold)
        d = -r + b*d

                

            






    



