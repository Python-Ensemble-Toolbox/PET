# External imports
import numpy as np
import time
import pprint
import warnings

from numpy import linalg as la
from functools import cache
from scipy.optimize import OptimizeResult
from scipy.optimize._dcsrch import DCSRCH
from scipy.optimize._linesearch import scalar_search_wolfe2

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.loop.optimize import Optimize

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
        Other options are 'BFGS' for the 'Broyden–Fletcher–Goldfarb–Shanno' method.   
        TODO: Include 'Newton-method'.
    
    hess: callable, optional
        Hessian function, hess(x, *args). Default is None. 
    
    args: tuple, optional
        Args passed to fun, jac and hess.

    bounds: list, optional
        (min, max) pairs for each element in x. None is used to specify no bound.
    
    callback: callable, optional
        A callable called after each successful iteration. The class instance of LineSearch 
        is passed as the only argument to the callback function: callback(self) 
    
    **options: keyword arguments, optional

    
    LineSearch Options (**options)
    ------------------------------
    maxiter: int
        Maximum number of iterations. Default is 20.
    
    step_size: float
        Step-size for optimizer. Default is 0.25/inf-norm(jac(x0)).

    step_size_maxiter: int
        Maximum number of iterations for the line search. Default is 10.

    step_size_max: float
        Maximum step-size. Default is 1000

    step_size_adapt: int
        Set method for choosing initial step-size for each iteration. If 0, step_size value is used.
        If 1, Equation (3.6) from "Numercal Optimization" [1] is used. If 2, the equation above Equation (3.6) is used. 
        Default is 0.
                    
    c1: float
        Tolerance parameter for the Armijo condition. Default is 1e-4.

    c2: float
        Tolerance parameter for the Curvature condition. Default is 0.9.

    line_search_method: int
        Sets method for proposing new step-size in the line search.  Default is 1.
        If line_search_method=0: step-size is cut in half.
        If line_search_method=1: Algorithm (3.5) from [1] with polynomial interpolation is used.
        If line_search_method=2: the DCSRCH implementation in scipy is used.

    saveit: bool
        If True, the results from each iteration is saved. Default is True.

    save_folder: str
        Name of folder to save the results to. Defaul is ./ (the current directory).

    fun0: float
        Function value of the intial control.

    jac0: ndarray
        Jacobian of the initial control.
    
    hess0: ndarray
        Hessian value of the initial control.

    hess0_inv: ndarray
        Initial inverse of hessian (only if method = 'BFGS').

    resample: int
        Number of jacobian re-computations allowed if a line search fails. Default is 0.
        (useful if jacobian is stochastic)

    savedata: list[str]
        Further specification of which class variables to save to the result files.

    restart: bool
        Restart optimization from a restart file. Default is False

    restartsave: bool
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
                  'line_search_maxiter': 10,
                  'step_size_adapt': 1,
                  'saveit': False}
    >>> res = LineSearch(fun=rosen, x=x0, jac=rosen_der, method='BFGS', **kwargs)
    >>> print(res)
    '''
    obj = LineSearchClass(fun, x, jac, method, hess, args, bounds, callback, **options)
    return obj.optimize_result


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
        
        # Set options for step-size
        self.step_size       = options.get('step_size', None)
        self.step_size_max   = options.get('step_size_max', 1000)
        self.step_size_adapt = options.get('step_size_adapt', 0)

        # Set options for line-search method
        self.line_search_kwargs  = {'c1': options.get('c1', 1e-4),
                                    'c2': options.get('c2', 0.9),
                                    'amax': self.step_size_max,
                                    'maxiter': options.get('line_search_maxiter', 10),
                                    'method' : options.get('line_search_method', 1)}

        # Set other options
        self.normalize = options.get('normalize', False)
        self.resample  = options.get('resample', 0)
        self.saveit    = options.get('saveit', True)

        # Check method
        valid_methods = ['GD', 'BFGS']
        if not self.method in valid_methods:
            raise ValueError(f"'{self.method}' is not a valid method. Valid methods are: {valid_methods}")
        
        # Calculate objective function of startpoint
        if not self.restart:
            self.start_time = time.perf_counter()

            # Check for initial callable values
            self.fk = options.get('fun0', self._fun(self.xk))
            self.jk = options.get('jac0', self._jac(self.xk))
            self.Hk = options.get('hess0', self._hess(self.xk))

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
            if self.saveit:
                self.optimize_result = self.update_results()
                ot.save_optimize_results(self.optimize_result)

            if self.logger is not None:
                self.logger.info(f'       ====== Running optimization - Line search ({method}) ======')
                self.logger.info('\n'+pprint.pformat(self.options))
                self.logger.info(f'       {'iter.':<10} {'fun':<15} {'step-size':<15} {'|grad|':<15}')
                self.logger.info(f'       {self.iteration:<10} {self.fk:<15.4e} {0.0:<15.4e}  {la.norm(self.jk):<15.4e}')
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
        return g
    
    def _hess(self, x):
        if self.hessian is None:
            return None
    
        x = ot.clip_state(x, self.bounds) # ensure bounds are respected
        if self.args is None:
            h = self.hessian(x)
        else:
            h = self.hessian(x, *self.args)
        return ot.get_sym_pos_semidef(h)
    
    def update_results(self):

        res = {'fun': self.fk, 
               'x': self.xk, 
               'jac': self.jk,
               'hess': self.Hk,
               'hess_inv': self.Hk_inv,
               'nfev': self.nfev,
               'njev': self.njev,
               'nit': self.iteration,
               'step-size': self.step_size,
               'save_folder': self.options.get('save_folder', './')}
        
        for a, arg in enumerate(self.args):
            res[f'args[{a}]'] = arg

        if 'savedata' in self.options:
            # Make sure "SAVEDATA" gives a list
            if isinstance(self.options['savedata'], list):
                savedata = self.options['savedata']
            else:
                savedata = [self.options['savedata']]

            # Loop over variables to store in save list
            for save_typ in savedata:
                if save_typ in locals():
                    res[save_typ] = eval('{}'.format(save_typ))
                elif hasattr(self, save_typ):
                    res[save_typ] = eval(' self.{}'.format(save_typ))
                else:
                    print(f'Cannot save {save_typ}!\n\n')

        return OptimizeResult(res)
    
    def _set_step_size(self, pk):
        ''' Sets the step-size '''

        # If first iteration
        if self.iteration == 1:
            if self.step_size is None:
                alpha = 0.25/la.norm(pk, np.inf)
            else:
                alpha = self.step_size
        else:
            if np.dot(pk, self.jk) != 0 and self.step_size_adapt != 0:
                if self.step_size_adapt == 1:
                    alpha = 2*(self.fk - self.f_old)/np.dot(pk, self.jk)
                if self.step_size_adapt == 2:
                    slope_old = np.dot(self.p_old, self.j_old)
                    slope_new = np.dot(pk, self.jk)
                    alpha = self.step_size*slope_old/slope_new
            else:
                alpha = self.step_size

        if alpha < 0: 
            alpha = abs(alpha)

        if self.method == 'BFGS':
            # From "Numerical Optimization"
            alpha = min(1, 1.01*alpha)
        
        return min(alpha, self.step_size_max)

   
    def calc_update(self, iter_resamp=0):

        # Initialize variables for this step
        success = False

        # If in resampling mode, compute jacobian 
        # Else, jacobian from in __init__ or latest line_search is used
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
        
        # Set step_size
        step_size = self._set_step_size(pk)
        
        # Perform line-search 
        self.logger.info('Performing line search...')
        kw = {'fun': self._fun,
              'jac': self._jac,
              'xk': self.xk,
              'pk': pk,
              'ak': step_size,
              'fk': self.fk,
              'gk': self.jk,
              'logger': self.logger}
        
        step_size, f_new, f_old, j_new = line_search(**kw, **self.line_search_kwargs)
    
        if not step_size is None:

            x_old = self.xk
            j_old = self.jk
            x_new = ot.clip_state(x_old + step_size*pk, self.bounds)

            # Update state
            self.xk = x_new
            self.fk = f_new
            self.jk = j_new

            self.j_old = j_old
            self.f_old = f_old
            self.p_old = pk
            self.step_size = step_size

            # Call the callback function
            if callable(self.callback):
                self.callback(self) 

            # Update BFGS
            if self.method == 'BFGS':
                sk  = x_new - x_old
                yk  = j_new - j_old
                rho = 1/np.dot(yk,sk)
                id_mat = np.eye(sk.size)

                matrix1 = (id_mat - rho*np.outer(sk, yk))
                matrix2 = (id_mat - rho*np.outer(yk, sk))
                self.Hk_inv = matrix1@self.Hk_inv@matrix2 + rho*np.outer(sk, sk)

            # Update status
            success = True

            # Save Results
            self.optimize_result = self.update_results()
            if self.saveit:
                ot.save_optimize_results(self.optimize_result)

            # Write logging info
            if self.logger is not None:
                self.logger.info('')
                self.logger.info(f'       {'iter.':<10} {'fun':<15} {'step-size':<15} {'|grad|':<15}')
                self.logger.info(f'       {self.iteration:<10} {self.fk:<15.4e} {step_size:<15.4e}  {la.norm(self.jk):<15.4e}')
                self.logger.info('')

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



def line_search(fun, jac, xk, pk, ak, fk=None, gk=None, c1=0.0001, c2=0.9, maxiter=10, **kwargs):
    '''
    Performs a single line search step
    '''
    line_search_step = LineSearchStepBase(fun, jac, xk, pk, ak, fk, gk, c1, c2, maxiter, **kwargs)
    return line_search_step()

class LineSearchStepBase:

    def __init__(self, fun, jac, xk, pk, ak, fk=None, gk=None, c1=0.0001, c2=0.9, maxiter=10, **kwargs):
        self.fun = fun
        self.jac = jac
        self.xk = xk
        self.pk = pk
        self.ak = ak
        self.fk = fk
        self.gk = gk
        self.c1 = c1
        self.c2 = c2
        self.maxiter = maxiter

        # kwargs
        self.amax   = kwargs.get('amax', 1000)
        self.amin   = kwargs.get('amin', 0.0)
        self.xtol   = kwargs.get('xtol', 1e-10)
        self.method = kwargs.get('method', 1)
        self.logger = kwargs.get('logger', None)

        # Check for initial values
        if self.fk is None:
            self.phi0 = self.phi(0, eval=False)
        else:
            self.phi0 = self.fk

        if self.gk is None:
            self.dphi0 = self.dphi(0, eval=False)
        else:
            self.dphi0 = np.dot(self.pk, self.gk)


    def __call__(self):

        if self.method == 0:
            step_size, fnew = self._line_search_alpha_cut(step_size=self.ak)
        
        if self.method == 1:
            step_size, fnew = self._line_search_alpha_interpol(step_size=self.ak)

        if self.method == 2:
            dcsrch = DCSRCH(self.phi, self.dphi, self.c1, self.c2, self.xtol, self.amin, self.amax)
            step_size, fnew, _,  _ = dcsrch(self.ak, phi0=self.phi0, derphi0=self.dphi0, maxiter=self.maxiter)

        if step_size is None:
            self.log('Line search method did not converge')
            return None, None, None, None
        else:
            step_size = min(step_size, self.amax)
            return step_size, fnew, self.phi0, self.jac_val


    def _line_search_alpha_cut(self, step_size):

        ak = step_size
        for i in range(self.maxiter):
            phi_new = self.phi(ak)

            # Check Armijo Condition
            if phi_new < self.phi0 + self.c1*ak*self.dphi0:
                dphi_new = self.dphi(ak)

                # Curvature condition
                if abs(dphi_new) <= abs(self.c2*self.dphi0):
                    return ak, phi_new
        
        return None, None
    
    def _line_search_alpha_interpol(self, step_size):
        from scipy.optimize._linesearch import _zoom
        ak  = step_size    
        a   = [0.0]
        ph  = [self.phi0]
        dph = [self.dphi0]

        for i in range(1, self.maxiter+1):
            
            # Append lists
            a.append(ak)
            ph.append(self.phi(ak))
            dph.append(self.dphi(ak))

            # Check Armijo Condition
            if ph[i] > self.phi0 + self.c1*a[i]*self.dphi0 or (ph[i] >= ph[i-1] and i>1):
                step_size_new, phi_new, _ = _zoom(a_lo=a[i-1], a_hi=a[i],
                                                  phi_lo=ph[i-1], phi_hi=ph[i],
                                                  derphi_lo=dph[i-1],
                                                  phi=self.phi, derphi=self.dphi,
                                                  phi0=self.phi0, derphi0=self.dphi0,
                                                  c1=self.c1,c2=self.c2,
                                                  extra_condition = lambda *args: True)
                return step_size_new, phi_new
            
            if abs(dph[i]) < -self.c2*self.dphi0:
                return a[i], ph[i]
            
            # Check # Curvature condition
            if dph[i] >= 0:
                step_size_new, phi_new, _ = _zoom(a_lo=a[i], a_hi=a[i-1],
                                                  phi_lo=ph[i], phi_hi=ph[i-1],
                                                  derphi_lo=dph[i],
                                                  phi=self.phi, derphi=self.dphi,
                                                  phi0=self.phi0, derphi0=self.dphi0,
                                                  c1=self.c1, c2=self.c2,
                                                  extra_condition = lambda *args: True)
                return step_size_new, phi_new
            
            if a[i] >= self.amax:
                return None, None
            else:
                ak = ak*2
        
        return None, None


    def log(self, msg):
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
            
    def phi(self, a, eval=True):
        if eval:
            self.log('  Evaluating Armijo Condition')
        return self.fun(self.xk + a*self.pk)

    def dphi(self, a, eval=True):
        if eval:
            self.log('  Evaluating Curvature Condition')
        jval = self.jac(self.xk + a*self.pk)
        self.jac_val = jval
        return np.dot(self.pk, jval)
    




    



