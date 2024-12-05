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

            Optimization options:
                - maxiter: maximum number of iterations. Default is 20.
                - alpha_maxiter: maximum number of iterations for the line search. Default is 10.
                - alpha_max: maximum step-size. Default is 1000
                - alpha0: initial step-size (for the first iteration). Default is 0.5/|g0|.
                - c1: tolerance parameter for the Armijo condition. Default is 1e-4.
                - c2: tolerance parameter for the Curvature condition. Default is 0.9.
                - alpha_iter_method: method for proposing new step-size in the line search.
                  If alpha_iter_method=0: step-size is cut in half.
                  If alpha_iter_method=1: polynomial interpolation is used to propose new step-size (Default).  
                  If alpha_iter_method=2: the DCSRCH implementation in scipy is used.
                - saveit: If True, the results from each iteration is saved. Default is True.
                - save_folder: Name of folder to save the results to. Defaul is ./ (the current directory).
                - f0: function value of the intial control. Default is None.
                - g0: jacobian of the initial control. Default is None. 
                - H0: initial inverse of hessian (only if method = 'BFGS'). Default is None. 
                - resample: Number of jacobian re-computations allowed if a line search fails. Default is 0.
                - savedata: further specification of which class variables to save to the result files.
                - restart: restart optimization from a restart file (default false)
                - restartsave: save a restart file after each successful iteration (defalut false)
    
    Returns
    -------
    res: OptimizeResult
        Important attributes:
            - x: optimized control
            - fun: objective function value
            - nfev: number of function evaluations
            - njev: number of jacobian evaluations
    
    Example use
    -----------
    >>> import numpy as np
    >>> from scipy.optimize import rosen, rosen_der
    >>> from popt.update_schemes.linesearch import LineSearch
    >>> x0 = np.random.uniform(-3, 3, 2)
    >>> kwargs = {'maxiter': 100,
                  'alpha_maxiter': 10,
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
        self.options = options   # options
        self.fun     = fun       # objective function
        self.jac     = jac       # gradient function
        self.hess    = hess      # hessian function
        self.method  = method    # step method
        self.bounds  = bounds    # parameter bounds
        self.xk      = x         # initial mean state
        self.args    = args

        # Callback
        if callable(callback):
            self.callback = callback
        else:
            self.callback = None
        
        # Set other optimization parameters
        self.normalize       = options.get('normalize', False)
        self.iter_resamp_max = options.get('resample', 0)
        self.saveit          = options.get('saveit', True)
        self.alpha_max       = options.get('alpha_max', 1000)
        self.alpha_adapt     = options.get('alpha_adapt', 2)
        self.alpha           = 0.0
        self.hessian         = None

        # Initialize line-search parameters (scipy defaults for c1, and c2)
        self.ls_options = {'c1': options.get('c1', 1e-4),
                           'c2': options.get('c2', 0.9),
                           'maxiter': options.get('alpha_maxiter', 5),
                           'ls_method':  options.get('alpha_iter_method', 2),
                           'amax': self.alpha_max}
        
        # Calculate objective function of startpoint
        if not self.restart:
            self.start_time = time.perf_counter()

            # Check for initial values
            f0 = options.get('f0', None)
            g0 = options.get('g0', None)
            self.fold = None
            self.gold = None
            self.pold = None

            if f0 is None: self.fk = self._fun(self.xk)
            else: self.fk = f0

            if g0 is None: self.gk = self._jac(self.xk)
            else: self.gk = g0

            # Choose method
            if self.method == 'BFGS':
                self.H = options.get('H0', np.eye(x.size))

            # Inital step-size
            self.alpha0 = options.get('alpha0', None)
            if self.alpha0 is not None:
                self.ls_options['a0'] = self.alpha0
        
            # Save Results
            self.optimize_result = self.update_results()
            if self.saveit:
                ot.save_optimize_results(self.optimize_result)

            if self.logger is not None:
                self.logger.info('       ====== Running optimization - EnOpt ======')
                self.logger.info('\n'+pprint.pformat(self.options))
                self.logger.info(f'       {'iter.':<10} {'fun':<15} {'step-size':<15} {'|grad|':<15}')
                self.logger.info(f'       {self.iteration:<10} {self.fk:<15.4e} {self.alpha:<15.4e}  {la.norm(self.gk):<15.4e}')
                self.logger.info('')

        self.run_loop() 

    
    def update_results(self):

        res = {'fun': self.fk, 
               'x': self.xk, 
               'jac': self.gk,
               'hess': self.hessian,
               'nfev': self.nfev,
               'njev': self.njev,
               'nit': self.iteration,
               'step-size': self.alpha,
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

    def _fun(self, x):
        self.nfev += 1
        x = ot.clip_state(x, self.bounds) # ensure bounds are respected
        if self.args is None:
            f = np.mean(self.fun(x))
        else:
            f = np.mean(self.fun(x, *self.args))
        return f

    def _jac(self, x):
        self.njev += 1
        x = ot.clip_state(x, self.bounds) # ensure bounds are respected
        if self.args is None:
            g = self.jac(x)
        else:
            g = self.jac(x, *self.args)
        return g
    
    def _set_step_size(self, pk):
        ''' Sets the step-size '''
        # If first iteration
        if self.iteration == 1:
            if self.alpha0 is None:
                a = 0.25/np.linalg.norm(pk)
            else:
                a = self.alpha0
        else:
            if np.dot(pk, self.gk) != 0 and self.alpha_adapt != 0:
                if self.alpha_adapt == 1:
                    a = 2*(self.fk - self.fold)/np.dot(pk, self.gk)
                if self.alpha_adapt == 2:
                    a = self.alpha*np.dot(self.pold, self.gold)/np.dot(pk, self.gk)
            else:
                a = self.alpha0

        if a < 0: 
            a = abs(a)

        if self.method == 'BFGS' and self.alpha0 is None:
            # From "Numerical Optimization"
            a = min(1, 1.01*a)
            a = min(self.alpha_max, a)
        else:
           a = min(self.alpha_max, a)

        return a

   
    def calc_update(self, iter_resamp=0):

        # Initialize variables for this step
        success = False

        #Compute gradient
        if self.gk is None:
            self.gk = self._jac(self.xk)
    
        # Compute the hessian (Not Used Currently)
        if self.hess is not None:
            self.hessian = self.hess()
            if self.normalize:
                self.hessian /= np.maximum(la.norm(self.hessian, np.inf), 1e-12)

        # Calculate search direction
        if self.method == 'GD':
            pk = - self.gk
        if self.method == 'BFGS':
            pk = - np.matmul(self.H, self.gk)
        
        # Set step_size
        step_size = self._set_step_size(pk)
        
        # Perform line-search 
        self.logger.info('Performing line search...')
        ls_kw = {'fun': self._fun,
                 'jac': self._jac,
                 'xk': self.xk,
                 'pk': pk,
                 'ak': step_size,
                 'fk': self.fk,
                 'gk': self.gk,
                 'logger': self.logger}
        step_size, fnew, fold, gnew = line_search(**ls_kw, **self.ls_options)
    
        if not step_size is None:

            xold = self.xk
            xnew = ot.clip_state(xold + step_size*pk, self.bounds)
            gold = self.gk

            # Update state
            self.xk = xnew
            self.fk = fnew
            self.gold  = self.gk
            self.gk    = gnew
            self.fold  = fold
            self.pold  = pk
            self.alpha = step_size

            # Call the callback function
            if callable(self.callback):
                self.callback(self) 

            # Update BFGS
            if self.method == 'BFGS':
                sk = xnew - xold
                yk = gnew - gold
                rho = 1/np.dot(yk,sk)
                I = np.eye(sk.size)
                self.H = (I-rho*np.outer(sk, yk)) @ self.H @ (I-rho*np.outer(yk, sk)) + rho*np.outer(sk, sk)

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
                self.logger.info(f'       {self.iteration:<10} {self.fk:<15.4e} {self.alpha:<15.4e}  {la.norm(self.gk):<15.4e}')
                self.logger.info('')

            # Update iteration
            self.iteration += 1
        
        else:
            if iter_resamp < self.iter_resamp_max:

                self.logger.info('Resampling Gradient')
                iter_resamp += 1
                self.gk = None

                # Recursivly call function
                success = self.calc_update(iter_resamp=iter_resamp)

            else:
                success = False
    
        return success



def line_search(fun, jac, xk, pk, ak, fk=None, gk=None, c1=0.0001, c2=0.9, maxiter=10, **kwargs):
    return LineSearchStepBase(fun, jac, xk, pk, ak, fk, gk, c1, c2, maxiter, **kwargs)()

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
        self.amax = kwargs.get('amax', 1000)
        self.amin = kwargs.get('amin', 0.0)
        self.xtol = kwargs.get('xtol', 1e-10)
        self.ls_method = kwargs.get('ls_method', 0)
        self.logger    = kwargs.get('logger', None)

        # Other variables
        self.jac_val = None
        self.iter = 0

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

        if self.ls_method == 0:
            step_size, fnew = self._line_search_alpha_cut(step_size=self.ak)
        
        if self.ls_method == 1:
            step_size, fnew = self._line_search_alpha_interpol(step_size=self.ak)

        if self.ls_method == 2:
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
            self.log('Evaluating Armijo Condition')
        return self.fun(self.xk + a*self.pk)

    def dphi(self, a, eval=True):
        if eval:
            self.log('Evaluating Curvature Condition')
        jval = self.jac(self.xk + a*self.pk)
        self.jac_val = jval
        return np.dot(self.pk, jval)
    




    



