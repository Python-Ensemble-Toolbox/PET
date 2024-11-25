# External imports
import numpy as np
import time
import pprint
import warnings

from numpy import linalg as la
from scipy.optimize import OptimizeResult
#from scipy.optimize import line_search

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.loop.optimize import Optimize


class LineSearch(Optimize):
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
    
    hess: callable
        Hessian function.
    '''
    def  __init__(self, fun, x, jac, method='GD', hess=None, args=(), bounds=None, callback=None,**options):

        # init PETEnsemble
        super(LineSearch, self).__init__(**options)

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
        self.normalize       = options.get('normalize', True)
        self.iter_resamp_max = options.get('resample', 0)
        self.saveit          = options.get('saveit', False)
        self.alpha           = 0.0
        self.hessian         = None

        # Initialize line-search parameters (scipy defaults for c1, and c2)
        self.ls_options = {'c1': options.get('c1', 0.0001),
                           'c2': options.get('c2', 0.9),
                           'maxiter': options.get('alpha_maxiter', 5)}
        
        # Calculate objective function of startpoint
        if not self.restart:
            self.start_time = time.perf_counter()

            # Check for initial values
            f0 = options.get('f0', None)
            g0 = options.get('g0', None)

            if f0 is None: self.fk = self._fun(self.xk)
            else: self.fk = self.f0

            if g0 is None: self.gk = self._jac(self.xk)
            else: self.fk = self.f0

            # Init
            if self.method == 'BFGS':
                self.H = options.get('H0', np.eye(x.size))
                self.alpha_max  = options.get('alpha_max', 1.0)
            else:
                self.alpha_max  = options.get('alpha_max', 100)


            # This sets the inital step-step such that the initial dx is sqrt(d)
            self.fold = self.fk + np.sqrt(x.size)*np.linalg.norm(self.gk)/2  

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
               'args': self.args,
               'step-size': self.alpha,
               'save_folder': self.options['save_folder']}

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

        self.logger.info('Performing line search...')
        ls_kw = {'fun': self._fun,
                 'grad': self._jac,
                 'amax': self.alpha_max,
                 'xk': self.xk,
                 'pk': pk,
                 'fk': self.fk,
                 'gk': self.gk,
                 'fold': self.fold,
                 'logger': self.logger}
        
        step_size, fnew, self.fold, gnew, i = line_search_step(**ls_kw, **self.ls_options)
        if not step_size is None:

            xold = self.xk
            xnew = ot.clip_state(xold + step_size*pk, self.bounds)
            gold = self.gk

            # Update state
            self.xk = xnew
            self.fk = fnew
            self.gk = gnew
            self.fold = self.fold 
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

def line_search_step(fun, grad, xk, pk, fk=None, gk=None, c1=0.0001, c2=0.9, maxiter=10, **kwargs):

    # Get kwargs
    fold = kwargs.get('fold', None)
    xtol = kwargs.get('xtol', 1e-10)
    amax = kwargs.get('amax', 1.0)
    amin = kwargs.get('amin', 1e-8)
    logger = kwargs.get('logger', None)

    def _log(message):
        if logger is None:
            print(message)
        else:
            logger.info(message)
    
    def phi(a):
        _log('Evaluating Armijo Condition')
        return fun(xk + a*pk)
    
    def dphi(a):
        _log('Evaluating Curvature Condition')
        gnew = grad(xk + a*pk)
        dphi.gnew = gnew
        return np.dot(pk, gnew)
    
    # Check for initial phi and dphi
    if fk is None:
        phi0 = phi(0)
    else:
        phi0 = fk
    
    if gk is None:
        dphi0 = dphi(0)
    else:
        dphi0 = np.dot(pk, gk)

    # Initial step-size
    if fold is None:
        a0 = kwargs.get('a0', 1/np.linalg.norm(pk))
    else:
        # From "Numerical Optimization"
        a0 = 2*abs(phi0 - fold)/max(abs(dphi0), 1e-20)

    a1 = min(amax, 1.01*a0)

    # Perform Line-Search
    step_size, fnew, fold, i = _lines_search(phi, dphi, a1, phi0, dphi0, c1, c2, maxiter)

    if step_size is None:
        _log(f'Line search method did not converge')
        return None, fnew, None, None, i
    else:
        step_size = min(step_size, amax)
        return step_size, fnew, fold, dphi.gnew, i 


def _lines_search(phi, dphi, a1, phi0, dphi0, c1, c2, maxiter):
    
    ai = a1
    a_list   = []
    phi_list = []
    
    for i in range(maxiter+1):
        phi_i = phi(ai)

        a_list.append(ai)
        phi_list.append(phi_i)

        if phi_i < phi0 + c1*ai*dphi0:
            dphi_i = dphi(ai)
            if abs(dphi_i) <= abs(c2*dphi0):
                return ai, phi_i, phi0, i

        # Set new alpha
        if i == 0:
            ai = quadratic_interpolation(a_list[i], phi_list[i], phi0, dphi0) 
        else:
            ai = qubic_interpolation(a_list[i], a_list[i-1], phi_list[i], phi_list[i-1], phi0, dphi0)

    # did not converge
    return None, None, phi0, i


def quadratic_interpolation(a, phi_a, phi0, dphi0):
    a_new = (dphi0*a**2)/(2*(phi_a - phi0 - dphi0*a))
    return a_new

def qubic_interpolation(ai, aold, phi_i, phi_old, phi0, dphi0):
    frac = 1/((aold**2)*(ai**2)*(ai-aold))
    mat = np.array([[aold**2, -ai**2],
                    [-aold**3, ai**3]])
    vec = np.array([[phi_i-phi0-dphi0*ai], [phi_old-phi_i-dphi0*aold]])
    k = frac*np.squeeze(mat@vec)

    sqrt_term = k[1]**2 - 3*k[0]*dphi0
    if sqrt_term >= 0:
        a_new =  (-k[1] + np.sqrt(sqrt_term))/(3*k[0])
    else:
        a_new = ai/2

    return a_new








    
