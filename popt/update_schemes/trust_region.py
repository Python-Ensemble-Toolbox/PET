# External imports
import numpy as np
import time
import pprint
import warnings

from numpy import linalg as la
from scipy.optimize import OptimizeResult

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.loop.optimize import Optimize
from popt.update_schemes.subroutines.subroutines import solve_trust_region_subproblem

# Some symbols for logger
subk = '\u2096'
fun_xk_symbol  = f'fun(x{subk})'
delta_k_symbol = f'\u0394{subk}'
rho_symbol     = '\u03C1'


def TrustRegion(fun, x, jac, hess, method='iterative', args=(), bounds=None, callback=None, **options):
    '''
    Trust region optimization algorithm.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized. The calling signature is `fun(x, *args)`.

    x : array_like
        Initial guess.
    
    jac : callable
        Gradient (Jacobian) of objective function. The calling signature is `jac(x, *args)`.
    
    hess : callable
        Hessian of objective function. The calling signature is `hess(x, *args)`.

    method : str, optional
        Method to use for solving the trust-region subproblem. Options are 'iterative' or 'CG-Steihaug'.
        Default is 'iterative'.
    
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives (Jacobian, Hessian).
    
    bounds : sequence, optional
        Bounds for variables. Each element of the sequence must be a tuple of two scalars, 
        representing the lower and upper bounds for that variable. Use None for one of the bounds if there are no bounds.
        Bounds are handle by clipping the state to the bounds before evaluating the objective function and its derivatives.
    
    callback: callable, optional
        A callable called after each successful iteration. The class instance 
        is passed as the only argument to the callback function: callback(self) 

    **options : keyword arguments, optional

    TrustRegion Options (**options)
    -------------------------------
    maxiter: int
        Maximum number of iterations. Default is 20.
    
    trust_radius: float
        Inital trust-region radius. Default is 1.0.
    
    trust_radius_max: float
        Maximum trust-region radius. Default is 10 times initial trust_radius.
    
    trust_radius_min: float
        Minimum trust-region radius. Optimization is terminated if trust_radius = trust_radius_min.
        Default is trust_radius/100.
    
    trust_radius_cuts: int
        Number of allowed trust-region radius reductions if a step is not successful. Default is 4.
    
    rho_tol: float
        Tolerance for rho (ratio of actual to predicted reduction). Default is 1e-6.
        
    eta1, eta2, gam1, gam2: float
        Parameters for updateing the trust-region radius.

        Δnew = max(gam2*Δold, Δmax)   if rho >= eta2. \n
        Δnew = Δold                   if eta1 <= rho < eta2. \n
        Δnew = gam1*Δold              if rho < eta1. \n

        Defults:
            eta1 = 0.001 \n
            eta2 = 0.1 \n
            gam1 = 0.7 \n
            gam2 = 1.5 \n
    
    saveit: bool
        If True, save the optimization results to a file. Default is True.
    
    convergence_criteria: callable
        A callable that takes the current optimization object as an argument and returns True if the optimization should stop.
        It can be used to implement custom convergence criteria. Default is None.

    save_folder: str
        Name of folder to save the results to. Defaul is ./ (the current directory).

    fun0: float
        Function value of the intial control.

    jac0: ndarray
        Jacobian of the initial control.
    
    hess0: ndarray
        Hessian value of the initial control.

    resample: bool
        If True, resample the Jacobian and Hessian if a step is not successful. Default is False.
        (Only makes sense if the Jacobian and Hessian are stochastic).

    savedata: list[str]
        Further specification of which class variables to save to the result files.

    restart: bool
        Restart optimization from a restart file. Default is False

    restartsave: bool
        Save a restart file after each successful iteration. Default is False


    Returns
    -------
    OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes:
        - x: optimized control
        - fun: objective function value
        - nfev: number of function evaluations
        - njev: number of jacobian evaluations
    '''
    tr_obj = TrustRegionClass(fun, x, jac, hess, method, args, bounds, callback, **options)
    return tr_obj.optimize_result

class TrustRegionClass(Optimize):

    def __init__(self, fun, x, jac, hess, method='iterative', args=(), bounds=None, callback=None, **options):
        
        # Initialize the parent class
        super().__init__(**options)

        # Set class attributes
        self.function = fun
        self._xk      = x
        self.jacobian = jac
        self.hessian  = hess
        self.method   = method
        self.args     = args
        self.bounds   = bounds
        self.options  = options

        # Check if the callback function is callable
        if callable(callback):
            self.callback = callback
        else:
            self.callback = None

        # Custom convergence criteria (callable)
        convergence_criteria = options.get('convergence_criteria', None)
        if callable(convergence_criteria):
            self.convergence_criteria = convergence_criteria
        else:
            self.convergence_criteria = None

        # Set options for trust-region radius       
        self.trust_radius      = options.get('trust_radius', 1.0) 
        self.trust_radius_max  = options.get('trust_radius_max', 100*self.trust_radius)
        self.trust_radius_min  = options.get('trust_radius_min', self.trust_radius/1000)
        self.trust_radius_cuts = options.get('trust_radius_cuts', 4)

        # Set other options
        self.resample = options.get('resample', False)
        self.saveit   = options.get('saveit', True)
        self.rho_tol  = options.get('rho_tol', 1e-6)
        self.eta1 = options.get('eta1', 0.1)  # reduce raduis if rho < 10%
        self.eta2 = options.get('eta2', 0.5)  # increase radius if rho > 50% 
        self.gam1 = options.get('gam1', 0.5)  # reduce by 50%
        self.gam2 = options.get('gam2', 1.5)  # increase by 50%
        self.rho  = 0.0

        # set tolerance for convergence
        self._xtol = options.get('xtol', 1e-8) # tolerance for control vector
        self._ftol = options.get('ftol', 1e-4) # relative tolerance for function value
        self._gtol = options.get('gtol', 1e-5) # tolerance for inf-norm of jacobian

        # Check if method is valid
        if callable(self.method):
            self.logger(f'Method is a callable!. Using custom subproblem solver.')
        elif isinstance(self.method, str):
            if self.method not in ['iterative', 'CG-Steihaug']:
                self.method = 'iterative'
                self.logger(f'Method {self.method} is not valid!. Method is set to "iterative"')
        else:
            self.logger(f'Method is a string or callable!. Method is set to "iterative"')
            self.method = 'iterative'

    
        if not self.restart:
            self.start_time = time.perf_counter()

            # Check for initial callable values
            self._fk = options.get('fun0', None)
            self._jk = options.get('jac0', None)
            self._Hk = options.get('hess0', None)

            if self.hessian == 'BFGS':
                self.hessian = None
                self.quasi_newton = True
                self.logger('Hessian approximation set to BFGS.')
            else:
                self.quasi_newton = False

            if self._fk is None: self._fk = self.fun(self._xk)
            if self._jk is None: self._jk = self.jac(self._xk)
            if self._Hk is None: self._Hk = self.hess(self._xk)

            if self.logger is not None:
                self.logger('================= Running Optimization - Trust Region =================')
                self.logger(f'\n \nUSER-SPECIFIED OPTIONS:\n{pprint.pformat(OptimizeResult(self.options))}\n')
                info = {
                    'Iter.': self.iteration,
                    fun_xk_symbol: self._fk,
                    f'{delta_k_symbol}': self.trust_radius,
                    f'{rho_symbol}': self.rho,
                    f'|p{subk}| = {delta_k_symbol}': 'N/A',
                }
                self.logger(**info)
            

            # Initial results
            self.optimize_result = self.get_intermediate_results()
            if self.saveit:
                ot.save_optimize_results(self.optimize_result)

        # Run the optimization
        self.run_loop()

    def fun(self, x, *args, **kwargs):
        self.nfev += 1

        if self.bounds is not None:
            lb = np.array(self.bounds)[:, 0]
            ub = np.array(self.bounds)[:, 1]
            x = np.clip(x, lb, ub)  # ensure bounds are respected
            
        if self.args is None:
            f = np.mean(self.function(x, epf=self.epf))
        else:
            f = np.mean(self.function(x, *self.args, epf=self.epf))
        return f

    @property
    def xk(self):
        return self._xk

    @property
    def fk(self):
        return self._fk

    @property
    def ftol(self):
        return self.obj_func_tol

    @ftol.setter
    def ftol(self, value):
        self.obj_func_tol = value

    def jac(self, x):
        self.njev += 1
        
        if self.bounds is not None:
            lb = np.array(self.bounds)[:, 0]
            ub = np.array(self.bounds)[:, 1]
            x = np.clip(x, lb, ub)  # ensure bounds are respected
            
        if self.args is None:
            g = self.jacobian(x, epf=self.epf)
        else:
            g = self.jacobian(x, *self.args, epf=self.epf)
        return g
    
    def hess(self, x):
        if self.hessian is None:
            return None
        
        if self.bounds is not None:
            lb = np.array(self.bounds)[:, 0]
            ub = np.array(self.bounds)[:, 1]
            x = np.clip(x, lb, ub)  # ensure bounds are respected

        if self.args is None:
            h = self.hessian(x)
        else:
            h = self.hessian(x, *self.args)
        return h

    def calc_update(self, inner_iter=0):

        # Initialize variables for this step
        success = True

        #print(self.quasi_newton, self._Hk is None, self.iteration)
        if self.quasi_newton and (self._Hk is None) and (self.iteration == 1):
            sk = - self._jk
            sk = sk / la.norm(sk, np.inf) * self.trust_radius
            hits_boundary = True
            if la.norm(sk) > self.trust_radius:
                sk = sk / la.norm(sk) * self.trust_radius
                hits_boundary = True

        else:
            # Solve subproblem
            self.logger(f'Solving subproblem ...................')
            if callable(self.method):
                sk, hits_boundary = self.method(
                    self._xk, 
                    self._fk, 
                    self._jk, 
                    self._Hk, 
                    self.trust_radius, 
                    **self.options
                )
            else:
                sk, hits_boundary = solve_trust_region_subproblem(
                    self._xk, 
                    self._fk, 
                    self._jk, 
                    self._Hk, 
                    self.trust_radius,
                    method=self.method, 
                    **self.options
                )

        # Truncate sk to respect bounds
        if self.bounds is not None:
            lb = np.array(self.bounds)[:, 0]
            ub = np.array(self.bounds)[:, 1]
            sk = np.clip(sk, lb - self._xk, ub - self._xk)

        # Calculate the actual function value
        xk_new = self._xk + sk
        fk_new = self.fun(xk_new)
        print(self._fk, fk_new)
        # Calculate rho (actual / predicted reduction)
        df = self._fk - fk_new
        if self.iteration == 1 and self.quasi_newton:
            dm = - np.dot(self._jk, sk)
        else:
            dm = - np.dot(self._jk, sk) - np.dot(sk, np.dot(self._Hk, sk))/2
        print(df, dm)
        self.rho = df/(dm + 1e-16)  # add small number to avoid division by zero
        
        if self.rho > self.rho_tol:
            
            # Save old values
            x_old = self._xk
            f_old = self._fk
            j_old = self._jk
            h_old = self._Hk

            # Update the control
            self._xk = xk_new
            self._fk = fk_new

            # Save Results
            self.optimize_result = ot.get_optimize_result(self)
            if self.saveit:
                ot.save_optimize_results(self.optimize_result)

            # Write logging info
            info = {
                'Iter.': self.iteration,
                f'{fun_xk_symbol}': self._fk,
                f'{delta_k_symbol}': self.trust_radius,
                f'{rho_symbol}': self.rho,
                f'|p{subk}| = {delta_k_symbol}': 'True' if hits_boundary else 'False',
            }
            self.logger(**info)

            # Call the callback function
            if callable(self.callback):
                self.callback(self)

            # Check for convergence
            if (la.norm(sk, np.inf) < self._xtol):
                self.msg = 'Convergence criteria met: |dx| < xtol'
                self.logger.info(self.msg)
                success = False
                return success
            if (np.abs(self._fk - f_old) < self._ftol * np.abs(f_old)):
                self.msg = 'Convergence criteria met: |f(x+dx) - f(x)| < ftol * |f(x)|'
                self.logger.info(self.msg)
                success = False
                return success

            # Check for custom convergence
            if callable(self.convergence_criteria):
                if self.convergence_criteria(self):
                    self.logger('Custom convergence criteria met. Stopping optimization.')
                    success = False
                    return success

            # Update the trust region radius
            delta_old = self.trust_radius
            if (self.rho >= self.eta2) and hits_boundary: 
                delta_new = min(self.gam2*delta_old, self.trust_radius_max)
            elif self.eta1 <= self.rho < self.eta2:
                delta_new = delta_old
            else:
                delta_new = self.gam1*delta_old
            
            # Log new trust-radius
            self.trust_radius = np.clip(delta_new, self.trust_radius_min, self.trust_radius_max)
            if not (delta_old == delta_new):
                self.logger(f'Tr-radius {delta_k_symbol} updated: {delta_old:<10.4e} ───> {delta_new:<10.4e}')
 
            # check for convergence
            if self.iteration == self.max_iter:
                success = False
            else:
                # Calculate the jacobian and hessian
                self._jk = self.jac(self._xk)

                if self.quasi_newton:
                    yk = self._jk - j_old
                    if self.iteration==1 and self._Hk is None:
                        self._Hk = np.dot(yk, yk) / np.dot(yk, sk) * np.eye(self._xk.size) 

                    self._Hk = self.bfgs_update(
                        Bk = self._Hk, 
                        sk = sk, 
                        yk = yk)
                else:
                    self._Hk = self.hess(self._xk)
            
            # Update iteration
            self.iteration += 1

        else:
            if inner_iter < self.trust_radius_cuts:
                
                # Log the failure
                self.logger(f'Step not successful: {rho_symbol} = {self.rho:<10.4e} < {self.rho_tol:<10.4e}')
                
                # Reduce trust region radius to 75% of current value
                self.logger(f'Reducing {delta_k_symbol} by 75%: {self.trust_radius:<10.4e} ───> {0.25*self.trust_radius:<10.4e}')
                self.trust_radius = 0.25*self.trust_radius

                if self.trust_radius < self.trust_radius_min:
                    self.msg = f'Tr-radius {delta_k_symbol} <= minimum {delta_k_symbol}'
                    self.logger(f'Trust radius {self.trust_radius:<10.4e} is below minimum {self.trust_radius_min:<10.4e}. Stopping optimization.')
                    success = False
                    return success

                # Check for resampling of Jac and Hess
                if self.resample:
                    self.logger('Resampling gradient and hessian')
                    self._jk = self.jac(self._xk)

                    if not self.quasi_newton:
                        self._Hk = self.hess(self._xk)

                # Recursivly call function
                success = self.calc_update(inner_iter=inner_iter+1)

            else:
                success = False

        return success
    
    def bfgs_update(self, Bk, sk, yk):
        sk = sk.reshape(-1, 1)
        yk = yk.reshape(-1, 1)
        term1 = (yk @ yk.T) / (yk.T @ sk)
        term2 = (Bk @ sk @ sk.T @ Bk) / (sk.T @ Bk @ sk)
        Bk_new = Bk + term1 - term2
        return Bk_new

    def get_intermediate_results(self):
        # Define default results
        results = {
            'fun': self._fk,
            'x': self._xk, 
            'jac': self._jk,
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


                

                    







        

