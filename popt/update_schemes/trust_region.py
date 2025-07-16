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

# Impors from scipy
from scipy.optimize._trustregion_ncg import CGSteihaugSubproblem
from scipy.optimize._trustregion_exact import IterativeSubproblem


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
        self.xk       = x
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
            self.convergence_criteria = self.convergence_criteria
        else:
            self.convergence_criteria = None

        # Set options for trust-region radius       
        self.trust_radius      = options.get('trust_radius', 1.0) 
        self.trust_radius_max  = options.get('trust_radius_max', 10*self.trust_radius)
        self.trust_radius_min  = options.get('trust_radius_min', self.trust_radius/100)
        self.trust_radius_cuts = options.get('trust_radius_cuts', 4)

        # Set other options
        self.resample = options.get('resample', False)
        self.saveit   = options.get('saveit', True)
        self.rho_tol  = options.get('rho_tol', 1e-6)
        self.eta1 = options.get('eta1', 0.1) # reduce raduis if rho < 10%
        self.eta2 = options.get('eta2', 0.5)  # increase radius if rho > 50% 
        self.gam1 = options.get('gam1', 0.5)  # reduce by 50%
        self.gam2 = options.get('gam2', 1.5)  # increase by 50%
        self.rho  = 0.0

        # Check if method is valid
        if self.method not in ['iterative', 'CG-Steihaug']:
            self.method = 'iterative'
            raise ValueError(f'Method {self.method} is not valid!. Method is set to "iterative"')
    
            
        if not self.restart:
            self.start_time = time.perf_counter()

            # Check for initial callable values
            self.fk = options.get('fun0', None)
            self.jk = options.get('jac0', None)
            self.Hk = options.get('hess0', None)

            if self.fk is None: self.fk = self._fun(self.xk)
            if self.jk is None: self.jk = self._jac(self.xk)
            if self.Hk is None: self.Hk = self._hess(self.xk)

            # Initial results
            self.optimize_result = self.update_results()
            if self.saveit:
                ot.save_optimize_results(self.optimize_result)

            self._log(f'       ====== Running optimization - Trust Region ======')
            self._log('\n'+pprint.pformat(OptimizeResult(self.options)))
            self._log(f'       {"iter.":<10} {"fun":<15} {"tr-radius":<15} {"rho":<15}')
            self._log(f'       {self.iteration:<10} {self.fk:<15.4e} {self.trust_radius:<15.4e}  {self.rho:<15.4e}')
            self._log('')
        
        # Run the optimization
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
        return h
    
    def update_results(self):
        res = {
            'fun': self.fk, 
            'x': self.xk, 
            'jac': self.jk,
            'hess': self.Hk,
            'nfev': self.nfev,
            'njev': self.njev,
            'nit': self.iteration,
            'trust_radius': self.trust_radius,
            'save_folder': self.options.get('save_folder', './')
        }
        
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

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def solve_subproblem(self, g, B, delta):
        """
        Solve the trust region subproblem using the iterative method.
        (A big thanks to copilot for the help with this implementation)

        Parameters:
        g (numpy.ndarray): Gradient vector at the current point.
        B (numpy.ndarray): Hessian matrix at the current point.
        delta (float): Trust region radius.

        Returns:
        pk (numpy.ndarray): Step direction.
        pk_hits_boundary (bool): True if the step hits the boundary of the trust region.
        """

        # Define quadratic model
        quad = lambda p: self.fk + np.dot(g,p) + np.dot(p,np.dot(B,p))/2


        if self.method == 'iterative':
            subproblem = IterativeSubproblem(
                x=self.xk, 
                fun=quad, 
                jac=lambda _: g, 
                hess=lambda _: B,
            )
            pk, pk_hits_boundary = subproblem.solve(tr_radius=delta)
        
        elif self.method == 'CG-Steihaug':
            subproblem = CGSteihaugSubproblem(
                x=self.xk, 
                fun=quad, 
                jac=lambda _: g, 
                hess=lambda _: B,
            )
            pk, pk_hits_boundary = subproblem.solve(trust_radius=delta)

        else:
            raise ValueError(f"Method {self.method} is not valid!")

        return pk, pk_hits_boundary


    def calc_update(self, inner_iter=0):

        # Initialize variables for this step
        success = True

        # Solve subproblem
        self._log('Solving trust region subproblem')
        sk, hits_boundary = self.solve_subproblem(self.jk, self.Hk, self.trust_radius)

        # truncate sk to respect bounds
        if self.bounds is not None:
            lb = np.array(self.bounds)[:, 0]
            ub = np.array(self.bounds)[:, 1]
            sk = np.clip(sk, lb - self.xk, ub - self.xk)

        # Calculate the actual function value
        xk_new  = self.xk + sk
        fun_new = self._fun(xk_new)

        # Calculate rho
        actual_reduction    = self.fk - fun_new
        predicted_reduction = - np.dot(self.jk, sk) - np.dot(sk, np.dot(self.Hk, sk))/2
        self.rho = actual_reduction/predicted_reduction

        if self.rho > self.rho_tol:
            
            # Update the control
            self.xk = xk_new
            self.fk = fun_new

            # Save Results
            self.optimize_result = self.update_results()
            if self.saveit:
                ot.save_optimize_results(self.optimize_result)

            # Write logging info
            self._log('')
            self._log(f'       {"iter.":<10} {"fun":<15} {"tr-radius":<15} {"rho":<15}')
            self._log(f'       {self.iteration:<10} {self.fk:<15.4e} {self.trust_radius:<15.4e}  {self.rho:<15.4e}')
            self._log('')

            # Call the callback function
            if callable(self.callback):
                self.callback(self)

            # update the trust region radius
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
                self._log(f'Trust-radius updated: {delta_old:<10.4e} --> {delta_new:<10.4e}')

            # Check for custom convergence
            if callable(self.convergence_criteria):
                if self.convergence_criteria(self):
                    self._log('Custom convergence criteria met. Stopping optimization.')
                    success = False
                    return success

            # check for convergence
            if self.iteration==self.max_iter:
                success = False
            else:
                # Calculate the jacobian and hessian
                self.jk = self._jac(self.xk)
                self.Hk = self._hess(self.xk)
            
            # Update iteration
            self.iteration += 1

        else:
            if inner_iter < self.trust_radius_cuts:
                
                # Log the failure
                self._log(f'Step not successful: rho < {self.rho_tol:<10.4e}')
                
                # Reduce trust region radius to 75% of current value
                self._log('Reducing trust-radius by 75%')
                self.trust_radius = 0.25*self.trust_radius

                if self.trust_radius < self.trust_radius_min:
                    self._log(f'Trust radius {self.trust_radius} is below minimum {self.trust_radius_min}. Stopping optimization.')
                    success = False
                    return success

                # Check for resampling of Jac and Hess
                if self.resample:
                    self._log('Resampling gradient and hessian')
                    self.jk = self._jac(self.xk)
                    self.Hk = self._hess(self.xk)

                # Recursivly call function
                success = self.calc_update(inner_iter=inner_iter+1)

            else:
                success = False

        return success

                

                    







        

