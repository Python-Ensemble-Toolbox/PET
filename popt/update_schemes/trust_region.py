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


def TrustRegion(fun, x, jac, hess, args=(), bounds=None, callback=None, **options):
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
    
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives (Jacobian, Hessian).
    
    bounds : sequence, optional
        Bounds for variables. Each element of the sequence must be a tuple of two scalars, 
        representing the lower and upper bounds for that variable. Use None for one of the bounds if there are no bounds.
    
    callback: callable, optional
        A callable called after each successful iteration. The class instance of LineSearch 
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

    save_folder: str
        Name of folder to save the results to. Defaul is ./ (the current directory).

    fun0: float
        Function value of the intial control.

    jac0: ndarray
        Jacobian of the initial control.
    
    hess0: ndarray
        Hessian value of the initial control.

    resample: int
        Number of jacobian re-computations allowed if a line search fails. Default is 4.
        (useful if jacobian is stochastic)

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
    tr_obj = TrustRegionClass(fun, x, jac, hess, args, bounds, callback, **options)
    return tr_obj.optimize_result

class TrustRegionClass(Optimize):

    def __init__(self, fun, x, jac, hess, args=(), bounds=None, callback=None, **options):
        
        # Initialize the parent class
        super().__init__(**options)

        # Set class attributes
        self.function = fun
        self.xk       = x
        self.jacobian = jac
        self.hessian  = hess
        self.args     = args
        self.bounds   = bounds
        self.options  = options

        # Check if the callback function is callable
        if callable(callback):
            self.callback = callback
        else:
            self.callback = None

        # Set options for trust-region radius       
        self.trust_radius     = options.get('trust_radius', 1.0) 
        self.trust_radius_max = options.get('trust_radius_max', 10*self.trust_radius)
        self.trust_radius_min = options.get('trust_radius_min', self.trust_radius/100)

        # Set other options
        self.resample = options.get('resample', 3)
        self.saveit   = options.get('saveit', True)
        self.rho_tol  = options.get('rho_tol', 1e-6)
        self.eta1 = options.get('eta1', 0.001)
        self.eta2 = options.get('eta2', 0.1)
        self.gam1 = options.get('gam1', 0.7)
        self.gam2 = options.get('gam2', 1.5)
        self.rho  = 0.0

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
        res = {'fun': self.fk, 
               'x': self.xk, 
               'jac': self.jk,
               'hess': self.Hk,
               'nfev': self.nfev,
               'njev': self.njev,
               'nit': self.iteration,
               'trust_radius': self.trust_radius,
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

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def calc_update(self, iter_resamp=0):

        # Initialize variables for this step
        success = True

        # Solve subproblem
        self._log('Solving trust region subproblem using the CG-Steihaug method')
        sk = self.solve_sub_problem_CG_Steihaug(self.jk, self.Hk, self.trust_radius)

        # Calculate the actual function value
        xk_new = ot.clip_state(self.xk + sk, self.bounds)
        fun_new = self._fun(xk_new)

        # Calculate rho
        actual_reduction    = self.fk - fun_new
        predicted_reduction = - np.dot(self.jk, sk) - 0.5*np.dot(sk, np.dot(self.Hk, sk))
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
            if self.rho >= self.eta2:
                delta_new = min(self.gam2*delta_old, self.trust_radius_max)
            elif self.eta1 <= self.rho < self.eta2:
                delta_new = delta_old
            else:
                delta_new = self.gam1*delta_old
            
            # Log new trust-radius
            self.trust_radius = delta_new 
            if not (delta_old == delta_new):
                self._log(f'Trust-radius updated: {delta_old:<10.4e} --> {delta_new:<10.4e}')

            # check for convergence
            if (self.trust_radius < self.trust_radius_min) or (self.iteration==self.max_iter):
                success = False
            else:
                # Calculate the jacobian and hessian
                self.jk = self._jac(self.xk)
                self.Hk = self._hess(self.xk)
            
            # Update iteration
            self.iteration += 1

        else:
            if iter_resamp < self.resample:

                iter_resamp += 1

                # Calculate the jacobian and hessian
                self._log('Resampling gradient and hessian')
                self.jk = self._jac(self.xk)
                self.Hk = self._hess(self.xk)

                # Reduce trust region radius to 50% of current value
                self._log('Reducing trust-radius by 50%')
                self.trust_radius = 0.5*self.trust_radius

                # Recursivly call function
                success = self.calc_update(iter_resamp=iter_resamp)

            else:
                success = False

        return success

    
    def solve_sub_problem_CG_Steihaug(self, g, B, delta):
        """
        Solve the trust region subproblem using Steihaug's Conjugate Gradient method.
        (A big thanks to copilot for the help with this implementation)

        Parameters:
        g (numpy.ndarray): Gradient vector at the current point.
        B (numpy.ndarray): Hessian matrix at the current point.
        delta (float): Trust region radius.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

        Returns:
        p (numpy.ndarray): Solution vector.
        """
        z = np.zeros_like(g)
        r = g
        d = -g

        # Set same default tolerance as scipy
        tol = min(0.5, la.norm(g)**2)*la.norm(g)

        if la.norm(g) <= tol:
            return z

        # make quadratic model
        mc = lambda s: self.fk + np.dot(g,s) + np.dot(s,np.dot(B,s))/2

        while True:
            dBd = np.dot(d, np.dot(B,d))

            if dBd <= 0:
                # Solve the quadratic equation: (p + tau*d)**2 = delta**2
                tau_lo, tau_hi = self.get_tau_at_delta(z, d, delta)
                p_lo = z + tau_lo*d
                p_hi = z + tau_hi*d

                if mc(p_lo) < mc(p_hi): 
                    return p_lo
                else:
                    return p_hi

            alpha = np.dot(r,r)/dBd
            z_new = z + alpha*d

            if la.norm(z_new) >= delta:
                # Solve the quadratic equation: (p + tau*d)**2 = delta**2, for tau > 0
                _ , tau = self.get_tau_at_delta(z, d, delta)
                return z + tau * d
                
            r_new = r + alpha*np.dot(B,d)

            if la.norm(r_new) < tol:
                return z_new

            beta = np.dot(r_new,r_new)/np.dot(r,r)
            d = -r_new + beta*d
            r = r_new
            z = z_new

    
    def get_tau_at_delta(self, p, d, delta):
        """
        Solve the quadratic equation: (p + tau*d)**2 = delta**2, for tau > 0
        """
        a = np.dot(d,d)
        b = 2*np.dot(p,d)
        c = np.dot(p,p) - delta**2
        tau_lo = -b/(2*a) - np.sqrt(b**2 - 4*a*c)/(2*a) 
        tau_hi = -b/(2*a) + np.sqrt(b**2 - 4*a*c)/(2*a) 
        return tau_lo, tau_hi
             

                

                    







        

