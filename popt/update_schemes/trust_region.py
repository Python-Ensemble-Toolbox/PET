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
    
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the current parameter vector.

    options : dict, optional

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
        if self.bounds is None:
            init_trust_radius = 1.0
        else:
            mean_bound_range  = np.mean([b[1]-b[0] for b in self.bounds])
            init_trust_radius = np.sqrt(x.size)*mean_bound_range
            
        self.trust_radius     = options.get('trust_radius', init_trust_radius) 
        self.trust_radius_max = options.get('trust_radius_max', 10*self.trust_radius)
        self.trust_radius_min = options.get('trust_radius_min', self.trust_radius/1000)

        # Set other options
        #self.ftol = options.get('ftol', 1e-6)
        #self.xtol = options.get('xtol', 1e-4)
        self.eta  = options.get('eta', 1e-6)
        self.eps1 = options.get('eps1', 0.1)
        self.eps2 = options.get('eps2', 0.5)
        self.eps3 = options.get('eps3', 0.5)
        self.rho  = 0.0

        self.normalize = options.get('normalize', False)
        self.resample  = options.get('resample', 0)
        self.saveit    = options.get('saveit', True)

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

            if self.logger is not None:
                self.logger.info(f'       ====== Running optimization - Trust Region ======')
                self.logger.info('\n'+pprint.pformat(OptimizeResult(self.options)))
                self.logger.info(f'       {"iter.":<10} {"fun":<15} {"tr-radius":<15} {"rho":<15}')
                self.logger.info(f'       {self.iteration:<10} {self.fk:<15.4e} {self.trust_radius:<15.4e}  {self.rho:<15.4e}')
                self.logger.info('')
        
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

    def calc_update(self):

        # Initialize variables for this step
        success = True
        fun_old = self.fk

        # Solve subproblem
        sk = self.solve_sub_problem_CG_Steihaug(self.jk, self.Hk, self.trust_radius)

        # Calculate the actual function value
        fun_new = self._fun(self.xk + sk)

        # Calculate rho
        actual_reduction    = self.fk - fun_new
        predicted_reduction = - np.dot(self.jk, sk) - 0.5*np.dot(sk, np.dot(self.Hk, sk))
        self.rho = actual_reduction/predicted_reduction

        if self.rho > self.eta:
            
            # Update the control
            self.xk = ot.clip_state(self.xk + sk, self.bounds)
            self.fk = fun_new

            # Update trust region radius
            if np.linalg.norm(sk) <= 0.8*self.trust_radius:
                self.trust_radius = 0.8*self.trust_radius
            else:  
                if self.rho > self.eps3:
                    self.trust_radius = min(self.trust_radius_max, 1.5*self.trust_radius)          
                elif self.eps1 <= self.rho <= self.eps2:
                    self.trust_radius = self.trust_radius
                else:
                    self.trust_radius = 0.95*self.trust_radius

        else:
            self.trust_radius = 0.5*self.trust_radius
    

        # Update results if we hav improvement in the function value
        if fun_new < fun_old:
            self.optimize_result = self.update_results()
        # Save results
        if self.saveit:
            ot.save_optimize_results(self.optimize_result)

        # Write logging info
        if self.logger is not None:
            self.logger.info('')
            self.logger.info(f'       {"iter.":<10} {"fun":<15} {"tr-radius":<15} {"rho":<15}')
            self.logger.info(f'       {self.iteration:<10} {self.fk:<15.4e} {self.trust_radius:<15.4e}  {self.rho:<15.4e}')
            self.logger.info('')
        
        # Call the callback function
        if callable(self.callback):
            self.callback(self) 

        # Calculate the jacobian and hessian
        self.jk = self._jac(self.xk)
        self.Hk = self._hess(self.xk)

        # Update iteration
        self.iteration += 1

        # check for convergence
        if self.trust_radius < self.trust_radius_min:
            success = False

        return success

    
    def solve_sub_problem_CG_Steihaug(self, g, B, delta, tol=1e-5, maxiter=1000):
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
        n = len(g)
        p = np.zeros(n)
        r = g.copy()
        d = -r
        rTr = np.dot(r, r)

        for k in range(maxiter):
            Bd  = np.dot(B, d)
            dBd = np.dot(d, Bd)

            if dBd <= 0:
                tau = - (np.dot(g,d) + np.dot(p,np.dot(B,d))) / (np.dot(d,np.dot(B,d)))
                if np.linalg.norm(p + tau*d) > delta:
                    tau = (-np.dot(p,d) + np.sqrt(np.dot(p,d)**2 - np.dot(d,d) * (np.dot(p,p) - delta**2))) / np.dot(d,d)
                    
                p = p + tau*d
                p = ot.clip_state(self.xk + p, self.bounds) - self.xk
                break

            alpha = rTr/dBd
            p_next = p + alpha * d

            if np.linalg.norm(p_next) >= delta:
                tau = (-np.dot(p, d) + np.sqrt(np.dot(p, d)**2 - np.dot(d, d) * (np.dot(p, p) - delta**2))) / np.dot(d, d)
                p = p + tau * d
                p = ot.clip_state(self.xk + p, self.bounds) - self.xk
                break

            p = p = ot.clip_state(self.xk + p_next, self.bounds) - self.xk
            r_next = r + alpha * Bd

            if np.linalg.norm(r_next) < tol:
                break

            beta = np.dot(r_next, r_next) / rTr
            d = -r_next + beta * d
            r = r_next
            rTr = np.dot(r, r)

        return p
             

                

                    







        

