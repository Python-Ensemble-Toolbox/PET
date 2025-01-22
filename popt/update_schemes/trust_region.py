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
        self.trust_radius_min = options.get('trust_radius_min', self.trust_radius/100)

        # Set other options
        #self.ftol = options.get('ftol', 1e-6)
        #self.xtol = options.get('xtol', 1e-4)
        self.eta1 = options.get('eta1', 1e-6)
        self.eta2 = options.get('eta2', 0.1)
        self.gam1 = options.get('gam1', 0.8)
        self.gam2 = options.get('gam2', 1.5)
        self.rho  = 0.0

        self.normalize = options.get('normalize', False)
        self.resample  = options.get('resample', 3)
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
        fun_old = self.fk

        # Solve subproblem
        self._log('Solving trust region subproblem')
        sk = self.solve_sub_problem_CG_Steihaug(self.jk, self.Hk, self.trust_radius)

        # Calculate the actual function value
        fun_new = self._fun(self.xk + sk)

        # Calculate rho
        actual_reduction    = self.fk - fun_new
        predicted_reduction = - np.dot(self.jk, sk) - 0.5*np.dot(sk, np.dot(self.Hk, sk))
        self.rho = actual_reduction/predicted_reduction

        if self.rho > self.eta1:
            
            # Update the control
            self.xk = ot.clip_state(self.xk + sk, self.bounds)
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
            if self.rho >= self.eta2:
                self.trust_radius = min(self.gam2*self.trust_radius, self.trust_radius_max)
            elif self.eta1 <= self.rho < self.eta2:
                self.trust_radius = self.trust_radius
            else:
                self.trust_radius = self.gam1*self.trust_radius

            # check for convergence
            if self.trust_radius < self.trust_radius_min:
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

                self._log('Resampling gradient and hessian')
                # Calculate the jacobian and hessian
                self.jk = self._jac(self.xk)
                self.Hk = self._hess(self.xk)

                self._log('Reducing trust region radius by 50%')
                # Reduce trust region radius to 50% of current value
                self.trust_radius = 0.5*self.trust_radius

                # Recursivly call function
                success = self.calc_update(iter_resamp=iter_resamp)

            else:
                success = False

        return success

    
    def solve_sub_problem_CG_Steihaug(self, g, B, delta, tol=1e-4, maxiter=1000):
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

        # make quadratic model
        mc = lambda p: np.dot(g,p) + np.dot(p,np.dot(B,p))/2

        for k in range(maxiter):
            dBd = np.dot(d, np.dot(B, d))

            if dBd <= 0:
                # Solve the quadratic equation: (p + tau*d)**2 = delta**2
                tau_values   = np.roots([np.dot(d,d), 2*np.dot(p,d), np.dot(p,p) - delta**2])
                model_values = [mc(p + t*d) for t in tau_values]
                tau = tau_values[np.argmin(model_values)]
                    
                p = p + tau*d
                p = ot.clip_state(self.xk + p, self.bounds) - self.xk
                break

            alpha = np.dot(r,r)/dBd
            p_new = p + alpha * d

            if np.linalg.norm(p_new) >= delta:

                # Solve the quadratic equation: (p + tau*d)**2 = delta**2, for tau > 0
                tau_values = np.roots([np.dot(d,d), 2*np.dot(p,d), np.dot(p,p) - delta**2])
                tau = np.max(tau_values[np.where(tau_values > 0)[0]])
                p = p + tau * d
                p = ot.clip_state(self.xk + p, self.bounds) - self.xk
                break

            r_new = r + alpha*np.dot(B, d)

            if np.linalg.norm(r_new) < tol*np.linalg.norm(r_new):
                p = ot.clip_state(self.xk + p_new, self.bounds) - self.xk
                break

            beta = np.dot(r_new, r_new)/np.dot(r,r)
            d = -r_new + beta * d
            r = r_new
            p = p_new

        return p
             

                

                    







        

