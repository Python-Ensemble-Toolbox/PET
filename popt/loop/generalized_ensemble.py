# External imports
import numpy as np
import scipy.stats as stats
import sys
import warnings

from copy import deepcopy

# Internal imports
from popt.misc_tools import optim_tools as ot
from pipt.misc_tools import analysis_tools as at
from popt.loop.base import EnsembleOptimizationBase

class GeneralizedEnsemble(EnsembleOptimizationBase):

    def __init__(self, kwargs_ens, sim, obj_func):
        '''
        Parameters
        ----------
        kwargs_ens : dict
            Options for the ensemble class
        
        sim : callable
            The forward simulator (e.g. flow)
        
        obj_func : callable
            The objective function (e.g. npv)
        '''
        super().__init__(kwargs_ens, sim, obj_func)

        self.dim   = self.get_state().size

        # construct corr matrix
        std = np.sqrt(np.diag(self.cov))
        self.corr = self.cov/np.outer(std, std)

        # choose marginal
        marginal = kwargs_ens.get('marginal', 'Beta')

        if marginal in ['Beta', 'BetaMC', 'Logistic']:

            if marginal == 'Beta':
                self.margs = Beta()
                self.theta = kwargs_ens.get('theta', np.array([[20.0, 20.0] for _ in range(self.dim)]))
                self.eps = self.var2eps()
                self.grad_scale = 1/(2*self.eps)
                self.hess_scale = 1/(4*self.eps**2)

            elif marginal == 'BetaMC':
                lb, ub = np.array(self.bounds).T
                self.margs = BetaMC(lb, ub)
                state = self.get_state()
                var = np.diag(self.cov)
                #default_theta = self.margs.var_to_concentration(state, var)
                default_theta = np.array([var_to_concentration(state[i], var[i], lb[i], ub[i]) for i in range(self.dim)])
                #print(type(default_theta[0]))
                #print(type(default_theta_old[0]))
                #stop
                self.theta = kwargs_ens.get('theta', default_theta)
                self.grad_scale = 1.0
                self.hess_scale = 1.0
                
            elif marginal == 'Logistic':
                self.margs = Logistic()
                self.theta = kwargs_ens.get('theta', self.margs.var_to_scale(np.diag(self.cov)))
                self.grad_scale = 1.0
                self.hess_scale = 1.0
    
    def get_theta(self):
        return self.theta
    
    def get_corr(self):
        return self.corr
    
    def sample(self, size=None):

        if size is None:
            size = self.num_samples
        #enZ = stats.qmc.MultivariateNormalQMC(np.zeros(self.dim), self.corr).random(n=size)
        enZ = np.random.multivariate_normal(np.zeros(self.dim), self.corr, size=size)
        enX = self.margs.ppf(stats.norm.cdf(enZ), self.theta, mean=self.get_state())

        return enX, enZ
    
    def gradient(self, x, *args, **kwargs):

        # Set the ensemble state equal to the input control vector x
        self.state = ot.update_optim_state(x, self.state, list(self.state.keys()))

        if args:
            self.theta, self.corr = args

        self.enZ = kwargs.get('enZ', None)
        self.enX = kwargs.get('enX', None)
        self.enJ = kwargs.get('enJ', None)

        ne  = self.num_samples
        nr  = self._aux_input()
        dim = self.dim

        # Sample
        if (self.enX is None) or (self.enZ is None):
            self.enX, self.enZ = self.sample(size=ne)
        
        # Evaluate
        if self.enJ is None:
            self.enJ = self.function(self._trafo_ensemble(x).T)

        self.avg_hess = np.zeros((dim,dim))
        self.avg_grad = np.zeros(dim)
        self.nat_grad = np.zeros_like(self.theta)

        H = np.linalg.inv(self.corr)-np.eye(dim)
        O = np.ones((dim,dim))-np.eye(dim)
        enJ = self.enJ - np.array(np.repeat(self.state_func_values, nr))

        for n in range(self.ne):

            X = self.enX[n]
            Z = self.enZ[n]
            
            rho = self.margs.pdf(X, self.theta, mean=x)/stats.norm.pdf(Z)   # p(X)/φ(Z)
            G = self.margs.grad_log_pdf(X, self.theta, mean=x)              # ∇log(p)
            K = self.margs.hess_log_pdf(X, self.theta, mean=x)              # ∇²log(p) 
            D = - rho*np.matmul(H,Z)                                        # ∇log(c)
            M_ii = (G+rho*Z)*D - np.diag(H)*rho**2
            M_ij = - np.outer(rho,rho)*H
            M = np.diag(M_ii) + M_ij*O                                      # ∇²log(c) 
            
            # calc grad and hess
            self.avg_grad += enJ[n] * (G + D)
            self.avg_hess += enJ[n] * (np.outer(G+D,G+D) + np.diag(K)+M)

        self.avg_grad = -self.avg_grad*self.grad_scale/ne
        self.avg_hess = self.avg_hess*self.hess_scale/ne

        return self.avg_grad

    def hessian(self, x, *args, **kwargs):

        # Set the ensemble state equal to the input control vector x
        self.state = ot.update_optim_state(x, self.state, list(self.state.keys()))

        if kwargs.get('sample', False): 
            self.gradient(x, *args, **kwargs)
        
        return self.avg_hess
        
    def var2eps(self):
        var = np.diag(self.cov)
        a = self.theta[:,0]
        b = self.theta[:,1]

        frac    = a*b / ( (a+b)**2 * (a+b+1) )
        epsilon = np.sqrt(0.25*var/frac)
        return epsilon
    
    def _trafo_ensemble(self, x):

        if self.margs.name == 'Beta':
            lb, ub = np.array(self.bounds).T
            return epsilon_trafo(x, self.enX, self.eps, lb, ub)
        else:
            return self.enX


class BetaMC:

    def __init__(self, lb=0, ub=1):
        self.name = 'BetaMC'
        self.lb = lb
        self.ub = ub

    def pdf(self, x, theta, **kwargs):
        mode = kwargs.get('mean')
        mode = (mode-self.lb)/(self.ub-self.lb)
        concentration = theta

        a = mode*concentration+1
        b = (1-mode)*concentration+1

        return stats.beta(a,b, loc=self.lb, scale=self.ub-self.lb).pdf(x)

    def ppf(self, x, theta, **kwargs):
        mode = kwargs.get('mean')
        mode = (mode-self.lb)/(self.ub-self.lb)
        concentration = theta

        a = mode*concentration+1
        b = (1-mode)*concentration+1
        return stats.beta(a,b, loc=self.lb, scale=self.ub-self.lb).ppf(x)
    
    def grad_log_pdf(self, x, theta, **kwargs):

        mode = kwargs.get('mean')
        mode = (mode-self.lb)/(self.ub-self.lb)
        concentration = theta

        a = mode*concentration+1
        b = (1-mode)*concentration+1

        u = (x-self.lb)/(self.ub-self.lb)
        d_log_p = (a-1)/u - (b-1)/(1-u)
        return d_log_p/(self.ub-self.lb)
    
    def hess_log_pdf(self, x, theta, **kwargs):

        mode = kwargs.get('mean')
        mode = (mode-self.lb)/(self.ub-self.lb)
        concentration = theta

        a = mode*concentration+1
        b = (1-mode)*concentration+1

        u = (x-self.lb)/(self.ub-self.lb)
        h_log_p = -(a-1)/u**2 - (b-1)/(1-u)**2
        return h_log_p/((self.ub-self.lb)**2)
    
    def var_to_concentration(self, var, mode):
        from scipy.optimize import fsolve
        kappa = []
        for i, v in enumerate(var):
            m = (mode[i]-self.lb[i])/(self.ub[i]-self.lb[i])
            v = v/(self.ub[i]-self.lb[i])**2
            def func(k):
                a = m*k+1
                b = (1-m)*k+1
                return v - a*b/((a+b)**2 * (a+b+1))
            
            kappa.append(fsolve(func, 50)[0])

        return np.array(kappa)


class Beta:

    name = 'Beta'

    def pdf(self, x, theta, **kwargs):
        a, b = theta.T
        return stats.beta(a,b).pdf(x)
    
    def ppf(self, u, theta, **kwargs):
        a, b = theta.T
        return stats.beta(a,b).ppf(u)
    
    def grad_log_pdf(self, x, theta, **kwargs):
        a, b = theta.T
        return (a-1)/x - (b-1)/(1-x)

    def hess_log_pdf(self, x, theta, **kwargs):
        a, b = theta.T
        return -(a-1)/x**2 - (b-1)/(1-x)**2


class Logistic:

    name = 'Logistic'

    def pdf(self, x, theta, **kwargs):
        scale = theta
        loc = kwargs.get('mean', 0)
        return stats.logistic(loc=loc, scale=scale).pdf(x)
    
    def ppf(self, u, theta, **kwargs):
        scale = theta
        loc = kwargs.get('mean', 0)
        return stats.logistic(loc=loc, scale=scale).ppf(u)

    def grad_log_pdf(self, x, theta,**kwargs):
        scale = theta
        loc = kwargs.get('mean', 0)
        u = (x-loc)/(2*scale)
        return - np.tanh(u)/scale
    
    def hess_log_pdf(self, x, theta,**kwargs):
        scale = theta
        loc = kwargs.get('mean', 0)
        u = (x-loc)/(2*scale)
        return -(1/np.cosh(u))**2/(2*scale**2)
    
    def var_to_scale(self, var):
        return np.sqrt( 3*var/(np.pi**2) )
        
        

def epsilon_trafo(x, enX, eps, lower=None, upper=None):

    if not (lower is None and upper is None):
        Psi = x + (enX-0.5)*np.minimum(2*eps, upper-lower)
        enY = Psi + np.maximum(0, lower-(x-eps)) - np.maximum(0, x+eps - upper)
    else:
        enY = x + 2*eps*(enX-0.5)
    
    return enY


from sympy import symbols, solve, im, re
def var_to_concentration(mode, var, lb=0, ub=1):

    mode = (mode-lb)/(ub-lb)
    var  = var/(ub-lb)**2

    if var >= 1/12:
        warnings.warn('Maximum variance for Beta distribution is 1/12. The variance is set to 0.08.')
        var = 0.08

    m, c, v = symbols('m c v')

    # define alpha and beta
    a = m*c+1
    b = (1-m)*c+1

    # define the variance expression
    var_expr = a*b/((a+b)**2 * (a+b+1))
    equation = var_expr - v
    equation = equation.subs({m:mode, v:var})

    # solve for c
    solution = solve(equation, c)
   
    # check if imaginary part is zero
    for i, sol in enumerate(solution):
        is_real = im(sol).evalf() < 1e-10
        if is_real:
            solution[i] = re(sol).evalf()
        else:
            solution[i] = np.nan

    # convert to numpy array
    solution = np.array(solution).astype(np.float64)

    # return the positive solution
    return np.max(solution)