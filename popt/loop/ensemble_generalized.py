# External imports
import numpy as np
import scipy.stats as stats
import sys
import warnings

from copy import deepcopy
from scipy.special import polygamma

# Internal imports
from popt.misc_tools import optim_tools as ot
from pipt.misc_tools import analysis_tools as at
from popt.loop.ensemble_base import EnsembleOptimizationBaseClass

__all__ = ['GeneralizedEnsemble']

class GeneralizedEnsemble(EnsembleOptimizationBaseClass):

    def __init__(self, options, simulator, objective):
        '''
        Parameters
        ----------
        options : dict
            Options for the ensemble class
        
        simulator : callable
            The forward simulator (e.g. flow). If None, no simulation is performed.
        
        objective : callable
            The objective function (e.g. npv)
        '''
        super().__init__(options, simulator, objective)

        # construct corr matrix
        std = np.sqrt(np.diag(self.covX))
        self.corr = self.covX/np.outer(std, std)
        self.dim  = std.size

        # choose marginal
        marginal = options.get('marginal', 'BetaMC')

        if marginal in ['Beta', 'BetaMC', 'Logistic', 'TruncGaussian', 'Gaussian']:

            self.grad_scale = 1.0
            self.hess_scale = 1.0

            if marginal == 'Beta':
                self.margs = Beta()
                self.theta = options.get('theta', np.array([[20.0, 20.0] for _ in range(self.dim)]))
                self.eps = self.var2eps()
                self.grad_scale = 1/(2*self.eps)
                self.hess_scale = 1/(4*self.eps**2)

            elif marginal == 'BetaMC':
                lb, ub = np.array(self.bounds).T
                state = self.get_state()
                var = np.diag(self.cov)
                self.margs = BetaMC(lb, ub, 0.1*np.sqrt(var[0]))
                default_theta = np.array([var_to_concentration(state[i], var[i], lb[i], ub[i]) for i in range(self.dim)])
                self.theta = options.get('theta', default_theta)
                
            elif marginal == 'Logistic':
                self.margs = Logistic()
                self.theta = options.get('theta', self.margs.var_to_scale(np.diag(self.covX)))

            elif marginal == 'TruncGaussian':
                lb, ub = np.array(self.bounds).T
                self.margs = TruncGaussian(lb,ub)
                self.theta = options.get('theta', np.sqrt(np.diag(self.covX)))

            elif marginal == 'Gaussian':
                self.margs = Gaussian()
                self.theta = options.get('theta', np.sqrt(np.diag(self.covX)))
    
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
        enX = ot.clip_state(enX, self.bounds)
        return enX, enZ
    
    def gradient(self, x, *args, **kwargs):

        # Update state vector
        self.stateX = x

        if args:
            self.theta, self.corr = args

        self.enZ = kwargs.get('enZ', None)
        self.enX = kwargs.get('enX', None)
        self.enF = kwargs.get('enF', None)

        ne  = self.num_samples
        nr  = self._aux_input()
        dim = self.dim

        # Sample
        if (self.enX is None) or (self.enZ is None):
            self.enX, self.enZ = self.sample(size=ne)
        
        # Evaluate
        if self.enF is None:
            self.enF = self.function(self._trafo_ensemble(x).T)

        self.avg_hess = np.zeros((dim,dim))
        self.avg_grad = np.zeros(dim)

        H = np.linalg.inv(self.corr)-np.eye(dim)
        O = np.ones((dim,dim))-np.eye(dim)
        enF = self.enF - np.repeat(self.stateF, nr)

        for n in range(self.ne):

            X = self.enX[n]
            Z = self.enZ[n]
            
            # Marginal terms
            G = self.margs.grad_log_pdf(X, self.theta, mean=x)              # ∇log(p)
            K = self.margs.hess_log_pdf(X, self.theta, mean=x)              # ∇²log(p)

            # Copula terms
            rho  = self.margs.pdf(X, self.theta, mean=x)/stats.norm.pdf(Z)   # p(X)/φ(Z) 
            D    = - rho*np.matmul(H,Z)                                      # ∇log(c)
            M_ii = (G+rho*Z)*D - np.diag(H)*rho**2
            M_ij = - np.outer(rho,rho)*H
            M    = np.diag(M_ii) + M_ij*O                                    # ∇²log(c) 
            
            # calc grad and hess
            grad_log_p = G + D
            hess_log_p = np.diag(K)+M
            self.avg_grad += enF[n]*grad_log_p
            self.avg_hess += enF[n]*(np.outer(grad_log_p, grad_log_p) + hess_log_p)

        self.avg_grad = -self.avg_grad*self.grad_scale/ne
        self.avg_hess = self.avg_hess*self.hess_scale/ne

        return self.avg_grad

    def hessian(self, x, *args, **kwargs):

        # Update state vector
        self.stateX = x

        if kwargs.get('sample', False): 
            self.gradient(x, *args, **kwargs)
        
        return self.avg_hess
    
    def mutation_gradient(self, x, *args, **kwargs):
        # Set the ensemble state equal to the input control vector x
        self.state = ot.update_optim_state(x, self.state, list(self.state.keys()))

        if args:
            self.theta, self.corr = args

        self.enZ = kwargs.get('enZ', None)
        self.enX = kwargs.get('enX', None)
        self.enF = kwargs.get('enF', None)

        ne  = self.num_samples
        nr  = self._aux_input()
        dim = self.dim

        # Sample
        if (self.enX is None) or (self.enZ is None):
            self.enX, self.enZ = self.sample(size=ne)
        
        # Evaluate
        if self.enF is None:
            self.enF = self.function(self._trafo_ensemble(x).T)

        enF = self.enF - np.repeat(self.stateF, nr)

        self.nat_grad = np.zeros(dim)
        self.nat_hess = np.zeros(dim)
        for n in range(ne):

            X = self.enX[n]
            dm_log_p = self.margs.grad_theta_log_pdf(X, self.theta, mean=x)
            hm_log_p = self.margs.hess_theta_log_pdf(X, self.theta, mean=x)

            self.nat_grad += enF[n]*dm_log_p
            self.nat_hess += enF[n]*(hm_log_p + dm_log_p**2)
        
        # Fisher
        self.nat_grad = self.nat_grad/ne
        self.nat_hess = np.diag(self.nat_hess/ne)
        return self.nat_grad

    def mutation_hessian(self, x, *args, **kwargs):

        # Update state vector
        self.stateX = x

        if kwargs.get('sample', False): 
            self.gradient(x, *args, **kwargs)
        
        return self.nat_hess
        
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

    def __init__(self, lb=0, ub=1, eps=0.01):
        self.name = 'BetaMC'
        self.lb  = lb
        self.ub  = ub
        self.eps = eps

    def _mc_to_ab(self, m, c):
        a = 1 + c*m
        b = 1 + c*(1-m)
        return a, b
    
    def _get_mode(self, **kwargs):
        mode = kwargs.get('mean')
        mode = (mode-self.lb)/(self.ub-self.lb)
        mode = np.clip(mode, self.eps, 1-self.eps)
        return mode

    def pdf(self, x, theta, **kwargs):
        mode = self._get_mode(**kwargs)
        a, b = self._mc_to_ab(mode, theta)
        return stats.beta(a,b, loc=self.lb, scale=self.ub-self.lb).pdf(x)

    def ppf(self, u, theta, **kwargs):
        mode = self._get_mode(**kwargs)
        a, b = self._mc_to_ab(mode, theta)
        return stats.beta(a,b, loc=self.lb, scale=self.ub-self.lb).ppf(u)
    
    def grad_log_pdf(self, x, theta, **kwargs):
        u = (x-self.lb)/(self.ub-self.lb)
        m = self._get_mode(**kwargs)
        c = theta
        return c*m/u - c*(1-m)/(1-u)

    def hess_log_pdf(self, x, theta, **kwargs):
        u = (x-self.lb)/(self.ub-self.lb)
        m = self._get_mode(**kwargs)
        c = theta
        return -c*m/u**2 - c*(1-m)/(1-u)**2
    
    def grad_theta_log_pdf(self, x, theta, **kwargs):
        u = (x-self.lb)/(self.ub-self.lb)
        m = self._get_mode(**kwargs)
        c = theta
        return c*np.log(u/(1-u)) - c*kappa(m,c)
    
    def hess_theta_log_pdf(self, x, theta, **kwargs):
        m = self._get_mode(**kwargs)
        c = theta
        p1 = polygamma(1, 1+c*m)
        p2 = polygamma(1, 1+c*(1-m))
        return -c**2*(p1+p2)
    
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
        loc = kwargs.get('mean', 0)
        return stats.logistic(loc=loc, scale=theta).pdf(x)
    
    def ppf(self, u, theta, **kwargs):
        loc = kwargs.get('mean', 0)
        return stats.logistic(loc=loc, scale=theta).ppf(u)

    def grad_log_pdf(self, x, theta, **kwargs):
        loc = kwargs.get('mean', 0)
        u = (x - loc) / (2 * theta)
        return -np.tanh(u)/theta
    
    def hess_log_pdf(self, x, theta, **kwargs):
        loc = kwargs.get('mean', 0)
        u = (x - loc) / (2 * theta)
        return -1/(2*theta**2 * np.cosh(u)**2)
    
    def var_to_scale(self, var):
        return np.sqrt(3*var)/np.pi
    
class TruncGaussian:

    def __init__(self, lb=0, ub=1):
        self.name = 'TruncGaussian'
        self.lb = lb
        self.ub = ub

    def pdf(self, x, theta, **kwargs):
        mu = kwargs.get('mean')
        a, b = (self.lb - mu)/theta, (self.ub - mu)/theta
        return stats.truncnorm(a, b, loc=mu, scale=theta).pdf(x)

    def ppf(self, u, theta, **kwargs):
        mu = kwargs.get('mean')
        a, b = (self.lb - mu)/theta, (self.ub - mu)/theta
        return stats.truncnorm(a, b, loc=mu, scale=theta).ppf(u)
    
    def grad_log_pdf(self, x, theta, **kwargs):
        mu = kwargs.get('mean')
        return -(x - mu)/theta**2

    def hess_log_pdf(self, x, theta, **kwargs):
        return -1/theta**2
    
    def grad_theta_log_pdf(self, x, theta, **kwargs):
        mu  = kwargs.get('mean')
        sig = theta
        phi = lambda z: stats.norm.pdf(z)
        phi_d = phi((self.ub-mu)/sig) - phi((self.lb-mu)/sig) 
        Phi_d = stats.norm.cdf((self.ub-mu)/sig) - stats.norm.cdf((self.lb-mu)/sig) 
        return (x-mu)/sig**2 + phi_d/(sig*Phi_d)

    def hess_theta_log_pdf(self, x, theta, **kwargs):
        mu  = kwargs.get('mean')
        sig = theta
        phi = lambda z: stats.norm.pdf(z)

        a = self.lb
        b = self.ub

        phi_d = phi((b-mu)/sig) - phi((a-mu)/sig) 
        Phi_d = stats.norm.cdf((b-mu)/sig) - stats.norm.cdf((a-mu)/sig)
        ratio = phi_d/Phi_d

        return -1/sig**2 - mu*ratio/sig**3 + (ratio/sig)**2 + (b*phi((b-mu)/sig) - a*phi((a-mu)/sig))/(sig**3 * Phi_d)

class Gaussian:

    name = 'Gaussian'

    def pdf(self, x, theta, **kwargs):
        mu = kwargs.get('mean')
        return stats.norm(loc=mu, scale=theta).pdf(x)

    def ppf(self, u, theta, **kwargs):
        mu = kwargs.get('mean')
        return stats.norm(loc=mu, scale=theta).ppf(u)
    
    def grad_log_pdf(self, x, theta, **kwargs):
        mu = kwargs.get('mean')
        return -(x - mu)/theta**2

    def hess_log_pdf(self, x, theta, **kwargs):
        return -1/theta**2
        
        

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

def kappa(m,c):
    p1 = polygamma(0, 1+c*m)
    p2 = polygamma(0, 1+c*(1-m))
    return p1-p2