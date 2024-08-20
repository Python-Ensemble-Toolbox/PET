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
        self._supported = ['beta', 'logistic']
        marginal = kwargs_ens.get('marginal', 'beta')

        if marginal in self._supported:

            if marginal == 'beta':
                self.margs = Beta()
                self.theta = kwargs_ens.get('theta', np.array([[20.0, 20.0] for _ in self.dim]))
                self.eps = self.var2eps()
                self.grad_scale = 1/(2*self.eps)
                self.hess_scale = 1/(4*self.eps**2)
            else:
                self.margs = Logistic()
                self.theta = kwargs_ens.get('theta', self.margs.var_to_scale(np.diag(self.cov)))
                self.grad_scale = 1.0
                self.hess_scale = 1.0
    
    def sample(self, size=None):

        if size is None:
            size = self.num_samples

        enZ = np.random.multivariate_normal(np.zeros(self.dim), self.corr, size=size)
        enU = stats.norm.cdf(enZ)
        enX = self.ppf(enU)

        return enX, enZ
    
    def gradient(self, x, *args, **kwargs):

        # Set the ensemble state equal to the input control vector x
        self.state = ot.update_optim_state(x, self.state, list(self.state.keys()))

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

        avg_hess = np.zeros((dim,dim))
        avg_grad = np.zeros(dim)
        nat_grad = np.zeros_like(self.theta)

        H = np.linalg.inv(self.corr)-np.eye(dim)
        O = np.ones((dim,dim))-np.eye(dim)

        for n in range(self.ne):

            X = self.enX[n]
            Z = self.enZ[n]
            
            rho  = self.pdf(X)/stats.norm.pdf(Z)   # p(X)/φ(Z)
            G, K = self.grad_and_hess_log_p(X)     # ∇log(p) and ∇²log(p) 
            D = - rho*np.matmul(H,Z)               # ∇log(c)

            M_ii = (G+rho*Z)*D - np.diag(H)*rho**2
            M_ij = - np.outer(rho,rho)*H
            M = np.diag(M_ii) + M_ij*O              # ∇²log(c) 
            
            # calc grad and hess
            avg_grad += (self.enJ[n]-self.enJ.mean()) * (G + D) / (ne-1)
            avg_hess += (self.enJ[n]-self.enJ.mean()) * (np.outer(G+D,G+D) + K+M) / (ne-1)
        
        return -avg_grad*self.grad_scale

    def hessian(self, x, *args, **kwargs):
        pass

    def var2eps(self):
        var = np.diag(self.cov)
        a = self.theta[:,0]
        b = self.theta[:,1]

        frac    = a*b / ( (a+b)**2 * (a+b+1) )
        epsilon = np.sqrt(0.25*var/frac)
        return epsilon
    
    def _trafo_ensemble(self, x):

        if self.margs.name == 'Beta':
            return x+2*self.eps*(self.enX-0.5)
        else:
            return self.enX


class Beta:

    name = 'Beta'

    def pdf(self, x, theta, **kwargs):
        a, b = theta.T
        return stats.beta(a,b).pdf(x)
    
    def ppf(self, u, theta, **kwargs):
        a, b = theta.T
        return stats.beta(a,b).ppf(u)
    
    def grad_log_pdf(self, x, theta,**kwargs):
        a, b = theta.T
        return (a-1)/x - (b-1)/(1-x)

    def hess_log_pdf(self, x, theta, **kwargs):
        a, b = theta.T
        return -(a-1)/x**2 - (b-1)/(1-x)**2
    
class Logistic:

    name = 'Logistic'

    def pdf(self, x, theta, **kwargs):
        scale = theta
        mean  = kwargs.get('mean', 0)
        return stats.logistic(loc=mean, scale=scale).pdf(x)
    
    def ppf(self, u, theta, **kwargs):
        scale = theta
        mean  = kwargs.get('mean', 0)
        return stats.logistic(loc=mean, scale=scale).ppf(u)

    def grad_log_pdf(self, x, theta,**kwargs):
        scale = theta
        mean  = kwargs.get('mean', 0)
        u = (x-mean)/(2*scale)
        return - np.tanh(u)/scale
    
    def hess_log_pdf(self, x, theta,**kwargs):
        scale = theta
        mean  = kwargs.get('mean', 0)
        u = (x-mean)/(2*scale)
        return -(1/np.cosh(u))**2/(2*scale**2)
    
    def var_to_scale(self, var):
        return np.sqrt( 3*var/(np.pi**2) )
        
        