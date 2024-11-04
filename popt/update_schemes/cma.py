"""Covariance matrix adaptation (CMA)."""
import numpy as np
from popt.misc_tools import optim_tools as ot

class CMA:

    def __init__(self, ne, dim, alpha_mu=None, n_mu=None, alpha_1=None, alpha_c=None, corr_update=False, equal_weights=True):
        '''
        This is a rather simple simple CMA class

        Parameters
        ----------------------------------------------------------------------------------------------------------
        ne : int
            Ensemble size
        
        dim : int
            Dimensions of control vector
        
        alpha_mu : float
            Learning rate for rank-mu update. If None, value proposed in [1] is used.
        
        n_mu : int, `n_mu < ne`
            Number of best samples of ne, to be used for rank-mu update.
            Default is int(ne/2).
        
        alpha_1 : float
            Learning rate fro rank-one update. If None, value proposed in [1] is used.
        
        alpha_c : float
            Parameter (inverse if backwards time horizen)for evolution path update 
            in the rank-one update. See [1] for more info. If None, value proposed in [1] is used.

        corr_update : bool
            If True, CMA is used to update a correlation matrix. Default is False.
        
        equal_weights : bool
            If True, all n_mu members are assign equal weighting, `w_i = 1/n_mu`.
            If False, the weighting scheme proposed in [1], where `w_i = log(n_mu + 1)-log(i)`,
            and normalized such that they sum to one. Defualt is True.

        References
        ----------------------------------------------------------------------------------------------------------
        [1] Hansen, N. (2006). The CMA evolution strategy: a comparing review. 
            In J. Lozano, P. Larranaga, I. Inza & E. Bengoetxea (ed.), Towards a new evolutionary computation. 
            Advances on estimation of distribution algorithms (pp. 75--102) . Springer .     
        '''
        self.alpha_mu       = alpha_mu
        self.n_mu           = n_mu
        self.alpha_1        = alpha_1
        self.alpha_c        = alpha_c
        self.ne             = ne
        self.dim            = dim
        self.evo_path       = 0
        self.corr_update    = corr_update

        #If None is given, default values are used
        if self.n_mu is None:
            self.n_mu = int(self.ne/2)
        
        if equal_weights:
            self.weights = np.ones(self.n_mu)/self.n_mu
        else:
            self.weights = np.array([np.log(self.n_mu + 1)-np.log(i+1) for i in range(self.n_mu)])
            self.weights = self.weights/np.sum(self.weights)

        self.mu_eff = 1/np.sum(self.weights**2)
        self.c_cov  = 1/self.mu_eff * 2/(dim+2**0.5)**2 +\
                    (1-1/self.mu_eff)*min(1, (2*self.mu_eff-1)/((dim+2)**2+self.mu_eff))

        if self.alpha_1 is None:
            self.alpha_1  = self.c_cov/self.mu_eff
        if self.alpha_mu is None:
            self.alpha_mu = self.c_cov*(1-1/self.mu_eff)
        if self.alpha_c is None:
            self.alpha_c  = 4/(dim+4)
        
    def _rank_mu(self, X, J):
        '''
        Calculates the rank-mu matrix of CMA-ES.
        '''
        index   = J.argsort() # lowest (best) to highest (worst)
        Xsorted = (X[index[:self.n_mu]] - np.mean(X, axis=0)).T # shape (d, ne)
        weights = self.weights
        Cmu     = (Xsorted*weights)@Xsorted.T

        if self.corr_update: 
            Cmu = ot.cov2corr(Cmu)
        
        return Cmu

    def _rank_one(self, step):
        '''
        Calculates the rank-one matrix of CMA-ES.
        '''
        s = self.alpha_c
        self.evo_path = (1-s)*self.evo_path + np.sqrt(s*(2-s)*self.mu_eff)*step
        C1 = np.outer(self.evo_path, self.evo_path)

        if self.corr_update: 
            C1 = ot.cov2corr(C1)

        return C1
    
    def __call__(self, cov, step, X, J):
        '''
        Performs the CMA update.

        Parameters
        --------------------------------------------------
        cov : array_like, of shape (d, d)
            Current covariance or correlation matrix.
        
        step : array_like, of shape (d,)
            New step of control vector.
            Used to update the evolution path.

        X : array_like, of shape (n, d)
            Control ensemble of size n.
        
        J : array_like, of shape (n,)
            Objective ensemble of size n.
        
        Returns
        --------------------------------------------------
        out : array_like, of shape (d, d)
            CMA updated covariance (correlation) matrix.
        '''
        a_mu  = self.alpha_mu
        a_one = self.alpha_1 
        C_mu  = self._rank_mu(X, J)
        C_one = self._rank_one(step)
        
        cov   =  (1 - a_one - a_mu)*cov + a_one*C_one + a_mu*C_mu       
        return cov
