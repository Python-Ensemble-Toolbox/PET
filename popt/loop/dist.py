"""Descriptive description."""
# External imports 
import numpy as np
from scipy import stats
from scipy.special import polygamma, digamma

# Internal imports
from popt.misc_tools import optim_tools as ot


class GenOptDistribution:
    '''
    Class that contains all the operations on the mutation distribution of GenOpt
    '''
    def __init__(self, x, cov, theta0=[20.0, 20.0], func=None, ne=None):
        '''
        Parameters
        ----------
        x : array_like, shape (d,)
            Initial control vector. Used initally to get the dimensionality of the problem.
        
        cov : array_like, shape (d,d)
            Initial covaraince matrix. Used to construct the correlation matrix and 
            epsilon parameter of GenOpt
        
        theta0 : list, of length 2 ([alpha, beta])
            Initial alpha and beta parameter of the marginal Beta distributions.
        
        func : callable (optional)
            An objective function that can be used later for the gradeint.
            Can also be passed directly to the gradeint fucntion.
        
        ne : int
        '''
        self.dim   = x.size                         # dimension of state                 
        self.corr  = ot.cov2corr(cov)               # initial correlation
        self.var   = np.diag(cov)                   # initial varaince
        self.theta = np.tile(theta0, (self.dim,1))  # initial theta parameters, shape (dim, 2)
        self.eps   = var2eps(self.var, self.theta)  # epsilon parameter(s). SET BY VARIANCE (NOT MANUALLY BU USER ANYMORE)!
        self.func  = func                           # an objective function (optional)
        self.size  = ne                             # ensemble size

    def update_distribution(self, theta, corr):
        '''
        Updates the parameters (theta and corr) of the distirbution.

        Parameters
        ----------
        theta : array_like, shape (d,2)
            Contains the alpha (first column) and beta (second column) 
            of the marginal distirbutions.

        corr : array_like, shape (d,d)
            Correlation matrix 
        '''
        self.theta = theta
        self.corr  = corr
        return 
    
    def get_theta(self):
        return self.theta
    
    def get_corr(self):
        return self.corr

    def get_cov(self):

        std = np.zeros(self.dim)
        for d in range(self.dim):
            std[d] = stats.beta(*self.theta[d]).std() * 2 * self.eps[d]
        
        return ot.corr2cov(self.corr, std=std)
        
    def sample(self, size):
        '''
        Samples the mutation distribution as described in the GenOpt paper (NOT PUBLISHED YET!)

        Parameters
        ----------
        size : int
            Ensemble size (ne). Size of the sample to be drawn.
        
        Returns
        -------
        out : tuple, (enZ, enX)

            enZ : array_like, shape (ne,d)
                Zero-mean Gaussain ensemble, drawn with the correlation matrix, corr
            
            enX : array_like, shape (ne,d)
                The drawn ensemble matrix, X ~ p(x|θ,R) (GenOpt pdf)
        '''
        # Sample normal distribution with correlation
        enZ = np.random.multivariate_normal(mean=np.zeros(self.dim),
                                            cov=self.corr,
                                            size=size)

        # Transform Z to a uniform variable U
        enU = stats.norm.cdf(enZ)

        # Initialize enX
        enX = np.zeros_like(enZ)

        # Loop over dim
        for d in range(self.dim):
            marginal = stats.beta(*self.theta[d]) # Make marginal dist.
            enX[:,d] = marginal.ppf(enU[:,d])     # Transform U to marginal variables X

        return enZ, enX

    def epsilon_transformation(self, x, enX):
        '''
        Performs the epsilon transformation, 
            X ∈ [0, 1] ---> Y ∈ [x-ε, x+ε]
        
        Parameters
        ----------
        x : array_like, shape (d,)
            Current state vector.
        
        enX : array_like, shape (ne,d)
            Ensemble matrix X sampled from GenOpt distribution
        
        Returns
        -------
        out : array_like, shape (ne,d)
            Epsilon transfromed ensemble matrix, Y
        '''
        enY = np.zeros_like(enX) # tranfomred ensemble 

        # loop over dimenstion   
        for d, xd in enumerate(x):
            eps = self.eps[d]

            a = (xd-eps) - ( (xd-eps)*(xd-eps < 0) ) \
                         - ( (xd+eps-1)*(xd+eps > 1) ) \
                         + (xd+eps-1)*(xd-eps < 0)*(xd+eps > 1) #Lower bound of ensemble
            
            b = (xd+eps) - ( (xd-eps)*(xd-eps < 0) ) \
                         - ( (xd+eps-1)*(xd+eps > 1) ) \
                         + (xd-eps)*(xd-eps < 0)*(xd+eps > 1)   #Upper bound of ensemble
    
            enY[:,d] =  a + enX[:, d]*(b-a)                 #Component-wise trafo.
        
        return enY

    def gradient(self, x, *args, **kwargs):
        '''
        Calcualtes the average gradient of func using Stein's Lemma.
        Described in GenOpt paper.

        Parameters
        ----------
        x : array_like, shape (d,)
            Current state vector.
        
        args : (theta, corr)
            theta (parameters of distribution), shape (d,2)
            corr (correlation matrix), shape (d,d)
        
        kwargs :
            func : callable objectvie function
            ne : ensemble size
    
        Returns
        -------
        out : array_like, shape (d,)
            The average gradient.
        '''
        # check for objective fucntion 
        if 'func' in kwargs:
            func = kwargs.get('func')
        elif self.func is None:
            raise ValueError('No objectvie fucntion given. Please pass keyword argument: func=<your func>')
        else:
            func = self.func
        
        # check for ensemble size
        if 'ne' in kwargs:
            ne = kwargs.get('ne')
        elif self.size is None:
            ne = max(int(0.25*self.dim), 5)
        else:
            ne = self.size

        # update dist
        if args:
            self.update_distribution(*args)

        # sample and eps-transfrom
        self.enZ, self.enX = self.sample(size=ne)
        self.enY = self.epsilon_transformation(x, self.enX)
        self.enJ = func(self.enY.T)

        # calculate the ensemble gradient
        self.enJ    = self.enJ[:,np.newaxis]                    # shape (ne,1)
        self.enZ    = self.enZ.T                                # shape (d,ne)
        self.enX    = self.enX.T                                # shape (d,ne)
        alphas      = (self.theta[:,0])[:,np.newaxis]           # shape (d,1)   
        betas       = (self.theta[:,1])[:,np.newaxis]           # shape (d,1)

        H = np.linalg.inv(self.corr) - np.identity(self.dim)    # shape (d, d)
        N = stats.norm.pdf(self.enZ)                            # shape (d, Ne)
        M = np.zeros((self.dim, ne))                            # shape (d, Ne)

        for d in range(self.dim):
            marginal = stats.beta(*self.theta[d])
            M[d,:]   = marginal.pdf(self.enX[d])

        # get cov matrix
        cov = self.get_cov()

        # grad via Stein's lemma
        ensembleX = (alphas-1)/self.enX - (betas-1)/(1-self.enX) - (H@self.enZ)*M/N 
        grad      = -cov@ensembleX@self.enJ/(ne-1)
        grad      = 2*self.eps*np.squeeze(grad) # scale gradient. grad-shape (d,)

        # mutation gradient
        finv = lambda th: np.linalg.inv(self.fisher_matrix(*th))
        fishers_inv = np.apply_along_axis(finv, axis=1, arr=self.theta) #shape (d, 2, 2)

        log_term = np.array([np.log(self.enX), np.log(1-self.enX)])
        psi_term = np.array([np.apply_along_axis(delA, axis=1, arr=self.theta),
                             np.apply_along_axis(delA, axis=1, arr=np.flip(self.theta, axis=1))])
        
        ensembleTheta   = log_term-psi_term[:,:,None]
        self.grad_theta = np.sum(ensembleTheta*self.enJ.T, axis=-1)/(ne-1)
        self.grad_theta = np.einsum('kij,jk->ki', fishers_inv, self.grad_theta, optimize=True)

        return grad
    
    def mutation_gradient(self, x=None, *args, **kwargs):
        '''
        Returns the mutation gradient of theta. It is actually calulated in 
        self.ensemble_gradient.

        Parameters
        ----------
        kwargs :
            return_ensemble : bool
                If True, all the ensemble matrices are also returned in a dictionary.
        
        Returns
        -------
        out : array_like, shape (d,2)
            Mutation gradeint of theta
        
        NB! If return_ensembles=True, the ensmebles are also returned!
        '''
        if 'return_ensembles' in kwargs:
            ensembles = {'gaussian'   : self.enZ,
                         'vanilla'    : self.enX,
                         'transformed': self.enY,
                         'objective'  : self.enJ}
            return self.grad_theta, ensembles
        else:
            return self.grad_theta
    
    def corr_gradient(self):
        '''
        Returns the mutation gradeint of the correlation matrix
        '''
        enZ = self.enZ
        enJ = self.enJ
        ne  = np.squeeze(enJ).size 
        grad_corr = np.zeros_like(self.corr)

        for n in range(ne):
            grad_corr += enJ[n]*(np.outer(enZ[:,n], enZ[:,n]) - self.corr)
        
        np.fill_diagonal(grad_corr, 0)
        corr_gradient = grad_corr/(ne-1)

        return corr_gradient
    
    def fisher_matrix(self, alpha, beta):
        '''
        Calculates the Fisher matrix of a Beta distribution.
        
        Parameters
        ----------------------------------------------
        alpha : float
            alpha parameter in Beta distribution 

        beta : float
            beta parameter in Beta distribution

        Returns
        ----------------------------------------------
        out : array_like, of shape (2, 2)
            Fisher matrix 
        '''
        a = alpha
        b = beta

        upper_row = [polygamma(1, a) - polygamma(1, a+b), -polygamma(1, a + b)]
        lower_row = [-polygamma(1, a + b), polygamma(1, b) - polygamma(1, a+b)]
        fisher_matrix = np.array([upper_row, 
                                  lower_row])
        return fisher_matrix


# Some helping functions
def var2eps(var, theta):
    alphas  = theta[:,0]
    betas   = theta[:,1]
    frac    = alphas*betas / ( (alphas+betas)**2 * (alphas+betas+1) )
    epsilon = np.sqrt(0.25*var/frac)
    return epsilon

def delA(theta):
    r'''
    Calculates the expression  $\psi(a) - \psi(a+b)$,
    where  $\psi$  is the digamma function.

    Parameters
    --------------------------------------------
    a : float
    b : float
    
    Returns
    --------------------------------------------
    out : float
    '''
    a = theta[0]
    b = theta[1]
    return digamma(a)-digamma(a+b)
