import numpy as np
import numpy.linalg as la
from functools import lru_cache
from scipy.optimize._linesearch import _quadmin, _cubicmin
from scipy.optimize._trustregion_ncg import CGSteihaugSubproblem
from scipy.optimize._trustregion_exact import IterativeSubproblem

__all__ = [
    'line_search', 
    'zoom', 
    'line_search_backtracking', 
    'bfgs_update', 
    'newton_cg',
    'solve_trust_region_subproblem'
] 


def line_search(step_size, xk, pk, fun, jac, fk=None, jk=None, **kwargs):
    '''
    Line search algorithm to find step size alpha that satisfies the Wolfe conditions.

    Parameters
    ----------
    step_size : float
        Initial step size to start the line search.

    xk : ndarray
        Current point in the optimization process.

    pk : ndarray
        Search direction.
    
    fun : callable
        Objective function

    jac : callable
        Gradient of the objective function
    
    fk : float, optional
        Function value at xk. If None, it will be computed.
    
    jk : ndarray, optional
        Gradient at xk. If None, it will be computed.
    
    **kwargs : dict
        Additional parameters for the line search, such as:
        - amax : float, maximum step size (default: 1000)
        - maxiter : int, maximum number of iterations (default: 10)
        - c1 : float, sufficient decrease condition (default: 1e-4)
        - c2 : float, curvature condition (default: 0.9)
    
    Returns
    -------
    alpha : float
        Step size that satisfies the Wolfe conditions.
    
    fval : float
        Function value at the new point xk + step_size*pk.
    
    jval : ndarray
        Gradient at the new point xk + step_size*pk.
    
    nfev : int
        Number of function evaluations.
    
    njev : int
        Number of gradient evaluations.
    '''

    global ls_nfev
    global ls_njev
    ls_nfev = 0
    ls_njev = 0

    # Unpack some kwargs
    amax = kwargs.get('amax', 1000)
    maxiter = kwargs.get('maxiter', 10)
    c1 = kwargs.get('c1', 1e-4)
    c2 = kwargs.get('c2', 0.9)

    # check for logger in kwargs
    global logger
    logger = kwargs.get('logger', None)
    if logger is None:
        logger = print

    logger('Performing line search..........')
    logger('──────────────────────────────────────────────────')

    # assertions
    assert step_size <= amax, "Initial step size must be less than or equal to amax."

    # Define phi and derivative of phi
    @lru_cache(maxsize=None)
    def phi(alpha):
        global ls_nfev
        if (alpha == 0):
            if (fk is None):
                phi.fun_val = fun(xk)
                ls_nfev += 1
            else:
                phi.fun_val = fk
        else:
            #logger('    Evaluating Armijo condition')
            phi.fun_val = fun(xk + alpha*pk)
            ls_nfev += 1
        return phi.fun_val
    
    @lru_cache(maxsize=None)
    def dphi(alpha):
        global ls_njev
        if (alpha == 0):
            if (jk is None):
                dphi.jac_val = jac(xk)
                ls_njev += 1
            else:
                dphi.jac_val = jk
        else:
            #logger('    Evaluating curvature condition')
            dphi.jac_val = jac(xk + alpha*pk)
            ls_njev += 1
        return np.dot(dphi.jac_val, pk)
    
    # Define initial values of phi and dphi
    phi_0  = phi(0)
    dphi_0 = dphi(0)

    # Start loop
    a = [0, step_size]
    for i in range(1, maxiter+1):
        logger(f'iteration: {i-1}')

        # Evaluate phi(ai)
        phi_i = phi(a[i])

        # Check for sufficient decrease
        if (phi_i > phi_0 + c1*a[i]*dphi_0) or (phi_i >= phi(a[i-1]) and i>0):
            logger('    Armijo condition: not satisfied')
            # Call zoom function
            step_size = zoom(a[i-1], a[i], phi, dphi, phi_0, dphi_0, maxiter+1-i, c1, c2, iter_id=i) 
            logger('──────────────────────────────────────────────────')
            return step_size, phi.fun_val, dphi.jac_val, ls_nfev, ls_njev
        
        logger('    Armijo condition: satisfied')

        # Evaluate dphi(ai)
        dphi_i = dphi(a[i])

        # Check curvature condition
        if abs(dphi_i) <= -c2*dphi_0:
            logger('    Curvature condition: satisfied')
            step_size = a[i]
            logger('──────────────────────────────────────────────────')
            return step_size, phi.fun_val, dphi.jac_val, ls_nfev, ls_njev

        logger('    Curvature condition: not satisfied')
        
        # Check for posetive derivative
        if dphi_i >= 0:
            # Call zoom function
            step_size = zoom(a[i], a[i-1], phi, dphi, phi_0, dphi_0, maxiter+1-i, c1, c2, iter_id=i)
            logger('──────────────────────────────────────────────────')
            return step_size, phi.fun_val, dphi.jac_val, ls_nfev, ls_njev
        
        # Increase ai
        a.append(min(2*a[i], amax))
        logger(f'    Step-size: {a[i]:.3e} ──> {a[i+1]:.3e}')
        
    # If we reached this point, the line search failed
    logger('Line search failed to find a suitable step size')
    logger('──────────────────────────────────────────────────')
    return None, None, None, ls_nfev, ls_njev
            

def zoom(alo, ahi, f, df, f0, df0, maxiter, c1, c2, iter_id=0):
    '''Zoom function for line search algorithm. (This is the same as for scipy)'''

    phi_lo = f(alo)
    phi_hi = f(ahi)
    dphi_lo = df(alo)

    for j in range(maxiter):
        logger(f'iteration: {iter_id+j}')

        tol_cubic = 0.2*(ahi-alo)
        tol_quad  = 0.1*(ahi-alo)

        if (j > 0):
            # cubic interpolation for alo, phi(alo), dphi(alo) and ahi, phi(ahi)
            aj = _cubicmin(alo, phi_lo, dphi_lo, ahi, phi_hi, aold, phi_old)
        if (j == 0) or (aj is None) or (aj < alo + tol_cubic) or (aj > ahi - tol_cubic):
            # quadratic interpolation for alo, phi(alo), dphi(alo) and ahi, phi(ahi)
            aj = _quadmin(alo, phi_lo, dphi_lo, ahi, phi_hi)

        # Ensure aj is within bounds
        if (aj is None) or (aj <  alo + tol_quad) or (aj > ahi - tol_quad):
                aj = alo + 0.5*(ahi - alo)

        
        logger(f'    New step-size ──> {aj:.3e}')

        # Evaluate phi(aj)
        phi_j = f(aj)

        # Check for sufficient decrease
        if (phi_j > f0 + c1*aj*df0) or (phi_j >= phi_lo):
            logger('    Armijo condition: not satisfied')
            # store old values
            aold = ahi
            phi_old = phi_hi
            # update ahi
            ahi = aj
            phi_hi = phi_j
        else:
            logger('    Armijo condition: satisfied')
            # check curvature condition
            dphi_j = df(aj)
            if abs(dphi_j) <= -c2*df0:
                logger('    Curvature condition: satisfied')
                return aj
            
            logger('    Curvature condition: not satisfied')
            if dphi_j*(ahi-alo) >= 0:
                # store old values
                aold = ahi
                phi_old = phi_hi
                # update alo
                ahi = alo
                phi_hi = phi_lo
            else:
                # store old values
                aold = alo
                phi_old = phi_lo

            alo = aj
            phi_lo = phi_j
            dphi_lo = dphi_j

    # If we reached this point, the line search failed
    logger('Line search failed to find a suitable step size')
    logger('──────────────────────────────────────────────────')
    return None




def line_search_backtracking(step_size, xk, pk, fun, jac, fk=None, jk=None, **kwargs):
    '''
    Backtracking line search algorithm to find step size alpha that satisfies the Wolfe conditions.

    Parameters
    ----------
    step_size : float
        Initial step size to start the line search.

    xk : ndarray
        Current point in the optimization process.

    pk : ndarray
        Search direction.
    
    fun : callable
        Objective function

    jac : callable
        Gradient of the objective function
    
    fk : float, optional
        Function value at xk. If None, it will be computed.
    
    jk : ndarray, optional
        Gradient at xk. If None, it will be computed.
    
    **kwargs : dict
        Additional parameters for the line search, such as:
        - rho : float, backtracking factor (default: 0.5)
        - maxiter : int, maximum number of iterations (default: 10)
        - c1 : float, sufficient decrease condition (default: 1e-4)
        - c2 : float, curvature condition (default: 0.9)
    
    Returns
    -------
    alpha : float
        Step size that satisfies the Wolfe conditions.
    
    fval : float
        Function value at the new point xk + step_size*pk.
    
    jval : ndarray
        Gradient at the new point xk + step_size*pk.
    
    nfev : int
        Number of function evaluations.
    
    njev : int
        Number of gradient evaluations.
    '''

    global ls_nfev
    global ls_njev
    ls_nfev = 0
    ls_njev = 0

    # Unpack some kwargs
    rho = kwargs.get('rho', 0.5)
    maxiter = kwargs.get('maxiter', 10)
    c1 = kwargs.get('c1', 1e-4)

    # check for logger in kwargs
    global logger
    logger = kwargs.get('logger', None)
    if logger is None:
        logger = print

    logger('Performing backtracking line search..........')
    logger('──────────────────────────────────────────────────')

    # Define phi and derivative of phi
    @lru_cache(maxsize=None)
    def phi(alpha):
        global ls_nfev
        if (alpha == 0):
            if (fk is None):
                fun_val = fun(xk)
                ls_nfev += 1
            else:
                fun_val = fk
        else:
            fun_val = fun(xk + alpha*pk)
            ls_nfev += 1
        return fun_val
    

    # run the backtracking line search loop
    for i in range(maxiter):
        logger(f'iteration: {i}')
        # Evaluate phi(alpha)
        phi_i = phi(step_size)

        # Check for sufficient decrease
        if (phi_i <= phi(0) + c1*step_size*np.dot(jk, pk)):
                # Evaluate jac at new point
                jac_new = jac(xk + step_size*pk)
                logger('──────────────────────────────────────────────────')
                return step_size, phi_i, jac_new, ls_nfev, ls_njev
        
        # Reduce step size
        step_size *= rho  

    # If we reached this point, the line search failed
    logger('Backtracking failed to find a suitable step size')
    logger('──────────────────────────────────────────────────')
    return None, None, None, ls_nfev, ls_njev  


def bfgs_update(Hk, sk, yk):
    """
    Perform the BFGS update of the inverse Hessian approximation.

    Parameters:
    - Hk: np.ndarray, current inverse Hessian approximation (n x n)
    - sk: np.ndarray, step vector (x_{k+1} - x_k), shape (n,)
    - yk: np.ndarray, gradient difference (grad_{k+1} - grad_k), shape (n,)

    Returns:
    - Hk_new: np.ndarray, updated inverse Hessian approximation
    """
    sk = sk.reshape(-1, 1)
    yk = yk.reshape(-1, 1)
    rho = 1.0 / (yk.T @ sk)

    if rho <= 0:
        print('Non-positive curvature detected. BFGS update skipped....')
        return Hk

    I = np.eye(Hk.shape[0])
    Vk = I - rho * sk @ yk.T
    Hk_new = Vk @ Hk @ Vk.T + rho * sk @ sk.T

    return Hk_new

def newton_cg(gk, Hk=None, maxiter=None, **kwargs):

    # Check for logger
    logger = kwargs.get('logger', None)
    if logger is None:
        logger = print
    
    logger('')
    logger('Running Newton-CG subroutine..........')

    if Hk is None:
        jac = kwargs.get('jac')
        eps = kwargs.get('eps', 1e-4)
        xk  = kwargs.get('xk')

        # define a finite difference approximation of the Hessian times a vector
        def Hessd(d):
            return (jac(xk + eps*d) - gk)/eps

    if maxiter is None:
        maxiter = 20*gk.size # Same dfault as in scipy

    tol = min(0.5, np.sqrt(la.norm(gk)))*la.norm(gk)
    z = 0
    r = gk
    d = -r

    for j in range(maxiter):
        logger(f'iteration: {j}')
        if Hk is None:
            Hd = Hessd(d)
        else:
            Hd = np.matmul(Hk, d)

        dTHd = np.dot(d, Hd)

        if dTHd <= 0:
            logger('Negative curvature detected, terminating subroutine')
            logger('')
            if j == 0:
                return -gk
            else:
                return z
            
        rold = r
        a = np.dot(r,r)/dTHd
        z = z + a*d
        r = r + a*Hd

        if la.norm(r) < tol:
            logger('Subroutine converged')
            logger('')
            return z

        b = np.dot(r, r)/np.dot(rold, rold)
        d = -r + b*d


def solve_trust_region_subproblem(xk, fk, gk, Hk, radius, method='iterative', **kwargs):
    '''
    Solve the trust-region subproblem.

    Parameters
    ----------
    xk : ndarray
        Current point in the optimization process.
    fk : float
        Function value at xk.
    gk : ndarray
        Gradient at xk.
    Hk : ndarray
        Hessian at xk.
    radius : float
        Trust-region radius.
    method : str, optional
        Method to solve the trust-region subproblem. Options are 'iterative' or 'CG-Steihaug'. Default is 'iterative'.
        If a callable is provided, it will be used as the solver with the signature:
        method(xk, fk, gk, Hk, radius, **kwargs)

    **kwargs : dict
        Additional parameters for the solver.
    
    Returns
    -------
    pk : ndarray
        Solution to the trust-region subproblem.
    hits_boundary : bool
        Indicates whether the solution lies on the boundary of the trust region.
    '''
    # Make quadratic model
    model = lambda p: fk + np.dot(gk, p) + 0.5*np.dot(p, np.matmul(Hk, p))

    # Solve the trust-region subproblem
    if method == 'iterative':
        subproblem = IterativeSubproblem(
                xk,
                model, 
                lambda _: gk, 
                lambda _: Hk,
            )
        pk, hits_boundary = subproblem.solve(radius)

    elif method == 'CG-Steihaug':
        subproblem = CGSteihaugSubproblem(
                xk,
                model, 
                lambda _: gk, 
                lambda _: Hk,
            )
        pk, hits_boundary = subproblem.solve(radius)
    
    else:
        raise ValueError("Invalid method for solving trust-region subproblem. Choose 'iterative' or 'CG-Steihaug'.")

    return pk, hits_boundary