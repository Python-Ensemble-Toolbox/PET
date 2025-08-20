# This is a an implementation of the Line Search Algorithm (Alg. 3.5) in Numerical Optimization from Nocedal 2006.

import numpy as np
from functools import cache
from scipy.optimize._linesearch import _quadmin, _cubicmin


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

    # assertions
    assert step_size <= amax, "Initial step size must be less than or equal to amax."

    # Define phi and derivative of phi
    @cache
    def phi(alpha):
        global ls_nfev
        if (alpha == 0):
            if (fk is None):
                phi.fun_val = fun(xk)
                ls_nfev += 1
            else:
                phi.fun_val = fk
        else:
            phi.fun_val = fun(xk + alpha*pk)
            ls_nfev += 1
        return phi.fun_val
    
    @cache
    def dphi(alpha):
        global ls_njev
        if (alpha == 0):
            if (jk is None):
                dphi.jac_val = jac(xk)
                ls_njev += 1
            else:
                dphi.jac_val = jk
        else:
            dphi.jac_val = jac(xk + alpha*pk)
            ls_njev += 1
        return np.dot(dphi.jac_val, pk)
    
    # Define initial values of phi and dphi
    phi_0  = phi(0)
    dphi_0 = dphi(0)

    # Start loop
    a = [0, step_size]
    for i in range(1, maxiter+1):
        # Evaluate phi(ai)
        phi_i = phi(a[i])

        # Check for sufficient decrease
        if (phi_i > phi_0 + c1*a[i]*dphi_0) or (phi_i >= phi(a[i-1]) and i>0):
            # Call zoom function
            step_size = zoom(a[i-1], a[i], phi, dphi, phi_0, dphi_0, maxiter+1-i, c1, c2) 
            return step_size, phi.fun_val, dphi.jac_val, ls_nfev, ls_njev

        # Evaluate dphi(ai)
        dphi_i = dphi(a[i])

        # Check curvature condition
        if abs(dphi_i) <= -c2*dphi_0:
            step_size = a[i]
            return step_size, phi.fun_val, dphi.jac_val, ls_nfev, ls_njev
        
        # Check for posetive derivative
        if dphi_i >= 0:
            # Call zoom function
            step_size = zoom(a[i], a[i-1], phi, dphi, phi_0, dphi_0, maxiter+1-i, c1, c2) 
            return step_size, phi.fun_val, dphi.jac_val, ls_nfev, ls_njev
        
        # Increase ai
        a.append(min(2*a[i], amax))
        
    # If we reached this point, the line search failed
    return None, None, None, ls_nfev, ls_njev
            

def zoom(alo, ahi, f, df, f0, df0, maxiter, c1, c2):
    '''Zoom function for line search algorithm. (This is the same as for scipy)'''

    phi_lo = f(alo)
    phi_hi = f(ahi)
    dphi_lo = df(alo)

    for j in range(maxiter):

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

        # Evaluate phi(aj)
        phi_j = f(aj)

        # Check for sufficient decrease
        if (phi_j > f0 + c1*aj*df0) or (phi_j >= phi_lo):
            # store old values
            aold = ahi
            phi_old = phi_hi
            # update ahi
            ahi = aj
            phi_hi = phi_j
        else:
            # check curvature condition
            dphi_j = df(aj)
            if abs(dphi_j) <= -c2*df0:
                return aj
            
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

    # Define phi and derivative of phi
    @cache
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
        # Evaluate phi(alpha)
        phi_i = phi(step_size)

        # Check for sufficient decrease
        if (phi_i <= phi(0) + c1*step_size*np.dot(jk, pk)):
                # Evaluate jac at new point
                jac_new = jac(xk + step_size*pk)
                return step_size, phi_i, jac_new, ls_nfev, ls_njev
        
        # Reduce step size
        step_size *= rho  

    # If we reached this point, the line search failed
    return None, None, None, ls_nfev, ls_njev  