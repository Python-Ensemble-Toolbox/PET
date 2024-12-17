import numpy as np
import os
import shutil

def NewtonsMethod(fun, x0, args=(), grad=None, hess=None, bounds=None, **options):

    # get options
    maxiter   = options.get('maxiter', 100)
    step_size = options.get('step_size', 1.0)
    ftol = options.get('ftol', 1e-4)
    norm = options.get('norm', np.inf)
    maxiter_step  = options.get('maxiter_step', 5)
    diag_hessian  = options.get('diag_hessian', True)
    save_progress = options.get('save_progress', True)
    save_folder   = options.get('save_folder', './iterations')

    # make save folder if progress is to be saved
    if save_progress:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)


    nfev = 0 # fucntion evaluations
    ngev = 0 # gradient evaluations
    nhev = 0 # hessian evaluations

    # functions
    def _fun(x):
        nonlocal nfev
        nfev += 1
        return fun(x, *args)

    def _grad(x):
        nonlocal ngev
        ngev += 1
        return grad(x, *args)

    def _hess(x):
        nonlocal nhev
        nhev += 1
        if diag_hessian:
            return np.diag(np.diag(hess(x, *args)))
        else:
            return hess(x, *args)
    
    # check for initial value
    func_val = options.get('f0', None)
    gradient = options.get('g0', None)
    hessian  = options.get('h0', None)

    info = {'x': x0,
            'fun': func_val,
            'grad': gradient,
            'hess': hess,
            'iter': 0,
            'njev': nfev,
            'ngev': ngev,
            'nhev': nhev}

    if save_progress:
        np.savez(f'{save_folder}/iter_{0}', **info)

    # logger
    def _log_info(i):
        print('\n')
        print('Iteration: ', i)
        print('Function value: ', func_val)
        print('Function evaluations: ', nfev)
        print('Gradient calculations: ', ngev)
        print('\n')

    # set control vector
    x = x0

    # some stuff
    converged = False

    # optimization loop
    for k in range(maxiter):

        if k == 0:
            if func_val is None:
                func_val = _fun(x)

            if gradient is None:
                gradient = _grad(x)

            if hessian is None:
                hessian = _hess(x)
            
            _log_info(i=0)
            
        else: 
            gradient = _grad(x)
            hessian  = _hess(x)

        dx = np.linalg.solve(hessian, gradient)
        dx = _normalize(dx, norm=norm)

        # Backtracking Line Search
        a = step_size
        ncuts = 0
        sufficient_decrease = False

        while not sufficient_decrease:
            
            # check if maximum number of step-size cuts is reached
            if ncuts == maxiter_step:
                converged = True
                break
            
            # trial update
            x_trial = _clip(x - a*dx, bounds=bounds)
            new_func_val = _fun(x_trial)

            # check if Armijo condidtion is satisfied
            if armijo_condition(func_val, new_func_val, c1=ftol, alpha=a, pk=dx, gk=gradient):
                sufficient_decrease = True
            else:
                a = 0.5*a
                ncuts += 1
        
        # update
        x = x_trial
        func_val = new_func_val

        # log progress
        _log_info(k+1)

        # update info dict
        info.update({'x': x,
                     'fun': func_val,
                     'grad': gradient,
                     'hess': hess,
                     'iter': k+1,
                     'njev': nfev,
                     'ngev': ngev,
                     'nhev': nhev})
        
        if save_progress:
            np.savez(f'{save_folder}/iter_{k+1}', **info)

        if converged:
            # terminate
            break
    
    return info
                

def _clip(x, bounds):
    lb, ub = np.array(bounds).T
    return np.clip(x, a_min=lb, a_max=ub)

def armijo_condition(fold, fnew, c1, alpha, pk, gk):
    if fnew <= fold - c1*alpha*np.abs(np.dot(pk, gk)):
        return True
    else:
        return False

def _normalize(x, norm):

    if norm == False:
        return x
    else:
        return x/np.linalg.norm(x, norm)





