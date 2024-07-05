"""
Collection of tools that can be used in optimization schemes. Only put tools here that are so general that they can
be used by several optimization schemes. If some method is only applicable to the update scheme you are
implementing, leave it in that class.
"""
import numpy as np
from scipy.linalg import block_diag
import os
from datetime import datetime
from scipy.optimize import OptimizeResult


def aug_optim_state(state, list_state):
    """
    Augment the state variables to get one augmented array.

    Input:
            - state:                Dictionary of state variable for optimization. OBS: 1D arrays!
            - list_state:           Fixed list of keys in state dict.

    Output:
            - aug_state:            Augmented 1D array of state variables.

    """
    # Start with ensemble of first state variable
    aug = state[list_state[0]]

    # Loop over the next states (if exists)
    for i in range(1, len(list_state)):
        aug = np.hstack((aug, state[list_state[i]]))

    # Return the augmented array
    return aug


def update_optim_state(aug_state, state, list_state):
    """
    Extract the separate state variables from an augmented state array. It is assumed that the augmented state
    array is made in aug_optim_state method, hence this is the reverse method.

    Input:
            - aug_state:                Augmented state array. OBS: 1D array.
            - state:                    Dictionary of state variable for optimization.
            - list_state:               Fixed list of keys in state dict.

    Output:
            - state:                    State dictionary updated with aug_state.

    """

    # Loop over all entries in list_state and extract an array with same number of rows as the key in state
    # determines from aug and replace the values in state[key].
    # Init. a variable to keep track of which row in 'aug' we start from in each loop
    aug_row = 0
    for _, key in enumerate(list_state):
        # Find no. rows in state[key] to determine how many rows from aug to extract
        no_rows = state[key].shape[0]

        # Extract the rows from aug and update 'state[key]'
        state[key] = aug_state[aug_row:aug_row + no_rows]

        # Update tracking variable for row in 'aug'
        aug_row += no_rows

    # Return
    return state


def corr2BlockDiagonal(state, corr):
    """
    Makes the correlation matrix block diagonal. The blocks are the state varible types.

    Parameters
    ----------
    state: dict
        Current control state, including state names

    corr : array_like
        Correlation matrix, of shape (d, d)

    Returns
    -------
    corr_blocks : list
        block matrices, one for each variable type

    """

    statenames = list(state.keys())
    corr_blocks = []
    for name in statenames:
        dim = state[name].size
        corr_blocks.append(corr[:dim, :dim])
        corr = corr[dim:, dim:]
    return corr_blocks


def time_correlation(a, state, n_timesteps, dt=1.0):
    """
    Constructs correlation matrix with time correlation
    using an autoregressive model.

    .. math::
        Corr(t_1, t_2) = a^{|t_1 - t_2|}
    
    Assumes that each varaible in state is time-order such that
    `x = [x1, x2,..., xi,..., xn]`, where `i` is the time index, 
    and `xi` is d-dimensional.

    Parameters
    -------------------------------------------------------------
    a : float
        Correlation coef, in range (0, 1).

    state : dict
        Control state (represented in a dict).
    
    n_timesteps : int
        Number of time-steps to correlate for each component.
    
    dt : float or int
        Duration between each time-step. Default is 1.

    Returns
    -------------------------------------------------------------
    out : numpy.ndarray
        Correlation matrix with time correlation    
    """
    dim_states = [int(state[name].size/n_timesteps) for name in list(state.keys())]
    blocks     = []

    # Construct correlation matrix
    # m: variable type index  
    # i: first time index
    # j: second time index
    # k: first dim index
    # l: second dim index
    for m in dim_states:
        corr_single_block = np.zeros((m*n_timesteps, m*n_timesteps))
        for i in range(n_timesteps):
            for j in range(n_timesteps):
                for k in range(m):
                    for l in range(m):
                       corr_single_block[i*m + k, j*m + l] = (k==l)*a**abs(dt*(i-j))
        blocks.append(corr_single_block)

    return block_diag(*blocks)


def cov2corr(cov):
    """
    Transfroms a covaraince matrix to a correlation matrix

    Parameters
    -------------
    cov : array_like
        The covaraince matrix, of shape (d,d).

    Returns
    -------------
    out : numpy.ndarray
        The correlation matrix, of shape (d,d)
    """
    std  = np.sqrt(np.diag(cov))
    corr = np.divide(cov, np.outer(std, std))
    return corr


def corr2cov(corr, std):
    """
    Transfroms a correlation matrix to a covaraince matrix

    Parameters
    ----------
    corr : array_like
        The correlation matrix, of shape (d,d).

    std : array_like
        Array of the standard deviations, of shape (d, ).

    Returns
    -------
    out : numpy.ndarray
        The covaraince matrix, of shape (d,d)
    """
    cov = np.multiply(corr, np.outer(std, std))
    return cov


def get_sym_pos_semidef(a):
    """
    Force matrix to positive semidefinite

    Parameters
    ----------
    a : array_like
        The input matrix, of shape (d,d)

    Returns
    -------
    a : numpy.ndarray
        The positive semidefinite matrix, of shape (d,d)
    """

    rtol = 1e-05
    if not isinstance(a, int):
        S, U = np.linalg.eigh(a)
        if not np.all(S > 0):
            S = np.clip(S, rtol, None)
            a = (U * S) @ U.T
    else:
        a = np.maximum(a, rtol)
    return a


def clip_state(x, bounds):
    """
    Clip a state vector according to the bounds

    Parameters
    ----------
    x : array_like
        The input state

    bounds : array_like
        (min, max) pairs for each element in x. None is used to specify no bound.

    Returns
    -------
    x : numpy.ndarray
        The state after truncation
    """

    any_not_none = any(any(item) for item in bounds)
    if any_not_none:
        lb = np.array(bounds)[:, 0]
        lb = np.where(lb is None, -np.inf, lb)
        ub = np.array(bounds)[:, 1]
        ub = np.where(ub is None, -np.inf, ub)
        x = np.clip(x, lb, ub)
    return x


def get_optimize_result(obj):
    """
    Collect optimize results based on requested

    Parameters
    ----------
    obj : popt.loop.optimize.Optimize
        An instance of an optimization class

    Returns
    -------
    save_dict : scipy.optimize.OptimizeResult
        The requested optimization results
    """

    # Initialize dictionary of variables to save
    save_dict = OptimizeResult({'success': True, 'x': obj.mean_state, 'fun': np.mean(obj.obj_func_values),
                                'nit':  obj.iteration, 'nfev': obj.nfev, 'njev': obj.njev})
    if 'savedata' in obj.options:

        # Make sure "SAVEDATA" gives a list
        if isinstance( obj.options['savedata'], list):
            savedata = obj.options['savedata']
        else:
            savedata = [ obj.options['savedata']]

        # Loop over variables to store in save list
        for save_typ in savedata:
            if 'mean_state' in save_typ:
                continue  # mean_state is alwaysed saved as 'x'
            if save_typ in locals():
                save_dict[save_typ] = eval('{}'.format(save_typ))
            elif hasattr( obj, save_typ):
                save_dict[save_typ] = eval(' obj.{}'.format(save_typ))
            else:
                print(f'Cannot save {save_typ}!\n\n')

    if 'save_folder' in obj.options:
        save_dict['save_folder'] = obj.options['save_folder']

    return save_dict


def save_optimize_results(intermediate_result):
    """
    Save optimize results

    Parameters
    ----------
    intermediate_result : scipy.optimize.OptimizeResult
        An instance of an OptimizeResult class
    """
    # Cast to OptimizeResult if a ndarray is passed as argument
    if type(intermediate_result) is np.ndarray:
        intermediate_result = OptimizeResult({'x': intermediate_result})

    # Make folder (if it does not exist)
    if 'save_folder' in intermediate_result:
        save_folder = intermediate_result['save_folder']
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    else:
        save_folder = './'

    if 'nit' in intermediate_result:
        suffix = str(intermediate_result['nit'])
    else:
        now = datetime.now()  # current date and time
        suffix = now.strftime("%m_%d_%Y_%H_%M_%S")

    # Save the variables
    np.savez(save_folder + '/optimize_result_{0}'.format(suffix), **intermediate_result)
