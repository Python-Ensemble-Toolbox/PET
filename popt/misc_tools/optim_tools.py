"""
Collection of tools that can be used in optimization schemes. Only put tools here that are so general that they can
be used by several optimization schemes. If some method is only applicable to the update scheme you are
implementing, leave it in that class.
"""
import numpy as np


def aug_optim_state(state, list_state):
    """
    Augment the state variables to get one augmented array.

    Input:
            - state:                Dictionary of state variable for optimization. OBS: 1D arrays!
            - list_state:           Fixed list of keys in state dict.

    Output:
            - aug_state:            Augmented 1D array of state variables.

    ST 14/5-18: Similar to misc_tools.analysis_tools.aug_state, but in this method we get 1D array.
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

    ST 14/5-18: Similar to misc_tools.analysis_tools.update_state, but in this method we have 1D array.
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
    Makes the correlation matrix block diagonal.
    The blocks are the state varible types.

    Parameters
    ---------------------------------------------
        corr : 2D-array_like, of shape (d, d)

    Returns
    ---------------------------------------------
        corr_blocks : list of block matrices, one for each variable type
    """

    statenames = list(state.keys())
    corr_blocks = []
    for name in statenames:
        dim = state[name].size
        corr_blocks.append(corr[:dim, :dim])
        corr = corr[dim:, dim:]
    return corr_blocks

def time_correlation(a, state, n_timesteps, dt=1.0):
    r'''
    Constructs correlation matrix with time correlation
    using an autoregressive model.

    .. math::
        Corr(t_1, t_2) = a^{|t_1 - t_2|}
    
    Assumes that each varaible in state is time-order such that
    x = [x1, x2,..., xi,..., xn], where i is the time index, 
    and xi is d-dimensional.

    Parameters:
    -------------------------------------------------------------
            Correlation coef.

        state : dict
            Control state (represented in a dict).
        
        n_timesteps : int
            Number of time-steps to correlate for each component.
        
        dt : float or int
            Duration between each time-step. Default is 1.
    
    Returns:
    -------------------------------------------------------------
        out : ndarray
            Correlation matrix with time correlation    
    '''
    dim_states = [int(state[name].size/n_timesteps) for name in list(state.keys())]
    blocks     = []

    #Construct correlation matrix
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
