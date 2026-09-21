"""
Collection of tools that can be used in optimization schemes. Only put tools here that are so general that they can
be used by several optimization schemes. If some method is only applicable to the update scheme you are
implementing, leave it in that class.
"""
import numpy as np
import os
from datetime import datetime

from scipy.optimize import OptimizeResult


def get_list_element(list, element):
    """
    Retrieve the value associated with a given element in a list of tuples.

    Parameters
    ----------
    list : list
        A list of tuples, where each tuple contains two elements.
    element : any
        The element to search for in the first position of the tuples.

    Returns
    -------
    any
        The value associated with the given element in the list of tuples, or None if the element is not found.
    """

    # Iterate through the list to find element
    for item in list:
        if item[0] == element:
            return item[1]


def toggle_ml_state(state, ml_ne):

    """
    Toggle the state from a dictionary to a list of levels, or from a list of levels to a dictionary.
    This is necessary when we are using multi-level ensembles.

    Parameters
    ----------
    state : dict or list
        The current state, either as a dictionary or a list of levels.
    ml_ne : list
        List of ensemble sizes for each level.

    Returns
    -------
    new_state : dict or list
        The toggled state, either as a list of levels or a dictionary.
    """

    if not isinstance(state,list):

        # initialize the state as an empty list of dictionaries with length equal self.tot_level
        new_state = []

        # distribute the initial ensemble of states to the levels according to the given ensemble size.
        start = 0 # initialize
        for l in range(len(ml_ne)):
            stop = start + ml_ne[l]
            new_state.append(state[:,start:stop])
            start = stop

        del state
    else:  # state is a list of levels
        new_state = np.hstack(state)

    return new_state

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

    if bounds is None or len(bounds) == 0:
        return x
    # None means "no bound on this side". The previous version tested
    # `lb is None` on a whole array (always False), defaulted the *upper*
    # bound to -inf, and skipped clipping altogether when every bound was 0.
    lb = np.array([-np.inf if lo is None else lo for lo, _ in bounds], dtype=float)
    ub = np.array([np.inf if hi is None else hi for _, hi in bounds], dtype=float)
    return np.clip(x, lb, ub)


def save_optimize_results(intermediate_result, folder=None):
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
    if folder is not None:
        save_folder = folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    elif 'save_folder' in intermediate_result:
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
    if 'epf_iteration' in intermediate_result:
        np.savez(save_folder + '/optimize_result_{0}_{1}'.format(str(intermediate_result['epf_iteration']), suffix),
                 **intermediate_result)
    else:
        np.savez(save_folder + '/optimize_result_{0}'.format(suffix), **intermediate_result)
