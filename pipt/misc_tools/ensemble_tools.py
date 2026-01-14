# This module contains functions and tools for ensembles

__all__ = [
    'matrix_to_dict',
    'matrix_to_list',
    'list_to_matrix',
    'generate_prior_ensemble',
    'clip_matrix'
]

# Imports
import numpy as np

# Internal imports
from geostat.decomp import Cholesky


def matrix_to_dict(matrix: np.ndarray, indecies: dict[tuple]) -> dict:
    '''
    Convert an ensemble matrix to a dictionary of arrays.

    Parameters
    ----------
    matrix : np.ndarray
        Ensemble matrix where each column represents an ensemble member.
    indecies : dict
        Dictionary with keys as variable names and values as tuples indicating the start and end row indices
        for each variable in the ensemble matrix.
    
    Returns
    -------
    ensemble_dict : dict
        Dictionary with keys as variable names and values as arrays of shape (nx, ne).
    '''
    ensemble_dict = {}
    for key, (start, end) in indecies.items():
        ensemble_dict[key] = matrix[start:end]
    
    return ensemble_dict


def matrix_to_list(matrix: np.ndarray, indecies: dict[tuple]) -> list[dict]:
    '''
    Convert an ensemble matrix to a list of dictionaries.

    Parameters
    ----------
    matrix : np.ndarray
        Ensemble matrix where each column represents an ensemble member.
    indecies : dict
        Dictionary with keys as variable names and values as tuples indicating the start and end row indices
        for each variable in the ensemble matrix.
    
    Returns
    -------
    ensemble_list : list of dict
    '''
    ne = matrix.shape[1]
    ensemble_list = []

    for n in range(ne):
        member = matrix_to_dict(matrix[:,n], indecies)
        ensemble_list.append(member)
    
    return ensemble_list


def list_to_matrix(ensemble_list: list[dict], indecies: dict[tuple]) -> np.ndarray:
    '''
    Convert a list of dictionaries to an ensemble matrix.

    Parameters
    ----------
    ensemble_list : list of dict
        List where each dictionary represents an ensemble member with variable names as keys.
    indecies : dict
        Dictionary with keys as variable names and values as tuples indicating the start and end row indices
        for each variable in the ensemble matrix.
    
    Returns
    -------
    matrix : np.ndarray
        Ensemble matrix where each column represents an ensemble member.
    '''
    ne = len(ensemble_list)
    nx = sum(end - start for start, end in indecies.values())
    matrix = np.zeros((nx, ne))

    for n, member in enumerate(ensemble_list):
        for key, (start, end) in indecies.items():
            if member[key].ndim == 2:
                matrix[start:end, n] = member[key][:,n]
            else:
                matrix[start:end, n] = member[key]
    
    return matrix


def generate_prior_ensemble(prior_info: dict, size: int, save: bool = True) -> tuple[np.ndarray, dict, dict]: 
    '''
    Generate a prior ensemble based on provided prior information.

    Parameters
    ----------
    prior_info : dict
        Dictionary containing prior information for each state variable.

    size : int
        Size of ensemble.

    save : bool, optional
        Whether to save the generated ensemble to a file. Default is True.  
    
    Returns
    -------
    enX : np.ndarray
        The generated ensemble matrix, shape: (nx, ne).

    idX : dict
        Dictionary with keys as variable names and values as tuples indicating the start and end row indices
        for each variable in the ensemble matrix.

    cov_prior : dict
        Dictionary containing the covariance matrices for each state variable.
    '''
    
    # Initialize sampler
    generator = Cholesky()

    # Initialize variables
    enX = None
    idX = {}
    cov_prior = {}

    # Loop over all state variables
    for name, info in prior_info.items():

        # Extract info
        nx = info.get('nx', 0)
        ny = info.get('ny', 0)
        nz = info.get('nz', 0)
        mean = info.get('mean', None)

        # if no dimensions are given, nothing is generated for this variable
        if nx == ny == 0: 
                break
        
        # Extract more options
        variance = info.get('variance', None)
        corr_length = info.get('corr_length', None)
        aniso = info.get('aniso', None)
        vario = info.get('vario', None)
        angle = info.get('angle', None)
        limits= info.get('limits',None)

        # Loop over nz to make layers of 2D priors
        index_stop = 0
        for idz in range(nz):
            # If mean is scalar, no covariance matrix is needed
            if isinstance(mean, (list, np.ndarray)) and len(mean) > 1:
                # Generate covariance matrix
                cov = generator.gen_cov2d(
                    x_size = nx, 
                    y_size = ny, 
                    variance = variance[idz], 
                    var_range = corr_length[idz], 
                    aspect = aniso[idz], 
                    angle = angle[idz], 
                    var_type = vario[idz]
                )
            else:
                cov = np.array(variance[idz])

            # Pick out the mean vector for the current layer
            index_start = index_stop
            index_stop  = int((idz + 1) * (len(mean)/nz))
            mean_layer  = mean[index_start:index_stop]

            # Generate realizations. If LIMITS have been entered, they must be taken account for here
            if limits is None:
                real = generator.gen_real(mean_layer, cov, size)
            else:
                real = generator.gen_real(mean_layer, cov, size, limits[idz])

            # Stack realizations for each layer
            if idz == 0:
                real_out = real
            else:
                real_out = np.vstack((real_out, real))
        
        # Fill in the ensemble matrix and indecies
        if enX is None:
            idX[name] = (0, real_out.shape[0])
            enX = real_out
        else:
            idX[name] = (enX.shape[0], enX.shape[0] + real_out.shape[0])
            enX = np.vstack((enX, real_out))
        
        # Store the covariance matrix
        cov_prior[name] = cov

    # Save prior ensemble
    if save:
        np.savez(
            'prior_ensemble.npz', 
            **{name: enX[idX[name][0]:idX[name][1]] for name in idX.keys()}
        )

    return enX, idX, cov_prior


def clip_matrix(matrix: np.ndarray, limits: dict|tuple|list, indecies: dict|None = None) -> np.ndarray:
    '''
    Clip the values in an ensemble matrix based on provided limits.

    Parameters
    ----------
    matrix : np.ndarray
        Ensemble matrix where each column represents an ensemble member.
    
    limits : dict, tuple, or list
        If tuple, it should be (lower_bound, upper_bound) applied to all variables.
        If dict, it should have variable names as keys and (lower_bound, upper_bound) as values.
        If list, it should contain (lower_bound, upper_bound) tuples for each variable in the order of indecies.
    
    indecies : dict, optional
        Dictionary with keys as variable names and values as tuples indicating the start and end row indices
        for each variable in the ensemble matrix. Required if limits is a dict or list. Default is None.
    
    Returns
    -------
    matrix : np.ndarray
    '''
    if isinstance(limits, tuple):
        lb, ub = limits
        if not (lb is None and ub is None):
            matrix = np.clip(matrix, lb, ub)

    elif isinstance(limits, dict) and isinstance(limits, dict):
        if indecies is None:
            raise ValueError("When limits is a dictionary, indecies must also be provided.")
        
        for key, (start, end) in indecies.items():
            if key in limits:
                lb, ub = limits[key]
                if not (lb is None and ub is None):
                    matrix[start:end] = np.clip(matrix[start:end], lb, ub)
    
    elif isinstance(limits, list):
        if indecies is None:
            raise ValueError("When limits is a list, indecies must also be provided.")
        
        if len(limits) != len(indecies):
            raise ValueError("Length of limits list must match number of variables in indecies.")
        
        for (key, (start, end)), (lb, ub) in zip(indecies.items(), limits):
            if not (lb is None and ub is None):
                matrix[start:end] = np.clip(matrix[start:end], lb, ub)

    return matrix

