"""Conversion of the stacked state matrix into a per-variable dictionary of arrays."""

__all__ = [
    'matrix_to_dict',
]

# Imports
import numpy as np


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
