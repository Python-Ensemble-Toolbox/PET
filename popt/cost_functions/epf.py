import numpy as np

def epf(r, c_eq=0, c_iq=0):
    r"""
    The external penalty function, given as
    .. math::
        0.5r(\sum_i c_{eq}^2 + \sum(\max(c_{iq},0)^2)
    We assume that the ensemble members are stacked as columns.
    """
    
    return r*0.5*( np.sum(c_eq**2, axis=0) + np.sum(np.maximum(-c_iq,0)**2, axis=0) )
