"""Rosenbrock objective function."""
from scipy.optimize import rosen

def _rosenbrock(state, *args, **kwargs):
    """
    Rosenbrock: http://en.wikipedia.org/wiki/Rosenbrock_function
    """
    x = state[0]['vector']
    x0 = x[:-1]
    x1 = x[1:]
    f = sum((1 - x0) ** 2) + 100 * sum((x1 - x0 ** 2) ** 2)
    return f

def rosenbrock(x, *args, **kwargs):
    """SciPy's Rosenbrock function of ``x``; extra arguments are ignored."""
    return rosen(x)
