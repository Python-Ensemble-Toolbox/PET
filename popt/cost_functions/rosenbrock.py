"""Rosenbrock objective function."""

def rosenbrock(state, *args):
    """
    Rosenbrock with negative sign (since we want to find the minimum)
    http://en.wikipedia.org/wiki/Rosenbrock_function
    """
    x = state[0]['vector']
    x0 = x[:-1]
    x1 = x[1:]
    ans = sum((1 - x0) ** 2) + 100 * sum((x1 - x0 ** 2) ** 2)
    return [-i for i in ans]
