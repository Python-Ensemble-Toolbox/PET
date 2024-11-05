"""Descriptive description."""

import numpy as np
from popt.cost_functions.epf import epf


def quadratic(state, *args, **kwargs):
	r"""Quadratic objective function

	$$ f(x) = ||x - b||^2_A $$
	"""

	r = kwargs.get('r', -1)

	x = state[0]['vector']
	dim, ne = x.shape
	A = 0.5*np.diag(np.ones(dim))
	b = 1.0*np.ones(dim)
	f = np.zeros(ne)
	for i in range(ne):
		u = x[:, i] - b
		f[i] = u.T@A@u

		# check for contraints
		if r >= 0:
			c_eq = g(x[:, i])
			c_iq = h(x[:, i])
			f[i] += epf(r, c_eq=c_eq, c_iq=c_iq) 

	return f

# Equality constraint saying that sum of x should be equal to dimention + 1
def g(x):
	return sum(x) - (x.size + 1)

# Inequality constrint saying that x_1 should be equal or less than 0
def h(x):
	return -x[0]

