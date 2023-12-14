import numpy as np


def quadratic(state, *args):
	r"""
	Quadratic objective function
	.. math::
		f(x) = ||x - b||^2_A
	"""

	x = state[0]['vector']
	dim, ne = x.shape
	A = 0.5*np.diag(np.ones(dim))
	b = 1.0*np.ones(dim)
	f = np.zeros(ne)
	for i in range(ne):
		u = x[:, i] - b
		f[i] = u.T@A@u

	return f
