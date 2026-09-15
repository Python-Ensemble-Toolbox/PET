"""Ensemble information filter (EnIF) and multiple data assimilation schemes.

The sparse regression and precision estimators follow ERT's EnIF analysis,
using ``graphite-maps``. PET supplies the perturbed observations and manages
forecasting, state limits, convergence diagnostics and acceptance of updates.
"""

from os import PathLike

import networkx as nx
import numpy as np
from graphite_maps.enif import EnIF
from graphite_maps.linear_regression import linear_boost_ic_regression
from graphite_maps.precision_estimation import fit_precision_cholesky_approximate
from scipy import linalg, sparse
from sklearn.preprocessing import StandardScaler

from pipt.update_schemes.esmda import esmdaMixIn


class enif_update:
    """Graph-informed information-space update, with PET's ``update`` interface."""

    def update(self, enX, enY, enE, **kwargs):
        """Compute ``self.step`` from the current ensemble and perturbed observations.

        Parameters
        ----------
        enX : ndarray
            State ensemble, shape (number of parameters, number of members).
        enY : ndarray
            Forecast ensemble, shape (number of observations, number of members).
        enE : ndarray
            Perturbed observations with covariance ``alpha * cov_data`` and
            the same shape as ``enY``. These are used without adding more noise.

        Notes
        -----
        Each parameter group has its own precision block. Parameters containing
        non-finite values, and parameters with no ensemble spread, are held fixed.
        The regression and prior precision are refitted at every MDA step.
        """
        if enX.ndim != 2 or enX.shape[1] < 2:
            raise ValueError('EnIF requires at least two ensemble members.')
        if enY.ndim != 2 or enY.shape[1] != enX.shape[1] or enE.shape != enY.shape:
            raise ValueError('EnIF state, forecast and observation ensembles have incompatible shapes.')
        if enY.shape[0] == 0 or self.vecObs.shape != (enY.shape[0],):
            raise ValueError('EnIF requires observations matching the forecast rows.')
        if not all(np.all(np.isfinite(value)) for value in (enY, enE, self.vecObs)):
            raise ValueError('EnIF observations and forecasts must be finite.')

        finite = np.all(np.isfinite(enX), axis=1)
        if not finite.any():
            raise ValueError('No finite parameter rows available for EnIF.')
        active = finite.copy()
        active[finite] = np.ptp(enX[finite], axis=1) > 0
        self.step = np.zeros(enX.shape, dtype=float)
        self.enif_active_rows = np.flatnonzero(active)
        if not active.any():
            return

        scaler = StandardScaler()
        U = scaler.fit_transform(enX[active].T)
        Y, E, d, self.Prec_eps = self._observation_precision(enY, enE)
        self.H = linear_boost_ic_regression(U=U, Y=Y.T)

        # Keep precision blocks in the same row order as the augmented state.
        blocks = []
        for name, (start, stop) in sorted(self.idX.items(), key=lambda item: item[1][0]):
            local_active = active[start:stop]
            if not local_active.any():
                continue
            graph = self._parameter_graph(name, stop - start)
            graph = graph.subgraph(np.flatnonzero(local_active))
            graph = nx.convert_node_labels_to_integers(graph, ordering='sorted')
            local_scaler = StandardScaler()
            local_U = local_scaler.fit_transform(enX[start:stop][local_active].T)
            blocks.append(fit_precision_cholesky_approximate(
                local_U,
                graph,
                neighbourhood_expansion=self.enif_options.get('neighbourhood_expansion', 2),
                use_tqdm=not self.disable_tqdm,
            ))
        self.Prec_u = sparse.csc_array(sparse.block_diag(blocks, format='csc'))

        gtmap = EnIF(Prec_u=self.Prec_u, Prec_eps=self.Prec_eps, H=self.H)
        self.update_indices = gtmap.get_update_indices(
            neighbor_propagation_order=self.enif_options.get('neighbor_propagation_order', 15),
        )
        canonical = gtmap.pushforward_to_canonical(U)
        residuals = gtmap.response_residual(U, Y.T)
        # ERT transport draws noise internally. Use PET's existing perturbations
        # instead: d - (residuals + d - E) == E - residuals.
        canonical = gtmap.update_canonical(
            canonical=canonical,
            residual_noisy=residuals + d - E.T,
            d=d,
        )
        updated = gtmap.pullback_from_canonical(
            updated_canonical=canonical,
            update_indices=self.update_indices,
            U_prior=U,
            iterative=False,
        )
        self.Prec_posterior = gtmap.Prec_u
        self.step[active] = scaler.inverse_transform(updated).T - enX[active]

    def _parameter_graph(self, name, size):
        """Load a group graph or build nearest-neighbour connectivity from its grid.

        Graph nodes are local parameter rows, numbered ``0 .. size-1``. Regular
        grids use y-fastest ordering, then x, then z, matching PET's layered
        prior ensembles. Without grid metadata, parameters are independent.
        """
        graph = self.enif_options.get('parameter_graphs', {}).get(name)
        if graph is not None:
            if isinstance(graph, (str, PathLike)):
                graph = sparse.load_npz(graph)
            if sparse.issparse(graph):
                if graph.shape != (size, size) or (graph != graph.T).nnz:
                    raise ValueError(f'EnIF graph for {name} must be a symmetric ({size}, {size}) adjacency.')
                graph = nx.from_scipy_sparse_array(graph)
            if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
                raise ValueError(f'EnIF graph for {name} must be an undirected simple graph.')
            if set(graph.nodes) != set(range(size)):
                raise ValueError(f'EnIF graph for {name} must have nodes 0 through {size - 1}.')
            return graph.copy()

        info = self.prior_info[name]
        if not all(key in info for key in ('nx', 'ny')):
            return nx.empty_graph(size)
        shape = (int(info.get('nz', 1)), int(info['nx']), int(info['ny']))
        if min(shape) < 1 or np.prod(shape) != size:
            raise ValueError(
                f'EnIF grid for {name} has {np.prod(shape)} cells but {size} parameter rows. '
                'Provide parameter_graphs for a reduced or irregular grid.'
            )
        cells = np.arange(size).reshape(shape)
        graph = nx.empty_graph(size)
        for axis in range(3):
            left = [slice(None)] * 3
            right = [slice(None)] * 3
            left[axis] = slice(None, -1)
            right[axis] = slice(1, None)
            graph.add_edges_from(zip(cells[tuple(left)].ravel(), cells[tuple(right)].ravel()))
        return graph

    def _observation_precision(self, enY, enE):
        """Inflate observation covariance once; whiten correlated observation errors."""
        covariance = np.asarray(self.cov_data, dtype=float)
        alpha = self.alpha[self.iteration - 1]
        nd = enY.shape[0]
        if not np.all(np.isfinite(covariance)):
            raise ValueError('EnIF observation covariance must be finite.')
        if covariance.ndim == 2:
            if covariance.shape != (nd, nd) or not np.allclose(covariance, covariance.T):
                raise ValueError('EnIF observation covariance must be square and symmetric.')
            if np.count_nonzero(covariance - np.diag(covariance.diagonal())):
                chol = linalg.cholesky(covariance, lower=True)
                Y = linalg.solve_triangular(chol, enY, lower=True)
                E = linalg.solve_triangular(chol, enE, lower=True)
                d = linalg.solve_triangular(chol, self.vecObs, lower=True)
                precision = sparse.diags_array(np.full(nd, 1.0 / alpha), format='csc')
                return Y, E, d, precision
            covariance = covariance.diagonal()
        if covariance.shape != (nd,) or np.any(covariance <= 0):
            raise ValueError('EnIF requires one strictly positive observation variance per forecast row.')
        precision = sparse.diags_array(1.0 / (alpha * covariance), format='csc')
        return enY, enE, self.vecObs, precision


class enifMixIn(esmdaMixIn):
    """Use PET's MDA lifecycle with EnIF-specific settings and inflation validation.

    Parameters
    ----------
    keys_da : dict
        Standard PET assimilation settings. Optional ``enif`` dictionary:

        - ``parameter_graphs``: maps state names to NetworkX graphs, sparse
          adjacency arrays, or files written with ``scipy.sparse.save_npz``.
        - ``neighbourhood_expansion``: precision fitting graph hops (default 2).
        - ``neighbor_propagation_order``: update propagation hops (default 15).

        Covariance localization and local analysis cannot be combined with
        this scheme; spatial dependence is specified by the parameter graphs.
    keys_en : dict
        Standard PET ensemble settings, including ``disable_tqdm``.
    sim : object
        PET forward simulator.
    """

    def __init__(self, keys_da, keys_en, sim):
        for key in ('localization', 'localanalysis', 'multilevel'):
            if key in keys_da or key in keys_en:
                raise ValueError(f'EnIF does not support {key}.')
        if keys_da.get('emp_cov') == 'yes':
            raise ValueError('EnIF requires observation variances, not emp_cov samples.')
        super().__init__(keys_da, keys_en, sim)
        self.enif_options = self.keys_da.get('enif', {})
        if not isinstance(self.enif_options, dict):
            raise ValueError('ENIF settings must be a dictionary.')
        for key, minimum in (('neighbourhood_expansion', 1), ('neighbor_propagation_order', 0)):
            value = self.enif_options.get(key, minimum)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
                raise ValueError(f'EnIF {key} must be an integer >= {minimum}.')
        graphs = self.enif_options.get('parameter_graphs', {})
        if not isinstance(graphs, dict) or set(graphs) - set(self.idX):
            raise ValueError('EnIF parameter_graphs must map known state names to graphs.')

    def _mda_options(self):
        """Accept PET's dictionary and legacy list forms for MDA settings."""
        if 'mda' not in self.keys_da:
            raise ValueError('EnIF-MDA requires MDA settings with tot_assim_steps.')
        options = self.keys_da['mda']
        try:
            return dict(options)
        except (TypeError, ValueError):
            return dict([options])

    def _ext_assim_steps(self):
        """Keep the full schedule on restart; PET resumes using ``iteration``."""
        steps = self._mda_options().get('tot_assim_steps')
        if not isinstance(steps, (int, float, np.integer)) or isinstance(steps, bool):
            raise ValueError('MDA tot_assim_steps must be a positive integer.')
        if not np.isfinite(steps) or steps < 1 or int(steps) != steps:
            raise ValueError('MDA tot_assim_steps must be a positive integer.')
        return list(range(int(steps)))

    def _ext_inflation_param(self):
        """Validate a positive MDA schedule whose reciprocal factors sum to one."""
        count = len(self._ext_assim_steps())
        alpha = np.asarray(self._mda_options().get('inflation_param', count), dtype=float)
        if alpha.ndim == 0:
            alpha = np.full(count, alpha.item())
        if alpha.shape != (count,) or not np.all(np.isfinite(alpha)) or np.any(alpha <= 0):
            raise ValueError('MDA requires one finite positive inflation factor per assimilation step.')
        if not np.isclose(np.sum(1.0 / alpha), 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError('The inverse MDA inflation factors must sum to one.')
        return alpha.tolist()


class enif_full(enifMixIn, enif_update):
    """Original, single-update EnIF: ``daalg=['enif', 'enif'], analysis='full'``."""

    def _ext_assim_steps(self):
        return [0]

    def _ext_inflation_param(self):
        return [1.0]


class enif_mda(enifMixIn, enif_update):
    """EnIF-MDA: ``daalg=['enif', 'enif'], analysis='mda'`` with PET's ``mda`` settings."""

    pass
