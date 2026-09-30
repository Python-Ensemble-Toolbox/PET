"""Graph-informed ensemble information-filter update (EnIF).

The ensemble information filter replaces the ensemble covariance of the
smoother updates with two sparse graph-informed estimates: a per-parameter
precision matrix fitted on a graph of the parameter connectivity, and a
boosted linear regression of the responses on the (standardised) state. The
estimators are ERT's, through the ``graphite-maps`` dependency -- which is
also why PET requires Python 3.12 through 3.14; PET supplies the MDA
lifecycle around them -- perturbed observations, inflation schedule,
forecasting, state limits and scoring.

The flavour is ES-MDA-specific, the way ``hybrid`` belongs to the multilevel
scheme and ``margis`` to GN-EnRML: it is wired into
``ESMDA.COMPATIBLE_ANALYSES`` rather than the global analysis registry, and
selectable as ``analysis='enif'``. The original single-update EnIF is the
one-step schedule, ``mda={tot_assim_steps: 1}``.
"""

from dataclasses import dataclass
from os import PathLike

import networkx as nx
import numpy as np
from graphite_maps.enif import EnIF
from graphite_maps.linear_regression import linear_boost_ic_regression
from graphite_maps.precision_estimation import fit_precision_cholesky_approximate
from scipy import linalg, sparse
from sklearn.preprocessing import StandardScaler

from pipt.update_schemes.analysis.base import AnalysisBase, AnalysisResult
import pipt.misc_tools.extract_tools as extract

__all__ = ["enif_update"]


@dataclass(frozen=True)
class _EnIFInformation:
    precision: sparse.sparray
    scales: np.ndarray
    active_rows: np.ndarray
    groups: tuple
    iteration: int


class enif_update(AnalysisBase):
    """Graph-informed information-space update, as an ES-MDA analysis flavour.

    Parameters
    ----------
    scheme : object, optional
        The ES-MDA scheme this analysis computes updates for. ``None`` leaves
        it unbound; the configuration is validated when it is bound, so an
        incompatible ``dataassim``/``ensemble`` combination or a bad ``enif``
        block fails at construction rather than mid-run.

    Notes
    -----
    The ``enif`` block of the ``dataassim`` section accepts:

    - ``parameter_graphs``: maps state names to NetworkX graphs, sparse
      adjacency arrays, or files written with ``scipy.sparse.save_npz``.
      Without one, a group with ``nx``/``ny`` (``nz``) grid metadata in its
      ``prior_`` block gets nearest-neighbour connectivity; a group without
      grid metadata is treated as independent.
    - ``neighbourhood_expansion``: precision fitting graph hops (default 2).
    - ``neighbor_propagation_order``: accepted for compatibility; MDA updates
      all retained state rows to preserve the accumulated information.

    Covariance localization, local analysis, multilevel ensembles and
    ``emp_cov`` cannot be combined with this flavour; spatial dependence is
    specified by the parameter graphs.

    Diagnostics of the last update -- the fitted regression ``H``, the prior
    and posterior precisions ``Prec_u``/``Prec_posterior``, the observation
    precision ``Prec_eps``, the ``update_indices`` and the active rows
    ``enif_active_rows`` -- are kept on the analysis object, not the scheme.
    """

    def __init__(self, scheme=None):
        super().__init__(scheme)
        if scheme is not None:
            self._validate_configuration(scheme)

    def update(self, enX, enY, enE, **kwargs):
        """Compute the graph-informed update step.

        Parameters
        ----------
        enX : np.ndarray
            State ensemble matrix, shape ``(nx, ne)``.
        enY : np.ndarray
            Predicted data ensemble matrix, shape ``(nd, ne)``.
        enE : np.ndarray
            Perturbed observations with covariance
            ``alpha * cov_data`` and the same shape as ``enY``. Additional
            noise is drawn when the fitted response has unexplained variance.

        Returns
        -------
        AnalysisResult
            The additive state-space ``step``.

        Notes
        -----
        Each parameter group has its own precision block. Parameters
        containing non-finite values, and parameters with no ensemble
        spread, are held fixed. The regression is refitted at every MDA step;
        the posterior precision is carried forward in the new standardized
        state coordinates.
        """
        scheme = self.scheme
        options = scheme.keys_da.get('enif', {})

        if enX.ndim != 2 or enX.shape[1] < 2:
            raise ValueError('EnIF requires at least two ensemble members.')
        if enY.ndim != 2 or enY.shape[1] != enX.shape[1] or enE.shape != enY.shape:
            raise ValueError('EnIF state, forecast and observation ensembles have incompatible shapes.')
        if enY.shape[0] == 0 or scheme.vecObs.shape != (enY.shape[0],):
            raise ValueError('EnIF requires observations matching the forecast rows.')
        if not all(np.all(np.isfinite(value)) for value in (enY, enE, scheme.vecObs)):
            raise ValueError('EnIF observations and forecasts must be finite.')

        finite = np.all(np.isfinite(enX), axis=1)
        if not finite.any():
            raise ValueError('No finite parameter rows available for EnIF.')
        active = finite.copy()
        active[finite] = np.ptp(enX[finite], axis=1) > 0
        step = np.zeros(enX.shape, dtype=float)
        self.enif_active_rows = np.flatnonzero(active)
        information = getattr(scheme, 'enif_information', None)
        groups = tuple(sorted(scheme.idX.items(), key=lambda item: item[1][0]))
        if scheme.iteration and information is None:
            raise ValueError('EnIF-MDA requires the preceding posterior information to resume.')
        if information is not None and (
            information.iteration != scheme.iteration
            or information.groups != groups
            or not np.array_equal(information.active_rows, self.enif_active_rows)
            or information.precision.shape != (len(self.enif_active_rows),) * 2
            or information.scales.shape != (len(self.enif_active_rows),)
        ):
            raise ValueError('EnIF-MDA information does not match the current state rows or iteration.')
        if not active.any():
            scheme.enif_information = _EnIFInformation(
                precision=sparse.csc_array((0, 0)),
                scales=np.empty(0),
                active_rows=self.enif_active_rows.copy(),
                groups=groups,
                iteration=scheme.iteration + 1,
            )
            return AnalysisResult(step=step)

        scaler = StandardScaler()
        U = scaler.fit_transform(enX[active].T)
        Y, E, d, self.Prec_eps = self._observation_precision(enY, enE)
        self.H = linear_boost_ic_regression(U=U, Y=Y.T)

        if information is None:
            # Keep precision blocks in the same row order as the augmented state.
            blocks = []
            for name, (start, stop) in groups:
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
                    neighbourhood_expansion=options.get('neighbourhood_expansion', 2),
                    use_tqdm=self._use_tqdm(scheme),
                ))
            self.Prec_u = sparse.csc_array(sparse.block_diag(blocks, format='csc'))
        else:
            change_of_scale = sparse.diags_array(scaler.scale_ / information.scales, format='csc')
            self.Prec_u = (change_of_scale @ information.precision @ change_of_scale).tocsc()

        gtmap = EnIF(Prec_u=self.Prec_u, Prec_eps=self.Prec_eps, H=self.H)
        self.update_indices = None
        canonical = gtmap.pushforward_to_canonical(U)
        residuals = gtmap.response_residual(U, Y.T)
        alpha = scheme.alpha[scheme.iteration]
        extra_variance = (alpha - 1) * gtmap.unexplained_variance
        if alpha > 1:
            self.Prec_eps = sparse.diags_array(
                1 / (1 / self.Prec_eps.diagonal() + extra_variance), format='csc',
            )
            gtmap.Prec_eps = self.Prec_eps
            extra_noise = scheme.ensemble.rng.standard_normal(residuals.shape) * np.sqrt(extra_variance)
        else:
            extra_noise = 0
        # PET already drew the measurement noise in E. The extra independent
        # draw inflates the full noisy-residual variance, including the fitted
        # unexplained response variance.
        canonical = gtmap.update_canonical(
            canonical=canonical,
            residual_noisy=residuals + d - E.T + extra_noise,
            d=d,
        )
        updated = gtmap.pullback_from_canonical(
            updated_canonical=canonical,
            update_indices=self.update_indices,
            U_prior=U,
            iterative=False,
        )
        self.Prec_posterior = gtmap.Prec_u
        step[active] = scaler.inverse_transform(updated).T - enX[active]
        scheme.enif_information = _EnIFInformation(
            precision=self.Prec_posterior,
            scales=scaler.scale_.copy(),
            active_rows=self.enif_active_rows.copy(),
            groups=groups,
            iteration=scheme.iteration + 1,
        )
        return AnalysisResult(step=step)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parameter_graph(self, name, size):
        """Load a group graph or build nearest-neighbour connectivity from its grid.

        Graph nodes are local parameter rows, numbered ``0 .. size-1``.
        Regular grids use y-fastest ordering, then x, then z, matching PET's
        layered prior ensembles. Without grid metadata, parameters are
        independent.

        Parameters
        ----------
        name : str
            State-variable name whose graph to build.
        size : int
            Number of parameter rows in the group.

        Returns
        -------
        networkx.Graph
        """
        scheme = self.scheme
        graph = scheme.keys_da.get('enif', {}).get('parameter_graphs', {}).get(name)
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

        info = scheme.prior_info[name]
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
        """Inflate observation covariance once; whiten correlated observation errors.

        Returns
        -------
        Y, E, d, Prec_eps
            The (possibly whitened) forecast and perturbation ensembles and
            observation vector, with the observation precision of the
            inflated covariance.
        """
        scheme = self.scheme
        covariance = np.asarray(scheme.cov_data, dtype=float)
        alpha = scheme.alpha[scheme.iteration]
        nd = enY.shape[0]
        if not np.isfinite(alpha) or alpha < 1:
            raise ValueError('EnIF-MDA inflation must be finite and at least one.')
        if not np.all(np.isfinite(covariance)):
            raise ValueError('EnIF observation covariance must be finite.')
        if covariance.ndim == 2:
            if covariance.shape != (nd, nd) or not np.allclose(covariance, covariance.T):
                raise ValueError('EnIF observation covariance must be square and symmetric.')
            if np.count_nonzero(covariance - np.diag(covariance.diagonal())):
                chol = linalg.cholesky(covariance, lower=True)
                Y = linalg.solve_triangular(chol, enY, lower=True)
                E = linalg.solve_triangular(chol, enE, lower=True)
                d = linalg.solve_triangular(chol, scheme.vecObs, lower=True)
                precision = sparse.diags_array(np.full(nd, 1.0 / alpha), format='csc')
                return Y, E, d, precision
            covariance = covariance.diagonal()
        if covariance.shape != (nd,) or np.any(covariance <= 0):
            raise ValueError('EnIF requires one strictly positive observation variance per forecast row.')
        precision = sparse.diags_array(1.0 / (alpha * covariance), format='csc')
        return enY, enE, scheme.vecObs, precision

    def _validate_configuration(self, scheme):
        """Reject keys EnIF cannot honour, and validate the ``enif`` block."""
        keys_da = scheme.keys_da
        keys_en = self._scheme_keys_en(scheme)
        for key in ('localization', 'localanalysis', 'multilevel'):
            if key in keys_da or key in keys_en:
                raise ValueError(f'EnIF does not support {key}.')
        if extract.is_enabled(keys_da.get('emp_cov', False)):
            raise ValueError('EnIF requires observation variances, not emp_cov samples.')

        options = keys_da.get('enif', {})
        if not isinstance(options, dict):
            raise ValueError('ENIF settings must be a dictionary.')
        for key, minimum in (('neighbourhood_expansion', 1), ('neighbor_propagation_order', 0)):
            value = options.get(key, minimum)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
                raise ValueError(f'EnIF {key} must be an integer >= {minimum}.')
        graphs = options.get('parameter_graphs', {})
        if not isinstance(graphs, dict) or set(graphs) - set(scheme.idX):
            raise ValueError('EnIF parameter_graphs must map known state names to graphs.')

    @staticmethod
    def _scheme_keys_en(scheme):
        """The ensemble section of the config, wherever the scheme keeps it.

        A real scheme exposes it as ``scheme.ensemble.keys_en``; a flat test
        double may carry ``keys_en`` directly; otherwise there is none.
        """
        keys_en = getattr(scheme, 'keys_en', None)
        if keys_en is None:
            keys_en = getattr(getattr(scheme, 'ensemble', None), 'keys_en', None)
        return keys_en or {}

    @classmethod
    def _use_tqdm(cls, scheme):
        """Show graphite-maps' progress bars unless the config disabled them."""
        return not bool(cls._scheme_keys_en(scheme).get('disable_tqdm', False))
