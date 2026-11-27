"""Covariance square-root products for the Laplace scheme, in PET state order."""

from dataclasses import dataclass
from functools import cached_property
from os import PathLike

import networkx as nx
import numpy as np
from graphite_maps.precision_estimation import optimize_sparse_affine_kr_map, reverse_cholesky, solve_row_closed_form
from scipy import sparse
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import spsolve_triangular
from sksparse.cholmod import CholmodNotPositiveDefiniteError

from misc.structures.layout import prior_layers
from pipt.update_schemes.analysis.enif import parameter_graph


@dataclass
class PriorBlock:
    rows: slice
    factor: object
    precision: bool = False
    permutation: np.ndarray | None = None

    def apply(self, x, transpose):
        if not self.precision:
            return (self.factor.T if transpose else self.factor) @ x
        if transpose:
            if self.permutation is not None:
                x = x[self.permutation]
            return spsolve_triangular(self.factor.T, x, lower=False)
        out = spsolve_triangular(self.factor, x, lower=True)
        if self.permutation is not None:
            out[self.permutation] = out.copy()
        return out

    def standardized(self, dx):
        if self.precision:
            return self.factor @ (dx if self.permutation is None else dx[self.permutation])
        if sparse.issparse(self.factor):  # PET's independent scalar variances
            return dx / self.factor.diagonal()
        return solve_triangular(self.factor, dx, lower=True)

    def precision_product(self, dx):
        z = self.standardized(dx)
        if self.precision:
            out = self.factor.T @ z
            if self.permutation is not None:
                out[self.permutation] = out.copy()
            return out
        if sparse.issparse(self.factor):
            return z / self.factor.diagonal()
        return solve_triangular(self.factor, z, lower=True, trans="T")


@dataclass
class MaternBlock:
    """matern-egrid SPDE root: F = diag(1/tau) K⁻¹ diag(sqrt(volumes))."""
    rows: slice
    K: object
    tau: np.ndarray
    sqrt_volume: np.ndarray

    @cached_property
    def solver(self):
        from matern_egrid.solver import make_solver
        return make_solver(self.K)

    def __getstate__(self):
        return {k: v for k, v in vars(self).items() if k != "solver"}

    def apply(self, x, transpose):
        tau, sd = (self.tau, self.sqrt_volume) if x.ndim == 1 else (self.tau[:, None], self.sqrt_volume[:, None])
        if transpose:
            return sd * self.solver.solve(x / tau, trans="T")
        return self.solver.solve(sd * x) / tau

    def standardized(self, dx):
        return (self.K @ (self.tau * dx)) / self.sqrt_volume

    def precision_product(self, dx):
        return self.tau * (self.K.T @ (self.standardized(dx) / self.sqrt_volume))


def _matern_block(rows, settings):
    from matern_egrid import EgridGeometry, MaternField3, build_operator
    geom = EgridGeometry.from_egrid(settings["egrid"])
    if geom.nc != rows.stop - rows.start:
        raise ValueError("Matérn active cell count must match the PET state block.")
    op = build_operator(geom, MaternField3(**settings["field"]), **settings.get("operator", {}))
    return MaternBlock(rows, op.K, op.tau, np.sqrt(op.volumes))


def apply_cov(blocks, rhs):
    """Apply B to one or more columns, without assembling B."""
    out = np.empty_like(rhs, dtype=float)
    for block in blocks:
        out[block.rows] = block.apply(block.apply(rhs[block.rows], True), False)
    return out


def prior_cost(blocks, dx):
    total = 0.0
    for block in blocks:
        z = block.standardized(dx[block.rows])
        total += z @ z
    return 0.5 * total


def apply_root(blocks, x, transpose=False):
    out = np.empty_like(x, dtype=float)
    for block in blocks:
        out[block.rows] = block.apply(x[block.rows], transpose)
    return out


def _precision_block(name, source, rows):
    q = sparse.load_npz(source) if isinstance(source, (str, PathLike)) else source
    n = rows.stop - rows.start
    if not sparse.issparse(q) or q.shape != (n, n):
        raise ValueError(f"Laplace precision for {name} must be a sparse ({n}, {n}) matrix.")
    q = q.tocsc()
    if np.iscomplexobj(q.data) or not np.isfinite(q.data).all() or (q != q.T).nnz:
        raise ValueError(f"Laplace precision for {name} must be finite, real and symmetric.")
    try:
        # Q[P,P] = C.T C, so the covariance root maps z to P.T C^-1 z.
        c, order = reverse_cholesky(q)
        return PriorBlock(rows, c.tocsr(), precision=True, permutation=order)
    except CholmodNotPositiveDefiniteError as error:
        raise ValueError(f"Laplace precision for {name} must be positive definite.") from error


def _covariance_root(cov, n):
    if np.iscomplexobj(cov):
        raise ValueError("Laplace covariance must be real.")
    if cov.ndim < 2:
        var = np.atleast_1d(cov)
        if var.shape != (n,) or not np.all(np.isfinite(var) & (var > 0)):
            raise ValueError("Laplace variances must be finite and positive, one per state row.")
        return sparse.diags(np.sqrt(var), format="csr")
    if cov.shape != (n, n) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T):
        raise ValueError(f"Laplace covariance must be finite, symmetric and of shape ({n}, {n}).")
    return np.linalg.cholesky(cov)


def _fit(name, samples, options, info, progress):
    if not np.all(np.ptp(samples, axis=1) > 0):
        raise ValueError(f"Laplace graph fit for {name} needs members that differ in every state.")
    hops = options.get("neighbourhood_expansion", 2)
    if type(hops) is not int or hops < 1:
        raise ValueError("neighbourhood_expansion must be a positive integer.")
    graph = nx.power(parameter_graph(name, len(samples), options, info), hops)
    ridge = options.get("ridge_per_entry", 2.0)
    if ridge == 2.0:
        return optimize_sparse_affine_kr_map(samples.T, graph, use_tqdm=progress).tocsr()
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge_per_entry must be finite and nonnegative.")
    # graphite-maps fixes its penalty at 2 per entry; use its row solver for overrides.
    sd = samples.std(axis=1)
    u = ((samples - samples.mean(axis=1, keepdims=True)) / sd[:, None]).T
    c = sparse.lil_matrix((len(sd), len(sd)))
    for k in range(len(sd)):
        cols = [j for j in graph[k] if j < k] + [k]
        off, diag = solve_row_closed_form(U_reduced=u[:, cols], lambda_l2=ridge * len(cols))
        c[k, cols] = np.append(off, diag) / sd[cols]
    return c.tocsr()


def build_prior(samples, layout, info, options, imported, progress):
    """Choose the root before replacing member zero by the centre."""
    if not np.isfinite(samples).all():
        raise ValueError("Laplace prior members must be finite.")
    mu = samples.mean(axis=1)
    precision = options.get("precision", {})
    if not isinstance(precision, dict):
        return mu, [_precision_block("the whole state", precision, slice(0, len(mu)))]
    covariance = options.get("covariance", {})
    factors = options.get("precision_factor", {})
    matern = options.get("matern", {})
    unknown = (precision.keys() | covariance.keys() | factors.keys() | matern.keys()
               | options.get("parameter_graphs", {}).keys()) - layout.indices.keys()
    if unknown:
        raise ValueError(f"Unknown Laplace prior variable(s): {sorted(unknown)}.")
    limit = options.get("max_dense_size", 4096)
    if type(limit) is not int or limit < 0:
        raise ValueError("max_dense_size must be a nonnegative integer.")
    blocks = []
    for name in layout.variables:
        rows = layout.rows(name)
        n = rows.stop - rows.start
        if name in precision:
            blocks.append(_precision_block(name, precision[name], rows))
        elif name in factors:
            source = factors[name]
            c = sparse.load_npz(source) if isinstance(source, (str, PathLike)) else source
            if (not sparse.issparse(c) or c.shape != (n, n) or np.iscomplexobj(c.data)
                    or not np.isfinite(c.data).all() or not np.all(c.diagonal() > 0)
                    or sparse.triu(c, k=1).count_nonzero()):
                raise ValueError(f"precision_factor for {name} must be finite, real, lower triangular "
                                 f"and ({n}, {n}) with positive diagonal.")
            blocks.append(PriorBlock(rows, c.tocsr(), precision=True))
        elif name in matern:
            blocks.append(_matern_block(rows, matern[name]))
        elif n <= limit and name in covariance:
            source = covariance[name]
            cov = np.load(source, mmap_mode="r") if isinstance(source, (str, PathLike)) else np.asarray(source)
            blocks.append(PriorBlock(rows, _covariance_root(cov, n)))
        elif n <= limit and not imported and "variance" in info[name]:
            start = rows.start
            for mean, cov in prior_layers(info[name]):
                layer = slice(start, start + len(mean))
                mu[layer] = mean
                blocks.append(PriorBlock(layer, _covariance_root(cov, len(mean))))
                start = layer.stop
            if start != rows.stop:
                raise ValueError(f"Prior model for {name} does not match its state rows.")
        else:
            c = _fit(name, samples[rows], options, info, progress)
            blocks.append(PriorBlock(rows, c, precision=True))
    return mu, blocks
