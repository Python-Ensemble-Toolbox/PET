"""lowrankgn against the known posterior of a linear-Gaussian problem."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from misc.structures import PETDataFrame
from pipt import LowRankGN
from pipt.update_schemes.laplace_prior import MaternBlock, apply_cov, apply_root, build_prior, prior_cost
from pipt.update_schemes.lowrank_gn import damped_step, laplace_perturbations
from misc.structures.layout import StateLayout

NX, ND, NE = 12, 5, 400


class Forward:
    """`d = G x`, one datum per index."""

    def __init__(self, G):
        self.G = G
        index = list(range(G.shape[0]))
        self.input_dict = {"reporttype": "index", "reportpoint": index, "datatype": ["value"], "parallel": 1}
        self.true_order = ["index", index]
        self.datatype = ["value"]

    def run_fwd_sim(self, state, member_index, *args, **kwargs):
        return [{"value": np.array([y])} for y in self.G @ state["x"]]



class Linear(Forward):
    """With the weighted adjoint, `W^T G`."""

    def run_vjp(self, state, weights):
        return {"x": np.stack(list(weights["value"])).T @ self.G}


def run(keys_en_extra=None, sim=Linear, laplace=None, reference=None, **iteration):
    rng = np.random.default_rng(0)
    G = rng.standard_normal((ND, NX))
    R = np.array([0.01, 0.1, 0.5, 1.0, 4.0])
    d = rng.standard_normal(ND)
    index = pd.Index(range(ND), name="index")
    PETDataFrame(pd.DataFrame({"value": d}, index=index)).to_pickle("data.pkl")
    PETDataFrame(pd.DataFrame({"value": [["abs", r] for r in R]}, index=index)).to_pickle("var.pkl")
    np.savez("mean.npz", x=np.linspace(-1, 1, NX))

    keys_da = {
        "scheme": "lowrankgn", "analysis": "laplace", "obsname": "index", "datatype": ["value"],
        "truedataindex": list(range(ND)), "assimindex": list(range(ND)),
        "data": "data.pkl", "datavar": "var.pkl", "nosave": True,
        "iteration": {"max_iter": 30, "energy": 1.0, "data_misfit_tol": 1e-14, **iteration},
    }
    keys_en = {
        "ne": NE, "state": ["x"], "seed": 1, "disable_tqdm": True, "save_prior": False,
        "prior_x": {"vario": "exp", "mean": "mean.npz", "var": 2.0, "range": 3.0, "aniso": 1.0, "angle": 0.0,
                    "grid": [NX, 1, 1]},
        **(keys_en_extra or {}),
    }
    if laplace is not None:
        keys_da["laplace"] = laplace
    scheme = LowRankGN(keys_da, keys_en, sim(G))
    result = scheme.run_assimilation()

    if reference is None:
        h = np.abs(np.arange(NX) - np.arange(NX)[:, None])
        reference = np.linspace(-1, 1, NX), 2 * np.exp(-3 * (h / 3)**1.9)
    m, C = reference
    K = C @ G.T @ np.linalg.inv(G @ C @ G.T + np.diag(R))
    return scheme, result, m + K @ (d - G @ m), C - K @ G @ C


def test_the_centre_is_the_posterior_mean_and_the_samples_have_the_posterior_covariance():
    scheme, result, mean, cov = run()
    sd = np.sqrt(np.diag(cov))

    np.testing.assert_allclose(scheme.centre, mean, atol=1e-6 * sd.max())
    assert result.data_misfit < result.prior_data_misfit

    # The samples are exact draws, so only their number limits the agreement
    samples = np.asarray(result.x)
    assert samples.shape == (NX, NE)
    assert np.abs((samples.mean(axis=1) - mean) / sd).max() < 4 / np.sqrt(NE)
    np.testing.assert_allclose(samples.var(axis=1, ddof=1) / sd**2, 1.0, atol=4 * np.sqrt(2 / NE))


def test_a_truncated_basis_keeps_more_of_the_prior_spread():
    _, result, _, cov = run(energy=0.5)
    assert (np.asarray(result.x).var(axis=1, ddof=1) / np.diag(cov)).mean() > 1.1


def test_only_the_prior_and_the_samples_are_simulated_as_an_ensemble():
    runs = []

    class Counting(Linear):
        def run_fwd_sim(self, state, member_index, *args, **kwargs):
            runs.append(member_index)
            return super().run_fwd_sim(state, member_index, *args, **kwargs)

    _, result, *_ = run(sim=Counting)
    assert runs[:NE] == runs[-NE:] == list(range(NE))  # the prior, and the samples
    trials = runs[NE:-NE]  # each a single simulation, of the trial centre
    assert len(trials) >= result.nit and set(trials) == {0}


def test_the_basis_is_no_larger_than_max_rank():
    scheme, *_ = run(max_iter=2, max_rank=2)
    assert scheme.rank == 2


def test_the_full_jacobian_does_without_the_basis():
    # A basis of one direction would leave the centre short of the posterior mean
    scheme, _, mean, cov = run(full_jacobian=True, max_rank=1)
    assert scheme.rank == ND
    np.testing.assert_allclose(scheme.centre, mean, atol=1e-6 * np.sqrt(np.diag(cov)).max())


def test_an_imported_prior_ensemble_is_enough():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal((NX, NE)), axis=0)
    np.savez("prior.npz", x=x.astype("float32"))
    # No grid or graph: graphite-maps fits independent marginals.
    scheme, _, mean, cov = run({"importstate": "prior.npz", "prior_x": {}},
                               reference=(x.mean(axis=1), np.diag(x.var(axis=1))))
    np.testing.assert_allclose(scheme.centre, mean, atol=1e-6 * np.sqrt(np.diag(cov)).max())


def test_joint_precision_with_cross_variable_correlations_and_nontrivial_ordering():
    q = sparse.diags([np.full(7, -.4), np.arange(2., 10), np.full(7, -.4)], [-1, 0, 1], format="csc")
    sparse.save_npz("Q.npz", q)
    layout = StateLayout({"x": (0, 5), "y": (5, 8)})
    _, blocks = build_prior(np.ones((8, 3)), layout, {}, {"precision": "Q.npz"}, True, False)
    root = apply_root(blocks, np.eye(8))
    np.testing.assert_allclose(root @ root.T, np.linalg.inv(q.toarray()), atol=1e-14)
    np.testing.assert_allclose(apply_root(blocks, np.eye(8), transpose=True), root.T, atol=1e-14)
    dx = np.arange(8.)
    np.testing.assert_allclose(prior_cost(blocks, dx), .5 * dx @ q @ dx)
    np.testing.assert_allclose(blocks[0].precision_product(dx), q @ dx)


def test_supplied_precision_gives_the_analytic_map_in_a_full_run():
    q = sparse.diags([np.full(NX - 1, -.3), np.arange(2., NX + 2), np.full(NX - 1, -.3)], [-1, 0, 1])
    sparse.save_npz("precision.npz", q)
    samples = np.random.default_rng(8).normal(size=(NX, NE))
    np.savez("prior.npz", x=samples)
    scheme, _, mean, cov = run({"importstate": "prior.npz"}, laplace={"precision": "precision.npz"},
                               reference=(samples.mean(axis=1), np.linalg.inv(q.toarray())))
    np.testing.assert_allclose(scheme.centre, mean, atol=1e-6 * np.sqrt(np.diag(cov)).max())


@pytest.mark.parametrize("rank", [0, 3, 5])
def test_aw_step_and_undamped_draws_match_the_state_space_hessian(rank):
    rng = np.random.default_rng(7)
    C = np.tril(rng.normal(scale=.15, size=(5, 5))) + np.eye(5)
    Q = C.T @ C
    _, blocks = build_prior(np.ones((5, 2)), StateLayout({"x": (0, 5)}), {},
                            {"precision_factor": {"x": sparse.csr_matrix(C)}}, True, False)
    A = rng.normal(size=(rank, 5))
    if rank == 5:
        A[-1] = A[0]  # singular AW
    grad = rng.normal(size=5)  # not confined to the row space
    W = apply_cov(blocks, A.T) if rank else np.empty((5, 0))
    gram = A @ W
    for lam in (0, .01, 10):
        step = damped_step(apply_cov(blocks, grad), A, W, gram, lam)
        np.testing.assert_allclose(((1 + lam) * Q + A.T @ A) @ step, -grad, atol=1e-12)
    # Feed coordinate vectors for both independent noises: their transformed Gram is the exact covariance.
    basis = np.eye(5 + rank)
    perturb = laplace_perturbations(A, W, gram, apply_root(blocks, basis[:5]), basis[5:])
    np.testing.assert_allclose(perturb @ perturb.T, np.linalg.inv(Q + A.T @ A), atol=1e-12)


def test_spde_transpose_and_cost_for_nonsymmetric_operator():
    from scipy.sparse.linalg import splu
    K = sparse.csc_matrix([[2., .25, 0], [1., 3., .75], [0, .5, 2.]])
    tau, volumes = np.array([1., 2., 3.]), np.array([4., 1., 5.])
    block = MaternBlock(slice(0, 3), K, tau, np.sqrt(volumes))
    block.solver = splu(K)  # same transpose-solve contract, without an optional grid package
    Q = np.diag(tau) @ K.T @ np.diag(1 / volumes) @ K @ np.diag(tau)
    np.testing.assert_allclose(apply_cov([block], np.eye(3)), np.linalg.inv(Q), atol=1e-12)
    np.testing.assert_allclose(prior_cost([block], tau), .5 * tau @ Q @ tau)
    np.testing.assert_allclose(block.precision_product(tau), Q @ tau)


def test_restoring_whitened_checkpoint_uses_saved_factor_and_mean():
    scheme = object.__new__(LowRankGN)
    scheme.ensemble = type("Ensemble", (), {"state_layout": StateLayout({"x": (0, 2)})})()
    scheme.limits = {}
    c = sparse.csr_matrix([[2., 0.], [1., 3.]])
    mu, xi = np.array([2., -1.]), np.array([1., 3.])
    scheme._set_restart_state({"C": c, "mu": mu, "xi": xi})
    np.testing.assert_allclose(scheme.centre, mu + np.linalg.solve(c.toarray(), xi))
    assert "xi" not in scheme._get_restart_state()


def test_mixed_sources_and_graph_overrides():
    layout = StateLayout({"x": (0, 1), "y": (1, 3), "z": (3, 6)})
    x = np.cumsum(np.random.default_rng(1).standard_normal((6, 50)), axis=0)
    info = {"z": {"nx": 3, "ny": 1}}
    np.save("y.npy", [[3., .4], [.4, 1.]])
    opts = {"max_dense_size": 2, "precision": {"x": sparse.eye(1) * 4},
            "covariance": {"x": "not-used.npy", "y": "y.npy", "z": "too-large.npy"},
            "neighbourhood_expansion": 1, "ridge_per_entry": .02}
    _, blocks = build_prior(x, layout, info, opts, True, False)
    root = apply_root(blocks, np.eye(6))
    np.testing.assert_allclose((root @ root.T)[:3, :3], [[.25, 0, 0], [0, 3, .4], [0, .4, 1]])
    assert blocks[-1].factor[2, 0] == 0 and blocks[-1].factor[2, 1] != 0
    _, defaults = build_prior(x[3:], StateLayout({"z": (0, 3)}), info, {}, True, False)
    assert defaults[0].factor[2, 0] != 0
    assert not np.isclose(defaults[0].factor[1, 0], blocks[-1].factor[1, 0])


@pytest.mark.parametrize("q", [sparse.eye(3), sparse.csr_matrix([[1., 1.], [0., 1.]]),
                               sparse.diags([1., -1.])])
def test_bad_explicit_precision_does_not_fall_back(q):
    with pytest.raises(ValueError, match="precision"):
        build_prior(np.ones((2, 4)), StateLayout({"x": (0, 2)}), {}, {"precision": {"x": q}}, False, False)


def test_what_it_cannot_honour_is_refused():
    with pytest.raises(TypeError, match="run_vjp"):
        run(sim=Forward)
    np.savez("prior.npz", x=np.ones((NX, NE)))
    with pytest.raises(ValueError, match="differ in every state"):
        run({"importstate": "prior.npz"})


def scalar_scheme(*, variance=1., limits=None, iteration=None, **options):
    index = pd.Index([0], name="index")
    PETDataFrame(pd.DataFrame({"value": [4.]}, index=index)).to_pickle("data.pkl")
    PETDataFrame(pd.DataFrame({"value": [["abs", variance]]}, index=index)).to_pickle("var.pkl")
    da = {"scheme": "lowrankgn", "analysis": "laplace", "obsname": "index", "datatype": ["value"],
          "truedataindex": [0], "assimindex": [0], "data": "data.pkl", "datavar": "var.pkl", "nosave": True,
          "iteration": {"max_iter": 4, "full_jacobian": True, "lambda": 0., **(iteration or {})}, **options}
    prior = {"mean": 0., "var": 1.}
    if limits is not None:
        prior["limits"] = limits
    return LowRankGN(da, {"ne": 20, "state": ["x"], "seed": 3, "disable_tqdm": True, "prior_x": prior},
                     Linear(np.eye(1)))


def test_map_at_a_bound_scores_and_samples_about_the_feasible_centre():
    scheme = scalar_scheme(limits=[-10., 1.])
    result = scheme.run_assimilation()
    assert result.success and result.nit == 1 and "Stationary" in result.message
    np.testing.assert_array_equal(scheme.centre, [1.])
    np.testing.assert_array_equal(scheme.centre_pred, scheme.centre)
    assert result.x.max() == 1. and result.data_misfit == 9.


def test_restart_after_convergence_repeats_only_the_posterior(monkeypatch):
    options = {"restartsave": True, "iteration": {"lambda": 10., "data_misfit_tol": .99}}
    reference = scalar_scheme(**options)
    result = reference.run_assimilation()
    resumed = scalar_scheme(restart=True, **options)
    monkeypatch.setattr(resumed, "update_step", lambda: pytest.fail("repeated converged step"))
    restarted = resumed.run_assimilation()
    assert restarted.success and restarted.nit == result.nit == 1
    np.testing.assert_array_equal(resumed.centre, reference.centre)
    np.testing.assert_array_equal(restarted.x, result.x)


@pytest.mark.parametrize("crash", [True, False])
def test_retry_exhaustion_is_not_convergence(monkeypatch, crash):
    scheme = scalar_scheme(iteration={"max_inner_iter": 2})
    forward = Linear.run_fwd_sim
    runs = []

    def simulate(self, state, member_index):
        runs.append(member_index)
        if scheme.ne < len(runs) <= scheme.ne + 2:
            return False if crash else [{"value": np.array([-10.])}]
        return forward(self, state, member_index)

    monkeypatch.setattr(Linear, "run_fwd_sim", simulate)
    result = scheme.run_assimilation()
    assert not result.success and result.nit == 0
    assert ("All 2 trial forecasts failed" if crash else "No step lowered") in result.message
    assert len(runs) == scheme.ne + 2 + (0 if crash else scheme.ne)
    if crash:
        np.testing.assert_array_equal(result.x, [[0.]])
        np.testing.assert_array_equal(scheme.pred_data.matrix, [[0.]])


@pytest.mark.parametrize("variance", [0., -1., np.inf])
def test_invalid_variance_reports_the_observation_before_forecast(variance):
    with pytest.raises(ValueError, match=r"positive observation variance.*0, 'value'"):
        scalar_scheme(variance=variance)


def test_post_processed_forecasts_are_refused():
    with pytest.raises(NotImplementedError, match="post_process_forecast"):
        scalar_scheme(post_process_forecast=True)


def test_nonfinite_proposal_is_redamped_before_projection_or_forecast(monkeypatch):
    from pipt.update_schemes import lowrank_gn
    scheme = scalar_scheme(limits=[-10., 1.], iteration={"max_iter": 1})
    lambdas = []

    def step(v, A, W, gram, lam):
        lambdas.append(lam)
        return np.full_like(v, np.inf) if len(lambdas) == 1 else damped_step(v, A, W, gram, lam)

    monkeypatch.setattr(lowrank_gn, "damped_step", step)
    result = scheme.run_assimilation()
    assert len(lambdas) == 2 and lambdas[1] > lambdas[0]
    assert result.nit == 1 and np.isfinite(result.x).all()
