"""SVGD: its kernel, its Stein direction, its optimizers, its prior term, the adjoint weights it hands the
simulator, and a full run.

The full run is on a linear-Gaussian problem whose answer is known. The prior
SVGD is told about is the Gaussian with the mean and covariance of its prior
sample, so the ensemble must reach the posterior of that prior and the
likelihood, :math:`N(\\hat x, (H^\\mathsf{T} C_d^{-1} H + C_x^{-1})^{-1})`. The
prior and the data weigh about equally in it, so an error in either term of
the gradient moves the answer.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from pipt import SVGD
from pipt.update_schemes.svgd import (LBFGS, AdaGrad, Adam, EnsemblePrior, GradientDescent, GraphPrior,
                                      fit_kernel, make_optimizer, stein_direction, stein_terms)

NX = 2
LABELS = [1, 2, 3, 4]
SIGMA = 0.5
PRIOR_MEAN = np.array([0.2, 0.3])
PRIOR_VAR = 0.16
PRIOR_COV = np.array([[PRIOR_VAR, 0.06], [0.06, PRIOR_VAR]])
N_PRIOR = 500


class LinearAdjointSim:
    """``d = H x``, returning the weighted adjoint ``H^T w`` for the weights it is handed, as the JutulDarcy wrapper does."""

    def __init__(self, H, parameter="x"):
        self.H = H
        self.parameter = parameter
        self.input_dict = {"parallel": 1}
        self.compute_adjoints = True
        self.weighted_parameters = [parameter]
        self.adjoint_weights = None

    def run_fwd_sim(self, state, member_index):
        output = pd.DataFrame({"value": self.H @ np.ravel(state["x"])}, index=pd.Index(LABELS, name="position"))
        weights = self.adjoint_weights(output.copy(), member_index)
        w = weights.reindex(index=output.index, columns=output.columns).fillna(0.0)["value"].to_numpy()
        return output, pd.Series({self.parameter: self.H.T @ w}, dtype=object)


def _operator():
    return np.array([[1.0, 0.5], [0.2, 1.0], [1.0, -1.0], [0.3, 0.3]])


def _write_case(n_prior=N_PRIOR, seed=3):
    """Observations of a known truth, and ``n_prior`` prior realisations; the members are the first ``ne`` of them."""
    rng = np.random.default_rng(seed)
    H = _operator()
    truth = np.array([1.0, -0.5])
    obs = H @ truth + SIGMA * rng.standard_normal(len(LABELS))
    pd.DataFrame({"value": obs}, index=pd.Index(LABELS, name="position")).to_pickle("true_data.pkl")
    pd.DataFrame({"value": [["abs", SIGMA ** 2]] * len(LABELS)},
                 index=pd.Index(LABELS, name="position")).to_pickle("var.pkl")
    prior = PRIOR_MEAN[:, None] + np.linalg.cholesky(PRIOR_COV) @ rng.standard_normal((NX, n_prior))
    np.savez("prior.npz", x=prior)
    return H, obs


def _configs(ne, **svgd):
    cfg_en = {"ne": ne, "state": "x", "importstate": "prior.npz", "prior_x": {"var": PRIOR_VAR}}
    cfg_da = {"scheme": "svgd", "analysis": "adjoint", "obsname": "position",
              "data": "true_data.pkl", "datavar": "var.pkl", "nosave": True, "svgd": svgd}
    return cfg_da, cfg_en


def _drawn_prior_configs(ne, **svgd):
    """As :func:`_configs`, with the prior described in its block and drawn by PET instead of imported."""
    cfg_da, cfg_en = _configs(ne, **svgd)
    del cfg_en["importstate"]
    cfg_en["prior_x"] = {"mean": list(PRIOR_MEAN), "var": PRIOR_VAR, "range": 2, "aniso": 1, "angle": 0,
                         "vario": "sph", "grid": [NX, 1]}
    return cfg_da, cfg_en


def _prior_sample(n=None):
    return np.load("prior.npz")["x"][:, :n]


# ---------------------------------------------------------------------- #
# Kernel and direction
# ---------------------------------------------------------------------- #
def test_p_kernel_passes_through_its_two_percentiles():
    Z = np.random.default_rng(0).standard_normal((50, 30))
    D = cdist(Z.T, Z.T)
    alpha = 0.05
    p, sigma = fit_kernel(D, "p-kernel", alpha)
    d_lo, d_hi = np.quantile(D[np.triu_indices_from(D, k=1)], [alpha, 1 - alpha])
    def kernel(d):
        return np.exp(-(d / sigma) ** p)

    assert kernel(d_lo) == pytest.approx(1 - alpha)
    assert kernel(d_hi) == pytest.approx(alpha)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
def test_p_variant_of_the_inverse_multiquadric_passes_through_the_two_percentiles_of_the_p_kernel(alpha):
    Z = np.random.default_rng(0).standard_normal((50, 30))
    D = cdist(Z.T, Z.T)
    p, sigma = fit_kernel(D, "p-imq", alpha)
    d_lo, d_hi = np.quantile(D[np.triu_indices_from(D, k=1)], [alpha, 1 - alpha])
    def kernel(d):
        return 1 / np.sqrt(1 + (d / sigma) ** p)

    assert kernel(d_lo) == pytest.approx(1 - alpha)
    assert kernel(d_hi) == pytest.approx(alpha)
    # Its exponent is a fixed multiple of the p-kernel's for the same distances.
    ratio = np.log((alpha ** -2 - 1) / ((1 - alpha) ** -2 - 1)) / np.log(np.log(alpha) / np.log(1 - alpha))
    assert p == pytest.approx(ratio * fit_kernel(D, "p-kernel", alpha)[0])
    if alpha == 0.05:
        assert ratio == pytest.approx(2.02, abs=5e-3)


def test_p_variant_of_the_inverse_multiquadric_with_p_two_is_the_inverse_multiquadric():
    rng = np.random.default_rng(8)
    Z, gradient = rng.standard_normal((4, 9)), rng.standard_normal((4, 9))
    D = cdist(Z.T, Z.T)
    for ours, theirs in zip(stein_terms(Z, gradient, D, 2.0, 1.7, "p-imq"), stein_terms(Z, gradient, D, None, 1.7)):
        np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("p", [0.7, 2.0, 5.9])
def test_repulsion_of_the_p_variant_of_the_inverse_multiquadric_is_the_gradient_of_its_kernel(p):
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((3, 6))
    sigma = 1.3
    def kernel(d):
        return 1 / np.sqrt(1 + (d / sigma) ** p)


    def kernel_sum(i, Z):
        return sum(kernel(np.linalg.norm(Z[:, j] - Z[:, i])) for j in range(Z.shape[1]) if j != i)

    phi = stein_direction(Z, np.zeros_like(Z), cdist(Z.T, Z.T), p, sigma, "p-imq")
    h = 1e-6
    for i in range(Z.shape[1]):
        expected = np.zeros(Z.shape[0])
        for k in range(Z.shape[0]):
            up, down = Z.copy(), Z.copy()
            up[k, i] += h
            down[k, i] -= h
            # the repulsion of member i is minus the gradient, with respect to its own position, of the kernel sum
            expected[k] = -(kernel_sum(i, up) - kernel_sum(i, down)) / (2 * h)
        np.testing.assert_allclose(phi[:, i] * Z.shape[1], expected, rtol=1e-5, atol=1e-8)


def test_p_variant_of_the_inverse_multiquadric_stays_finite_when_the_members_are_almost_equally_far_apart():
    """In many dimensions the distances differ by a few percent and p is in the hundreds."""
    Z = np.random.default_rng(2).standard_normal((20000, 30))
    D = cdist(Z.T, Z.T)
    p, sigma = fit_kernel(D, "p-imq")
    assert p > 200
    attraction, repulsion, K = stein_terms(Z, np.ones_like(Z), D, 5 * p, sigma, "p-imq")      # and far beyond
    assert np.all(np.isfinite(attraction)) and np.all(np.isfinite(repulsion))
    assert np.all((K >= 0) & (K <= 1)) and np.allclose(np.diag(K), 1.0)


def test_gaussian_kernel_follows_the_median_rule():
    Z = np.random.default_rng(1).standard_normal((5, 12))
    D = cdist(Z.T, Z.T)
    p, sigma = fit_kernel(D, "gaussian")
    assert p == 2.0
    assert sigma ** 2 == pytest.approx(np.median(D[np.triu_indices_from(D, k=1)]) ** 2 / np.log(12))


@pytest.mark.parametrize("level", [0.3, 0.7, 0.95])
def test_inverse_multiquadric_has_its_level_at_the_median_distance(level):
    Z = np.random.default_rng(1).standard_normal((5, 12))
    D = cdist(Z.T, Z.T)
    p, sigma = fit_kernel(D, "imq", level=level)
    assert p is None
    median = np.median(D[np.triu_indices_from(D, k=1)])
    assert 1 / np.sqrt(1 + (median / sigma) ** 2) == pytest.approx(level)


@pytest.mark.parametrize("n", [12, 50])
def test_inverse_multiquadric_is_one_over_root_n_at_the_median_distance_by_default(n):
    Z = np.random.default_rng(1).standard_normal((5, n))
    D = cdist(Z.T, Z.T)
    p, sigma = fit_kernel(D, "imq")
    assert p is None
    median = np.median(D[np.triu_indices_from(D, k=1)])
    assert 1 / np.sqrt(1 + (median / sigma) ** 2) == pytest.approx(1 / np.sqrt(n))
    assert sigma == pytest.approx(median / np.sqrt(n - 1))


def test_svgd_takes_the_default_level_of_the_inverse_multiquadric_from_the_number_of_members():
    _write_case()
    assert SVGD(*_configs(16, kernel="imq"), LinearAdjointSim(_operator())).imq_level == pytest.approx(0.25)
    assert SVGD(*_configs(16, kernel="imq", imq_level=0.7), LinearAdjointSim(_operator())).imq_level == 0.7


def test_kernel_ignores_duplicated_members():
    Z = np.random.default_rng(2).standard_normal((4, 10))
    Z_dup = np.hstack([Z, Z[:, :1]])   # a failed member replaced by a copy of member 0
    p, sigma = fit_kernel(cdist(Z_dup.T, Z_dup.T), "p-kernel")
    assert np.isfinite(p) and np.isfinite(sigma) and p > 0
    # the copy's direction has no term from coinciding with its original
    phi = stein_direction(Z_dup, np.zeros_like(Z_dup), cdist(Z_dup.T, Z_dup.T), p, sigma)
    assert np.all(np.isfinite(phi))


@pytest.mark.parametrize("p", [0.7, 1.5, 2.0, 3.0, None])   # None: the inverse multiquadric
def test_repulsion_is_the_gradient_of_the_kernel(p):
    """With no data gradient, the direction is the mean of grad_{z_j} K(z_j, z_i) over j."""
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((3, 6))
    sigma = 1.3
    kernel = (lambda d: 1 / np.sqrt(1 + (d / sigma) ** 2)) if p is None else (lambda d: np.exp(-(d / sigma) ** p))

    def kernel_sum(i, Z):
        return sum(kernel(np.linalg.norm(Z[:, j] - Z[:, i])) for j in range(Z.shape[1]) if j != i)

    phi = stein_direction(Z, np.zeros_like(Z), cdist(Z.T, Z.T), p, sigma)
    h = 1e-6
    for i in range(Z.shape[1]):
        expected = np.zeros(Z.shape[0])
        for j in range(Z.shape[1]):
            if j == i:
                continue
            for k in range(Z.shape[0]):
                up, down = Z.copy(), Z.copy()
                up[k, j] += h
                down[k, j] -= h
                # only the j-th term of member i's sum moves with z_j
                expected[k] += (kernel_sum(i, up) - kernel_sum(i, down)) / (2 * h)
        np.testing.assert_allclose(phi[:, i], expected / Z.shape[1], rtol=1e-5, atol=1e-9)


def test_repulsion_stays_finite_when_the_members_are_almost_equally_far_apart():
    """In many dimensions the distances differ by a few percent and the p-kernel needs an exponent in the
    hundreds; ``sigma ** p`` is then far beyond the largest float."""
    Z = 3 * np.random.default_rng(31).standard_normal((20000, 30))
    D = cdist(Z.T, Z.T)
    p, sigma = fit_kernel(D, "p-kernel")
    assert p > 100 and p * np.log10(sigma) > 308                 # sigma ** p overflows

    phi = stein_direction(Z, np.zeros_like(Z), D, p, sigma)
    assert np.isfinite(phi).all() and np.abs(phi).max() > 0

    # The coefficient of the repulsion from its logarithm, which has no large number in it.
    off = ~np.eye(30, dtype=bool)
    log_c = np.log(p) - (D[off] / sigma) ** p + (p - 2) * np.log(D[off]) - p * np.log(sigma)
    c = np.zeros_like(D)
    c[off] = np.exp(log_c)
    # Entries that nearly cancel are compared on the scale of the largest one.
    np.testing.assert_allclose(phi, (Z * c.sum(axis=0) - Z @ c) / 30, rtol=1e-7, atol=1e-9 * np.abs(phi).max())


def test_repulsion_is_zero_between_members_the_kernel_does_not_connect():
    """Two groups far apart: the kernel is zero between them, and so is the repulsion, without a NaN."""
    rng = np.random.default_rng(32)
    Z = np.hstack([rng.standard_normal((4, 5)), 1e6 + rng.standard_normal((4, 5))])
    D = cdist(Z.T, Z.T)
    phi = stein_direction(Z, np.zeros_like(Z), D, 200.0, 2.0)
    assert np.isfinite(phi).all()


def test_direction_weights_the_gradients_with_the_kernel():
    rng = np.random.default_rng(5)
    Z, G = rng.standard_normal((3, 5)), rng.standard_normal((3, 5))
    D = cdist(Z.T, Z.T)
    p, sigma = 1.5, 2.0
    K = np.exp(-(D / sigma) ** p)
    data_part = stein_direction(Z, G, D, p, sigma) - stein_direction(Z, np.zeros_like(G), D, p, sigma)
    np.testing.assert_allclose(data_part, G @ K / 5)


def test_direction_weights_the_gradients_with_the_inverse_multiquadric():
    rng = np.random.default_rng(5)
    Z, G = rng.standard_normal((3, 5)), rng.standard_normal((3, 5))
    D = cdist(Z.T, Z.T)
    sigma = 2.0
    K = 1 / np.sqrt(1 + (D / sigma) ** 2)
    data_part = stein_direction(Z, G, D, None, sigma) - stein_direction(Z, np.zeros_like(G), D, None, sigma)
    np.testing.assert_allclose(data_part, G @ K / 5)


# ---------------------------------------------------------------------- #
# Optimizers
# ---------------------------------------------------------------------- #
def test_adagrad_is_the_default_optimizer():
    optimizer = make_optimizer({})
    assert isinstance(optimizer, AdaGrad)
    assert (optimizer.step_size, optimizer.autocorr) == (0.1, 0.9)


@pytest.mark.parametrize("name, cls", [("adagrad", AdaGrad), ("adam", Adam), ("gd", GradientDescent), ("lbfgs", LBFGS),
                                       ("Adam", Adam)])
def test_optimizer_is_chosen_by_name(name, cls):
    optimizer = make_optimizer({"optimizer": name, "step_size": 0.3})
    assert type(optimizer) is cls
    assert optimizer.step_size == 0.3


def test_optimizers_read_their_own_options():
    assert make_optimizer({"optimizer": "adagrad", "autocorr": 0.5}).autocorr == 0.5
    adam = make_optimizer({"optimizer": "adam", "beta1": 0.8, "beta2": 0.99})
    assert (adam.beta1, adam.beta2) == (0.8, 0.99)


@pytest.mark.parametrize("options, match", [
    ({"optimizer": "newton"}, "optimizer"),
    ({"step_size": 0.0}, "step_size"),
    ({"autocorr": 1.0}, "autocorr"),
    ({"optimizer": "adam", "beta2": 1.5}, "beta2"),
    ({"optimizer": "lbfgs", "memory": 0}, "memory"),
])
def test_optimizer_options_that_cannot_work_are_refused(options, match):
    with pytest.raises(ValueError, match=match):
        make_optimizer(options)


def test_gradient_descent_steps_along_the_direction():
    phi = np.random.default_rng(7).standard_normal((3, 4))
    np.testing.assert_array_equal(GradientDescent(0.2).step(phi), 0.2 * phi)


def test_adagrad_divides_by_the_running_root_mean_square():
    rng = np.random.default_rng(8)
    first, second = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    optimizer = AdaGrad(0.1, autocorr=0.9)
    # the first step has only its own direction to scale by: step_size along its sign
    np.testing.assert_allclose(optimizer.step(first), 0.1 * np.sign(first), rtol=1e-4)
    mean_square = 0.9 * first ** 2 + 0.1 * second ** 2
    np.testing.assert_allclose(optimizer.step(second), 0.1 * second / (AdaGrad.FUDGE + np.sqrt(mean_square)))


def test_adam_follows_its_two_running_means():
    rng = np.random.default_rng(9)
    first, second = rng.standard_normal((3, 4)), rng.standard_normal((3, 4))
    optimizer = Adam(0.1, beta1=0.9, beta2=0.999)
    # the start from zero is corrected for, so the first step is step_size along the sign
    np.testing.assert_allclose(optimizer.step(first), 0.1 * np.sign(first), rtol=1e-4)
    mean = (0.9 * 0.1 * first + 0.1 * second) / (1 - 0.9 ** 2)
    mean_square = (0.999 * 0.001 * first ** 2 + 0.001 * second ** 2) / (1 - 0.999 ** 2)
    np.testing.assert_allclose(optimizer.step(second), 0.1 * mean / (Adam.FUDGE + np.sqrt(mean_square)))


def _quadratic(seed=13, n=6, ne=3):
    """phi = -A (Z - target): minus the gradient of a convex quadratic with condition number 1000."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    A = Q @ np.diag(np.logspace(0, 3, n)) @ Q.T
    target = rng.standard_normal((n, ne))
    return (lambda Z: -A @ (Z - target)), target, rng.standard_normal((n, ne))


def test_lbfgs_reads_its_memory_option():
    assert make_optimizer({"optimizer": "lbfgs"}).memory == 10
    assert make_optimizer({"optimizer": "lbfgs", "memory": 4}).memory == 4


def test_lbfgs_first_step_is_step_size_along_the_direction():
    phi = np.random.default_rng(14).standard_normal((3, 4))
    step = LBFGS(0.2).step(phi, np.zeros_like(phi))
    np.testing.assert_allclose(step, 0.2 * phi / np.sqrt(np.mean(phi ** 2)))


def test_lbfgs_finds_the_minimum_of_an_ill_conditioned_quadratic():
    """With curvature from its pairs it closes in within a few dozen steps; the largest stable plain step does not."""
    direction, target, start = _quadratic()

    def distance_after(optimizer, iterations=40):
        Z = start.copy()
        for _ in range(iterations):
            Z = Z + optimizer.step(direction(Z), Z)
        return np.linalg.norm(Z - target) / np.linalg.norm(start - target)

    assert distance_after(LBFGS(step_size=1.0, memory=10)) < 1e-3
    assert distance_after(GradientDescent(1.9e-3)) > 1e-2     # the largest stable plain step is 2 / 1000


def test_lbfgs_steps_never_exceed_step_size():
    direction, _, Z = _quadratic()
    optimizer = LBFGS(step_size=0.05)
    for _ in range(15):
        step = optimizer.step(direction(Z), Z)
        assert np.sqrt(np.mean(step ** 2)) <= 0.05 * (1 + 1e-12)
        Z = Z + step


def test_lbfgs_step_always_has_a_component_along_the_direction():
    """Pairs are kept only with positive curvature, so the inverse Hessian stays positive definite."""
    rng = np.random.default_rng(15)
    optimizer = LBFGS(step_size=0.1, memory=5)
    Z = rng.standard_normal((4, 3))
    for _ in range(30):
        phi = rng.standard_normal((4, 3))      # directions unrelated to the steps: curvature of either sign
        step = optimizer.step(phi, Z)
        assert np.sum(step * phi) > 0
        Z = Z + step
    assert optimizer.skipped > 0 and len(optimizer.pairs) > 0


def test_lbfgs_skips_a_pair_with_negative_curvature():
    optimizer = LBFGS(step_size=0.1)
    Z = np.random.default_rng(16).standard_normal((3, 2))
    for _ in range(3):
        Z = Z + optimizer.step(Z, Z)           # phi = +Z: moving along it makes it grow
    assert optimizer.skipped == 2 and optimizer.pairs == []


def test_lbfgs_measures_the_step_the_members_actually_took():
    """The scheme clips the proposal to the prior limits; the pair must hold the clipped move."""
    direction, _, Z = _quadratic()
    optimizer = LBFGS(step_size=0.1)
    proposed = Z + optimizer.step(direction(Z), Z)
    clipped = np.clip(proposed, -0.5, 0.5)
    optimizer.step(direction(clipped), clipped)
    np.testing.assert_array_equal(optimizer.pairs[-1][0], (clipped - Z).ravel())


def test_lbfgs_reset_forgets_the_pairs_but_still_learns_from_the_step_being_taken():
    phi, _, Z = _quadratic()
    optimizer = LBFGS(step_size=0.1)
    for _ in range(3):
        Z = Z + optimizer.step(phi(Z), Z)
    assert optimizer.pairs
    optimizer.reset()
    assert optimizer.pairs == []
    Z = Z + 0.01 * phi(Z) / np.abs(phi(Z)).max()
    optimizer.step(phi(Z), Z)   # the move since the last call is remembered as usual
    assert len(optimizer.pairs) == 1


# ---------------------------------------------------------------------- #
# The prior term
# ---------------------------------------------------------------------- #
def test_prior_gradient_is_that_of_the_sample_covariance():
    """With more realisations than state variables, C^-1 (mu - x) for the sample's mean and covariance."""
    rng = np.random.default_rng(10)
    samples = rng.standard_normal((6, 6)) @ rng.standard_normal((6, 200)) + 3.0
    X = rng.standard_normal((6, 4))
    prior = EnsemblePrior(samples)
    assert prior.rank == 6
    expected = np.linalg.solve(np.cov(samples), samples.mean(axis=1, keepdims=True) - X)
    np.testing.assert_allclose(prior.gradient(X), expected, rtol=1e-8)


def test_prior_gradient_acts_only_where_the_sample_has_variance():
    """Fewer realisations than state variables: the pseudo-inverse, zero outside the sample's span."""
    rng = np.random.default_rng(11)
    samples = rng.standard_normal((30, 8))
    prior = EnsemblePrior(samples)
    assert prior.rank == 7                                        # centring takes one direction
    mean = samples.mean(axis=1, keepdims=True)
    X = rng.standard_normal((30, 4))
    np.testing.assert_allclose(prior.gradient(X), np.linalg.pinv(np.cov(samples)) @ (mean - X), rtol=1e-8, atol=1e-10)

    anomalies = samples - mean
    outside = rng.standard_normal((30, 1))
    outside -= anomalies @ np.linalg.lstsq(anomalies, outside, rcond=None)[0]   # orthogonal to the sample
    np.testing.assert_allclose(prior.gradient(mean + outside), 0.0, atol=1e-10)


def test_prior_truncation_drops_the_smallest_directions():
    rng = np.random.default_rng(12)
    samples = np.diag([10.0, 5.0, 1.0, 0.01]) @ rng.standard_normal((4, 300))
    assert EnsemblePrior(samples, energy=1.0).rank == 4
    truncated = EnsemblePrior(samples, energy=0.99)
    assert truncated.rank == 3
    # the dropped direction, the fourth axis, gets no pull
    assert abs(truncated.gradient(samples.mean(axis=1, keepdims=True) + np.eye(4)[:, 3:])[3, 0]) < 1e-2


def test_prior_misfit_is_the_quadratic_form_of_the_sample_covariance():
    rng = np.random.default_rng(21)
    sample = np.array([[1.0], [2.0], [0.5]]) * rng.standard_normal((3, 400)) + np.array([[0.3], [-1.0], [2.0]])
    X = rng.standard_normal((3, 5))
    deviation = X - sample.mean(axis=1, keepdims=True)
    expected = np.sum(deviation * np.linalg.solve(np.cov(sample), deviation), axis=0)
    np.testing.assert_allclose(EnsemblePrior(sample).misfit(X), expected)


def test_prior_gradient_is_minus_half_the_gradient_of_the_prior_misfit():
    """The line search compares misfits and the direction uses gradients; they must be those of one function."""
    rng = np.random.default_rng(22)
    prior = EnsemblePrior(rng.standard_normal((4, 300)))
    x, eps = rng.standard_normal((4, 1)), 1e-6
    numeric = [(prior.misfit(x + eps * e[:, None]) - prior.misfit(x - eps * e[:, None]))[0] / (2 * eps)
               for e in np.eye(4)]
    np.testing.assert_allclose(prior.gradient(x).ravel(), -0.5 * np.array(numeric), rtol=1e-6)


# ---------------------------------------------------------------------- #
# The scheme
# ---------------------------------------------------------------------- #
def test_svgd_needs_a_weighted_adjoint():
    _write_case()
    sim = LinearAdjointSim(_operator())
    sim.weighted_parameters = None
    with pytest.raises(ValueError, match="weighted_adjoint"):
        SVGD(*_configs(10), sim)


@pytest.mark.parametrize("kernel", ["laplace", "p"])   # "p" is not a short form of "p-kernel"
def test_svgd_rejects_an_unknown_kernel(kernel):
    _write_case()
    with pytest.raises(ValueError, match="'p-kernel', 'gaussian', 'imq' or 'p-imq'"):
        SVGD(*_configs(10, kernel=kernel), LinearAdjointSim(_operator()))


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
def test_svgd_rejects_an_imq_level_outside_zero_to_one(level):
    _write_case()
    with pytest.raises(ValueError, match="imq_level"):
        SVGD(*_configs(10, kernel="imq", imq_level=level), LinearAdjointSim(_operator()))


def test_p_kernel_is_the_default_kernel():
    H, _ = _write_case()
    assert SVGD(*_configs(10), LinearAdjointSim(H)).kernel == "p-kernel"


def test_svgd_rejects_an_unknown_optimizer():
    _write_case()
    with pytest.raises(ValueError, match="optimizer"):
        SVGD(*_configs(10, optimizer="newton"), LinearAdjointSim(_operator()))


@pytest.mark.parametrize("with_prior", [True, False])
def test_one_iteration_steps_along_the_stein_direction_of_the_log_posterior(with_prior):
    """One plain step moves the members by step_size times the direction, in prior standard deviations."""
    ne = 10
    H, obs = _write_case()
    svgd = {} if with_prior else {"prior_cov_samples": 0}
    scheme = SVGD(*_configs(ne, maxiter=1, optimizer="gd", step_size=0.01, **svgd), LinearAdjointSim(H))
    members = scheme.enX.copy()
    np.testing.assert_array_equal(members, _prior_sample(ne))
    result = scheme.run_assimilation()

    gradient = H.T @ (obs[:, None] - H @ members) / SIGMA ** 2
    if with_prior:
        sample = _prior_sample()
        gradient += np.linalg.solve(np.cov(sample), sample.mean(axis=1, keepdims=True) - members)

    scale = np.sqrt(PRIOR_VAR)
    Z = members / scale
    D = cdist(Z.T, Z.T)
    phi = stein_direction(Z, gradient * scale, D, *fit_kernel(D, "p-kernel"))
    np.testing.assert_allclose(result.x, (Z + 0.01 * phi) * scale)


def test_prior_sample_is_the_first_realisations_of_the_imported_file():
    H, _ = _write_case()
    scale = np.sqrt(PRIOR_VAR)   # the scheme keeps its prior in prior standard deviations
    everything = SVGD(*_configs(10), LinearAdjointSim(H))
    assert everything.prior.mean.ravel() * scale == pytest.approx(_prior_sample().mean(axis=1))
    first_fifty = SVGD(*_configs(10, prior_cov_samples=50), LinearAdjointSim(H))
    assert first_fifty.prior.mean.ravel() * scale == pytest.approx(_prior_sample(50).mean(axis=1))


def test_prior_sample_is_drawn_from_the_prior_blocks_without_moving_the_ensembles_stream():
    H, _ = _write_case()
    np.random.seed(5)
    without = SVGD(*_drawn_prior_configs(10, prior_cov_samples=0), LinearAdjointSim(H))
    stream_without = without.ensemble.rng.get_state()
    np.random.seed(5)
    scheme = SVGD(*_drawn_prior_configs(10, prior_cov_samples=4000), LinearAdjointSim(H))

    assert without.prior is None
    np.testing.assert_array_equal(scheme.enX, without.enX)
    np.testing.assert_array_equal(scheme.ensemble.rng.get_state()[1], stream_without[1])

    scale = np.sqrt(PRIOR_VAR)
    np.testing.assert_allclose(scheme.prior.mean.ravel() * scale, PRIOR_MEAN, atol=0.03)
    # the block describes a spherical variogram: unit-lag correlation 1 - 1.5/2 + 0.5/8
    correlation = scheme.prior.U * scheme.prior.S ** 2 @ scheme.prior.U.T
    np.testing.assert_allclose(correlation, [[1.0, 0.3125], [0.3125, 1.0]], atol=0.06)


@pytest.mark.parametrize("svgd, match", [
    ({"prior_cov_samples": N_PRIOR + 1}, "holds 500 realisations"),
    ({"prior_cov_samples": 2}, "prior_cov_samples"),
    ({"prior_cov_energy": 0.0}, "prior_cov_energy"),
])
def test_prior_options_that_cannot_work_are_refused(svgd, match):
    H, _ = _write_case()
    with pytest.raises(ValueError, match=match):
        SVGD(*_configs(10, **svgd), LinearAdjointSim(H))


@pytest.mark.parametrize("optimizer, configs", [("adagrad", _configs), ("adam", _configs), ("lbfgs", _configs),
                                                ("adagrad", _drawn_prior_configs)])
def test_a_resumed_svgd_run_matches_an_uninterrupted_one(optimizer, configs):
    """The optimizer's memory and the prior sample are those of the interrupted run."""
    ne, interrupt_at = 10, 2
    H, _ = _write_case()

    def run_configs(**da):
        cfg_da, cfg_en = configs(ne, maxiter=5, optimizer=optimizer)
        return {**cfg_da, **da}, cfg_en

    np.random.seed(1)
    reference = SVGD.assimilate(*run_configs(), LinearAdjointSim(H))

    class Interrupted(Exception):
        pass

    np.random.seed(1)
    scheme = SVGD(*run_configs(restartsave=True, restart_file="checkpoint.pkl"), LinearAdjointSim(H))
    hook = scheme.after_accepted_iteration

    def hook_then_die():
        hook()
        if scheme.iteration == interrupt_at:
            raise Interrupted

    scheme.after_accepted_iteration = hook_then_die
    with pytest.raises(Interrupted):
        scheme.run_assimilation()

    # Resume in a fresh process with a different random state.
    np.random.seed(12345)
    resumed = SVGD.assimilate(*run_configs(restart=True, restart_file="checkpoint.pkl"), LinearAdjointSim(H))
    assert resumed.nit == reference.nit
    np.testing.assert_array_equal(resumed.x, reference.x)


# ---------------------------------------------------------------------- #
# Stopping on the size of the Stein direction
# ---------------------------------------------------------------------- #
def test_stein_tol_stops_the_run_when_the_stein_direction_has_become_small():
    H, _ = _write_case()
    scheme = SVGD(*_configs(20, maxiter=200, step_size=0.05, stein_tol=0.1), LinearAdjointSim(H))
    lines = []
    scheme.logger = lambda *args, **kwargs: lines.extend(str(arg) for arg in args)
    moves = []
    step = scheme.update_step
    scheme.update_step = lambda: moves.append(scheme.enX.copy()) or step()
    result = scheme.run_assimilation()

    assert 0 < result.nit < 200
    assert result.why_stop == {"stein_tol": True}
    assert scheme.stein_ratio < 0.1
    assert "below stein_tol = 0.1" in scheme.conv_msg
    assert any(line.startswith(" SVGD finished: The Stein direction is") for line in lines)
    # The last call found the direction small: it simulated nothing and left the members where they were.
    assert len(moves) == result.nit + 1
    np.testing.assert_array_equal(result.x, moves[-1])
    assert scheme.ensemble_forecasts == result.nit + 1                # the prior, and one per iteration taken
    ratios = [float(line.split()[-1]) for line in lines if "relative to the first iteration" in line]
    assert ratios[0] == 1.0 and ratios[-1] < 0.1 and min(ratios[:-1]) >= 0.1


def test_the_run_table_shows_the_stein_direction_in_percent_of_the_first():
    H, _ = _write_case()
    scheme = SVGD(*_configs(20, maxiter=6, step_size=0.05), LinearAdjointSim(H))
    lines, table = [], []
    scheme.logger = lambda *args, **kwargs: table.append(kwargs) if kwargs else lines.extend(str(arg) for arg in args)
    scheme.run_assimilation()

    assert [row["Iteration"] for row in table] == ["0", "1", "2", "3", "4", "5", "6"]
    assert table[0]["Φ (%)"] == "" and table[1]["Φ (%)"] == "100"
    ratios = [float(line.split()[-1]) for line in lines if "relative to the first iteration" in line]
    np.testing.assert_allclose([float(row["Φ (%)"]) for row in table[1:]], 100 * np.array(ratios), rtol=5e-3)
    assert float(table[-1]["Φ (%)"]) < 100


def test_stein_change_keeps_a_run_going_while_the_stein_direction_still_changes():
    """With ``stein_change`` the direction must be small and have stopped changing. A run stopped by ``stein_tol``
    alone is taken as the reference: with a change no run can meet it goes on, with one every iteration meets it
    stops in the same iteration, and in between it stops later, in the first iteration that meets both."""
    H, _ = _write_case()

    def run(**options):
        options = {"stein_tol": 0.3, **options}
        scheme = SVGD(*_configs(20, maxiter=40, step_size=0.05, **options), LinearAdjointSim(H))
        lines = []
        scheme.logger = lambda *args, **kwargs: lines.extend(str(arg) for arg in args)
        result = scheme.run_assimilation()
        sizes = np.array([float(line.split()[-1]) for line in lines if "relative to the first iteration" in line])
        return scheme, result, sizes, lines

    alone, reference, _, lines = run()
    assert reference.why_stop == {"stein_tol": True} and alone.stein_change is None
    assert not any("has changed by less" in line for line in lines)

    _, restless, sizes, lines = run(stein_change=1e-9)
    assert restless.why_stop == {"maxiter": True} and sizes[-1] < 0.3
    assert any("has changed by less than this part of its size" in line and "1e-09" in line for line in lines)

    scheme, same, _, _ = run(stein_change=10.0)
    assert same.why_stop == {"stein_tol": True} and same.nit == reference.nit
    assert "stein_tol = 0.3" in scheme.conv_msg and "stein_change = 10" in scheme.conv_msg

    # In between: a tolerance and a change picked from a run that is not stopped, such that the first iteration
    # below the tolerance has changed too much and the next one has not.
    _, _, sizes, _ = run(stein_tol=0)
    changes = np.abs(np.diff(sizes)) / sizes[:-1]                         # changes[k - 1]: over step k
    k = int(np.flatnonzero(np.diff(changes) < 0)[0]) + 1                  # step k + 1 changed it less than step k
    tol = float(sizes[k] + sizes[k - 1]) / 2                              # met after k steps, not before
    level = float(changes[k] + changes[k - 1]) / 2                        # met by step k + 1, not by step k
    _, alone, _, _ = run(stein_tol=tol)
    scheme, both, _, _ = run(stein_tol=tol, stein_change=level)
    assert alone.why_stop == both.why_stop == {"stein_tol": True}
    assert (alone.nit, both.nit) == (k, k + 1)


def test_stein_change_must_be_positive():
    H, _ = _write_case()
    with pytest.raises(ValueError, match="stein_change"):
        SVGD(*_configs(10, stein_change=0.0), LinearAdjointSim(H))


def test_stein_tol_of_zero_takes_every_iteration():
    H, _ = _write_case()
    result = SVGD.assimilate(*_configs(20, maxiter=60, step_size=0.05, stein_tol=0), LinearAdjointSim(H))
    assert result.nit == 60
    assert result.why_stop == {"maxiter": True}


def test_stein_tol_is_a_small_fraction_by_default():
    H, _ = _write_case()
    assert SVGD(*_configs(10), LinearAdjointSim(H)).stein_tol == 0.001


@pytest.mark.parametrize("stein_tol", [-0.1, 1.0])
def test_stein_tol_outside_zero_to_one_is_refused(stein_tol):
    H, _ = _write_case()
    with pytest.raises(ValueError, match="stein_tol"):
        SVGD(*_configs(10, stein_tol=stein_tol), LinearAdjointSim(H))


def test_a_resumed_run_compares_with_the_stein_direction_of_the_first_iteration():
    """The size the criterion compares with is that of the interrupted run's first iteration, not of the first
    iteration after resuming, so the resumed run stops where an uninterrupted one does."""
    H, _ = _write_case()

    def run_configs(**da):
        cfg_da, cfg_en = _configs(20, maxiter=200, step_size=0.05, stein_tol=0.1)
        return {**cfg_da, **da}, cfg_en

    reference = SVGD.assimilate(*run_configs(), LinearAdjointSim(H))
    assert reference.why_stop == {"stein_tol": True} and reference.nit > 12

    class Interrupted(Exception):
        pass

    scheme = SVGD(*run_configs(restartsave=True, restart_file="checkpoint.pkl"), LinearAdjointSim(H))
    hook = scheme.after_accepted_iteration

    def hook_then_die():
        hook()
        if scheme.iteration == 10:
            raise Interrupted

    scheme.after_accepted_iteration = hook_then_die
    with pytest.raises(Interrupted):
        scheme.run_assimilation()

    resumed = SVGD.assimilate(*run_configs(restart=True, restart_file="checkpoint.pkl"), LinearAdjointSim(H))
    assert resumed.nit == reference.nit
    assert resumed.why_stop == {"stein_tol": True}
    np.testing.assert_array_equal(resumed.x, reference.x)


# ---------------------------------------------------------------------- #
# The log
# ---------------------------------------------------------------------- #
def _logged(scheme):
    """Run ``scheme`` and return the messages it logged, one per line."""
    lines = []
    scheme.logger = lambda *args, **kwargs: lines.extend(str(arg) for arg in args)
    scheme.run_assimilation()
    return lines


def test_a_run_logs_the_problem_every_iteration_and_a_summary():
    H, _ = _write_case()
    scheme = SVGD(*_configs(10, maxiter=3, kernel="gaussian"), LinearAdjointSim(H))
    lines = _logged(scheme)

    assert " SVGD: Stein variational gradient descent" in lines
    assert " Problem" in lines and " User-specified options" in lines
    assert "   kernel   gaussian" in lines                       # the options as they were given
    assert [line.split()[2] for line in lines if line.startswith("── Iteration")] == ["1", "2", "3"]
    for title in ("Members", "Gradient of the log-posterior", "Kernel", "Stein direction", "Step"):
        assert lines.count(title) == 3
    assert " SVGD finished: Maximum number of iterations reached" in lines
    assert scheme.ensemble_forecasts == 4                        # the prior and three iterations


def test_the_log_gives_the_mean_data_misfit_the_members_start_an_iteration_with():
    H, _ = _write_case()
    scheme = SVGD(*_configs(10, maxiter=1), LinearAdjointSim(H))
    lines = _logged(scheme)
    logged = [line.split()[-1] for line in lines if line.strip().startswith("Data misfit, mean")]
    assert logged == [f"{scheme.prior_data_misfit_mean:.3e}"]


def test_stein_terms_are_the_attraction_and_the_repulsion_of_the_direction():
    rng = np.random.default_rng(21)
    Z, gradient = rng.standard_normal((3, 12)), rng.standard_normal((3, 12))
    D = cdist(Z.T, Z.T)
    p, sigma = fit_kernel(D, "p-kernel")
    attraction, repulsion, K = stein_terms(Z, gradient, D, p, sigma)
    np.testing.assert_allclose(K, np.exp(-(D / sigma) ** p))
    np.testing.assert_allclose(attraction, gradient @ K)
    np.testing.assert_allclose(repulsion / 12, stein_direction(Z, np.zeros_like(Z), D, p, sigma))
    np.testing.assert_array_equal(stein_direction(Z, gradient, D, p, sigma), (attraction + repulsion) / 12)


def test_svgd_is_registered_under_its_gradient_source():
    from pipt.update_schemes.registry import get_scheme
    assert get_scheme("svgd", "adjoint") is SVGD
    with pytest.raises(KeyError, match="no 'approx' analysis flavour"):
        get_scheme("svgd", "approx")


@pytest.mark.parametrize("kernel, optimizer", [("p-kernel", "adagrad"), ("gaussian", "adagrad"), ("p-kernel", "adam"), ("p-kernel", "gd"),
                                               ("p-kernel", "lbfgs"), ("gaussian", "lbfgs"), ("imq", "adagrad"),
                                               ("p-imq", "adagrad")])
def test_svgd_reaches_the_linear_gaussian_posterior(kernel, optimizer):
    ne = 100
    H, obs = _write_case()
    sample = _prior_sample()
    prior_mean, prior_precision = sample.mean(axis=1), np.linalg.inv(np.cov(sample))
    cov = np.linalg.inv(H.T @ H / SIGMA ** 2 + prior_precision)
    mean = cov @ (H.T @ obs / SIGMA ** 2 + prior_precision @ prior_mean)

    result = SVGD.assimilate(*_configs(ne, maxiter=300, step_size=0.05, kernel=kernel, optimizer=optimizer),
                             LinearAdjointSim(H))
    X = result.x

    std = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(X.mean(axis=1), mean, atol=0.2 * std.min())
    np.testing.assert_allclose(X.std(axis=1, ddof=1), std, rtol=0.15)
    assert np.corrcoef(X)[0, 1] == pytest.approx(cov[0, 1] / (std[0] * std[1]), abs=0.15)


def test_the_posterior_is_not_the_likelihood():
    """The case tells the two apart: the prior moves the answer by far more than the tolerances above."""
    H, obs = _write_case()
    sample = _prior_sample()
    prior_precision = np.linalg.inv(np.cov(sample))
    likelihood_precision = H.T @ H / SIGMA ** 2
    cov = np.linalg.inv(likelihood_precision + prior_precision)
    mean = cov @ (H.T @ obs / SIGMA ** 2 + prior_precision @ sample.mean(axis=1))
    likelihood_mean = np.linalg.solve(likelihood_precision, H.T @ obs / SIGMA ** 2)
    assert np.abs(mean - likelihood_mean).max() > 5 * 0.2 * np.sqrt(np.diag(cov)).min()


# ---------------------------------------------------------------------- #
# The prior from a sparse factor fitted on a graph (``prior_cov_estimate = "graph"``)
# ---------------------------------------------------------------------- #
def _chain_sample(p=12, n=6000, seed=21):
    """A sample of a chain of ``p`` variables whose inverse covariance is tridiagonal, and that inverse covariance."""
    precision = 2.0 * np.eye(p) - 0.8 * (np.eye(p, k=1) + np.eye(p, k=-1))
    rng = np.random.default_rng(seed)
    samples = 1.5 + np.linalg.cholesky(np.linalg.inv(precision)) @ rng.standard_normal((p, n))
    return samples, precision


def test_graph_prior_recovers_an_inverse_covariance_that_is_sparse_on_its_graph():
    samples, precision = _chain_sample()
    prior = GraphPrior(samples, [((0, 12), nx.path_graph(12))], hops=1)
    assert prior.rank == 12
    np.testing.assert_allclose(prior.mean.ravel(), 1.5, atol=0.05)
    np.testing.assert_allclose((prior.L_inv.T @ prior.L_inv).toarray(), precision, atol=0.12)
    # and nothing outside the graph: the factor has a diagonal and one neighbour a row
    assert prior.L_inv.nnz <= 2 * 12 - 1


def test_graph_prior_factor_whitens_and_its_solves_invert_it():
    samples, precision = _chain_sample()
    prior = GraphPrior(samples, [((0, 12), nx.path_graph(12))], hops=2)
    deviation = samples[:, :50] - prior.mean
    L = prior.L @ np.eye(12)
    np.testing.assert_allclose(prior.L @ (prior.L_inv @ deviation), deviation, atol=1e-10)
    np.testing.assert_allclose(prior.L.T @ np.eye(12), L.T, atol=1e-12)
    np.testing.assert_allclose(L @ L.T, np.linalg.inv(precision), atol=0.08)          # C = L L'
    np.testing.assert_allclose(np.cov(prior.L_inv @ (samples - prior.mean)), np.eye(12), atol=0.06)
    assert prior.L.shape == (12, 12) == prior.L.T.shape


def test_graph_prior_gradient_is_minus_half_the_gradient_of_its_misfit():
    samples, _ = _chain_sample(n=300)
    prior = GraphPrior(samples, [((0, 12), nx.path_graph(12))])
    rng = np.random.default_rng(8)
    x, direction, h = rng.standard_normal((12, 1)), rng.standard_normal((12, 1)), 1e-5
    numeric = -(prior.misfit(x + h * direction) - prior.misfit(x - h * direction)) / (4 * h)
    assert float((prior.gradient(x) * direction).sum()) == pytest.approx(float(numeric[0]), rel=1e-6)


def test_graph_prior_fits_its_groups_apart_and_leaves_out_a_variable_that_does_not_vary():
    """Two groups of rows, the second with a row that is the same in every realisation. The groups are independent
    in the prior, and the constant row has no prior term and is not moved by the factor."""
    first, _ = _chain_sample(p=5, n=400, seed=1)
    second, _ = _chain_sample(p=4, n=400, seed=2)
    second[2] = 0.7
    samples = np.vstack([first, second])
    prior = GraphPrior(samples, [((0, 5), nx.path_graph(5)), ((5, 9), nx.path_graph(4))], hops=2)

    assert prior.rank == 8 and list(prior.rows) == [0, 1, 2, 3, 4, 5, 6, 8]
    precision = (prior.L_inv.T @ prior.L_inv).toarray()
    assert np.all(precision[:5, 5:] == 0) and np.all(precision[7] == 0)
    X = samples[:, :3] + 0.1
    assert np.all(prior.gradient(X)[7] == 0)
    assert np.all((prior.L @ np.ones((8, 2)))[7] == 0)
    # rows 6 and 8 are not neighbours once row 7 is gone: its node is taken out, the gap is not bridged
    assert precision[6, 8] == 0


def test_graph_prior_gives_a_full_rank_prior_from_fewer_realisations_than_variables():
    """Ten realisations of twelve variables: their covariance has nine directions, the fitted factor has all twelve."""
    samples, _ = _chain_sample(n=10)
    assert EnsemblePrior(samples).rank == 9
    prior = GraphPrior(samples, [((0, 12), nx.path_graph(12))])
    assert prior.rank == 12
    assert np.linalg.matrix_rank((prior.L_inv.T @ prior.L_inv).toarray()) == 12


def test_prior_cov_estimate_graph_gives_the_scheme_the_fitted_factor():
    """The graph is EnIF's: given under ``enif.parameter_graphs``, or independent variables without one."""
    H, _ = _write_case()
    assert isinstance(SVGD(*_configs(10), LinearAdjointSim(H)).prior, EnsemblePrior)       # the sample, unless asked

    cfg_da, cfg_en = _configs(10, prior_cov_estimate="graph")
    independent = SVGD(cfg_da, cfg_en, LinearAdjointSim(H))
    assert isinstance(independent.prior, GraphPrior) and independent.prior.rank == NX
    precision = (independent.prior.L_inv.T @ independent.prior.L_inv).toarray()
    assert precision[0, 1] == 0                                                            # no edge, no dependence

    cfg_da, cfg_en = _configs(10, prior_cov_estimate="graph")
    cfg_da["enif"] = {"parameter_graphs": {"x": nx.path_graph(NX)}, "neighbourhood_expansion": 1}
    scheme = SVGD(cfg_da, cfg_en, LinearAdjointSim(H))
    precision = (scheme.prior.L_inv.T @ scheme.prior.L_inv).toarray()
    # the scheme keeps its prior in prior standard deviations
    np.testing.assert_allclose(precision, np.linalg.inv(PRIOR_COV / PRIOR_VAR), atol=0.12)
    Z = scheme.enX / scheme.state_scaling[:, None]
    np.testing.assert_allclose(scheme.prior_gradient(Z), precision @ (scheme.prior.mean - Z), atol=1e-12)


def test_svgd_runs_with_the_prior_from_the_fitted_factor():
    H, _ = _write_case()
    cfg_da, cfg_en = _configs(20, maxiter=6, step_size=0.05, prior_cov_estimate="graph")
    cfg_da["enif"] = {"parameter_graphs": {"x": nx.path_graph(NX)}}
    scheme = SVGD(cfg_da, cfg_en, LinearAdjointSim(H))
    scheme.logger = lambda *args, **kwargs: None
    scheme.run_assimilation()
    assert scheme.iteration == 6
    assert scheme.data_misfit_mean < scheme.prior_data_misfit_mean


def test_prior_cov_estimate_refuses_what_it_does_not_know():
    H, _ = _write_case()
    with pytest.raises(ValueError, match="prior_cov_estimate"):
        SVGD(*_configs(10, prior_cov_estimate="glasso"), LinearAdjointSim(H))
