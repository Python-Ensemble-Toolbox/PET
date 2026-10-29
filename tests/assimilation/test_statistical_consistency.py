"""Statistical consistency: every method against the known posterior, not against pinned numbers.

A linear sanity check. The model is linear, `d = G x + noise`, and the prior and the noise
are Gaussian, so the posterior is known exactly. Every registered (scheme, analysis)
combination that claims to sample it is run on the same problem and compared with it, on
both its mean and its spread. The characterisation tests pin today's output, wrong or not;
the Van der Pol tests check that the misfit falls; neither says whether the answer is
right.

The cases are multivariate but not spatial: one state vector of 1, 2 or 10 unknowns, with
a random prior correlation and a dense random `G`, observed through fewer data than
unknowns (except in 1D), with noise variances spanning three orders of magnitude.
Nothing has a position, so localization is out of scope, and EnIF is given a complete
graph (no assumed sparsity) rather than the nearest-neighbour one it builds from a grid.

The reference is the exact update of the prior *ensemble*: its own mean and covariance,
through the Kalman gain. That is the right answer for those members, so sampling error in
the prior does not count against a method; what is left (the perturbed observations) is
averaged out over a few seeds. With 100 members and five seeds, the correct methods were
measured within 0.14 posterior standard deviations on the mean and 0.90--1.11 on the
variance ratio, over four independent sets of seeds; the tolerances leave a margin above that.

A method passes if it does so in every dimension: a case that happens not to show a fault
does not make the method right there. Expected deviations are listed in ``DEVIATIONS``,
per method, each with its reason. They are strict xfails: when one is fixed, its test
fails until the entry is removed. Those in ``BORDERLINE`` are not, since they deviate by
about as much as the tolerance, and so pass on some seeds.
"""

import networkx
import numpy as np
import pandas as pd
import pytest

from misc.structures import PETDataFrame
from pipt.update_schemes.registry import available_schemes, get_scheme

NE = 100
SEEDS = range(5)

#: Bounds on the seed-averaged error: of the mean, in posterior standard deviations, and
#: of the variance, as a ratio to the posterior variance. Per unknown, the worst counts.
MEAN_TOL = 0.2
VAR_RATIO = (0.85, 1.2)

#: Left out of the parametrization, and why.
EXCLUDED = {
    # Aliases; test_legacy_scheme_names checks they are exactly what they name.
    ("co_lm_enrml", "approx"): "alias",
    ("gn_enrml", "subspace"): "alias",
    # Multilevel ES-MDA needs a multilevel ensemble; test_multilevel runs it.
    ("esmda", "hybrid"): "multilevel",
    # `approx` drops the prior increment term (Chen & Oliver 2013). For ES, EnKF and ES-MDA
    # that term is zero, but the iterative schemes converge to a different point: with
    # 2000 members, 1.14 x the posterior variance in 1D and 0.4 sd off the mean in 10D.
    ("lmenrml", "approx"): "approximation",
    ("gnenrml", "approx"): "approximation",
}

#: Expected deviations from the posterior, per (scheme, analysis).
DEVIATIONS = {
    ("esmda", "full"): "Exact with one MDA step, not with four: 1.6 x the posterior variance in 1D",
    **{
        ("esmda", analysis): "Exact with one MDA step, not with four: up to 3 x the posterior "
        "variance in 1D; suspected to re-update the prior at every step"
        for analysis in ("subspace", "subspace2")
    },
    ("lmenrml", "subspace"): "Can stop at about the prior: the first step, at the default "
    "lambda, barely moves the misfit, which reads as convergence",
    ("esmda", "enif"): "About 0.2 sd off the mean in 10D at 100 members, though exact at 2000: "
    "presumably its regularised estimators",
}

#: Deviations about the size of the tolerance: non-strict, so a change in the noise does not fail them.
BORDERLINE = {("esmda", "enif")}

DIMENSIONS = (1, 2, 10)

COMBOS = [c for c in available_schemes() if c not in EXCLUDED]


class Linear:
    """`d = G x`: one datum per index, all of type `value`."""

    def __init__(self, G):
        self.G = G
        index = list(range(G.shape[0]))
        self.input_dict = {"reporttype": "index", "reportpoint": index, "datatype": ["value"], "parallel": 1}
        self.true_order = ["index", index]
        self.datatype = ["value"]

    def run_fwd_sim(self, state, member_index, *args, **kwargs):
        return [{"value": np.array([y])} for y in self.G @ state["x"]]


def problem(nx):
    """The prior, the model, the noise and the observations, for `nx` unknowns."""
    rng = np.random.default_rng(nx)
    if nx == 1:
        G, R = np.array([[2.0]]), np.array([0.5])
    elif nx == 2:
        G, R = np.array([[1.0, 1.0]]), np.array([0.1])
    else:
        G, R = rng.standard_normal((4, nx)), np.array([0.01, 0.1, 1.0, 10.0])
    A = rng.standard_normal((nx, nx + 2))
    S = A @ A.T
    sd = np.linspace(1, 3, nx)
    C = sd[:, None] * S / np.sqrt(np.outer(np.diag(S), np.diag(S))) * sd[None]
    m = np.linspace(-1, 1, nx)
    truth = rng.multivariate_normal(m, C)
    d = G @ truth + np.sqrt(R) * rng.standard_normal(len(R))
    return {"G": G, "R": R, "C": C, "m": m, "d": d}


def exact_update(X, G, R, d):
    """The Kalman update of the ensemble's own mean and covariance."""
    m, C = X.mean(axis=1), np.atleast_2d(np.cov(X))
    K = C @ G.T @ np.linalg.inv(G @ C @ G.T + np.diag(R))
    return m + K @ (d - G @ m), C - K @ G @ C


def assimilate(scheme, analysis, P, seed):
    """The posterior ensemble, and the prior ensemble it started from."""
    nx, nd = len(P["m"]), len(P["d"])
    prior = np.random.default_rng(1000 + seed).multivariate_normal(P["m"], P["C"], size=NE).T
    np.savez("prior.npz", x=prior)
    index = pd.Index(range(nd), name="index")
    PETDataFrame(pd.DataFrame({"value": P["d"]}, index=index)).to_pickle("data.pkl")
    PETDataFrame(pd.DataFrame({"value": [["abs", r] for r in P["R"]]}, index=index)).to_pickle("var.pkl")

    keys_da = {
        "scheme": scheme, "analysis": analysis, "obsname": "index", "datatype": ["value"],
        "truedataindex": list(range(nd)),
        # EnKF assimilates one datum at a time; the others, all at once.
        "assimindex": [[i] for i in range(nd)] if scheme == "enkf" else list(range(nd)),
        "data": "data.pkl", "datavar": "var.pkl", "energy": 1.0,
        "mda": {"tot_assim_steps": 4, "inflation_param": [4, 4, 4, 4]},
        "iteration": {"max_iter": 20, "energy": 1.0},
        "enif": {"parameter_graphs": {"x": networkx.complete_graph(nx)}},
    }
    keys_en = {
        "ne": NE, "state": ["x"], "importstate": "prior.npz", "seed": seed, "disable_tqdm": True,
        # The variance only sets the state scaling, to which the update is invariant.
        "prior_x": {"mean": P["m"].tolist(), "var": float(np.diag(P["C"]).mean()), "grid": [nx, 1]},
    }
    run = get_scheme(scheme, analysis)(keys_da, keys_en, Linear(P["G"]))
    run.run_assimilation()
    return np.asarray(run.enX, dtype=float), prior


def deviation(scheme, analysis, nx):
    """How the seed-averaged posterior misses the exact one in `nx` unknowns; `None` if it does not."""
    P = problem(nx)
    mean_errors, var_ratios = [], []
    for seed in SEEDS:
        posterior, prior = assimilate(scheme, analysis, P, seed)
        mean, cov = exact_update(prior, P["G"], P["R"], P["d"])
        sd = np.sqrt(np.diag(cov))
        mean_errors.append((posterior.mean(axis=1) - mean) / sd)
        var_ratios.append(posterior.var(axis=1, ddof=1) / sd**2)
    mean_error = np.abs(np.mean(mean_errors, axis=0)).max()
    var_ratio = np.mean(var_ratios, axis=0)

    misses = []
    if mean_error > MEAN_TOL:
        misses.append(f"mean off by {mean_error:.2f} posterior sd")
    if not VAR_RATIO[0] <= var_ratio.min() <= var_ratio.max() <= VAR_RATIO[1]:
        misses.append(f"variance {var_ratio.min():.2f}..{var_ratio.max():.2f} x the posterior's")
    return ", ".join(misses) or None


def cases():
    for scheme, analysis in COMBOS:
        reason = DEVIATIONS.get((scheme, analysis))
        strict = (scheme, analysis) not in BORDERLINE
        marks = [pytest.mark.xfail(reason=reason, strict=strict)] if reason else []
        yield pytest.param(scheme, analysis, marks=marks, id=f"{scheme}-{analysis}")


@pytest.mark.parametrize("scheme, analysis", list(cases()))
def test_the_posterior_is_the_exact_one(scheme, analysis):
    misses = {nx: deviation(scheme, analysis, nx) for nx in DIMENSIONS}
    misses = {nx: miss for nx, miss in misses.items() if miss}
    assert not misses, "; ".join(f"{nx} unknowns: {miss}" for nx, miss in misses.items())


def test_every_combination_is_checked_or_excluded_for_a_reason():
    assert set(EXCLUDED) <= set(available_schemes())
    assert set(DEVIATIONS) <= set(COMBOS)
    assert BORDERLINE <= set(DEVIATIONS)
