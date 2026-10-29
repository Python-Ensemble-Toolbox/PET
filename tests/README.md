# Tests

```sh
pytest                    # everything
pytest -m "not slow"      # without the end-to-end runs with a parallel forecast
pytest tests/assimilation/test_numerical_characterisation.py
```

The MiniRes tests are skipped unless the `minires` extra is installed (`pip install -e .[minires]`).
Every test starts in its own temporary directory (`conftest.py`), since PET writes its output to
the working directory.

Most of the suite is unit and regression tests of one module or one fixed bug; their docstrings say
what each pins. This page is about the rest: the **cases**, i.e. the synthetic problems that the
methods are run on end to end, and which **methods** run on which case.

## Layout

| | |
|---|---|
| `assimilation/` | `pipt`: the update schemes, their analyses, localization, the ensemble, data reading. |
| `optimization/` | `popt`: the optimizers, their ensembles and gradients, bounds, constraints. |
| the top level | What both share: data structures, config parsing, simulators, the CLI, imports, logging. |

## The cases

### Assimilation

#### Van der Pol

The workhorse. The oscillator `x'' - μ(1 - x²)x' + x = 0` (`simulator/vanderpol.py`), with the
initial conditions `x1`, `x2` and the damping `μ` as the unknowns (truth `1, 0, 1`; prior means
`0.05, 0.05, 1.5`). `x1` is observed at 15 steps with noise σ = 0.1. The simulator also
integrates the sensitivities, so it provides adjoints.

It comes in two sizes:

- **1000 members** (`test_assimilation_pipeline.py`, marked `slow`): asserts *properties*, i.e.
  that the data misfit falls below a threshold and that the posterior mean of `μ` is within 20 %
  of the prior's error. Catches a method that is broken, not one that has changed.
- **100 members, seeded, serial** (`test_numerical_characterisation.py`): asserts *numbers*. The
  posterior ensemble and the misfit trajectory are compared to `characterisation_reference.npz`
  at `rtol = 1e-9`. Catches any change to the numerics, intended or not. If a change is intended,
  regenerate with `python tests/assimilation/test_numerical_characterisation.py --regenerate`
  and check the diff; a mismatch can also come from a new numpy or BLAS rather than from the code.

The small case is reused (via `_write_synthetic_case` and `_write_config`) wherever a real run is
needed but its quality is not the point: restarts, seeding, the forecast backends, QA/QC, the
legacy scheme names, and the multilevel scheme (2 levels × 10 members).

#### Statistical consistency (vs. known posterior), not just previous/pinned values

`test_statistical_consistency.py`, a linear sanity check. The model is `d = G x + noise`, with a
Gaussian prior and Gaussian noise, so the correct posterior is known exactly. Every registered
scheme and analysis is run on it and compared with that posterior, on both the mean and the spread.

The cases are multivariate but not spatial: 1, 2 or 10 unknowns, a random prior correlation, a
dense random `G` with fewer data than unknowns, and noise variances spanning three orders of
magnitude. The reference is the exact update of the prior *ensemble*'s own mean and covariance, so
prior sampling error does not count against a method. The error is averaged over 20 seeds of 100
members each. Correct methods come within 0.08 posterior sd on the mean and 0.96–1.06 on the
variance ratio, and the tolerances (0.2, and 0.85–1.2) leave a margin above that. Known
deviations are listed, with their reasons, in `DEVIATIONS` (strict xfails, so a fix shows up)
and `EXCLUDED`.

#### Linear 1D field

`simulator.simple_models.lin_1d`: the state is a 1D field, 150 cells in a row, with a spherical
variogram of range 20, and is observed directly at every fifth cell. The model being the identity, the
case isolates the analysis from any nonlinearity.

- `test_linear_model.py`: LM-EnRML, `full` analysis, 250 members; the posterior mean at six cells
  is pinned.
- `test_enif.py`: one cell, prior N(0, 1), observed at 1 with variance 0.25, 400 members. The
  posterior is known exactly, N(0.8, 0.2), so ES-MDA with the `enif` analysis is checked against
  it, rather than against stored numbers.

#### MiniRes

A two-phase waterflood (`simulator/minires.py`, wrapping [MiniRes](https://github.com/patnr/MiniRes)),
the only case resembling a reservoir.

- `test_minires_sim.py`: the wrapper on an 8 × 8 grid: the simulator protocol, pickling, the state
  as log-permeability in the declared grid order, crashed members, and the adjoint.
- `test_benchmark_minires.py` (marked `slow`): runs every method of the
  [benchmark](../benchmarks/minires/README.md) on one tiny twin experiment (coarse grid, 6 members)
  and asserts that the scores are finite and the misfit did not increase. It checks that the
  benchmark still runs, not how well the methods do; that is `benchmarks/minires/RESULTS.md`.

### Optimization

#### Rosenbrock and quadratic

The Rosenbrock function (`scipy.optimize.rosen`), mostly in 2D from about `(-1, -1)` with bounds
`[-2, 2]`, and the quadratic in `popt/cost_functions`. Given the exact gradient and Hessian, the
optimizers must reach the minimum `(1, 1)` to `1e-4`, with and without the transform of the bounds.
Given only an ensemble of the controls (Gaussian or generalized), they estimate the gradient, and
the tolerance is looser; the hardest such case is a 100-dimensional Rosenbrock, from `-2`
everywhere, with 100 members.

### Hand-built cases

Many tests use no simulator at all: a fake ensemble or scheme, or frames built by hand, so that one
piece runs without the rest. These are local to their file; e.g. `test_compression.py` (seismic
vintages over a masked grid), `test_distance_loc.py` (a small grid with point observations), and
`test_scheme_base.py` (the loop with a fake ensemble).

## The methods

### Update schemes (`pipt`)

A scheme is an algorithm (ES, ES-MDA, …) and an *analysis*, the linear algebra of its update
(`approx`, `full`, `subspace`, …). `R` = pinned by the characterisation reference,
`P` = property test at 1000 members, `M` = MiniRes benchmark, `L` = matches the exact posterior in
the linear sanity checks, `L✗` = a known deviation there in some or all dimensions.

| Scheme | `approx` | `full` | `subspace` | `subspace2` | `margis` | `enif` | `hybrid` |
|---|---|---|---|---|---|---|---|
| ES | R M L | R L | L ¹ | | | | |
| EnKF | R L | L | L ¹ | | | | |
| ES-MDA | R P M L | R M L✗ | R M L✗ | R M L✗ | | M L✗ ² | ³ |
| LM-EnRML | R P M ⁴ | R L ⁵ | R L✗ | R L | | | |
| GN-EnRML | R P M ⁴ | R L | R L | R M L | R M L | | |

1. Crashes on the Van der Pol case, so it is not in the characterisation reference.
2. Exact with 1 and 2 unknowns; about 0.2 sd off the mean with 10, at 100 members. Also
   checked against its exact posterior in `test_enif.py`.
3. The multilevel scheme (`test_multilevel.py`): runs, keeps its levels, and lowers the misfit;
   no reference numbers.
4. Not in the linear sanity checks: `approx` drops the prior increment term, so the iterative
   schemes converge to a different point by design.
5. Also pinned on the linear 1D field (`test_linear_model.py`).

Further:

- `subspace2` is checked against an identity: it is `margis` with the error scale fixed at 1
  (`test_subspace2.py`).
- `co_lm_enrml` and `gn_enrml` are old names for LM-EnRML/`approx` and GN-EnRML/`subspace`, and must
  give exactly their numbers (`test_legacy_scheme_names.py`).
- Localization: adaptive and distance-based localization are unit-tested
  (`test_autoadaloc.py`, `test_distance_loc.py`) and run end to end only in the MiniRes benchmark.

### Optimizers (`popt`)

| Optimizer | Case | File |
|---|---|---|
| EnOpt | quadratic, from a Gaussian ensemble | `test_ensemble_optimization.py` |
| LineSearch: GD, BFGS, Newton-CG | Rosenbrock, exact derivatives; quadratic and 100-D Rosenbrock, ensemble gradient | `test_line_search.py`, `test_ensemble_optimization.py` |
| TrustRegion: iterative, CG-Steihaug, BFGS | Rosenbrock, exact derivatives | `test_trust_region.py` |
| GenOpt, with and without CMA | Rosenbrock, generalized ensemble | `test_genopt.py` |
| EPF (constraints) | the outer loop's convergence | `test_epf_convergence.py` |

The ensemble gradients and Hessians themselves are checked in `test_ensembles.py`, robust
optimization over several models in `test_robust_evaluation.py`, and a crashed simulation in
`test_crashed_evaluation.py`.

## Not covered

- The GIES schemes (`pipt/update_schemes/gies`) and SmcOpt (`popt/optimization_methods/smcopt.py`).
