# Ensemble information filter (EnIF)

PET offers the EnIF analysis as a flavour of ES-MDA. It uses the sparse
regression and precision estimation from ERT's
[`_enif_update.py`](https://github.com/equinor/ert/blob/main/src/ert/analysis/_enif_update.py)
through `graphite-maps`.

## Installation

Install PET from your checkout, including EnIF and its dependencies:

```sh
python -m pip install -e .
```

PET requires Python 3.12 through 3.14, matching its `graphite-maps` dependency.

## Select the analysis

Keep your existing ensemble, observation and simulator settings. EnIF is an
analysis flavour of ES-MDA, so select it in the `dataassim` section:

```yaml
scheme: esmda
analysis: enif
mda:
  tot_assim_steps: 3
  inflation_param: [2, 4, 4]
```

The original, single-update EnIF is the one-step schedule:

```yaml
scheme: esmda
analysis: enif
mda:
  tot_assim_steps: 1
```

The equivalent Python entry point is `ESMDA(keys_da, keys_en, sim,
analysis="enif")`, and `("esmda", "enif")` resolves through the scheme
registry like any other combination.

EnIF-MDA reruns the simulator and refits the response regression after each
update. It estimates the graph-based prior precision on the first pass, then
passes the accumulated posterior precision to the next pass, converting it to
that pass's standardized state coordinates. MDA requires positive, finite
inflation factors satisfying `sum(1 / alpha) = 1`. If you omit
`inflation_param`, PET uses `tot_assim_steps` for each factor. A scalar factor
repeats across the schedule.
Checkpoints retain both the schedule position and the accumulated precision;
restarting a multi-pass EnIF run requires a checkpoint with that information.

## Parameter graphs

EnIF estimates a separate prior precision block for each state in `idX`.
By default:

- A state with `grid` metadata in `prior_<state>` uses nearest-neighbour
  connectivity. PET's prior parser converts `grid` to `nx`, `ny` and `nz`.
- A state without grid metadata uses independent graph nodes.

The regular-grid ordering matches PET's layered prior generator:
`row = z * nx * ny + x * ny + y` (y varies fastest). For imported ensembles
with a different ordering, reduced active-cell arrays or irregular geometry,
provide a graph whose node numbers match the imported parameter rows.

You can configure graphs and the initial precision-fitting neighbourhood under
`enif`:

```yaml
enif:
  parameter_graphs:
    perm: perm_graph.npz
  neighbourhood_expansion: 2
```

`neighbor_propagation_order` is accepted in existing configurations but is
ignored by EnIF-MDA: every retained state row is solved for at each pass so
the carried information is reflected in the ensemble.

Write a graph file as a symmetric sparse adjacency array with
`scipy.sparse.save_npz`. For example, for five parameters arranged in a chain:

```python
import networkx as nx
from scipy import sparse

graph = nx.path_graph(5)
sparse.save_npz('perm_graph.npz', nx.to_scipy_sparse_array(graph, format='csc'))
```

Python configurations can also supply NetworkX graphs or SciPy sparse
adjacency arrays directly in `parameter_graphs`. Use local node numbers
`0` through `number_of_parameter_rows - 1` for each state. Graph weights do
not affect the fit; EnIF uses connectivity.

EnIF excludes rows containing non-finite values and rows with zero ensemble
spread from estimation. It removes their graph nodes without connecting
neighbours across the resulting gaps. The analysis gives those rows a zero
increment, then applies PET's configured state limits.

## Update and diagnostics

EnIF uses PET's perturbed observations, random-number stream, state clipping,
forecast loop and misfit reporting. It scales observation covariance by the
current MDA factor once. It estimates the unexplained response variance and
inflates the **total** noisy-residual variance: for step factor `alpha`, this
is `alpha * (observation variance + unexplained variance)`. PET's existing
observation perturbations already include the inflated measurement error; an
independent draw supplies the remaining `(alpha - 1) * unexplained variance`.
For a correlated observation covariance, PET whitens the observations,
forecasts and perturbations before fitting the response map and drawing this
additional noise in whitened coordinates.

The analysis lives in `pipt/update_schemes/analysis/enif.py` and binds to the
ES-MDA scheme like the `approx`, `full` and `subspace` flavours; it returns an
additive state-space step and uses the direct sparse solver, matching ERT's
non-iterative transport setting.

After an update, the bound analysis object (`scheme.analysis`) exposes the
fitted `H`, `Prec_u`, `Prec_eps` and `Prec_posterior`. `Prec_u` is the initial
graph-based fit on the first pass and the carried, rescaled posterior precision
on later passes. These matrices use standardized, retained state rows;
`enif_active_rows` maps them back to the full state. The scheme stores the
information needed by the next pass in `scheme.enif_information`, including
the posterior precision, scaling and
retained rows. Changes to the retained rows across passes are rejected rather
than silently discarding accumulated information. With correlated observation
errors, `H` and `Prec_eps` use whitened observation coordinates.

This analysis requires at least two ensemble members and positive observation
variances. It does not support PET's covariance localization, local analysis,
multilevel ensembles or `emp_cov` sample input. Use parameter graphs to specify
spatial dependence.
