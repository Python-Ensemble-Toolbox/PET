# Ensemble information filter (EnIF)

PET provides the original, single-update EnIF and an EnIF-MDA variant. Both
use the sparse regression and precision estimation from ERT's
[`_enif_update.py`](https://github.com/equinor/ert/blob/main/src/ert/analysis/_enif_update.py)
through `graphite-maps`.

## Installation

Install PET from your checkout, including EnIF and its dependencies:

```sh
python -m pip install -e .
```

PET requires Python 3.12 through 3.14, matching its `graphite-maps` dependency.

## Select the analysis

Keep your existing ensemble, observation and simulator settings. For one EnIF
update, set these entries in `dataassim`:

```yaml
daalg: [enif, enif]
analysis: full
```

For EnIF-MDA, use:

```yaml
daalg: [enif, enif]
analysis: mda
mda:
  tot_assim_steps: 3
  inflation_param: [2, 4, 4]
```

The corresponding classes are `enif_full` and `enif_mda` in
`pipt.update_schemes.enif`. PET's `pipt_init.init_da` loads them through the
existing configuration interface.

Both variants assimilate all selected `assimindex` entries together. Standard
EnIF performs one update with inflation 1; it does not need `mda` settings.
EnIF-MDA reruns the simulator and refits the regression and state precision
after each update.

MDA requires positive, finite inflation factors satisfying
`sum(1 / alpha) = 1`. If you omit `inflation_param`, PET uses
`tot_assim_steps` for each factor. A scalar factor repeats across the schedule.
The schedule retains its original indexing on restart.

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

You can configure graphs and neighbourhood sizes under `enif`:

```yaml
enif:
  parameter_graphs:
    perm: perm_graph.npz
  neighbourhood_expansion: 2
  neighbor_propagation_order: 15
```

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
current MDA factor once. It also estimates the unexplained response variance,
as in ERT. For a correlated observation covariance, it whitens the observations,
forecasts and perturbations before fitting the response map.

The EnIF-specific update and helpers live in `pipt/update_schemes/enif.py`.
The scheme inherits PET's `esmdaMixIn` lifecycle and returns an additive
`step`, following the existing update-method interface. It uses the direct
sparse solver, matching ERT's non-iterative transport setting.

For `analysisdebug`, the scheme exposes `H`, `Prec_u`, `Prec_eps` and
`Prec_posterior`. These matrices use standardized, retained state rows;
`enif_active_rows` maps them back to the full state. With correlated observation
errors, `H` and `Prec_eps` use whitened observation coordinates.

This scheme requires at least two ensemble members and positive observation
variances. It does not support PET's covariance localization, local analysis,
multilevel ensembles or `emp_cov` sample input. Use parameter graphs to specify
spatial dependence.
