# Architecture

PET is three layers. `ensemble` is the foundation: the base ensemble that runs
a forward simulator over the members, the checkpoint mixin, the loggers, the
`ForwardSimulator` protocol. `pipt` (data assimilation) and `popt`
(optimisation) build on it and never import each other; `ensemble` imports
neither. `misc` holds the data structures and the observed-data reader,
`input_output` the configuration boundary, `simulator` the analytical models
and the wrappers around external simulators.

## A run

1. **Configuration.** `input_output.read_config.read(file)` returns the
   problem, simulator and ensemble sections as plain dictionaries in one
   canonical form (`input_output.config.normalize`); `validate` says what a run
   would fail on. A dictionary built in a script gets the same treatment when
   the ensemble is constructed. See the [configuration reference](configuration.md).
2. **Ensemble.** `pipt.ensembles.AssimilationEnsemble(keys_da, keys_en, sim)`
   draws or loads the prior (`(nx, ne)` array; its variable rows are a
   `StateLayout`), reads the observations, fixes the **data layout** and
   builds the observation vector and variance in that order, and sets up
   localization and compression. It owns the random stream (`seed`).
3. **Scheme.** `ESMDA(keys_da, keys_en, sim, analysis="approx")` and the
   other schemes build the ensemble (or take one passed as `ensemble=`), bind
   an analysis object, and `run_assimilation()`: forecast the prior, then call
   `update_step()` until a criterion fires or `maxiter` updates are done.
   `ESMDA.assimilate(...)` is the one-line form.
4. **Result.** An `AssimilationResult` (`x`, `data_misfit`,
   `prior_data_misfit`, `nit`, `success`, `message`, `why_stop`), a
   `dict` subclass with attribute access like SciPy's `OptimizeResult`.

## The scheme contract

`pipt.update_schemes.core.AssimilationScheme` owns the loop, the convergence
checks (`misfit_tol`, `step_tol`, plus the scheme's own `check_convergence`),
the run table, restart, QA/QC and saving. A scheme supplies:

- `update_step() -> StepReport(accepted, state, misfit, why_stop)`: one
  iteration, retries included (LM-EnRML re-damps inside it). The loop commits
  the returned state and misfit; a scheme never assigns them itself.
- `score()`: the per-member data misfit of the current forecast.
- `log_columns()`: its columns of the run table.
- Hooks: `after_prior_forecast`, `after_analysis`, `after_forecast`,
  `after_accepted_iteration`, `after_loop`.
- `COMPATIBLE_ANALYSES`: the analysis flavours it accepts.
- `RESTART_ATTRIBUTES`: the attributes a checkpoint must carry for it.

Registration is one line: `register_scheme(name, analysis, cls)` in
`pipt.update_schemes.registry` (`available_schemes()` lists every pair). The
iterative family shares `IterativeEnRML`, where LM-EnRML and GN-EnRML differ
only in ten small hooks around their control parameter.

## Analyses

An analysis (`pipt.update_schemes.analysis`) is a class with
`update(enX, enY, enE, **kwargs) -> AnalysisResult`, returning exactly one of
a state-space `step`, a weight-space `w_step` or a `W_step`. The scheme turns
it into a proposal with `propose_state(result, step_scale)`. The flavours are
`approx`, `full`, `subspace`, `margis` and the multilevel `hybrid`;
`register_analysis` adds one. Analyses read what they need from the scheme:
`state_scaling`, `scale_data`, `proj`, `cov_data`, `trunc_energy`, `lam`.

## Data on the analysis path

Everything the analyses see is a matrix in one fixed row order:

- `DataLayout` (`misc.structures`): the order of the data vector, computed
  once from the observed frame -- label-major, then data type, empty cells
  skipped. The ensemble exposes `obs_vector` and `obs_variance` built from it.
- `PredictedData`: the `(nd, ne)` forecast filled directly from each member's
  simulator output through the layout, scaled as the observations were, with
  compressed vintages reduced on the way in. `pred_data.matrix` is what the
  schemes read; `pred_data.to_frame()` is the frame view.
- Adjoints: an `(nd, nx, ne)` array in the same rows, when the simulator
  computes them.
- `StateLayout`: the `{variable: (start, stop)}` rows of the state array; the
  ensemble's `state_layout` converts to and from dictionaries, builds member
  inputs for the simulator, and clips to the prior's limits.

`PETDataFrame` remains as the table observed data arrive in and results are
saved as; it is built from the matrices on demand, never on the analysis path.

## Forecast

`BaseEnsemble.calc_prediction(enX)` runs one level: member inputs
(`_simulator_input`), a backend (`_run_members`: in sequence, a local process
pool, or the wrapper's HPC queue), crash replacement, adjoint splitting, and
the raw outputs kept as `member_outputs`. `ForecastMixin.forecast` then fills
`pred_data`, corrects multilevel levels, applies the `scale` option and saves
what was asked for. Outlier replacement (`OutlierMixin`) reorders the raw
outputs and the state together.

A simulator is anything satisfying `ensemble.protocols.ForwardSimulator`: an
`input_dict` and `run_fwd_sim(state, member_index)` returning one dict per
report point (or a DataFrame), `False` on failure, or `(output, adjoint)`.
`simulator/vanderpol.py` is the smallest complete example.

## Restart, random numbers, logging

- One checkpoint per run, on `RestartMixin` (`ensemble.checkpoint`): the
  loop's bookkeeping, the scheme's `RESTART_ATTRIBUTES`, and the ensemble's
  state, prior, forecast, scaling and random stream, so a resumed run
  continues the interrupted one exactly. Driven by `restart`, `restartsave`,
  `restart_file`.
- Every draw comes from `ensemble.rng`: a private `RandomState` when the
  config gives a `seed`, otherwise NumPy's global stream as before.
  `misc.sampling.gen_real` is the Gaussian sampler.
- One named logger per log file (`ensemble.logger.PetLogger`); `NullLogger`
  stands in when logging is off.

## popt

`popt.optimization_methods.OptimizerBase` has the same shape as the scheme
base: it owns the loop, the starting evaluation, the callback, result
recording, the log and the function, state and projected-gradient checks. An
optimizer implements `update_step()`, committing an improving point with
`_commit_step(x, f, jac=..., hess=...)` and returning a `StepReport(accepted,
message)`, and `log_columns()`. `EnOpt`, `LineSearch`, `TrustRegion` and
`SmcOpt` are exported from `popt`, as are the ensembles that estimate
gradients and Hessians (`GaussianEnsemble`, `GeneralizedEnsemble`).

## Where a new piece goes

| Adding | Do |
| --- | --- |
| a scheme | subclass `AssimilationScheme` (or `IterativeEnRML`), implement `update_step`/`score`/`log_columns`, declare `COMPATIBLE_ANALYSES`, `register_scheme` |
| an analysis | subclass `AnalysisBase`, implement `update` returning `AnalysisResult`, `register_analysis` |
| a localization | a builder taking `info` (and `rng`, `data`, ...), `register_localization` |
| an optimizer | subclass `OptimizerBase`, implement `update_step` with `_commit_step`, `log_columns` |
| a simulator | a class with `input_dict` and `run_fwd_sim`; see the protocol's docstring for the optional hooks |
| a config key | read it from the normalised section; add it to `KNOWN_DATAASSIM`/`KNOWN_ENSEMBLE` in `input_output.config` and to the [configuration reference](configuration.md) |

The two notebooks under *Extending PIPT* in the tutorials walk through the
first two.
