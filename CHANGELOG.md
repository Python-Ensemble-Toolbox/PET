# Changelog

All notable changes to PET are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking changes
- **`conv_crit` in the `epf` section means a penalty magnitude, not a state change.** The outer EPF loop used to stop once no control moved more than `conv_crit` relative to its previous value; it now stops once `mean(epf['penalty']) / epf['r']` falls below it. The old test asked the wrong question: it reported success whenever the inner optimizer stalled, however badly the constraints were still violated, and refused to finish while a single control kept jittering. The default is still `1e-5`, so a config written for the old criterion loads unchanged, but the number now carries the units of the objective rather than being dimensionless — check it against your penalty's scale. The objective must write `penalty` into the `epf` dict it is handed; one that does not now raises `KeyError` instead of silently converging on the step size. Ported from upstream 5358e07 and 4b8d878.
- `max_iter` in the `iteration` section is the number of update iterations. It used to count the prior forecast as iteration 0, so `max_iter: 5` performed four updates; the same config now performs five. To keep an existing run as it was, lower `max_iter` by one. The run table, the convergence message and the `assimilation_result_{i}` files already numbered updates from 1 with the prior as 0, and are unchanged.
- `PETStateArray` is gone. The state ensemble is a plain `(nx, ne)` NumPy array; its variable layout is the ensemble's `idX` dictionary, wrapped by `misc.structures.StateLayout` (`ensemble.state_layout`), which owns what the subclass carried: `to_dict(enX)`, `member_dicts(enX)` (was `to_list_of_dicts`), `clip(enX, limits)` (was `clip_matrix`), and the constructors `StateLayout.from_dict(...)` and `StateLayout.from_prior_info(...)`, both returning `(matrix, layout)`. The subclass copied the row map onto every slice and view, so a five-row slice still claimed the full layout, and lost it on unpickling; twenty operator overrides existed only so a type checker inferred the subclass. Code that did `enX.to_dict()` or `enX.indices` now goes through the layout.
- Restart is one mechanism: the scheme's checkpoint (`RestartMixin`), driven by `restart`, `restartsave` and `restart_file` in the `[dataassim]` block and written to `<scheme>_restart.pkl` (default) after the prior forecast and every accepted iteration. The ensemble no longer loads `emergency_dump` when `restart` is set; that file is written only when every realisation of a forecast fails, for inspection. A resumed run continues the interrupted one exactly: the checkpoint carries the loop's bookkeeping, the scheme's declared state (`RESTART_ATTRIBUTES`: perturbed observations, damping, the subspace `W`), and the ensemble's state, prior, forecast, scaling and random stream, so it does not depend on the random state of the resuming process. Before this, the keys never reached the scheme (every scheme passed only zero tolerances to its base), so `restartsave` pickled the ensemble and a `restart` run re-initialised the scheme from scratch.
- `EnOpt` and `SmcOpt` constructors take `(x0, fun, ...)` like `LineSearch`, `TrustRegion` and every `minimize`; they took `(fun, x, ...)`. Callers using the keyword `x=` write `x0=`.
- `OptimizerBase.update_step()` returns a `StepReport(accepted, message)` instead of a bool and commits its point through `_commit_step(x, f, jac=..., hess=...)`; the base then runs the callback, records and saves the result, logs a row (from `log_columns()`) and checks convergence. Custom optimizers built on the old contract need those four changes.
- `SmcOpt` no longer runs the optimization inside its constructor (the `autorun` option is gone); call `run_optimization()` or use `SmcOpt.minimize(...)`, which has not changed.

- **Config: `daalg` is replaced by `scheme`.** The analysis flavour is a
  parameter of an algorithm rather than a separate algorithm, so the
  two-element `daalg` key has nothing left to encode. Only its second entry
  ever selected the class; the first was a module hint the registry no longer
  needs.

  ```toml
  # before                          # after
  [dataassim]                       [dataassim]
  daalg = ["esmda", "esmda"]        scheme = "esmda"
  analysis = "approx"               analysis = "approx"
  ```

  Loading a config that still uses `daalg` raises an error showing the rewrite
  and naming the migration command. To migrate:

  ```sh
  pet migrate my_config.toml          # rewrites in place, keeps <config>.bak
  pet migrate my_config.toml --dry-run
  pet convert my_case.pipt && pet migrate my_case.toml   # legacy text configs
  ```

  Existing `.pipt`/`.popt` files are unaffected until converted.

- **`pipt.loop.assimilation.Assimilate` is removed, with no shim.** Schemes own
  their iteration loop now, as popt's optimizers do. The whole `pipt.loop`
  package is gone, including the `pipt.loop.ensemble` compatibility shim.

  ```python
  # before                                  # after
  from pipt.loop.assimilation import Assimilate
  scheme = pipt_init.init_da(kd, ke, sim)   scheme = ESMDA(kd, ke, sim)
  Assimilate(scheme).run()                  result = scheme.run_assimilation()
  ```

  `pipt_init.init_da(...)` still works and still returns the scheme; only the
  driver changed. `Scheme.assimilate(kd, ke, sim)` is the one-line form.

- **`optimization_loop()` and `assimilation_loop()` are renamed** to
  `run_optimization()` and `run_assimilation()`, with no aliases. `_loop` named
  the mechanism rather than the job — nobody calls it because they want a loop
  — and `assimilation_loop` sat awkwardly beside the `run_forecast` /
  `run_prior_forecast` already on the same class. The rename affects both
  packages so they keep the same shape.

  ```python
  # before                       # after
  enopt.optimization_loop()      enopt.run_optimization()
  esmda.assimilation_loop()      esmda.run_assimilation()
  ```

  The class-level shortcuts are unchanged: `EnOpt.minimize(...)` and
  `ESMDA.assimilate(...)` still construct and run in one call.

- **Per-iteration result files renamed.** `debug_analysis_step_{i}.npz` is now
  `assimilation_result_{i}.npz`, the assimilation counterpart of popt's
  `optimize_result_{i}.npz`. The files were never a debugging aid — they are
  the record of a run, one per iteration, with iteration 0 the prior — and the
  old name said otherwise. **Post-processing that globs
  `debug_analysis_step_*` must be updated**; nothing can alias a filename.

  The config key that selects them follows: `analysisdebug` is now `savedata`,
  again matching popt. The old spelling still works and warns, and
  `pet migrate` rewrites it in place alongside `daalg`. There is no `saveit`
  switch to go with it: listing variables turns saving on and omitting the key
  turns it off, so a config cannot name variables that are silently discarded.

  ```toml
  # before                                   # after
  [dataassim]                                [dataassim]
  analysisdebug = ["state", "pred_data"]     savedata = ["state", "pred_data"]
  ```

  `analysis_tools.save_analysisdebug` is likewise deprecated in favour of
  `save_assimilation_result`; the alias writes the new filename, not the old
  one.

- **Eighteen scheme classes collapsed into five, and the per-flavour names
  removed.** `ESMDA`, `EnKF`, `ES`, `LMEnRML` and `GNEnRML` are classes taking
  `analysis` as an argument, and replace both the factory functions of the
  same names and the per-flavour classes (`esmda_approx`, `lmenrml_full`,
  ...): each was one line pinning a flavour the constructor argument already
  expresses. Use `ESMDA(..., analysis="approx")` and friends instead --
  `registry.get_scheme(scheme, analysis)` still resolves a `(scheme,
  analysis)` pair for config-driven code, now to the algorithm class with
  `analysis` pre-bound rather than to a stored class per combination.

  Two combinations are not aliases and keep their own classes: `esmda_hybrid`
  (multilevel ES-MDA) and `gnenrml_margis` (a private, externally-implemented
  strategy) are algorithms in their own right that happen to share a name,
  reachable via `registry.get_scheme("esmda", "hybrid")` /
  `("gnenrml", "margis")`. `esmda_geo` is gone outright: its `__init__` took
  the wrong arguments and referenced an attribute the class never set, so it
  could not have been constructed successfully; nothing exercised it.

  Not source-compatible: the removed classes used to *inherit* their
  strategy, so `issubclass(esmda_approx, approx_update)` held. The replacement
  *holds* one instead. Behaviour and numbers are unchanged -- pinned by the
  characterisation suite -- only the type relationship goes.

  Each algorithm class now declares, right on the class, which flavours it
  supports and which class handles each -- `ESMDA.COMPATIBLE_ANALYSES = {
  "approx": approx_update, "full": full_update, "subspace": subspace_update}`
  -- so reading one scheme's source shows everything it supports, with no
  registry lookup needed to find out. `EnKF`/`ES` requesting `analysis="full"`
  used to resolve to the `approx` strategy only through the per-flavour
  classes; requesting it directly on `EnKF`/`ES` ran the (numerically
  identical, more expensive) `full` strategy. `EnKF.COMPATIBLE_ANALYSES`
  now points `"full"` at the same class as `"approx"`, which is what the
  removed classes' docstrings already claimed ("EnKF/ES take a single step,
  so full and approx coincide") but did not, in fact, apply to direct
  construction. `ES` inherits the dict unchanged, so the fact lives in one
  place and applies regardless of entry point.

  `register_strategy` (`pipt.update_schemes.analysis.registry`) no longer
  makes a newly registered flavour automatically selectable on an existing
  scheme -- each scheme's `COMPATIBLE_ANALYSES` is what a config's `analysis`
  key is actually checked against. Add the flavour to a scheme's dict
  directly, or register a whole `(scheme, analysis)` combination via
  `pipt.update_schemes.registry.register_scheme`.

  `esmda_hybrid` (multilevel ES-MDA) moved off the mixed-in path onto this
  same bound-strategy pattern: `hybrid_update` now inherits `AnalysisStrategy`
  and `esmda_hybrid.COMPATIBLE_ANALYSES = {"hybrid": hybrid_update}`, in place
  of `class esmda_hybrid(hybrid_update, ESMDA)`. Its calling convention
  (`update(enX, enY, enE, **kwargs)`) already matched the bound shape; only
  the values are lists of per-level matrices rather than single ones, which
  the attribute-forwarding that binding relies on does not care about. One
  consequence: `esmda_hybrid.COMPATIBLE_ANALYSES` deliberately does *not*
  include `approx`/`full`/`subspace` -- those strategies expect a single
  `enX`/`proj` matrix, which this scheme's per-level state never gives them;
  requesting one now raises a clear error instead of the previous, unrelated
  behaviour of silently running the hybrid update regardless of what
  `analysis` was asked for. Verified bit-for-bit unchanged against the
  pre-conversion code (no committed reference existed to pin, so this was
  checked directly rather than through the characterisation suite).
  `gnenrml_margis` remains the one scheme still wired up the old way -- see
  below.

- **The config's `analysis` key is no longer overridden by a default.**
  `build_scheme`/`ESMDA(...)` took `analysis="approx"` as a parameter default
  and never consulted the config, so a config asking for `subspace` silently
  built the `approx` scheme through that entry point while `init_da` built the
  right one. Precedence is now explicit argument, then config, then `"approx"`.

- **Analysis strategies moved** from `pipt.update_schemes.update_methods_ns` to
  `pipt.update_schemes.analysis`, joining the base class and registry that
  already lived there. Modules are renamed to `approx`/`full`/`subspace`/
  `hybrid`/`margis`; the class names are unchanged.

  This affects code outside this repository: `enrml.py` walked
  `update_methods_ns` with `pkgutil` so a private namespace package could supply
  `margIS_update` alongside what shipped here. A private overlay must now
  target `pipt.update_schemes.analysis`, or the module below is used instead —
  silently.

  `analysis/margis.py` itself is no longer an inert placeholder: it now
  carries a real port of the margIS math from an older layout, with attribute
  names (`self.ne`, `self.proj`, `self.lam`, `self.scale_data`) matching this
  codebase's current conventions, plus fixes against Stordal, Lorentzen &
  Fossum (2023), *Marginalized iterative ensemble smoothers for data
  assimilation*:

  - **`GNEnRML.calc_analysis` was missing a branch.** `margIS_update`
    delivers its result via `self.W_step` (capital W) -- the ensemble
    *matrix* update ("following e.g. Raanes et al. 2019" in the code this
    was ported from), reconstructed as
    `enX = mean(prior_enX) + prior_enX @ proj * sqrt(ne-1) @ W`. Only the
    lowercase `w_step` *vector* update ("following e.g. Evensen et al. 2019",
    a different reconstruction for a differently-initialised `W`) had
    survived in this codebase's `GNEnRML.calc_analysis`. The first attempt
    at a fix renamed `self.W_step` to `self.w_step` to match what existed --
    which was wrong, and confirmed wrong by running it: routed through the
    vector-update branch, the assimilation made the misfit *worse* by five
    orders of magnitude, unchanged however small the step length shrank --
    the signature of the wrong formula entirely, not a scale problem. Fixed
    properly by restoring the missing `hasattr(self, 'W_step')` branch to
    `GNEnRML.calc_analysis`, gamma-scaled to match the existing `w_step`
    branch's convention, and reverting this file to deliver `self.W_step` as
    it always did.
  - **The first-call check used the wrong iteration convention.** `if
    self.iteration == 1` guarded initialising `current_W`/`current_w`/`D`.
    This codebase's schemes count from `self.iteration = 0` (confirmed
    against `GNEnRML.__init__` and against `subspace_update`, which checks
    `if self.iteration == 0` for the same reason), so initialisation never
    ran and the first real call failed outright with `AttributeError:
    'AssimilationEnsemble' object has no attribute 'current_W'`. Fixed to
    check `== 0`.
  - **The update loop was hardcoded to 70 individual data points**, each its
    own "type" of one (`M = 1`), instead of the paper's Eq. 8/9 sum over
    actual data types with each type's real count as `M`. Now groups rows by
    data type (`self.data_df`'s columns) instead.
  - **It carried its own `scale()`**, duplicating `AnalysisStrategy.solve` --
    the same duplication `approx`/`full`/`subspace` had before they were
    consolidated onto the shared base. Now inherits `AnalysisStrategy` and
    calls `self.solve` directly, picking up the same robustness fix
    consolidation made (`np.ndim` instead of `scaling.shape`, so a covariance
    passed as a plain list or scalar works).

  That inheritance change surfaced a fifth, pre-existing bug, unrelated to
  any of the above: the old `gnenrml_margis(GNEnRML, margIS_update)` listed
  `GNEnRML` first, so `StrategyMixin.update` -- reachable through `GNEnRML`'s
  own MRO chain -- was what plain attribute lookup actually found, not
  `margIS_update.update`, regardless of what `bind_strategy` decided about
  `self.strategy`. That `update` raises immediately for a mixed-in flavour,
  so the scheme could not run at all, independent of anything above.

  **`gnenrml_margis` is gone.** Once `margIS_update` took the same
  `(enX, enY, enE, **kwargs)` shape as every other strategy, mixing it into a
  separate class was no longer the only way to wire it up -- and, as the bug
  above shows, was actively worse than the alternative. `GNEnRML.
  COMPATIBLE_ANALYSES` now has a `"margis"` entry like `"approx"` and friends;
  `GNEnRML(..., analysis="margis")` binds `margIS_update` by ordinary
  composition, the same way `ESMDA(..., analysis="approx")` binds
  `approx_update`, with no MRO shadowing possible because binding never
  touches the class hierarchy. `("gnenrml", "margis")` resolves through the
  generic `ALGORITHMS` + `COMPATIBLE_ANALYSES` path now, not
  `SPECIAL_SCHEMES` -- unlike `("esmda", "hybrid")`, which stays special
  because `esmda_hybrid` really is a distinct class (multilevel ES-MDA), not
  an alias for an existing one.

  Run against real data for the first time (PIPT's own `TinyBox` tutorial
  case, 9 data types across 6 wells): misfit prior 1.96e10, after one
  iteration 1.18e8, a 99.4% reduction -- a large, sensible improvement, not
  just an absence of errors. Still not a golden reference, though: one run,
  one case, no committed values pinning today's numbers the way
  `test_numerical_characterisation` does for the other flavours -- see
  `pipt.update_schemes.analysis.margis`.

- **`iterinfo` hooks receive the scheme**, not the removed `Assimilate` object.
  Custom `main(self)` hooks reading loop attributes need adjusting.

- **Scheme machinery moved to `pipt.update_schemes.core`** —
  `AssimilationSchemeBase`, `AnalysisBindingMixin`, `AssimilationWorkflowMixin`
  — so `pipt.update_schemes` lists algorithms rather than mixing them with the
  scaffolding they stand on.

- **`log_update()` moved to the scheme base.** ES-MDA, LM-EnRML and GN-EnRML
  each carried a near-identical 14-line copy differing only in the trailing
  column -- `α`, `λ` and `γ` respectively. The base now builds the shared row
  and calls `log_columns()`, which a scheme overrides to add its control
  parameter:

  ```python
  def log_columns(self, prior_run: bool = False) -> dict:
      return {"λ": self.lam}
  ```

  The rendered table is unchanged, verified by capturing every row a run
  logs before and after.

- **`forecast()` takes the state to predict on.** It used to read
  `ensemble.enX_temp`, falling back to `enX` -- an ambient slot a scheme had
  to park its trial state in before calling, and clear afterwards. It is now
  `ensemble.forecast(enX)`, and `run_forecast(state) -> state` hands back the
  state actually used. `after_forecast(state) -> state` and
  `remove_outliers(state) -> state` follow suit: outlier replacement resamples
  members, so it returns the resampled state rather than writing it back to
  whichever slot happened to be set. `enX_temp` is gone from the
  scheme/ensemble contract entirely; it survives only inside the (already
  unimplemented) local-analysis path.

- **`update_step()` returns a `StepReport`, not a `bool`.** The loop needed
  four things back from a step but could only see one of them in the
  signature; the rest were attribute side effects a scheme could silently
  forget, leaving `data_misfit` as `None` and convergence permanently
  unreachable. A scheme now returns

  ```python
  StepReport(accepted=..., misfit=...)      # why_stop optional
  ```

  where `state` is the state the attempt produced and `misfit` is the
  *per-realisation* array. The loop commits `state` when `accepted` and
  clears the trial either way, so a scheme no longer has to remember
  `ensemble.enX = deepcopy(enX_temp); ensemble.enX_temp = None` — forgetting
  that gave a run which iterated and logged normally while returning the
  prior untouched. The loop derives
  `data_misfit` and `data_misfit_std` from it, so those three can no longer
  disagree — as they previously could after a rejected LM-EnRML step, which
  restored the scalar but left `ensemble_misfit` holding the rejected attempt.
  Schemes no longer set `step_accepted`, `data_misfit`, `data_misfit_std` or
  `ensemble_misfit` at all.

- **One name for the analysis concept.** The code called the same thing an
  "analysis" (the config key, `COMPATIBLE_ANALYSES`) and a "strategy" (the
  base class, the registry, the bound attribute). It is now "analysis"
  throughout:

  | before | after |
  | --- | --- |
  | `AnalysisStrategy` | `AnalysisBase` |
  | `StrategyMixin` | `AnalysisBindingMixin` |
  | `pipt.update_schemes.core.strategy` | `pipt.update_schemes.core.analysis_binding` |
  | `STRATEGIES` | `ANALYSES` |
  | `get_strategy` / `register_strategy` / `available_strategies` | `get_analysis` / `register_analysis` / `available_analyses` |
  | `bind_strategy()` | `bind_analysis()` |
  | `scheme.strategy` (object) + `scheme.analysis` (name) | `scheme.analysis` (object) + `scheme.analysis_name` (name) |

  Note the last row: `analysis` is both the constructor argument (a flavour
  *name*) and the attribute holding the resulting object, the way
  `Model(optimizer="adam").optimizer` is an optimizer instance.
  `pipt.localization` keeps its own, unrelated use of "strategy".

- **`score_prior()` is replaced by `score()`.** Every scheme spelled out its
  own prior-scoring hook: the misfit expression, then the same five
  assignments around it. The expression is now a `score()` method and the
  bookkeeping belongs to the base, which calls it through
  `record_prior_score()` before the loop. `run_prior_forecast()` went the same
  way -- it wrapped a single line that now sits in the loop that used it.

  ```python
  # before: once per scheme
  def score_prior(self):
      misfit = at.calc_objectivefun(
          self.enObs, self.pred_data.to_matrix(), self.cov_data)
      self.ensemble_misfit = misfit
      self.data_misfit_mean = np.mean(misfit)
      self.prior_data_misfit_mean = np.mean(misfit)
      self.data_misfit_std = np.std(misfit)

  # after: the expression only, and only when it differs from the default
  def score(self, pred_data=None):
      pred = self.pred_data if pred_data is None else pred_data
      return at.calc_objectivefun(
          self.enObs_conv, self._as_matrix(pred), self.cov_data)
  ```

  `score()` is called for the prior *and* for every attempt inside a step, so
  a scheme has one definition of its own misfit instead of two copies that
  could drift. The base implements the default — perturbed observations
  `enObs` against `cov_data` — so a scheme that binds those needs no override
  at all; ES-MDA overrides it to score against its un-inflated `enObs_conv`,
  and the EnKF family to use `scale_data`. A scheme with nothing to score
  returns `None` and the base leaves its misfit bookkeeping alone.

  Prior scoring now also logs its row through `log_update(prior_run=True)` for
  every scheme, so the EnKF and ES print an iteration-0 row in the run table
  where they previously printed a one-line info message.

- **The damping loop moved into `update_step()`.** LM-EnRML and GN-EnRML used
  to iterate λ and γ through the *base* loop: reject the step, return
  `accepted=False`, and let `run_assimilation` retry at the same iteration
  number. The retry now happens inside `update_step()`, so one call is one
  iteration however many attempts it takes — the shape popt's optimizers
  already had, where `EnOpt.update_step` backtracks over its own step length
  before returning.

  The sequence of analyses, forecasts and λ updates is unchanged, and the
  numerical characterisation tests confirm the schemes produce identical
  numbers. What changes is where the loop lives, and how a scheme that cannot
  improve gives up: both schemes take a new `max_inner_iter` option
  (default 10) in the `iteration` block and stop with `why_stop['inner_stop']`
  when they exhaust it. GN-EnRML has no `gamma_min`, so previously it kept
  shortening its step until the base loop's `max_rejected` valve fired after
  `10 * max_iter` attempts; it now gives up after 10 consecutive failures.

  With the retries inside the step, the base loop no longer counts rejections:
  `max_rejected` and its "stopped after N consecutive rejected steps" ending
  are gone. A report coming back `accepted=False` now means the scheme has
  exhausted its own attempts, so the run stops — asking again would only
  repeat the step it just said it could not improve on. Convergence is still
  checked on that final report, so a scheme that rejects *and* converges (LM-
  EnRML reaching `lambda_max`) is still reported as converged.

- **The run table is logged by the loop.** `log_update()` was called from
  inside each scheme's `score_and_commit`, twelve times across five schemes,
  once per *attempt*. `run_assimilation` now logs one row per accepted
  iteration, and the schemes do not log at all:

  ```python
  # scheme_base.run_assimilation()
  if self.step_accepted:
      self.log_update(success=True)
      self.iteration += 1
      self.after_accepted_iteration()
  ```

  Rejected attempts no longer produce a `Failed` row — with the damping loop
  inside `update_step`, those attempts are the scheme's business. The EnKF and
  ES, which never called `log_update` at all, now get rows like every other
  scheme. `log_columns()` is unchanged and remains how a scheme adds its
  control parameter; LM-EnRML and GN-EnRML report the λ and γ the logged
  iteration actually ran with, since by the time the loop logs, the scheme has
  already adjusted them for the next one.

- **One class: `AssimilationScheme`.** Schemes used to inherit a combination
  of `AssimilationWorkflowMixin` and `AssimilationSchemeBase`, in that order
  and no other: the mixin *overrides* five hooks (`after_analysis`,
  `after_forecast`, `after_loop`, `after_accepted_iteration`,
  `after_prior_forecast`) that the base declared as no-op defaults, so listing
  it second silently stopped a run from saving anything.

  The split bought nothing — every shipped scheme wanted both halves — so the
  two are now one class named `AssimilationScheme`, and
  `pipt/update_schemes/core/workflow.py` is gone.

  ```python
  # before                                          # after
  from ...core.workflow import AssimilationScheme   from ...core import AssimilationScheme, StepReport
  from ...core.scheme_base import StepReport
  ```

  `AssimilationSchemeBase` and `AssimilationWorkflowMixin` no longer exist
  under any name. Anything subclassing the mixin on its own — a test double,
  say — should subclass `AssimilationScheme` and supply an ensemble stand-in,
  since `keys_da`, `save_folder` and friends are read-only views of the
  ensemble rather than attributes to assign.

  The ensemble collaborator protocol grew accordingly: a scheme now always
  carries the workflow, so its ensemble must also expose `keys_da`, `sim`
  (for `input_dict`) and `_saving_enabled`. The module docstring lists it.

- **LM-EnRML's damping factor is `lam_factor`, not `gamma`.** The config key
  is unchanged (`lambda_factor`); only the attribute is renamed. `gamma` named
  two different quantities in one file -- LM-EnRML's damping multiplier and
  GN-EnRML's step length -- which is a poor trap to leave beside two classes
  whose inner loops now read almost identically. A `savedata` entry or
  `iterinfo` script reading `gamma` off an LM-EnRML scheme should read
  `lam_factor` instead.

- **Empty hook declarations are gone.** The five no-op `after_*` stubs
  disappeared with the merge — the workflow bodies took their place — and
  `_get_restart_state()` / `_set_restart_state()` moved to
  `ensemble.checkpoint.RestartMixin` as defaults, so neither PIPT's schemes
  nor popt's `OptimizerBase` declare an empty pair to satisfy the protocol.
  Hosts that checkpoint their own state (`EnOpt`, `TrustRegion`, `LineSearch`,
  `SmcOpt`) override them exactly as before.

### Added
- Documentation: a configuration reference (`docs/configuration.md`) listing every key of the `dataassim`, `ensemble`, `optim` and `simulator` sections with meaning and default, and an architecture page (`docs/architecture.md`) describing the layers, the scheme, analysis and optimizer contracts, the data layouts, restart, random numbers, and where a new piece goes. Both are in the site navigation and linked from the README and the developer guide. `popt` exports its public API (`EnOpt`, `LineSearch`, `TrustRegion`, `SmcOpt`, `OptimizerBase`, `StepReport`, `GaussianEnsemble`, `GeneralizedEnsemble`). Every public function and class now has a docstring.
- `misc.structures.DataLayout`: the order of the data vector, derived once from the observed frame (label-major, then data type, empty cells skipped). The ensemble builds it after scaling and exposes `obs_vector`, which the schemes now use in place of `data_df.to_matrix()`; the frame remains as the view (`DataLayout.to_frame`). First step of replacing frame flattening on the analysis path.
- `seed` option in the ensemble config (`[ensemble] seed = 7` for pipt, `options['seed']` for popt). Every draw a run makes -- prior realisations, perturbed observations, outlier and crash replacement, the auto-adaptive localization's shuffle, popt's control perturbations -- now comes from the ensemble's `rng`: a private `numpy.random.RandomState(seed)` when a seed is given, so the run reproduces on its own and leaves NumPy's global state untouched; otherwise the global stream, exactly as before, so `np.random.seed(...)` before a run keeps working and every reference number is unchanged. The geostat sampler PET used for these draws is replicated draw for draw in `misc.sampling.gen_real`, which takes the stream as an argument; geostat remains a dependency for its covariance builder.

- **Every scheme takes a ready-made `ensemble=`.** The default collaborator
  is declared once, as `AssimilationScheme.ENSEMBLE_CLASS`, and built by
  `build_ensemble` only when none is handed in; multilevel ES-MDA keeps its
  override. Two schemes can share one prior and its forecasts, and a test can
  substitute a stand-in without the config, data files and simulator a real
  ensemble needs.
- **`pipt.localization.register_localization`.** Strategies are selected from
  the `LOCALIZATIONS` table by the config's `name` instead of an `if`/`elif`
  chain in the factory, so a new strategy is one registration call;
  `available_localizations()` lists them and an unknown name reports them.

- **`ensemble.protocols.ForwardSimulator`** writes down the simulator
  contract the base ensemble drives: `input_dict` and
  `run_fwd_sim(state, member_index)` are required, and the docstring lists
  the optional hooks (`setup_fwd_run`, `true_order`, `datatype`,
  `compute_adjoints`) and the four return shapes the ensemble accepts. It is
  a runtime-checkable `Protocol`, so `isinstance(sim, ForwardSimulator)`
  works, and a test holds every bundled simulator to it. Until now the
  contract could only be recovered by reading `calc_prediction`.

- **One constructor per algorithm**, with the flavour as an argument, so five
  names reach what previously took eighteen:

  ```python
  from pipt import ESMDA, available_schemes
  scheme = ESMDA(cfg_da, cfg_en, sim, analysis="approx")
  available_schemes()   # every valid (scheme, analysis) pair
  ```

- **`pipt.update_schemes.registry`** — an explicit scheme table replacing
  dispatch by string surgery. Unknown keys now report the valid alternatives
  instead of failing on a missing attribute. Third-party and private schemes
  can join via `register_scheme()`.

- **`AssimilationSchemeBase`** (`pipt.update_schemes.core`) — the PIPT
  counterpart to popt's `OptimizerBase`, with a matching contract
  (`update_step`/`run_assimilation`/`check_*_convergence`/`assimilate`). The
  ensemble is a collaborator rather than a superclass. Every scheme is now
  migrated onto it.

- **`AnalysisStrategy`** (`pipt.update_schemes.analysis`) — shared base for the
  approx/full/subspace flavours, the counterpart to popt's `subroutines`.

- **`pipt.localization`** — replaces the 888-line `cov_regularization` monolith
  with a package: an ABC and config builder, one module per strategy, and a
  factory dispatching on a `name` attribute.

- **`pet` command line**: `validate`, `convert`, `migrate`, `version`.

- **`ensemble.checkpoint.RestartMixin`** — checkpoint/restart logic shared by
  PIPT and POPT rather than duplicated.

### Fixed
- Flags written as strings were read by truthiness in several places, so `scale_data = "no"` in a config enabled scaling; every flag is a boolean after the boundary.
- A NaN data variance for an observed cell was silently dropped when the covariance was assembled, leaving it one entry shorter than the observation vector; it is now reported with the cell.
- The `scale` option of `[dataassim]` (multiply the predictions of named data types by a factor) never did anything: it iterated the characters of the column names. It now scales the named rows of the prediction matrix.
- ES-MDA's restart branch referenced an undefined `loop_ind`; the step to resume at now comes from the restored iteration counter.
- `LineSearch(recompute_jac=n)` crashed with `TypeError` on its first retry: the gradient was cleared but not recomputed before the next search direction.
- popt's `save_prediction` option raised `AttributeError`: the base ensemble read `self.ensemble.keys_da`, an attribute it never had. The folder now comes from the ensemble's own options (`savefolder` or `save_folder`, default `Predictions`) and is created before writing.

- **Six small crash and correctness fixes.** `OpenBlasSingleThread` (and the
  other environment context managers) called `os.environ.unsetenv`, which does
  not exist, so leaving the block raised whenever the variable had been unset
  beforehand; they use `os.environ.pop`. The `lin_1d` and `nonlin_onedimmodel`
  test simulators returned their shared output list, so in a serial forecast
  every member aliased the last one evaluated; they return a copy. The
  localization factory returned `None` for an unknown `name`, which then
  failed far away on `localization.name`; it raises with the valid names. The
  multilevel row batch could be zero for a single-row state. The outlier
  filter called `.ndim` on empty (`None`) cells; they are left alone. And the
  end-of-run summary said "Convergence was met." after every run, including
  those stopped by the iteration limit; it now says which.

- **popt: five verified bugs in the numerical subroutines.** Steihaug's
  boundary step divided only the square root by the squared direction length,
  so every step that hit the trust region had the wrong length. Adam, AdaMax
  and Steihaug did not take the backtracking factor, and EnOpt's `TypeError`
  fallback halved them on the pre-trial call with factor 1.0, before the
  first attempt of every iteration; every step rule now takes `shrink`.
  EnOpt's covariance step used `beta * cov` where `beta` is documented as
  momentum, shrinking the covariance by `1 - beta` on every accepted step
  whatever the gradient said; it is `beta * cov_step` now, like the state
  step (no effect at the default `beta = 0`). `LineSearch` passed its
  `lsmaxiter` option under a key the line searches never read, so the cap was
  always 10. Newton-CG fell off its loop without a `return` when it reached
  the iteration cap, handing `None` to a caller that took its norm. And
  `clip_state` tested `lb is None` on a whole array, defaulted the upper bound
  to `-inf`, and skipped clipping when every bound was 0.

- **Distance localization placed kernels with their axes swapped.** Kernels
  are built `(nx, ny)`-major like the field, but placement unpacked them as
  `(ky, kx)`. Square, symmetric kernels away from the edges came out right by
  coincidence; an anisotropic kernel raised a shape error everywhere, and an
  isotropic one raised near any grid edge where the x and y clipping differed.
  Placement now uses the kernel's own axes, with tests at the edges and for
  an anisotropic kernel. Existing results for interior, isotropic kernels are
  unchanged.

- **A crashed realisation no longer crashes the run.** The forecast handed
  `_replace_failed_simulations` the list of member inputs where it expected
  the state matrix, so the first failed member raised `AttributeError` on
  `.shape` instead of being replaced. It now receives the trial state, and a
  crashed member takes both the prediction and the state of the successful
  member drawn to replace it, so the two stay a matched pair.
- **The emergency dump could not pickle the ensemble** when the config asked
  for no localization: the stand-in was an instance of an anonymous class
  created with `type(...)`. It is now a module-level `NoLocalization` class,
  so `emergency_dump` and the restart file work on the runs that need them.

- **The `approx` and `full` analyses now apply the state scaling.** Both read
  a `scale_state` attribute that nothing ever set, so the per-row prior
  standard deviation the ensemble computes as `state_scaling` was silently
  replaced by ones. For `approx` this cancels exactly (anomalies are divided
  by it and the step multiplied back), except in the empirical-covariance
  branch of distance localization, whose gain matrix now returns to physical
  units like the other branches. For `full` it did not cancel: `Am` was built
  from the prior anomalies *multiplied* by the standard deviation while the
  anomalies and the prior misfit were left unscaled, so the regularisation
  term was off by the squared standard deviation for any variable whose prior
  standard deviation was not 1. `Am` is now built in the same scaled space as
  the rest of the update. A test checks that rescaling one variable's units
  rescales only its rows of the step. The goldens are unchanged: every prior
  variance in the characterisation case is 1.

- **EnKF and ES reported the misfit divided by sigma, not sigma squared.**
  `EnKF.score` passed `scale_data` -- the square root (or Cholesky factor) of
  the data covariance -- into the objective, which expects a variance. The
  override is gone and the family scores with `cov_data` like every other
  scheme, so its misfits are comparable with ES-MDA's and mean what the run
  table says. The posterior states are unchanged (neither scheme feeds the
  misfit back into its update); the ES and EnKF `data_misfit` and
  `prior_data_misfit` goldens were regenerated, and the regeneration step
  asserted that no state entry moved.

- **The `subspace` analysis now whitens the predicted anomalies before its
  SVD.** It took the SVD of `enY @ PI` while whitening only the residual and
  the observation perturbations, so the weight-space step depended on the
  units of the data: identical to the pre-refactor `gn_enrml` step when every
  data variance was 1, tens of percent apart otherwise. The omission dated
  from the strategy's first extraction. With `Y = self.solve(scy, enY @ PI)`
  the step matches that transcription to 1e-16 under every scaling tried, and
  a scale-invariance test pins it. The `esmda`, `lmenrml` and `gnenrml`
  `subspace` goldens were regenerated for this change; the ten other entries
  are unchanged.

- `check_state_convergence()` was inert: `enX_old` was initialised to `None`
  and never assigned, so it returned `False` for every scheme. It is the
  counterpart of a criterion that works on the popt side, where each optimizer
  assigns `xk_old` itself. `run_assimilation` now takes the snapshot centrally
  -- one site rather than the seven a per-scheme approach would need -- and
  only when `step_tol > 0`, since `enX` is `(nx, ne)` and a copy per attempt
  would cost memory for schemes that never use the criterion. Every shipped
  scheme still passes `step_tol=0.0`, so behaviour is unchanged; the criterion
  now works for anyone who opts in.

- `step_accepted` could disagree with what `update_step()` returned. Only the
  EnRML family maintained it, so for other schemes it stayed at its default of
  `True` regardless. The loop now syncs it from the return value. This matters
  because a rejected step leaves `enX` untouched: without an accurate flag,
  state convergence would read the resulting zero-norm as instant convergence
  on every rejection.

- A converged `LMEnRML`/`GNEnRML` run reported `no stopping reason recorded`.
  Both schemes set their converged flag in `score_and_commit()` but never set
  `conv_msg`, and they disable the base class's generic criteria -- which are
  the only other thing that sets it. `result.message` and the closing log line
  now name the criterion that fired (the data-misfit tolerance, or
  `lambda_max` for LM-EnRML).

- `LMEnRML`/`GNEnRML` re-armed a convergence criterion they had just
  disabled. Both pass `step_tol=0.0` to switch off the base class's generic
  state-change check, then set `self.step_tol` from config (default `0.01`) a
  few lines later. Neither reads the value itself — the only consumer is the
  check they opted out of. The assignment was vestigial, carried over from the
  never-constructed `co_lm_enrml`/`gn_enrml`, and is removed. No behaviour
  change today, because `check_state_convergence()` cannot fire at all (see
  Known issues).

- `hybrid_update` carried its own `scale()`, a duplicate of the inherited
  `AnalysisBase.solve()` with the arguments in the opposite order. Removed in
  favour of `solve`, which additionally accepts a covariance given as a plain
  list or scalar.

- **`savedata` could not record the prior.** Every scheme computed its
  prior misfit inside the first `calc_analysis`, which runs *after* the
  iteration-0 artifacts are written. So the step-0 file never
  contained `ensemble_misfit`, `data_misfit` or `prior_data_misfit`; the run
  printed `Cannot save ensemble_misfit, because it is a local variable!` and
  carried on. Prior scoring moved to a new `score_prior()` hook that the loop
  calls between the prior forecast and `after_prior_forecast`, so step 0 is
  described by the same attributes as every later step. Numbers are unchanged
  — the characterisation suite pins all nine scheme/flavour combinations.

  Two consequences beyond the saved files:

  - LM-EnRML and GN-EnRML no longer recompute `prior_data_misfit` from the
    *rejected* forecast each time they reject their first step. The old
    `iteration == 0` branch also re-clobbered `data_misfit` right after
    `score_and_commit` had restored it.
  - `ensemble_misfit` is now set by EnKF, ES and the multilevel hybrid too;
    only ES-MDA and the EnRML pair kept it before.

- **`save_folder` in a `dataassim` block was silently ignored.** Only the
  unspaced `savefolder` was read, so a config using the underscored spelling —
  which popt's optimizers accept — wrote to the default `Results` folder
  instead. Both spellings are now accepted.
- **ES discarded its own update.** The posterior came back bit-identical to the
  prior: the analysis ran, the forecast ran, the log reported a reduced misfit,
  but the state promotion sat inside an equal-misfit branch that is essentially
  never taken, so `enX_temp` was never committed. Anyone running ES was handed
  their prior ensemble back.
- **`enkf` could not run at all.** `check_convergence` read
  `self.full_cov_data`, which nothing assigns, so every run raised
  `AttributeError` at the end of its first iteration. Commit 6401e6e rewrote the
  two sibling call sites to use `scale_data` and missed this one.
- **The multilevel scheme had never completed a run.** Four faults: the level
  loop iterated ensemble *sizes* while using the value as an *index*;
  `treat_modeling_error` was called before `pred_data` existed;
  `calc_analysis` overwrote the step `hybrid_update` had just computed with the
  `None` it returns, discarding every update; and `esmda_hybrid` relied on C3
  linearisation to reach the scheme's `__init__`, which stopped happening when
  schemes left the ensemble hierarchy. It now runs end to end.
- `gies/rlmmac_update.py` imported `_calc_loc` from the removed
  `cov_regularization` module, so importing the GIES-RLMMAC scheme raised
  `ImportError`.
- `co_lm_enrml.calc_analysis` added the imported *function* `aug_state` to an
  ndarray — there is no local variable of that name — raising `TypeError` on
  every run.
- `approx_update.solve` used `A.ndim` where the other two flavours used
  `np.ndim(A)`, so a covariance supplied as a list or scalar raised
  `AttributeError` with that flavour only.
- `convert_txt_to_yaml` opened its output in binary mode while `yaml.dump`
  writes `str`, so every call raised `TypeError`.
- Two uses of `np.bool`, removed in modern NumPy.
- popt's line-search `zoom()` read `aold`/`phi_old` before binding them on the
  first branch.

### Changed
- One configuration boundary, `input_output.config`. Every reader (TOML, YAML, legacy `.pipt`/`.popt`) and every ensemble constructor now passes the sections through `normalize`: the canonical name where a key has had two spellings (`data` for `truedata`, `datavar` for `var`, `savefolder` for `save_folder`, `restart_file` for `restartfile`, `importstate` for `importstaticvar`), booleans for the yes/no flags, dictionaries for the sub-blocks the text format wrote as rows (`iteration`, `mda`, `compress`, `localization`, `multilevel`, `prior_*`), and the field conversions done once. The result is a copy: the ensemble no longer writes `datatype`, `truedataindex` and `assimindex` back into the caller's dictionary, and the helpers that rewrote `iteration`, `mda` and `compress` in place work on copies. Consumers read one name. `validate` replaces the assert-based mandatory-keyword checks: problems are reported by section and key, `pet validate` prints all of them plus keys nothing in PET reads (misspellings), and building an ensemble raises `ConfigError` listing the fatal ones instead of a `KeyError` inside the run. The legacy text reader returns three sections like the others (the third empty) and no longer asserts at read time. `is_enabled` and `list_to_dict` in `extract_tools` are the boundary's `as_flag` and `pairs_to_dict` under their old names.
- Seismic compression happens while the prediction matrix is filled: a compressed data type's raw vintage becomes its leading wavelet coefficients through the same `SparseRepresentation` that reduced the observed vintage, member by member, as the values enter `PredictedData`. The frame-based `post_process_forecast` rewrite is gone; `post_process_forecast` now only enables the `sim2seis` scaling (`scale_results.pkl`), and compression follows from `compress` alone -- a config with `compress` but without `post_process_forecast` used to leave predictions uncompressed against compressed observations. Reconstructions of compressed members are computed only when `saveforecast` will write them (`rec_results.pkl` unchanged). Checked against an AVO case: the reader reproduces a previous run's compressed observations and variances bit for bit (7376 and 7122 coefficients over two vintages), and real member vintages filled through the new path equal their direct compression.
- The ensemble builds `obs_variance` once from the layout (`(nd,)`, or `(nd, ne)` for an empirical error ensemble); the schemes, the observation perturbation and outlier detection read it. `construct_data_cov` is gone.
- The full forecast is kept as what the members returned (`member_outputs`); the `sim_data` frame is built from them when something asks for it -- saving, QA/QC, popt's objective -- and cached until the next forecast. Outlier replacement reorders the raw outputs instead of rewriting every frame cell. Adjoints are an `(nd, nx, ne)` array in layout order, scaled with the data, instead of a frame flattened on every analysis; the frame path stacked every row the simulator reported, not only the observed ones. Adjoint-based updates move at the 1e-13 level: the legacy stack was a non-contiguous array, so the member mean summed in a different order (values are identical; verified on the Van der Pol case).
- Predictions are a `PredictedData` container -- the `(nd, ne)` matrix in `DataLayout` order plus the layout -- filled directly from what each member's simulation returned, scaled as the observations were. The schemes read `pred_data.matrix`; nothing on the analysis path flattens a frame any more. `pred_data.to_frame()` is the frame view (QA/QC, inspection); `sim_data`, the full forecast, is still a frame and still what gets saved. Observations and predictions now share one row order by construction, so an unobserved cell can no longer leave the observation vector shorter than the prediction matrix. The multilevel model-error correction and outlier detection work on the matrices. In `savedata` files, `pred_data` is the matrix rather than a list of records. The seismic compression path (`post_process_forecast`) still runs on the frame and is wrapped into the container afterwards.
- `BaseEnsemble.calc_prediction` is orchestration over four named steps: `_simulator_input` (one dict per member), `_run_members` (the serial, HPC and process-pool backends), `_collect_adjoints` and `_collect_sim_data` (the output coercion and scaling). Same operations in the same order; the characterisation goldens are unchanged, and a new test pins the pooled backend against the serial one.
- `OptimizerBase` owns what the four optimizers each repeated: `minimize`, the starting evaluation (now at the start of `run_optimization()` rather than in the constructor, so an optimizer can be built without evaluating anything), the callback, result recording and saving, the iteration log, and the projected-gradient convergence check (`gtol`). `enopt.py`, `linesearch.py`, `trust_region.py` and `smcopt.py` lost about 500 lines between them. Results are unchanged: 21 deterministic cases across all optimizers, search directions and step rules give bit-identical `x`, `fun`, `nit`, `nfev`, `njev` and `nhev`.
- `LineSearch` results no longer carry `hess` after the first step: the Hessian on hand belonged to the previous iterate and was reported against the new `x`.
- The Steihaug step rule's diagnostic output (a dozen lines per CG iteration, printed unconditionally) is now emitted at `DEBUG` level on the `popt.optimization_methods.subroutines.optimizers` logger; the BFGS 'non-positive curvature' notice is a logging warning instead of a print.
- `restart_sim_results.pkl` (a saved forecast copied to that name so a restarted run can skip the forecast it had already finished) is now honoured only when `restart` is enabled; it used to be consumed by any run that found it in the working directory. Once used it is moved into the results folder as `sim_results.pkl`, where a saved forecast goes, instead of being renamed in the working directory.

- **Library code keeps to its own logger and raises instead of exiting.**
  `PetLogger` gives each log file its own named logger with its own file and
  console handlers; it used to call `logging.basicConfig`, which configures
  the root logger once per process and does nothing the second time, so a
  second logger (popt beside pipt, or a re-run in a notebook) kept writing
  into the first file and any application that had touched the root logger
  got no file at all. Records still propagate, so root handlers see them.
  Reading `save_folder` no longer creates the directory; it is created where
  something is written. The `sys.exit` calls in the ensemble (every member
  failed), the wavelet compression, the `sevenmountains` objective, popt's
  ensemble base and the Steihaug subroutine are exceptions now, and the
  remaining `print` calls beside a logger go through it. A `savedata` entry
  naming a variable the scheme does not have is a `UserWarning`. The
  Gaussian ensemble's `warnings.filterwarnings('ignore')`, which silenced
  warnings for the rest of the process, is scoped to the method that needed
  it. popt's EPF refresh saved its result to the working directory instead of
  `savefolder`.

- **LM-EnRML and GN-EnRML share one implementation.** `IterativeEnRML`
  holds the construction, the analysis call, the retry loop inside
  `update_step`, the scoring and the accept/reject bookkeeping the two
  schemes had as near-verbatim copies (about 400 lines); each subclass now
  supplies only how its control parameter reacts -- LM-EnRML's damping
  `lambda` (grows on rejection, stops at `lambda_max`) and GN-EnRML's step
  length `gamma` (scales the step, shrinks on rejection) -- through ten small
  hooks. A new iterative smoother with a different damping policy is those
  hooks and nothing else. Numbers, `why_stop` contents, stop messages and
  the run table are unchanged, pinned by the goldens for all six LM/GN pairs;
  the one visible difference is that LM-EnRML's "converged after an increase"
  log line no longer carries a leading space, since both schemes log it the
  same way now.

- **Analyses return their result instead of writing it onto the scheme.**
  `update()` now returns an `AnalysisResult` holding exactly one of `step`
  (state space), `w_step` (ensemble-weight space, `W_0 = 0`) or `W_step`
  (ensemble-transform space, `W_0 = I`), and the scheme base turns any of them
  into the trial state in one place, `propose_state(result, step_scale)`.
  Before, `subspace_update` and `margIS_update` assigned `scheme.w_step` /
  `scheme.W_step` and returned `None`, `hybrid_update` assigned
  `scheme.step`, and four copies of `calc_analysis` chose a reconstruction
  with `hasattr` chains -- attributes that were never cleared, so the branch
  taken depended on what an earlier flavour had left behind, and a single
  letter (`w_step` vs `W_step`) selected a different formula. A plain array
  is still accepted as a state-space step, so an analysis written the way
  the tutorial shows keeps working. Numbers are unchanged: the goldens for
  all thirteen scheme/analysis pairs pass untouched. The `approx` analysis
  now raises `NotImplementedError` for the `localanalysis` and
  `parallel_update` localizations instead of warning and returning nothing,
  which left the posterior equal to the prior.

- **QA/QC works again, on the current data structures.** `QAQC` was still
  written against the pre-refactor layout (lists of dicts for observations,
  variances and predictions), and the scheme handed it `ensemble.obs_data`
  and `ensemble.datavar`, which no longer exist, so any config with `qa` or
  `qc` failed at construction. It now takes the ensemble's frames and adapts
  them once, per data type, into the arrays its four diagnostics use; the
  diagnostics themselves (coverage, the ES-style Kalman-gain ranking, the
  Mahalanobis diagnostic, update statistics) keep their algorithms. Along the
  way: the closures over a dozen loop variables became methods with
  arguments; the module no longer reseeds the global random state (it uses a
  private generator), no longer shells out to ImageMagick (`bbox_inches`
  trims the plots), and no longer needs OpenCV (one HLS colour conversion,
  now a few lines of numpy, so `opencv-python` is dropped); `actnum` is read
  from the config's `actnum` file rather than from the working directory;
  outputs go to `QAQC/` under the run's save folder; the localization used
  by the gain diagnostic is the scheme's own; and the level-2 and level-3
  Mahalanobis scores, the grid-dimension lookup for field plots and the
  cross-plots with fewer than four data are fixed. Multilevel ensembles are
  refused with a clear message: that branch could never run (`ne` was 0).
  Unit tests cover the adapter and each diagnostic on hand-built frames, and
  an end-to-end test runs `qa` and `qc` through ES-MDA and LM-EnRML.

- The characterisation suite pins thirteen `(scheme, analysis)` pairs instead
  of eight: LM-EnRML and GN-EnRML with `full` and `subspace`, and GN-EnRML with
  `margis`, are now under golden reference for the first time. The reference
  file was regenerated to add them; the eight existing entries moved by at
  most 3e-13 relative, the floating-point noise from the `eigh` and column-sum
  changes in the analysis kernel accumulated over three iterations.

- Tests run in a temporary directory by default (a suite-wide fixture in
  `tests/conftest.py`), so no test writes into the repository or the launch
  directory. The three end-to-end pipeline tests are seeded and carry a
  `slow` marker for `pytest -m "not slow"`. CI
  reports line coverage (`pytest-cov` is in the `dev` extra).

- **`co_lm_enrml` and `gn_enrml` are constructible and selectable again.**
  Both had been left in `enrml.py` as pre-refactor bodies that could not be
  constructed (a one-argument `__init__` against a three-argument parent) and
  read ensemble attributes that no longer exist. Neither was a distinct
  algorithm: `co_lm_enrml` only ever mixed the approximate analysis into
  LM-EnRML, and `gn_enrml`'s inline weight-space update is the `subspace`
  analysis with GN-EnRML's step-length schedule under the name `lambda`. They
  are now thin subclasses -- `co_lm_enrml` is `LMEnRML(analysis="approx")`,
  `gn_enrml` is `GNEnRML(analysis="subspace")` -- registered under their own
  names so a migrated config saying `scheme = "co_lm_enrml"` or
  `scheme = "gn_enrml"` runs, with a test that their results are identical to
  the algorithm they alias. Asking either for a different flavour raises the
  registry's usual "no such flavour" error.

- **Packaging and import time.** `mako`, `psutil` and `six` are no longer
  dependencies: nothing in PET imports the first two, and `six` served only
  Python 2 shims in the vendored Eclipse reader, now written with the
  standard library. `geostat` is pinned to a commit instead of tracking
  `main`, so a fresh install gets the code the tests were run against.
  `import pipt` no longer imports matplotlib, OpenCV or PyWavelets: QA/QC and
  sparse compression import them when a run asks for them, and
  `misc.structures` imports geostat only when it generates a prior. Cold
  import time drops from about 0.85 s to 0.5 s.

- **`truncSVD` keeps at least the requested energy fraction.** For
  `energy=e` the rank used to be the index at which the cumulative
  singular-value fraction first *reaches* `e`, which keeps everything before
  that point and so always retained *less* than `e`. It is now that index plus
  one, so the retained fraction is the first value at or above `e` -- the
  reading anyone gives "retain 98 percent", the scikit-learn convention, and
  what `full_update.ext_Am` in the same package already did, so the two
  truncations inside one `full` analysis now agree.

  ```
  S = [3, 2, 1], energy = 0.8
  before: rank 1, retains 0.50        after: rank 2, retains 0.83
  ```

  Two edge cases change with it. `energy=1` fell into the percentage branch
  and meant 1 percent, keeping a single singular value; `1` and `100` now both
  mean keep everything, with the fraction/percentage split at `energy > 1`. A
  zero spectrum keeps everything instead of dividing by zero.

  **Every analysis keeps one more singular value than before at the same
  `trunc_energy`**, so posteriors shift -- by up to 5.7 percent in the
  synthetic characterisation case. No config needs changing. The
  characterisation reference and the `test_lin_1d` expected values were
  regenerated for this change and nothing else. The fraction is still of the
  singular values themselves (the nuclear norm), not their squares; switching
  to Frobenius energy would be a modelling change and is not made here.

- **A scheme reaches its ensemble through declared properties, not
  `__getattr__`.** Reads a scheme does not own (`enX`, `pred_data`,
  `keys_da`, `localization`, ...) were forwarded to the ensemble by a blanket
  `__getattr__`, which resolved *any* name, was invisible to `dir()`,
  autocompletion and type checkers, and silently absorbed typos. Each of the
  25 names that actually crosses that boundary is now an explicit `property`
  on `AssimilationSchemeBase`: 21 read-only, plus `cov_data`, `scale_data`,
  `proj` and `Am`, which a scheme may legitimately compute for itself and so
  have setters. Reading is unchanged (`self.enX` still works everywhere);
  *assigning* a read-only one now raises `AttributeError` instead of quietly
  creating a shadow the forecast would never see. Ensemble state is still
  written explicitly through `self.ensemble.<name> = ...`.

- `logit` and `logger_name` are real `[dataassim]` options. Both were
  documented on the scheme base but could never take effect: the ensemble
  built its logger unconditionally, hardcoded to `assim.log`, and every scheme
  overwrote the scheme-side logger with the ensemble's. The ensemble now
  honours both, defaulting to `ASSIM.log`, and `logit = false` installs a
  no-op logger so no file is created at all.

- Packaging: corrected the license path (pointed at a nonexistent
  `LICENSE.txt`), moved test tooling to a `dev` extra, added classifiers and a
  supported-Python floor matching CI.
- CI runs a lint job, previously a `# TODO: Lint` comment.
- Removed 71 unused imports; replaced 33 bare `except:` clauses so
  `KeyboardInterrupt`/`SystemExit` are no longer swallowed. `ruff check src` is
  clean and enforced.
- The legacy `.pipt`/`.popt` parser's nested try/except cascade was rewritten as
  named helpers with identical behaviour.

### Removed
- `read_config.check_mand_keywords_fwd/da/opt/en`; `input_output.config.validate` is the check.
- The schemes' `max_iter` attribute. It existed only to derive `maxiter`, the loop's budget of updates, by subtracting the prior forecast the legacy loop counted as iteration 0. The config key `max_iter` and its meaning are unchanged.
- `pipt.ensembles.CompressionMixin` and its `compress_manager`, which rewrote the observation, variance and prediction frames cell by cell.
- `misc.read_input_csv`'s module-level readers (`read_data_df`, `read_var_df`, `read_data_csv`, `read_var_csv`, `convert_to_array`, `to_array_if_sequence`, 470 lines): nothing called them; `DataReader` is the reader.
- `BaseEnsemble.load()` and the `if self.restart is False:` guards around every scheme's and the ensemble's initialisation, which were always true. Construction now always initialises; a checkpoint is overlaid afterwards when `run_assimilation()` starts.

- `opencv-python` is no longer a dependency; QA/QC was its only user.

- `analysis_tools.screen_data`, whose only callers were the two unreachable
  `screendata` branches above; it also read `cov_data.p` from the working
  directory across iterations.

- **`pipt.misc_tools.ensemble_tools` keeps only `matrix_to_dict`.**
  `matrix_to_list`, `list_to_matrix` and `generate_prior_ensemble` had no
  callers, and `clip_matrix` duplicated `PETStateArray.clip_matrix` line for
  line; EnKF, its one caller, now clips through the state array's method like
  every other scheme (checked identical on tuple, dict and list limits).

- **Twelve unreferenced functions in `pipt.misc_tools.analysis_tools`**:
  `data_mismatch`, `calc_crosscov`, `update_datavar`,
  `extract_tot_empirical_cov`, `calc_kalmangain`, `calc_subspace_kalmangain`,
  `compute_x`, `resample_state`, `block_diag_cov`, `calc_kalman_filter_eq`,
  `subsample_state` and `get_obs_size`. Their last callers were the
  pre-refactor `co_lm_enrml`/`gn_enrml` bodies; the Kalman-gain trio was the
  superseded predecessor of the `analysis` package. The module's private
  `_is_enabled` was a copy of `extract_tools.is_enabled` and is gone too.

- **Dead code with no callers anywhere in the repository**, confirmed by grep
  over src, tests and docs: `pipt.misc_tools.data_tools` (every function
  duplicated a `PETDataFrame` method); `popt.misc_tools.basic_tools`
  (duplicated `input_output.get_ecl_key_val`); the `CMA` class in
  `popt.optimization_methods.subroutines`; the `lmenrmlMixIn`,
  `gnenrmlMixIn`, `esmdaMixIn`, `enkfMixIn` and `esMixIn` aliases; six
  functions in `popt.misc_tools.optim_tools` (`aug_optim_state`,
  `update_optim_state`, `corr2BlockDiagonal`, `time_correlation`, `corr2cov`
  and `get_optimize_result`, the last of which built its result with `eval`
  and referenced a module that no longer exists); the empty
  `pipt.update_schemes.update_methods_ns` directory; and a never-collected
  plotting helper in `test_autoadaloc.py` together with the ruff exemption
  that existed only for it. Code outside this repository that imported any of
  these should use the surviving equivalent: the `PETDataFrame` methods for
  `data_tools`, `get_ecl_key_val` for `basic_tools`, and the class names for
  the aliases.

### Known issues

- **`use_ensemble` in the `compress` section is not supported and now says so.** It
  meant: widen the leading wavelet indices with the first forecast, then
  compress the observations with them. Observations are perturbed when the
  scheme is built, before any forecast exists, and for this option the reader
  kept the observed vintage raw while giving it the compressed-length variance,
  so the two could never be used together; the perturbation step failed on the
  shape mismatch. A config that enables it now gets a `ValueError` explaining
  this. Supporting it means perturbing observations after the prior forecast,
  the same change `screendata` needs.

- **`screendata` is not supported and now says so.** Data screening inflates
  the variance of observations the ensemble cannot reach, which needs
  predictions; observations are perturbed when the scheme is built, before any
  forecast has run. The two remaining calls could never have worked (they
  passed four arguments to a five-argument function and read an `enPred` the
  ensemble never had), so a config that enables the option now gets a
  `ValueError` explaining this instead of an `AttributeError`. Supporting it
  again means perturbing observations after the prior forecast, which changes
  the order of random draws for every scheme.

- **Local analysis is broken along both routes.** `localization = {name =
  "localanalysis"}` reaches a branch that warns and returns `None`, so no update
  is applied and the run completes reporting a misfit — the posterior is the
  prior. Separately, `LocalAnalysisMixin` calls `self._ext_obs()`, which is
  defined nowhere in the codebase.
- **`es`/`enkf` with `analysis="subspace"`** raise `ValueError: Length of values
  (11) does not match length of index (15)`. `esmda/subspace` is unaffected, so
  the fault is in the sequential path.
- **The GIES schemes cannot be constructed.** `GIESMixIn.__init__` uses the
  pre-ensemble-matrix API (`self.state`, `self.obs_data`) and calls
  `self._ext_obs()`, which does not exist. Reproduced unchanged before the
  Phase 8 work, so this predates it.
- `docs/tutorials/pipt/TinyBox/tutorial_pipt.ipynb` has been updated to the
  current API but **not re-executed** — running it needs the OPM `flow`
  simulator through the external `subsurface` package, so its stored outputs
  are from the old code.
- `docs/tutorials/popt/5Spot/tutorial_popt.ipynb` targets the current API
  (`popt.optimization_methods.LineSearch`, `popt.ensembles.GaussianEnsemble`)
  but likewise needs `subsurface.multphaseflow.opm.flow`, which is not a
  dependency of this repository, so neither notebook is executed by the docs
  build or CI.
