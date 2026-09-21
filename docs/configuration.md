# Configuration reference

A run is described by three sections: the **problem** (`[dataassim]` for
assimilation, `[optim]` for optimisation), the **ensemble**, and the
**simulator** (`[fwdsim]` is accepted as its name too). They can be written in
TOML, YAML or the legacy `.pipt`/`.popt` text format, or built as Python
dictionaries in a script. Whichever way they arrive, `input_output.config`
normalises them once -- canonical key names, booleans for flags, dictionaries
for sub-blocks -- and everything downstream reads that one form. `pet validate
my_config.toml` reports what is missing, by section and key, and points out
keys nothing in PET reads.

Flags accept `true`/`false`, `yes`/`no` and the Python booleans. A key marked
*presence* is on when it is present at all, whatever its value.

## `[dataassim]`

### The problem

| Key | Meaning | Default |
| --- | --- | --- |
| `scheme` | Algorithm: `esmda`, `es`, `enkf`, `lmenrml`, `gnenrml`. `pipt.available_schemes()` lists every `(scheme, analysis)` pair. | required |
| `analysis` | Analysis flavour the scheme runs: `approx`, `full`, `subspace` (all schemes); `subspace2` (ES-MDA, LM-EnRML, GN-EnRML); `margis` (GN-EnRML). `subspace2` solves for the ensemble transform directly and uses the analytic data covariance, so it reads neither `energy` nor `iteration.energy`. | `approx` |
| `energy` | Truncation energy of the SVD in ES-MDA, ES and EnKF; a fraction, or a percentage when greater than 1. The iterative schemes read `iteration.energy`. | `0.98` |
| `emp_cov` | The variance file holds an ensemble of observation errors; the analyses use that empirical covariance. Flag. | off |

### Observed data

| Key | Meaning | Default |
| --- | --- | --- |
| `data` | Observations: a `.csv`, `.pkl` or `.npz` file with one row per report label and one column per data type. A cell may name a `.npz` file holding a vector (seismic). `truedata` is the older spelling. | required |
| `datavar` | Variance file on the same geometry. Each cell is `['abs', v]`, `['rel', percent]`, `['emp', ensemble]` or `['cd', covariance.npz]`. `var` is the older spelling. | required |
| `obsname` | Name of the report-label index (times, dates, steps). Not needed when `data` is a dict carrying `index_name`. | required |
| `datatype` | Data types to assimilate. Normally given in the simulator section and copied here. | from the data file |
| `assimindex`, `truedataindex` | Derived from the data file at load time; a value written here is replaced. | derived |
| `scale_data` | Max-min scale observations and predictions per data type before the analysis. Flag. | off |
| `scale` | `[types, factor]`: multiply the predictions of the named data types by `factor`. | none |
| `remove_outliers` | Replace members whose normalised misfit is more than four standard deviations from the mean after every forecast. *Presence.* | off |
| `actnum` | `.npz` with an `actnum` mask, used by the iterative schemes and QA/QC to map a field back to the grid. | none |

### Iteration (`lmenrml`, `gnenrml`): the `[dataassim.iteration]` block

| Key | Meaning | Default |
| --- | --- | --- |
| `max_iter` | Number of update iterations. The prior forecast is not one of them. | required |
| `data_misfit_tol` | Stop when the relative change of the mean data misfit is below this. | `0.01` |
| `energy` | Truncation energy of the SVD; a fraction, or a percentage when greater than 1. | `0.95` |
| `max_inner_iter` | Attempts one iteration may make (tightening the control after each rejected step) before giving up. | `10` |
| `lambda` | LM-EnRML: initial damping. `auto` sizes it from the prior misfit. | `10` |
| `lambda_max`, `lambda_min` | LM-EnRML: bounds on the damping; reaching `lambda_max` stops the run. | `1e10`, `0.01` |
| `lambda_factor` | LM-EnRML: factor the damping is divided (accepted step) or multiplied (rejected step) by. | `5` |
| `gamma` | GN-EnRML: step length. `auto` starts at `0.1`. | `0.2` |
| `gamma_max`, `gamma_factor` | GN-EnRML: bound and update factor for the step length. | `1.0`, `2.0` |

### ES-MDA: the `[dataassim.mda]` block

| Key | Meaning | Default |
| --- | --- | --- |
| `tot_assim_steps` | Number of inflated assimilation steps; one update each. | required |
| `inflation_param` | Inflation factor per step (a list) or one factor for all. The inverses must sum to 1. | `tot_assim_steps` for every step |

### Localization: the `[dataassim.localization]` block

`name` selects the strategy; `pipt.localization.available_localizations()`
lists them, `register_localization` adds one. All strategies take `field`
(grid dimensions as a list of integers) and an optional `actnum` (`.npz` mask).

A block that gives no `name` is still understood: the mode is inferred from the
keyword that used to select it — `autoadaloc`, `localanalysis` or `dist_loc` (as a
key or as a bare value), a `.p`/`.pkl` mask file for `distance_loc`, and none of
them for the parallel update. An explicit `name` always wins.

| `name` | Keys | Meaning |
| --- | --- | --- |
| `autoadaloc` | `threshold` (`adaptive`, `fixed`, `universal`), `cutoff`, `type` (`hard`, `soft`, `sigm`), `projection` (`rank-r`, `ensemble`), `parameters` | Auto-adaptive localization from the correlations the ensemble itself shows. `cutoff` is how many noise standard deviations a correlation must clear (default `0.3`); it is also read from `nstd`, or from the value of `autoadaloc` itself. |
| `distance_loc` | `taper_func` (`gaspari_cohn`, `furrer_bengtsson`, `region`), `entries` (list of rows or a `.csv`) | Distance-based tapering around each datum: per entry a data type, report label, parameter, radius, anisotropy and vertical range. |

`localanalysis` and the parallel update are **not supported**: both need per-subset
observation machinery the scheme rewrite replaced, and naming either raises a
`ConfigError` saying so. Use `distance_loc` or `autoadaloc` instead.

### Seismic compression: the `[dataassim.compress]` block

Observed vintages named in `compress_data` are wavelet-compressed when read,
and every prediction of that data type is reduced to the same leading
coefficients as it enters the prediction matrix.

| Key | Meaning |
| --- | --- |
| `compress_data` | Data type (or list) to compress. |
| `dim` | Grid dimensions of a vintage. |
| `mask` | One `.npz` (key `mask`) per vintage; a missing file means all cells active. |
| `level`, `wname` | Wavelet decomposition level and PyWavelets wavelet name (`db2`). |
| `threshold_rule`, `th_mult`, `use_hard_th`, `keep_ca` | `universal` or `bayesian` thresholding, its multiplier, hard vs soft thresholding, whether the approximation coefficients are kept. |
| `inactive_value`, `order`, `min_noise`, `colored_noise` | Fill value outside the mask, flatten order (`C`/`F`), noise floor per vintage, per-subband noise estimate. |
| `use_ensemble` | Not supported; refused with the reason. |

`post_process_forecast` (flag) additionally divides `sim2seis` data types by
the factor in `scale_results.pkl`, when that file is present.

### Restart

| Key | Meaning | Default |
| --- | --- | --- |
| `restartsave` | Write a checkpoint after the prior forecast and every accepted iteration. Flag. | off |
| `restart` | Resume from the checkpoint instead of starting. Flag. | off |
| `restart_file` | Path of the checkpoint. `restartfile` is the older spelling. | `<scheme>_restart.pkl` |

A resumed run continues the interrupted one exactly: the checkpoint carries the
loop's bookkeeping, the scheme's state and the ensemble's state, prior,
forecast and random stream.

### Output

| Key | Meaning | Default |
| --- | --- | --- |
| `savefolder` | Folder for results. `save_folder` is the older spelling. | `Results` |
| `nosave` | Write no result files at all. *Presence.* | off |
| `savedata` | Attribute names saved per iteration to `assimilation_result_{i}.npz` (iteration 0 is the prior); `state` expands to one array per variable. `analysisdebug` is the deprecated spelling. | none |
| `iterinfo` | Python modules (`name.py`) whose `main(scheme)` runs after the prior and every accepted iteration. | none |
| `obsvarsave` | Also save the observed data and variance frames as `obs_data.pkl` and `obs_var.pkl`. Flag. | off |
| `qa`, `qc` | Run quality-assurance plots / quality-control statistics after the prior and every iteration. *Presence.* | off |
| `logit`, `logger_name` | Whether to log, and the log file. | on, `ASSIM.log` |

`screendata` is not supported and says so.

## `[ensemble]`

| Key | Meaning | Default |
| --- | --- | --- |
| `ne` | Ensemble size. | `100` when a prior is generated |
| `state` | State variable name(s). | required (or `controls`) |
| `prior_<name>` | Prior of each state variable; see below. | required unless `importstate` |
| `importstate` | `.npz` with one `(n, ne)` array per state variable, used instead of generating a prior. `importstaticvar` is the older spelling. | none |
| `seed` | Seed for the run's private random stream: prior, perturbed observations, outlier and crash replacement, localization shuffles. Without it NumPy's global stream is used. | none |
| `save_prior` | Write the generated prior as `prior_ensemble.npz`. Flag. | on |
| `sim_limit` | Wall-time limit passed to the simulator. | none |
| `disable_tqdm` | Hide progress bars. Flag. | off |
| `multilevel` | Multilevel ES-MDA: `levels`, `en_size` (members per level), `ml_weights` or `cov_wgt` (weights per level, normalised). | none |
| `savefolder` | Folder for popt's `save_prediction` output. | `Predictions` |

### `[ensemble.prior_<name>]`

| Key | Meaning |
| --- | --- |
| `mean` | A number, a list per cell, or a `.npz` holding the mean field. |
| `var` (or `variance`) | Variance per layer. |
| `range` (or `corr_length`), `aniso`, `angle`, `vario` | Correlation length, anisotropy, angle and variogram type (`sph`, `exp`, `gau`) of a field prior. |
| `grid` | `[nx, ny, nz]`; scalars are `[1, 1, 1]`. |
| `limits` | `[lower, upper]` the realisations and every update are clipped to. |

### popt additions

| Key | Meaning | Default |
| --- | --- | --- |
| `controls` | `{name: {initial or mean, var/variance or std ('5%' of the range needs limits), limits}}`; values may be `.npy`, `.npz` or `.csv` files. | required |
| `natural_gradient` | Gaussian ensemble: scale the gradient by the covariance. Flag. | on |
| `num_models` | Realisations per control for robust optimisation. | `1` |
| `save_prediction` | Pickle each forecast under `savefolder` with this name. | none |
| `marginal`, `theta` | Generalized ensemble: marginal family (`BetaMC`, `Beta`, `Logistic`, `TruncGaussian`, `Gaussian`) and its parameters. | `BetaMC` |

## `[optim]`

Options every optimizer takes (`popt.optimization_methods.OptimizerBase`):

| Key | Meaning | Default |
| --- | --- | --- |
| `maxiter` | Number of update iterations. | `100` |
| `ftol`, `xtol`, `gtol` | Stop on relative objective change, on state-change norm, on projected-gradient infinity norm. | `1e-5`, `1e-8`, `1e-5` |
| `transform` | Optimise in the unit cube `[0, 1]^n` (needs bounds). Flag. | off |
| `saveit`, `savefolder` | Save the result after every iteration, and where. | off, `Iteration_Results` |
| `restart`, `restartsave`, `restart_file` | Checkpointing, as for the schemes. | off, off, `<optimizer>_restart.pkl` |
| `logit`, `logger_name` | Whether to log, and the log file. | on, `OPTIM.log` |
| `fun0`, `jac0`, `hess0` | Starting values to reuse instead of evaluating. | none |
| `epf` | Exterior penalty: `r`, `r_factor`, `tol_factor`, `conv_crit`, `max_epf_iter`. `conv_crit` is compared against the mean penalty with `r` divided out, so the objective must write `penalty` into the `epf` dict it is handed. | none |

Per optimizer:

| Optimizer | Keys |
| --- | --- |
| `EnOpt` | `tol`, `alpha` (or `step_size`), `alpha_cov`, `beta`, `nesterov`, `alpha_maxiter`, `resample`, `cov_factor`, `hessian`, `normalize`, `optimizer` (`GD`, `Adam`, `AdaMax`, `Steihaug`) |
| `GenOpt` | `tol`, `alpha` (or `step_size`), `alpha_theta`, `alpha_corr`, `beta`, `nesterov`, `alpha_maxiter`, `resample`, `cov_factor`, `normalize`, `optimizer` (`GD`, `Adam`). Takes `args = (theta, corr)`, a `jac_mut` mutation gradient, and an optional `corr_adapt` (a `CMA` instance or any callable). |
| `LineSearch` | `step_size`, `step_size_max`, `step_size_adapt`, `c1`, `c2`, `rho`, `lsmaxiter`, `lsmethod` (0 backtracking, 1 Wolfe), `normalize`, `recompute_jac`, `hess0_inv` |
| `TrustRegion` | `trust_radius`, `trust_radius_max`, `trust_radius_min`, `trust_radius_cuts`, `rho_tol`, `eta1`, `eta2`, `gam1`, `gam2`, `resample`, `convergence_criteria` |
| `SmcOpt` | `tol`, `alpha`, `alpha_maxiter`, `resample`, `cov_factor`, `inflation_factor`, `survival_factor`, `best_func` |

The constructors document each key.

## `[simulator]` (or `[fwdsim]`)

PET reads a few keys; the rest belong to the simulator wrapper.

| Key | Meaning | Default |
| --- | --- | --- |
| `datatype` | Data types the simulator reports, in order. | required |
| `reporttype`, `reportpoint` | Name and values of the report labels: a list, a `.csv`/`.txt`/`.yaml` file, or `{start, end, freq}` for a date range. | required by most wrappers |
| `parallel` | Members run at once in a local process pool; `1` runs them in sequence. | `1` |
| `hpc` | Run the members through the wrapper's HPC queue in batches of `parallel`. Flag. | off |
| `compute_adjoints` | The wrapper returns `(prediction, adjoint)` per member; the adjoints reach the analysis as an `(nd, nx, ne)` array. Flag. | off |
| `saveforecast` | Save each full forecast (`sim_results.pkl`) and the reconstructed compressed vintages (`rec_results.pkl`). *Presence.* | off |

## Legacy text files

`.pipt`/`.popt` files keep the ensemble's keys in the `DATAASSIM` block and
lower-case every value, so data-type names written in upper case must match
lower-case columns. `pet convert my_case.pipt` writes the same content as TOML,
and `pet migrate` updates a file from `daalg` to `scheme`.
