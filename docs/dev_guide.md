# Developer guide

## Repository layout

PET is one repository holding two toolboxes on a shared foundation. Every
package lives under `src/`.

| Package | Role |
| --- | --- |
| `ensemble` | The foundation both toolboxes build on: the base ensemble (prior generation, forecast orchestration), checkpoint/restart, logging. It must not import `pipt` or `popt` at module level; `tests/test_import_hygiene.py` enforces this. |
| `pipt` | Data assimilation. Schemes in `update_schemes/`, analysis flavours in `update_schemes/analysis/`, the assimilation ensemble in `ensembles/`, localization in `localization/`, numerical helpers in `misc_tools/`. |
| `popt` | Optimisation. Optimizers in `optimization_methods/`, the ensembles that estimate gradients in `ensembles/`, cost functions in `cost_functions/`. |
| `misc` | Data structures (`PETDataFrame` as the table view, `DataLayout`/`PredictedData` for the data matrices, `StateLayout` for the state's variable rows), the observed-data reader, and vendored Eclipse grid and output readers used by external simulator wrappers. |
| `input_output` | Config parsing (`.toml`, `.yaml`, and the legacy `.pipt`/`.popt` text format) and report-point handling. |
| `simulator` | Small analytical simulators used by the tests and tutorials. Reservoir simulators live in the external SimulatorWrap repository. |
| `pet_cli` | The `pet` command: `validate`, `convert`, `migrate`, `version`. |

### How a run is put together

The [architecture page](architecture.md) describes the layers and contracts in
full and the [configuration reference](configuration.md) every key; this is
the short version.

A **scheme** (`pipt.update_schemes.core.AssimilationScheme`) owns the
iteration loop, the convergence checks and the checkpointing. It holds an
**ensemble** collaborator (`pipt.ensembles.AssimilationEnsemble`) that owns
the state realisations, the observed data and the forward simulator, and it
binds an **analysis** object (`pipt.update_schemes.analysis`) that computes the
update step from the state, predicted-data and perturbed-observation matrices.
Which flavours a scheme supports is declared on the class in
`COMPATIBLE_ANALYSES`; the registry (`pipt.update_schemes.registry`) derives
every selectable `(scheme, analysis)` pair from those tables. Every scheme
also accepts a ready-made `ensemble=`, so two schemes can share one prior
and a test can hand in a stand-in. Localization strategies are selected from
`pipt.localization.LOCALIZATIONS` by the config's `name`; a new one is a
call to `register_localization`. The two notebooks under *Extending PIPT* in
the tutorials walk through adding an analysis and adding a scheme.

A forward simulator is anything satisfying `ensemble.protocols.ForwardSimulator`:
an `input_dict` and a `run_fwd_sim(state, member_index)` method, plus the
optional hooks the protocol's docstring lists. The analytical models in
`simulator/` are the smallest complete examples.

`popt` has the same shape: an optimizer
(`popt.optimization_methods.optimizer_base.OptimizerBase`) owns its loop and is
handed `fun`/`jac`/`hess` callables, typically the methods of an ensemble from
`popt.ensembles`. A new optimizer implements `update_step()`, which commits an
improving point with `_commit_step(x, f, jac=..., hess=...)` and returns a
`StepReport`, and `log_columns()` for its row of the log. The base evaluates
the starting point, runs the callback, records and saves the result, logs,
and checks function, state and projected-gradient convergence.

## Tests

The suite is `pytest`, configured in `pyproject.toml` and run in CI on
Python 3.10 to 3.12.

```sh
pytest                    # everything, about two minutes
pytest -m "not slow"      # skip the three end-to-end pipeline tests
pytest --cov=src          # with line coverage (pytest-cov is in the dev extra)
ruff check src tests      # lint; CI fails on findings
```

Every test starts in its own temporary directory (`tests/conftest.py`), so a
test may write files freely without touching the repository.

`tests/assimilation/test_numerical_characterisation.py` pins the numbers every
shipped `(scheme, analysis)` pair produces on a small Van der Pol case. A
refactor that is meant to preserve behaviour should leave it green. When a
change to the numbers is intended, regenerate the reference deliberately and
say so in the CHANGELOG:

```sh
python tests/assimilation/test_numerical_characterisation.py --regenerate
```

## Changelog

User-visible changes are recorded in `CHANGELOG.md`, following
[Keep a Changelog](https://keepachangelog.com/). A change that alters results
gets an entry that names the change and states that the reference was
regenerated for it.

## Writing documentation

The documentation is built with `mkdocs` and the Material theme.

- Pages are [Markdown](https://www.markdownguide.org/cheat-sheet/), augmented
  by [several pymdown extensions](https://squidfunk.github.io/mkdocs-material/reference/).
- **Docstrings** are rendered by `mkdocstrings`. Declare parameters and return
  values in the [numpy style](https://mkdocstrings.github.io/griffe/reference/docstrings/#numpydoc-style),
  and put `>>>` examples under an "Examples" heading.

!!! note
    Preview the rendered site with
    ```sh
    mkdocs serve
    ```
    Temporarily disable `mkdocs-jupyter` in `mkdocs.yml` to speed up reloads,
    and set `validation: unrecognized_links: warn` to surface broken links.

### Linking to pages

Use relative page links including the `.md` extension, for example
`[link label](sibling-page.md)`; these are validated by the build. Absolute
links are not, and neither GitHub's Markdown rendering nor an editor can follow
them, so avoid them.

### Linking to headers and API items

Thanks to the `autorefs` plugin, a heading anywhere in the site can be linked
without its page path: `[visible label][anchor]`, or the shorthand
`[anchor][]`. Anchors are lowercase. This also covers

- **API items**, for example ``[`pipt.update_schemes.esmda.ESMDA`][]``, and
- **references**, for example ``[`chen2013`][]``.

### Docstring injection

`::: pipt.update_schemes.esmda` injects a module's rendered docstrings. This
is rarely written by hand: `docs/gen_ref_pages.py` generates one such page per
module under `src/` at build time, which is what the *Reference* section is.

### Including other files

The `pymdown` ["snippets"](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/#snippets-notation)
extension includes text from another file:
`--8<-- "path/from/project/root/filename.ext"`. The home page includes
`README.md` this way.

### Tutorials

Tutorials are Jupyter notebooks under `docs/tutorials/`, listed in
`docs/tutorials/README.md`. The build renders their stored outputs and does
not execute them (`execute: false`): the reservoir cases need the OPM `flow`
simulator through the external `subsurface` package.

### Bibliography

Add new references as BibTeX to `docs/bib/refs.bib`, then run
`docs/bib/bib2md.py`, which formats them into `docs/references.md` so they can
be cited with the cross-reference syntax, e.g. `[chen2013][]`.

## Hosting

`.github/workflows/deploy-docs.yml` builds the site and publishes it to GitHub
Pages with `mhausenblas/mkdocs-deploy-gh-pages` whenever `main` is updated.
