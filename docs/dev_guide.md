For now `index.html` redirects to this module
(lookup `"dev_guide.py"` or `"dev_guide.md"` in the source)
making it the front page.
Once there is a sensible root package to point to,
(containing references to the others) we will change that.
Similarly, manually making the following list will be unnecessary
once all of the packages (folders) are put into a root folder,
(as they should be).

- `ip`
- `fwd_sim`
- `input_output`
- `geostat`
- `optimization`

However, only a top-level lists will be generated.
If we want a full (recursive) tree/index of modules,
we have to wait for [this](https://github.com/pdoc3/pdoc/issues/101).
Meanwhile, the following was generating using

```sh
tree | grep -v "pyc" | grep -v "__init__.py" | grep -v "txt$"

├── debug
│   └── traceme.py
├── docs
│   ├── dev_guide.md
│   ├── dev_guide.py
│   ├── index.html
│   ├── manual_inv_prob_python.pdf
│   ├── templates
│   │   └── config.mako
├── fwd_sim
│   ├── deterministic_loops.py
│   ├── deterministic.py
│   ├── ensemble_loops.py
│   ├── ensemble.py
│   ├── optim_loops.py
│   ├── optim.py
├── geostat
│   ├── decomp.py
│   ├── gaussian_sim.py
│   ├── get_variogram.py
│   ├── gslib.py
├── input_output
│   ├── mare2dem.py
│   ├── modem.py
│   ├── pipt_init.py
├── ip.py
├── log_optim_loops.log
├── misc_tools
│   ├── analysis_tools.py
│   ├── basic_tools.py
│   ├── coordtrans.py
│   ├── cov_regularization.py
│   ├── genetic.py
│   ├── gradient_field.py
│   ├── green_tools.py
│   ├── grid_select.py
│   ├── heaviside_dirac.py
│   ├── levelset_tools.py
│   ├── LS_matchspace.py
│   ├── matchspace.py
│   ├── old_matchspace.py
│   ├── optim_tools.py
│   ├── seismic_tools.py
│   └── wavelet_tools.py
├── optimization
│   ├── levelset_optim.py
├── paramrep
│   ├── levelset.py
├── PIPT.egg-info
│   ├── PKG-INFO
├── plot
│   ├── csem_plot.py
│   ├── levelset_plot.py
│   ├── upscale_field.py
│   ├── upscale_plot.py
│   └── upscale_stats.py
├── post_processing
│   ├── bayesian_averaging.py
├── README.md
├── rockphysics
│   ├── ekofisk_pem.py
│   ├── em.py
│   ├── nornep.py
│   └── standardrp.py
├── setup.py
├── simulator
│   ├── csem.py
│   ├── debug_simulator.py
│   ├── flow_rock_feat.py
│   ├── flow_rock.py
│   ├── gravimetry.py
│   ├── log_gp.py
│   ├── log.py
│   ├── mt.py
│   ├── Ray_runners
│   │   ├── simple_models.py
│   │   └── subsurf_flow.py
│   ├── seismic.py
│   ├── shallowlogs.py
│   ├── simple_models.py
│   └── subsurf_flow.py
├── system_tools
│   ├── environ_var.py
├── tests
│   └── test_stuff.py
└── update_schemes
    ├── ensemble_optimization.py
    ├── iter_ensemble.py
    ├── ml_ensemble.py
    └── seq_ensemble.py

```

## Documentation generation

The documentation is generated using [pdoc3](https://pdoc3.github.io/pdoc/doc/pdoc),
which grabs the docstrings of modules/classes/functions,
and renders them into pretty html.
The docstrings should be written using markdown syntax,
[with some extras (cross-referencing, math, etc)](https://pdoc3.github.io/pdoc/doc/pdoc/#supported-docstring-formats)

### Run pdoc locally

Unlike the typical case with `sphinx`,
we do not use `readthedocs` for generation and hosting,
but rather GitHub Actions and Pages.
To preview you changes to the docs,
do the following (manually including all of the folders/packages will become unnecessary
once they've all been put into a root folder (which they should be)):

```sh
pdoc --force --html --template-dir docs/templates -o ./docs/ \
    docs/dev_guide.py ip.py fwd_sim input_output geostat optimization
```

Open `docs/index.html` to preview the result.

There is also a nice `--http` flag to get live reloading
as you make changes. Use it for example as follows:

```sh
pdoc --template-dir docs/templates --http : ip.py
```

## Tests

Run the tests by the command

```sh
pytest
```

It will discover all [appropriately named tests](https://docs.pytest.org)
in the source (see the `tests` dir).

Use (for example) `pytest --doctest-modules docs/dev_guide.py` to
*also* run any example code **within** docstrings.

The CI also simply runs `pytest`.
