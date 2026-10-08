# Installing PET on Windows

On Windows, install PET with the `conda` package manager. A plain
`pip install -e .` (as well as `uv pip install -e .`) fails, because of
`graphite-maps` and its dependencies: PET's EnIF analysis depends on
`graphite-maps`, which in turn needs `scikit-sparse` and the CHOLMOD C library.
Neither package has a Windows build on PyPI; conda-forge provides
`scikit-sparse` and CHOLMOD ready-made.

## Prerequisites

- [Miniforge](https://conda-forge.org/download/), or another conda
  installation that can reach the conda-forge channel.
- [Git for Windows](https://git-scm.com/download/win) on your `PATH`.
  `graphite-maps` and PET's `geostat` dependency are installed from git.

## Install

From the root of your PET folder:

```powershell
conda env create -f environment.yml -n pet
conda activate pet
```

This creates an environment named `pet` with Python 3.13, installs the compiled
dependencies from conda-forge, `graphite-maps` 0.0.11 from GitHub, and PET
itself in editable mode, so changes to the source take effect immediately.

To run the Eclipse or OPM simulators, also install
[SimulatorWrap](https://github.com/Python-Ensemble-Toolbox/SimulatorWrap) from
its checkout:

```powershell
pip install -e ..\SimulatorWrap
```

## Check the installation

```powershell
python -c "import pipt; from sksparse.cholmod import cholesky; print('PET OK')"
pet version
```

To run the test suite, install `pytest` and run it from the PET folder:

```powershell
pip install pytest
python -m pytest -m "not slow"
```

## Things to know

- **Always activate the environment** (`conda activate pet`) before running
  PET. Calling `envs\pet\python.exe` directly, without activation, crashes in
  NumPy's linear algebra because its DLLs are not found.
- **Keep `scikit-sparse` below 0.5.** `graphite-maps` 0.0.11 requires it, and
  conda-forge also offers 0.5.
- **After updating PET**, rerun `pip install -e .` in the activated
  environment if `pyproject.toml` has changed. An editable install picks up
  source changes, but not new dependencies.
- **Behind a corporate proxy**, `mamba` and conda-forge's own `git` package can
  fail with `schannel: the revocation status is unknown`. Use `conda` rather
  than `mamba`, and Git for Windows rather than a conda-installed git. This is
  why `git` is not listed in `environment.yml`.
- **`conda list` may show `scipy` as coming from `pypi`.** It is still the
  conda-forge build; pip does not reinstall it.

## Known test failures on Windows

With the environment above, 583 of 589 non-slow tests pass (October 2026). The
six failures are not installation problems:

- Five tests in `tests/assimilation/test_subspace2.py` require two analyses to
  agree bit for bit. On Windows they differ in the last floating-point digit
  (about 1e-16).
- `tests/test_logging_and_paths.py::test_save_folder_is_not_created_by_reading_it`
  expects a `/` path separator and gets `\`.

## Conda on Linux

The same `environment.yml` is expected to work on Linux for those who use
`conda` as a package manager, though it has not been tested.
