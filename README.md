# PET: Python Ensemble Toolbox

<h1 align="center">
<img src="https://github.com/Python-Ensemble-Toolbox/.github/blob/main/profile/pictures/logo.png?raw=true" width="300">
</h1><br>

PET is a toolbox for ensemble-based Data Assimilation and Optimisation.
It is developed and maintained by the eponymous group
at NORCE Norwegian Research Centre AS.

[![CI status](https://github.com/Python-Ensemble-Toolbox/PET/actions/workflows/tests.yml/badge.svg)](https://github.com/Python-Ensemble-Toolbox/PET/actions/workflows/tests.yml)


## Installation

PET requires Python 3.12 through 3.14. The standard installation includes
EnIF and EnIF-MDA with their dependencies.

To install PET, first clone the repo (assuming you have added the SSH key)

```
git clone git@github.com:Python-Ensemble-Toolbox/PET.git PET
```
Alternatively, [click here to download ZIP](https://github.com/Python-Ensemble-Toolbox/PET/archive/refs/heads/main.zip).

### On Linux

PET is installed with [uv](https://docs.astral.sh/uv/), which also provides Python itself.
Install uv (see its [installation page](https://docs.astral.sh/uv/getting-started/installation/)
for alternatives):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Inside the PET folder, run

```sh
uv sync
```

- This creates a virtual environment in `.venv` and installs PET into it, with every dependency
  at the version recorded in `uv.lock`, so that everyone runs the same versions.
- PET is installed such that changes to it take effect immediately (without re-installation).
- Run commands in the environment with `uv run`, e.g. `uv run pet ...`, or activate it with
  `source .venv/bin/activate`.

To also install the tools needed for running tests and linting locally:

```sh
uv sync --extra dev --extra minires
uv run pytest
```

`uv sync` removes packages that are not in `uv.lock`. If you have installed others into the
environment yourself (e.g. `uv pip install -e ../SimulatorWrap`), keep them with `uv sync --inexact`.

After changing the dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock`;
`uv lock --upgrade` moves every dependency to its latest allowed version.

#### With pip

pip works as before, but ignores `uv.lock`: it installs the latest versions that
`pyproject.toml` allows. In a virtual environment, inside the PET folder:

```sh
python3 -m venv venv-PET
source venv-PET/bin/activate
python3 -m pip install -e ".[dev]"
```

### On Windows

TL;DR: use `conda` to create the environment and install PET. Inside the PET folder,
run:

```powershell
conda env create -f environment.yml -n pet
conda activate pet
```
For more details refer to
[Installing PET on Windows](docs/installation_windows.md).
This setup may also work on Linux, though it has not been tested yet.

## Documentation

The [configuration reference](docs/configuration.md) lists every key of the
`dataassim`, `ensemble`, `optim` and `simulator` sections; the
[architecture page](docs/architecture.md) explains how a run is put together
and where a new scheme, analysis, localization, optimizer or simulator goes.

## Command-line interface

Installing PET also installs a `pet` command for working with config files:

```sh
pet validate my_config.toml   # check a config file for missing/invalid keys
pet convert my_case.pipt      # convert a legacy .pipt/.popt file to .toml (or --to yaml)
pet migrate my_config.toml    # update a config file to the current schema
pet version                   # print the installed PET version
```

## Examples

PET needs to be set up with a configuration file. See the example [repository](https://github.com/Python-Ensemble-Toolbox/Examples) for inspiration.

## Simulation wrappers

To use the subsurface simulators Eclipse or OPM, you need to install the [SimulatorWrap](https://github.com/Python-Ensemble-Toolbox/SimulatorWrap) repository. 
This repository also contains instructions on how to link your own simulator to PET.

## Visualization
Some basic plotting functionality is provided [here](https://github.com/Python-Ensemble-Toolbox/Plotting). The functions should be copied and adapted for each specific use cases.

## Tutorials

- A PIPT tutorial is found [here](https://python-ensemble-toolbox.github.io/PET/tutorials/pipt/TinyBox/tutorial_pipt)
- A POPT tutorial is found [here](https://python-ensemble-toolbox.github.io/PET/tutorials/popt/5Spot/tutorial_popt)

## Suggested readings:

If you use PET in a scientific publication, we would appreciate it if you cited one of the first papers where the PET was introduced. Each of them describes some of the PET's functionalities:

### Bayesian data assimilation with EnRML and ES-MDA for History-Matching Workflow with AI-Geomodeling
#### Cite as
Fossum, Kristian, Sergey Alyaev, and Ahmed H. Elsheikh. "Ensemble history-matching workflow using interpretable SPADE-GAN geomodel." First Break 42.2 (2024): 57-63. https://doi.org/10.3997/1365-2397.fb2024014

```
@article{fossum2024ensemble,
  title={Ensemble history-matching workflow using interpretable SPADE-GAN geomodel},
  author={Fossum, Kristian and Alyaev, Sergey and Elsheikh, Ahmed H},
  journal={First Break},
  volume={42},
  number={2},
  pages={57--63},
  year={2024},
  publisher={European Association of Geoscientists \& Engineers},
  url = {https://doi.org/10.3997/1365-2397.fb2024014}
}
```

###  Bayesian inversion technique, localization, and data compression for history matching of the Edvard Grieg field using 4D seismic data
#### Cite as

Lorentzen, R.J., Bhakta, T., Fossum, K. et al. Ensemble-based history matching of the Edvard Grieg field using 4D seismic data. Comput Geosci 28, 129–156 (2024). https://doi.org/10.1007/s10596-024-10275-0


```
@article{lorentzen2024ensemble,
  title={Ensemble-based history matching of the Edvard Grieg field using 4D seismic data},
  author={Lorentzen, Rolf J and Bhakta, Tuhin and Fossum, Kristian and Haugen, Jon Andr{\'e} and Lie, Espen Oen and Ndingwan, Abel Onana and Straith, Knut Richard},
  journal={Computational Geosciences},
  volume={28},
  number={1},
  pages={129--156},
  year={2024},
  publisher={Springer},
  url={https://doi.org/10.1007/s10596-024-10275-0}
}
```

###  Offshore wind farm layout optimization using ensemble methods
#### Cite as

Eikrem, K.S., Lorentzen, R.J., Faria, R. et al. Offshore wind farm layout optimization using ensemble methods. Renewable Energy 216, 119061 (2023). https://www.sciencedirect.com/science/article/pii/S0960148123009758

```
@article{Eikrem2023offshore,
title = {Offshore wind farm layout optimization using ensemble methods},
journal = {Renewable Energy},
volume = {216},
pages = {119061},
year = {2023},
issn = {0960-1481},
doi = {https://doi.org/10.1016/j.renene.2023.119061},
url = {https://www.sciencedirect.com/science/article/pii/S0960148123009758},
author = {Kjersti Solberg Eikrem and Rolf Johan Lorentzen and Ricardo Faria and Andreas St{\o}rksen Stordal and Alexandre Godard},
keywords = {Wind farm layout optimization, Ensemble optimization (EnOpt and EPF-EnOpt), Constrained optimization, Levelized cost of energy (LCOE), Floating offshore wind},
}
```
