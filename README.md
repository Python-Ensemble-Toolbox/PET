# PET: Python Ensemble Toolbox

<h1 align="center">
<img src="https://github.com/Python-Ensemble-Toolbox/.github/blob/main/profile/pictures/logo.png" width="300">
</h1><br>

PET is a toolbox for ensemble-based Data Assimilation and Optimisation.
It is developed and maintained by the eponymous group
at NORCE Norwegian Research Centre AS.

[![CI status](https://github.com/Python-Ensemble-Toolbox/PET/actions/workflows/tests.yml/badge.svg)](https://github.com/Python-Ensemble-Toolbox/PET/actions/workflows/tests.yml)


## Installation

Before installing ensure you have python3 pre-requisites. On a Debian system run:

```
sudo upt-get update
sudo apt-get install python3
sudo apt-get install python3-pip
sudo apt-get install python3-venv
```

To install PET, first clone the repo (assuming you have added the SSH key)

```sh
git clone git@github.com:Python-Ensemble-Toolbox/PET.git PET
```

Make sure you have the latest version of `pip` and `setuptools`:

```sh
python3 -m pip install --upgrade pip setuptools
```

Optionally (but recommended): Create and activate a virtual environment:

```sh
python3 -m venv venv-PET
source venv-PET/bin/activate
```

Some additional features might be not part of your default installation and need to be set in the Python (virtual) environment manually:

```
python3 -m pip install wheel
python3 setup.py bdist_wheel
```

If you do not install PET inside a virtual environment,
you may have to include the `--user` option in the following
(to install to your local Python site packages, usually located in `~/.local`).

Inside the PET folder, run

```sh
python3 -m pip install -e .
```

- The dot is needed to point to the current directory.
- The `-e` option installs PET such that changes to it take effect immediately
  (without re-installation).

## Examples

PET needs to be setup with a configuration file. See the example folder for inspiration.the
