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
