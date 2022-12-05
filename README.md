<h1 align="center">
<img src="https://github.com/Python-Ensemble-Toolbox/.github/blob/main/profile/pictures/logo.png" width="300">
</h1><br>

PET is a toolbox for ensemble based Data-Assimilation developed and maintained by the data-assimilation and optimization group at NORCE Norwegian Research Centre AS.


## Installation

Two tips before installation: use virtual environments and add your SSH key to Github! You can read more about virtual environments [here](https://docs.python.org/3/tutorial/venv.html) and adding SSH key [here](https://help.github.com/en/articles/adding-a-new-ssh-key-to-your-github-account).

To install PET, first clone the repo with (assuming you have added the SSH key)

```
git clone git@github.com:Python-Ensemble-Toolbox/PET.git PET
```

Make sure you have the latest version of pip and setuptools:

```
python3 -m pip install --upgrade pip setuptools
```

Inside the PET folder, run

```
python3 -m pip install -e .
```

Note that the ```-e``` option installs the package in a way you can make changes to the source code and have the changes take effect immediately (no re-installation required). Alternatively, you can install with ``` python3 -m pip install .``` where it installs it like a third-party package. Note also that if you do not install PIPT inside a virtual environment, you may have to add a ```--user``` option to install to your local Python site packages (usually located in ```~/.local```).

If you have not added your SSH key to Github, you might get problems installing the ```pyresito``` package. If so, you have to install it separately and ensure that they are a part of the ```PYTHONPATH```.

## Examples

PET needs to be setup with a configuration file. See the example folder for inspiration.

