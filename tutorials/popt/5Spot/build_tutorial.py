"""Generate docs/tutorials/popt/5Spot/tutorial_popt.ipynb.

Written as a generator rather than by hand-editing JSON so the cell sources stay
readable and reviewable in one place.
"""

import json
from pathlib import Path

OUT = Path("docs/tutorials/popt/5Spot/tutorial_popt.ipynb")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = []

# ----------------------------------------------------------------------
cells.append(md("""\
# Tutorial for running the Python Optimization Toolbox (POPT)

<font size=4em>As an illustrative example we choose a 2D five-spot pattern: one producer at the centre of the field and four (water) injectors, one at each corner. The figure below shows the permeability field and the well positions. The grid is 50x50, and the porosity is 0.2. The optimization problem is to find the water injection rate for each injector, one value per year of the eight-year production period, that maximizes the net present value (NPV).

<img src="../permx.png" alt="drawing" width="500"/>
<br>
<font size=4em>POPT mirrors PIPT: an *ensemble* object owns the control perturbations and the gradient estimate, and an *optimizer* owns its own iteration loop. The first step is to load the necessary external and local modules.
"""))

cells.append(code("""\
# Import global modules
import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import local modules
from input_output import read_config                # the config reader
from popt.ensembles import GaussianEnsemble          # control perturbations and gradients
from popt.optimization_methods import LineSearch     # the optimizer; it owns its own loop
from subsurface.multphaseflow.opm import flow        # the simulator we want to use
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Set the random seed:
"""))

cells.append(code("""\
np.random.seed(10_08_1997)
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Each simulator call runs in its own <span style="font-family:Courier;">En_&lt;member&gt;</span> folder, which it creates with <span style="font-family:Courier;">os.mkdir</span> &mdash; so a folder left behind by an interrupted run makes the next one fail with <span style="font-family:Courier;">FileExistsError</span>. PET clears them when an ensemble is constructed, but not between runs, so we define a helper and call it before each optimization. That keeps the run cells safe to re-execute on their own.
"""))

cells.append(code("""\
def clean_run_folders(*result_folders):
    \"\"\"Remove simulator scratch folders, and any results being replaced.\"\"\"
    for folder in glob('En_*'):
        shutil.rmtree(folder, ignore_errors=True)
    for folder in result_folders:
        shutil.rmtree(folder, ignore_errors=True)
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Read the input file. In this tutorial the input file is written as a .toml file, and consists of three main keys: <span style="font-family:Courier;">ensemble</span>, <span style="font-family:Courier;">optim</span> and <span style="font-family:Courier;">simulator</span>. The first contains keys related to the ensemble of control perturbations, the second the options for the optimization algorithm, and the third the options for the forward simulation model.

<font size=4em>The <span style="font-family:Courier;">ensemble.controls</span> table lists the control variables directly: each entry names one .mako placeholder, together with its mean, standard deviation and bounds. This is a simpler alternative to PIPT's <span style="font-family:Courier;">prior_&lt;name&gt;</span> tables &mdash; there is no need for a separate <span style="font-family:Courier;">state</span> list, since the keys of <span style="font-family:Courier;">controls</span> already give the names.
"""))

cells.append(code("""\
!cat init_optim.toml
ko, kf, ke = read_config.read('init_optim.toml')
# ko  -->  Optimization settings
# kf  -->  Simulator settings
# ke  -->  Ensemble settings
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Set the initial controls. The filename given as <span style="font-family:Courier;">mean</span> in the input file above must exist before the ensemble is built, and its arrays must match the .mako placeholders <span style="font-family:Courier;">rate_inj1</span>&ndash;<span style="font-family:Courier;">rate_inj4</span>. Each array holds one rate per year of the eight-year schedule, so all four injectors start at a flat 200 Sm3/day.
"""))

cells.append(code("""\
rate = 8 * [200]
np.savez(
    'initrates.npz',
    rate_inj1=rate,
    rate_inj2=rate,
    rate_inj3=rate,
    rate_inj4=rate,
)
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Define the objective function. This is the one piece POPT does not supply: you hand it any callable that takes the simulated data and returns a scalar to be **minimized**. Here it is the discounted net present value, with the economic constants read from the <span style="font-family:Courier;">npv_const</span> block of the input file.

<font size=4em>Note the <span style="font-family:Courier;">obj_scaling</span> of -1e9: the negative sign turns maximizing NPV into a minimization, and the 1e9 puts the value in billions so the optimizer works on a sensible scale.
"""))

cells.append(code("""\
DEFAULT_ECON = {
    'wop': 400.0,  # Oil price: $/Sm3
    'wgp': 0.4,    # Gas price: $/Sm3
    'wwp': 20.0,   # Cost of water production per unit volume
    'wwi': 10.0,   # Cost of water injection per unit volume
    'disc': 0.08,  # Discount rate per year
}


def npv(pred_data: pd.DataFrame, **kwargs):
    \"\"\"Discounted net present value of one simulated production profile.\"\"\"
    # Economic parameters, from the config's npv_const block if present
    input_dict = kwargs.get('input_dict', {})
    econ = dict(input_dict.get('npv_const', DEFAULT_ECON))
    scaling_factor = econ.pop('obj_scaling', 1.0)

    # Incremental volumes per report step
    vol_oil = pred_data['FOPT'].diff()
    vol_gas = pred_data['FGPT'].diff()
    vol_water_prod = pred_data['FWPT'].diff()
    vol_water_inj = pred_data['FWIT'].diff()

    # Time in years since the start of the run
    time_index = pred_data.index.to_numpy()
    years = (time_index - time_index[0]) / np.timedelta64(365, 'D')

    # Revenue, cost, and discounting
    revenue = vol_oil * econ['wop'] + vol_gas * econ['wgp']
    operating_cost = vol_water_prod * econ['wwp'] + vol_water_inj * econ['wwi']
    discount_factor = (1.0 + econ['disc']) ** years

    return ((revenue - operating_cost) / discount_factor).sum() / scaling_factor
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Initialize the ensemble with the ensemble keys, the simulator and the objective function, then extract the initial control vector (<span style="font-family:Courier;">x0</span>), its covariance (<span style="font-family:Courier;">cov</span>) and the bounds. The ensemble is what turns a non-differentiable simulator into something gradient-based methods can use: it perturbs the controls, runs the simulator on each perturbation, and forms an ensemble approximation of the gradient.
"""))

cells.append(code("""\
sim = flow(kf)
ensemble = GaussianEnsemble(ke, sim, npv)

x0 = ensemble.get_state()
cov = ensemble.get_cov()
bounds = ensemble.get_bounds()

print(f'controls: {x0}')
print(f'bounds:   {bounds[0]} ... (x{len(bounds)})')
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Run the optimization with <span style="font-family:Courier;">LineSearch</span>, using BFGS as the search direction. During the run, useful information is written to the screen and to a log file. As in PIPT, there are two ways to do this &mdash; the class-level shortcut that constructs and runs in one call, or an instance you keep and drive yourself.

<font size=4em>The other supported <span style="font-family:Courier;">method</span> values are <span style="font-family:Courier;">'GD'</span> (steepest descent, needs only the gradient) and <span style="font-family:Courier;">'Newton-CG'</span> (needs a Hessian as well, passed as <span style="font-family:Courier;">hess=ensemble.hessian</span>). BFGS builds a curvature estimate from successive gradients, so it needs no Hessian.
"""))

cells.append(code("""\
clean_run_folders(ko.get('savefolder', 'Results'))

# There are two ways to run the optimization:

# Option 1: the class-level shortcut, when the optimizer object is not needed afterwards
res_bfgs = LineSearch.minimize(
    x0=x0,
    fun=ensemble.function,
    method='BFGS',
    jac=ensemble.gradient,
    args=(cov,),
    bounds=bounds,
    **ko,
)

# Option 2: keep the optimizer, then run it
# ls = LineSearch(x0=x0, fun=ensemble.function, method='BFGS', jac=ensemble.gradient,
#                 args=(cov,), bounds=bounds, **ko)
# res_bfgs = ls.run_optimization()

print(f'NPV: {-res_bfgs.fun:.4f} billion $ after {res_bfgs.nit} iterations')
print(res_bfgs)
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Plot the objective function against iteration. The optimizer writes one file per iteration, <span style="font-family:Courier;">optimize_result_{i}.npz</span>, into the folder named by the <span style="font-family:Courier;">savefolder</span> key &mdash; the counterpart of PIPT's <span style="font-family:Courier;">assimilation_result_{i}.npz</span>. Saving happens only when <span style="font-family:Courier;">saveit</span> is true.
"""))

cells.append(code("""\
def read_npv_history(folder):
    \"\"\"Collect the NPV, in million $, at each iteration from the saved result files.\"\"\"
    values = []
    it = 0
    while True:
        file = f'{folder}/optimize_result_{it}.npz'
        if not os.path.exists(file):
            break
        info = np.load(file)
        # 'fun' is the objective value at that iteration, in billion $ with a flipped sign
        # (obj_scaling = -1e9). Undo both to get NPV in million $.
        values.append(-1000.0 * float(np.mean(info['fun'])))
        it += 1
    return values


npv_bfgs = read_npv_history(ko.get('savefolder', 'Results'))

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor='white')
ax.plot(npv_bfgs, 's-', color='#4C78A8', linewidth=2, markersize=7, label='BFGS')
ax.set_xlabel('Iteration no.', size=13)
ax.set_ylabel('NPV [million $]', size=13)
ax.set_title('Objective function', size=14)
ax.set_xticks(range(len(npv_bfgs)))
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>The same problem with a different search direction. <span style="font-family:Courier;">method='GD'</span> takes a plain steepest-descent step instead of the BFGS quasi-Newton direction &mdash; simpler, and it does not accumulate curvature information across iterations, so its step sizes are driven entirely by <span style="font-family:Courier;">step_size_adapt</span> rather than an approximated Hessian. Everything else &mdash; the ensemble, the objective, the bounds &mdash; is reused unchanged, which is the point of keeping the optimizer separate from the ensemble.

<font size=4em>Both runs start from the same <span style="font-family:Courier;">x0</span> captured above, so the comparison is fair. Note that <span style="font-family:Courier;">ensemble.get_state()</span> would <em>not</em> do here: it returns the ensemble's current controls, which the first optimization has already moved.
"""))

cells.append(code("""\
from copy import deepcopy

ko_gd = deepcopy(ko)
ko_gd['savefolder'] = 'Results_gd'   # keep the BFGS files for the comparison below

clean_run_folders(ko_gd['savefolder'])

res_gd = LineSearch.minimize(
    x0=x0,
    fun=ensemble.function,
    method='GD',
    jac=ensemble.gradient,
    args=(cov,),
    bounds=bounds,
    **ko_gd,
)

print(f'NPV: {-res_gd.fun:.4f} billion $ after {res_gd.nit} iterations')
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
<font size=4em>Compare the two:
"""))

cells.append(code("""\
npv_gd = read_npv_history(ko_gd['savefolder'])

fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor='white')
ax.plot(npv_bfgs, 's-', color='#4C78A8', linewidth=2, markersize=7, label='BFGS')
ax.plot(npv_gd, 'o--', color='#E45756', linewidth=2, markersize=7, label='GD')
ax.set_xlabel('Iteration no.', size=13)
ax.set_ylabel('NPV [million $]', size=13)
ax.set_title('BFGS vs. GD', size=14)
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
## Setting up the .mako file
<font size=4em>The optimization relies on a .mako file for writing the current control variables to the flow simulator input. In this case, the flow simulator is opm-flow [opm-projects.org](opm-projects.org), and the input file is provided as a text file (.DATA file). Once a year, the .mako file writes a <span style="font-family:Courier;">WCONINJE</span> block that sets that year's rate for each injector:

    WCONINJE
    INJ1 WATER OPEN RATE ${rate_inj1[index]} 1* 500.0  /
    INJ2 WATER OPEN RATE ${rate_inj2[index]} 1* 500.0  /
    INJ3 WATER OPEN RATE ${rate_inj3[index]} 1* 500.0  /
    INJ4 WATER OPEN RATE ${rate_inj4[index]} 1* 500.0  /
    /

<font size=4em>The names <span style="font-family:Courier;">rate_inj1</span>&ndash;<span style="font-family:Courier;">rate_inj4</span> are the keys of the <span style="font-family:Courier;">ensemble.controls</span> table in the input file, so the .mako placeholders and the config have to agree. The producer's bottom-hole pressure is fixed for the whole run and is not a control.
"""))

# ----------------------------------------------------------------------
cells.append(md("""\
## Running locally

<font size=4em>It is recommended to run the notebook from a virtual environment. Follow these steps to run this notebook on your own computer:

<font size=4em>*Step 1: Create virtual environment as normal*

    python3 -m venv pet_venv

<font size=4em>Then activate the environment using:

    source pet_venv/bin/activate

<font size=4em>*Step 2: Install Jupyter Notebook into virtual environment*

    python3 -m pip install ipykernel

<font size=4em>*Step 3: Install PET in the virtual environment, see [PET installation](https://github.com/Python-Ensemble-Toolbox/PET)*

<font size=4em>*Step 4: Allow Jupyter access to the kernel within the virtual environment*

    python3 -m ipykernel install --user --name=pet_venv

<font size=4em>Start jupyter notebook, and load tutorial_popt.ipynb (this file). On the jupyter notebook toolbar, select 'Kernel' and 'Change Kernel'.
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

OUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {OUT} with {len(cells)} cells")
