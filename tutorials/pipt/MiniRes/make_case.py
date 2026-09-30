"""Make this tutorial's synthetic case: a truth, and observations of it.

A twin experiment: the "true" permeability is itself a draw from the prior that
``CONFIG_ESMDA.toml`` specifies, MiniRes is run on it, and the well reports are
perturbed by the observation noise. Nothing external is needed -- which is the
point of this tutorial's simulator.

    python make_case.py
"""

from copy import deepcopy

import numpy as np
import pandas as pd

from input_output import read_config
from misc.sampling import random_stream
from misc.structures.layout import StateLayout
from pipt.misc_tools.extract_tools import extract_prior_info
from simulator.minires import MiniRes

SIGMA = 0.05  # observation noise (absolute, on both the water cut and the oil rate)

kwda, kwsim, kwens = read_config.read("CONFIG_ESMDA.toml")
nx, ny, _ = kwens["prior_permx"]["grid"]

# The prior's mean: a flat log-permeability field. Also fixes the state's length.
np.savez("priormean.npz", permx=np.zeros(nx * ny))

# The truth: one draw from that prior
truth, _ = StateLayout.from_prior_info(
    extract_prior_info(deepcopy(kwens)), 1, rng=random_stream(7), save=False
)
truth = truth[:, 0]
np.savez("truth.npz", permx=truth)

# The observations: MiniRes on the truth, plus noise
records = MiniRes(kwsim).run_fwd_sim({"permx": truth}, 0)
report = kwsim["reportpoint"]

data = pd.DataFrame([{k: float(v[0]) for k, v in row.items()} for row in records], index=report)
data += SIGMA * np.random.default_rng(42).standard_normal(data.shape)
data.index.name = kwsim["reporttype"]
data.to_csv("data.csv")

var = pd.DataFrame({col: [f"['abs', {SIGMA**2}]"] * len(report) for col in data.columns}, index=report)
var.index.name = data.index.name
var.to_csv("var.csv")

print(f"Wrote priormean.npz, truth.npz, data.csv, var.csv ({data.shape[0]} report points"
      f" x {data.shape[1]} data types)")
