"""The cases: one waterflood, at two resolutions, observed over its history and scored on its forecast.

The reservoir is a 1000 x 1000 x 20 m box in metric units (m, day, bar, mD, cP), with
one injector at the centre on rate, four producers towards the corners on bottom-hole
pressure, 40 report steps of 60 days. The producers' rates, and when they water out, are
therefore outcomes of the permeability around them.

The unknowns are the log-permeability field and two scalars of the fluid, the residual oil
saturation ``sor`` and the oil viscosity ``vo``. Each twin experiment draws its truth from
the same prior as the ensemble, simulates it to the end, and perturbs the history part into
observations; the forecast part is what the posterior is scored against, unseen.
"""

from dataclasses import dataclass

import numpy as np

from misc.sampling import random_stream
from misc.structures.layout import StateLayout
from pipt.misc_tools.extract_tools import extract_prior_info
from simulator.minires import MiniRes

LX = 1000.0  # extent [m]
DZ = 20.0  # thickness [m]: MiniRes is areal, so the rates are per unit thickness

DT = 60.0  # report step [days]
NSTEPS = 40
HISTORY = 16  # the observed steps, of which every fourth is assimilated
OBS_STEPS = list(range(4, HISTORY + 1, 4))
FORECAST_STEPS = list(range(HISTORY + 1, NSTEPS + 1))

PRODUCERS = ["PROD1", "PROD2", "PROD3", "PROD4"]
INJECTOR = "INJ"
DATATYPES = [f"WWCT:{w}" for w in PRODUCERS] + [f"WOPR:{w}" for w in PRODUCERS] + [f"WBHP:{INJECTOR}"]

# Observation error, as (relative, absolute): the larger applies.
OBS_ERROR = {"WWCT": (0.05, 0.02), "WOPR": (0.10, 5.0), "WBHP": (0.01, 1.0)}

# The prior. log K about log(200 mD), sd 0.8, spherical variogram of range RANGE metres.
LOGK_MEAN, LOGK_SD, RANGE = np.log(200.0), 0.8, 300.0
SCALARS = {
    #       mean   sd     limits
    "sor": (0.22, 0.04, [0.05, 0.40]),
    "vo": (5.0, 1.0, [1.5, 10.0]),
}


@dataclass(frozen=True)
class Case:
    name: str
    n: int  # cells per side
    truths: int  # the number of twin experiments run by default

    @property
    def dx(self):
        return LX / self.n

    def model(self) -> dict:
        """``[simulator.model]``, less what the state supplies (K, and the scalars' values)."""
        wells = [{"name": INJECTOR, "xy": [LX / 2, LX / 2], "rate": 1000.0 / DZ, "rw": 0.15}]
        for name, (fx, fy) in zip(PRODUCERS, [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]):
            wells.append({"name": name, "xy": [fx * LX, fy * LX], "bhp": 250.0, "rw": 0.15})
        return {
            "Lx": LX, "Ly": LX, "Nx": self.n, "Ny": self.n,
            "por": 0.2, "ct": 1e-5, "cdarcy": 0.008527,
            "fluid": {"vw": 1.0, "vo": SCALARS["vo"][0], "swc": 0.2, "sor": SCALARS["sor"][0]},
            "wells": wells,
        }

    def simulator(self, reportpoint) -> dict:
        """The ``[simulator]`` section, reporting at ``reportpoint``."""
        return {
            "dt": DT, "reporttype": "steps", "reportpoint": list(reportpoint),
            "datatype": DATATYPES, "parallel": 1, "p0": 300.0, "s0": 0.2,
            "model": self.model(),
        }

    def ensemble(self, ne: int, seed: int, mean_file: str) -> dict:
        """The ``[ensemble]`` section; ``mean_file`` is where :meth:`write_prior_mean` put the field's mean."""
        ens = {
            "ne": ne, "seed": seed, "disable_tqdm": True, "save_prior": False,
            "state": ["permx", *SCALARS],
            "prior_permx": {
                "vario": "sph", "mean": mean_file, "var": LOGK_SD**2,
                "range": RANGE / self.dx, "aniso": 1.0, "angle": 0.0, "grid": [self.n, self.n, 1],
            },
        }
        for name, (mean, sd, limits) in SCALARS.items():
            ens[f"prior_{name}"] = {"mean": mean, "var": sd**2, "limits": limits}
        return ens

    def distance_entries(self, radius: float) -> list:
        """Distance-localization entries: a taper of ``radius`` metres on the field, at each datum's well."""
        cells = {w["name"]: [int(w["xy"][0] // self.dx), int(w["xy"][1] // self.dx)] for w in self.model()["wells"]}
        return [
            {"x": x, "y": y, "z": 0, "radius": int(round(radius / self.dx)), "z_range": ":", "aniso": 1.0,
             "rotation": 0.0, "data_type": d, "time": "*", "param": "permx"}
            for d in DATATYPES
            for x, y in [cells[d.split(":")[1]]]
        ]

    def write_prior_mean(self, path: str) -> None:
        np.savez(path, permx=np.full(self.n**2, LOGK_MEAN))

    def run(self, state: dict) -> np.ndarray:
        """Simulate one member to the end: ``(NSTEPS, len(DATATYPES))``, or ``nan`` if it failed."""
        sim = MiniRes(self.simulator(range(1, NSTEPS + 1)))
        records = sim.run_fwd_sim(state, 0)
        if records is False:
            return np.full((NSTEPS, len(DATATYPES)), np.nan)
        return np.array([[row[d][0] for d in DATATYPES] for row in records])


CASES = {c.name: c for c in [Case("coarse", 25, truths=20), Case("fine", 50, truths=8)]}


@dataclass
class Twin:
    """One twin experiment: a truth, its full response, and the observations of its history."""

    truth: dict  # the true state, {variable: values}
    response: np.ndarray  # (NSTEPS, len(DATATYPES)), noiseless
    obs: np.ndarray  # (len(OBS_STEPS), len(DATATYPES)), perturbed
    std: np.ndarray  # the observation error, same shape


def twin(case: Case, seed: int, mean_file: str) -> Twin:
    """Draw a truth from the prior, simulate it, and perturb the history into observations."""
    keys = case.ensemble(1, seed, mean_file)
    enX, layout = StateLayout.from_prior_info(extract_prior_info(keys), 1, rng=random_stream(seed), save=False)
    truth = {k: v[:, 0] for k, v in layout.to_dict(enX).items()}
    response = case.run(truth)

    exact = response[np.array(OBS_STEPS) - 1]
    rel, ab = np.array([OBS_ERROR[d.split(":")[0]] for d in DATATYPES]).T
    std = np.maximum(rel * np.abs(exact), ab)
    obs = exact + std * np.random.default_rng(seed).standard_normal(exact.shape)
    # A water cut outside [0, 1], or a negative rate, would be unphysical
    quantity = [d.split(":")[0] for d in DATATYPES]
    lower = np.where(np.isin(quantity, ["WWCT", "WOPR"]), 0.0, -np.inf)
    upper = np.where(np.isin(quantity, ["WWCT"]), 1.0, np.inf)
    return Twin(truth, response, obs.clip(lower, upper), std)
