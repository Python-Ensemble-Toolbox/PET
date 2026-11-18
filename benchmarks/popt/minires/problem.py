"""The problem: a waterflood's injection rates, chosen for net present value under two facility limits.

The reservoir is that of ``benchmarks/pipt/minires`` turned round: a 1000 x 1000 x 20 m box in
metric units (m, day, bar, mD, cP), with four injectors towards the corners on rate and one
producer at the centre on bottom-hole pressure, over 40 steps of 60 days. The permeability is
known (one heterogeneous field per case), so the problem is deterministic.

The controls are the injectors' rates in each of 8 intervals of 300 days, 32 numbers. They are
chosen to maximise the net present value of the oil, less the cost of the water produced and
injected, subject to

- the water produced by the field in each interval staying below what the facility can
  treat -- a nonlinear constraint, known only through the simulation -- and
- the water injected by the four wells together, in each interval, staying below what the
  pumps deliver -- a linear constraint on the controls.

Both are written ``c >= 0`` and scaled to order one.
"""

from dataclasses import dataclass

import numpy as np

from scipy.ndimage import gaussian_filter

from simulator.minires import MiniRes

LX = 1000.0  # extent [m]
DZ = 20.0  # thickness [m]: MiniRes is areal, so rates and volumes are per unit thickness
N = 25  # cells per side

DT = 60.0  # step [days]
NSTEPS = 40
INTERVALS = 8  # control intervals, of NSTEPS / INTERVALS steps each
INTERVAL_DAYS = DT * NSTEPS / INTERVALS
REPORT = list(range(NSTEPS // INTERVALS, NSTEPS + 1, NSTEPS // INTERVALS))  # the end of each interval

INJECTORS = {"INJ1": (0.1, 0.1), "INJ2": (0.9, 0.1), "INJ3": (0.1, 0.9), "INJ4": (0.9, 0.9)}
PRODUCER = "PROD"
CONTROLS = [f"rate_{w.lower()}" for w in INJECTORS]
DATATYPES = ["FOPT", "FWPT", "FWIT"]

# The permeability: log K about log(200 mD), sd 0.8, smooth over about 300 m (the statistics of benchmarks/pipt/minires).
LOGK_MEAN, LOGK_SD, RANGE = np.log(200.0), 0.8, 300.0

# Rates are in m3/day per metre of thickness
RATE_MAX = 500.0 / DZ  # one injector's upper bound
RATE_START = 150.0 / DZ  # the initial rate of every injector in every interval
RATE_SD = 0.05 * RATE_MAX  # the perturbations' standard deviation
CAPACITY = 800.0 / DZ  # the four injectors together
WATER_CAP = 300.0 / DZ  # the field's water production, mean over an interval

# Economics: USD per m3, and the yearly discount rate
OIL_PRICE, WATER_PRODUCTION_COST, WATER_INJECTION_COST, DISCOUNT = 500.0, 60.0, 20.0, 0.08
NPV_SCALE = 1e7 / DZ  # the objective is NPV / (1e7 USD)


@dataclass(frozen=True)
class Problem:
    """One case: the reservoir with the ``index``-th permeability field."""

    index: int
    seed: int

    def logk(self) -> np.ndarray:
        """The log-permeability field, ``(N, N)``: smoothed white noise, drawn with this case's seed."""
        noise = np.random.default_rng(self.seed).standard_normal((N, N))
        field = gaussian_filter(noise, sigma=RANGE / 3.0 / (LX / N), mode="reflect")
        return LOGK_MEAN + LOGK_SD * (field - field.mean()) / field.std()

    def simulator(self) -> dict:
        """The ``[simulator]`` section: the injectors' rates are the controls, the permeability is the model's."""
        wells = [{"name": w, "xy": [fx * LX, fy * LX], "rate": RATE_START, "rw": 0.15} for w, (fx, fy) in INJECTORS.items()]
        wells.append({"name": PRODUCER, "xy": [LX / 2, LX / 2], "bhp": 250.0, "rw": 0.15})
        return {
            "dt": DT, "reporttype": "steps", "reportpoint": REPORT, "datatype": DATATYPES,
            "parallel": 1, "p0": 300.0, "s0": 0.2,
            "controls": {c: {"well": w, "type": "rate"} for c, w in zip(CONTROLS, INJECTORS)},
            "model": {
                "Lx": LX, "Ly": LX, "Nx": N, "Ny": N, "K": np.exp(self.logk()),
                "por": 0.2, "ct": 1e-5, "cdarcy": 0.008527,
                "fluid": {"vw": 1.0, "vo": 5.0, "swc": 0.2, "sor": 0.22},
                "wells": wells,
            },
        }

    @staticmethod
    def ensemble(ne: int, seed: int) -> dict:
        """The ``[ensemble]`` section: the controls with their start, perturbation size and bounds."""
        control = {"mean": [RATE_START] * INTERVALS, "var": RATE_SD**2, "limits": [0.0, RATE_MAX]}
        return {"ne": ne, "seed": seed, "disable_tqdm": True, "controls": {c: dict(control) for c in CONTROLS}}


# ----------------------------------------------------------------------
# Objective and constraints
# ----------------------------------------------------------------------
def parts(cumulatives: dict, rates: np.ndarray):
    """Objective and constraints of an ensemble of schedules.

    Parameters
    ----------
    cumulatives : dict
        ``FOPT``, ``FWPT`` and ``FWIT`` at the end of each interval, each ``(INTERVALS, ne)``.
    rates : ndarray
        The controls, ``(len(CONTROLS) * INTERVALS, ne)``, injector by injector.

    Returns
    -------
    f : ndarray, ``(ne,)``
        Minus the net present value, in units of 1e7 USD.
    c : ndarray, ``(2 * INTERVALS, ne)``
        The water-production constraints, then the injection-capacity constraints; ``>= 0`` when met.
    """
    def per_interval(cumulative):
        return np.diff(cumulative, axis=0, prepend=0.0)  # nothing is produced before the first interval

    oil, water, injected = (per_interval(cumulatives[k]) for k in DATATYPES)
    years = (np.arange(INTERVALS) + 1) * INTERVAL_DAYS / 365.0
    cash = OIL_PRICE * oil - WATER_PRODUCTION_COST * water - WATER_INJECTION_COST * injected
    npv = np.sum(cash / (1.0 + DISCOUNT) ** years[:, None], axis=0)

    c_water = 1.0 - water / INTERVAL_DAYS / WATER_CAP
    total = rates.reshape(len(CONTROLS), INTERVALS, -1).sum(axis=0)
    c_capacity = 1.0 - total / CAPACITY
    return -npv / NPV_SCALE, np.vstack([c_water, c_capacity])


def from_ensemble(pred_data, state: dict):
    """:func:`parts` of what the popt ensemble hands an objective: the simulated data frame and the state."""
    cumulatives = {k: np.array([np.atleast_1d(v) for v in pred_data[k]], dtype=float) for k in DATATYPES}
    ne = cumulatives[DATATYPES[0]].shape[1]
    rates = np.vstack([np.asarray(state[c], dtype=float).reshape(INTERVALS, ne) for c in CONTROLS])
    return parts(cumulatives, rates)


def objective_with_constraints(pred_data, state=None, **kwargs):
    """For EnSQP: the objective and the constraints, side by side."""
    f, c = from_ensemble(pred_data, state)
    return {"f": f, "c_iq": c}


def objective_with_penalty(pred_data, state=None, epf=None, **kwargs):
    """For EnOpt with the exterior penalty function: the objective plus the penalty on the violations."""
    from popt.cost_functions.epf import epf as penalty_function

    f, c = from_ensemble(pred_data, state)
    if not epf:
        return f
    penalty = penalty_function(epf["r"], c_iq=c)
    epf["penalty"] = penalty
    return f + penalty


def objective_alone(pred_data, state=None, **kwargs):
    """The objective with the constraints ignored."""
    return from_ensemble(pred_data, state)[0]


def evaluate(sim: MiniRes, x: np.ndarray):
    """Objective and constraints of one schedule ``x`` (all controls, injector by injector), simulated directly."""
    x = np.asarray(x, dtype=float)
    state = {c: x[i * INTERVALS:(i + 1) * INTERVALS] for i, c in enumerate(CONTROLS)}
    records = sim.run_fwd_sim(state, 0)
    if records is False:
        return np.inf, np.full(2 * INTERVALS, np.nan)
    cumulatives = {k: np.array([[row[k][0]] for row in records]) for k in DATATYPES}
    f, c = parts(cumulatives, x[:, None])
    return float(f[0]), c[:, 0]
