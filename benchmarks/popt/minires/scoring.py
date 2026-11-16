"""What a run is scored on: the schedule it ended at, simulated once more."""

import numpy as np

from problem import INTERVALS, evaluate

FEASIBLE = 1e-3  # a constraint is taken as met when violated by less than this fraction of its limit


def scores(sim, x: np.ndarray) -> dict:
    """Scores of the schedule ``x`` (all controls, in the units of the simulator).

    - ``npv``: the net present value, in 1e7 USD;
    - ``water``: the largest excess of an interval's water production over the cap, as a fraction of the cap;
    - ``capacity``: the largest excess of an interval's injection over the capacity, as a fraction of it;
    - ``feasible``: whether both are below :data:`FEASIBLE`.
    """
    f, c = evaluate(sim, x)
    water = float(max(0.0, -c[:INTERVALS].min()))
    capacity = float(max(0.0, -c[INTERVALS:].min()))
    return {"npv": -f, "water": water, "capacity": capacity, "feasible": float(max(water, capacity) <= FEASIBLE)}
