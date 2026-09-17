"""
Integration test for 1D linear model with LM-EnRML assimilation.
"""

import os
import numpy as np

from misc.structures import PETDataFrame
from simulator.simple_models import lin_1d
from pipt import LMEnRML


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATE_SIZE = 150

CFG_ENS = {
    "ne": 250,
    "state": "x",
    "prior_x": {
        "vario": "sph",
        "mean": [0.0] * STATE_SIZE,
        "var": 1.0,
        "range": 20.0,
        "aniso": 1.0,
        "angle": 0.0,
        "grid": [STATE_SIZE, 1],
    },
}

CFG_DA = {
    "scheme": "lmenrml",
    "analysis": "full",
    "energy": 0.95,
    "obsname": "position",
    "data": "true_data.pkl",
    "datavar": "var.pkl",
    "iteration": {
        # Four updates. The expected numbers below were pinned when `max_iter`
        # counted the prior forecast as iteration 0, i.e. with `max_iter: 5`.
        "max_iter": 4,
        "data_misfit_tol": 1e-3,
        "step_tol": 0.0,
        "lambda": 50.0,
        "lambda_factor": 4.0,
        "lambda_max": 1e8,
    },
}

CFG_SIM = {
    "reporttype": "position",
    "reportpoint": list(range(5, 150, 5)),
    "datatype": ["value"],
    "parallel": 4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_synthetic_data():
    """Generate synthetic observations and associated variances."""
    np.random.seed(10)

    simulator = lin_1d(CFG_SIM)
    simulator.setup_fwd_run()

    # Generate a random state realization
    state = {
        "x": np.random.multivariate_normal(
            mean=np.zeros(STATE_SIZE),
            cov=np.eye(STATE_SIZE),
        )
    }

    # Forward simulation
    prediction = simulator.run_fwd_sim(state, 0)
    prediction = PETDataFrame.from_records(
        prediction,
        index=CFG_SIM["reportpoint"]
    )

    # Construct observation data and variance
    data = prediction.copy()
    data_var = prediction.copy()

    for column in data.columns:
        data[column] = data[column].apply(np.squeeze)
        data_var[column] = data_var[column].apply(lambda _: ["abs", 1.0])

    data.to_pickle("true_data.pkl")
    data_var.to_pickle("var.pkl")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_lin_1d(tmp_path):
    """
    End-to-end test of the LM-EnRML assimilation workflow.
    """
    # --- Setup temporary working directory
    workdir = tmp_path / "lin_1d_test"
    workdir.mkdir()
    os.chdir(workdir)

    # --- Generate synthetic dataset
    setup_synthetic_data()

    # --- Initialize and run
    np.random.seed(10)
    result = LMEnRML.assimilate(CFG_DA, CFG_ENS, lin_1d(CFG_SIM), analysis="full")

    # --- Validate results. `result.x` is the posterior state ensemble.
    ensemble_mean = result.x.mean(axis=-1)
    # Regenerated 2026-09-07 for the truncSVD energy-rank change (see CHANGELOG).
    expected = np.array([
        -0.08340785,
         0.00542748,
        -0.04776080,
         0.47335038,
         0.45950017,
         0.37760764,
    ])
    result = ensemble_mean[[1, 2, 3, -3, -2, -1]]
    np.testing.assert_array_almost_equal(result, expected, decimal=5)
