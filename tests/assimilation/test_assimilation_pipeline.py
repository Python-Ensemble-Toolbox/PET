"""
Integration tests for Data Assimilation workflows using the Van der Pol oscillator.

Tested algorithms:
- ESMDA (Ensemble Smoother with Multiple Data Assimilation)
- LM-EnRML (Levenberg-Marquardt Ensemble Randomized Maximum Likelihood)
- GN-EnRML (Gauss-Newton Ensemble Randomized Maximum Likelihood)

These tests validate multiple ensemble-based assimilation algorithms by
verifying:
1. Reduction in data misfit
2. Improvement of inferred parameters relative to prior
"""

import os
from pathlib import Path

import yaml
import pytest
import numpy as np
import pandas as pd

from simulator.vanderpol import VanDerPolOscillator, _integrate
from input_output import read_config
from pipt import ES, ESMDA, EnKF, GNEnRML, LMEnRML


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def num_cores():
    """
    Return number of CPU cores for parallel execution.

    Uses half of available cores, with a minimum of 1.
    """
    return max(os.cpu_count() // 2, 1)


# ----------------------------------------------------------------------
# Test utilities
# ----------------------------------------------------------------------

#: Members in the synthetic prior and in the config that consumes it. The
#: quality thresholds in assert_assimilation_quality hold at this size; at
#: 300 the posterior mean of mu misses the 0.2 x prior-error bar.
ENSEMBLE_SIZE = 1000


def setup_synthetic_case(seed: int = 12345):
    """
    Create synthetic prior ensemble and observation data.

    Outputs:
        - prior_ensemble.npz
        - true_data.pkl
        - var.pkl
    """
    rng = np.random.default_rng(seed)

    # True parameters
    x1_true, x2_true, mu_true = 1.0, 0.0, 1.0

    # Prior ensemble
    ne = ENSEMBLE_SIZE
    X1 = 0.05 + 0.1 * rng.standard_normal(ne)
    X2 = 0.05 + 0.1 * rng.standard_normal(ne)
    MU = 1.5 + 0.5 * rng.standard_normal(ne)

    np.savez(
        "prior_ensemble.npz",
        x1=X1[np.newaxis, :],
        x2=X2[np.newaxis, :],
        mu=MU[np.newaxis, :],
    )

    # Time configuration
    time_steps = np.arange(0, 16, dtype=float)
    report_points = np.arange(1, 16)

    # True simulation
    result = _integrate(x1_true, x2_true, mu_true, time_steps,
                        atol=1e-5, rtol=1e-5)

    # Observations (with noise)
    sigma = 0.1
    observations = result[report_points, 0] + sigma * rng.standard_normal(len(report_points))

    # Store observations
    df_obs = pd.DataFrame({"x1": observations}, index=report_points)
    df_obs.index.name = "steps"
    df_obs.to_pickle("true_data.pkl")

    # Store variance (PET format)
    variance = sigma ** 2
    df_var = pd.DataFrame(
        {"x1": [f"['abs', {variance}]" for _ in report_points]},
        index=report_points,
    )
    df_var.index.name = "steps"
    df_var.to_pickle("var.pkl")


def create_config_file(filename: str, data_assimilation_cfg: dict, parallel_runs: int):
    """
    Write YAML configuration file for data assimilation run.
    """
    ensemble_cfg = {
        "ne": ENSEMBLE_SIZE,
        "state": ["x1", "x2", "mu"],
        "importstate": "prior_ensemble.npz",
        "prior_x1": {"var": 1.0},
        "prior_x2": {"var": 1.0},
        "prior_mu": {"var": 1.0},
    }

    simulator_cfg = {
        "reporttype": "steps",
        "reportpoints": list(range(1, 16)),
        "datatype": ["x1"],
        "parallel": parallel_runs,
        "compute_adjoints": False,
    }

    config = {
        "ensemble": ensemble_cfg,
        "dataassim": data_assimilation_cfg,
        "simulator": simulator_cfg,
    }

    with open(f"{filename}.yaml", "w") as f:
        yaml.dump(config, f)


def compute_data_misfit(observed, predicted, cov):
    """
    Compute normalized data misfit across ensemble members.
    """
    n_ens = predicted.shape[1]
    misfit = 0.0

    for i in range(n_ens):
        residual = predicted[:, i] - observed
        misfit += (residual.T @ np.linalg.solve(cov, residual)) / n_ens

    return float(np.squeeze(misfit))


#: The public class per algorithm.
SCHEME_CLASSES = {
    "enkf": EnKF,
    "es": ES,
    "esmda": ESMDA,
    "lmenrml": LMEnRML,
    "gnenrml": GNEnRML,
}


def run_case(config_file: str):
    """Initialize and run assimilation given a config file.

    Constructs the scheme class directly, as a user would. The scheme itself is
    returned rather than only the result, because the assertions here read
    `vecObs`, `pred_data` and `cov_data`, which the result object does not
    carry.
    """
    cfg_da, cfg_sim, cfg_ens = read_config.read(config_file)

    scheme = SCHEME_CLASSES[cfg_da["scheme"]](
        cfg_da,
        cfg_ens,
        VanDerPolOscillator(cfg_sim),
    )

    scheme.run_assimilation()
    return scheme


def assert_assimilation_quality(ensemble, misfit_threshold=60.0):
    """
    Validate assimilation performance:
        - Data misfit is below threshold
        - Parameter estimate improves
    """
    # Data misfit check
    dm = compute_data_misfit(
        observed=ensemble.vecObs,
        predicted=ensemble.pred_data.matrix,
        cov=np.diag(ensemble.cov_data),
    )

    assert dm < misfit_threshold, f"Data mismatch too high: {dm:.2f} >= {misfit_threshold}"

    # Parameter improvement (mu)
    mu_true = 1.0
    mu_prior = ensemble.prior_enX[2, :].mean()
    mu_post = ensemble.enX[2, :].mean()

    prior_error = abs(mu_prior - mu_true)
    post_error = abs(mu_post - mu_true)

    assert post_error < 0.2 * prior_error, (
        f"Insufficient parameter improvement: "
        f"{post_error:.3f} >= 0.2 * {prior_error:.3f}"
    )


def assert_savedata_files(scheme, expected):
    """Every saved iteration file carries every requested variable.

    Iteration 0 is the interesting one. Its file is written from
    ``after_prior_forecast``, and the schemes used to compute the prior misfit
    inside their first ``calc_analysis`` -- which runs later -- so
    ``ensemble_misfit`` was silently dropped from step 0 with a printed
    "Cannot save ... because it is a local variable!" and no failure.
    """
    folder = Path(scheme.save_folder)
    saved = sorted(folder.glob("assimilation_result_*.npz"))
    assert saved, f"no savedata files written to {folder}"

    for path in saved:
        with np.load(path, allow_pickle=True) as archive:
            keys = set(archive.files)
        missing = [name for name in expected if name not in keys]
        assert not missing, f"{path.name} is missing {missing}; has {sorted(keys)}"


def prepare_test_environment(tmp_path: Path, folder_name: str):
    """
    Create isolated test directory and initialize synthetic data.
    """
    path = tmp_path / folder_name
    path.mkdir()
    os.chdir(path)
    setup_synthetic_case(seed=12345)
    # The schemes perturb observations from the global numpy state; seed it so
    # the quality thresholds below are checked against the same run every time.
    np.random.seed(12345)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_esmda_approx(tmp_path, num_cores):
    """Test ESMDA (approx analysis)."""
    prepare_test_environment(tmp_path, "esmda_test")

    da_cfg = {
        "scheme": "esmda",
        "analysis": "approx",
        "mda": {
            "tot_assim_steps": 8,
            "inflation_param": 8 * [8],
        },
        "energy": 0.99,
        "obsname": "steps",
        "data": "true_data.pkl",
        "datavar": "var.pkl",
        "save_folder": "results",
        "savedata": ["state", "pred_data", "ensemble_misfit"],
    }
    create_config_file("config_esmda", da_cfg, num_cores)

    ensemble = run_case("config_esmda.yaml")
    assert_assimilation_quality(ensemble)
    assert_savedata_files(ensemble, ["pred_data", "ensemble_misfit", "x1", "x2", "mu"])


@pytest.mark.slow
def test_lm_enrml_approx(tmp_path, num_cores):
    """Test LM-EnRML (approx analysis)."""
    prepare_test_environment(tmp_path, "lm_enrml_test")

    da_cfg = {
        "scheme": "lmenrml",
        "analysis": "approx",
        "iteration": {
            "max_iter": 8,
            "lambda": 10,
            "lambda_factor": 5,
            "trunc_energy": 0.99,
        },
        "energy": 0.99,
        "obsname": "steps",
        "data": "true_data.pkl",
        "datavar": "var.pkl",
        "save_folder": "results",
        "savedata": ["state", "pred_data", "ensemble_misfit"],
    }
    create_config_file("config_lm_enrml", da_cfg, num_cores)

    ensemble = run_case("config_lm_enrml.yaml")
    assert_assimilation_quality(ensemble)
    assert_savedata_files(ensemble, ["pred_data", "ensemble_misfit", "x1", "x2", "mu"])


@pytest.mark.slow
def test_gn_enrml_approx(tmp_path, num_cores):
    """Test GN-EnRML (approx analysis)."""
    prepare_test_environment(tmp_path, "gn_enrml_test")

    da_cfg = {
        "scheme": "gnenrml",
        "analysis": "approx",
        "iteration": {
            "max_iter": 8,
            "gamma": 0.5,
            "gamma_factor": 5,
            "trunc_energy": 0.99,
        },
        "energy": 0.99,
        "obsname": "steps",
        "data": "true_data.pkl",
        "datavar": "var.pkl",
        "save_folder": "results",
        "savedata": ["state", "pred_data", "ensemble_misfit"],
    }
    create_config_file("config_gn_enrml", da_cfg, num_cores)

    ensemble = run_case("config_gn_enrml.yaml")
    assert_assimilation_quality(ensemble)
    assert_savedata_files(ensemble, ["pred_data", "ensemble_misfit", "x1", "x2", "mu"])
