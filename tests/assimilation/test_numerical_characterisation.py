"""Characterisation tests pinning the current numerical output of PIPT schemes.

Purpose
-------
The rest of the assimilation suite asserts *properties* -- misfit went down,
the parameter estimate improved -- with generous thresholds. That catches an
algorithm that is badly broken, but not one that quietly produces different
numbers. Before refactoring the assimilation mathematics, we need the stronger
statement: *these inputs still produce exactly these outputs.*

These tests run each scheme against a fixed synthetic case and compare the
posterior ensemble and the data-misfit trajectory against committed reference
values. A refactor that changes the numerics fails here with a diff, instead of
sliding under a loose threshold.

Determinism
-----------
The schemes draw observation perturbations from the *global* ``numpy.random``
state, so a run is only reproducible if that state is seeded. The rest of the
suite does not seed it, which makes those tests non-deterministic: repeated
runs of the same case were measured to differ by up to 0.37 in the posterior
state. Every test here seeds ``np.random`` explicitly and runs single-threaded
(``parallel = 1``), which was verified to give bit-identical results across
repeated runs.

Regenerating the references
---------------------------
The references are floating-point results and can legitimately shift across
BLAS implementations or numpy versions, so a mismatch is not automatically a
regression -- check whether the environment moved before concluding the code
did. To regenerate deliberately, after confirming a change is intended::

    python tests/assimilation/test_numerical_characterisation.py --regenerate

Then inspect the diff on ``characterisation_reference.npz`` before committing:
a refactor that is meant to preserve behaviour should produce *no* diff.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

import pandas as pd
import yaml

from input_output import read_config
from pipt import ES, ESMDA, EnKF, GNEnRML, LMEnRML
from simulator.vanderpol import VanDerPolOscillator, _integrate

#: The public class per algorithm. Cases run through these so the numbers pin
#: the documented entry point, not just the internals.
SCHEME_CLASSES = {
    "enkf": EnKF,
    "es": ES,
    "esmda": ESMDA,
    "lmenrml": LMEnRML,
    "gnenrml": GNEnRML,
}

REFERENCE_FILE = Path(__file__).with_name("characterisation_reference.npz")

#: Small enough to run quickly, large enough that any change to the analysis
#: mathematics moves the numbers well outside the comparison tolerance.
ENSEMBLE_SIZE = 100
GLOBAL_SEED = 42

#: Tight enough to catch a real change in the mathematics, loose enough to
#: absorb last-bit reassociation from an unrelated refactor.
RTOL = 1e-9
ATOL = 1e-11


def _write_synthetic_case(seed=12345, ne=ENSEMBLE_SIZE):
    """Create the prior ensemble and observations for the Van der Pol case.

    A local, deliberately small copy of the pipeline test's setup: the shared
    helper hardcodes a 1000-member ensemble, which made each characterisation
    case take minutes. The physics is identical, just cheaper.
    """
    rng = np.random.default_rng(seed)
    x1_true, x2_true, mu_true = 1.0, 0.0, 1.0

    np.savez(
        "prior_ensemble.npz",
        x1=(0.05 + 0.1 * rng.standard_normal(ne))[np.newaxis, :],
        x2=(0.05 + 0.1 * rng.standard_normal(ne))[np.newaxis, :],
        mu=(1.5 + 0.5 * rng.standard_normal(ne))[np.newaxis, :],
    )

    time_steps = np.arange(0, 16, dtype=float)
    report_points = np.arange(1, 16)
    truth = _integrate(x1_true, x2_true, mu_true, time_steps, atol=1e-5, rtol=1e-5)

    sigma = 0.1
    observations = truth[report_points, 0] + sigma * rng.standard_normal(len(report_points))

    df_obs = pd.DataFrame({"x1": observations}, index=report_points)
    df_obs.index.name = "steps"
    df_obs.to_pickle("true_data.pkl")

    df_var = pd.DataFrame(
        {"x1": [f"['abs', {sigma ** 2}]" for _ in report_points]}, index=report_points
    )
    df_var.index.name = "steps"
    df_var.to_pickle("var.pkl")

    return report_points


def _write_config(name, scheme, analysis, report_points, ne=ENSEMBLE_SIZE):
    """Write the YAML config for one characterisation case."""
    if scheme == "esmda":
        extra = {"mda": {"tot_assim_steps": 3, "inflation_param": 3 * [3]}}
    else:
        extra = {
            "iteration": {
                # Two updates. The reference numbers were generated when `max_iter`
                # counted the prior forecast as iteration 0, i.e. with `max_iter: 3`;
                # the meaning changed, the runs did not.
                "max_iter": 2,
                "lambda": 10,
                "lambda_factor": 5,
                "trunc_energy": 0.99,
            }
        }

    config = {
        "ensemble": {
            "ne": ne,
            "state": ["x1", "x2", "mu"],
            "importstate": "prior_ensemble.npz",
            "prior_x1": {"var": 1.0},
            "prior_x2": {"var": 1.0},
            "prior_mu": {"var": 1.0},
        },
        "dataassim": {
            "scheme": scheme,
            "analysis": analysis,
            "energy": 0.99,
            "obsname": "steps",
            "data": "true_data.pkl",
            "datavar": "var.pkl",
            "nosave": True,
            **extra,
        },
        "simulator": {
            "reporttype": "steps",
            # plain ints: yaml.dump emits numpy scalars as objects the loader
            # cannot reconstruct
            "reportpoints": [int(p) for p in report_points],
            "datatype": ["x1"],
            "parallel": 1,   # single-threaded: required for reproducibility
            "compute_adjoints": False,
        },
    }

    with open(f"{name}.yaml", "w") as handle:
        yaml.dump(config, handle)
    return f"{name}.yaml"


#: (scheme, analysis) combinations under characterisation.
#:
#: The ``subspace`` flavour of both ``es`` and ``enkf`` is absent: it raises
#: ``ValueError: Length of values (11) does not match length of index (15)`` on
#: this case, which predates the Phase 8 work and is untested elsewhere.
#: ``esmda/subspace`` is fine, so the fault is in the sequential path rather
#: than in the subspace analysis.
CASES = [
    ("esmda", "approx"),
    ("esmda", "full"),
    ("esmda", "subspace"),
    ("lmenrml", "approx"),
    ("lmenrml", "full"),
    ("lmenrml", "subspace"),
    ("gnenrml", "approx"),
    ("gnenrml", "full"),
    ("gnenrml", "subspace"),
    ("gnenrml", "margis"),
    ("es", "approx"),
    ("es", "full"),
    ("enkf", "approx"),
]


def run_case(scheme, analysis, tmpdir):
    """Run one case deterministically and return its numerical fingerprint."""
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.chdir(tmpdir)

    report_points = _write_synthetic_case()

    # The schemes perturb observations from the global numpy random state, so
    # this seed is what makes the run reproducible at all.
    np.random.seed(GLOBAL_SEED)

    config_file = _write_config(
        f"characterise_{scheme}_{analysis}", scheme, analysis, report_points
    )
    cfg_da, cfg_sim, cfg_ens = read_config.read(config_file)

    # Exactly what a user writes. Driving the cases through this means the
    # reference numbers pin the public entry point and the result object's
    # contents, not only the internal loop.
    result = SCHEME_CLASSES[scheme].assimilate(
        cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim), analysis=analysis
    )

    return {
        "enX": np.asarray(result.x, dtype=float),
        "data_misfit": np.atleast_1d(np.asarray(result.data_misfit, dtype=float)),
        "prior_data_misfit": np.atleast_1d(
            np.asarray(result.prior_data_misfit, dtype=float)
        ),
    }


def _key(scheme, analysis, field):
    return f"{scheme}__{analysis}__{field}"


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE_FILE.exists():
        pytest.skip(
            f"No reference file at {REFERENCE_FILE.name}. Generate it with:\n"
            f"  python {Path(__file__).name} --regenerate"
        )
    return np.load(REFERENCE_FILE)


@pytest.mark.parametrize("scheme,analysis", CASES)
def test_matches_reference(scheme, analysis, tmp_path, reference):
    """The scheme still produces exactly the reference numbers."""
    result = run_case(scheme, analysis, tmp_path)

    for field, value in result.items():
        key = _key(scheme, analysis, field)
        if key not in reference:
            pytest.skip(f"Reference has no entry for {key}; regenerate it.")

        expected = reference[key]
        assert value.shape == expected.shape, (
            f"{scheme}/{analysis} {field}: shape changed "
            f"{expected.shape} -> {value.shape}"
        )
        np.testing.assert_allclose(
            value, expected, rtol=RTOL, atol=ATOL,
            err_msg=(
                f"{scheme}/{analysis} {field} no longer matches the reference. "
                f"If this change is intended, regenerate the reference and "
                f"review the diff; if not, the refactor changed the numerics."
            ),
        )


@pytest.mark.parametrize("scheme,analysis", CASES[:1])
def test_run_is_reproducible(scheme, analysis, tmp_path):
    """Two seeded runs of the same case agree bit-for-bit.

    Guards the determinism the other tests here depend on: if seeding stops
    being sufficient, this fails directly rather than showing up as a confusing
    reference mismatch.
    """
    first = run_case(scheme, analysis, tmp_path / "a")
    second = run_case(scheme, analysis, tmp_path / "b")
    np.testing.assert_array_equal(
        first["enX"], second["enX"],
        err_msg="Seeded runs diverged; the schemes have an unseeded random source.",
    )


def regenerate():
    """Write the reference file from the current code."""
    import tempfile

    payload = {}
    for scheme, analysis in CASES:
        print(f"running {scheme}/{analysis} ...", flush=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            for field, value in run_case(scheme, analysis, tmpdir).items():
                payload[_key(scheme, analysis, field)] = value

    np.savez_compressed(REFERENCE_FILE, **payload)
    print(f"\nWrote {REFERENCE_FILE} with {len(payload)} arrays.")


@pytest.mark.parametrize("scheme,analysis", [("esmda", "approx")])
def test_config_driven_entry_point_matches_reference(scheme, analysis, tmp_path, reference):
    """``init_da(...)`` then ``run_assimilation()`` agrees with ``assimilate()``.

    The cases above all run through ``Scheme.assimilate(...)``, so this pins the
    other supported path -- config-driven construction through the registry --
    against the same reference. The roles used to be reversed, and
    ``assimilate()`` was the entry point nothing exercised, which is how it
    stayed inert through the whole Phase 8 migration.
    """
    from pipt import pipt_init

    os.chdir(tmp_path)
    report_points = _write_synthetic_case()
    np.random.seed(GLOBAL_SEED)
    config_file = _write_config(f"cfg_{scheme}_{analysis}", scheme, analysis, report_points)
    cfg_da, cfg_sim, cfg_ens = read_config.read(config_file)

    scheme_obj = pipt_init.init_da(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    result = scheme_obj.run_assimilation()

    # Both spellings must work: AssimilationResult subclasses scipy's
    # OptimizeResult so PIPT and POPT results are handled alike.
    np.testing.assert_array_equal(
        np.asarray(result["x"], dtype=float), np.asarray(result.x, dtype=float)
    )
    np.testing.assert_allclose(
        np.asarray(result.x, dtype=float),
        reference[_key(scheme, analysis, "enX")],
        rtol=RTOL, atol=ATOL,
        err_msg="init_da + run_assimilation does not reproduce the reference posterior.",
    )


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        cwd = os.getcwd()
        try:
            regenerate()
        finally:
            os.chdir(cwd)
    else:
        print(__doc__)
