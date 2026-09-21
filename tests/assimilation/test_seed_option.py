"""A run with a `seed` in its ensemble config is reproducible on its own.

Every draw -- prior realisations, perturbed observations, outlier and crash
replacement -- comes from the ensemble's private stream, so the result does
not depend on NumPy's global state and does not disturb it either.
"""

import numpy as np
import pytest

from input_output import read_config
from pipt import ESMDA
from simulator.vanderpol import VanDerPolOscillator
from test_numerical_characterisation import _write_config, _write_synthetic_case

NE = 40


def _run(tmp_path, monkeypatch, name, global_seed, seed=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config(name, "esmda", "approx", report_points, ne=NE))
    if seed is not None:
        cfg_ens["seed"] = seed
    np.random.seed(global_seed)
    result = ESMDA.assimilate(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim), analysis="approx")
    return np.asarray(result.x, dtype=float)


def test_a_seeded_run_reproduces_regardless_of_the_global_state(tmp_path, monkeypatch):
    first = _run(tmp_path / "a", monkeypatch, "seeded_a", global_seed=1, seed=7)
    second = _run(tmp_path / "b", monkeypatch, "seeded_b", global_seed=2, seed=7)
    np.testing.assert_array_equal(first, second)


def test_a_seeded_run_leaves_the_global_stream_untouched(tmp_path, monkeypatch):
    np.random.seed(3)
    before = np.random.get_state()
    _run(tmp_path, monkeypatch, "seeded_c", global_seed=3, seed=7)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2] == after[2]


def test_without_a_seed_the_global_state_still_governs_the_run(tmp_path, monkeypatch):
    # Unchanged behaviour: np.random.seed(...) before the run is what reproduces it.
    first = _run(tmp_path / "a", monkeypatch, "unseeded_a", global_seed=1)
    second = _run(tmp_path / "b", monkeypatch, "unseeded_b", global_seed=1)
    third = _run(tmp_path / "c", monkeypatch, "unseeded_c", global_seed=2)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


@pytest.mark.parametrize("seed", [7, "7"])
def test_the_seed_is_read_from_the_ensemble_config(tmp_path, monkeypatch, seed):
    from pipt.ensembles import AssimilationEnsemble

    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("seed_type", "esmda", "approx", report_points, ne=NE))
    cfg_ens["seed"] = seed
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    assert isinstance(ensemble.rng, np.random.RandomState)
