"""The MiniRes benchmark (benchmarks/minires) still runs: one tiny twin experiment per method."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("minires", reason="pip install PET[minires]")

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks" / "minires"))
from methods import METHODS  # noqa: E402
from run import experiment  # noqa: E402


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS, ids=[m["label"] for m in METHODS])
def test_a_twin_experiment_runs_and_scores(method):
    out = experiment("coarse", method, 0, ne=6)
    assert out["runs"] > 0
    for stage in ("prior", "posterior"):
        assert all(np.isfinite(v) for v in out[stage].values()), out[stage]
    assert out["posterior"]["misfit"] < out["prior"]["misfit"]
