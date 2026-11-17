"""The popt benchmark on MiniRes (benchmarks/popt/minires) still runs: one tiny optimization per method."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("minires", reason="pip install PET[minires]")

FOLDER = Path(__file__).parents[1] / "benchmarks" / "popt" / "minires"

# Run in a process of its own: the benchmark pins the BLAS threads before numpy is imported
# (unpinned, the small solves of each simulation are many times slower), and its module names
# (``run``, ...) would otherwise meet those of ``benchmarks/pipt/minires`` in ``sys.modules``.
SCRIPT = """
import json, sys
import run
out = {}
for method in run.METHODS:
    method = dict(method)
    method["options"] = {**method["options"], "maxiter": 2}
    if "epf" in method["options"]:
        method["options"]["epf"] = {**method["options"]["epf"], "max_epf_iter": 2}
    out[method["label"]] = run.experiment(method, 0, 4)
json.dump(out, sys.stdout)
"""


@pytest.fixture(scope="module")
def outcomes():
    done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=FOLDER, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout)


@pytest.mark.slow
def test_every_method_runs_and_is_scored(outcomes):
    assert len(outcomes) == 5
    for label, out in outcomes.items():
        assert out["runs"] > 0, label
        assert all(np.isfinite(out[k]) for k in ("npv", "water", "capacity")), (label, out)
        assert len(out["trace"]["runs"]) == len(out["trace"]["npv"]) >= 1, label


@pytest.mark.slow
def test_the_start_violates_the_water_cap_and_ignoring_the_constraints_makes_it_worse(outcomes):
    start = outcomes["EnSQP (BFGS)"]["trace"]
    assert start["runs"][0] == 0 and start["violation"][0] > 0.05
    assert outcomes["EnOpt, constraints ignored"]["npv"] > start["npv"][0]
