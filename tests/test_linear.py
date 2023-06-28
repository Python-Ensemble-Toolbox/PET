import os
import sys
from pathlib import Path, PosixPath

import numpy as np
import subprocess

# Logger (since we cannot print during testing)
# -- there is probably a more official way to do this.
logfile = Path.cwd() / "PET-test-log"
with open(logfile, "w") as file:
    pass
def prnt(*args, **kwargs):
    with open(logfile, "a") as file:
        print(*args, **kwargs, file=file)


def test_git_clone(temp_examples_dir):
    # prnt(cwd)
    # prnt(os.listdir(cwd))
    assert (temp_examples_dir / "3Spot").is_dir()


def test_mod(temp_examples_dir: PosixPath):
    """Validate a few values of the result of the `LinearModel` example."""
    cwd = temp_examples_dir / "LinearModel"
    old = Path.cwd()

    try:
        os.chdir(cwd)
        sys.path.append(str(cwd))
        subprocess.run(["python", "write_true_and_data.py"], cwd=temp_examples_dir)
        import run_script
    finally:
        os.chdir(old)

    result = run_script.assimilation.ensemble.state['permx'].mean(axis=1)
    np.testing.assert_array_almost_equal(
        result[[1, 2, 3, -3, -2, -1]],
        [-0.07294738, 0.00353635, -0.06393236, 0.45394362, 0.44388684, 0.37096157],
        decimal=5)
