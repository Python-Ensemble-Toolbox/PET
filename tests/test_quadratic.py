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
    assert (temp_examples_dir / "Quadratic").is_dir()


def test_mod(temp_examples_dir: PosixPath):
    """Validate a few values of the result of the `Quadratic` example."""
    cwd = temp_examples_dir / "Quadratic"
    old = Path.cwd()

    try:
        os.chdir(cwd)
        sys.path.append(str(cwd))
        import run_opt
        run_opt.main()
        files = os.listdir('./')
        results = [name for name in files if "optimize_result" in name]
        num_iter = len(results) - 1
        state = np.load(f'optimize_result_{num_iter}.npz', allow_pickle=True)['x']
        obj = np.load(f'optimize_result_{num_iter}.npz', allow_pickle=True)['obj_func_values']
    finally:
        os.chdir(old)

    np.testing.assert_array_almost_equal(state, [0.5, 0.5, 0.5], decimal=1)
    np.testing.assert_array_almost_equal(obj, [0.0], decimal=0)
