"""Guards against import cycles between the top-level packages.

``ensemble`` is the foundation package that both ``pipt`` and ``popt`` build on.
If it imports from either of them at module level, the layering inverts and
importing ``ensemble`` first raises a partially-initialized-module error.

That regression existed for a long time without being noticed, because the full
test suite happened to import the packages in an order that avoided it -- only
running a single test file surfaced it. These tests each import in a fresh
subprocess so import order cannot mask the problem.
"""

import subprocess
import sys

import pytest

TOP_LEVEL_PACKAGES = ["ensemble", "misc", "input_output", "pet_cli", "pipt", "popt", "simulator"]


@pytest.mark.parametrize("package", TOP_LEVEL_PACKAGES)
def test_package_imports_standalone(package):
    """Each package must import cleanly as the very first import."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {package}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`import {package}` failed as a first import:\n{result.stderr}"
    )


def test_ensemble_does_not_import_pipt_or_popt_at_module_level():
    """The foundation package must not depend upward at import time.

    Uses a fresh interpreter and checks which modules are resolved: importing
    ``ensemble`` must not drag in ``pipt`` or ``popt``.
    """
    code = (
        "import sys; import ensemble; "
        "print(','.join(sorted(m for m in sys.modules "
        "if m.split('.')[0] in ('pipt', 'popt'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    leaked = [m for m in result.stdout.strip().split(",") if m]
    assert not leaked, (
        "Importing `ensemble` pulled in upward dependencies: "
        f"{leaked}. Keep pipt/popt imports inside the functions that use them."
    )


def _modules_loaded_by(statement):
    """Run ``statement`` in a fresh interpreter and return the modules it loaded."""
    code = f"import sys; {statement}; print(','.join(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return set(result.stdout.strip().split(","))


def test_pipt_does_not_import_plotting_or_wavelets_at_module_level():
    """QA/QC (matplotlib, cv2) and sparse compression (PyWavelets) are
    optional features; a run that does not ask for them must not pay their
    import cost, and a machine without them must still be able to assimilate."""
    loaded = _modules_loaded_by("import pipt")
    heavy = {"cv2", "pywt", "matplotlib.pyplot"}
    assert not (loaded & heavy), f"`import pipt` loaded {sorted(loaded & heavy)}"


def test_misc_does_not_import_pipt_or_geostat_at_module_level():
    """``misc`` sits below ``pipt``: its data structures and readers must not
    depend upward, nor need the geostat git dependency, at import time."""
    loaded = _modules_loaded_by("import misc.structures, misc.read_input_csv")
    upward = sorted(m for m in loaded if m.split(".")[0] in ("pipt", "popt", "geostat"))
    assert not upward, f"importing misc pulled in {upward}"
