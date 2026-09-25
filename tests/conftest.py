"""Suite-wide fixtures.

PET's ensembles and optimizers write to the current working directory by
default: ``En_*`` folders, ``prior_ensemble.npz``, ``ASSIM.log``/``OPTIM.log``,
restart files. Until that default changes, every test starts in its own
temporary directory so nothing lands in the repository or wherever pytest was
launched. Tests that need a particular layout still call ``monkeypatch.chdir``
or ``os.chdir`` themselves; this fixture only sets the starting point.
"""

import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
