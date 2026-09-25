"""Asking for data screening fails with an explanation, not an AttributeError."""

import pytest

from pipt.ensembles.ensemble_base import AssimilationEnsemble


def test_screendata_raises_a_clear_error():
    ens = object.__new__(AssimilationEnsemble)   # perturb_observations reads only keys_da first
    ens.keys_da = {"screendata": True}

    with pytest.raises(ValueError, match="'screendata' is not supported"):
        ens.perturb_observations(None)
