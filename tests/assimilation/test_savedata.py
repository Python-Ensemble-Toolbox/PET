"""Per-iteration result saving: the ``savedata`` key and its output files.

Unit-level counterpart to the end-to-end assertions in
``test_assimilation_pipeline.py``. Those run a real scheme and are slow; these
drive the saving path of :class:`~pipt.update_schemes.core.AssimilationScheme`
directly, so the naming contract and the deprecated alias are cheap to pin.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from pipt.update_schemes.core import AssimilationScheme


class FakeScheme(AssimilationScheme):
    """Enough of a scheme for the saving path, and nothing else.

    ``keys_da`` and ``save_folder`` are read-only views of the ensemble, so
    they are supplied through a stand-in for it rather than assigned.
    """

    def __init__(self, keys_da, save_folder, iteration=0, **attrs):
        self.ensemble = SimpleNamespace(
            keys_da=keys_da,
            save_folder=str(save_folder),
            multilevel=None,
        )
        self.iteration = iteration
        for name, value in attrs.items():
            setattr(self, name, value)

    def update_step(self):
        """Never called: declared only because the class is abstract."""
        raise NotImplementedError


def _saved(folder, iteration):
    path = folder / f"assimilation_result_{iteration}.npz"
    assert path.exists(), f"expected {path.name}, found {sorted(p.name for p in folder.iterdir())}"
    with np.load(path, allow_pickle=True) as archive:
        return {name: archive[name] for name in archive.files}


# ----------------------------------------------------------------------
# Naming
# ----------------------------------------------------------------------
def test_file_is_named_for_the_iteration(tmp_path):
    """``assimilation_result_{i}.npz``, mirroring popt's ``optimize_result_{i}``.

    Was ``debug_analysis_step_{i}.npz``, which described the mechanism as a
    debugging aid rather than as the record of the run that it is.
    """
    scheme = FakeScheme(
        {"savedata": ["ensemble_misfit"]},
        tmp_path,
        iteration=3,
        ensemble_misfit=np.array([1.0, 2.0]),
    )
    scheme._save_iteration_data()

    np.testing.assert_array_equal(_saved(tmp_path, 3)["ensemble_misfit"], [1.0, 2.0])


def test_a_single_name_need_not_be_a_list(tmp_path):
    scheme = FakeScheme({"savedata": "data_misfit"}, tmp_path, data_misfit=7.5)
    scheme._save_iteration_data()

    assert _saved(tmp_path, 0)["data_misfit"] == 7.5


def test_unresolvable_names_are_skipped_not_fatal(tmp_path):
    """A variable can legitimately be absent for a given scheme.

    ``lam`` exists for the Levenberg-Marquardt family and not for ES-MDA, so a
    shared config naming it must not fail the ES-MDA run.
    """
    scheme = FakeScheme(
        {"savedata": ["data_misfit", "lam"]}, tmp_path, data_misfit=1.0
    )
    with pytest.warns(UserWarning, match="Cannot save 'lam'"):
        scheme._save_iteration_data()
    assert set(_saved(tmp_path, 0)) == {"data_misfit"}


# ----------------------------------------------------------------------
# The deprecated spelling
# ----------------------------------------------------------------------
def test_analysisdebug_still_works_and_warns(tmp_path):
    scheme = FakeScheme({"analysisdebug": ["data_misfit"]}, tmp_path, data_misfit=2.0)

    with pytest.deprecated_call(match="analysisdebug"):
        keys = scheme._savedata_keys

    assert keys == ["data_misfit"]


def test_savedata_wins_over_the_old_spelling(tmp_path):
    """Not merged: a config carrying both is mid-migration.

    Unioning them would keep honouring whichever one the user meant to delete.
    """
    scheme = FakeScheme(
        {"savedata": ["data_misfit"], "analysisdebug": ["ensemble_misfit"]},
        tmp_path,
        data_misfit=1.0,
        ensemble_misfit=np.array([1.0]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert scheme._savedata_keys == ["data_misfit"]


def test_no_key_means_no_saving(tmp_path):
    assert FakeScheme({}, tmp_path)._savedata_keys == []
