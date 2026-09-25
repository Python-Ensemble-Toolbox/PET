"""`calc_prediction(..., save_prediction=name)` writes the forecast where the ensemble's options say.

popt is the caller (its `save_prediction` option). The branch read
`self.ensemble.keys_da`, an attribute the base ensemble never had, so using
the option raised AttributeError; and it wrote into a folder it never created.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

from test_failed_member_replacement import _bare_ensemble, _members


# `save_folder` is mapped to `savefolder` at the config boundary (tests/test_config_boundary.py);
# a bare ensemble built without it holds canonical keys only.
@pytest.mark.parametrize("options, folder", [({}, "Predictions"), ({"savefolder": "out"}, "out")])
def test_the_forecast_is_pickled_under_the_named_folder(tmp_path, monkeypatch, options, folder):
    monkeypatch.chdir(tmp_path)
    ens = _bare_ensemble()
    ens.keys_en = options
    enX, _ = _members()

    np.random.seed(0)
    ens.calc_prediction(enX, save_prediction="forecast")

    path = Path(folder) / "forecast.pkl"
    assert path.exists()
    with open(path, "rb") as file:
        saved = pickle.load(file)
    np.testing.assert_array_equal(np.asarray(saved.loc[1, "d"]), np.asarray(ens.sim_data.loc[1, "d"]))
