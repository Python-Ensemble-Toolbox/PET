"""The ensemble works on canonical copies of the config sections it is given."""

import numpy as np

from input_output import read_config
from pipt.ensembles import AssimilationEnsemble
from simulator.vanderpol import VanDerPolOscillator
from test_numerical_characterisation import _write_config, _write_synthetic_case


def test_the_ensemble_keeps_its_own_copy_of_the_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=6)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("copy", "esmda", "approx", report_points, ne=6))
    cfg_da["save_folder"] = "elsewhere"                      # the alias, as a script might write it
    cfg_da.pop("nosave", None)                               # so the folder is in use, not switched off
    snapshot = {key: (value if not isinstance(value, (list, dict)) else repr(value)) for key, value in cfg_da.items()}
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))

    assert ensemble.keys_da["savefolder"] == "elsewhere" and ensemble.save_folder == "elsewhere"
    assert "datatype" in ensemble.keys_da and ensemble.keys_da["assimindex"] is not None
    for key in ("datatype", "truedataindex"):
        assert key not in cfg_da or repr(cfg_da[key]) == snapshot[key]      # nothing written back into the caller's dict
    np.testing.assert_array_equal(ensemble.enX.shape, (3, 6))
