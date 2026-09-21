"""``co_lm_enrml`` and ``gn_enrml`` are names, not algorithms.

Each is a thin subclass pinning one flavour of a live scheme -- ``co_lm_enrml``
is ``LMEnRML(analysis="approx")``, ``gn_enrml`` is ``GNEnRML(analysis="subspace")``
-- so on the same case, with the same seed, each must produce exactly the
numbers of the algorithm it names, whether constructed directly or selected
from a config by name.
"""

import numpy as np
import pytest

from input_output import read_config
from pipt import GNEnRML, LMEnRML, pipt_init
from pipt.update_schemes.enrml import co_lm_enrml, gn_enrml
from simulator.vanderpol import VanDerPolOscillator
from test_numerical_characterisation import GLOBAL_SEED, _write_config, _write_synthetic_case

CASES = [
    pytest.param("co_lm_enrml", co_lm_enrml, LMEnRML, "approx", id="co_lm_enrml"),
    pytest.param("gn_enrml", gn_enrml, GNEnRML, "subspace", id="gn_enrml"),
]


def _run(scheme_name, analysis, build):
    """Run the golden synthetic case; ``build(cfg_da, cfg_ens, sim)`` returns the result."""
    report_points = _write_synthetic_case()
    cfg_da, cfg_sim, cfg_ens = read_config.read(
        _write_config("legacy_names", scheme_name, analysis, report_points)
    )
    np.random.seed(GLOBAL_SEED)
    result = build(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    return np.asarray(result.x, dtype=float), np.asarray(result.data_misfit, dtype=float)


@pytest.mark.parametrize("name, legacy, live, analysis", CASES)
def test_class_gives_the_numbers_of_the_algorithm_it_names(tmp_path, monkeypatch, name, legacy, live, analysis):
    monkeypatch.chdir(tmp_path)
    x_live, misfit_live = _run(name, analysis, lambda da, en, sim: live.assimilate(da, en, sim, analysis=analysis))

    def build_legacy(da, en, sim):
        da.pop("analysis")            # the name alone must supply the flavour
        return legacy.assimilate(da, en, sim)

    x_legacy, misfit_legacy = _run(name, analysis, build_legacy)

    np.testing.assert_array_equal(x_legacy, x_live)
    np.testing.assert_array_equal(misfit_legacy, misfit_live)


@pytest.mark.parametrize("name, legacy, live, analysis", CASES)
def test_config_naming_the_scheme_runs_the_same_algorithm(tmp_path, monkeypatch, name, legacy, live, analysis):
    monkeypatch.chdir(tmp_path)
    x_live, _ = _run(name, analysis, lambda da, en, sim: live.assimilate(da, en, sim, analysis=analysis))
    x_config, _ = _run(name, analysis, lambda da, en, sim: pipt_init.init_da(da, en, sim).run_assimilation())

    np.testing.assert_array_equal(x_config, x_live)
