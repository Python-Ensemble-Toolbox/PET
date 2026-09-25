"""A scheme can run on an ensemble it was handed, instead of building one.

Two schemes can then share one prior and its forecasts, and a test can hand a
scheme a stand-in without the config, data files and simulator a real ensemble
needs.
"""

import numpy as np
import pytest

from input_output import read_config
from pipt import ESMDA, EnKF, GNEnRML, LMEnRML
from pipt.ensembles import AssimilationEnsemble
from pipt.update_schemes.core import AssimilationScheme
from simulator.vanderpol import VanDerPolOscillator
from test_numerical_characterisation import _write_config, _write_synthetic_case


@pytest.fixture
def configs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case()
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("inject", "esmda", "approx", report_points))
    # The ES-MDA writer emits only the `mda` block; the iterative schemes read `iteration`.
    cfg_da["iteration"] = {"max_iter": 3, "lambda": 10, "lambda_factor": 5, "trunc_energy": 0.99}
    return cfg_da, cfg_sim, cfg_ens


@pytest.mark.parametrize("scheme_cls, analysis", [(ESMDA, "approx"), (LMEnRML, "approx"), (GNEnRML, "subspace"), (EnKF, "approx")])
def test_a_handed_in_ensemble_is_used_and_no_second_one_is_built(configs, monkeypatch, scheme_cls, analysis):
    cfg_da, cfg_sim, cfg_ens = configs
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))

    built = []
    original = AssimilationEnsemble.__init__

    def counting_init(self, *args, **kwargs):
        built.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(AssimilationEnsemble, "__init__", counting_init)
    scheme = scheme_cls(cfg_da, cfg_ens, ensemble.sim, analysis=analysis, ensemble=ensemble)

    assert scheme.ensemble is ensemble
    assert built == []


def test_the_default_collaborator_is_declared_on_the_base():
    assert AssimilationScheme.ENSEMBLE_CLASS is AssimilationEnsemble
    for scheme_cls in (ESMDA, LMEnRML, GNEnRML, EnKF):
        assert scheme_cls.ENSEMBLE_CLASS is AssimilationEnsemble


def test_two_schemes_can_share_one_prior(configs):
    cfg_da, cfg_sim, cfg_ens = configs
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    prior = np.array(ensemble.enX, dtype=float)

    first = ESMDA(cfg_da, cfg_ens, ensemble.sim, analysis="approx", ensemble=ensemble)
    second = LMEnRML(cfg_da, cfg_ens, ensemble.sim, analysis="approx", ensemble=ensemble)

    assert first.ensemble is second.ensemble
    np.testing.assert_array_equal(np.array(second.prior_enX, dtype=float), prior)
