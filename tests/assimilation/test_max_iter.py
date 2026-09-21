"""`max_iter` is the number of update iterations a run may take."""

import numpy as np
import pytest

from input_output import read_config
from pipt import ESMDA, GNEnRML, LMEnRML
from simulator.vanderpol import VanDerPolOscillator
from test_numerical_characterisation import _write_config, _write_synthetic_case

NE = 20


@pytest.mark.parametrize("scheme_cls, analysis", [(LMEnRML, "approx"), (GNEnRML, "subspace")])
@pytest.mark.parametrize("max_iter", [1, 3])
def test_an_iterative_scheme_takes_exactly_max_iter_updates_when_nothing_else_stops_it(tmp_path, monkeypatch, scheme_cls, analysis, max_iter):
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("budget", "lmenrml", analysis, report_points, ne=NE))
    # A tolerance no step meets, and a generous inner budget, so the outer limit is what stops the run.
    cfg_da["iteration"] = {"max_iter": max_iter, "lambda": 10, "lambda_factor": 5, "trunc_energy": 0.99,
                           "data_misfit_tol": 1e-12, "max_inner_iter": 50}
    np.random.seed(0)
    result = scheme_cls.assimilate(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim), analysis=analysis)
    assert result.nit == max_iter
    assert result.message == "Maximum number of iterations reached"


def test_esmda_takes_one_update_per_assimilation_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config("steps", "esmda", "approx", report_points, ne=NE))
    assert cfg_da["mda"]["tot_assim_steps"] == 3
    result = ESMDA.assimilate(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim), analysis="approx")
    assert result.nit == 3
