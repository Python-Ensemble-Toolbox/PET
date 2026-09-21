"""A run resumed from a checkpoint continues the interrupted one exactly.

The checkpoint is the scheme's (RestartMixin), driven by `restart`,
`restartsave` and `restart_file` in the `[dataassim]` block. It carries the
loop's bookkeeping, the scheme's declared state and the ensemble's state and
random stream, so resuming does not depend on the random state of the
process that resumes.
"""

import numpy as np
import pytest

from input_output import read_config
from pipt import ESMDA, GNEnRML, LMEnRML
from pipt.ensembles import AssimilationEnsemble
from pipt.update_schemes.core import restart_options
from simulator.vanderpol import VanDerPolOscillator
from test_numerical_characterisation import _write_config, _write_synthetic_case

NE = 20
INTERRUPT_AT = 2


class Interrupted(Exception):
    pass


def _configs(tmp_path, monkeypatch, name, **da):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=NE)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_config(name, "esmda", "approx", report_points, ne=NE))
    cfg_da["iteration"] = {"max_iter": 4, "lambda": 10, "lambda_factor": 5, "trunc_energy": 0.99}
    cfg_da.update(da)
    return cfg_da, cfg_sim, cfg_ens


# EnKF is not here: with one assimilation index the case has one step, so there is nothing to interrupt.
@pytest.mark.parametrize("scheme_cls, analysis", [(ESMDA, "approx"), (LMEnRML, "approx"), (GNEnRML, "subspace")])
def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path, monkeypatch, scheme_cls, analysis):
    checkpoint = str(tmp_path / "checkpoint.pkl")

    # Uninterrupted reference.
    cfg_da, cfg_sim, cfg_ens = _configs(tmp_path / "ref", monkeypatch, "ref")
    np.random.seed(1)
    reference = scheme_cls.assimilate(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim), analysis=analysis)
    assert reference.nit > INTERRUPT_AT, "the case must run past the interruption point"

    # The same run, checkpointing, killed after its second accepted iteration.
    cfg_da, cfg_sim, cfg_ens = _configs(tmp_path / "run", monkeypatch, "run", restartsave=True, restart_file=checkpoint)
    np.random.seed(1)
    scheme = scheme_cls(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim), analysis=analysis)
    hook = scheme.after_accepted_iteration

    def hook_then_die():
        hook()
        if scheme.iteration == INTERRUPT_AT:
            raise Interrupted

    monkeypatch.setattr(scheme, "after_accepted_iteration", hook_then_die)
    with pytest.raises(Interrupted):
        scheme.run_assimilation()

    # Resume in a fresh process with a different random state.
    cfg_da, cfg_sim, cfg_ens = _configs(tmp_path / "resume", monkeypatch, "resume", restart=True, restart_file=checkpoint)
    np.random.seed(12345)
    resumed_scheme = scheme_cls(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim), analysis=analysis)
    resumed = resumed_scheme.run_assimilation()

    assert resumed_scheme.ensemble.restart is True
    assert resumed.nit == reference.nit
    np.testing.assert_array_equal(np.asarray(resumed.x, dtype=float), np.asarray(reference.x, dtype=float))
    np.testing.assert_array_equal(np.asarray(resumed.data_misfit), np.asarray(reference.data_misfit))


def test_the_checkpoint_is_written_after_the_prior_forecast(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ck.pkl"
    cfg_da, cfg_sim, cfg_ens = _configs(tmp_path, monkeypatch, "prior", restartsave="yes", restart_file=str(checkpoint))
    scheme = ESMDA(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim), analysis="approx")
    monkeypatch.setattr(scheme, "update_step", lambda: (_ for _ in ()).throw(Interrupted()))
    with pytest.raises(Interrupted):
        scheme.run_assimilation()
    assert checkpoint.exists()


def test_restart_options_read_the_dataassim_keys():
    assert restart_options({}) == {"restart": False, "restartsave": False}
    assert restart_options({"restart": "yes", "restartsave": "no", "restart_file": "x.pkl"}) == {
        "restart": True, "restartsave": False, "restart_file": "x.pkl"}


def test_the_ensemble_no_longer_looks_for_an_emergency_dump(tmp_path, monkeypatch):
    # `restart = yes` used to make the ensemble assert that `emergency_dump` sat in the working directory.
    cfg_da, cfg_sim, cfg_ens = _configs(tmp_path, monkeypatch, "nodump", restart="yes")
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))
    assert ensemble.restart is False
    assert not hasattr(ensemble, "load")
