"""A failure while extracting one member's results costs that member, not the batch.

``en_pred`` is positional -- ``calc_prediction`` reads member *i* out of slot *i* --
so the invariant these tests defend is that the list comes back exactly ``ne`` long
with the failure in its own slot, however the extraction failed.
"""

from types import SimpleNamespace

import numpy as np

import pipt.misc_tools.analysis_tools as at
from ensemble.ensemble import BaseEnsemble

NE, NX = 4, 2


def _host(sim):
    log = []
    return SimpleNamespace(
        ne=NE,
        sim=sim,
        logger=SimpleNamespace(info=log.append, error=log.append),
    ), log


def _sim(*, extract_raises_on=(), saveinfo=None):
    """A simulator whose HPC hooks all succeed, except extraction for named members."""

    def extract_data(member_i):
        if member_i in extract_raises_on:
            raise RuntimeError(f"no results for {member_i}")
        sim.pred_data = [{"d": np.array([float(member_i)])}]

    sim = SimpleNamespace(
        file="case",
        options={"mpiarray": False},
        saveinfo=saveinfo,
        pred_data=None,
        run_fwd_sim=lambda state, member_index, nosim=True: None,
        SLURM_HPC_run=lambda n_e, **kwargs: "job-1",
        wait_for_jobs=lambda job_id: [True] * NE,
        extract_data=extract_data,
        remove_folder=lambda member_i: None,
    )
    return sim


def _member_of(pred):
    return int(pred[0]["d"][0])


def test_an_extraction_failure_costs_only_its_own_member():
    host, log = _host(_sim(extract_raises_on=(1,)))

    en_pred = BaseEnsemble.run_on_HPC(host, np.zeros((NE, NX)), batch_size=NE)

    assert len(en_pred) == NE
    assert en_pred[1] is False
    assert [_member_of(en_pred[i]) for i in (0, 2, 3)] == [0, 2, 3]
    assert any("Could not extract data for ensemble member 1" in m for m in log)


def test_a_saveinfo_failure_does_not_shift_the_members_after_it(monkeypatch):
    """Upstream's c629c0f appends inside the try *and* in the except, so a raise from
    store_ensemble_sim_information appends twice for one member and every later
    member reads one slot too early."""
    def boom(saveinfo, member_i):
        if member_i == 1:
            raise RuntimeError("disk full")

    monkeypatch.setattr(at, "store_ensemble_sim_information", boom)
    host, log = _host(_sim(saveinfo={"store": True}))

    en_pred = BaseEnsemble.run_on_HPC(host, np.zeros((NE, NX)), batch_size=NE)

    assert len(en_pred) == NE
    assert [_member_of(p) for p in en_pred] == [0, 1, 2, 3]
    assert any("Could not store sim information for member 1" in m for m in log)


def test_a_crashed_simulation_still_gets_its_own_slot():
    sim = _sim()
    sim.wait_for_jobs = lambda job_id: [True, False, True, True]
    host, _ = _host(sim)

    en_pred = BaseEnsemble.run_on_HPC(host, np.zeros((NE, NX)), batch_size=NE)

    assert len(en_pred) == NE
    assert en_pred[1] is False
    assert [_member_of(en_pred[i]) for i in (0, 2, 3)] == [0, 2, 3]
