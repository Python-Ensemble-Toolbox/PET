"""Seismic vintages are compressed as they enter the prediction matrix, with the observations' leading indices.

A small case shaped like a real one: a seismic type observed at two vintages,
each a vector over a masked grid, reduced by wavelet thresholding when read;
an uncompressed point type; a fake simulator that returns the raw vectors.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

from input_output import read_config
from pipt import ESMDA
from pipt.ensembles import AssimilationEnsemble

pytest.importorskip("pywt")

DIM = [16, 12, 12]
N_RAW = int(np.prod(DIM))
LABELS = [1000, 2000, 3000, 4000]
VINTAGES = [2000, 4000]
NE = 6


class SeismicSimulator:
    """Returns one record per report point: a smooth seismic vector plus a scalar, both depending on the state."""

    def __init__(self):
        self.input_dict = {"parallel": 1, "reporttype": "time", "reportpoints": LABELS, "datatype": ["avo", "grav"]}
        self.true_order = ["time", LABELS]
        self.redund_sim = None
        self.compute_adjoints = False

    @staticmethod
    def vintage(x1, label):
        grid = np.linspace(0.0, 3.0, N_RAW)
        return np.sin(grid * (1 + label / 4000.0)) * (1.0 + 0.2 * x1) + 0.05 * np.cos(7 * grid)

    def run_fwd_sim(self, state, member_index):
        x1 = float(np.ravel(state["x1"])[0])
        return [{"avo": self.vintage(x1, label), "grav": 10.0 * x1 + label / 1000.0} for label in LABELS]


def _write_case(tmp_path, use_ensemble=False, saveforecast=False):
    rng = np.random.default_rng(3)
    np.savez("prior_ensemble.npz", x1=(0.5 + 0.3 * rng.standard_normal(NE))[np.newaxis, :])
    for i in range(len(VINTAGES)):
        np.savez(f"mask_{i}.npz", mask=np.ones(DIM, dtype=bool))
    truth = 0.6
    obs = pd.DataFrame({"avo": [np.nan] * len(LABELS), "grav": [np.nan] * len(LABELS)}, index=LABELS, dtype=object)
    obs.index.name = "time"
    for label in VINTAGES:
        np.savez(f"avo_{label}.npz", SeismicSimulator.vintage(truth, label) + 0.02 * rng.standard_normal(N_RAW))
        obs.at[label, "avo"] = f"avo_{label}.npz"
    for label in LABELS:
        obs.at[label, "grav"] = 10.0 * truth + label / 1000.0 + 0.1 * rng.standard_normal()
    obs.to_pickle("true_data.pkl")
    var = pd.DataFrame({"avo": ["['abs', 1.0]"] * len(LABELS), "grav": ["['abs', 0.01]"] * len(LABELS)}, index=LABELS)
    var.index.name = "time"
    var.to_pickle("var.pkl")

    config = {
        "ensemble": {"ne": NE, "state": ["x1"], "importstate": "prior_ensemble.npz", "prior_x1": {"var": 1.0}},
        "dataassim": {
            "scheme": "esmda", "analysis": "approx", "energy": 0.99, "obsname": "time",
            "data": "true_data.pkl", "datavar": "var.pkl", "nosave": True,
            "mda": {"tot_assim_steps": 1, "inflation_param": [1]},
            "compress": {
                "compress_data": "avo", "dim": DIM, "mask": [f"mask_{i}.npz" for i in range(len(VINTAGES))],
                "level": 2, "wname": "db2", "threshold_rule": "universal", "th_mult": 1, "use_hard_th": True,
                "keep_ca": False, "inactive_value": 0.0, "use_ensemble": use_ensemble, "order": "F",
                "min_noise": [1e-9, 1e-9], "colored_noise": False,
            },
        },
        "simulator": {"reporttype": "time", "reportpoints": LABELS, "datatype": ["avo", "grav"], "parallel": 1},
    }
    if saveforecast:
        config["simulator"]["saveforecast"] = True
    with open("case.yaml", "w") as handle:
        yaml.dump(config, handle)
    cfg_da, cfg_sim, cfg_ens = read_config.read("case.yaml")
    sim = SeismicSimulator()
    sim.input_dict.update({k: v for k, v in cfg_sim.items() if k == "saveforecast"})
    return cfg_da, cfg_ens, sim


def test_observations_are_compressed_and_predictions_follow_with_the_same_leading_indices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_da, cfg_ens, sim = _write_case(tmp_path)
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, sim)

    # The reader reduced each observed vintage; the layout rows are the coefficient counts.
    assert len(ensemble.sparse_data) == len(VINTAGES)
    for vintage, label in enumerate(VINTAGES):
        row = ensemble.data_layout.row(label, "avo")
        representation = ensemble.sparse_data[vintage]
        assert 0 < row.size < representation.num_total_coeff          # thresholding dropped coefficients
        assert row.size == representation.cd_leading_index.size + representation.ca_leading_index.size
        np.testing.assert_allclose(ensemble.obs_variance[row.rows], representation.est_noise ** 2, rtol=1e-14)

    ensemble.forecast(ensemble.enX)
    pred = ensemble.pred_data
    assert pred.nd == ensemble.obs_vector.size == ensemble.obs_variance.size

    # Each member's raw vintage, compressed by hand with the observed vintage's representation, is what was filled.
    for vintage, label in enumerate(VINTAGES):
        row = ensemble.data_layout.row(label, "avo")
        for j, member in enumerate(ensemble.member_outputs[0]):
            raw = member[LABELS.index(label)]["avo"]
            expected, _ = ensemble.sparse_data[vintage].compress(raw)
            np.testing.assert_array_equal(pred.matrix[row.rows, j], expected)
    # The uncompressed type is untouched.
    row = ensemble.data_layout.row(1000, "grav")
    np.testing.assert_array_equal(pred.matrix[row.rows, 0], ensemble.member_outputs[0][0][0]["grav"])


def test_a_scheme_runs_on_the_compressed_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_da, cfg_ens, sim = _write_case(tmp_path)
    result = ESMDA.assimilate(cfg_da, cfg_ens, sim, analysis="approx")
    assert np.all(np.isfinite(result.x))
    assert result.data_misfit < result.prior_data_misfit


def test_use_ensemble_is_refused_with_the_reason(tmp_path, monkeypatch):
    """Observations are perturbed at construction; there is no forecast yet to widen the indices with."""
    monkeypatch.chdir(tmp_path)
    cfg_da, cfg_ens, sim = _write_case(tmp_path, use_ensemble=True)
    with pytest.raises(ValueError, match="use_ensemble"):
        AssimilationEnsemble(cfg_da, cfg_ens, sim)


def test_reconstructions_are_saved_only_when_the_forecast_is(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_da, cfg_ens, sim = _write_case(tmp_path, saveforecast=True)
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, sim)
    ensemble.forecast(ensemble.enX)
    with open("rec_results.pkl", "rb") as file:
        rec = pickle.load(file)
    assert len(rec) == len(VINTAGES) and all(r.shape == (N_RAW, NE) for r in rec)

    (tmp_path / "plain").mkdir()
    monkeypatch.chdir(tmp_path / "plain")
    cfg_da, cfg_ens, sim = _write_case(tmp_path / "plain")
    ensemble = AssimilationEnsemble(cfg_da, cfg_ens, sim)
    ensemble.forecast(ensemble.enX)
    assert ensemble.data_rec == [[], []] and not (tmp_path / "plain" / "rec_results.pkl").exists()
