"""Robust optimization: every control vector is evaluated on every geological model.

With ``num_models > 1`` a single control vector must run once per model, each copy
with its own model index, and the gradient ensemble must pair its perturbations
with the models in the order the gradient subtracts the single-point values.
"""

import numpy as np
import pandas as pd

from popt import GaussianEnsemble


class _ModelRecorder:
    """A simulator whose output is the sum of the controls plus 100 per model index."""

    def __init__(self):
        self.input_dict = {'datatype': ['v'], 'parallel': 1}
        self.datatype = ['v']
        self.true_order = None
        self.redund_sim = None
        self.models = []

    def setup_fwd_run(self, **kwargs):
        pass

    def run_fwd_sim(self, state, member_index):
        self.models.append(int(state['aux_input']))
        return pd.DataFrame({'v': [float(np.sum(state['x'])) + 100 * state['aux_input']]}, index=[0])


def _objective(data, **kwargs):
    return np.array([np.asarray(v).ravel() for v in data['v']]).ravel()


def _ensemble(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = _ModelRecorder()
    keys = {
        'ne': 10, 'num_models': 5, 'seed': 1, 'save_prior': False, 'disable_tqdm': True,
        'controls': {'x': {'mean': [1.0, 2.0], 'var': 0.01, 'limits': [0, 10]}},
    }
    return GaussianEnsemble(keys, sim, _objective), sim


def test_a_single_control_vector_runs_on_every_model(tmp_path, monkeypatch):
    ensemble, sim = _ensemble(tmp_path, monkeypatch)

    values = ensemble.function(ensemble.get_state())

    assert sim.models == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(values, [3.0, 103.0, 203.0, 303.0, 403.0])


def test_the_gradient_pairs_each_perturbation_with_its_model(tmp_path, monkeypatch):
    ensemble, sim = _ensemble(tmp_path, monkeypatch)
    x = ensemble.get_state()
    ensemble.function(x)
    sim.models.clear()

    gradient = ensemble.gradient(x, ensemble.get_cov())

    assert sim.models == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # The model offsets cancel against the single-point values, leaving a small
    # natural gradient of sum(x) rather than one of order 100.
    assert np.all(np.abs(gradient) < 1.0)
