"""Numerical and PET lifecycle tests for the EnIF schemes."""

from copy import deepcopy

import networkx as nx
import numpy as np
import pytest
from graphite_maps.enif import EnIF
from graphite_maps.linear_regression import linear_boost_ic_regression
from graphite_maps.precision_estimation import fit_precision_cholesky_approximate
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from pipt.loop.assimilation import Assimilate
from pipt.pipt_init import init_da
from pipt.update_schemes.enif import enif_full, enif_mda, enif_update
from simulator.simple_models import lin_1d


class LinearModel(lin_1d):
    """Return independent response buffers for PET's serial simulation loop."""

    def run_fwd_sim(self, state, member_i):
        return deepcopy(super().run_fwd_sim(state, member_i))


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def update_scheme():
    scheme = enif_update()
    scheme.idX = {'field': (0, 6)}
    scheme.prior_info = {'field': {'nx': 3, 'ny': 2, 'nz': 1}}
    scheme.enif_options = {}
    scheme.disable_tqdm = True
    scheme.alpha = [1.0]
    scheme.iteration = 1
    scheme.vecObs = np.array([1.2, -0.3])
    scheme.cov_data = np.array([0.2, 0.5])
    return scheme


@pytest.mark.parametrize('alpha', [1.0, 4.0])
def test_matches_ert_transport(update_scheme, alpha):
    """Compare the PET step with ERT's fit-and-transport recipe, member by member."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(6, 80)) * np.arange(1, 7)[:, None] + 5
    Y = np.vstack((X[0] + 0.3 * X[1] ** 2, X[4] - X[5]))
    graph = nx.grid_2d_graph(3, 2)
    graph = nx.convert_node_labels_to_integers(graph)
    scaler = StandardScaler()
    U = scaler.fit_transform(X.T)
    H = linear_boost_ic_regression(U=U, Y=Y.T)
    precision = fit_precision_cholesky_approximate(U, graph, use_tqdm=False)
    reference = EnIF(
        Prec_u=precision,
        Prec_eps=sparse.diags_array(1 / (alpha * update_scheme.cov_data), format='csc'),
        H=H,
    )
    noise = reference.generate_observation_noise(X.shape[1], seed=19)
    expected = reference.transport(
        U, Y.T, update_scheme.vecObs,
        update_indices=reference.get_update_indices(neighbor_propagation_order=15),
        iterative=False, seed=19,
    )
    expected = scaler.inverse_transform(expected).T

    update_scheme.alpha = [alpha]
    E = update_scheme.vecObs[:, None] - noise.T
    update_scheme.update(X, Y, E)

    np.testing.assert_allclose(X + update_scheme.step, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(update_scheme.Prec_posterior.toarray(), reference.Prec_u.toarray())


def test_parameter_grid_order_and_custom_graphs(update_scheme, tmp_path):
    update_scheme.prior_info['field']['nz'] = 2
    graph = update_scheme._parameter_graph('field', 12)
    assert set(graph.neighbors(0)) == {1, 2, 6}
    assert set(graph.neighbors(5)) == {3, 4, 11}
    assert graph.number_of_edges() == 20

    custom = nx.path_graph(6)
    filename = tmp_path / 'graph.npz'
    sparse.save_npz(filename, nx.to_scipy_sparse_array(custom))
    update_scheme.enif_options['parameter_graphs'] = {'field': filename}
    assert set(update_scheme._parameter_graph('field', 6).edges) == set(custom.edges)

    update_scheme.enif_options['parameter_graphs']['field'] = nx.empty_graph(6)
    assert update_scheme._parameter_graph('field', 6).number_of_edges() == 0
    update_scheme.enif_options['parameter_graphs']['field'] = nx.path_graph(5)
    with pytest.raises(ValueError, match='nodes 0 through 5'):
        update_scheme._parameter_graph('field', 6)
    update_scheme.enif_options = {}
    with pytest.raises(ValueError, match='12 cells but 6 parameter rows'):
        update_scheme._parameter_graph('field', 6)


def test_masks_and_group_precision(update_scheme):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, 60))
    X[1] = np.nan
    X[3, 0] = np.nan
    X[4] = 2.0
    Y = np.vstack((X[0], X[5]))
    update_scheme.idX = {'other': (5, 6), 'field': (0, 5)}
    update_scheme.prior_info = {'field': {'nx': 5, 'ny': 1}, 'other': {}}
    update_scheme.update(X, Y, np.tile(update_scheme.vecObs[:, None], (1, X.shape[1])))

    np.testing.assert_array_equal(update_scheme.enif_active_rows, [0, 2, 5])
    np.testing.assert_array_equal(update_scheme.step[[1, 3, 4]], 0)
    assert np.isfinite(update_scheme.step).all()
    assert np.linalg.norm(update_scheme.step[[0, 5]]) > 0
    # Removing inactive nodes must not bridge across the hole in the field.
    assert update_scheme.Prec_u[0, 1] == 0
    assert update_scheme.Prec_u[:2, 2:].nnz == 0


def test_correlated_observations(update_scheme):
    """Whitened EnIF agrees with a Gaussian update using the full covariance."""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(6, 400))
    X -= X.mean(axis=1, keepdims=True)
    X /= X.std(axis=1, keepdims=True)
    Y = np.vstack((X[0], X[1]))
    covariance = np.array([[0.4, 0.2], [0.2, 0.6]])
    update_scheme.cov_data = covariance
    update_scheme.prior_info = {'field': {}}
    E = np.tile(update_scheme.vecObs[:, None], (1, X.shape[1]))
    update_scheme.update(X, Y, E)

    expected_mean = np.linalg.solve(np.eye(2) + covariance, update_scheme.vecObs)
    np.testing.assert_allclose((X + update_scheme.step).mean(axis=1)[:2], expected_mean, atol=0.025)


@pytest.mark.parametrize('covariance', [np.array([0.0, 1.0]), np.array([-1.0, 1.0]),
                                      np.array([np.nan, 1.0]), np.ones(3),
                                      np.array([[1.0, 0.2], [0.0, 1.0]])])
def test_invalid_observation_covariance(update_scheme, covariance):
    update_scheme.cov_data = covariance
    with pytest.raises(ValueError):
        update_scheme._observation_precision(np.ones((2, 20)), np.ones((2, 20)))


def test_constant_and_nonfinite_parameters(update_scheme):
    X = np.ones((6, 20))
    Y = np.ones((2, 20))
    update_scheme.update(X, Y, Y)
    np.testing.assert_array_equal(update_scheme.step, 0)
    with pytest.raises(ValueError, match='No finite parameter rows'):
        update_scheme.update(X * np.nan, Y, Y)
    with pytest.raises(ValueError, match='at least two ensemble members'):
        update_scheme.update(X[:, :1], Y[:, :1], Y[:, :1])
    with pytest.raises(ValueError, match='must be finite'):
        update_scheme.update(X, Y * np.nan, Y)


@pytest.fixture
def pet_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(42)
    np.savez('prior.npz', field=rng.normal(size=(1, 400)))
    keys_da = {
        'daalg': ['enif', 'enif'], 'analysis': 'full',
        'obsname': 'index', 'truedataindex': [0], 'assimindex': [0],
        'datatype': ['value'], 'truedata': [1.0], 'datavar': ['abs', 0.25],
    }
    keys_en = {
        'ne': 400, 'state': ['field'], 'staticvar': ['field'],
        'prior_field': {'mean': 0.0, 'var': 1.0},
        'importstaticvar': 'prior.npz', 'disable_tqdm': True,
    }
    sim = LinearModel({'reporttype': 'index', 'reportpoint': [0], 'datatype': ['value']})
    return keys_da, keys_en, sim


@pytest.mark.parametrize('analysis, mda, steps', [
    ('full', None, 1),
    ('mda', {'tot_assim_steps': 1}, 1),
    ('mda', {'tot_assim_steps': 3}, 3),
    ('mda', {'tot_assim_steps': 3, 'inflation_param': [2, 4, 4]}, 3),
])
def test_pet_assimilation_loop(pet_inputs, analysis, mda, steps):
    keys_da, keys_en, sim = pet_inputs
    keys_da['analysis'] = analysis
    if mda is not None:
        keys_da['mda'] = mda
    np.random.seed(21)
    ensemble = init_da(keys_da, keys_en, sim)
    prior = ensemble.enX.copy()
    assimilation = Assimilate(ensemble)
    assimilation.run()

    assert isinstance(ensemble, enif_full if analysis == 'full' else enif_mda)
    assert ensemble.iteration == steps + 1
    assert ensemble.enX_temp is None
    assert ensemble.data_misfit < ensemble.prior_data_misfit
    np.testing.assert_array_equal(ensemble.prior_enX, prior)
    # N(0, 1) prior observed at 1 with variance 0.25 has N(0.8, 0.2) posterior.
    np.testing.assert_allclose(ensemble.enX.mean(), 0.8, atol=0.07)
    np.testing.assert_allclose(ensemble.enX.var(), 0.2, atol=0.05)
    posterior = np.load('SaveOutputs/posterior_state_estimate.npz')['field']
    np.testing.assert_array_equal(posterior, ensemble.enX)
    np.testing.assert_allclose(ensemble.pred_data[0]['value'], ensemble.enX)


def test_one_step_mda_is_original_enif(pet_inputs):
    results = []
    for analysis in ('full', 'mda'):
        keys_da, keys_en, sim = deepcopy(pet_inputs)
        keys_da.update(analysis=analysis, mda={'tot_assim_steps': 1})
        np.random.seed(22)
        ensemble = init_da(keys_da, keys_en, sim)
        ensemble.iteration = 1
        ensemble.pred_data = [{'value': ensemble.enX.copy()}]
        ensemble.calc_analysis()
        results.append(ensemble.enX_temp.copy())
    np.testing.assert_array_equal(*results)


def test_state_limits(pet_inputs):
    keys_da, keys_en, sim = pet_inputs
    keys_en['prior_field']['limits'] = [-0.1, 0.1]
    ensemble = init_da(keys_da, keys_en, sim)
    ensemble.iteration = 1
    ensemble.pred_data = [{'value': ensemble.enX.copy()}]
    ensemble.calc_analysis()
    assert np.min(ensemble.enX_temp) >= -0.1
    assert np.max(ensemble.enX_temp) <= 0.1


@pytest.mark.parametrize('options', [
    {'tot_assim_steps': 0}, {'tot_assim_steps': 1.5},
    {'tot_assim_steps': 2, 'inflation_param': [2]},
    {'tot_assim_steps': 2, 'inflation_param': [1, 1]},
    {'tot_assim_steps': 2, 'inflation_param': [0, 2]},
    {'tot_assim_steps': 2, 'inflation_param': [np.inf, 1]},
    {'tot_assim_steps': 2, 'inflation_param': [-1, 0.5]},
])
def test_invalid_mda_schedule(options):
    ensemble = enif_mda.__new__(enif_mda)
    ensemble.keys_da = {'mda': options}
    with pytest.raises(ValueError):
        ensemble._ext_inflation_param()


@pytest.mark.parametrize('options', [
    {'tot_assim_steps': 3, 'inflation_param': [2, 4, 4]},
    [['tot_assim_steps', 3], ['inflation_param', [2, 4, 4]]],
])
def test_restart_preserves_full_schedule(options):
    ensemble = enif_mda.__new__(enif_mda)
    ensemble.keys_da = {'mda': options}
    ensemble.restart = True
    ensemble.iteration = 2
    ensemble.loop_ind = 1
    assert ensemble._ext_assim_steps() == [0, 1, 2]
    assert ensemble._ext_inflation_param() == [2, 4, 4]
