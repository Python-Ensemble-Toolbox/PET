"""Numerical and PET lifecycle tests for the EnIF analysis flavour.

The first block binds ``enif_update`` to a flat scheme double, following
``test_analysis_binding``: a real scheme exposes the ensemble-owned context
(``idX``, ``prior_info``, ``cov_data``) as properties of its own, so a double
just needs those names present. These tests pin the numerics against ERT's
own fit-and-transport recipe, the graph construction, and the input
validation.

The second block runs ES-MDA with the flavour bound, through the ordinary
config path, on a single-parameter identity model where the Gaussian
posterior is known analytically.
"""

import numpy as np
import pandas as pd
import pytest

from graphite_maps.enif import EnIF
from graphite_maps.linear_regression import linear_boost_ic_regression
from graphite_maps.precision_estimation import fit_precision_cholesky_approximate
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import networkx as nx

from misc.structures import PETDataFrame
from pipt import ESMDA
from pipt.update_schemes import registry
from pipt.update_schemes.analysis.enif import enif_update
from simulator.simple_models import lin_1d


class FakeScheme:
    """The context the EnIF analysis reads, and nothing else (see module docstring)."""

    def __init__(self):
        self.keys_da = {}
        self.keys_en = {'disable_tqdm': True}
        self.idX = {'field': (0, 6)}
        self.prior_info = {'field': {'nx': 3, 'ny': 2, 'nz': 1}}
        self.alpha = [1.0]
        self.iteration = 0
        self.vecObs = np.array([1.2, -0.3])
        self.cov_data = np.array([0.2, 0.5])


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def scheme_double():
    return FakeScheme()


@pytest.mark.parametrize('alpha', [1.0, 4.0])
def test_matches_ert_transport(scheme_double, alpha):
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
        Prec_eps=sparse.diags_array(1 / (alpha * scheme_double.cov_data), format='csc'),
        H=H,
    )
    noise = reference.generate_observation_noise(X.shape[1], seed=19)
    expected = reference.transport(
        U, Y.T, scheme_double.vecObs,
        update_indices=reference.get_update_indices(neighbor_propagation_order=15),
        iterative=False, seed=19,
    )
    expected = scaler.inverse_transform(expected).T

    scheme_double.alpha = [alpha]
    E = scheme_double.vecObs[:, None] - noise.T
    analysis = enif_update(scheme_double)
    result = analysis.update(X, Y, E)

    np.testing.assert_allclose(X + result.step, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(analysis.Prec_posterior.toarray(), reference.Prec_u.toarray())


def test_parameter_grid_order_and_custom_graphs(scheme_double, tmp_path):
    analysis = enif_update(scheme_double)
    scheme_double.prior_info['field']['nz'] = 2
    graph = analysis._parameter_graph('field', 12)
    assert set(graph.neighbors(0)) == {1, 2, 6}
    assert set(graph.neighbors(5)) == {3, 4, 11}
    assert graph.number_of_edges() == 20

    custom = nx.path_graph(6)
    filename = tmp_path / 'graph.npz'
    sparse.save_npz(filename, nx.to_scipy_sparse_array(custom))
    scheme_double.keys_da['enif'] = {'parameter_graphs': {'field': filename}}
    assert set(analysis._parameter_graph('field', 6).edges) == set(custom.edges)

    scheme_double.keys_da['enif']['parameter_graphs']['field'] = nx.empty_graph(6)
    assert analysis._parameter_graph('field', 6).number_of_edges() == 0
    scheme_double.keys_da['enif']['parameter_graphs']['field'] = nx.path_graph(5)
    with pytest.raises(ValueError, match='nodes 0 through 5'):
        analysis._parameter_graph('field', 6)
    scheme_double.keys_da['enif'] = {}
    with pytest.raises(ValueError, match='12 cells but 6 parameter rows'):
        analysis._parameter_graph('field', 6)


def test_masks_and_group_precision(scheme_double):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, 60))
    X[1] = np.nan
    X[3, 0] = np.nan
    X[4] = 2.0
    Y = np.vstack((X[0], X[5]))
    scheme_double.idX = {'other': (5, 6), 'field': (0, 5)}
    scheme_double.prior_info = {'field': {'nx': 5, 'ny': 1}, 'other': {}}
    analysis = enif_update(scheme_double)
    result = analysis.update(X, Y, np.tile(scheme_double.vecObs[:, None], (1, X.shape[1])))

    np.testing.assert_array_equal(analysis.enif_active_rows, [0, 2, 5])
    np.testing.assert_array_equal(result.step[[1, 3, 4]], 0)
    assert np.isfinite(result.step).all()
    assert np.linalg.norm(result.step[[0, 5]]) > 0
    # Removing inactive nodes must not bridge across the hole in the field.
    assert analysis.Prec_u[0, 1] == 0
    assert analysis.Prec_u[:2, 2:].nnz == 0


def test_correlated_observations(scheme_double):
    """Whitened EnIF agrees with a Gaussian update using the full covariance."""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(6, 400))
    X -= X.mean(axis=1, keepdims=True)
    X /= X.std(axis=1, keepdims=True)
    Y = np.vstack((X[0], X[1]))
    covariance = np.array([[0.4, 0.2], [0.2, 0.6]])
    scheme_double.cov_data = covariance
    scheme_double.prior_info = {'field': {}}
    E = np.tile(scheme_double.vecObs[:, None], (1, X.shape[1]))
    analysis = enif_update(scheme_double)
    result = analysis.update(X, Y, E)

    expected_mean = np.linalg.solve(np.eye(2) + covariance, scheme_double.vecObs)
    np.testing.assert_allclose((X + result.step).mean(axis=1)[:2], expected_mean, atol=0.025)


@pytest.mark.parametrize('covariance', [np.array([0.0, 1.0]), np.array([-1.0, 1.0]),
                                        np.array([np.nan, 1.0]), np.ones(3),
                                        np.array([[1.0, 0.2], [0.0, 1.0]])])
def test_invalid_observation_covariance(scheme_double, covariance):
    scheme_double.cov_data = covariance
    analysis = enif_update(scheme_double)
    with pytest.raises(ValueError):
        analysis._observation_precision(np.ones((2, 20)), np.ones((2, 20)))


def test_constant_and_nonfinite_parameters(scheme_double):
    X = np.ones((6, 20))
    Y = np.ones((2, 20))
    analysis = enif_update(scheme_double)
    np.testing.assert_array_equal(analysis.update(X, Y, Y).step, 0)
    with pytest.raises(ValueError, match='No finite parameter rows'):
        analysis.update(X * np.nan, Y, Y)
    with pytest.raises(ValueError, match='at least two ensemble members'):
        analysis.update(X[:, :1], Y[:, :1], Y[:, :1])
    with pytest.raises(ValueError, match='must be finite'):
        analysis.update(X, Y * np.nan, Y)


# ----------------------------------------------------------------------
# Configuration validation, at binding time
# ----------------------------------------------------------------------
@pytest.mark.parametrize('section, key', [
    ('keys_da', 'localization'), ('keys_da', 'localanalysis'),
    ('keys_da', 'multilevel'), ('keys_en', 'multilevel'),
    ('keys_en', 'localization'),
])
def test_rejects_unsupported_options(scheme_double, section, key):
    getattr(scheme_double, section)[key] = {}
    with pytest.raises(ValueError, match=f'EnIF does not support {key}'):
        enif_update(scheme_double)


def test_rejects_emp_cov(scheme_double):
    scheme_double.keys_da['emp_cov'] = True
    with pytest.raises(ValueError, match='emp_cov'):
        enif_update(scheme_double)


@pytest.mark.parametrize('options', [
    'not-a-dictionary',
    {'neighbourhood_expansion': 0},
    {'neighbourhood_expansion': 1.5},
    {'neighbourhood_expansion': True},
    {'neighbor_propagation_order': -1},
])
def test_rejects_invalid_enif_options(scheme_double, options):
    scheme_double.keys_da['enif'] = options
    with pytest.raises(ValueError):
        enif_update(scheme_double)


def test_rejects_unknown_parameter_graph_names(scheme_double):
    scheme_double.keys_da['enif'] = {'parameter_graphs': {'unknown': nx.path_graph(6)}}
    with pytest.raises(ValueError, match='known state names'):
        enif_update(scheme_double)


# ----------------------------------------------------------------------
# ES-MDA with the EnIF flavour, through the ordinary config path
# ----------------------------------------------------------------------
@pytest.fixture
def pet_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(42)
    np.savez('prior.npz', field=rng.normal(size=(1, 400)))

    data = PETDataFrame(pd.DataFrame({'value': [1.0]}, index=pd.Index([0], name='index')))
    var = PETDataFrame(pd.DataFrame({'value': [['abs', 0.25]]}, index=pd.Index([0], name='index')))
    data.to_pickle('true_data.pkl')
    var.to_pickle('var.pkl')

    keys_da = {
        'scheme': 'esmda', 'analysis': 'enif',
        'obsname': 'index', 'truedataindex': [0], 'assimindex': [0],
        'datatype': ['value'], 'data': 'true_data.pkl', 'datavar': 'var.pkl',
    }
    keys_en = {
        'ne': 400, 'state': ['field'], 'prior_field': {'mean': 0.0, 'var': 1.0},
        'importstate': 'prior.npz', 'disable_tqdm': True,
    }
    sim = lin_1d({'reporttype': 'index', 'reportpoint': [0], 'datatype': ['value']})
    return keys_da, keys_en, sim


@pytest.mark.parametrize('mda, steps', [
    ({'tot_assim_steps': 1}, 1),
    ({'tot_assim_steps': 3}, 3),
    ({'tot_assim_steps': 3, 'inflation_param': [2, 4, 4]}, 3),
])
def test_esmda_enif_assimilation_loop(pet_inputs, mda, steps):
    keys_da, keys_en, sim = pet_inputs
    keys_da['mda'] = mda
    np.random.seed(21)
    scheme = ESMDA(keys_da, keys_en, sim)
    prior = scheme.prior_enX.copy()
    result = scheme.run_assimilation()

    assert isinstance(scheme.analysis, enif_update)
    assert scheme.analysis_name == 'enif'
    assert scheme.iteration == steps
    assert result.data_misfit < result.prior_data_misfit
    np.testing.assert_array_equal(scheme.prior_enX, prior)
    # N(0, 1) prior observed at 1 with variance 0.25 has N(0.8, 0.2) posterior.
    np.testing.assert_allclose(scheme.enX.mean(), 0.8, atol=0.07)
    np.testing.assert_allclose(scheme.enX.var(), 0.2, atol=0.05)
    posterior = np.load('Results/posterior_state_estimate.npz')['field']
    np.testing.assert_array_equal(posterior, scheme.enX)
    np.testing.assert_allclose(scheme.pred_data.matrix, scheme.enX)


def test_registry_offers_the_flavour_on_esmda_only():
    assert ('esmda', 'enif') in registry.available_schemes()
    ctor = registry.get_scheme('esmda', 'enif')
    assert ctor.func is ESMDA
    assert ctor.keywords == {'analysis': 'enif'}
    from pipt.update_schemes.analysis.registry import available_analyses
    assert 'enif' not in available_analyses()


def test_state_limits(pet_inputs):
    keys_da, keys_en, sim = pet_inputs
    keys_da['mda'] = {'tot_assim_steps': 1}
    keys_en['prior_field']['limits'] = [-0.1, 0.1]
    np.random.seed(23)
    scheme = ESMDA(keys_da, keys_en, sim)
    scheme.run_assimilation()
    assert np.min(scheme.enX) >= -0.1
    assert np.max(scheme.enX) <= 0.1
