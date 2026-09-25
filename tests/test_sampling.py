"""``misc.sampling`` draws exactly what geostat drew, from whichever stream it is handed."""

import pickle

import numpy as np
import pytest
from geostat.decomp import Cholesky

from misc.sampling import GlobalRandomStream, gen_real, random_stream


def _spd(n, seed):
    a = np.random.RandomState(seed).randn(n, n)
    return a @ a.T + n * np.eye(n)


CASES = {
    "variance vector": (np.arange(1.0, 5.0), np.array([0.5, 1.0, 2.0, 4.0])),
    "diagonal covariance": (np.arange(1.0, 5.0), np.diag([0.5, 1.0, 2.0, 4.0])),
    "full covariance": (np.arange(1.0, 5.0), _spd(4, 3)),
    "single element": (np.array([2.0]), np.array(9.0)),
}


@pytest.mark.parametrize("mean, var", CASES.values(), ids=CASES.keys())
@pytest.mark.parametrize("limits", [None, {"lower": 0.5, "upper": 3.0}])
def test_gen_real_reproduces_geostat_draw_for_draw(mean, var, limits):
    np.random.seed(11)
    expected, expected_factor = Cholesky().gen_real(mean, var, 7, limits=limits, return_chol=True)
    np.random.seed(11)
    actual, actual_factor = gen_real(mean, var, 7, limits=limits, return_chol=True)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_factor, expected_factor)


def test_gen_real_draws_from_the_stream_it_is_handed():
    mean, var = CASES["full covariance"]
    a = gen_real(mean, var, 5, rng=np.random.RandomState(4))
    b = gen_real(mean, var, 5, rng=np.random.RandomState(4))
    c = gen_real(mean, var, 5, rng=np.random.RandomState(5))

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_default_stream_is_the_global_one():
    np.random.seed(2)
    expected = np.random.randn(3, 2)
    np.random.seed(2)
    stream = random_stream(None)

    assert isinstance(stream, GlobalRandomStream)
    np.testing.assert_array_equal(stream.randn(3, 2), expected)


def test_a_seed_gives_a_private_stream():
    assert isinstance(random_stream(7), np.random.RandomState)
    np.testing.assert_array_equal(random_stream(7).randn(4), random_stream(7).randn(4))
    np.testing.assert_array_equal(random_stream("7").randn(4), random_stream(7).randn(4))


def test_the_global_stream_survives_pickling():
    # The ensemble is pickled by its emergency dump; the numpy.random module itself cannot be.
    stream = pickle.loads(pickle.dumps(GlobalRandomStream()))
    assert isinstance(stream, GlobalRandomStream)
    np.random.seed(9)
    expected = np.random.permutation(6)
    np.random.seed(9)
    np.testing.assert_array_equal(stream.permutation(6), expected)
