import numpy as np
import pytest

from popt.optimization_methods import BoundTransformHandler

def test_finite_bounds():
    bounds = [(0.0, 10.0)]
    h = BoundTransformHandler(bounds, transform=True)
    x = np.array([5.0])
    u = h.state_to_unit_cube(x)
    assert np.allclose(u, [0.5])


def test_multiple_dimensions():
    bounds = [
        (0.0, 10.0),
        (-5.0, 5.0),
        (100.0, 200.0),
    ]
    h = BoundTransformHandler(bounds, transform=True)
    x = np.array([5.0, 0.0, 150.0])
    u = h.state_to_unit_cube(x)
    expected = np.array([
        0.5,
        0.5,
        0.5,
    ])
    assert np.allclose(u, expected)


def test_lower_boundary():
    bounds = [
        (0.0, 10.0),
        (-5.0, 5.0),
    ]
    h = BoundTransformHandler(bounds, transform=True)
    x = np.array([0.0, -5.0])
    u = h.state_to_unit_cube(x)
    assert np.allclose(u, [0.0, 0.0])


def test_upper_boundary():
    bounds = [
        (0.0, 10.0),
        (-5.0, 5.0),
    ]
    h = BoundTransformHandler(bounds, transform=True)
    x = np.array([10.0, 5.0])
    u = h.state_to_unit_cube(x)
    assert np.allclose(u, [1.0, 1.0])


def test_no_transform():
    bounds = [(0.0, 10.0)]
    h = BoundTransformHandler(bounds, transform=False)
    x = np.array([7.5])
    assert np.allclose(
        h.state_to_unit_cube(x),
        x,
    )


def test_none_bounds():
    h = BoundTransformHandler(None, transform=True)
    x = np.array([1.0, 2.0])
    assert np.allclose(
        h.state_to_unit_cube(x),
        x,
    )


def test_round_trip_single_dimension():
    bounds = [(0.0, 10.0)]
    h = BoundTransformHandler(bounds, transform=True)
    x = np.array([7.5])
    u = h.state_to_unit_cube(x)
    x2 = h.unit_cube_to_state(u)
    assert np.allclose(x, x2)


def test_round_trip_multiple_dimensions():
    bounds = [
        (0.0, 10.0),
        (-5.0, 5.0),
        (100.0, 200.0),
    ]
    h = BoundTransformHandler(bounds, transform=True)
    x = np.array([3.7,-1.4, 175.8])
    u = h.state_to_unit_cube(x)
    x2 = h.unit_cube_to_state(u)
    assert np.allclose(x, x2)


def test_round_trip_random_points():
    bounds = [
        (-5.0, 5.0),
        (0.0, 100.0),
        (10.0, 20.0),
    ]
    h = BoundTransformHandler(bounds, transform=True)
    rng = np.random.default_rng(42)
    for _ in range(1000):
        x = np.array([
            rng.uniform(-5.0, 5.0),
            rng.uniform(0.0, 100.0),
            rng.uniform(10.0, 20.0),
        ])
        u = h.state_to_unit_cube(x)
        x2 = h.unit_cube_to_state(u)
        assert np.allclose(x, x2)


def test_project_state_space():
    bounds = [
        (0.0, 10.0),
        (-5.0, 5.0),
    ]
    h = BoundTransformHandler(bounds, transform=False)
    x = np.array([-1.0, 10.0])

    projected = h.project_to_bounds(x)
    assert np.allclose(
        projected,
        [0.0, 5.0],
    )


def test_project_unit_cube():
    bounds = [
        (0.0, 10.0),
        (-5.0, 5.0),
    ]
    h = BoundTransformHandler(bounds, transform=True)
    u = np.array([-0.2, 1.5])

    projected = h.project_to_bounds(u)
    assert np.allclose(
        projected,
        [0.0, 1.0],
    )


def test_invalid_state_outside_bounds():
    bounds = [(0.0, 10.0)]
    h = BoundTransformHandler(bounds, transform=True)
    with pytest.raises(ValueError):
        h.state_to_unit_cube(np.array([11.0]))


def test_invalid_unit_cube_coordinate():
    bounds = [(0.0, 10.0)]
    h = BoundTransformHandler(bounds, transform=True)
    with pytest.raises(ValueError):
        h.unit_cube_to_state(np.array([1.1]))


def test_invalid_bounds():
    with pytest.raises(ValueError):
        BoundTransformHandler([(10.0, 0.0)])
