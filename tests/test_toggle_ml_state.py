"""
Test suite for toggle_ml_state function in popt.misc_tools.optim_tools
"""

import numpy as np
import pytest
from popt.misc_tools.optim_tools import toggle_ml_state


def test_toggle_ml_state_matrix_to_list():
    """Test converting a matrix state to a list of levels"""
    # Create a sample state matrix (rows=state_dim, cols=total_ensemble)
    state_dim = 10
    total_ensemble = 15
    state = np.random.rand(state_dim, total_ensemble)
    
    # Define multilevel ensemble sizes
    ml_ne = [5, 7, 3]  # 3 levels with 5, 7, and 3 members respectively
    
    # Toggle to list format
    result = toggle_ml_state(state, ml_ne)
    
    # Check that result is a list
    assert isinstance(result, list)
    
    # Check that we have the correct number of levels
    assert len(result) == len(ml_ne)
    
    # Check that each level has the correct ensemble size
    for i, ne in enumerate(ml_ne):
        assert result[i].shape == (state_dim, ne)
    
    # Check that the data is correctly distributed
    start = 0
    for i, ne in enumerate(ml_ne):
        stop = start + ne
        np.testing.assert_array_equal(result[i], state[:, start:stop])
        start = stop


def test_toggle_ml_state_list_to_matrix():
    """Test converting a list of levels back to a matrix state"""
    # Create sample state as list of levels
    state_dim = 10
    ml_ne = [5, 7, 3]
    
    state_list = [
        np.random.rand(state_dim, ml_ne[0]),
        np.random.rand(state_dim, ml_ne[1]),
        np.random.rand(state_dim, ml_ne[2])
    ]
    
    # Toggle to matrix format
    result = toggle_ml_state(state_list, ml_ne)
    
    # Check that result is a numpy array
    assert isinstance(result, np.ndarray)
    
    # Check dimensions
    total_ensemble = sum(ml_ne)
    assert result.shape == (state_dim, total_ensemble)
    
    # Check that data is correctly concatenated
    start = 0
    for i, ne in enumerate(ml_ne):
        stop = start + ne
        np.testing.assert_array_equal(result[:, start:stop], state_list[i])
        start = stop


def test_toggle_ml_state_roundtrip():
    """Test that toggling back and forth preserves the data"""
    # Create initial state matrix
    state_dim = 8
    total_ensemble = 12
    original_state = np.random.rand(state_dim, total_ensemble)
    
    ml_ne = [4, 5, 3]
    
    # Toggle to list then back to matrix
    state_list = toggle_ml_state(original_state, ml_ne)
    restored_state = toggle_ml_state(state_list, ml_ne)
    
    # Check that we get back the original state
    np.testing.assert_array_equal(restored_state, original_state)


def test_toggle_ml_state_single_level():
    """Test with a single level (edge case)"""
    state_dim = 5
    ensemble_size = 10
    state = np.random.rand(state_dim, ensemble_size)
    
    ml_ne = [ensemble_size]
    
    # Toggle to list
    result = toggle_ml_state(state, ml_ne)
    
    assert isinstance(result, list)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], state)
    
    # Toggle back
    restored = toggle_ml_state(result, ml_ne)
    np.testing.assert_array_equal(restored, state)


def test_toggle_ml_state_many_levels():
    """Test with many levels"""
    state_dim = 6
    ml_ne = [2, 3, 1, 4, 2, 3]  # 6 levels
    total_ensemble = sum(ml_ne)
    
    state = np.random.rand(state_dim, total_ensemble)
    
    # Toggle to list
    result = toggle_ml_state(state, ml_ne)
    
    assert len(result) == len(ml_ne)
    for i, ne in enumerate(ml_ne):
        assert result[i].shape[1] == ne
    
    # Toggle back and verify
    restored = toggle_ml_state(result, ml_ne)
    np.testing.assert_array_equal(restored, state)


def test_toggle_ml_state_preserves_values():
    """Test that specific values are preserved correctly"""
    # Create a state with known values for verification
    state = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0, 40.0, 50.0]
    ])
    
    ml_ne = [2, 3]
    
    # Toggle to list
    result = toggle_ml_state(state, ml_ne)
    
    # Check first level
    expected_level0 = np.array([[1.0, 2.0], [10.0, 20.0]])
    np.testing.assert_array_equal(result[0], expected_level0)
    
    # Check second level
    expected_level1 = np.array([[3.0, 4.0, 5.0], [30.0, 40.0, 50.0]])
    np.testing.assert_array_equal(result[1], expected_level1)


def test_toggle_ml_state_empty_level():
    """Test behavior with empty levels (ensemble size = 0)"""
    state_dim = 4
    ml_ne = [3, 0, 2]  # Middle level has no members
    total_ensemble = sum(ml_ne)
    
    state = np.random.rand(state_dim, total_ensemble)
    
    # Toggle to list
    result = toggle_ml_state(state, ml_ne)
    
    assert len(result) == len(ml_ne)
    assert result[0].shape == (state_dim, 3)
    assert result[1].shape == (state_dim, 0)  # Empty array
    assert result[2].shape == (state_dim, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
