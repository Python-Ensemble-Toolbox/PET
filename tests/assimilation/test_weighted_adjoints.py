"""A weighted adjoint reaches the scheme as an ``(nx, ne)`` gradient.

A simulator in weighted-adjoint mode is handed weights ``w`` and returns the
gradient of ``sum_i w_i d_i`` per parameter, one adjoint solve per member.
"""

import numpy as np
import pandas as pd
import pytest

from pipt.ensembles import ForecastMixin


# ---------------------------------------------------------------------- #
# Weighted adjoints into the state
# ---------------------------------------------------------------------- #
def _ensemble_with_two_states():
    class Holder(ForecastMixin):
        idX = {"permx": (0, 3), "poro": (3, 5)}

    return Holder()


def test_weighted_gradients_are_placed_by_name_not_order():
    holder = _ensemble_with_two_states()
    member = pd.Series({"poro": np.array([4.0, 5.0]), "log_permx": np.array([1.0, 2.0, 3.0])}, dtype=object)
    np.testing.assert_array_equal(holder._weighted_gradient(member), [1, 2, 3, 4, 5])


@pytest.mark.parametrize("member", [
    pd.Series({"permx": np.zeros(3)}, dtype=object),                                         # one missing
    pd.Series({"permx": np.zeros(3), "permy": np.zeros(2)}, dtype=object),                  # wrong name
    pd.Series({"permx": np.zeros(3), "log_permx": np.zeros(3), "poro": np.zeros(2)}, dtype=object),  # twice
    pd.Series({"permx": np.zeros(2), "poro": np.zeros(3)}, dtype=object),                   # wrong sizes
])
def test_weighted_gradients_that_do_not_match_the_state_are_refused(member):
    with pytest.raises(ValueError, match="weighted adjoint"):
        _ensemble_with_two_states()._weighted_gradient(member)
