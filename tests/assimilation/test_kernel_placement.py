"""A kernel is placed on the grid with its own axes, whatever its shape.

Kernels are built ``(nx, ny)``-major, like the field ``(nz, nx, ny)``. Placement
used to unpack the kernel as ``(ky, kx)``, which only worked for square,
symmetric kernels placed away from the edges: an anisotropic kernel raised a
shape error everywhere, and an isotropic one raised near any edge where the x
and y clipping differed.
"""

import numpy as np
import pytest

from pipt.localization.distance_localization import DistanceLocalization


def _placer(field):
    loc = object.__new__(DistanceLocalization)   # _place_kernel reads only self.field
    loc.field = field
    return loc


def test_anisotropic_kernel_is_placed_with_its_own_orientation():
    loc = _placer((2, 20, 30))
    kernel = np.arange(3 * 5, dtype=float).reshape(3, 5) + 1   # kx = 3, ky = 5

    placed = loc._place_kernel(kernel, [10, 15, 0])

    np.testing.assert_array_equal(placed[0, 9:12, 13:18], kernel)
    assert placed.sum() == kernel.sum()
    assert not placed[1].any()


def test_kernel_is_clipped_consistently_at_a_corner():
    loc = _placer((2, 20, 30))
    kernel = np.arange(3 * 5, dtype=float).reshape(3, 5) + 1

    placed = loc._place_kernel(kernel, [0, 0, 1])

    # x_min = -1 keeps kernel rows 1:3 on grid rows 0:2; y_min = -2 keeps
    # kernel columns 2:5 on grid columns 0:3.
    np.testing.assert_array_equal(placed[1, 0:2, 0:3], kernel[1:3, 2:5])
    assert placed.sum() == kernel[1:3, 2:5].sum()


@pytest.mark.parametrize("position", [[1, 15, 0], [10, 1, 0], [19, 29, 1]])
def test_isotropic_kernel_survives_every_edge(position):
    loc = _placer((2, 20, 30))
    kernel = np.ones((5, 5))

    placed = loc._place_kernel(kernel, position)

    assert 0 < placed.sum() <= kernel.sum()
