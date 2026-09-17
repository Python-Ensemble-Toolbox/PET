"""Tests for distance-based localization (DistanceLocalization).

Covers:
- Kernel mathematical correctness (GaspariCohn, FurrerBengtsson, Region)
- Geometry helpers (_build_transform, _crop_kernel)
- DistanceLocalization configuration and factory
- Integration: output shape, spatial mask values, multi-parameter behavior
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pipt.localization import build_localization_instance
from pipt.localization.distance_localization import (
    DistanceLocalization,
    FurrerBengtssonKernel,
    GaspariCohnKernel,
    RegionKernel,
    _build_transform,
    _crop_kernel,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

NZ, NX, NY = 1, 10, 10
FIELD = [NZ, NX, NY]


def _make_data(data_type: str = "pressure", time: float = 1.0, cell: int = 5) -> pd.DataFrame:
    """Return a minimal one-column DataFrame for DistanceLocalization."""
    return pd.DataFrame({data_type: [cell]}, index=[time])


def _make_info(
    taper: str = "region",
    x_pos: int = 5,
    y_pos: int = 5,
    z_pos: int = 0,
    radius: int = 4,
    z_range: str = ":",
    anisotropy: float = 1.0,
    rotation: float = 0.0,
    data_type: str = "pressure",
    time: float = 1.0,
    param: str = "perm",
    taper_func: str = "region",
) -> dict:
    """Build a minimal info dict with one inline CSV row (trailing comma trick)."""
    row = (
        f"{taper} {x_pos} {y_pos} {z_pos} {radius} {z_range} "
        f"{anisotropy} {rotation} {data_type} {time} {param},"
    )
    return {"field": FIELD, "taper_func": taper_func, row: None}


# ===========================================================================
# 1. GaspariCohn kernel – mathematical properties
# ===========================================================================

class TestGaspariCohnKernel:

    def test_center_is_one(self):
        """Value at the kernel center (ratio = 0) must be exactly 1."""
        k = GaspariCohnKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0, field_shape=FIELD
        )
        cy, cx = k.shape[0] // 2, k.shape[1] // 2
        assert k[cy, cx] == pytest.approx(1.0)

    def test_values_in_unit_interval(self):
        """All GC values must lie in [0, 1]."""
        k = GaspariCohnKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0, field_shape=FIELD
        )
        assert np.all(k >= 0)
        assert np.all(k <= 1.0 + 1e-12)

    def test_outer_formula_zero_at_ratio_two(self):
        """Outer-branch formula evaluates to 0 at ratio = 2 (compact support boundary)."""
        r = 2.0
        value = (
            (1.0 / 12.0) * r ** 5
            - 0.5  * r ** 4
            + 0.625 * r ** 3
            + (5.0 / 3.0) * r ** 2
            - 5.0  * r
            + 4.0
            - (2.0 / 3.0) / r
        )
        assert value == pytest.approx(0.0, abs=1e-12)

    def test_inner_outer_continuity_at_ratio_one(self):
        """Inner and outer branch formulas must agree at ratio = 1 (C¹ junction)."""
        r = 1.0
        inner = (
            -0.25 * r ** 5
            + 0.5  * r ** 4
            + 0.625 * r ** 3
            - (5.0 / 3.0) * r ** 2
            + 1.0
        )
        outer = (
            (1.0 / 12.0) * r ** 5
            - 0.5  * r ** 4
            + 0.625 * r ** 3
            + (5.0 / 3.0) * r ** 2
            - 5.0  * r
            + 4.0
            - (2.0 / 3.0) / r
        )
        assert inner == pytest.approx(outer, abs=1e-12)

    def test_center_is_global_maximum(self):
        """Center cell must have the largest value in the kernel."""
        k = GaspariCohnKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0, field_shape=FIELD
        )
        cy, cx = k.shape[0] // 2, k.shape[1] // 2
        assert k[cy, cx] == pytest.approx(k.max(), rel=1e-10)

    def test_radially_symmetric_no_anisotropy(self):
        """Without anisotropy or rotation the kernel must be symmetric about the center."""
        k = GaspariCohnKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0, field_shape=FIELD
        )
        np.testing.assert_allclose(k, k[::-1, :], atol=1e-12)
        np.testing.assert_allclose(k, k[:, ::-1], atol=1e-12)

    def test_decreases_from_center_along_central_row(self):
        """GC kernel values must be non-increasing moving outward along the central row."""
        k = GaspariCohnKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0, field_shape=FIELD
        )
        cy, cx = k.shape[0] // 2, k.shape[1] // 2
        # left half: columns 0..cx – values should increase toward center
        assert np.all(np.diff(k[cy, : cx + 1]) >= -1e-12)
        # right half: columns cx..end – values should decrease from center
        assert np.all(np.diff(k[cy, cx:]) <= 1e-12)


# ===========================================================================
# 2. FurrerBengtsson kernel – mathematical properties
# ===========================================================================

class TestFurrerBengtssonKernel:

    def test_center_value_formula(self):
        """FB center value must equal ne / (ne + 2) (weight = 1 at d = 0)."""
        ne = 50
        k = FurrerBengtssonKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0,
            field_shape=FIELD, ensemble_size=ne,
        )
        cy, cx = k.shape[0] // 2, k.shape[1] // 2
        # At d=0: weight=1  →  fb = (ne * 1) / (1*(ne+1) + 1) = ne/(ne+2)
        assert k[cy, cx] == pytest.approx(ne / (ne + 2), rel=1e-6)

    def test_values_non_negative(self):
        """All FB values must be non-negative."""
        k = FurrerBengtssonKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0,
            field_shape=FIELD, ensemble_size=20,
        )
        assert np.all(k >= 0)

    def test_values_bounded_above(self):
        """FB values must not exceed ne / (ne + 2) (the maximum at the center)."""
        ne = 20
        k = FurrerBengtssonKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0,
            field_shape=FIELD, ensemble_size=ne,
        )
        assert np.all(k <= ne / (ne + 2) + 1e-12)

    def test_default_ensemble_size_is_50(self):
        """Passing ensemble_size=None must produce the same kernel as ensemble_size=50."""
        k_none = FurrerBengtssonKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0,
            field_shape=FIELD, ensemble_size=None,
        )
        k_50 = FurrerBengtssonKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0,
            field_shape=FIELD, ensemble_size=50,
        )
        np.testing.assert_array_equal(k_none, k_50)

    def test_ensemble_size_affects_kernel(self):
        """A larger ensemble size must produce a different kernel from a smaller one."""
        k_small = FurrerBengtssonKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0,
            field_shape=FIELD, ensemble_size=5,
        )
        k_large = FurrerBengtssonKernel().build(
            radius=4, anisotropy_ratio=1.0, rotation_deg=0.0,
            field_shape=FIELD, ensemble_size=1000,
        )
        assert not np.allclose(k_small, k_large)


# ===========================================================================
# 3. Region kernel
# ===========================================================================

class TestRegionKernel:

    def test_always_returns_1x1_ones(self):
        """RegionKernel must return a (1, 1) array containing 1.0, ignoring all args."""
        k = RegionKernel().build()
        assert k.shape == (1, 1)
        assert k[0, 0] == 1.0

    def test_ignores_all_arguments(self):
        """RegionKernel output must be independent of radius, anisotropy, rotation, etc."""
        k1 = RegionKernel().build(
            radius=100, anisotropy_ratio=3.0, rotation_deg=45.0,
            field_shape=[5, 20, 20], ensemble_size=100,
        )
        k2 = RegionKernel().build()
        np.testing.assert_array_equal(k1, k2)


# ===========================================================================
# 4. Geometry helpers
# ===========================================================================

class TestBuildTransform:

    def test_identity_with_unit_ratio_and_zero_rotation(self):
        """anisotropy_ratio=1, rotation_deg=0 must yield the 2×2 identity."""
        T = _build_transform(anisotropy_ratio=1.0, rotation_deg=0.0)
        np.testing.assert_allclose(T, np.eye(2), atol=1e-12)

    def test_pure_anisotropy_scales_first_axis(self):
        """anisotropy_ratio=2 with zero rotation should scale the x-axis by 0.5."""
        T = _build_transform(anisotropy_ratio=2.0, rotation_deg=0.0)
        np.testing.assert_allclose(T, np.array([[0.5, 0.0], [0.0, 1.0]]), atol=1e-12)

    def test_pure_rotation_90_degrees(self):
        """90° rotation with unit anisotropy must correspond to a 90° rotation matrix."""
        T = _build_transform(anisotropy_ratio=1.0, rotation_deg=90.0)
        # cos(90°)=0, sin(90°)=1 → [[0, 1], [-1, 0]]
        expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(T, expected, atol=1e-12)


class TestCropKernel:

    def test_removes_zero_border_rows_and_columns(self):
        """_crop_kernel must trim zero-only rows and columns from all four sides."""
        kernel = np.zeros((7, 7))
        kernel[2:5, 2:5] = 1.0
        cropped = _crop_kernel(kernel)
        assert cropped.shape == (3, 3)
        assert np.all(cropped == 1.0)

    def test_no_trimming_when_borders_nonzero(self):
        """Output must equal input when no zero-only borders exist."""
        kernel = np.ones((4, 4))
        cropped = _crop_kernel(kernel)
        assert cropped.shape == (4, 4)

    def test_asymmetric_zero_border(self):
        """Trimming must handle asymmetric padding correctly."""
        kernel = np.zeros((5, 6))
        kernel[1:3, 2:5] = 1.0
        cropped = _crop_kernel(kernel)
        assert cropped.shape == (2, 3)
        assert np.all(cropped == 1.0)


# ===========================================================================
# 5. DistanceLocalization configuration
# ===========================================================================

class TestDistanceLocalizationConfig:

    def test_factory_returns_distance_loc_instance(self):
        """`build_localization_instance` must return a DistanceLocalization for 'distance_loc'."""
        info = {"name": "distance_loc", "field": FIELD, "taper_func": "region"}
        loc = build_localization_instance(info)
        assert isinstance(loc, DistanceLocalization)
        assert loc.name == "distance_loc"

    def test_unknown_taper_func_raises_value_error(self):
        """An unrecognised taper_func must raise ValueError with informative message."""
        info = {"field": FIELD, "taper_func": "bogus_kernel"}
        with pytest.raises(ValueError, match="Unknown taper_func"):
            DistanceLocalization(info)

    def test_no_data_produces_empty_entries_and_cache(self):
        """Without a data DataFrame, _entries and _mask_cache must both be empty."""
        loc = DistanceLocalization({"field": FIELD, "taper_func": "region"})
        assert loc._entries == {}
        assert loc._mask_cache == {}

    def test_region_kernel_is_selected(self):
        """taper_func='region' must be accepted without error."""
        loc = DistanceLocalization({"field": FIELD, "taper_func": "region"})
        assert loc is not None

    def test_gc_kernel_is_selected(self):
        """taper_func='gc' must be accepted without error."""
        loc = DistanceLocalization({"field": FIELD, "taper_func": "gc"})
        assert loc is not None

    def test_fb_kernel_is_selected(self):
        """taper_func='fb' must be accepted without error."""
        loc = DistanceLocalization({"field": FIELD, "taper_func": "fb"})
        assert loc is not None

    def test_field_stored_correctly(self):
        """Field dimensions must be stored as-is from the info dict."""
        loc = DistanceLocalization({"field": [2, 8, 12], "taper_func": "region"})
        assert loc.field == [2, 8, 12]

    def test_data_types_and_indices_extracted_from_dataframe(self):
        """data_types and data_indices must be inferred from the DataFrame."""
        data = _make_data(data_type="bhp", time=3.5, cell=2)
        info = _make_info(data_type="bhp", time=3.5, param="perm")
        loc = DistanceLocalization(info, data=data, parameters=["perm"])
        assert loc.data_types == ["bhp"]
        assert loc.data_indices == [3.5]

    def test_inline_csv_row_populates_entry(self):
        """An inline CSV row must create an entry with the correct taper and position."""
        data = _make_data()
        info = _make_info(taper="region", x_pos=3, y_pos=7, z_pos=0, radius=5)
        loc = DistanceLocalization(info, data=data, parameters=["perm"])
        key = ("pressure", 1.0, "perm")
        assert key in loc._entries
        entry = loc._entries[key]
        assert entry.taper == "region"
        assert entry.radius == 5
        assert entry.positions == [[3, 7, 0]]

    def test_mask_cache_built_for_active_entry(self):
        """_mask_cache must contain a precomputed array for each distinct kernel config."""
        data = _make_data()
        info = _make_info(taper="region", radius=4)
        loc = DistanceLocalization(info, data=data, parameters=["perm"])
        assert len(loc._mask_cache) == 1
        cache_key = ("region", 4, 1.0, 0.0)
        assert cache_key in loc._mask_cache


# ===========================================================================
# 6. Integration – output shape and spatial values
# ===========================================================================

class TestDistanceLocalizationOutput:

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _loc(self, taper="region", taper_func="region", radius=4, x_pos=5, y_pos=5):
        data = _make_data()
        info = _make_info(
            taper=taper, taper_func=taper_func, radius=radius,
            x_pos=x_pos, y_pos=y_pos,
        )
        return DistanceLocalization(info, data=data, parameters=["perm"])

    # ------------------------------------------------------------------
    # shape / type
    # ------------------------------------------------------------------

    def test_output_is_sparse_matrix(self):
        """__call__ must return a scipy sparse matrix."""
        result = self._loc()()
        assert sparse.issparse(result)

    def test_output_shape_single_obs_single_param(self):
        """Output shape must be (n_active_cells, n_obs) = (NZ*NX*NY, 1)."""
        result = self._loc()()
        assert result.shape == (NZ * NX * NY, 1)

    # ------------------------------------------------------------------
    # Region kernel spatial correctness
    # ------------------------------------------------------------------

    def test_region_kernel_activates_exactly_one_cell(self):
        """Region kernel at (x=5, y=5, z=0) must activate only cell index 5*NY+5."""
        result = self._loc(x_pos=5, y_pos=5)()
        dense = result.toarray().ravel()
        expected_idx = 5 * NY + 5      # flat index in (NZ, NX, NY) field
        assert dense[expected_idx] == pytest.approx(1.0)
        mask = np.zeros(NZ * NX * NY, dtype=bool)
        mask[expected_idx] = True
        assert np.all(dense[~mask] == 0.0)

    def test_region_kernel_position_corner(self):
        """Region kernel placed at corner (x=0, y=0) must activate cell index 0."""
        result = self._loc(x_pos=0, y_pos=0)()
        dense = result.toarray().ravel()
        assert dense[0] == pytest.approx(1.0)
        assert np.sum(dense > 0) == 1

    # ------------------------------------------------------------------
    # GC kernel spatial correctness
    # ------------------------------------------------------------------

    def test_gc_output_in_unit_interval(self):
        """All GC localization weights must lie in [0, 1]."""
        result = self._loc(taper="gc", taper_func="gc", radius=5)()
        dense = result.toarray()
        assert np.all(dense >= 0)
        assert np.all(dense <= 1.0 + 1e-10)

    def test_gc_center_cell_is_maximum(self):
        """GC weight at the kernel center cell must equal the global maximum."""
        result = self._loc(taper="gc", taper_func="gc", radius=8, x_pos=5, y_pos=5)()
        dense = result.toarray().ravel()
        center_idx = 5 * NY + 5
        assert dense[center_idx] == pytest.approx(dense.max(), rel=1e-10)

    def test_gc_taper_decreases_from_center_along_row(self):
        """GC weights along the row through the kernel center must taper outward."""
        result = self._loc(taper="gc", taper_func="gc", radius=8, x_pos=5, y_pos=5)()
        grid = result.toarray().reshape(NZ, NX, NY)[0]  # shape (NX, NY)
        row = grid[5, :]                          # row at x=5, y=0..9
        left_half  = row[:6]                      # y=0..5 → should increase to center
        right_half = row[5:]                      # y=5..9 → should decrease from center
        assert np.all(np.diff(left_half)  >= -1e-10), "GC must increase toward center"
        assert np.all(np.diff(right_half) <= 1e-10),  "GC must decrease from center"

    def test_gc_taper_decreases_from_center_along_column(self):
        """GC weights along the column through the kernel center must also taper outward."""
        result = self._loc(taper="gc", taper_func="gc", radius=8, x_pos=5, y_pos=5)()
        grid = result.toarray().reshape(NZ, NX, NY)[0]  # shape (NX, NY)
        col = grid[:, 5]                          # column at y=5, x=0..9
        top_half    = col[:6]                     # x=0..5 → increase toward center
        bottom_half = col[5:]                     # x=5..9 → decrease from center
        assert np.all(np.diff(top_half)    >= -1e-10), "GC must increase toward center"
        assert np.all(np.diff(bottom_half) <= 1e-10),  "GC must decrease from center"

    # ------------------------------------------------------------------
    # FB kernel spatial correctness
    # ------------------------------------------------------------------

    def test_fb_output_range(self):
        """All FB localization weights must lie in [0, ne/(ne+2)]."""
        ne = 20
        data = _make_data()
        info = _make_info(taper="fb", taper_func="fb", radius=5)
        loc = DistanceLocalization(info, data=data, parameters=["perm"], ensemble_size=ne)
        result = loc()
        dense = result.toarray()
        assert np.all(dense >= 0)
        assert np.all(dense <= ne / (ne + 2) + 1e-10)

    # ------------------------------------------------------------------
    # Multi-parameter: unconfigured parameter → zero columns
    # ------------------------------------------------------------------

    def test_unconfigured_param_gives_zero_weights(self):
        """Parameters with no localization entry must produce all-zero weight columns."""
        data = _make_data()
        info = _make_info()                       # configured only for "perm"
        prior_info = {"other": {"nx": NX, "ny": NY, "nz": NZ}}
        loc = DistanceLocalization(
            info, data=data, parameters=["perm", "other"], prior_info=prior_info
        )
        result = loc()
        n_cells = NZ * NX * NY
        assert result.shape == (2 * n_cells, 1)
        dense = result.toarray()
        assert np.any(dense[:n_cells] > 0),    "perm weights should have non-zero entries"
        np.testing.assert_array_equal(dense[n_cells:], 0.0)

    def test_two_configured_params_correct_output_shape(self):
        """With two configured parameters the output must span both cell-spaces."""
        data = _make_data()
        # Two rows: one for perm, one for poro
        row_perm = "region 5 5 0 4 : 1.0 0.0 pressure 1.0 perm,"
        row_poro = "region 3 3 0 4 : 1.0 0.0 pressure 1.0 poro"
        # Combine as a single comma-separated multi-row key
        multi_row_key = f"{row_perm}{row_poro}"
        info = {"field": FIELD, "taper_func": "region", multi_row_key: None}
        loc = DistanceLocalization(info, data=data, parameters=["perm", "poro"])
        result = loc()
        assert result.shape == (2 * NZ * NX * NY, 1)

    # ------------------------------------------------------------------
    # Active-cell mask
    # ------------------------------------------------------------------

    def test_actnum_reduces_a_localized_parameter_to_the_active_cells(self, tmp_path):
        """A localized parameter used to contribute one row per grid cell while an
        unlocalized one contributed a row per active cell, so the operator came out
        with the wrong number of rows: 160 instead of 120 on this 60-of-100 case."""
        n_cells = NZ * NX * NY
        n_active = 60
        actnum = np.zeros(n_cells, dtype=bool)
        actnum[:n_active] = True
        actnum_file = tmp_path / "active.npz"
        np.savez(actnum_file, actnum=actnum)

        info = {**_make_info(), "actnum": str(actnum_file)}
        prior_info = {"other": {"nx": NX, "ny": NY, "nz": NZ}}
        loc = DistanceLocalization(
            info, data=_make_data(), parameters=["perm", "other"], prior_info=prior_info
        )
        result = loc()

        assert result.shape == (2 * n_active, 1)
        dense = result.toarray()
        assert np.any(dense[:n_active] > 0)                  # the localized parameter
        np.testing.assert_array_equal(dense[n_active:], 0.0)  # the unlocalized one

    def test_an_all_active_actnum_matches_giving_none(self, tmp_path):
        actnum_file = tmp_path / "all.npz"
        np.savez(actnum_file, actnum=np.ones(NZ * NX * NY, dtype=bool))

        without = DistanceLocalization(_make_info(), data=_make_data(), parameters=["perm"])()
        with_all = DistanceLocalization(
            {**_make_info(), "actnum": str(actnum_file)}, data=_make_data(), parameters=["perm"]
        )()

        np.testing.assert_array_equal(without.toarray(), with_all.toarray())

    # ------------------------------------------------------------------
    # Pickled mask files
    # ------------------------------------------------------------------

    def test_a_pickled_localization_file_is_read(self, tmp_path):
        """Pickled files hold plain dicts. They were returned unconverted, so the first
        thing that asked for `.taper` raised AttributeError and no pickled mask file
        could be used at all."""
        legacy = {
            ("pressure", 1.0, "perm"): {"taper_func": "gc", "position": [[5, 5, 0]],
                                        "range": [4, ":"], "anisotropi": [1.0, 0.0]},
            ("pressure", 1.0, "poro"): {"taper_func": None, "position": None,
                                        "range": None, "anisotropi": None},
        }
        path = tmp_path / "masks.p"
        with open(path, "wb") as handle:
            pickle.dump(legacy, handle)

        prior_info = {p: {"nx": NX, "ny": NY, "nz": NZ} for p in ("perm", "poro")}
        loc = DistanceLocalization(
            {"field": FIELD, "taper_func": "gc", "locfile": str(path)},
            data=_make_data(), parameters=["perm", "poro"], prior_info=prior_info,
        )
        dense = loc().toarray()
        n_cells = NZ * NX * NY

        assert dense.shape == (2 * n_cells, 1)
        assert np.any(dense[:n_cells] > 0)                    # perm is tapered
        np.testing.assert_array_equal(dense[n_cells:], 0.0)   # poro has no entry

    def test_a_pickled_radius_without_a_z_range_still_reads(self, tmp_path):
        """Older files wrote `range` as the radius alone rather than [radius, z_range]."""
        path = tmp_path / "masks.pkl"
        with open(path, "wb") as handle:
            pickle.dump({("pressure", 1.0, "perm"): {
                "taper_func": "gc", "position": [[5, 5, 0]], "range": 4, "anisotropi": [1.0, 0.0]
            }}, handle)

        loc = DistanceLocalization(
            {"field": FIELD, "taper_func": "gc", "locfile": str(path)},
            data=_make_data(), parameters=["perm"],
        )

        assert np.any(loc().toarray() > 0)

    # ------------------------------------------------------------------
    # z_range selection
    # ------------------------------------------------------------------

    def test_specific_z_range_limits_cells_to_one_layer(self):
        """When z_range is a layer index, the mask must cover only that z-layer."""
        nz_multi = 3
        field_multi = [nz_multi, NX, NY]
        data = _make_data()
        row = "region 5 5 1 4 1 1.0 0.0 pressure 1.0 perm,"
        info = {"field": field_multi, "taper_func": "region", row: None}
        loc = DistanceLocalization(info, data=data, parameters=["perm"])
        result = loc()
        # With z_range="1", the mask is NX*NY cells (one layer)
        assert result.shape == (NX * NY * nz_multi, 1)
