import numpy as np
import pytest

from awr.life.civil.support_recovery import recover


def flat(xy, hint):
    return np.zeros(len(xy)), np.ones(len(xy), bool)


def test_propagation_is_measured_bounded_and_batch_independent():
    z = np.full((7, 9), np.nan)
    z[:, 0] = 0
    before = z.copy()
    allow = np.ones_like(z, bool)
    a, info = recover(z, allow, 1.0, flat, max_rounds=3, batch=2)
    b, other = recover(z, allow, 1.0, flat, max_rounds=3, batch=999)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(z, before)
    assert info == other and info["recovered_cells"] == 21
    assert np.isfinite(a[:, :4]).all() and np.isnan(a[:, 4:]).all()


def test_hint_is_never_used_as_a_height_without_mesh_hit():
    z = np.array([[0.0, np.nan, np.nan]])
    out, info = recover(z, np.ones_like(z, bool), 1.0, lambda xy, h: (h, np.zeros(len(h), bool)))
    np.testing.assert_array_equal(z, out)
    assert info["recovered_cells"] == 0


def test_cannot_cross_unknown_gap_permission_hole_or_roof_jump():
    z = np.full((1, 7), np.nan)
    z[0, 0] = 0
    allow = np.ones_like(z, bool)

    def gap(xy, h):
        return np.zeros(len(h)), xy[:, 0] < 2

    out, _ = recover(z, allow, 1.0, gap)
    assert np.isfinite(out[0, :2]).all() and np.isnan(out[0, 2:]).all()
    allow[0, 2] = False
    out, _ = recover(z, allow, 1.0, flat)
    assert np.isnan(out[0, 2:]).all()
    out, _ = recover(z, np.ones_like(z, bool), 1.0, lambda xy, h: (h + 2, np.ones(len(h), bool)))
    np.testing.assert_array_equal(out, z)


def test_agrees_with_all_measured_neighbours_and_preserves_existing():
    z = np.array([[0.0, np.nan, 0.6]])
    out, _ = recover(z, np.ones_like(z, bool), 1.0, lambda xy, h: (h, np.ones(len(h), bool)))
    np.testing.assert_array_equal(out, z)
    z = np.array([[0.0, np.nan, 0.4]])
    out, _ = recover(z, np.ones_like(z, bool), 1.0, lambda xy, h: (h, np.ones(len(h), bool)))
    assert out[0, 1] == pytest.approx(0.2)


def test_slope_and_origin_are_measured_not_flattened():
    z = np.full((1, 5), np.nan)
    z[0, 0] = 2.05

    def slope(xy, h):
        return 0.1 * (xy[:, 0] - 100) + 2, np.ones(len(h), bool)

    out, _ = recover(z, np.ones_like(z, bool), 1.0, slope, origin=(100.0, -20.0))
    np.testing.assert_allclose(out, [[2.05, 2.15, 2.25, 2.35, 2.45]])


def test_bad_callback_or_geometry_fails_closed():
    z = np.array([[0.0, np.nan]])
    with pytest.raises(ValueError):
        recover(z, np.ones_like(z, bool), 0.0, flat)
    with pytest.raises(ValueError):
        recover(z, np.ones_like(z, bool), 1.0, lambda xy, h: (h, True))
