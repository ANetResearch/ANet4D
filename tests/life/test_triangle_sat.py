import numpy as np
import pytest

from awr.life.civil.mesh_geometry import triangle_box_intersections as reference
from awr.life.motion.triangle_sat import triangle_box_intersections as compiled


def test_random_and_degenerate_triangles_match_all_original_axes():
    rng = np.random.default_rng(184)
    for scale in (1e-4, 1.0, 1000.0):
        tri = rng.normal(size=(10000, 3, 3)) * scale
        tri[:100, 1] = tri[:100, 0]
        tri[100:200, 1:] = tri[100:200, :1]
        lo = np.array([-0.4, -0.5, -0.6]) * scale
        hi = np.array([0.8, 0.9, 1.0]) * scale
        np.testing.assert_array_equal(compiled(tri, lo, hi), reference(tri, lo, hi))


def test_closed_contacts_and_original_tolerance_are_preserved():
    lo = np.array([-1.0, -1.0, -1.0])
    hi = -lo
    rows = []
    for axis in range(3):
        for sign in (-1, 1):
            for offset in (-1e-8, 0.0, 0.5e-9, 1e-9, 1.5e-9, 1e-8):
                tri = np.zeros((3, 3))
                tri[:, axis] = sign * (1 + offset)
                tri[1, (axis + 1) % 3] = 0.5
                tri[2, (axis + 2) % 3] = 0.5
                rows.append(tri)
    tri = np.array(rows)
    np.testing.assert_array_equal(compiled(tri, lo, hi), reference(tri, lo, hi))
    assert compiled(np.array([[[1, 0, 0]] * 3]), lo, hi)[0]
    assert compiled(np.empty((0, 3, 3)), lo, hi).shape == (0,)


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        compiled([[[np.nan] * 3] * 3], [-1] * 3, [1] * 3)
    with pytest.raises(ValueError):
        compiled(np.zeros((1, 3, 3)), [1] * 3, [-1] * 3)
