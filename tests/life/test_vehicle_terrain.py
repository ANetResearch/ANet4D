"""Continuous terrain proof rejects hazards hidden between route samples."""

import numpy as np
import pytest

from awr.life.civil.surface import SurfaceMap
from awr.life.civil.vehicle_terrain import TrianglePatch, VehicleTerrain

SIZE = np.array([2.0, 1.0, 1.0])
CONTACTS = np.array([[-0.64, -0.37], [-0.64, 0.37], [0.64, -0.37], [0.64, 0.37]])


def terrain(*, grade=(0.0, 0.0), extra=(), bounds=(-10.0, -10.0, 10.0, 10.0), depth=12):
    s = SurfaceMap.__new__(SurfaceMap)
    s.x0 = s.y0 = -10.0
    s.cell_m = 0.1
    s.nx = s.ny = 200
    yy, xx = np.mgrid[:200, :200]
    s.z = grade[0] * (s.x0 + (xx + 0.5) * s.cell_m) + grade[1] * (s.y0 + (yy + 0.5) * s.cell_m)
    s.walk = {"vehicle": np.ones((200, 200), bool)}
    v = np.array([[-10, -10], [-10, 10], [10, 10], [10, -10]], float)
    v = np.column_stack([v, v @ grade])
    triangles = np.concatenate([v[[[0, 1, 2], [0, 2, 3]]], np.asarray(extra).reshape(-1, 3, 3)])
    return VehicleTerrain(
        s, TrianglePatch(triangles, bounds), SIZE, CONTACTS, [-1, -0.5, 0.2], [1, 0.5, 1.0], max_depth=depth
    )


def mask(t, points, yaw=None):
    return t.vehicle_support_mask(points, np.zeros(len(points)) if yaw is None else yaw, SIZE)


def block(t, xy, *, missing=False):
    x, y = np.floor((np.asarray(xy) + 10) / 0.1).astype(int)
    if missing:
        t.surface.z[y, x] = np.nan
    else:
        t.surface.walk["vehicle"][y, x] = False


def test_supported_tyres_and_clear_body_do_not_need_ground_under_overhang():
    t = terrain()
    block(t, [0.95, 0.45])
    assert mask(t, [[0, 0]]).all()
    block(t, CONTACTS[0])
    assert not mask(t, [[0, 0]]).any()


def test_continuous_flat_and_sloped_ground_paths_are_certified():
    for grade in ((0.0, 0.0), (0.04, 0.02)):
        t = terrain(grade=grade)
        assert mask(t, [[-3, 0], [3, 0]], [-0.1, 0.1]).all()
        assert t.stats["certified_intervals"] > 2
        assert not t.release_ready


def test_obstacle_between_clear_endpoints_rejects_entire_crossing():
    t = terrain(extra=[[[0.43, -0.2, 0.3], [0.43, 0.2, 0.3], [0.43, 0, 0.9]]])
    assert mask(t, [[-3, 0]]).all() and mask(t, [[3, 0]]).all()
    assert not mask(t, [[-3, 0], [3, 0]]).any()
    assert t.stats["body_collision"] > 0


@pytest.mark.parametrize("missing", [False, True])
def test_support_hole_between_endpoints_never_becomes_free_space(missing):
    t = terrain()
    block(t, [0.43, 0], missing=missing)
    assert mask(t, [[-3, 0]]).all() and mask(t, [[3, 0]]).all()
    assert not mask(t, [[-3, 0], [3, 0]]).any()


def test_pose_probes_are_required_even_when_caller_supplies_only_tyres():
    t = terrain()
    block(t, [0, 0.37])
    assert not mask(t, [[0, 0]]).any()


def test_patch_boundary_and_subdivision_exhaustion_fail_closed():
    assert not mask(terrain(bounds=(-0.9, -0.9, 0.9, 0.9)), [[0, 0]]).any()
    t = terrain(depth=1)
    assert not mask(t, [[-3, 0], [3, 0]]).any()
    assert t.stats["uncertified_interval"] > 0


def test_failed_interval_does_not_poison_later_unrelated_intervals():
    t = terrain()
    block(t, [-3, 0])
    # First failed endpoint invalidates its adjoining interval only.
    assert mask(t, [[-3, 0], [0, 0], [3, 0], [4, 0]]).tolist() == [False, False, True, True]


def test_invalid_mesh_and_mismatched_vehicle_rejected():
    with pytest.raises(ValueError):
        TrianglePatch(np.full((1, 3, 3), np.nan), [-1, -1, 1, 1])
    with pytest.raises(ValueError):
        terrain().vehicle_support_mask([[0, 0]], [0], [3, 1, 1])
    with pytest.raises(ValueError):
        mask(terrain(), [[np.nan, 0]])


def test_early_rejection_agrees_with_full_mask_on_clear_and_blocked_routes():
    from awr.life.civil.curb import valid_route
    from awr.life.civil.driving_graph import route_arrays

    rng = np.random.default_rng(34)
    for case in range(24):
        t = terrain(grade=(0.02, 0.01))
        points = np.column_stack([np.linspace(-3, 3, 9), np.full(9, rng.uniform(-1, 1))])
        yaw = np.linspace(-0.05, 0.05, 9)
        if case % 3 == 1:
            block(t, points[4])
        if case % 3 == 2:
            block(t, (points[4] + points[5]) / 2, missing=True)
        full = bool(mask(t, points, yaw).all())
        assert t.vehicle_route_supported(points, yaw, SIZE) == full
        assert valid_route(t, route_arrays(points, yaw), SIZE) == full


def test_early_route_rejection_does_not_do_unneeded_mesh_proofs():
    t = terrain()
    block(t, [0, 0])
    assert not t.vehicle_route_supported([[-3, 0], [0, 0], [3, 0]], [0, 0, 0], SIZE)
    assert t.stats["early_unsupported_route"] == 1
    assert t.stats["intervals"] == 0
