import unittest

import numpy as np
from scipy.spatial import cKDTree

from awr.life.civil.collision import actor_blocked, collision_pairs, overlapping, rectangle
from awr.life.civil.surface import SurfaceError, SurfaceMap


class TestBatchedGeometry(unittest.TestCase):
    def test_sat_matches_polygon_reference_and_neighbor_queries(self):
        rng = np.random.default_rng(2051)
        for _ in range(12):
            xy = rng.uniform(-10, 10, (80, 2))
            yaw = rng.uniform(-np.pi, np.pi, 80)
            sizes = rng.uniform(0.1, 8, (80, 3))
            polygons = [rectangle(p, a, s) for p, a, s in zip(xy, yaw, sizes, strict=False)]
            expected = [(i, j) for i in range(80) for j in range(i + 1, 80) if overlapping(polygons[i], polygons[j])]
            self.assertEqual(collision_pairs(xy, yaw, sizes), expected)
            radii = np.linalg.norm(sizes[:, :2], axis=1) / 2
            tree = cKDTree(xy)
            for i in range(80):
                self.assertEqual(
                    actor_blocked(i, xy[i], yaw[i], xy, yaw, sizes, tree, radii), any(i in p for p in expected)
                )

    def test_touching_and_containment(self):
        size = np.array([[4.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertEqual(collision_pairs(np.array([[0.0, 0.0], [2.5, 0.0]]), np.zeros(2), size), [])
        self.assertEqual(collision_pairs(np.zeros((2, 2)), np.zeros(2), size), [(0, 1)])

    def test_batched_support_matches_sample_including_invalid_cells(self):
        s = SurfaceMap.__new__(SurfaceMap)
        s.x0 = -10
        s.y0 = 5
        s.nx = 20
        s.ny = 20
        s.cell_m = 1.0
        s.z = np.zeros((20, 20))
        s.z[5, 5] = np.nan
        mask = np.ones((20, 20), bool)
        mask[10:12, 10:12] = False
        s.walk = {"person": mask}
        rng = np.random.default_rng(2048)
        xy = np.vstack([rng.uniform([-11, 4], [11, 26], (1000, 2)), [-9.5, 5.5], [9.5, 24.5], [np.nan, 1], [np.inf, 3]])
        expected = []
        for p in xy:
            try:
                s.sample(p)
                expected.append(True)
            except SurfaceError:
                expected.append(False)
        np.testing.assert_array_equal(s.supported(xy), expected)


class TestStoppingCorridor(unittest.TestCase):
    def test_approaching_body_reserves_braking_sweep(self):
        xy = np.array([[0.0, 0.0], [5.0, 0.0]])
        yaw = np.array([0.0, np.pi])
        sizes = np.array([[2.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
        radii = np.linalg.norm(sizes[:, :2], axis=1) / 2
        self.assertFalse(actor_blocked(0, xy[0], 0, xy, yaw, sizes, cKDTree(xy), radii))
        future = xy.copy()
        future[1, 0] -= 2.5
        envelope = sizes.copy()
        envelope[1, 0] += 5
        radii = np.linalg.norm(envelope[:, :2], axis=1) / 2
        self.assertTrue(actor_blocked(0, xy[0], 0, future, yaw, envelope, cKDTree(future), radii, body_size=sizes[0]))
