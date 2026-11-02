import unittest

import numpy as np

from awr.life.civil.grounded import at, route_length
from awr.life.civil.surface import SurfaceMap
from awr.life.civil.walking import WalkNetwork, polyline


class WalkingNetworkTests(unittest.TestCase):
    def test_dynamic_obstacle_detour_preserves_exact_start_and_destination(self):
        free = np.ones((40, 80), bool)
        network = WalkNetwork(free, [0.0, 0.0])
        start, goal = np.array([1.13, 9.91]), np.array([35.0, 10.0])
        blocked = network.obstacle_mask([[18.0, 10.0]], [[4.0, 4.0, 2.0]], 0.7)
        route = network.plan(start, goal, blocked)
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route["points"][0], start)
        np.testing.assert_array_equal(route["points"][-1], goal)
        self.assertGreater(route_length(route), np.linalg.norm(goal - start))
        for a, b in zip(route["points"][:-1], route["points"][1:], strict=False):
            self.assertTrue(network.clear_segment(a, b, blocked))

    def test_disconnected_map_and_blocked_current_pose_do_not_snap(self):
        free = np.ones((40, 80), bool)
        free[:, 40] = False
        network = WalkNetwork(free, [0.0, 0.0])
        self.assertIsNone(network.plan([5.0, 10.0], [30.0, 10.0]))
        self.assertIsNone(network.plan([20.0, 10.0], [30.0, 10.0]))

    def test_only_annotated_crossing_opens_road_interior(self):
        surface = SurfaceMap.__new__(SurfaceMap)
        surface.x0 = surface.y0 = 0
        surface.nx = 80
        surface.ny = 60
        surface.cell_m = 1.0
        surface.width_m = 80
        surface.height_m = 60
        surface.z = np.zeros((60, 80))
        surface.walk = {"person": np.ones((60, 80), bool)}
        # Road runs beyond the map, so its endpoints cannot provide a bypass.
        sem = {"roads": [{"line": [[-10, 30], [90, 30]], "width_m": 6}], "crossings": []}
        network = WalkNetwork.from_scene(surface, sem, 0.7)
        self.assertIsNone(network.plan([10.25, 25.25], [60.25, 34.75]))
        sem["crossings"] = [{"ends": [[35, 27], [35, 33]], "width_m": 5}]
        network = WalkNetwork.from_scene(surface, sem, 0.7)
        route = network.plan([10.25, 25.25], [60.25, 34.75])
        self.assertIsNotNone(route)
        for d in np.arange(0, route_length(route), 0.1):
            p, _ = at(route, d)
            if 27 < p[1] < 33:
                self.assertLess(abs(p[0] - 35), 2.5)

    def test_walker_inside_planning_margin_can_step_away_and_pass(self):
        # Two walkers stopped face to face, closer than the planning margin
        # (legal: their boxes do not touch). The one at A must still find a
        # route past B; its first connector never closes the gap to B.
        from awr.life.civil.walking import PLANNING_MARGIN_M

        network = WalkNetwork(np.ones((40, 80), bool), [0.0, 0.0])
        body = np.array([1.2, 0.7, 1.7])
        free = network.obstacle_mask([[20.0, 10.0]], [body], 0.35, [0.0])
        start, goal = np.array([18.95, 10.0]), np.array([30.0, 10.0])
        limit = 0.35 + 0.08 + PLANNING_MARGIN_M

        def gap(p):
            return float(np.linalg.norm(np.maximum(np.abs(np.asarray(p) - [20.0, 10.0]) - body[:2] / 2, 0)))

        self.assertLess(gap(start), limit)
        route = network.plan(start, goal, free)
        self.assertIsNotNone(route)
        a, b = route["points"][0], route["points"][1]
        samples = [gap(a + u * (b - a)) for u in np.linspace(0, 1, 50)]
        self.assertGreaterEqual(min(samples), gap(start) - 1e-6)
        for p0, p1 in zip(route["points"][1:-1], route["points"][2:], strict=False):
            self.assertTrue(network.clear_segment(p0, p1, free))
        # A connector straight at B is still refused.
        self.assertFalse(network.clear_segment(start, [19.2, 10.0], free))

    def test_passing_cells_and_turning_cells(self):
        # A 2.75 m wide walkway: two rows of cells where the body passes (half
        # width), one row where it can also rotate in place; turning cells are
        # a subset of the walking cells.
        from awr.life.civil.walking import restrict

        surface = SurfaceMap.__new__(SurfaceMap)
        surface.x0 = surface.y0 = 0
        surface.nx = 320
        surface.ny = 240
        surface.cell_m = 0.25
        surface.width_m = 80
        surface.height_m = 60
        surface.z = np.zeros((240, 320))
        walk = np.zeros((240, 320), bool)
        walk[112:123, :] = True
        surface.walk = {"person": walk}
        sem = {"roads": [{"line": [[-10, 29.5], [90, 29.5]], "width_m": 6}], "crossings": []}
        network = WalkNetwork.from_scene(surface, sem, 0.37, shared_streets={"carriageway_cost": 2.0}, turn_radius=0.72)
        rows_free = np.flatnonzero(network.free.any(1))
        rows_turn = np.flatnonzero(network.turn_free.any(1))
        self.assertTrue(np.all(network.free[network.turn_free]))
        self.assertEqual(len(rows_free), 2)
        self.assertEqual(len(rows_turn), 1)
        route = network.plan([10.25, 29.25], [70.25, 29.25])
        self.assertIsNotNone(route)
        narrow = restrict(network.free, network.turn_free)
        self.assertTrue(narrow.turn_only)
        self.assertLessEqual(int(narrow.sum()), int(network.free.sum()))

    def test_rotation_check_uses_the_body_box(self):
        # 2.75 m walkway (y 28-30.75): a 1.27 x 0.73 m body turns a right angle
        # on the centre line, not 0.5 m off it; a parked body nearby forbids it.
        surface = SurfaceMap.__new__(SurfaceMap)
        surface.x0 = surface.y0 = 0
        surface.nx = 320
        surface.ny = 240
        surface.cell_m = 0.25
        surface.width_m = 80
        surface.height_m = 60
        surface.z = np.zeros((240, 320))
        walk = np.zeros((240, 320), bool)
        walk[112:123, :] = True
        surface.walk = {"person": walk}
        sem = {"roads": [{"line": [[-10, 29.5], [90, 29.5]], "width_m": 6}], "crossings": []}
        network = WalkNetwork.from_scene(surface, sem, 0.37, shared_streets={"carriageway_cost": 2.0}, turn_radius=0.72)
        network.turn_body = (1.27, 0.73)
        self.assertTrue(network.can_turn([40.0, 29.375], 0.0, np.pi / 2))
        self.assertFalse(network.can_turn([40.0, 28.75], 0.0, np.pi / 2))
        network.turn_clear = np.asarray(network.obstacle_mask([[41.0, 29.4]], [[4.0, 1.8, 1.5]], 0.72, [0.0]))
        self.assertFalse(network.can_turn([40.0, 29.375], 0.0, np.pi / 2))

    def test_true_arc_length_and_corner_heading(self):
        route = polyline([[0, 0], [3, 0], [3, 4]], "test")
        self.assertEqual(route_length(route), 7)
        np.testing.assert_allclose(at(route, 2)[0], [2, 0])
        np.testing.assert_allclose(at(route, 3)[0], [3, 0])
        self.assertAlmostEqual(at(route, 3)[1], np.pi / 2)
        np.testing.assert_allclose(at(route, 6)[0], [3, 3])

    def test_line_cannot_clip_a_forbidden_cell_between_samples(self):
        free = np.ones((20, 20), bool)
        free[10, 10] = False
        network = WalkNetwork(free, [0.0, 0.0], step=1.0)
        # A microscopic overlap near (9.5, 9.5) is still an illegal route.
        self.assertFalse(network.clear_segment([8.0, 11.000001], [11.000001, 8.0], free))

    def test_long_vehicle_does_not_block_parallel_sidewalk_with_its_radius(self):
        network = WalkNetwork(np.ones((60, 80), bool), [0.0, 0.0])
        free = network.obstacle_mask([[20.0, 10.0]], [[12.0, 2.5, 3.0]], 0.7, [0.0])
        self.assertIsNotNone(network.plan([10.0, 13.0], [30.0, 13.0], free))
        self.assertIsNone(network.plan([20.0, 10.0], [30.0, 13.0], free))


if __name__ == "__main__":
    unittest.main()


def test_path_cost_keeps_to_cheap_cells_and_field_matches_grid_search():
    import numpy as np

    from awr.life.civil.walking import WalkNetwork

    free = np.ones((40, 120), bool)
    weight = np.ones(free.shape, np.float32)
    weight[10:30, :] = 3.0  # a costly band (carriageway) between two cheap edges
    net = WalkNetwork(free, (0.0, 0.0), 0.5, weight=weight)
    route = net.plan((1.0, 2.0), (58.0, 2.0), free)
    assert route is not None
    # The route stays on the cheap rows (y < 5 m) instead of crossing the band.
    assert np.all(route["points"][:, 1] < 5.0)
    # The distance field equals the A* cost to the same goal.
    field = net.distance_field(tuple(net.cell((58.0, 2.0))))
    region, index, cost = field
    c = tuple(net.cell((1.0, 2.0)))
    assert abs(float(cost[index[c]]) - (net.cell((58.0, 2.0))[1] - c[1])) < 1e-4
    # Long legs use the field heuristic and still find the crossing route.
    across = net.plan((1.0, 1.0), (58.0, 19.0), free)
    assert across is not None and across["arc"][-1] < 75.0


def test_unit_cost_network_unchanged_without_weights():
    import numpy as np

    from awr.life.civil.walking import WalkNetwork

    free = np.ones((20, 20), bool)
    free[5:15, 10] = False
    net = WalkNetwork(free, (0.0, 0.0), 0.5)
    route = net.plan((1.0, 5.0), (9.0, 5.0), free)
    assert route is not None and net.weight is None


def test_shared_street_routes_round_their_corners():
    import numpy as np

    from awr.life.civil.walking import WalkNetwork

    free = np.zeros((60, 60), bool)
    free[5:15, 5:55] = True
    free[5:55, 45:55] = True  # an L-shaped street
    net = WalkNetwork(free, (0.0, 0.0), 0.5, weight=np.ones(free.shape, np.float32))
    route = net.plan((4.0, 5.0), (25.0, 25.0), free)
    assert route is not None
    p = route["points"]
    d = np.diff(p, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turns = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    assert turns.max() <= 0.25 + 1e-6
    # every chord stays on free cells
    assert all(net.clear_segment(p[k], p[k + 1], free) for k in range(len(p) - 1))


def test_verified_shared_parking_allows_real_walk_access_but_keeps_body_and_ground_checks():
    import copy

    import pytest

    surface = SurfaceMap.__new__(SurfaceMap)
    surface.x0 = surface.y0 = 0.0
    surface.nx = 80
    surface.ny = 60
    surface.cell_m = 1.0
    surface.width_m = 80.0
    surface.height_m = 60.0
    surface.z = np.zeros((60, 80))
    surface.walk = {"person": np.ones((60, 80), bool)}
    lot = dict(
        id="surveyed-lot",
        verified=True,
        evidence="synthetic geometry fixture",
        polygon=[[10, 12], [35, 12], [35, 28], [10, 28]],
        aisles=[[[18, 14], [22, 14], [22, 27], [18, 27]]],
        entrances=[[[18, 25], [22, 25], [22, 28], [18, 28]]],
    )
    sem = dict(roads=[dict(line=[[-10, 30], [90, 30]], width_m=6)], crossings=[], verified_parking_regions=[lot])
    without = WalkNetwork.from_scene(surface, sem, 0.37)
    assert without.plan([40.0, 25.0], [28.0, 18.0]) is None
    network = WalkNetwork.from_scene(surface, sem, 0.37, shared_streets={"carriageway_cost": 2.0})
    assert network.parking_walk_regions == ("surveyed-lot",)
    route = network.plan([40.0, 30.0], [28.0, 18.0])
    assert route is not None
    assert network.carriageway[tuple(network.cell([20.0, 20.0]))]
    assert not network.carriageway[tuple(network.cell([28.0, 18.0]))]
    parked = network.obstacle_mask([[28.0, 18.0]], [[4.2, 2.1, 1.5]], 0.37, [0.0])
    assert network.plan([40.0, 30.0], [28.0, 18.0], parked) is None
    surface.walk["person"][15:22, 25:32] = False
    broken = WalkNetwork.from_scene(surface, sem, 0.37, shared_streets={"carriageway_cost": 2.0})
    assert broken.plan([40.0, 30.0], [28.0, 18.0]) is None
    unreviewed = copy.deepcopy(sem)
    unreviewed["verified_parking_regions"][0]["verified"] = False
    with pytest.raises(ValueError, match="surveyed"):
        WalkNetwork.from_scene(surface, unreviewed, 0.37, shared_streets={"carriageway_cost": 2.0})
