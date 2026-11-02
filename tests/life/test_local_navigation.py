import unittest

import numpy as np
from scipy.spatial import cKDTree

from awr.life.civil.local_navigation import distance_to_polyline, rolling_route, step_aside_route, yield_route
from awr.life.civil.prediction import blocked_disks, route_at_many
from awr.life.civil.walking import PLANNING_MARGIN_M, WalkNetwork, polyline


class LocalNavigationTests(unittest.TestCase):
    def test_far_occupied_destination_does_not_stop_local_progress(self):
        network = WalkNetwork(np.ones((30, 100), bool), [0.0, 0.0], 0.5)
        start, goal = [2.0, 7.0], [40.0, 7.0]
        free = network.obstacle_mask([goal], [[1.0, 0.6, 1.8]], 0.65, circle_radii=[0.65])
        self.assertIsNone(network.plan(start, goal, free))
        route, temporary = rolling_route(network, start, goal, network.free, free)
        self.assertIsNotNone(route)
        self.assertTrue(temporary)
        self.assertGreater(route["points"][-1][0], 10.0)
        self.assertLess(route["points"][-1][0], goal[0])
        for a, b in zip(route["points"][:-1], route["points"][1:], strict=False):
            self.assertTrue(network.clear_segment(a, b, free))

    def test_occupied_guide_points_lead_to_the_open_space_beside_them(self):
        # People standing on every sampled guide point (2, 5 and 6 m) of a short
        # leg: the walker heads for the open space beside its path instead of
        # reporting the destination unreachable. The destination is retained.
        network = WalkNetwork(np.ones((30, 100), bool), [0.0, 0.0], 0.5)
        start, goal = [2.0, 7.0], [8.0, 7.0]
        bodies = [[4.0, 7.0], [7.0, 7.0], [8.0, 7.0]]
        free = network.obstacle_mask(bodies, [[0.42, 0.68, 1.7]] * 3, 0.36, yaws=[0.0, 0.0, 0.0])
        self.assertFalse(any(free[tuple(network.cell(q))] for q in bodies))
        route, temporary = rolling_route(network, start, goal, network.free, free)
        self.assertIsNotNone(route)
        self.assertTrue(temporary)
        end = route["points"][-1]
        self.assertLessEqual(min(np.linalg.norm(end - np.array(q)) for q in ([4.0, 7.0], [7.0, 7.0], [8.0, 7.0])), 1.6)
        for a, b in zip(route["points"][:-1], route["points"][1:], strict=False):
            self.assertTrue(network.clear_segment(a, b, free))

    def test_exact_safe_pose_can_leave_a_dynamically_blocked_raster_cell(self):
        network = WalkNetwork(np.ones((40, 60), bool), [0.0, 0.0], 0.5)
        start = np.array([10.24, 10.0])
        free = network.obstacle_mask([[8.85, 10.0]], [[1.0, 0.6, 1.8]], 0.6, circle_radii=[0.6])
        self.assertFalse(free[tuple(network.cell(start))])
        self.assertTrue(network.clear_segment(start, start, free))
        route = network.plan(start, [20.0, 10.0], free)
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route["points"][0], start)
        # An actually overlapping body is still rejected, with no snapping.
        self.assertIsNone(network.plan([9.9, 10.0], [20.0, 10.0], free))

    def test_static_wall_is_never_treated_as_temporary_occupancy(self):
        mask = np.ones((30, 100), bool)
        mask[:, 40] = False
        network = WalkNetwork(mask, [0.0, 0.0], 0.5)
        free = network.obstacle_mask([], [], 0.65)
        route, temporary = rolling_route(network, [2.0, 7.0], [40.0, 7.0], mask, free)
        self.assertIsNone(route)
        self.assertFalse(temporary)

    def test_lower_priority_person_yields_without_moving_the_other(self):
        network = WalkNetwork(np.ones((40, 60), bool), [0.0, 0.0], 0.5)
        positions = np.array([[10.0, 10.0], [12.0, 10.0]])
        radii = np.array([0.65, 0.65])
        free = network.obstacle_mask(positions[:1], [[1.0, 0.6, 1.8]], 0.65, circle_radii=radii[:1])
        route = yield_route(network, positions[1], np.pi, free, 1, positions, radii, np.ones(2, bool))
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route["points"][0], positions[1])
        self.assertTrue(network.clear_segment(*route["points"], free))
        self.assertIsNone(yield_route(network, positions[0], 0.0, free, 0, positions, radii, np.ones(2, bool)))

    def test_planned_connectors_are_never_refused_by_the_lookahead(self):
        # A standing body at exactly the lookahead threshold must be rejected by
        # the planner; whatever the planner accepts must pass the lookahead at
        # every .025 m sample. This is the mechanism behind recorded stalls of
        # pedestrians that held a complete route and never moved.
        network = WalkNetwork(np.ones((40, 60), bool), [0.0, 0.0], 0.5)
        body, radius = np.array([[10.0, 10.0]]), 0.65
        sizes = np.array([[1.1, 0.7, 1.7]])
        yaws = np.zeros(1)
        free = network.obstacle_mask(body, sizes, radius, yaws, circle_radii=[0.6])
        grazing = [8.0, 10.0 + 0.6 + radius + 0.08]
        self.assertFalse(network.clear_segment(grazing, [12.0, grazing[1]], free))
        self.assertTrue(PLANNING_MARGIN_M > 0.05)
        tree = cKDTree(body)
        obstacle_sizes = np.array([[1.2, 1.2, 1.7]])
        obstacle_radii = np.linalg.norm(obstacle_sizes[:, :2], axis=1) / 2
        for offset in np.arange(0.60, 1.00, 0.005):
            a, b = [8.0, 10.0 + offset], [12.0, 10.0 + offset]
            if not network.clear_segment(a, b, free):
                continue
            route = polyline([a, b], "t")
            distances = np.arange(0.025, route["arc"][-1], 0.025)
            points, _ = route_at_many(route, distances)
            blocked = blocked_disks(
                1, points, radius, body, yaws, obstacle_sizes, tree, obstacle_radii, np.array([0.6])
            )
            self.assertFalse(blocked.any(), offset)

    def test_step_aside_refuge_clears_the_corridor_or_is_absent(self):
        network = WalkNetwork(np.ones((40, 60), bool), [0.0, 0.0], 0.5)
        corridor = np.array([[0.0, 10.0], [20.0, 10.0]])
        free = network.obstacle_mask([], [], 0.65)
        route = step_aside_route(network, [10.0, 10.2], corridor, free, 1.6)
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route["points"][0], [10.0, 10.2])
        self.assertGreaterEqual(distance_to_polyline(corridor, route["points"][-1]), 1.6)
        self.assertTrue(network.clear_segment(*route["points"], free))
        # Single-file passage: no refuge exists and none is invented.
        mask = np.zeros((40, 60), bool)
        mask[19:22, :] = True
        narrow = WalkNetwork(mask, [0.0, 0.0], 0.5)
        self.assertIsNone(step_aside_route(narrow, [10.0, 10.0], corridor, narrow.obstacle_mask([], [], 0.65), 1.6))


if __name__ == "__main__":
    unittest.main()


def test_retreat_route_reaches_a_passing_place_off_the_other_path():
    import numpy as np

    from awr.life.civil.local_navigation import retreat_route
    from awr.life.civil.walking import WalkNetwork

    free = np.zeros((40, 80), bool)
    free[18:22, :] = True  # a 2 m single-file passage
    free[10:30, 5:15] = True  # a wider place behind the walker
    net = WalkNetwork(free, (0.0, 0.0), 0.5)
    start = (20.0, 10.0)
    other_path = [(40.0, 10.0), (2.0, 10.0)]
    route = retreat_route(net, start, other_path, free, 2.0)
    assert route is not None
    end = route["points"][-1]
    assert abs(end[1] - 10.0) >= 2.0 and end[0] <= 8.0
