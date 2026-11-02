import numpy as np

from awr.life.civil.static_walk import StaticWalkRouter
from awr.life.civil.walking import WalkNetwork


def check_route(network, router, route, start, goal):
    assert route is not None
    np.testing.assert_array_equal(route["points"][0], start)
    np.testing.assert_array_equal(route["points"][-1], goal)
    for a, b in zip(route["points"][:-1], route["points"][1:], strict=False):
        assert network.clear_segment(a, b, router.free)


def test_one_source_tree_serves_many_distinct_facility_queries():
    mask = np.ones((70, 100), bool)
    mask[25:45, 40:60] = False
    network = WalkNetwork(mask, [0.0, 0.0])
    router = StaticWalkRouter(network, network.free)
    start = np.array([2.13, 16.71])
    for y in range(4, 30, 3):
        goal = np.array([42.0, float(y)])
        check_route(network, router, router.plan(start, goal), start, goal)
    assert router.stats["static_source_searches"] == 1
    assert router.stats["static_source_cache_hits"] == 8


def test_return_proofs_reuse_symmetric_tree_but_recheck_real_body_connectors():
    network = WalkNetwork(np.ones((60, 100), bool), [0.0, 0.0])
    free = network.obstacle_mask([[24.0, 15.0]], [[5.0, 2.0, 2.0]], 0.4, [np.pi / 3])
    router = StaticWalkRouter(network, free)
    home = [3.0, 15.0]
    for goal in ([45.0, 15.0], [45.0, 20.0], [40.0, 10.0]):
        check_route(network, router, router.plan(home, goal), home, goal)
        check_route(network, router, router.plan(goal, home), goal, home)
    assert router.stats["static_source_searches"] == 1
    assert router.stats["static_reverse_tree_hits"] == 3
    # A cached destination tree does not authorize an origin inside a body.
    assert router.plan([24.0, 15.0], home) is None


def test_reverse_tree_respects_weighted_road_cost_and_cache_eviction():
    free = np.ones((50, 80), bool)
    weight = np.ones_like(free, dtype=np.float32)
    weight[20:31, 15:65] = 15.0
    network = WalkNetwork(free, [0.0, 0.0], weight=weight)
    router = StaticWalkRouter(network, free, cache_size=1)
    home, goal = [3.0, 12.5], [36.0, 12.5]
    check_route(network, router, router.plan(home, goal), home, goal)
    route = router.plan(goal, home)
    check_route(network, router, route, goal, home)
    assert np.min(route["points"][:, 1]) < 10.0 or np.max(route["points"][:, 1]) > 15.0
    assert router.stats["static_source_searches"] == 1
    router.plan([4.0, 3.0], [35.0, 3.0])  # evict home tree
    check_route(network, router, router.plan(goal, home), goal, home)
    assert router.stats["static_source_searches"] == 3


def test_dynamic_mask_snapshot_does_not_cross_walls_or_change_after_mutation():
    network = WalkNetwork(np.ones((40, 80), bool), [0.0, 0.0])
    mask = network.free.copy()
    mask[:, 40] = False
    router = StaticWalkRouter(network, mask)
    assert router.plan([5.0, 10.0], [30.0, 10.0]) is None
    mask[:] = True  # caller must invalidate/rebuild; the old snapshot stays safe
    assert router.plan([5.0, 10.0], [30.0, 10.0]) is None
    rebuilt = StaticWalkRouter(network, mask)
    check_route(network, rebuilt, rebuilt.plan([5.0, 10.0], [30.0, 10.0]), [5.0, 10.0], [30.0, 10.0])


def test_oriented_obstacle_metadata_and_exact_pose_are_preserved():
    network = WalkNetwork(np.ones((60, 100), bool), [0.0, 0.0])
    free = network.obstacle_mask([[24.0, 15.0]], [[5.0, 2.0, 2.0]], 0.4, [np.pi / 3])
    router = StaticWalkRouter(network, free)
    start, goal = [3.13, 14.81], [45.0, 15.0]
    check_route(network, router, router.plan(start, goal), start, goal)
    free.positions[:] = 0.0
    np.testing.assert_array_equal(router.free.positions, [[24.0, 15.0]])
    assert router.plan([24.0, 15.0], goal) is None  # no snapping out of a car


def test_costly_carriageway_remains_costly_and_no_diagonal_corner_cutting():
    free = np.ones((50, 80), bool)
    weight = np.ones_like(free, dtype=np.float32)
    weight[20:31, 15:65] = 15.0
    network = WalkNetwork(free, [0.0, 0.0], weight=weight)
    router = StaticWalkRouter(network, free)
    route = router.plan([3.0, 12.5], [36.0, 12.5])
    check_route(network, router, route, [3.0, 12.5], [36.0, 12.5])
    assert np.min(route["points"][:, 1]) < 10.0 or np.max(route["points"][:, 1]) > 15.0
    free[1, 2] = False
    free[2, 1] = False
    free[1, 1] = True
    network = WalkNetwork(free, [0.0, 0.0])
    # Isolate one cell diagonally from a neighbouring reachable region.
    free[0:3, 0:3] = False
    free[1, 1] = True
    router = StaticWalkRouter(network, free)
    assert router.plan([0.5, 0.5], [4.0, 4.0]) is None


def test_activity_router_rebuilds_when_a_parked_vehicle_closes_a_passage():
    from awr.life.civil.activity.grounded import GroundedActivities

    free = np.ones((40, 140), bool)
    network = WalkNetwork(free, [0.0, 0.0])
    pools = dict(
        sites=[
            dict(
                id="park",
                kind="park",
                affordances=("outdoor_stay",),
                semantic_source="fixture:park",
                slots=[dict(id="slot", xy=(60.0, 10.0))],
            )
        ]
    )
    runtime = GroundedActivities(network, pools, free, {}, 17)
    assert runtime.route("walk", (5.0, 10.0), (60.0, 10.0), "p") is not None
    assert runtime.route_stats["static_graph_builds"] == 1
    free[:, 60] = False
    runtime.invalidate_routes()
    assert runtime.static_router is None
    assert runtime.route("walk", (5.0, 10.0), (60.0, 10.0), "p") is None
    free[:, 60] = True
    runtime.invalidate_routes()
    assert runtime.route("walk", (5.0, 10.0), (60.0, 10.0), "p") is not None
    assert runtime.route_stats["static_graph_builds"] == 2


def test_newly_parked_body_remains_visible_to_exact_route_simplification():
    from awr.life.civil.walking import merge_obstacles

    network = WalkNetwork(np.ones((50, 100), bool), [0.0, 0.0])
    free = network.obstacle_mask([], [], 0.4)
    car = network.obstacle_mask([[24.0, 12.0]], [[5.0, 2.0, 1.5]], 0.4, [0.0])
    # Reproduce the previous failure: masking node centres alone does not
    # communicate a new body to the exact segment checker.
    free[...] &= car
    assert network.clear_segment([2.0, 12.0], [46.0, 12.0], free)
    merge_obstacles(free, car)
    assert not network.clear_segment([2.0, 12.0], [46.0, 12.0], free)
    router = StaticWalkRouter(network, free)
    route = router.plan([2.0, 12.0], [46.0, 12.0])
    check_route(network, router, route, [2.0, 12.0], [46.0, 12.0])
    assert route["arc"][-1] > 44.0


def test_legal_start_connector_can_leave_a_disconnected_nearest_node_pocket():
    network = WalkNetwork(np.ones((6, 10), bool), [0.0, 0.0], step=1.0)
    free = network.obstacle_mask([], [], 0.3)
    # Dynamic candidate cells are not solid geometry. There is an exact legal
    # connector to column 4, but the closer column 2 is in the wrong component.
    free[:, 3] = False
    start, goal = [2.6, 2.0], [8.0, 2.0]
    router = StaticWalkRouter(network, free)
    for route in (network.plan(start, goal, free), router.plan(start, goal)):
        check_route(network, router, route, start, goal)
    # This does not waive a real obstacle along that alternative connector.
    wall = network.obstacle_mask([[3.5, 2.5]], [[0.5, 8.0, 2.0]], 0.3)
    assert network.plan(start, goal, wall) is None
    assert StaticWalkRouter(network, wall).plan(start, goal) is None
