"""All walking planners retain exact ingress to a berth beside its parked car."""

import numpy as np
import pytest

from awr.life.civil.activity.grounded import GroundedActivities
from awr.life.civil.static_walk import StaticWalkRouter
from awr.life.civil.walking import WalkNetwork
from awr.life.motion.route import LifeRouter


@pytest.mark.parametrize("backend", ["grid", "static", "life", "activity"])
def test_car_side_access_remains_reachable_without_moving_or_shrinking_the_car(backend):
    n = WalkNetwork(np.ones((60, 60), bool), [0.0, 0.0])
    free = n.obstacle_mask([[10.0, 10.0]], [[4.0, 2.0, 2.0]], 0.4, [0.0])
    start, goal = np.array([5.0, 15.0]), np.array([10.0, 11.6])
    assert not free[tuple(n.cell(goal))]
    if backend == "grid":

        def plan(a, b):
            return n.plan(a, b, free)
    elif backend == "static":
        plan = StaticWalkRouter(n, free).plan
    elif backend == "life":
        plan = LifeRouter(n, free, landmarks=2).plan
    else:
        runtime = GroundedActivities(n, dict(sites=[], street=[]), free, {}, 17)
        runtime.static_router = LifeRouter(n, free, landmarks=2)
        plan = runtime.measured_route
    for a, b in [(start, goal), (goal, start)]:
        route = plan(a, b)
        assert route is not None
        np.testing.assert_array_equal(route["points"][0], a)
        np.testing.assert_array_equal(route["points"][-1], b)
        assert all(n.clear_segment(p, q, free) for p, q in zip(route["points"][:-1], route["points"][1:], strict=False))
    # Real collision and the original planning margin remain rejection conditions.
    assert plan(start, [10.0, 10.0]) is None
    assert plan(start, [10.0, 11.45]) is None
    np.testing.assert_array_equal(free.positions, [[10.0, 10.0]])
    np.testing.assert_array_equal(free.sizes, [[4.0, 2.0, 2.0]])


@pytest.mark.parametrize("backend", [StaticWalkRouter, LifeRouter])
def test_endpoint_fallback_does_not_escape_static_terrain_or_connect_components(backend):
    mask = np.ones((60, 60), bool)
    mask[:, 30] = False
    n = WalkNetwork(mask, [0.0, 0.0])
    free = n.obstacle_mask([[10.0, 10.0]], [[4.0, 2.0, 2.0]], 0.4, [0.0])
    router = backend(n, free)
    assert router.plan([20.0, 15.0], [10.0, 11.6]) is None
    assert router.plan([5.0, 15.0], [15.0, 10.0]) is None
