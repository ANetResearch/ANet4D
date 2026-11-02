import math

import numpy as np
from test_activity_engine import route, world

from awr.life.civil.activity.grounded import GroundedActivities
from awr.life.civil.walking import WalkNetwork
from awr.life.motion.parked_space import ParkedSpace


def space(corridor=False):
    cells = np.ones((60, 60), bool)
    if corridor:
        cells[:] = False
        cells[2:9] = True
    network = WalkNetwork(cells, [0.0, 0.0], step=0.5)
    free = network.obstacle_mask([], [], 0.3, [])
    activity = GroundedActivities(network, dict(sites=[], street=[]), free, {}, 17)
    return ParkedSpace(activity, network, 0.3)


def test_future_car_is_not_admitted_on_the_wrong_side_of_its_own_blockage():
    s = space(corridor=True)
    old = (6.0, 2.5, math.pi / 2, 4.0, 2.0, 1.5)
    future = (15.0, 2.5, math.pi / 2, 4.0, 2.0, 1.5)
    s.update({1: old})
    before = np.array(s.activity.static_free, copy=True)
    start, end = (13.4, 2.5), (24.0, 2.5)
    assert s.activity.measured_route(start, end) is not None
    assert s.plan_after_parking(1, future, start, end) is None
    assert s.parked == {1: old}
    np.testing.assert_array_equal(s.activity.static_free, before)
    assert s.plan_after_parking(1, future, start, end) is None
    assert s.future_stats["cache_hits"] == 1


def test_future_route_can_use_the_space_freed_at_the_old_berth():
    s = space(corridor=True)
    s.update({1: (15.0, 2.5, math.pi / 2, 4.0, 2.0, 1.5)})
    start, end = (10.0, 2.5), (24.0, 2.5)
    assert s.activity.measured_route(start, end) is None
    assert s.plan_after_parking(1, (5.0, 2.5, math.pi / 2, 4.0, 2.0, 1.5), start, end) is not None


# Adapted from codex 232db5e.
def test_parking_changes_reuse_only_landmarks_not_paths_or_connectivity(monkeypatch):
    s = space(corridor=True)
    start, end = (10.0, 2.5), (24.0, 2.5)
    s.update({})
    base = s.landmark_base
    assert s.activity.measured_route(start, end) is not None

    def unexpected(*args, **kwargs):
        raise AssertionError("Parking update recomputed all landmark distances")

    monkeypatch.setattr("awr.life.motion.route.dijkstra", unexpected)
    s.update({1: (15.0, 2.5, math.pi / 2, 4.0, 2.0, 1.5)})
    assert s.activity.measured_route(start, end) is None
    s.update({})
    assert s.activity.measured_route(start, end) is not None
    assert s.landmark_base is base
    assert s.activity.route_stats["life_router_landmark_reuses"] == 3


def test_unaffected_route_reuses_proof_and_parking_change_invalidates_cache():
    s = space()
    old = (5.0, 5.0, 0.0, 4.0, 2.0, 1.5)
    s.update({1: old})
    future = (12.0, 5.0, 0.0, 4.0, 2.0, 1.5)
    start, end = (10.0, 15.0), (20.0, 15.0)
    assert s.plan_after_parking(1, future, start, end) is not None
    assert s.future_stats["reused_routes"] == 1
    assert s.future_stats["router_builds"] == 0
    # Another car now blocks the endpoint; an earlier positive proof is stale.
    s.update({1: old, 2: (20.0, 15.0, 0.0, 4.0, 2.0, 1.5)})
    assert s.plan_after_parking(1, future, start, end) is None


def test_parking_that_changes_the_route_can_still_offer_a_real_detour():
    s = space()
    s.update({1: (5.0, 5.0, 0.0, 4.0, 2.0, 1.5)})
    start, end = (10.0, 15.0), (22.0, 15.0)
    future = (15.0, 15.0, math.pi / 2, 4.0, 2.0, 1.5)
    path = s.plan_after_parking(1, future, start, end)
    assert path is not None
    assert np.linalg.norm(np.diff(path["points"], axis=0), axis=1).sum() > 12.0
    mask = s.walking.obstacle_mask([future[:2]], [future[3:]], 0.3, [future[2]])
    assert all(
        s.walking.clear_segment(a, b, mask) for a, b in zip(path["points"][:-1], path["points"][1:], strict=False)
    )
    # Found on the present router round the future body's cells, then checked on the future geometry: no grid built.
    assert s.future_stats["round_routes"] == 1 and s.future_stats["router_builds"] == 0


def test_a_way_round_the_future_body_gives_the_verdict_of_the_future_grid(monkeypatch):
    # ParkedSpace._round may only accept a route the future grid admits, and its refusal leaves the verdict to that
    # grid: with and without it, the same queries have a route or none.
    rng = np.random.default_rng(5)

    def verdicts(round_way):
        s = space()
        if not round_way:
            monkeypatch.setattr(s, "_round", lambda *a: None)
        s.update({1: (5.0, 5.0, 0.0, 4.0, 2.0, 1.5), 2: (20.0, 20.0, 0.3, 4.0, 2.0, 1.5)})
        out = []
        for _ in range(30):
            body = (
                float(rng.uniform(4.0, 26.0)),
                float(rng.uniform(4.0, 26.0)),
                float(rng.uniform(0.0, math.pi)),
                4.0,
                2.0,
                1.5,
            )
            start, end = tuple(rng.uniform(1.0, 29.0, 2)), tuple(rng.uniform(1.0, 29.0, 2))
            path = s.plan_after_parking(1, body, start, end)
            if path is not None:
                light = s._exact(1, body)
                assert all(
                    s.walking.clear_segment(a, b, light)
                    for a, b in zip(path["points"][:-1], path["points"][1:], strict=False)
                )
            out.append(path is not None)
        return out, s.future_stats

    rng = np.random.default_rng(5)
    fast, stats = verdicts(True)
    rng = np.random.default_rng(5)
    exact, _ = verdicts(False)
    assert fast == exact and stats["round_routes"] > 0 and True in fast


def test_director_rejects_unreachable_future_walk_without_reserving_a_berth():
    d = world(car=True)
    seen = []

    def proof(*args):
        seen.append(args)
        return None

    d.parking_walk = proof
    holders = dict(d.holders)
    a, p = d.actors["p"], d.places["shop"]
    assert d._journey(a, (0.0, 0.0), p, "car", 0.0, 10.0) is None
    assert seen == [("car", "shop-bay", (90.0, 2.0), (100.0, 0.0), "p")]
    assert d.holders == holders
    assert d.car_rejections["no_walk_after_parking"] == 1
    d.parking_walk = lambda v, b, s, g, p: route("walk", s, g, p)
    assert d._journey(a, (0.0, 0.0), p, "car", 0.0, 10.0) is not None


def test_exact_future_geometry_checks_segments_as_the_built_future_grid():
    # plan_after_parking checks the present route against _exact instead of building the future grid for every
    # candidate berth (cruise r11f v9); every clear_segment answer must be the same.
    from awr.life.civil.walking import merge_obstacles
    from awr.life.motion.parked_space import _copy

    s = space()
    parked = {
        1: (5.0, 5.0, 0.0, 4.0, 2.0, 1.5),
        2: (20.0, 12.0, 0.7, 4.0, 2.0, 1.5),
        3: (12.0, 22.0, 1.9, 4.4, 2.1, 1.5),
    }
    s.update(parked)
    rng = np.random.default_rng(3)
    seen = set()
    for trial in range(12):
        car = int(rng.integers(1, 4))
        body = (
            float(rng.uniform(4.0, 26.0)),
            float(rng.uniform(4.0, 26.0)),
            float(rng.uniform(-3.0, 3.0)),
            *parked[car][3:],
        )
        future = _copy(s.base)
        for i in sorted(s.masks):
            if i != car:
                merge_obstacles(future, s.masks[i][1])
        merge_obstacles(
            future, s.walking.obstacle_mask(np.array([body[:2]]), np.array([body[3:6]]), s.radius, np.array([body[2]]))
        )
        light = s._exact(car, body)
        for _ in range(25):
            a, b = rng.uniform(1.0, 29.0, 2), rng.uniform(1.0, 29.0, 2)
            clear = s.walking.clear_segment(a, b, future)
            assert s.walking.clear_segment(a, b, light) == clear
            assert s.walking.clear_segment(a, a, light) == s.walking.clear_segment(a, a, future)
            seen.add(clear)
    assert seen == {True, False}
