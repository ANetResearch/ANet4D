"""Compiled walking-route parts used by the life runtime: segment check equivalence and exact A* costs."""

import math

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

from awr.life.civil.grid_search import compiled_segment_check
from awr.life.civil.static_walk import StaticWalkRouter
from awr.life.civil.walking import WalkNetwork
from awr.life.motion.route import LifeRouter


def town(seed, shape=(90, 120)):
    rng = np.random.default_rng(seed)
    free = np.ones(shape, bool)
    for _ in range(40):
        y, x = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        free[y : y + rng.integers(2, 14), x : x + rng.integers(2, 14)] = False
    weight = np.where(rng.random(shape) < 0.15, 3.0, 1.0).astype(np.float32)
    return WalkNetwork(free, np.array([-3.1, 7.2]), step=0.5, weight=weight)


def test_compiled_segment_check_matches_numpy():
    check = compiled_segment_check()
    assert check is not None
    for seed in range(3):
        net = town(seed)
        rng = np.random.default_rng(100 + seed)
        lo, hi = net.origin, net.origin + np.array(net.free.shape[::-1]) * net.step
        for _ in range(3000):
            a = rng.uniform(lo - 1, hi + 1)
            b = a + rng.normal(0, 6, 2) if rng.random() < 0.8 else a + np.array([rng.normal(0, 6), 0.0])
            if rng.random() < 0.1:  # exactly on cell boundaries / centres
                a = net.origin + np.round((a - net.origin) / net.step * 2) / 2 * net.step
            mw = None if rng.random() < 0.5 else float(rng.choice([1.0, 2.0, 3.0]))
            q = (a - net.origin) / net.step
            end = (b - net.origin) / net.step
            expected = net._grid_segment_clear(q, end, end - q, net.free, net.free, mw)
            weight = net.weight if mw is not None else np.empty((0, 0), np.float32)
            got = check(
                float(q[0]), float(q[1]), float(end[0]), float(end[1]), net.free, weight, math.inf if mw is None else mw
            )
            assert got == expected, (seed, a, b, mw)


def test_life_router_costs_are_shortest():
    for seed in range(3):
        net = town(seed)
        static = StaticWalkRouter(net, net.free)
        life = LifeRouter(net, net.free, landmarks=4)
        rng = np.random.default_rng(seed)
        cells = np.argwhere(net.free)
        for _ in range(60):
            s, t = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
            si, ti = int(life.index[tuple(s)]), int(life.index[tuple(t)])
            path = life.cells(si, ti)
            exact = dijkstra(static.graph, directed=False, indices=int(static.index[tuple(s)]))[static.index[tuple(t)]]
            if not np.isfinite(exact):
                assert path is None
                continue
            assert path[0] == tuple(t) and path[-1] == tuple(s)
            assert all(
                abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 and net.free[p] for p, q in zip(path, path[1:], strict=False)
            )
            c = np.array([life.cost[life.index[p]] for p in path])
            assert abs(float(((c[1:] + c[:-1]) / 2).sum()) - exact) < 1e-6


def test_a_route_ends_beside_a_body_whose_clearance_closes_the_goal_cell():
    """A goal clear of a parked body whose clearance closes the goal's cell at grid resolution (the door side of a
    parked car) is reached by a clear last segment from the nearest free cell, by every router; a goal inside the
    clearance is not."""
    from awr.life.civil.walking import PLANNING_MARGIN_M

    net = WalkNetwork(np.ones((60, 60), bool), np.array([0.0, 0.0]), step=0.5)
    radius, size = 0.3, np.array([4.2, 2.1, 1.6])
    limit = radius + 0.08 + PLANNING_MARGIN_M
    goal, start = np.array([15.4, 18.2]), np.array([5.0, 25.0])  # the goal's cell centre (15.5, 18.0) is .2 m nearer
    for gap, reached in ((0.05, True), (-0.05, False)):
        car = np.array([15.0, goal[1] - size[1] / 2 - limit - gap])
        free = net.obstacle_mask(car[None], size[None], radius, np.zeros(1))
        assert not free[tuple(net.cell(goal))]
        for route in (
            LifeRouter(net, free).plan(start, goal),
            StaticWalkRouter(net, free).plan(start, goal),
            net.plan(start, goal, free),
        ):
            assert (route is not None) == reached
            if reached:
                points = np.asarray(route["points"])
                assert np.allclose(points[-1], goal)
                assert all(net.clear_segment(a, b, free) for a, b in zip(points[:-1], points[1:], strict=False))


def test_a_walker_on_a_snapshot_island_joins_the_network_by_a_supported_connector():
    """A walker whose local avoidance left it on a free cell that the 4-connected graph joins to no other (its cells
    close round it diagonally) is planned from there by a connector on supported ground, as a walker off the snapshot
    is; without supported ground there is no route (cq3p-d12-s3 v3, 1233.7 s: two walkers stranded 60 s)."""
    from types import SimpleNamespace

    free = np.zeros((12, 24), bool)
    free[2:6, :] = True  # a street
    free[6, 13] = True  # a recess of it
    free[7, 12] = True  # the island, diagonal to the recess
    net = WalkNetwork(free, np.array([0.0, 0.0]), step=0.5)
    start, goal = net.origin + np.array([12.0, 7.0]) * net.step, np.array([0.5, 1.5])
    for ground, reached in ((True, True), (False, False)):
        net.surface = SimpleNamespace(supported=lambda points, kind, ground=ground: np.full(len(points), ground))
        router = LifeRouter(net, net.free, landmarks=2)
        assert router.on_snapshot(start)
        route = router.plan(start, goal)
        assert (route is not None) == reached
        if reached:
            points = np.asarray(route["points"])
            assert np.allclose(points[0], start) and np.allclose(points[-1], goal)
            assert np.linalg.norm(points[1] - start) <= 1.5
            assert all(net.clear_segment(a, b, net.free) for a, b in zip(points[1:-1], points[2:], strict=False))
        probe = router.probe(start, goal)
        assert probe["on_snapshot"] and probe["start_component"] != probe["target_component"]
        assert any(
            n["component"] == probe["target_component"] and not n["segment_clear"] and n["connector_clear"] == ground
            for n in probe["nodes"]
        )


# Adapted from codex 232db5e.
def test_landmark_reuse_retains_shortest_costs_on_changed_obstacles(monkeypatch):
    # Fresh obstacles split components; old potentials must not grant a path.
    # Changing parking repeatedly closes AND reopens cells relative to the
    # previous snapshot, while all remain subsets of the immutable parent.
    net = town(42)
    base = LifeRouter(net, net.free, landmarks=4)

    def unexpected(*args, **kwargs):
        raise AssertionError("Reused landmarks ran global Dijkstra again")

    monkeypatch.setattr("awr.life.motion.route.dijkstra", unexpected)
    rng = np.random.default_rng(88)
    for column in (35, 65, 35):
        free = net.free.copy()
        free[:, column] = False
        child = LifeRouter(net, free, landmark_source=base)
        reference = StaticWalkRouter(net, free)
        for _ in range(40):
            a, b = rng.integers(len(child.yx), size=2)
            start, end = tuple(child.yx[a]), tuple(child.yx[b])
            expected = dijkstra(reference.graph, directed=False, indices=int(reference.index[start]))[
                reference.index[end]
            ]
            path = child.cells(int(a), int(b))
            if not np.isfinite(expected):
                assert path is None
            else:
                assert path[0] == end and path[-1] == start
                assert all(free[p] for p in path)
                assert all(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for p, q in zip(path, path[1:], strict=False))
                costs = np.array([child.cost[child.index[p]] for p in path])
                assert abs(((costs[:-1] + costs[1:]) * 0.5).sum() - expected) < 1e-6
        assert not np.shares_memory(child.lm, base.lm)
        assert child.stats["life_router_landmark_reuses"] == 1


def test_landmark_parent_cannot_hide_new_cells_cheaper_costs_or_other_network():
    net = town(3)
    free = net.free.copy()
    free[:, 30] = False
    base = LifeRouter(net, free, landmarks=2)
    with pytest.raises(ValueError, match="subset"):
        LifeRouter(net, net.free, landmark_source=base)
    with pytest.raises(ValueError, match="same walking network"):
        LifeRouter(town(3), free, landmark_source=base)
    # Edge costs decreasing would invalidate the old metric's lower bound.
    net.weight[net.free] = 4.0
    base = LifeRouter(net, net.free, landmarks=2)
    net.weight[net.free] = 1.0
    with pytest.raises(ValueError, match="reduced cell costs"):
        LifeRouter(net, net.free, landmark_source=base)
