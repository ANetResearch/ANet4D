"""Route identity, not just reachability, across execution backends."""

import heapq
import math

import numpy as np
import pytest

from awr.life.civil.grid_search import grid_path, implementation
from awr.life.civil.walking import WalkNetwork


def reference(free, weight, source, target, field=None):
    # Independent retained pre-acceleration A*: global tuple keys and nested
    # heap tuples make tie/crop regressions visible to this comparison.
    queue = [(0.0, source)]
    costs = {source: 0.0}
    parent = {}
    closed = set()
    while queue:
        _, u = heapq.heappop(queue)
        if u == target:
            break
        if u in closed:
            continue
        closed.add(u)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            v = u[0] + dy, u[1] + dx
            if not (0 <= v[0] < free.shape[0] and 0 <= v[1] < free.shape[1]) or not free[v]:
                continue
            cost = costs[u] + (1 if weight is None else (float(weight[u]) + float(weight[v])) / 2)
            if cost < costs.get(v, math.inf):
                parent[v] = u
                costs[v] = cost
                h = abs(v[0] - target[0]) + abs(v[1] - target[1])
                if field is not None:
                    region, index, distance = field
                    h = float(distance[index[v[0] - region[0].start, v[1] - region[1].start]])
                heapq.heappush(queue, (cost + h, v))
    if target not in costs:
        return []
    cells = [target]
    while cells[-1] != source:
        cells.append(parent[cells[-1]])
    return cells


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_exact_old_grid_paths_with_costs_dynamic_masks_and_cropped_component(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    rng = np.random.default_rng(914)
    free = np.zeros((63, 107), bool)
    free[7:54, 11:99] = True
    weight = rng.choice([1.0, 1.1, 1.5, 3.0], free.shape).astype(np.float32)
    net = WalkNetwork(free, [0.0, 0.0], weight=weight, search_backend=backend)
    region = net.component_slices[0]
    for case in range(40):
        dynamic = free & (rng.random(free.shape) > (0.25 if case < 30 else 0.8))
        source, target = (12, 18), (49, 94)
        if case == 39:
            target = source
        dynamic[source] = dynamic[target] = True
        costs = weight if case % 3 else None
        field = net.distance_field(target) if case % 3 == 2 else None
        actual = grid_path(dynamic, costs, source, target, region=region, field=field, backend=backend)
        assert actual == reference(dynamic, costs, source, target, field)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_accelerated_route_still_checks_off_grid_body_and_turning_geometry(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    net = WalkNetwork(np.ones((80, 100), bool), [0.0, 0.0], search_backend=backend)
    free = net.obstacle_mask([[20.0, 20.0]], [[6.0, 2.0, 2.0]], 0.4, [np.pi / 4])
    start, goal = [2.13, 20.21], [44.2, 20.31]
    route = net.plan(start, goal, free)
    assert route is not None
    np.testing.assert_array_equal(route["points"][0], start)
    np.testing.assert_array_equal(route["points"][-1], goal)
    assert all(net.clear_segment(a, b, free) for a, b in zip(route["points"][:-1], route["points"][1:], strict=False))
    assert net.plan([20.0, 20.0], goal, free) is None
    # Turning clearance uses the same original physical finishing stage.
    net.turn_body = (1.0, 0.7)
    net.turn_free = np.ones(net.free.shape, bool)
    net.can_turn = lambda *args, **kw: False
    checked = net.plan(start, goal, free)
    assert checked is not None
    assert all(
        net.clear_segment(a, b, free) for a, b in zip(checked["points"][:-1], checked["points"][1:], strict=False)
    )


def test_bad_backend_and_mismatched_heuristic_fail_explicitly():
    with pytest.raises(ValueError):
        implementation("unknown")
    with pytest.raises(ValueError):
        grid_path(
            np.ones((5, 5), bool),
            None,
            (0, 0),
            (4, 4),
            region=(slice(0, 5), slice(0, 5)),
            field=((slice(1, 5), slice(0, 5)), np.zeros((4, 5), int), np.ones(20)),
        )
