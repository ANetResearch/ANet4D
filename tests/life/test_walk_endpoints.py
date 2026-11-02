from types import SimpleNamespace

import numpy as np

from awr.life.civil.static_walk import StaticWalkRouter
from awr.life.civil.walk_endpoints import endpoint_pair, standing_anchor_clear
from awr.life.civil.walking import WalkNetwork


def setup(mask=None):
    n = WalkNetwork(np.ones((60, 60), bool) if mask is None else mask, [0.0, 0.0])
    free = n.obstacle_mask([[10.0, 10.0]], [[4.0, 2.0, 2.0]], 0.4, [0.0])
    return n, StaticWalkRouter(n, free)


def test_valid_continuous_goal_in_blocked_node_cell_keeps_exact_body_connector():
    n, r = setup()
    start = [5.0, 15.0]
    goal = [10.0, 11.6]
    assert not r.free[tuple(n.cell(goal))]
    pair = endpoint_pair(n, r.free, r.index, r.components, start, goal)
    assert pair is not None
    source, target = pair
    target_xy = n.origin + np.array(target[::-1]) * n.step
    assert not np.array_equal(target_xy, goal)
    assert n.clear_segment(target_xy, goal, r.free)
    assert n.clear_segment(start, n.origin + np.array(source[::-1]) * n.step, r.free)


def test_actual_body_and_static_ground_constraints_cannot_be_bypassed_by_endpoint_search():
    n, r = setup()
    assert endpoint_pair(n, r.free, r.index, r.components, [5.0, 15.0], [10.0, 10.0]) is None
    # Actual body not overlapping but final approach enters planning margin.
    assert endpoint_pair(n, r.free, r.index, r.components, [5.0, 15.0], [10.0, 11.45]) is None
    mask = np.ones((60, 60), bool)
    mask[23, :] = False
    n, r = setup(mask)
    assert endpoint_pair(n, r.free, r.index, r.components, [5.0, 15.0], [10.0, 11.6]) is None


def test_original_free_goal_node_is_preferred_and_disconnected_components_stay_separate():
    n, r = setup()
    goal = [20.0, 20.0]
    pair = endpoint_pair(n, r.free, r.index, r.components, [5.0, 15.0], goal)
    assert pair[1] == tuple(n.cell(goal))
    mask = np.ones((60, 60), bool)
    mask[:, 30] = False
    n, r = setup(mask)
    assert endpoint_pair(n, r.free, r.index, r.components, [5.0, 15.0], [20.0, 20.0]) is None


def test_standing_anchor_retains_turning_radius_margin_and_measured_ground():
    n, r = setup()
    surface = SimpleNamespace(supported=lambda xy, kind: np.ones(len(xy), bool))
    assert standing_anchor_clear(n, surface, r.free, [10.0, 11.6], 0.4)
    assert not standing_anchor_clear(n, surface, r.free, [10.0, 11.45], 0.4)
    assert not standing_anchor_clear(n, surface, r.free, [10.0, 10.0], 0.4)
    assert not standing_anchor_clear(n, surface, r.free, [-1.0, 0.0], 0.4)
    unsupported = SimpleNamespace(supported=lambda xy, kind: np.zeros(len(xy), bool))
    assert not standing_anchor_clear(n, unsupported, r.free, [10.0, 11.6], 0.4)
    mask = np.ones((60, 60), bool)
    mask[23, :] = False
    n, r = setup(mask)
    assert not standing_anchor_clear(n, surface, r.free, [10.0, 11.6], 0.4)
