"""Connector broad phase must not discard a real narrow-phase collision."""

import numpy as np

from awr.life.civil.walking import WalkNetwork


def test_round_obstacle_uses_actual_radius_even_when_asset_box_is_small():
    net = WalkNetwork(np.ones((50, 80), bool), [0.0, 0.0])
    free = net.obstacle_mask([[10.0, 10.0]], [[0.2, 0.2, 1.7]], 0.3, [0.0], circle_radii=[2.0])
    assert not net.clear_segment([7.9, 10.0], [7.9, 10.1], free)
    assert net.clear_segment([7.5, 10.0], [7.5, 10.1], free)


def test_rotated_corner_preserves_entire_planning_margin_and_escape_direction():
    net = WalkNetwork(np.ones((50, 80), bool), [0.0, 0.0])
    free = net.obstacle_mask([[10.0, 10.0]], [[2.0, 2.0, 1.7]], 0.3, [np.pi / 4])
    start = np.array([10.0 + np.sqrt(2) + 0.42, 10.0])
    # Inside the .44 m planning margin, outside actual .3 m contact. The old
    # enclosing ball added only .10 m and missed this body on a short segment.
    assert not net.clear_segment(start, start - [0.005, 0.0], free)
    assert net.clear_segment(start, start + [0.005, 0.0], free)
    touching = np.array([10.0 + np.sqrt(2) + 0.29, 10.0])
    assert not net.clear_segment(touching, touching + [0.005, 0.0], free)


def test_long_connector_ignores_off_axis_crowd_but_keeps_rotated_vehicle():
    net = WalkNetwork(np.ones((140, 240), bool), [0.0, 0.0])
    crowd = [[float(x), 30.0] for x in range(10, 100, 2)]
    free = net.obstacle_mask(crowd, [[0.8, 0.6, 1.7]] * len(crowd), 0.3)
    assert net.clear_segment([2.0, 10.0], [110.0, 10.0], free)
    free = net.obstacle_mask(
        crowd + [[60.0, 10.0]],
        [[0.8, 0.6, 1.7]] * len(crowd) + [[12.0, 2.5, 3.0]],
        0.3,
        [0.0] * len(crowd) + [np.pi / 3],
    )
    assert not net.clear_segment([2.0, 10.0], [110.0, 10.0], free)


def test_empty_capsule_keeps_static_cells_and_point_contact_checks():
    terrain = np.ones((60, 80), bool)
    terrain[10:30, 40] = False
    net = WalkNetwork(terrain, [0.0, 0.0])
    free = net.obstacle_mask([[5.0, 25.0]], [[2.0, 2.0, 2.0]], 0.3)
    assert not net.clear_segment([2.0, 10.0], [35.0, 10.0], free)
    assert net.clear_segment([2.0, 20.0], [35.0, 20.0], free)
    assert not net.clear_segment([5.0, 25.0], [5.0, 25.0], free)
    assert net.clear_segment([10.0, 25.0], [10.0, 25.0], free)
