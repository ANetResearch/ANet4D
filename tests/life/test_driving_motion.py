import math

import numpy as np
import pytest

from awr.life.civil.driving_graph import route_arrays
from awr.life.civil.driving_motion import Limits, Schedule
from awr.life.civil.vehicle_connections import candidates, three_point_path


def test_acceleration_braking_and_reverse_cusps():
    route = three_point_path([0, 0, 0], [0, 3.2, math.pi], 4.0)
    motion = Schedule(route)
    assert len(motion.gear_stops) == 2
    p = motion.phases
    moving = p[:, -1] != 0
    acceleration = (p[moving, 5] - p[moving, 4]) / (p[moving, 1] - p[moving, 0])
    assert acceleration.max() <= 1.5 + 1e-9
    assert acceleration.min() >= -2.0 - 1e-9
    for start, end, arc in motion.gear_stops:
        assert end - start == pytest.approx(1.0)
        for t in np.linspace(start + 1e-8, end - 1e-8, 11):
            pose = motion.at(t)
            assert pose["speed_mps"] == 0 and pose["gear"] == 0
            assert pose["arc_m"] == pytest.approx(arc)
    for t in np.arange(0, motion.duration_s, 0.1):
        pose = motion.at(t)
        assert pose["speed_mps"] <= (1.4 if pose["gear"] < 0 else 6.0) + 1e-9
    assert motion.at(motion.duration_s)["arrived"]
    np.testing.assert_allclose(motion.at(motion.duration_s)["xy"], [0, 3.2], atol=1e-8)


def test_short_reverse_interval_gets_a_real_stop_and_motion_not_teleport():
    route = route_arrays([[0, 0], [1, 0], [0.999, 0], [2, 0]], [0, 0, 0, 0], [1, -1, 1])
    schedule = Schedule(route)
    assert len(schedule.gear_stops) == 2
    a, b = schedule.gear_stops
    midpoint = schedule.at((a[1] + b[0]) / 2)
    assert midpoint["gear"] == -1
    assert 0.999 < midpoint["xy"][0] < 1.0
    assert midpoint["speed_mps"] > 0


def test_sparse_straight_motion_and_absolute_time_prefix():
    r = route_arrays([[0, 0], [100, 0]], [0, 0])
    # Sparse input only limits achievable speed, never the safety limits.
    m = Schedule(r)
    for t in (0.0, 1.0, 3.7, m.duration_s, m.duration_s + 50.0):
        a, b = m.at(t), Schedule(r).at(t)
        np.testing.assert_array_equal(a["xy"], b["xy"])
        assert a["speed_mps"] == b["speed_mps"]
    assert m.at(-1)["arc_m"] == 0
    assert m.at(m.duration_s + 1)["arc_m"] == 100


def test_turning_speed_obeys_yaw_and_lateral_limits():
    r = candidates([0, 0, 0], [20, 10, math.pi / 2], [4.2, 1.8, 1.5])[0]
    m = Schedule(r)
    for t in np.arange(0.01, m.duration_s, 0.07):
        pose = m.at(t)
        i = min(len(r["gear"]) - 1, np.searchsorted(r["arc"], pose["arc_m"], side="right") - 1)
        curvature = abs(r["yaw"][i + 1] - r["yaw"][i]) / (r["arc"][i + 1] - r["arc"][i])
        assert pose["speed_mps"] * curvature <= m.limits.yaw_rps + 1e-8
        assert pose["speed_mps"] ** 2 * curvature <= m.limits.lateral_mps2 + 1e-8


def test_invalid_limits_and_nonfinite_time_rejected():
    with pytest.raises(ValueError):
        Limits(gear_dwell_s=0)
    with pytest.raises(ValueError):
        Limits(reverse_mps=float("nan"))
    m = Schedule(route_arrays([[0, 0], [1, 0]], [0, 0]))
    with pytest.raises(ValueError):
        m.at(float("nan"))
