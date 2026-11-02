import numpy as np
import pytest

from awr.life.civil.driving_graph import route_arrays
from awr.life.civil.driving_motion import Limits, RouteExecution, Schedule, UnsafeStop


def straight(length=30):
    return route_arrays([[0, 0], [length, 0]], [0, 0])


def test_clear_execution_matches_nominal_without_horizon_or_actor_id():
    route = straight()
    execution = RouteExecution(route)
    nominal = Schedule(route)
    for k in range(1, 200):
        current = execution.advance(0.1)
        expected = nominal.at(k * 0.1)
        for key in ("xy", "yaw", "arc_m", "speed_mps", "signed_speed_mps"):
            np.testing.assert_allclose(current[key], expected[key], atol=1e-10)
        assert current["arrived"] == expected["arrived"]


def test_obstruction_brakes_moves_while_braking_waits_and_resumes_exactly():
    execution = RouteExecution(straight())
    for _ in range(25):
        execution.advance(0.1)
    before = execution.state()
    stop = before["arc_m"] + before["speed_mps"] ** 2 / (2 * execution.limits.braking_mps2) + 0.2
    history = [before]
    for _ in range(40):
        history.append(execution.advance(0.1, stop_before_arc_m=stop))
    assert history[1]["mode"] == "braking"
    assert history[1]["arc_m"] > before["arc_m"]
    assert 0 < history[1]["speed_mps"] < before["speed_mps"]
    assert max(r["arc_m"] for r in history) <= stop
    assert history[-1]["mode"] == "waiting" and history[-1]["speed_mps"] == 0
    np.testing.assert_allclose(np.diff([r["speed_mps"] for r in history])[:10], -0.2, atol=1e-10)
    parked = history[-1]
    restart = execution.advance(0.1)
    assert 0 < restart["arc_m"] - parked["arc_m"] <= 0.5 * execution.limits.acceleration_mps2 * 0.1**2 + 1e-10
    assert restart["speed_mps"] == pytest.approx(0.15)
    for _ in range(200):
        execution.advance(0.1)
    assert execution.state()["arrived"] and execution.brake_count == 1
    assert execution.time_s > Schedule(straight()).duration_s


def test_late_obstacle_fails_before_any_pose_clock_or_speed_mutation():
    execution = RouteExecution(straight())
    for _ in range(20):
        execution.advance(0.1)
    before = execution.state()
    with pytest.raises(UnsafeStop):
        execution.advance(0.1, stop_before_arc_m=before["arc_m"] + 0.1)
    after = execution.state()
    for key in before:
        np.testing.assert_equal(before[key], after[key])
    for dt in (0, -0.1, 0.2, float("nan")):
        with pytest.raises(ValueError):
            execution.advance(dt)


def test_reverse_motion_recovery_keeps_heading_signed_speed_and_cusp_dwell():
    route = route_arrays([[0, 0], [8, 0], [4, 0], [10, 0]], [0, 0, 0, 0], [1, -1, 1])
    execution = RouteExecution(route)
    records = []
    obstructed = False
    hold_until = None
    for k in range(600):
        before = execution.state()
        stop = None
        if before["gear"] == -1 and before["speed_mps"] > 0.8 and not obstructed:
            stop = before["arc_m"] + before["speed_mps"] ** 2 / (2 * execution.limits.braking_mps2) + 0.1
            obstruction = stop
            hold_until = k + 25
            obstructed = True
        if hold_until is not None and k < hold_until:
            stop = obstruction
        records.append(execution.advance(0.1, stop_before_arc_m=stop))
    assert obstructed and records[-1]["arrived"]
    assert all(r["yaw"] == 0 for r in records)
    assert all(r["signed_speed_mps"] <= 0 for r in records if r["gear"] == -1)
    for cusp in (8.0, 12.0):
        stopped = [r for r in records if abs(r["arc_m"] - cusp) < 1e-8 and r["speed_mps"] == 0]
        assert len(stopped) >= 9  # >=1 s nominal dwell, excluding boundary rounding.
    reverse = [r for r in records if r["gear"] == -1]
    assert max(r["speed_mps"] for r in reverse) <= Limits().reverse_mps + 1e-9
    assert execution.brake_count == 1


def test_wait_inside_cusp_retains_remaining_gear_dwell():
    route = route_arrays([[0, 0], [3, 0], [1, 0]], [0, 0, 0], [1, -1])
    execution = RouteExecution(route)
    while True:
        s = execution.advance(0.1)
        if s["gear"] == 0 and s["arc_m"] > 0:
            break
    first = execution.time_s
    # Interrupt at a cusp, immediately clear the obstacle, and ensure the
    # remaining dwell is not replaced by an immediate reverse acceleration.
    execution.advance(0.1, stop_before_arc_m=3.0)
    while execution.state()["gear"] != -1:
        execution.advance(0.1)
    assert execution.time_s - first >= 0.9 - 1e-8
