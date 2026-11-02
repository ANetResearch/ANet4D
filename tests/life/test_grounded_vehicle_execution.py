import numpy as np
import pytest

from awr.life.civil.driving_graph import route_arrays
from awr.life.civil.vehicle_execution import advance_job, braking_direction, confirm_job, selected


def job(reverse=False):
    r = route_arrays([[0, 0], [-4 if reverse else 4, 0]], [0, 0], [-1 if reverse else 1])
    return dict(route=r, s=0.0, goal_s=4.0, speed=0.0, direction=1, max_speed=3.0)


def test_grounded_reverse_moves_back_without_turning_and_reserves_backwards():
    j = job(True)
    xy = np.array([0.0, 0.0])
    yaw = 0.0
    for _ in range(10):
        p = advance_job(j, xy, yaw, 0.1)
        j["speed"] = p["speed_mps"]
        confirm_job(j, p["xy"], p["yaw"], p["arc_m"])
        j["s"] = p["arc_m"]
        xy = p["xy"]
        yaw = p["yaw"]
        assert xy[0] < 0 and yaw == 0 and braking_direction(j) == -1


def test_grounded_late_hold_route_change_and_partial_leg_fail_closed():
    j = job()
    p = advance_job(j, np.zeros(2), 0.0, 0.1)
    j["speed"] = p["speed_mps"]
    with pytest.raises(RuntimeError, match="grounded body"):
        confirm_job(j, np.zeros(2), 0.0, 0.0)
    j["s"] = p["arc_m"]
    j["route"] = dict(j["route"])
    with pytest.raises(RuntimeError, match="route replaced"):
        advance_job(j, p["xy"], p["yaw"], 0.1)
    j = job()
    j["goal_s"] = 3.0
    with pytest.raises(ValueError, match="complete admitted leg"):
        advance_job(j, np.zeros(2), 0.0, 0.1)


def test_opt_in_requires_supported_adapter_and_keeps_legacy_explicit():
    assert not selected({})
    assert selected(dict(vehicle_motion_model="reactive_schedule_v1", life=dict(vehicles=dict(behavior="curb_v1"))))
    for cfg in (
        dict(vehicle_motion_model="typo"),
        dict(vehicle_motion_model="reactive_schedule_v1"),
        dict(vehicle_motion_model="reactive_schedule_v1", delivery={"enabled": True}),
    ):
        with pytest.raises(ValueError):
            selected(cfg)


def test_first_handoff_mismatch_carries_expected_and_actual_pose():
    from awr.life.civil.vehicle_execution import PoseMismatch

    j = job()
    with pytest.raises(PoseMismatch) as error:
        advance_job(j, np.zeros(2), 0.001, 0.1)
    assert error.value.diagnostic["expected"]["yaw"] == 0.0
    assert error.value.diagnostic["observed"]["yaw"] == 0.001
    assert "_vehicle_execution" not in j
