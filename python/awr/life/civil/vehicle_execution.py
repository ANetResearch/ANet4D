"""Connect admitted complete vehicle legs to the grounded collision loop."""

import numpy as np

from .driving_motion import Limits, RouteExecution
from .vehicle_connections import angle_delta


class PoseMismatch(RuntimeError):
    def __init__(self, expected, xy, yaw, arc, speed):
        super().__init__("Vehicle execution differs from grounded body; refusing a pose/speed correction")
        self.diagnostic = dict(expected=expected, observed=dict(xy=np.asarray(xy), yaw=yaw, arc_m=arc, speed_mps=speed))


def selected(cfg):
    mode = cfg.get("vehicle_motion_model", "legacy")
    if mode not in ("legacy", "reactive_schedule_v1"):
        raise ValueError("Unknown vehicle motion model")
    if mode == "reactive_schedule_v1":
        # The old delivery adapter mutates goals to intermediate scalar stops.
        # It needs explicit complete-leg handoff before it can use this backend.
        if cfg.get("delivery") or (cfg.get("life") or {}).get("vehicles", {}).get("behavior") != "curb_v1":
            raise ValueError("Reactive vehicle execution requires complete curb legs and no legacy delivery adapter")
        return True
    return False


def assert_pose(execution, xy, yaw, arc, speed):
    state = execution.state()
    if (
        not np.isfinite([*xy, yaw, arc, speed]).all()
        or np.linalg.norm(state["xy"] - xy) > 1e-6
        or abs(angle_delta(state["yaw"], yaw)) > 1e-6
        or abs(state["arc_m"] - arc) > 1e-6
        or abs(state["speed_mps"] - speed) > 1e-6
    ):
        raise PoseMismatch(state, xy, yaw, arc, speed)


def advance_job(job, xy, yaw, dt, *, stop_before_arc_m=None):
    execution = job.get("_vehicle_execution")
    if execution is None:
        route = job["route"]
        if (
            "arc" not in route
            or not route.get("smooth_vehicle_heading")
            or job["direction"] != 1
            or abs(job["s"]) > 1e-9
            or abs(job["goal_s"] - route["arc"][-1]) > 1e-6
        ):
            raise ValueError("Vehicle execution requires a complete admitted leg from its exact start")
        limits = Limits(forward_mps=float(job["max_speed"]), reverse_mps=min(1.4, float(job["max_speed"])))
        execution = RouteExecution(route, limits)
        job["_vehicle_execution_route"] = route
    elif job["_vehicle_execution_route"] is not job["route"]:
        raise RuntimeError("Vehicle route replaced without completing and handing off the previous leg")
    assert_pose(execution, xy, yaw, job["s"], job["speed"])
    state = execution.advance(dt, stop_before_arc_m=stop_before_arc_m)
    job["_vehicle_execution"] = execution
    return state


def confirm_job(job, xy, yaw, arc):
    execution = job.get("_vehicle_execution")
    if execution is not None:
        # A late collision/support veto may not quietly rewind the execution
        # while leaving its speed/clock advanced. The build must preserve and
        # diagnose that failure; it cannot publish that trajectory.
        assert_pose(execution, xy, yaw, arc, job["speed"])


def braking_direction(job):
    execution = job.get("_vehicle_execution")
    return execution.gear if execution is not None else 1
