"""Body steering constraints around native crowd execution, not a new planner.

Detour retains routes, avoidance, acceleration and world positions. This layer
owns body facing and uses its public speed parameter to avoid accelerating at
full speed before facing the requested movement. It does not assert foot contact
or play authored turn/start/stop clips; those remain a separate execution gap.
"""

import math
from dataclasses import asdict, dataclass


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


@dataclass
class BodyMotion:
    yaw: float
    desired_yaw: float
    speed_limit_mps: float = 0.0
    direction_source: str = "pending_path"


class NativeLocomotion:
    def __init__(self, crowd, *, max_yaw_rate=1.5, initial_heading=0.0):
        if not crowd.motion_capability:
            raise ValueError("Native steering/speed capability required")
        if not math.isfinite(max_yaw_rate) or max_yaw_rate <= 0 or not math.isfinite(initial_heading):
            raise ValueError("Finite heading and positive yaw rate required")
        self.crowd = crowd
        self.max_yaw_rate = max_yaw_rate
        self.bodies = {who: BodyMotion(initial_heading, initial_heading) for who in crowd.actors}
        self.last_time = None

    def before_step(self, dt):
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("Step must be in (0,0.1] seconds")
        if set(self.bodies) != set(self.crowd.actors):
            raise ValueError("Population changed without motion binding")
        if self.last_time is not None and self.crowd.time <= self.last_time:
            raise ValueError("Body control cannot advance twice at the same world time")
        for who, body in self.bodies.items():
            state = self.crowd.state(who)
            steering = self.crowd.steering(who)
            direction = None
            body.direction_source = "idle"
            if state.goal is not None:
                # The path corner remains available when maxSpeed is zero, unlike
                # dvel/nvel. This permits alignment without cancelling the goal.
                if steering.next_corner is not None:
                    direction = tuple(a - b for a, b in zip(steering.next_corner[:2], state.position[:2], strict=False))
                    body.direction_source = "native_path_corner"
                else:
                    body.direction_source = "pending_path"
                # Follow native avoidance while already travelling. Never replace
                # its velocity with a straight-to-goal application vector.
                if direction is not None and state.speed > 0.08 and math.hypot(*steering.avoidance_velocity[:2]) > 0.08:
                    direction = steering.avoidance_velocity[:2]
                    body.direction_source = "native_avoidance"
            elif state.speed > 0.08:
                direction = state.velocity[:2]
                body.direction_source = "braking_velocity"
            if direction is not None and math.hypot(*direction) > 1e-5:
                body.desired_yaw = math.atan2(direction[1], direction[0])
                turn = angle(body.desired_yaw - body.yaw)
                body.yaw += max(-self.max_yaw_rate * dt, min(self.max_yaw_rate * dt, turn))
            if state.goal is None or body.direction_source == "pending_path":
                limit = 0.0
            else:
                limit = steering.nominal_speed * max(0.0, math.cos(angle(body.desired_yaw - body.yaw)))
            self.crowd.speed_limit(who, limit)
            body.speed_limit_mps = limit
        self.last_time = self.crowd.time

    def sample(self, who):
        body = self.bodies[who]
        state = self.crowd.state(who)
        if state.goal is None:
            phase = "braking" if state.speed > 0.08 else "idle"
        elif body.direction_source == "pending_path":
            phase = "path_pending"
        elif body.speed_limit_mps < 0.08:
            phase = "aligning"
        elif state.speed < 0.3:
            phase = "low_speed"
        else:
            phase = "travelling"
        error = angle(math.atan2(state.velocity[1], state.velocity[0]) - body.yaw) if state.speed > 0.03 else 0.0
        return dict(
            **asdict(body),
            phase=phase,
            movement_heading_error_rad=error,
            authored_transition=False,
            time_s=self.crowd.time,
        )
