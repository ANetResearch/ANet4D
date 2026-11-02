"""Pose-to-pose driving candidates and feedback, independent of actor agendas.

Paths use the rear axle as the no-slip reference. Rendered body origins are
different points: planning/control must not silently treat them as that axle.
Candidates still require road, ground, body, traffic and berth admission.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from awr.life.civil.driving_graph import route_arrays
from awr.life.civil.vehicle_connections import angle_delta, candidates

from .bullet import WIDTH, Control


def axle(profile):
    rear = profile.wheels[profile.wheels[:, 0] < 0, :3]
    front = profile.wheels[profile.wheels[:, 0] > 0, :3]
    if (
        len(rear) != 2
        or len(front) != 2
        or np.ptp(rear[:, 0]) > 1e-7
        or np.ptp(front[:, 0]) > 1e-7
        or abs(rear[:, 1].mean()) > 1e-7
    ):
        raise ValueError("Two aligned rear and front wheels required")
    return float(rear[:, 0].mean()), float(front[:, 0].mean() - rear[:, 0].mean())


def body_to_rear(pose, rear_x):
    p = np.asarray(pose, float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("Finite body pose required")
    return p + np.r_[rear_x * np.array([np.cos(p[2]), np.sin(p[2])]), 0.0]


@dataclass(frozen=True)
class Maneuver:
    points: np.ndarray
    yaw: np.ndarray
    gear: np.ndarray
    rear_x: float

    def __post_init__(self):
        if not np.isfinite(self.rear_x) or self.rear_x >= 0:
            raise ValueError("Explicit rear axle offset required")
        r = route_arrays(self.points, self.yaw, self.gear)
        # A midpoint tangent follows the rear wheel's longitudinal axis, with
        # travel direction reversed for backing. Reject a body-origin path
        # masquerading as an axle path or a lateral translation in place.
        heading = (r["yaw"][:-1] + r["yaw"][1:]) * 0.5
        delta = np.diff(r["points"], axis=0) * r["gear"][:, None]
        direction = np.arctan2(delta[:, 1], delta[:, 0])
        error = np.arctan2(np.sin(direction - heading), np.cos(direction - heading))
        if (np.abs(error) > 0.01).any():
            raise ValueError("Route tangent differs from rear axle heading")
        for name in ["points", "yaw", "gear", "arc"]:
            a = r[name].copy()
            a.flags.writeable = False
            object.__setattr__(self, name, a)

    def body_pose(self, index):
        heading = np.array([np.cos(self.yaw[index]), np.sin(self.yaw[index])])
        return np.r_[self.points[index] - self.rear_x * heading, self.yaw[index]]


def berth_candidates(
    profile,
    start,
    goal,
    *,
    reverse_departure_m=(0.0,),
    reverse_arrival_m=(0.0,),
    allow_three_point=False,
    max_length_m=100.0,
    spacing=0.15,
    approach_m=None,
):
    rear, wheelbase = axle(profile)
    if approach_m is None:
        result = []
        seen = set()
        for length in (0.0, 1.5 * wheelbase, 3 * wheelbase):
            if length >= max_length_m:
                continue
            for r in berth_candidates(
                profile,
                start,
                goal,
                reverse_departure_m=reverse_departure_m,
                reverse_arrival_m=reverse_arrival_m,
                allow_three_point=allow_three_point,
                max_length_m=max_length_m,
                spacing=spacing,
                approach_m=length,
            ):
                key = (r.points.tobytes(), r.yaw.tobytes(), r.gear.tobytes())
                if key not in seen:
                    result.append(r)
                    seen.add(key)
        return sorted(result, key=lambda r: float(r.arc[-1]))
    approach = float(approach_m)
    if not np.isfinite(approach) or approach < 0 or approach >= max_length_m:
        raise ValueError("Finite approach shorter than the journey bound required")
    finish = body_to_rear(goal, rear)
    pre = finish.copy()
    pre[:2] -= approach * np.array([np.cos(pre[2]), np.sin(pre[2])])
    routes = candidates(
        body_to_rear(start, rear),
        pre,
        profile.body_hi - profile.body_lo,
        reverse_departure_m=reverse_departure_m,
        reverse_arrival_m=reverse_arrival_m,
        allow_three_point=allow_three_point,
        max_length_m=max_length_m - approach,
        spacing=spacing,
    )
    result = []
    for r in routes:
        # The older candidate family can emit sub-nanometre connector segments
        # at coincident tangencies. Do not silently delete or smooth these.
        if np.any(np.diff(r["arc"]) <= 1e-9):
            continue
        points, yaw, gear = r["points"], r["yaw"], r["gear"]
        if approach:
            # A berth is a full stopped pose, not a junction waypoint. Reserve
            # an explicit straight alignment approach, subject to the same
            # measured road/parking permission as every other path segment.
            tail = np.linspace(pre[:2], finish[:2], math.ceil(approach / spacing) + 1)[1:]
            points = np.vstack([points, tail])
            yaw = np.r_[yaw, np.full(len(tail), yaw[-1])]
            gear = np.r_[gear, np.ones(len(tail), np.int8)]
        result.append(Maneuver(points, yaw, gear, rear))
    return result


class ManeuverFollower:
    """Bounded force/steering control with actual stopped-pose handoffs.

    No body pose updates. Each direction change requires a full observed stop
    and dwell. Full final pose/speed are checked before reporting arrival.
    Safety caps are supplied by the caller; this class is not a traffic policy.
    """

    def __init__(self, profile, route, observed, *, speed=2.0, reverse_speed=1.0, gear_dwell_s=1.0):
        self.profile = profile
        self.rear, self.wheelbase = axle(profile)
        if abs(self.rear - route.rear_x) > 1e-7:
            raise ValueError("Maneuver and vehicle axle differ")
        self.speed = float(speed)
        self.reverse_speed = float(reverse_speed)
        self.gear_dwell_s = float(gear_dwell_s)
        if not np.isfinite([speed, reverse_speed, gear_dwell_s]).all() or min(speed, reverse_speed, gear_dwell_s) <= 0:
            raise ValueError("Positive finite motion parameters required")
        state = np.asarray(observed, float)
        if state.shape != (WIDTH,) or not np.isfinite(state).all():
            raise ValueError("Finite physical state required")
        if not self.at_pose(state, route.body_pose(0), position=0.1, yaw=0.05, speed=0.08):
            raise ValueError("Maneuver must begin at the actual stopped pose")
        curvature = np.abs(np.diff(route.yaw)) / np.diff(route.arc)
        if curvature.max() > math.tan(profile.max_steer) / self.wheelbase + 1e-5:
            raise ValueError("Maneuver exceeds vehicle steering curvature")
        self.route = route
        self.bounds = np.r_[0, np.flatnonzero(np.diff(route.gear)) + 1, len(route.points) - 1]
        self.section = 0
        self.progress = 0.0
        self.arrived = False
        self.last_time = None
        self.origin_time = float(state[0])
        self.hold_since = None
        self.park_steer = 0.0
        self.gear_events = []
        self.presteer = True
        self.terminal_speed_integral = 0.0

    @staticmethod
    def at_pose(state, goal, *, position=0.06, yaw=0.045, speed=0.04):
        return bool(
            np.linalg.norm(state[1:3] - goal[:2]) <= position
            and abs(angle_delta(state[8], goal[2])) <= yaw
            and abs(state[11]) <= speed
            and np.linalg.norm(state[12:14]) <= speed
            and abs(state[17]) <= 0.04
        )

    def command(self, state, *, dt=1 / 240, speed_limit=None):
        row = np.asarray(state, float)
        if row.shape != (WIDTH,) or not np.isfinite(row).all():
            raise ValueError("Finite physical state required")
        if not np.isfinite(dt) or not 0 < dt <= 0.02:
            raise ValueError("Bounded physics timestep required")
        t = float(row[0])
        if t < self.origin_time or (self.last_time is not None and t <= self.last_time):
            raise ValueError("Fresh world frame required")
        lo, hi = map(int, self.bounds[self.section : self.section + 2])
        r = self.route
        goal = r.body_pose(hi)
        direction = int(r.gear[lo])
        cap = self.speed if direction > 0 else self.reverse_speed
        if speed_limit is not None:
            if not np.isfinite(speed_limit) or not 0 <= speed_limit <= cap:
                raise ValueError("Invalid safety speed cap")
            cap = float(speed_limit)
        elapsed = 0.0 if self.last_time is None else min(0.1, t - self.last_time)
        self.last_time = t
        if self.arrived:
            return Control(self.park_steer, 0.0, self.profile.max_brake_impulse)
        # Capture inside 6 cm, then retain the brake through at most 2 cm of
        # physical settling. The outer hold band stays inside DrivingRoutes'
        # 10 cm handoff contract; yaw, speed and dwell are unchanged. Without
        # hysteresis a millimetre of settling can reapply drive at a cusp.
        if self.at_pose(row, goal, position=0.08 if self.hold_since is not None else 0.06):
            self.terminal_speed_integral = 0.0
            if self.hold_since is None:
                self.hold_since = t
                self.park_steer = float(row[33])
            dwell = self.gear_dwell_s if hi < len(r.points) - 1 else 0.3
            if t - self.hold_since >= dwell - 1e-9:
                if hi == len(r.points) - 1:
                    self.arrived = True
                else:
                    self.gear_events.append(
                        dict(
                            t_s=t,
                            from_gear=direction,
                            to_gear=int(r.gear[hi]),
                            observed_pose=row[[1, 2, 8]].tolist(),
                            speed_mps=float(row[11]),
                        )
                    )
                    self.section += 1
                    self.hold_since = None
                    self.progress = float(r.arc[hi])
                    self.presteer = True
            return Control(self.park_steer, 0.0, self.profile.max_brake_impulse)
        self.hold_since = None
        # Braking is how a moving vehicle becomes stopped. Waiting for the
        # stopped-pose predicate before requesting the brake leaves a low-speed
        # grade bias driving through a valid cusp. Enter capture only when the
        # measured pose plus one control interval of coasting fits INSIDE the
        # existing capture bounds. This does not acknowledge arrival or shift
        # gear: the next actual frame must still satisfy the stopped predicate
        # and complete its ordinary dwell. No velocity or transform is assigned.
        xy_speed = float(np.linalg.norm(row[12:14]))
        if (
            np.linalg.norm(row[1:3] - goal[:2]) + 0.1 * xy_speed <= 0.06
            and abs(angle_delta(row[8], goal[2])) + 0.1 * abs(row[17]) <= 0.045
        ):
            self.terminal_speed_integral = 0.0
            return Control(float(row[33]), 0.0, self.profile.max_brake_impulse)
        # Ground projection of the actual chassis rear reference. Pitch/roll
        # are retained; never fit or reset the physical pose to route geometry.
        rear_xy = row[1:3] + Rotation.from_quat(row[4:8]).apply([self.rear, 0, 0])[:2]
        # The existing causal projection window is contiguous in arc length.
        # Slice it before vector geometry: a long trip must not scan every
        # segment on every vehicle's control tick. Bounds and tie order match
        # the former full-array mask exactly, including both closed ends.
        first = max(lo, int(np.searchsorted(r.arc, self.progress - 0.5, side="left")) - 1)
        stop = min(hi, int(np.searchsorted(r.arc, self.progress + 8.0, side="right")))
        a, b = r.points[first:stop], r.points[first + 1 : stop + 1]
        delta = b - a
        u = np.clip(np.sum((rear_xy - a) * delta, axis=1) / np.sum(delta * delta, axis=1), 0, 1)
        distances = np.linalg.norm(rear_xy - a - u[:, None] * delta, axis=1)
        i = first + int(np.argmin(distances))
        self.progress = max(self.progress, float(r.arc[i] + u[i - first] * (r.arc[i + 1] - r.arc[i])))
        point = a[i - first] + u[i - first] * delta[i - first]
        heading = float(np.interp(self.progress, r.arc, r.yaw))
        lateral = float(np.array([-np.sin(heading), np.cos(heading)]) @ (rear_xy - point))
        heading_error = angle_delta(row[8], heading)
        curvature = float((r.yaw[i + 1] - r.yaw[i]) / (r.arc[i + 1] - r.arc[i]))
        # A finite-rate rack cannot acquire a CSC curve's steering angle at
        # the instant the rear axle crosses its tangent point. Preview half
        # the maximum rack transition at the actual speed, bounded by this
        # gear section. This also slows for the upcoming bend before entry.
        # Feedback still uses the current measured pose; no pose is advanced.
        preview_m = 0.5 * abs(float(row[11])) * self.profile.max_steer / self.profile.steer_rate
        preview_arc = min(float(r.arc[hi]), self.progress + preview_m)
        j = min(hi - 1, max(lo, int(np.searchsorted(r.arc, preview_arc, side="right")) - 1))
        preview_curvature = float((r.yaw[j + 1] - r.yaw[j]) / (r.arc[j + 1] - r.arc[j]))
        remaining = float(r.arc[hi] - self.progress)
        # Critically damped rear-path error in travelled-distance coordinates:
        # e_y'' + (4/L) e_y' + (4/L**2) e_y = 0. Scale by wheelbase rather
        # than a particular vehicle/berth's measured trajectory.
        steer = float(
            np.clip(
                np.arctan(
                    self.wheelbase
                    * (
                        direction * preview_curvature
                        - direction * (4 / self.wheelbase) * heading_error
                        - (4 / self.wheelbase**2) * lateral
                    )
                ),
                -self.profile.max_steer,
                self.profile.max_steer,
            )
        )
        if self.presteer:
            if abs(steer - row[33]) > 0.03:
                return Control(steer, 0.0, self.profile.max_brake_impulse)
            self.presteer = False
        cap = min(cap, math.sqrt(0.25 / max(abs(curvature), abs(preview_curvature), 1e-8)))
        cap = min(cap, max(0.25, cap * (1 - abs(steer - row[33]) / 0.45)))
        desired = direction * min(cap, 0.8 * remaining)
        # Near a stop the grade feed-forward is imperfect: tire contact and
        # static friction need not match chassis pitch. Pure proportional
        # control can keep applying a small brake forever just outside the
        # capture radius. Integrate measured speed error only in the terminal
        # approach, with bounded acceleration authority and no pose changes.
        # Reset on safety stops, direction/pose holds and ordinary cruising so
        # a blocked car cannot store an acceleration surge for release.
        if cap > 0.0 and remaining < self.wheelbase:
            self.terminal_speed_integral = float(
                np.clip(self.terminal_speed_integral + (desired - row[11]) * elapsed, -0.5, 0.5)
            )
        else:
            self.terminal_speed_integral = 0.0
        force = float(
            np.clip(
                self.profile.mass_kg * (4 * (desired - row[11]) + self.terminal_speed_integral - 9.81 * np.sin(row[9])),
                -self.profile.max_force,
                self.profile.max_force,
            )
        )
        if direction * force >= 0:
            return Control(steer, force, 0.0)
        return Control(steer, 0.0, min(self.profile.max_brake_impulse, abs(force) * dt / 4))
