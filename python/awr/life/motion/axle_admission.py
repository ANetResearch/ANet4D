"""Planar candidate gates for offline rear-axle maneuver search.

No invented surface or height is supplied to the legacy planner. This adapter
proves only body permission/obstacle/steering conditions; the candidate must
subsequently pass its native physical world, continuous terrain and traffic
checks. The shifted rear reference is never used as a chassis bounding center.
"""

import math

import numpy as np

from awr.life.civil.curb_allocation import footprint

from .maneuver import Maneuver, axle
from .road_guides import body_envelope


class AxleCandidateAdmission:
    def __init__(self, profile, *, steering_fraction=1.0):
        if not np.isfinite(steering_fraction) or not 0 < steering_fraction <= 1:
            raise ValueError("Steering reserve must tighten the physical curvature bound")
        self.profile = profile
        self.steering_fraction = steering_fraction
        self.rear, self.wheelbase = axle(profile)
        self.size = profile.body_hi - profile.body_lo

    def _size(self, size):
        if not np.allclose(size, self.size, rtol=0, atol=1e-8):
            raise ValueError("Candidate search must use the actual physical body")

    def pose(self, pose, size, permission, obstacles):
        self._size(size)
        p = np.asarray(pose, float)
        c, s = math.cos(p[2]), math.sin(p[2])
        middle = (self.profile.body_lo + self.profile.body_hi) / 2
        center = (
            p[:2]
            - self.rear * np.array([c, s])
            + np.array([c * middle[0] - s * middle[1], s * middle[0] + c * middle[1]])
        )
        body = footprint(dict(xy=center, yaw=p[2]), size, margin_m=0.0)
        if not permission.covers(body):
            return False, "outside_permitted_area"
        if any(body.buffer(0.4).intersects(o) for o in obstacles):
            return False, "static_obstacle"
        return True, None

    def route(self, route, size, permission, reverse_permission, obstacles):
        self._size(size)
        try:
            m = Maneuver(route["points"], route["yaw"], route["gear"], self.rear)
        except ValueError:
            return False, "rear_tangent"
        if (
            np.abs(np.diff(m.yaw)) / np.diff(m.arc)
            > math.tan(self.profile.max_steer) * self.steering_fraction / self.wheelbase + 1e-5
        ).any():
            return False, "steering"
        body = body_envelope(m, self.profile)
        if not permission.covers(body):
            return False, "outside_permitted_area"
        if any(body.buffer(0.4).intersects(o) for o in obstacles):
            return False, "static_obstacle"
        if (m.gear < 0).any():
            if reverse_permission is None:
                return False, "reverse_permission_required"
            boundaries = np.r_[0, np.flatnonzero(np.diff(m.gear)) + 1, len(m.gear)]
            for a, b in zip(boundaries, boundaries[1:], strict=False):
                if m.gear[a] == -1:
                    piece = Maneuver(m.points[a : b + 1], m.yaw[a : b + 1], m.gear[a:b], self.rear)
                    if not reverse_permission.covers(body_envelope(piece, self.profile)):
                        return False, "reverse_outside_permitted_area"
        return True, None


def search_access(
    profile,
    start,
    goal,
    kind,
    permission,
    reverse_area,
    *,
    max_expansions=128,
    max_length_m=50.0,
    steering_fraction=0.8,
    support=None,
    max_reverse_leg_m=None,
):
    """Offline bounded fallback with a straight final alignment runway.

    Reuses the existing search and pinned analytic library. No simulated world
    is advanced and no measured surface is replaced by a flat stand-in.
    """
    from awr.life.civil.maneuver_search import connect
    from awr.life.civil.vehicle_connections import PreparedPermission

    from .maneuver import body_to_rear

    if max_reverse_leg_m is not None and (not np.isfinite(max_reverse_leg_m) or max_reverse_leg_m <= 0):
        raise ValueError("Positive finite reverse-leg limit required")
    if support is None:
        gate = AxleCandidateAdmission(profile, steering_fraction=steering_fraction)
    else:
        from .local_road_search import MeasuredAxleAdmission

        gate = MeasuredAxleAdmission(profile, support, steering_fraction=steering_fraction)
    rear, wheelbase = axle(profile)
    alignment = 1.5 * wheelbase
    start, goal = body_to_rear(start, rear), body_to_rear(goal, rear)
    reverse_area = reverse_area.intersection(permission)
    if reverse_area.is_empty:
        return None, [{"reason": "no_explicit_reverse_area"}]
    attempts = []
    for gear in (1, -1) if kind == "entry" else (1,):
        if gear < 0 and max_reverse_leg_m is not None and alignment > max_reverse_leg_m:
            attempts.append(dict(alignment_gear=gear, counts={"alignment_reverse_leg_budget": 1}))
            continue
        pre = goal.copy()
        pre[:2] -= gear * alignment * np.array([math.cos(goal[2]), math.sin(goal[2])])
        route, stats = connect(
            None,
            start,
            pre,
            gate.size,
            maneuver_area=permission,
            reverse_area=reverse_area,
            max_expansions=max_expansions,
            max_length_m=max_length_m - alignment,
            candidate_admission=gate,
            max_reverse_leg_m=max_reverse_leg_m,
            terminal_reverse_reserve_m=alignment if gear < 0 and max_reverse_leg_m is not None else 0.0,
            turn_radius_m=1.01 * wheelbase / (math.tan(profile.max_steer) * steering_fraction),
        )
        attempts.append(dict(alignment_gear=gear, counts=stats))
        if route is None:
            continue
        tail = np.linspace(pre[:2], goal[:2], math.ceil(alignment / 0.15) + 1)[1:]
        result = Maneuver(
            np.vstack([route["points"], tail]),
            np.r_[route["yaw"], np.full(len(tail), route["yaw"][-1])],
            np.r_[route["gear"], np.full(len(tail), gear)],
            rear,
        )
        r = dict(points=result.points, yaw=result.yaw, gear=result.gear)
        if max_reverse_leg_m is not None:
            from awr.life.civil.maneuver_search import reverse_run

            if not reverse_run(dict(r, arc=result.arc), limit_m=max_reverse_leg_m)[0]:
                raise RuntimeError("Access alignment exceeded planned reverse-leg budget")
        ok, reason = gate.route(r, gate.size, PreparedPermission(permission), PreparedPermission(reverse_area), ())
        attempts[-1]["assembled_reason"] = reason
        if ok and result.arc[-1] <= max_length_m + 1e-7:
            return result, attempts
    return None, attempts
