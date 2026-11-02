"""Bounded offline road repair using the existing search and pinned rsplan.

The reference is the actual rear axle. Every primitive has full planar body
permission and sampled measured center support. Output is candidate-only;
native ground, wheels, body, traffic and activity checks remain mandatory.
No road width, reverse permission, endpoint or target agenda is changed.
"""

import math

import numpy as np

from awr.life.civil.maneuver_search import connect

from .axle_admission import AxleCandidateAdmission
from .guide_support import check_centers
from .maneuver import Maneuver, body_to_rear
from .road_guides import body_route


class MeasuredAxleAdmission(AxleCandidateAdmission):
    def __init__(self, profile, support, *, steering_fraction=0.8, body_clearance=None):
        super().__init__(profile, steering_fraction=steering_fraction)
        self.support = support
        self.body_clearance = body_clearance
        if body_clearance is not None and body_clearance.profile.identity_sha256() != profile.identity_sha256():
            raise ValueError("Mesh clearance must use the same physical profile")

    def pose(self, pose, size, permission, obstacles):
        ok, reason = super().pose(pose, size, permission, obstacles)
        if not ok:
            return ok, reason
        p = np.asarray(pose, float)
        body = p[:2] - self.rear * np.array([math.cos(p[2]), math.sin(p[2])])
        if not self.support.supported([body], "vehicle")[0]:
            return False, "measured_center_support"
        return (True, None) if self.body_clearance is None else self.body_clearance.pose(np.r_[body, p[2]])

    def route(self, route, size, permission, reverse_permission, obstacles):
        ok, reason = super().route(route, size, permission, reverse_permission, obstacles)
        if not ok:
            return ok, reason
        m = Maneuver(route["points"], route["yaw"], route["gear"], self.rear)
        r = check_centers(body_route(m)["points"], self.support)
        if r["rejected_samples"]:
            return False, "measured_center_support"
        return (
            (True, None) if self.body_clearance is None else self.body_clearance.route(body_route(m)["points"], m.yaw)
        )


def search(
    profile,
    support,
    start,
    goal,
    permission,
    *,
    max_expansions=256,
    max_length_m=80.0,
    steering_fraction=0.8,
    body_clearance=None,
    obstacles=(),
):
    """Forward-only local road candidate with exact fixed body endpoints."""
    if (
        type(max_expansions) is not int
        or not 1 <= max_expansions <= 2000
        or not np.isfinite(max_length_m)
        or not 1 <= max_length_m <= 150
    ):
        raise ValueError("Bounded offline road search required")
    gate = MeasuredAxleAdmission(profile, support, steering_fraction=steering_fraction, body_clearance=body_clearance)
    # Chord sampling slightly overestimates curvature. Reserve another 1%
    # beyond the existing steering gate instead of loosening that gate.
    radius = 1.01 * gate.wheelbase / (math.tan(profile.max_steer) * steering_fraction)
    route, stats = connect(
        None,
        body_to_rear(start, gate.rear),
        body_to_rear(goal, gate.rear),
        gate.size,
        maneuver_area=permission,
        obstacles=tuple(obstacles),
        candidate_admission=gate,
        max_expansions=max_expansions,
        max_length_m=max_length_m,
        max_gear_changes=0,
        turn_radius_m=radius,
        allowed_gears=(1,),
    )
    result = None if route is None else Maneuver(route["points"], route["yaw"], route["gear"], gate.rear)
    if result is not None:
        for actual, expected in ((result.body_pose(0), start), (result.body_pose(-1), goal)):
            if (
                np.linalg.norm(actual[:2] - np.asarray(expected)[:2]) > 1e-8
                or abs(math.remainder(float(actual[2] - expected[2]), 2 * math.pi)) > 1e-8
            ):
                stats["final_body_seam_rejected"] = 1
                result = None
                break
    return result, dict(
        stats=stats,
        turn_radius_m=radius,
        max_expansions=max_expansions,
        max_length_m=max_length_m,
        steering_fraction=steering_fraction,
        allowed_gears=[1],
        candidate_only=True,
        release_evidence=False,
        mesh_body_screened=body_clearance is not None,
    )
