"""Optional offline parking planner with explicit, exactly sampled gear cusps.

rsplan supplies analytic segment families, not collision or terrain admission.
Pinning its version also pins the enumeration API used here. Simulation and
ordinary forward road routing do not import this optional dependency.
"""

from __future__ import annotations

import math
from importlib.metadata import version
from types import SimpleNamespace

import numpy as np

from .vehicle_connections import _join, angle_delta


def sample_segments(start, segments, *, spacing=0.25):
    """Integrate signed straight/constant-curvature segments through every cusp.

    A sample is inserted at *each* analytic segment endpoint before joining.
    No interpolated chord is allowed to cross a forward/reverse transition.
    """
    pose = np.asarray(start, float)
    if pose.shape != (3,) or not np.isfinite(pose).all() or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Invalid Reeds-Shepp sampling request")
    point, yaw = pose[:2], float(pose[2])
    pieces = []
    total = 0.0
    for segment in segments:
        length, radius, gear = abs(float(segment.length)), float(segment.turn_radius), segment.direction
        if (
            segment.type not in ("left", "right", "straight")
            or gear not in (-1, 1)
            or not np.isfinite([length, radius]).all()
            or radius <= 0
        ):
            raise ValueError("Invalid analytic car segment")
        if length <= 1e-10:
            continue
        s = np.linspace(0.0, length, max(1, math.ceil(length / spacing)) + 1)
        curvature = {"left": 1.0, "right": -1.0, "straight": 0.0}[segment.type] / radius
        angles = yaw + gear * curvature * s
        if curvature:
            pp = point + np.column_stack([np.sin(angles) - math.sin(yaw), math.cos(yaw) - np.cos(angles)]) / curvature
        else:
            pp = point + gear * s[:, None] * np.array([math.cos(yaw), math.sin(yaw)])
        pieces.append((pp, angles, gear))
        total += length
        point, yaw = pp[-1], float(angles[-1])
    route = _join(pieces)
    if route is not None:
        route["analytic_length_m"] = total
        route["construction"] = "ReedsShepp:rsplan-1.0.10"
    return route


def candidates(
    start,
    goal,
    size,
    *,
    max_length_m=65.0,
    spacing=0.25,
    departure_m=0.0,
    arrival_m=0.0,
    turn_radius_m=None,
    allowed_gears=(1, -1),
):
    """Enumerate bounded-radius parking candidates; caller must admit each one.

    Do not invoke this on unrestricted roads: reversing is permitted only in
    explicitly mapped manoeuvre areas. These are obstacle-free analytic
    families, not a complete solution to obstacle-aware parking.
    """
    start, goal, size = (np.asarray(v, float) for v in (start, goal, size))
    allowed_gears = tuple(allowed_gears)
    if (
        allowed_gears not in ((1,), (-1,), (1, -1))
        or any(type(g) is not int for g in allowed_gears)
        or (turn_radius_m is not None and (not math.isfinite(turn_radius_m) or turn_radius_m <= 0))
    ):
        raise ValueError("Explicit positive radius and canonical gear permission required")
    if (
        any(v.shape != (3,) for v in (start, goal, size))
        or not np.isfinite(np.r_[start, goal, size, max_length_m, spacing, departure_m, arrival_m]).all()
        or np.any(size <= 0)
        or min(max_length_m, spacing) <= 0
    ):
        raise ValueError("Invalid Reeds-Shepp connection request")
    if version("rsplan") != "1.0.10":
        raise RuntimeError("Offline parking planner requires rsplan==1.0.10")
    from rsplan import curves, helpers

    radius = (
        max(4.0, float(size[0]) * 0.65 / math.tan(math.radians(32.0)))
        if turn_radius_m is None
        else float(turn_radius_m)
    )
    # Signed runways reserve straight travel while the body clears its bay.
    # A forward departure or reverse arrival must not cut neighbouring bays.
    a, b = start.copy(), goal.copy()
    a[:2] += departure_m * np.array([math.cos(a[2]), math.sin(a[2])])
    b[:2] -= arrival_m * np.array([math.cos(b[2]), math.sin(b[2])])
    x, y, phi = helpers.change_base(tuple(a), tuple(b))

    def runway(length):
        return SimpleNamespace(
            type="straight", direction=1 if length > 0 else -1, length=abs(length), turn_radius=radius
        )

    result = []
    seen = set()
    for scale in (1.0, 1.25, 1.5):
        for family in (curves.csc, curves.ccc, curves.cccc, curves.ccsc, curves.cscc, curves.ccscc):
            for path in family(tuple(a), tuple(b), spacing, x, y, phi, radius * scale):
                if any(s.direction not in allowed_gears for s in path.segments if abs(s.length) > 1e-10):
                    continue
                if (departure_m and (1 if departure_m > 0 else -1) not in allowed_gears) or (
                    arrival_m and (1 if arrival_m > 0 else -1) not in allowed_gears
                ):
                    continue
                if path.total_length + abs(departure_m) + abs(arrival_m) > max_length_m:
                    continue
                signature = tuple(
                    (s.type, s.direction, round(abs(s.length), 9), round(s.turn_radius, 9))
                    for s in path.segments
                    if abs(s.length) > 1e-10
                )
                if signature in seen:
                    continue
                seen.add(signature)
                segments = ([runway(departure_m)] if departure_m else []) + path.segments
                segments += [runway(arrival_m)] if arrival_m else []
                route = sample_segments(start, segments, spacing=spacing)
                if route is None:
                    continue
                # Never snap a solver endpoint onto the requested pose.
                if (
                    np.linalg.norm(route["points"][-1] - goal[:2]) > 1e-7
                    or abs(angle_delta(route["yaw"][-1], goal[2])) > 1e-7
                ):
                    raise ValueError("Analytic parking solution misses the requested pose")
                result.append(route)
    return sorted(result, key=lambda r: (r["analytic_length_m"], len(r["required_stops_m"])))
