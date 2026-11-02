"""Live car lookahead and joint-frame checks for the mixed life backend.

Global driving geometry is admitted by DrivingRoutes. This module only handles
the bodies present in the current frame. It does not widen streets or compute
new global routes. Prediction is a conservative local heuristic; the joint
swept-body check remains mandatory and a failure cannot become a frozen car.
"""

import math

import numpy as np
from numba import njit

from . import kernels as K


@njit(cache=True)
def first_contact(car, points, headings, arcs, current_arc, xy, yaw, speed, active, ext, radii, prediction_s):
    for q in range(len(arcs)):
        # Predict neighbours at this portion of the braking horizon. A small
        # uncertainty margin grows with prediction age; it is not a route offset.
        age = min((arcs[q] - current_arc) / max(abs(speed[car]), 1.0), prediction_s)
        margin = 0.15 + 0.1 * age
        ax, ay, al, aw = K.box(points[q, 0], points[q, 1], headings[q], 0.0, ext[car])
        for j in range(len(xy)):
            if j == car or not active[j]:
                continue
            px = xy[j, 0] + speed[j] * math.cos(yaw[j]) * age
            py = xy[j, 1] + speed[j] * math.sin(yaw[j]) * age
            if (px - points[q, 0]) ** 2 + (py - points[q, 1]) ** 2 > (radii[car] + radii[j] + margin) ** 2:
                continue
            bx, by, bl, bw = K.box(px, py, yaw[j], abs(speed[j]), ext[j])
            if K.overlap(ax, ay, headings[q], al, aw, bx, by, yaw[j], bl, bw, -margin):
                return q
    return -1


def stopping_arc(driving, evidence, car, xy, yaw, speed, active, crowd):
    plan = driving.plans[evidence]
    route = plan["schedule"].route
    execution = driving.executions.get(plan["vehicle"])
    s = 0.0 if execution is None else execution.arc_m
    v = abs(float(speed[car]))
    horizon = v / driving.limits.braking_mps2 + 1.0
    reach = v * v / (2 * driving.limits.braking_mps2) + max(3.0, v)
    end = min(float(route["arc"][-1]), s + reach)
    arcs = np.linspace(s, end, max(2, int(math.ceil((end - s) / 0.2)) + 1))
    points = np.column_stack([np.interp(arcs, route["arc"], route["points"][:, k]) for k in (0, 1)])
    headings = np.interp(arcs, route["arc"], route["yaw"])
    hit = first_contact(car, points, headings, arcs, s, xy, yaw, speed, active, crowd.ext, crowd.radius, horizon)
    if hit < 0:
        return None
    # Back off from the first occupied sample, including sampling uncertainty.
    return max(s, float(arcs[hit]) - 0.25)


def joint_conflicts(crowd, old_xy, old_yaw, xy, yaw, speed, active):
    """Read-only gate: all active bodies, including vehicle/vehicle conflicts."""
    gx, gy, cell, nx, ny = crowd.grid
    order, start = K.build_grid(
        np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1]), active, gx, gy, cell, nx, ny
    )
    reach = 2.0 * float(crowd.radius.max()) + 2.0 * float(np.linalg.norm(xy - old_xy, axis=1).max()) + 0.2
    pairs = K.swept_conflicts(
        np.flatnonzero(active),
        old_xy[:, 0].copy(),
        old_xy[:, 1].copy(),
        old_yaw,
        xy[:, 0].copy(),
        xy[:, 1].copy(),
        yaw,
        np.abs(speed),
        crowd.ext,
        crowd.radius,
        active,
        order,
        start,
        gx,
        gy,
        cell,
        nx,
        ny,
        reach,
        0.005,
        6,
    )
    return sorted({tuple(sorted(p)) for p in pairs.tolist()})


def free_door(crowd, person, point, heading, xy, yaw, speed, active, *, carrier=None):
    """Free visible endpoint, plus no other body across the transfer corridor.

    Only the carrier's own interior is exempt in the latter check. This does
    not supply an interior animation or reinterpret an occupied door as free.
    """
    if not K.in_cells(*point, crowd.free, crowd.fx0, crowd.fy0, crowd.fstep):
        return False
    if not K.body_supported(
        *point, heading, 0.0, crowd.ext[person], crowd.z_ok, crowd.walk, crowd.sx0, crowd.sy0, crowd.scell
    ):
        return False
    ax, ay, al, aw = K.box(*point, heading, 0.0, crowd.ext[person])
    for j in np.flatnonzero(active):
        if j == person:
            continue
        bx, by, bl, bw = K.box(*xy[j], yaw[j], abs(speed[j]), crowd.ext[j])
        if K.overlap(ax, ay, heading, al, aw, bx, by, yaw[j], bl, bw, -0.02):
            return False
    if carrier is not None:
        # The containment representation has a body-origin change between the
        # carrier centre and door. It cannot pass through a third party.
        start = xy[carrier]
        distance = float(np.linalg.norm(np.asarray(point) - start))
        for u in np.linspace(0.0, 1.0, max(2, int(math.ceil(distance / 0.1)) + 1)):
            q = start + u * (np.asarray(point) - start)
            ax, ay, al, aw = K.box(*q, heading, 0.0, crowd.ext[person])
            for j in np.flatnonzero(active):
                if j in (person, carrier):
                    continue
                bx, by, bl, bw = K.box(*xy[j], yaw[j], abs(speed[j]), crowd.ext[j])
                if K.overlap(ax, ay, heading, al, aw, bx, by, yaw[j], bl, bw, -0.02):
                    return False
    return True
