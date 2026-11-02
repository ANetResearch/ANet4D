"""Bounded weight shifts when a close neighbour or ground edge blocks turning.

Inspired by upstream a2af05c/bf1d5d7's slow edging primitive. This implementation
runs after normal velocity selection and safety resolution, never overwrites a
rigid-body controller, and checks the complete observed rest body against the
same measured support and continuous pair gates. It does not shrink a person,
move an obstacle, skip collision checks or reset the activity stall clock.
"""

import math

import numpy as np
from numba import njit

from . import kernels as K

EDGE_MPS = 0.015  # remains in the asset's existing rest-pose regime, including .01 sweep allowance


@njit(cache=True)
def ground_clearance(x, y, walk, z_ok, x0, y0, cell, reach):
    """Distance from the body pivot to nearby unsupported squares/height patches.

    Local analytic distances, bounded by the body's turning reach. No global
    distance-field rebuild or search. The actual oriented body still needs the
    full support and joint collision checks before any shift is accepted.
    """
    best = reach
    ny, nx = walk.shape
    for shift in (0.5, 1.0):
        c0 = math.floor((x - reach - x0) / cell - (shift - 0.5))
        c1 = math.floor((x + reach - x0) / cell - (shift - 0.5))
        r0 = math.floor((y - reach - y0) / cell - (shift - 0.5))
        r1 = math.floor((y + reach - y0) / cell - (shift - 0.5))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if shift == 0.5:
                    good = 0 <= r < ny and 0 <= c < nx and walk[r, c]
                else:
                    good = (
                        0 <= r < ny - 1
                        and 0 <= c < nx - 1
                        and z_ok[r, c]
                        and z_ok[r + 1, c]
                        and z_ok[r, c + 1]
                        and z_ok[r + 1, c + 1]
                    )
                if good:
                    continue
                dx = max(abs(x - (x0 + (c + shift) * cell)) - cell / 2.0, 0.0)
                dy = max(abs(y - (y0 + (r + shift) * cell)) - cell / 2.0, 0.0)
                best = min(best, math.hypot(dx, dy))
    return best


@njit(cache=True)
def _recover(
    eligible,
    ground_priority,
    xy,
    yaw,
    speed,
    next_xy,
    next_yaw,
    next_speed,
    active,
    cap,
    ext,
    radius,
    z_ok,
    walk,
    sx0,
    sy0,
    scell,
    free,
    fx0,
    fy0,
    fstep,
    gx0,
    gy0,
    cell,
    nx,
    ny,
    dt,
):
    ox = np.ascontiguousarray(xy[:, 0])
    oy = np.ascontiguousarray(xy[:, 1])
    px = np.ascontiguousarray(next_xy[:, 0])
    py = np.ascontiguousarray(next_xy[:, 1])
    sw = np.maximum(np.abs(speed), np.abs(next_speed)) + 0.01
    order, start = K.build_grid(px, py, active, gx0, gy0, cell, nx, ny)
    reach = 2.0 * radius.max() + 0.2 + 2.0 * np.sqrt(((next_xy - xy) ** 2).sum(axis=1)).max()
    nearby = np.empty(max(0, len(xy) - 1), np.int64)
    changed = 0
    attempts = 0
    for i in eligible:
        count = K.near(i, ox, oy, reach, order, start, gx0, gy0, cell, nx, ny, nearby)
        closest = -1
        distance = math.inf
        for k in range(count):
            j = nearby[k]
            d = (ox[j] - ox[i]) ** 2 + (oy[j] - oy[i]) ** 2
            if abs(speed[j]) <= 0.05 and abs(next_speed[j]) <= 0.05 and d < distance:
                closest = j
                distance = d
        close_body = closest >= 0 and distance <= (radius[i] + radius[closest] + 0.2) ** 2
        # A walker can also be unable to rotate at an irregular ground edge,
        # with no close person. Move its pivot toward more measured ground;
        # never step forward into the missing support or snap to a grid cell.
        room = ground_clearance(ox[i], oy[i], walk, z_ok, sx0, sy0, scell, radius[i] + 0.1)
        ground_edge = room < radius[i] + 0.05
        if not close_body and not ground_edge:
            continue
        away = math.atan2(oy[i] - oy[closest], ox[i] - ox[closest]) if close_body else 0.0
        oldx, oldy, oldh, oldv, olds = px[i], py[i], next_yaw[i], next_speed[i], sw[i]
        velocity = min(EDGE_MPS, cap[i])
        if velocity <= 0.0:
            continue
        accepted = False
        angles = np.empty(21)
        scores = np.zeros(21)
        nangle = 0
        if close_body:
            for turn in (0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 2.0, -math.pi / 2.0):
                angle = away + turn
                score = 1.0
                if ground_priority[i]:
                    xx = ox[i] + velocity * dt * math.cos(angle)
                    yy = oy[i] + velocity * dt * math.sin(angle)
                    score = ground_clearance(xx, yy, walk, z_ok, sx0, sy0, scell, radius[i] + 0.1) - room
                    if score <= 1e-8:
                        continue
                angles[nangle] = angle
                scores[nangle] = score
                nangle += 1
        if ground_edge:
            for k in range(16):
                angle = k * math.pi / 8.0
                xx = ox[i] + velocity * dt * math.cos(angle)
                yy = oy[i] + velocity * dt * math.sin(angle)
                gain = ground_clearance(xx, yy, walk, z_ok, sx0, sy0, scell, radius[i] + 0.1) - room
                if gain > 1e-8:
                    angles[nangle] = angle
                    scores[nangle] = gain
                    nangle += 1
        # Stable highest-gain choice. If a full intended turn is ground-blocked,
        # body-avoidance shifts must also increase ground clearance: alternating
        # safe shifts with opposite objectives otherwise create a limit cycle.
        # Every candidate still needs clearance from every observed body.
        for attempt in range(nangle):
            pick = np.argmax(scores[:nangle])
            angle = angles[pick]
            scores[pick] = -math.inf
            attempts += 1
            px[i] = ox[i] + velocity * dt * math.cos(angle)
            py[i] = oy[i] + velocity * dt * math.sin(angle)
            next_yaw[i] = yaw[i]
            next_speed[i] = velocity
            sw[i] = max(abs(speed[i]), velocity) + 0.01
            if not K.swept_supported(
                i,
                ox[i],
                oy[i],
                yaw[i],
                px[i],
                py[i],
                next_yaw[i],
                sw[i],
                ext,
                z_ok,
                walk,
                sx0,
                sy0,
                scell,
                free,
                fx0,
                fy0,
                fstep,
                6,
            ):
                continue
            pairs = K.swept_conflicts(
                np.array([i], np.int64),
                ox,
                oy,
                yaw,
                px,
                py,
                next_yaw,
                sw,
                ext,
                radius,
                active,
                order,
                start,
                gx0,
                gy0,
                cell,
                nx,
                ny,
                reach,
                -K.CONTACT_CLEARANCE_M,
                6,
            )
            if len(pairs) == 0:
                accepted = True
                changed += 1
                next_xy[i, 0] = px[i]
                next_xy[i, 1] = py[i]
                # A shift may cross a hash-cell boundary. Rebuild only after an
                # accepted change so later walkers see the exact joint proposal.
                order, start = K.build_grid(px, py, active, gx0, gy0, cell, nx, ny)
                break
        if not accepted:
            px[i], py[i], next_yaw[i], next_speed[i], sw[i] = oldx, oldy, oldh, oldv, olds
    return changed, attempts


def recover(
    crowd, xy, yaw, speed, next_xy, next_yaw, next_speed, active, mover, wanted, vmax, goal_cap, dt, desired_yaw=None
):
    stalled = (
        active & mover & (crowd.kinds == "person") & (wanted > 0.1) & (np.abs(speed) < 0.03) & (crowd.waited >= 3.0)
    )
    blocked = (np.abs(next_speed) < 0.03) & (np.abs(np.arctan2(np.sin(next_yaw - yaw), np.cos(next_yaw - yaw))) < 1e-6)
    ground_priority = np.zeros(len(xy), bool)
    if desired_yaw is not None:
        # A few degrees of alternating partial turn are not an escape from a
        # ground-edge lock. If the complete intended rest-pose turn cannot fit,
        # permit a supported weight shift before resuming that turn. A feasible
        # normal turn is left untouched (including legitimate long rotations).
        for i in np.flatnonzero(stalled):
            room = ground_clearance(
                *xy[i], crowd.walk, crowd.z_ok, crowd.sx0, crowd.sy0, crowd.scell, crowd.radius[i] + 0.1
            )
            if room >= crowd.radius[i] + 0.05:
                continue
            next_room = ground_clearance(
                *next_xy[i], crowd.walk, crowd.z_ok, crowd.sx0, crowd.sy0, crowd.scell, crowd.radius[i] + 0.1
            )
            if next_room > room + 1e-8:
                continue  # an ordinary step already opens turning room
            turn = K.wrap(desired_yaw[i] - yaw[i])
            for u in np.linspace(0.0, 1.0, max(2, int(math.ceil(abs(turn) / 0.05)) + 1)):
                if not K.body_supported(
                    *xy[i],
                    yaw[i] + turn * u,
                    0.0,
                    crowd.ext[i],
                    crowd.z_ok,
                    crowd.walk,
                    crowd.sx0,
                    crowd.sy0,
                    crowd.scell,
                ):
                    blocked[i] = True
                    ground_priority[i] = True
                    break
    eligible = np.flatnonzero(stalled & blocked)
    if not len(eligible):
        return 0, 0
    return _recover(
        eligible,
        ground_priority,
        xy,
        yaw,
        speed,
        next_xy,
        next_yaw,
        next_speed,
        active,
        np.minimum(vmax, goal_cap),
        crowd.ext,
        crowd.radius,
        crowd.z_ok,
        crowd.walk,
        crowd.sx0,
        crowd.sy0,
        crowd.scell,
        crowd.free,
        crowd.fx0,
        crowd.fy0,
        crowd.fstep,
        *crowd.grid,
        dt,
    )
