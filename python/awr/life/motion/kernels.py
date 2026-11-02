"""Compiled per-tick kernels of the crowd runtime (Numba; no fast-math; fixed iteration order except that velocity
selection runs walkers in parallel threads, each walker's result being independent of the others and of the schedule).

Bodies are oriented boxes. A person's extents depend on its speed exactly as in awr.life.civil.body (rest pose at or
below REST_BELOW_MPS, linear blend to the walk extents at FULL_WALK_MPS); vehicles and objects have equal rest and
walk extents. `overlap` is the separating-axis predicate of awr.life.civil.collision.obb_overlap; a negative `eps`
demands that much clearance.

Velocity selection follows the sampling and scoring scheme of Recast/DetourCrowd's obstacle-avoidance query,
with candidates expressed as lateral offsets from the walker's lane and speeds (a pedestrian sidesteps rather
than turning to a fixed new heading): each is scored by deviation from the desired and current velocity, a side
bias (keep right) and the time to the first predicted contact. Contacts are predicted with the oriented pose
boxes (own body integrated with turn-rate and acceleration limits; neighbours at constant velocity), static
support with the measured support surface and the walking cells. Path following (`desire_all`) is the compiled
form of awr.life.motion.crowd.Path.desire.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from awr.life.civil.body import FULL_WALK_MPS, REST_BELOW_MPS

# The same minimum separation applies to forecasts and committed swept motion.
# A softer forecast can promise a pass that the execution gate cannot accept,
# after the walker has already consumed its braking distance. Social margins
# may be larger; close-neighbour negotiation may not reduce this physical rule.
CONTACT_CLEARANCE_M = 0.01
PASS_SIDE_M = 0.25
TIGHT_CLOSING_MPS = 0.3
TIGHT_GAP_M = (
    CONTACT_CLEARANCE_M + 0.001
)  # just above the safety pass's clearance (Crowd.step): a step chosen here passes it
TIGHT_SLACK_M = 0.002  # how much closer than it is now a walker may come to a neighbour already within its margin
PASS_ROOM_S = 0.08
PASS_SLOW_MPS = 1.2
TURN_ALIGNED_RAD = 0.15
SIDESTEP_RAD = (math.pi / 4.0, 3.0 * math.pi / 8.0, math.pi / 2.0)
SIDESTEP_KEEP = 0.5  # cost bonus for keeping last tick's sidestep: no flipping between two sides while turning
EDGE_RELIEF = 0.5  # edging costs this much less than standing when standing is futile (blocked ahead, long wait)


@njit(cache=True)
def box(px, py, yaw, speed, ext):
    """Centre and half sizes of the pose box (ext: (2, 4) rest/walk extents [x_back, x_front, y_right, y_left])."""
    b = 0.0 if speed <= REST_BELOW_MPS else min(speed / FULL_WALK_MPS, 1.0)
    e0 = ext[0, 0] * (1.0 - b) + ext[1, 0] * b
    e1 = ext[0, 1] * (1.0 - b) + ext[1, 1] * b
    e2 = ext[0, 2] * (1.0 - b) + ext[1, 2] * b
    e3 = ext[0, 3] * (1.0 - b) + ext[1, 3] * b
    ox, oy = (e0 + e1) / 2.0, (e2 + e3) / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    return px + ox * c - oy * s, py + ox * s + oy * c, (e1 - e0) / 2.0, (e3 - e2) / 2.0


@njit(cache=True)
def overlap(ax, ay, ayaw, ahl, ahw, bx, by, byaw, bhl, bhw, eps):
    dx, dy = bx - ax, by - ay
    ux, uy = math.cos(ayaw), math.sin(ayaw)
    bux, buy = math.cos(byaw), math.sin(byaw)
    c, s = abs(math.cos(byaw - ayaw)), abs(math.sin(byaw - ayaw))
    if abs(dx * ux + dy * uy) >= ahl + bhl * c + bhw * s - eps:
        return False
    if abs(-dx * uy + dy * ux) >= ahw + bhl * s + bhw * c - eps:
        return False
    if abs(dx * bux + dy * buy) >= bhl + ahl * c + ahw * s - eps:
        return False
    if abs(-dx * buy + dy * bux) >= bhw + ahl * s + ahw * c - eps:
        return False
    return True


@njit(cache=True)
def gap(ax, ay, ayaw, ahl, ahw, bx, by, byaw, bhl, bhw):
    """Separation of two pose boxes along their separating axes: overlap(..., -m) holds exactly when it is below m
    (negative when the boxes overlap)."""
    dx, dy = bx - ax, by - ay
    ux, uy = math.cos(ayaw), math.sin(ayaw)
    bux, buy = math.cos(byaw), math.sin(byaw)
    c, s = abs(math.cos(byaw - ayaw)), abs(math.sin(byaw - ayaw))
    return max(
        abs(dx * ux + dy * uy) - (ahl + bhl * c + bhw * s),
        abs(-dx * uy + dy * ux) - (ahw + bhl * s + bhw * c),
        abs(dx * bux + dy * buy) - (bhl + ahl * c + ahw * s),
        abs(-dx * buy + dy * bux) - (bhw + ahl * s + ahw * c),
    )


@njit(cache=True)
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@njit(cache=True)
def supported(x, y, z_ok, walk, x0, y0, cell):
    """SurfaceMap.supported for one point: inside, four finite heights around it and a walkable cell."""
    gx, gy = (x - x0) / cell - 0.5, (y - y0) / cell - 0.5
    ix, iy = int(math.floor(gx)), int(math.floor(gy))
    ny, nx = walk.shape
    if ix < 0 or iy < 0 or ix + 1 >= nx or iy + 1 >= ny:
        return False
    if not (z_ok[iy, ix] and z_ok[iy, ix + 1] and z_ok[iy + 1, ix] and z_ok[iy + 1, ix + 1]):
        return False
    cx = min(max(int(math.floor((x - x0) / cell)), 0), nx - 1)
    cy = min(max(int(math.floor((y - y0) / cell)), 0), ny - 1)
    return walk[cy, cx]


@njit(cache=True)
def in_cells(x, y, free, fx0, fy0, step):
    """Centre within the walking cells (WalkNetwork.free, origin at the centre of cell [0, 0])."""
    c = int(math.floor((x - fx0) / step + 0.5))
    r = int(math.floor((y - fy0) / step + 0.5))
    if r < 0 or c < 0 or r >= free.shape[0] or c >= free.shape[1]:
        return False
    return free[r, c]


@njit(cache=True)
def body_supported(px, py, yaw, speed, ext, z_ok, walk, x0, y0, cell):
    """The pose box on the measured support: its four corners pass the tape QA's point test (`supported`), and no
    support cell that the box overlaps lacks walkable ground or a height. The corner test alone admits a box that
    straddles such a cell and is not monotone in the box: a walker whose walking box spans a hole on its corners
    could not stop there (its shorter rest box has a corner in the hole). With the cell test, a supported box
    implies that every box nested in it is supported, in particular the rest pose at the same place."""
    cx, cy, hl, hw = box(px, py, yaw, speed, ext)
    c, s = math.cos(yaw), math.sin(yaw)
    for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        if not supported(cx + sx * hl * c - sy * hw * s, cy + sx * hl * s + sy * hw * c, z_ok, walk, x0, y0, cell):
            return False
    ex, ey = abs(hl * c) + abs(hw * s), abs(hl * s) + abs(hw * c)  # half extents of the box's bounding rectangle
    ny, nx = walk.shape
    c0, c1 = int(math.floor((cx - ex - x0) / cell)), int(math.floor((cx + ex - x0) / cell))
    r0, r1 = int(math.floor((cy - ey - y0) / cell)), int(math.floor((cy + ey - y0) / cell))
    half = cell / 2.0
    for r in range(r0, r1 + 1):
        for q in range(c0, c1 + 1):
            if 0 <= r < ny and 0 <= q < nx and walk[r, q] and z_ok[r, q]:
                continue
            # separating axes of the box and the cell (the cell's axes are tested by the bounding rectangle)
            dx, dy = x0 + (q + 0.5) * cell - cx, y0 + (r + 0.5) * cell - cy
            if abs(dx) >= ex + half - 1e-9 or abs(dy) >= ey + half - 1e-9:
                continue
            if (
                abs(dx * c + dy * s) >= hl + half * (abs(c) + abs(s)) - 1e-9
                or abs(-dx * s + dy * c) >= hw + half * (abs(c) + abs(s)) - 1e-9
            ):
                continue
            return False
    return True


@njit(cache=True)
def body_clear(px, py, yaw, speed, ext, z_ok, walk, x0, y0, cell, sdist):
    """body_supported with a fast accept: `sdist` holds, per support cell, the distance in cells (capped at 255)
    to the nearest cell that is not walkable or lacks a height. Every cell a corner or cell test reads lies within
    the box radius plus 2.2 cells of the centre's cell, so a larger clearance there proves the tests pass."""
    cx, cy, hl, hw = box(px, py, yaw, speed, ext)
    c = int(math.floor((cx - x0) / cell))
    r = int(math.floor((cy - y0) / cell))
    ny, nx = walk.shape
    if 0 <= c < nx and 0 <= r < ny and sdist[r, c] > math.hypot(hl, hw) / cell + 2.2:
        return True
    return body_supported(px, py, yaw, speed, ext, z_ok, walk, x0, y0, cell)


@njit(cache=True)
def support_away(x, y, sdist, x0, y0, cell):
    """Direction in which the clearance to unsupported cells grows at (x, y) (central differences of `sdist`
    over two cells), or NaN where it is flat."""
    c = int(math.floor((x - x0) / cell))
    r = int(math.floor((y - y0) / cell))
    ny, nx = sdist.shape
    if r < 2 or c < 2 or r >= ny - 2 or c >= nx - 2:
        return np.nan
    gx = (float(sdist[r, c + 1]) + float(sdist[r, c + 2])) - (float(sdist[r, c - 1]) + float(sdist[r, c - 2]))
    gy = (float(sdist[r + 1, c]) + float(sdist[r + 2, c])) - (float(sdist[r - 1, c]) + float(sdist[r - 2, c]))
    if gx == 0.0 and gy == 0.0:
        return np.nan
    return math.atan2(gy, gx)


@njit(cache=True)
def build_grid(px, py, active, gx0, gy0, gcell, gnx, gny):
    """Bodies sorted by grid cell: (order, start) with start[c]..start[c+1] the bodies of cell c."""
    n = len(px)
    cellof = np.full(n, -1, np.int64)
    counts = np.zeros(gnx * gny + 1, np.int64)
    for i in range(n):
        if not active[i]:
            continue
        cx = min(max(int((px[i] - gx0) / gcell), 0), gnx - 1)
        cy = min(max(int((py[i] - gy0) / gcell), 0), gny - 1)
        cellof[i] = cy * gnx + cx
        counts[cellof[i] + 1] += 1
    start = np.cumsum(counts)
    fill = start[:-1].copy()
    order = np.empty(start[-1], np.int64)
    for i in range(n):
        if cellof[i] >= 0:
            order[fill[cellof[i]]] = i
            fill[cellof[i]] += 1
    return order, start


@njit(cache=True)
def near(i, px, py, reach, order, start, gx0, gy0, gcell, gnx, gny, out):
    """Indices of bodies within `reach` of body i (excluding i) into `out`; returns the count."""
    k = 0
    x, y = px[i], py[i]
    c0 = max(int((x - reach - gx0) / gcell), 0)
    c1 = min(int((x + reach - gx0) / gcell), gnx - 1)
    r0 = max(int((y - reach - gy0) / gcell), 0)
    r1 = min(int((y + reach - gy0) / gcell), gny - 1)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cell = r * gnx + c
            for q in range(start[cell], start[cell + 1]):
                j = order[q]
                if j != i and (px[j] - x) ** 2 + (py[j] - y) ** 2 <= reach * reach and k < len(out):
                    out[k] = j
                    k += 1
    return k


@njit(cache=True, parallel=True)
def choose_velocities(
    walkers,
    px,
    py,
    yaw,
    speed,
    vx,
    vy,
    ext,
    radius,
    active,
    recip,
    des_yaw,
    des_speed,
    max_speed,
    ax_,
    ay_,
    tangent,
    lane,
    order,
    start,
    gx0,
    gy0,
    gcell,
    gnx,
    gny,
    z_ok,
    walk,
    sx0,
    sy0,
    scell,
    sdist,
    free,
    fx0,
    fy0,
    fstep,
    offsets,
    speed_fracs,
    creep,
    creep_room,
    moving_room,
    times,
    turn_rate,
    accel,
    decel,
    margin,
    margin_growth,
    w_des,
    w_cur,
    w_side,
    w_toi,
    reach_walk,
    goal_dist,
    personal,
    stand_pen,
    nudge,
    sidestep,
    prev_aim,
    pred_x,
    pred_y,
    pred_yaw,
    seen_standing,
):
    """Best (heading, speed, direction of motion) per walker and the predicted time to the first contact for it.
    The direction of motion is NaN (along the heading) except for an edging step (`nudge`, below).

    A candidate is a lateral offset from the walker's kept lane (`lane` metres left of the path line with anchor
    point `ax_, ay_` and direction `tangent`) and a speed: the predicted walker steers towards the offset line
    (aim point 0.4 m + 0.7 s ahead) and then follows it, as a pedestrian keeps a sidestep while passing. Standing
    still (turning towards the desired heading) is always a candidate. The returned heading is the candidate's
    first commanded heading. The predicted walker stops at its goal (`goal_dist` ahead). Neighbours follow
    `predict_bodies` (`pred_*`, one column per sample of `times`: walking people keep turning towards the heading
    they last chose), except an oncoming walking person, who is predicted at its present velocity (with the
    reciprocal sidestep below): two people passing each other would otherwise each credit the other's last turn,
    relax their own sidestep in the same tick and resume it in the next, in step, and the body `seen_standing[i]`
    (-1 for none), which walker i predicts where it stands: a car about to set off that the walker is already in
    the way of (the caller's choice). The samples are dense in the near term and sparse later; where two bodies are near, the
    interval between samples is tested at steps of at most 0.15 m of closing. Leaving the support counts as a
    contact.

    People keep `personal[i]` metres more from a person who stands or crosses their way (not one walking along
    with them, nor an oncoming one, who is passed by the sidestep below), so that they wait before a crossing
    flow instead of closing in on it; the caller lowers it for a person who has been waiting (people accept a
    closer pass rather than wait indefinitely). From an oncoming walking person they keep PASS_ROOM_S seconds of
    the closing speed above PASS_SLOW_MPS more (0.1 m at 2.5 m/s): a fast pass leaves room for the other's
    deviation, and two people in a narrow passage slow down to pass closely.

    Passing on the right is reciprocal: for a candidate to the right, an oncoming walking person (`recip`) is
    predicted to make the mirrored sidestep to its own right, so that two people in a narrow passage each take
    half of the clearance. Two people already set to pass left to left by more than PASS_SIDE_M (the signed miss
    distance of their positions moving along their path directions, which is the same for both) keep their sides
    instead (the mirrored sidestep applies to candidates to the left), rather than both crossing over. The side
    is decided for the pair: judged from each walker's own path frame, two people whose paths are not exactly
    opposite may each see the other on a different side and sidestep the same way. The safety pass still checks
    the realised steps.

    Besides the speeds `speed_fracs` of the desired speed, every offset is also tried at the creep speed `creep`:
    the body stays close to its compact rest pose, so a walker can turn and edge through a dense group. A creeping
    walker keeps `creep_room` more from people, so that it does not edge into someone's face. Everybody keeps
    `moving_room` more from a moving vehicle (predicted along its route by the caller), also from a standing one
    predicted to set off (awr.life.motion.engine.car_intent_mps), unless the walker sees it standing (`seen_standing`):
    a walker that kept only the margin from a car about to set off came within the room the car keeps from people
    and held it (cq3p-d8-s4 v1, person-022 beside vehicle-042).
    `stand_pen` is added to the standing candidate: it grows while the walker waits, so that a waiting walker
    accepts a detour or a slow step rather than waiting for the others indefinitely. A waiting walker also tries
    creeping along its present heading and up to 0.6 rad to either side, turning as it goes: a walker whose turn
    on the spot is blocked can then edge out of the knot first.

    A neighbour that is already closer than the margins (two people who ended up abeam, not closing in at
    TIGHT_CLOSING_MPS or more) only has to keep TIGHT_GAP_M, just above the gap the safety pass requires, so that the
    two can slide apart instead of freezing; the safety pass keeps its 0.01 m. The walker may not close in on such
    a neighbour that is walking, though: the separation of the two pose boxes (`gap`) must not fall more than
    TIGHT_SLACK_M below both the separation now and the one the walker would have if it stayed where it is (same
    box, the neighbour where it is predicted), so that only the walker's own motion counts. Otherwise two people
    walking abreast inside each other's margin (one overtaking the other) drift together to the 0.011 m, and when
    one of them then turns, every step of either would touch the other. Standing bodies are exempt: turning beside
    a parked car or a person at a stay may come closer down to TIGHT_GAP_M.

    A walker with `nudge[i]` > 0 (it has waited long; the caller gives a speed at which the body keeps its rest
    pose) also tries edging aside: moving at that speed in one of eight directions around its heading, which it
    keeps. Such a step costs as much as standing plus one (and up to two more the less directly it moves away from
    the nearest body), so it is taken only where standing is itself blocked (two people locked abeam, each needing
    to turn into the other, or one whose start would swing its walking pose into the person behind it) and moving
    off is not possible; it then opens the gap.

    The support is tested between the samples too (at most 0.05 m apart, and at the safety pass's five points in
    the first step): thin unsupported features must not slip between two samples."""
    m = len(walkers)
    best_h = np.empty(m)
    best_s = np.empty(m)
    best_t = np.empty(m)
    best_m = np.full(m, np.nan)
    horizon = times[-1]
    last = len(times) - 1
    sets_off = np.zeros(len(px), np.bool_)  # standing vehicles predicted to move (to set off)
    for j in range(len(px)):
        sets_off[j] = (
            not recip[j] and speed[j] <= 0.05 and (pred_x[j, last] - px[j]) ** 2 + (pred_y[j, last] - py[j]) ** 2 > 1e-4
        )
    for w in prange(m):  # walkers are independent: the result does not depend on the thread schedule
        i = walkers[w]
        nb = np.empty(256, np.int64)
        if des_speed[i] <= 0.0:
            best_h[w] = des_yaw[i]
            best_s[w] = 0.0
            best_t[w] = horizon  # turning on the spot or standing
            continue
        cnt = near(i, px, py, reach_walk, order, start, gx0, gy0, gcell, gnx, gny, nb)
        # Neighbours already within the margin (two people who ended up abeam, closer than planning would bring
        # them) only have to stay out of contact, so that the two can slide apart instead of freezing.
        tight = np.zeros(cnt, np.bool_)
        gap0 = np.full(cnt, -math.inf)  # separation from a walking tight neighbour now (box as in the samples)
        ox_, oy_, ohl, ohw = box(px[i], py[i], yaw[i], speed[i], ext[i])
        for a in range(cnt):
            j = nb[a]
            d2 = (px[j] - px[i]) ** 2 + (py[j] - py[i]) ** 2
            # (not two who close in on each other: they keep their margins)
            closing = ((vx[i] - vx[j]) * (px[j] - px[i]) + (vy[i] - vy[j]) * (py[j] - py[i])) / math.sqrt(
                max(d2, 1e-12)
            )
            if d2 <= (radius[i] + radius[j] + 0.3) ** 2 and closing < TIGHT_CLOSING_MPS:
                qx_, qy_, qhl, qhw = box(px[j], py[j], yaw[j], speed[j], ext[j])
                tight[a] = overlap(
                    ox_, oy_, yaw[i], ohl, ohw, qx_, qy_, yaw[j], qhl, qhw, -(margin + personal[i] + creep_room)
                )
                if tight[a] and speed[j] > 0.05:
                    sx_, sy_, shl, shw = box(px[i], py[i], yaw[i], speed[i] + 0.01, ext[i])
                    gap0[a] = gap(sx_, sy_, yaw[i], shl, shw, qx_, qy_, yaw[j], qhl, qhw)
                else:
                    gap0[a] = -math.inf  # no closing-in test
        vmax = max(max_speed[i], 1e-3)
        dvx, dvy = des_speed[i] * math.cos(des_yaw[i]), des_speed[i] * math.sin(des_yaw[i])
        cvx, cvy = speed[i] * math.cos(yaw[i]), speed[i] * math.sin(yaw[i])
        tx, ty = math.cos(tangent[i]), math.sin(tangent[i])
        nx, ny = -ty, tx
        lat0 = (px[i] - ax_[i]) * nx + (py[i] - ay_[i]) * ny
        best = 1e300
        bh, bs, bt = yaw[i], 0.0, 0.0
        nsp = len(speed_fracs) + 1
        ncand = len(offsets) * nsp + 1
        nfree = 5 if stand_pen[i] > 0.0 else 0
        nnudge = 8 if nudge[i] > 0.0 else 0
        nside = 2 * len(SIDESTEP_RAD) if sidestep[i] else 0
        static_block = False  # the straight candidate meets a body that stands, or the end of the support, within 1 s
        support_block = False  # ... the end of the support
        bm = np.nan
        bturn = False
        away = np.nan  # an edging step opens the gap to the nearest body: the direction away from it
        if nnudge:
            dbest = 1e300
            for a in range(cnt):
                j = nb[a]
                dd = (px[j] - px[i]) ** 2 + (py[j] - py[i]) ** 2
                if j != i and dd < dbest:
                    dbest = dd
                    away = math.atan2(py[i] - py[j], px[i] - px[j])
        for q in range(ncand + nfree + nnudge + nside):
            off = 0.0
            rel = 0.0
            aim = 0.4
            creeping = False
            mdir = np.nan
            turnfirst = False
            fixed = False
            if q >= ncand + nfree + nnudge:
                # Stepping aside from a standing obstacle close ahead (a parked car, a person at a stay, the edge of
                # the support) after a long wait: turn on the spot 45-90 degrees off the path, then creep, keeping the
                # heading. Its walking pose reaches further than its rest pose, so a walker facing the obstacle cannot
                # set off towards it, nor turn away while walking.
                if not static_block:
                    continue
                qq = q - ncand - nfree - nnudge
                h0 = wrap(tangent[i] + (1.0 if qq % 2 == 0 else -1.0) * SIDESTEP_RAD[qq // 2])
                sp = min(creep, des_speed[i])
                creeping = True
                fixed = True
                turnfirst = abs(wrap(h0 - yaw[i])) > TURN_ALIGNED_RAD
            elif q >= ncand + nfree:
                # Edging aside: a slow step in any direction, keeping the heading and the rest pose.
                h0, sp, creeping = yaw[i], nudge[i], True
                mdir = wrap(yaw[i] + (q - ncand - nfree) * math.pi / 4.0)
            elif q >= ncand:
                # A waiting walker may also edge off along its present heading, turning as it goes, but not away
                # from its way: facing away from it (it could not turn), creeping on only takes it further in.
                h0, sp = wrap(yaw[i] + (q - ncand - 2) * 0.3), min(creep, des_speed[i])
                creeping = True
                if abs(wrap(h0 - des_yaw[i])) > math.pi / 2.0:
                    continue
            elif q == ncand - 1:
                h0, sp = des_yaw[i], 0.0  # stop where it stands, turning towards the desired heading
            else:
                rel = offsets[q // nsp]
                off = lane[i] + rel
                if q % nsp == nsp - 1:
                    sp = min(creep, des_speed[i])
                    creeping = True
                else:
                    sp = des_speed[i] * speed_fracs[q % nsp]
                aim = 0.4 + 0.7 * sp  # a slower walker sidesteps within a shorter distance
                h0 = wrap(tangent[i] + math.atan2(off - lat0, aim))
            x, y, th, v = px[i], py[i], yaw[i], speed[i]
            h = h0
            turning = turnfirst
            tmin = horizon
            trav = 0.0
            moved = 0.0
            for k in range(1, len(times) + 1):
                t = times[k - 1]
                dt = t - (times[k - 2] if k > 1 else 0.0)
                xp, yp, thp, vp, movedp = x, y, th, v, moved
                lat = (x - ax_[i]) * nx + (y - ay_[i]) * ny
                if q < ncand - 1 and not fixed:
                    h = wrap(tangent[i] + math.atan2(off - lat, aim))
                if mdir == mdir:
                    v = sp
                    step = v * dt
                    x += step * math.cos(mdir)
                    y += step * math.sin(mdir)
                else:
                    dth = wrap(h - th)
                    th = th + min(max(dth, -turn_rate * dt), turn_rate * dt)
                    if turning and abs(wrap(h - th)) <= TURN_ALIGNED_RAD:
                        turning = False  # turned: now it walks
                    dv = (0.0 if turning else sp) - v
                    v = v + min(max(dv, -decel * dt), accel * dt)
                    step = min(v * dt, max(goal_dist[i] - trav, 0.0))
                    trav += step
                    x += step * math.cos(th)
                    y += step * math.sin(th)
                if not in_cells(x, y, free, fx0, fy0, fstep):
                    if q == 0 and t <= 1.0:
                        static_block = True
                        support_block = True
                    tmin = min(tmin, t - dt)
                    break
                nsup = max(5 if k == 1 else 1, int(math.ceil(math.hypot(x - xp, y - yp) / 0.05)))
                unsupported = False
                for us in range(1, nsup + 1):
                    u = us / nsup
                    if not body_clear(
                        xp + (x - xp) * u,
                        yp + (y - yp) * u,
                        thp + wrap(th - thp) * u,
                        max(vp, v) + 0.01,
                        ext[i],
                        z_ok,
                        walk,
                        sx0,
                        sy0,
                        scell,
                        sdist,
                    ):
                        unsupported = True
                        break
                if unsupported:
                    if q == 0 and t <= 1.0:
                        static_block = True
                        support_block = True  # it stays where it is, like a standing body
                    tmin = min(tmin, t - dt)
                    break
                mg0 = margin + margin_growth * t
                moved = (x - px[i]) * nx + (y - py[i]) * ny
                hit = False
                ch, sh = math.cos(th), math.sin(th)
                for a in range(cnt):
                    j = nb[a]
                    mg = mg0
                    along = vx[j] * ch + vy[j] * sh
                    if tight[a]:
                        mg = TIGHT_GAP_M
                    elif recip[j]:
                        if along <= 0.5 * speed[j] and along >= -0.7 * speed[j]:
                            mg += personal[i]
                        elif along < -0.7 * speed[j]:
                            mg += PASS_ROOM_S * max(
                                v + speed[j] - PASS_SLOW_MPS, 0.0
                            )  # oncoming: pass closely only slowly
                        if creeping:
                            mg += creep_room
                    elif speed[j] > 0.05 or (sets_off[j] and j != seen_standing[i]):
                        mg += moving_room  # a moving vehicle, or one about to set off
                    oncoming = recip[j] and speed[j] > 0.2 and vx[j] * tx + vy[j] * ty < -0.7 * speed[j]
                    if oncoming:
                        # An oncoming walker is predicted at its present velocity: crediting the turn it chose
                        # last tick makes both relax their sidesteps in the same tick, and resume, in step.
                        t0 = times[k - 2] if k > 1 else 0.0
                        qx, qy, qth = px[j] + vx[j] * t, py[j] + vy[j] * t, yaw[j]
                        rx, ry, rth = px[j] + vx[j] * t0, py[j] + vy[j] * t0, yaw[j]
                    elif j == seen_standing[i]:
                        qx, qy, qth = px[j], py[j], yaw[j]
                        rx, ry, rth = qx, qy, qth
                    else:
                        qx, qy, qth = pred_x[j, k - 1], pred_y[j, k - 1], pred_yaw[j, k - 1]
                        if k > 1:
                            rx, ry, rth = pred_x[j, k - 2], pred_y[j, k - 2], pred_yaw[j, k - 2]
                        else:
                            rx, ry, rth = px[j], py[j], yaw[j]
                    if rel != 0.0 and oncoming:
                        # Reciprocal passing: on the right, unless the two are already clearly set to pass left to
                        # left (then both keep their sides). The side is that of the pair, the same for both: the
                        # signed miss distance of their positions moving along their path directions.
                        wx = math.cos(tangent[j]) - tx
                        wy = math.sin(tangent[j]) - ty
                        wn = math.hypot(wx, wy)
                        left_pass = wn > 1e-9 and ((px[j] - px[i]) * wy - (py[j] - py[i]) * wx) / wn < -PASS_SIDE_M
                        if (rel < 0.0) != left_pass:
                            qx -= moved * nx
                            qy -= moved * ny
                            rx -= movedp * nx
                            ry -= movedp * ny
                    close = (v + speed[j]) * dt  # what the two may close between the samples
                    reach_mg = max(mg, gap0[a])  # the distance at which a test can fail
                    if (qx - x) ** 2 + (qy - y) ** 2 > (radius[i] + radius[j] + reach_mg + close) ** 2:
                        continue
                    # test the interval between the samples, not only its end (no tunnelling at speed)
                    nsub = max(1, int(math.ceil(close / 0.15)))
                    for us in range(1, nsub + 1):
                        u = us / nsub
                        ix, iy, ith = xp + (x - xp) * u, yp + (y - yp) * u, thp + wrap(th - thp) * u
                        jx, jy, jth = rx + (qx - rx) * u, ry + (qy - ry) * u, rth + wrap(qth - rth) * u
                        if (ix - jx) ** 2 + (iy - jy) ** 2 > (radius[i] + radius[j] + reach_mg) ** 2:
                            continue
                        # the pose box of the safety pass (the step's higher speed and its allowance), so that a
                        # step judged clear here is not undone there
                        bx_, by_, ahl, ahw = box(ix, iy, ith, max(vp, v) + 0.01, ext[i])
                        cx_, cy_, bhl, bhw = box(jx, jy, jth, speed[j], ext[j])
                        if overlap(bx_, by_, ith, ahl, ahw, cx_, cy_, jth, bhl, bhw, -mg):
                            hit = True
                            break
                        if gap0[a] > -math.inf:
                            # closing in on a walking neighbour already within the margin: compared with standing
                            # still with the same box
                            sx_, sy_, shl, shw = box(px[i], py[i], yaw[i], max(vp, v) + 0.01, ext[i])
                            ref = min(gap0[a], gap(sx_, sy_, yaw[i], shl, shw, cx_, cy_, jth, bhl, bhw))
                            if gap(bx_, by_, ith, ahl, ahw, cx_, cy_, jth, bhl, bhw) < ref - TIGHT_SLACK_M:
                                hit = True
                                break
                    if hit:
                        if q == 0 and speed[j] <= 0.05 and t <= 1.0:
                            static_block = True
                        break
                if hit:
                    tmin = min(tmin, t - dt)
                    break
            if mdir == mdir:
                # Edging aside costs as much as standing plus one, and the less the more directly it opens the gap.
                # When the way ahead is shut by something that stays (a standing body, the end of the support)
                # standing is futile: edging then costs less than standing, away from the end of the support when
                # that is what shuts the way (a walker that stopped where only its rest pose fits, and can neither
                # set off nor turn on the spot).
                if static_block:
                    pen = stand_pen[i] - EDGE_RELIEF + w_toi * (1.0 / (0.1 + tmin / horizon))
                    ref = away
                    if support_block:
                        ref = support_away(px[i], py[i], sdist, sx0, sy0, scell)
                        if ref != ref:
                            ref = away
                    if ref == ref:
                        pen += 1.0 - math.cos(mdir - ref)
                else:
                    pen = stand_pen[i] + 1.0 + w_toi * (1.0 / (0.1 + tmin / horizon))
                    if away == away:
                        pen += 1.0 - math.cos(mdir - away)
            else:
                cx, cy = sp * math.cos(h0), sp * math.sin(h0)
                pen = w_des * math.hypot(cx - dvx, cy - dvy) / vmax + w_cur * math.hypot(cx - cvx, cy - cvy) / vmax
                if rel > 0.0:
                    pen += w_side * math.atan2(rel, aim) / math.pi  # keep right
                pen += w_toi * (1.0 / (0.1 + tmin / horizon))
                if q == ncand - 1:
                    pen += stand_pen[i]
                if fixed and abs(wrap(h0 - prev_aim[i])) < 0.05:
                    pen -= SIDESTEP_KEEP
            if pen < best:
                best, bh, bs, bt, bm, bturn = pen, h0, sp, tmin, mdir, turnfirst
        # a candidate turning on the spot first stands this tick while it turns
        best_h[w] = bh
        best_s[w] = 0.0 if bturn else bs
        best_t[w] = bt
        best_m[w] = bm
    return best_h, best_s, best_t, best_m


@njit(cache=True)
def predict_bodies(px, py, yaw, speed, des_yaw, steer, times, turn_rate):
    """Predicted poses (n, len(times)) of all bodies at constant speed at the sample `times`; bodies with `steer`
    turn towards the heading `des_yaw` at the turn rate (as the walker's own prediction does), others keep their
    heading."""
    n = len(px)
    m = len(times)
    qx = np.empty((n, m))
    qy = np.empty((n, m))
    qyaw = np.empty((n, m))
    for j in range(n):
        x, y, th = px[j], py[j], yaw[j]
        for k in range(m):
            dt = times[k] - (times[k - 1] if k > 0 else 0.0)
            if steer[j]:
                d = wrap(des_yaw[j] - th)
                th = th + min(max(d, -turn_rate * dt), turn_rate * dt)
            x += speed[j] * math.cos(th) * dt
            y += speed[j] * math.sin(th) * dt
            qx[j, k] = x
            qy[j, k] = y
            qyaw[j, k] = th
    return qx, qy, qyaw


@njit(cache=True)
def swept_conflicts(
    ids,
    ox,
    oy,
    oyaw,
    nx_,
    ny_,
    nyaw,
    sw,
    ext,
    radius,
    active,
    order,
    start,
    gx0,
    gy0,
    gcell,
    gnx,
    gny,
    reach,
    eps,
    samples,
):
    """Pairs (i, j), i in `ids`, whose bodies overlap at any of `samples` interpolation points of the tick.

    Boxes use the swept speed `sw` (the larger of the previous and new speed), as the builder does; the
    grid is built on the new positions and `reach` covers the largest per-tick displacement plus both radii.
    """
    out = np.empty((4096, 2), np.int64)
    k = 0
    nb = np.empty(256, np.int64)
    for a in range(len(ids)):
        i = ids[a]
        cnt = near(i, nx_, ny_, reach, order, start, gx0, gy0, gcell, gnx, gny, nb)
        for b in range(cnt):
            j = nb[b]
            if not active[j]:
                continue
            hit = False
            for s in range(samples):
                u = s / (samples - 1.0)
                ix, iy = ox[i] + (nx_[i] - ox[i]) * u, oy[i] + (ny_[i] - oy[i]) * u
                jx, jy = ox[j] + (nx_[j] - ox[j]) * u, oy[j] + (ny_[j] - oy[j]) * u
                if (ix - jx) ** 2 + (iy - jy) ** 2 > (radius[i] + radius[j] + 0.1) ** 2:
                    continue
                iyw = oyaw[i] + wrap(nyaw[i] - oyaw[i]) * u
                jyw = oyaw[j] + wrap(nyaw[j] - oyaw[j]) * u
                ax, ay, ahl, ahw = box(ix, iy, iyw, sw[i], ext[i])
                bx, by, bhl, bhw = box(jx, jy, jyw, sw[j], ext[j])
                if overlap(ax, ay, iyw, ahl, ahw, bx, by, jyw, bhl, bhw, eps):
                    hit = True
                    break
            if hit and k < len(out):
                out[k, 0] = i
                out[k, 1] = j
                k += 1
    return out[:k]


@njit(cache=True)
def pair_swept(ox, oy, oyaw, nx_, ny_, nyaw, sw, ext, i, j, eps, samples):
    """Whether the swept boxes of bodies i and j overlap at any of `samples` interpolation points of the tick."""
    for s in range(samples):
        u = s / (samples - 1.0)
        ix, iy = ox[i] + (nx_[i] - ox[i]) * u, oy[i] + (ny_[i] - oy[i]) * u
        jx, jy = ox[j] + (nx_[j] - ox[j]) * u, oy[j] + (ny_[j] - oy[j]) * u
        iyw = oyaw[i] + wrap(nyaw[i] - oyaw[i]) * u
        jyw = oyaw[j] + wrap(nyaw[j] - oyaw[j]) * u
        ax, ay, ahl, ahw = box(ix, iy, iyw, sw[i], ext[i])
        bx, by, bhl, bhw = box(jx, jy, jyw, sw[j], ext[j])
        if overlap(ax, ay, iyw, ahl, ahw, bx, by, jyw, bhl, bhw, eps):
            return True
    return False


@njit(cache=True)
def swept_supported(
    i, ox, oy, oyaw, nx_, ny_, nyaw, sw, ext, z_ok, walk, sx0, sy0, scell, free, fx0, fy0, fstep, samples
):
    for s in range(1, samples):
        u = s / (samples - 1.0)
        x, y = ox + (nx_ - ox) * u, oy + (ny_ - oy) * u
        th = oyaw + wrap(nyaw - oyaw) * u
        if not body_supported(x, y, th, sw, ext[i], z_ok, walk, sx0, sy0, scell):
            return False
    return in_cells(nx_, ny_, free, fx0, fy0, fstep)


@njit(cache=True)
def _path_at(pts, arc, f, m, s):
    total = arc[f + m - 1]
    s = min(max(s, 0.0), total)
    k = f
    while k < f + m - 2 and arc[k + 1] <= s:
        k += 1
    L = arc[k + 1] - arc[k]
    u = 0.0 if L < 1e-9 else (s - arc[k]) / L
    return pts[k, 0] + u * (pts[k + 1, 0] - pts[k, 0]), pts[k, 1] + u * (pts[k + 1, 1] - pts[k, 1])


@njit(cache=True)
def desire_all(
    ids,
    pts,
    arc,
    first,
    count,
    s_arr,
    k_arr,
    px,
    py,
    speed,
    max_speed,
    yaw,
    dt,
    creep,
    lane_m,
    turn_rate,
    heading,
    want,
    cap,
    anchor_x,
    anchor_y,
    tangent,
    lane,
):
    """awr.life.motion.crowd.Path.desire for the walkers `ids` on the flat path store (same arithmetic).

    Path i occupies rows first[i]..first[i]+count[i]-1 of `pts` with cumulative arc lengths `arc` (0 at the
    first row); s_arr/k_arr hold the progress (arc length, segment index relative to first). Writes the
    outputs at index i; heading is NaN where Path.desire returns None."""
    for a in range(len(ids)):
        i = ids[a]
        f = first[i]
        m = count[i]
        x, y = px[i], py[i]
        # update: project onto the next segments, never backwards along the path
        best = 1e300
        bk = k_arr[i]
        bs = s_arr[i]
        for k in range(k_arr[i], min(k_arr[i] + 8, m - 1)):
            ax, ay = pts[f + k, 0], pts[f + k, 1]
            sx, sy = pts[f + k + 1, 0] - ax, pts[f + k + 1, 1] - ay
            L = arc[f + k + 1] - arc[f + k]
            u = 0.0 if L < 1e-9 else min(max(((x - ax) * sx + (y - ay) * sy) / (L * L), 0.0), 1.0)
            d = math.hypot(ax + u * sx - x, ay + u * sy - y)
            if d < best - 1e-9:
                best = d
                bk = k
                bs = arc[f + k] + u * L
        if bs >= s_arr[i]:
            k_arr[i] = bk
            s_arr[i] = bs
        s = s_arr[i]
        gx, gy = pts[f + m - 1, 0], pts[f + m - 1, 1]
        to_goal = math.hypot(gx - x, gy - y)
        total = arc[f + m - 1]
        remaining = max(total - s, to_goal)
        look = min(max(0.6 + 0.6 * speed[i], 0.6), 1.5)
        qx, qy = _path_at(pts, arc, f, m, s)
        ex, ey = _path_at(pts, arc, f, m, s + max(look, 1.0))
        ah = math.hypot(ex - qx, ey - qy)
        ln, nx, ny = 0.0, 0.0, 0.0
        if ah >= 1e-6:
            nx, ny = -(ey - qy) / ah, (ex - qx) / ah
            band = min(lane_m, max(0.0, 0.5 * (remaining - 1.0)))
            ln = min(max((x - qx) * nx + (y - qy) * ny, -band), band)
        if remaining <= look:
            tx, ty = gx, gy
        else:
            tx, ty = _path_at(pts, arc, f, m, s + look)
            tx += ln * nx
            ty += ln * ny
        h = math.nan
        if math.hypot(tx - x, ty - y) > 1e-6:
            h = math.atan2(ty - y, tx - x)
        w = min(max_speed[i], math.sqrt(2.0 * 1.2 * max(remaining - 0.02, 0.0)))
        # corners within the stopping reach: at most 1.4 rad/s of heading change through a bend
        j = 0
        while j < m and arc[f + j] <= s:
            j += 1
        while 0 < j < m - 1 and arc[f + j] - s <= 3.0:
            ax_, ay_ = pts[f + j, 0] - pts[f + j - 1, 0], pts[f + j, 1] - pts[f + j - 1, 1]
            bx_, by_ = pts[f + j + 1, 0] - pts[f + j, 0], pts[f + j + 1, 1] - pts[f + j, 1]
            turn = abs(math.atan2(ax_ * by_ - ay_ * bx_, ax_ * bx_ + ay_ * by_))
            if turn > 0.05:
                l0 = arc[f + j] - arc[f + j - 1]
                l1 = arc[f + j + 1] - arc[f + j]
                bend = max(min(l0, l1, 1.0) / turn * 1.4, creep * 2)
                w = min(w, math.sqrt(bend**2 + 2.0 * 1.2 * max(arc[f + j] - s, 0.0)))
            j += 1
        if to_goal > 1e-6:
            # the circle through the goal tangent to the present heading must be within the turn rate
            err = abs(wrap(math.atan2(gy - y, gx - x) - yaw[i]))
            if err > 1e-3:
                w = min(w, 0.8 * turn_rate * to_goal / (2.0 * math.sin(min(err, math.pi / 2))))
        heading[i] = h
        want[i] = w
        cap[i] = to_goal / dt
        lane[i] = ln
        anchor_x[i] = qx
        anchor_y[i] = qy
        if ah < 1e-6 or math.isnan(h):
            tangent[i] = h
        else:
            tangent[i] = math.atan2(ey - qy, ex - qx)
