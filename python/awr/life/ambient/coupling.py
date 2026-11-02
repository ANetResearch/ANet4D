"""Cars and people on shared ground: people wait before the path of a moving car, cars stop for people on theirs.

Few streets of the town have a sidewalk, so people walk at the road edges and cross where their route goes. Each tick
every car's corridor is its body and the path ahead of its front (awr.life.ambient.traffic.Traffic.ahead); a person near
it is measured against the corridor: the arc `a` of the person's body along the path (from the car's front; negative
beside the body) and the lateral gap between the person's body (awr.life.civil.body extents, largest asset) and the car's
side.

- A person waits (desired speed zero, awr.life.ambient.crowd) when the next HOLD_LOOK_S of walking would bring the body
  closer than HOLD_GAP_M to the part of a car's corridor where the car could not stop: its body and, for a moving car,
  the path ahead within its stopping distance (0.5 m for a standing car). Both sideways approach (level with that
  stretch) and head-on approach to its front end count. Moving away, or along a car's side, is always allowed.
- A car stops STOP_GAP_M before a person whose body comes within YIELD_GAP_M of the corridor, now or after HOLD_LOOK_S
  of its intended walking unless the person waits for this car: its speed is limited to what brakes at
  YIELD_DECEL_MPS2 by then (traci setSpeed). So of a car and a person about to meet exactly one gives way: the person
  where the car could not stop, else the car. A person waiting for one car of a queue is let across by the next car
  that can stop, so a slow queue does not keep it waiting until the queue has passed.
- Whatever the margins, nobody comes within TOUCH_M of a car's body (and braking distance), people counted as discs
  of a walking body's reach about their positions (WALK_REACH_M): turning on the spot sweeps no closer than that.
  JuPedSim moves a walker along its own heading, which lags the target's direction, so a walker released after a
  wait may first step where it did not intend: the next HOLD_LOOK_S along that heading counts as well.
- A person kept waiting by cars for SQUEEZE_S passes closer for a while, as people do in slow traffic: it keeps
  TOUCH_M from the bodies and clear of the braking distances only (cars still stop for it in their corridors).
- A person beside the front half of a car's body (closer than 0.05 m) stops the car unless the car's body, moved
  SWEEP_M along its way, keeps SWEEP_GAP_M from it; then the car creeps past at CREEP_MPS, and the person, within
  PASS_HOLD_M of a moving car's body, stands still meanwhile. A person walking between the two lanes of a narrow
  street may otherwise end up between two cars that stopped beside it, unable to step towards either, and neither
  car would drive on.
- A car stopped inside a junction for a person, or queued there behind a car that is, is in the way of the other
  cars passing it: SUMO lets links whose lanes converge or pass close by run at once, so these stop before its body
  (cars standing in the junction only for those that stood there before them). Generally the first car in a junction
  goes first: the cars that entered after it, or are about to enter, stop before its body where it lies on their way
  (unless they go its way, following it).
- A person held by a standing car is a blocked walker (awr.life.ambient.crowd plans around the standing cars, `cars`,
  and off their ways ahead where there is room, `waiting`);
  a person in the way of a waiting car stands aside (`blocking`) until the car has passed, and is not held by that car
  meanwhile; not for a car held at the entry of a single-lane section until its turn (awr.life.ambient.traffic.Sections).

HOLD_GAP_M is wider than YIELD_GAP_M, so a person never steps where a car would have to brake harder than it can.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# walking extents of the largest person asset ([x_back, x_front, y_right, y_left], awr.life.civil.body with margin)
PERSON_WALK = np.array([-0.423, 0.576, -0.369, 0.369])
PERSON_REST = np.array([-0.145, 0.249, -0.332, 0.33])
WALK_REACH_M = float(
    np.hypot((PERSON_WALK[1] - PERSON_WALK[0]) / 2, PERSON_WALK[3]) + (PERSON_WALK[0] + PERSON_WALK[1]) / 2
)
YIELD_GAP_M = 0.3
HOLD_GAP_M = 0.6
TOUCH_M = 0.15
STOP_GAP_M = 1.0
YIELD_DECEL_MPS2 = 2.0
REACT_S = 0.5
HOLD_LOOK_S = 1.0
SOON_S = 0.3
SQUEEZE_S = 3.0  # a person kept waiting by cars this long accepts closer passes for SQUEEZE_FOR_S:
SQUEEZE_FOR_S = 10.0  # clear of the bodies by TOUCH_M and of the braking distances, without the comfort margins
HELD_DECAY = 2.0  # the waiting time counts down this many times faster while the person is free to walk
MOVING_MPS = 0.3
LOOK_M = 10.0  # corridor ahead beyond the stopping distance
STRAIGHT_M = 5.0  # a standing car's axis ahead of its front, for the people keeping clear of it
QUEUE_GAP_M = 3.0  # a standing car this close behind a standing car queues behind it
SAME_WAY_DEG = 30.0  # a car heading within this of another's follows it (SUMO), it does not yield to its body
SWEEP_M = 3.0  # a car with somebody beside the front half of its body drives on only if its body, moved this
SWEEP_GAP_M = 0.02  # far along its way, keeps this from that person's body, and passes it no faster than
CREEP_MPS = 1.5  # this; people this close to a moving car's body stand still (a body that starts walking
PASS_HOLD_M = 0.1  # widens by 0.04 m, awr.life.civil.body)


def person_box(xy, yaw, speed):
    """Centre (n, 2), heading (n,) and half sizes (n, 2) of the bodies at their speeds."""
    b = np.where(speed > 0.03, np.minimum(speed / 0.5, 1.0), 0.0)[:, None]
    e = PERSON_REST * (1 - b) + PERSON_WALK * b
    h = np.c_[np.cos(yaw), np.sin(yaw)]
    centre = xy + h * ((e[:, 0] + e[:, 1]) / 2)[:, None] + np.c_[-h[:, 1], h[:, 0]] * ((e[:, 2] + e[:, 3]) / 2)[:, None]
    return centre, yaw, np.c_[(e[:, 1] - e[:, 0]) / 2, (e[:, 3] - e[:, 2]) / 2]


def project(points, line, arcs):
    """Arc along `line` (with arcs `arcs` at its points), signed lateral offset (left +) and unit tangent of each point."""
    a, b = line[:-1], line[1:]
    ab = b - a
    L2 = np.maximum((ab**2).sum(1), 1e-12)
    f = np.clip(((points[:, None, :] - a[None]) * ab[None]).sum(-1) / L2[None], 0.0, 1.0)
    foot = a[None] + f[..., None] * ab[None]
    d = np.linalg.norm(points[:, None, :] - foot, axis=-1)
    k = d.argmin(1)
    r = np.arange(len(points))
    t = ab[k] / np.sqrt(L2[k])[:, None]
    rel = points - foot[r, k]
    lateral = t[:, 0] * rel[:, 1] - t[:, 1] * rel[:, 0]
    along = (rel * t).sum(1)  # beyond the ends of the corridor
    arc = (
        arcs[k]
        + f[r, k] * np.sqrt(L2[k])
        + np.where((k == 0) & (f[r, k] == 0.0) | (k == len(ab) - 1) & (f[r, k] == 1.0), along, 0.0)
    )
    return arc, lateral, t


def measure(centre, pos, yaw, half, line, arcs, half_w, radius=None):
    """(arc, half extent along the path, lateral gap to a car of half width `half_w`) of person boxes (centres, the
    people's positions, headings, half sizes) against a corridor, or of discs of `radius` about the positions."""
    if radius is not None:
        a, lat, _ = project(pos, line, arcs)
        return a, np.full(len(pos), radius), np.abs(lat) - radius - half_w
    a, lat, t = project(centre, line, arcs)
    et, en = extents(t, yaw, half)
    return a, et, np.abs(lat) - en - half_w


def approach(now, after, to, back, margin, length):
    """Whether person boxes moving from `now` to `after` ((arc, half extent along the path, lateral gap) each) come
    closer than `margin` to the stretch of a corridor from arc `back` to arc `to`: sideways while level with the
    stretch, or at its front end (head-on, ahead of the car's front at arc 0) or back end (catching up with a slow
    car, behind its back at arc -length) while across from it. Moving away, or along the side, is no approach."""
    (a, et, gap), (a2, et2, gap2) = now, after
    f, f2 = a - et - to, a2 - et2 - to
    r, r2 = back - (a + et), back - (a2 + et2)
    side = (gap2 < np.minimum(gap, margin) - 0.01) & (np.maximum(f2, r2) < margin)
    head = (f2 < np.minimum(f, margin) - 0.01) & (a - et > -0.2) & (gap2 < margin)
    tail = (r2 < np.minimum(r, margin) - 0.01) & (a + et < 0.2 - length) & (gap2 < margin)
    return side | head | tail


def reach(m, to, back):
    """Distance of discs ((arc, radius, lateral gap), measure with a radius) to the stretch of a corridor from arc
    `back` to arc `to` (the rectangle in arc and lateral offset, corners included); negative where they overlap it."""
    a, r, gap = m
    return np.hypot(np.maximum(np.maximum(back - a, a - to), 0.0), np.maximum(gap + r, 0.0)) - r


def closing(now, after, to, back, margin):
    """Whether discs moving from `now` to `after` ((arc, radius, lateral gap) each, measure with a radius) come
    closer than `margin` to the stretch of a corridor from arc `back` to arc `to`: their distance to it shrinks to
    below `margin`. Moving away, or along it, is no approach."""
    d, d2 = reach(now, to, back), reach(after, to, back)
    return d2 < np.minimum(d, margin) - 0.01


def box_gap(ca, ya, ha, cb, yb, hb):
    """Largest separation of pairs of oriented boxes (centres (n, 2), headings (n,), half sizes (n, 2)) along their
    own edge directions: negative where they overlap."""
    ua, ub = np.c_[np.cos(ya), np.sin(ya)], np.c_[np.cos(yb), np.sin(yb)]
    va, vb = np.c_[-ua[:, 1], ua[:, 0]], np.c_[-ub[:, 1], ub[:, 0]]
    d = cb - ca
    out = np.full(len(ca), -np.inf)
    for ax in (ua, va, ub, vb):
        ra = ha[:, 0] * np.abs((ua * ax).sum(1)) + ha[:, 1] * np.abs((va * ax).sum(1))
        rb = hb[:, 0] * np.abs((ub * ax).sum(1)) + hb[:, 1] * np.abs((vb * ax).sum(1))
        out = np.maximum(out, np.abs((d * ax).sum(1)) - ra - rb)
    return out


def sweep_gap(line, larc, length, half_w, centre, yaw, half, reach):
    """Least gap between person boxes (centres, headings, half sizes) and a car's body moved up to `reach` along its
    way (`line` with arcs `larc` from its front; the body spans the way from arc s-length to s), per person."""
    s = np.arange(0.25, reach + 1e-9, 0.25)
    fx, fy = np.interp(s, larc, line[:, 0]), np.interp(s, larc, line[:, 1])
    rx, ry = np.interp(s - length, larc, line[:, 0]), np.interp(s - length, larc, line[:, 1])
    cc, cyaw = np.c_[(fx + rx) / 2, (fy + ry) / 2], np.arctan2(fy - ry, fx - rx)
    n, k = len(centre), len(s)
    g = box_gap(
        np.repeat(cc[None], n, 0).reshape(-1, 2),
        np.tile(cyaw, n),
        np.full((n * k, 2), (length / 2, half_w)),
        np.repeat(centre, k, 0),
        np.repeat(yaw, k),
        np.repeat(half, k, 0),
    )
    return g.reshape(n, k).min(1)


def oncoming(now, after, to, length, margin):
    """Whether discs ((arc, radius, lateral gap) each, measure with a radius) within `margin` of the stretch of a
    corridor from the body's back to arc `to` come closer to the car's body. `closing` measures the distance to the
    whole stretch, which stays the same for a disc already within it walking on towards the car."""
    return (reach(now, to, -length) < margin) & (reach(after, 0.0, -length) < reach(now, 0.0, -length) - 0.01)


def car_corners(centre, yaw, hl, hw):
    """The four corners and the front and back middles of a car body, (6, 2)."""
    h, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    return centre + np.array([[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw], [hl, 0.0], [-hl, 0.0]]) @ np.array([h, n])


def extents(t, yaw, half):
    """Half extents of person boxes along the tangent `t` and across it."""
    c, s = np.cos(yaw), np.sin(yaw)
    ht = np.abs(t[:, 0] * c + t[:, 1] * s)  # |cos| between body x axis and tangent
    hn = np.abs(-t[:, 1] * c + t[:, 0] * s)
    return half[:, 0] * ht + half[:, 1] * hn, half[:, 0] * hn + half[:, 1] * ht


class Coupling:
    def __init__(self, traffic, crowd):
        self.traffic, self.crowd = traffic, crowd
        self.limited = {}
        self.for_people = {}  # car id -> its speed limit for the people in its way alone
        self.blocker_since = {}  # car id -> time it became a junction blocker
        self.entered = {}  # car id -> time its front entered the junction it is in
        self.holds = 0
        self.standing_holds = 0
        self.yields = 0

    def update(self):
        tr, cr = self.traffic, self.crowd
        cxy, cyaw, cv = tr.poses()
        pxy, pyaw, pv = cr.poses()
        intent = np.array([p.intent if p.intent is not None else (0.0, 0.0) for p in cr.people])
        box_c, box_yaw, half = person_box(pxy, pyaw, pv)
        later = pxy + intent * HOLD_LOOK_S
        lyaw = np.where(np.hypot(*intent.T) > 0.1, np.arctan2(intent[:, 1], intent[:, 0]), pyaw)
        lbox_c, _, lhalf = person_box(later, lyaw, np.maximum(np.hypot(*intent.T), pv))
        # ... and after HOLD_LOOK_S of walking on as in the last tick: JuPedSim knows no cars, and its avoidance of
        # other people can drift a walker sideways (held only if that would bring its body within TOUCH_M of a car)
        vel = np.array([p.last for p in cr.people]) / cr.step_s
        dyaw = np.where(np.hypot(*vel.T) > 0.1, np.arctan2(vel[:, 1], vel[:, 0]), pyaw)
        drifted = pxy + vel * HOLD_LOOK_S
        dbox_c, _, dhalf = person_box(drifted, dyaw, pv)
        # ... and after HOLD_LOOK_S along JuPedSim's heading at the intended speed: JuPedSim turns a walker towards its
        # target gradually, so one released after a wait first steps along its old heading (a walker that waited
        # beside a car, then is sent along it, would edge towards it a step at a time)
        heading = np.array([p.heading for p in cr.people])
        hyaw = np.arctan2(heading[:, 1], heading[:, 0])
        headed = pxy + heading * np.hypot(*intent.T)[:, None] * HOLD_LOOK_S
        hbox_c, _, hhalf = person_box(headed, hyaw, np.hypot(*intent.T))
        # ... and after SOON_S of the intended walking (a body turns early on, near where it stands; at the speed it can
        # reach by then, walkers accelerating at 1 m/s², awr.life.ambient.crowd)
        soon = pxy + intent * SOON_S
        sbox_c, _, shalf = person_box(soon, lyaw, np.minimum(np.hypot(*intent.T), pv + 1.0 * SOON_S))
        tree = cKDTree(pxy)
        hold = np.zeros(len(pxy), bool)
        hold_standing = np.zeros(len(pxy), bool)
        seen = []
        corridors = []  # every car's body and path ahead, for the crowd's choice of where to stand aside
        squeeze = np.array([cr.t < p.squeeze_until for p in cr.people])
        paths = []  # every car's corridor line and its arcs
        waiting = {}  # standing cars' ways ahead of their fronts, for the crowd's routes
        # people: no step towards a corridor where its car could not stop
        for i, vid in enumerate(tr.ids):
            v = float(cv[i])
            length = tr.length[tr.types[i]]
            half_w = tr.width[tr.types[i]] / 2
            stop = v * v / (2 * YIELD_DECEL_MPS2) + v * REACT_S + STOP_GAP_M
            ahead, arcs = tr.ahead(vid, stop + LOOK_M, 0.5)
            h = np.array([np.cos(cyaw[i]), np.sin(cyaw[i])])
            front = cxy[i] + h * length / 2
            back = front[None] - h[None] * np.arange(length + 0.5, 0.0, -0.5)[:, None]
            line = np.r_[back, front[None], ahead[1:] if len(ahead) and np.hypot(*(ahead[0] - front)) < 0.3 else ahead]
            larc = np.r_[
                -np.arange(length + 0.5, 0.0, -0.5),
                0.0,
                arcs[1:] if len(ahead) and np.hypot(*(ahead[0] - front)) < 0.3 else arcs,
            ]
            corridors.append((vid, line, half_w))
            paths.append((line, larc))
            if v <= MOVING_MPS:
                waiting[vid] = (line[larc >= 0.0], half_w)
            near = tree.query_ball_point(front, stop + LOOK_M + 3.0)
            if not near:
                seen.append(None)
                continue
            near = np.array(near)
            moving = v > MOVING_MPS

            def measures(line, larc):
                # now, after HOLD_LOOK_S and SOON_S of the intended walking; and as discs of a walking body's reach,
                # the same and after HOLD_LOOK_S of drift, for the TOUCH_M margins (a body turning or starting to walk
                # sweeps no closer than that)
                def m(c, p, y, h, r=None):
                    return measure(c[near], p[near], y[near], h[near], line, larc, half_w, r)

                return (
                    m(box_c, pxy, box_yaw, half),
                    m(lbox_c, later, lyaw, lhalf),
                    m(sbox_c, soon, lyaw, shalf),
                    m(box_c, pxy, box_yaw, half, WALK_REACH_M),
                    m(lbox_c, later, lyaw, lhalf, WALK_REACH_M),
                    m(sbox_c, soon, lyaw, shalf, WALK_REACH_M),
                    m(dbox_c, drifted, dyaw, dhalf, WALK_REACH_M),
                    m(hbox_c, headed, hyaw, hhalf, WALK_REACH_M),
                )

            now, after, soon_, now_t, after_t, soon_t, drift_t, head_t = measures(line, larc)
            a, et, gap = now
            front_gap = a - et
            gap2, front2 = after[2], after[0] - after[1]
            if not moving:
                # what a person keeps clear of at a standing car is its body and the half metre before it, measured
                # along the car's axis: the way ahead of a car standing at a junction may turn at once, and would bend
                # the measure about its front (a person stepping off sideways in front of it would seem to close in)
                ext = np.arange(0.5, STRAIGHT_M + 0.01, 0.5)
                now, after, soon_, now_t, after_t, soon_t, drift_t, head_t = measures(
                    np.r_[back, front[None], front[None] + h[None] * ext[:, None]], np.r_[larc[: len(back) + 1], ext]
                )
            danger_to = stop if moving else 0.5

            def towards(to, back, margin):
                return approach(now, after, to, back, margin, length) | approach(now, soon_, to, back, margin, length)

            def touching(to):
                return (
                    closing(now_t, after_t, to, -length, TOUCH_M)
                    | closing(now_t, soon_t, to, -length, TOUCH_M)
                    | oncoming(now_t, after_t, to, length, TOUCH_M)
                    | oncoming(now_t, soon_t, to, length, TOUCH_M)
                )

            # nobody comes within TOUCH_M of a car's body and braking distance; with the comfort margins, unless kept
            # waiting long (squeeze)
            brake = v * v / (2 * YIELD_DECEL_MPS2) + v * REACT_S if moving else 0.0
            held = ~squeeze[near] & towards(danger_to, -length - 0.5, HOLD_GAP_M) | touching(brake)
            # a person standing aside keeps clear only of the bodies of standing cars, and none of the car it stands
            # aside for in front of it (that car waits for it) while clear of its front
            giving = np.array([cr.people[j].give is not None for j in near])
            aside = giving & np.array(
                [cr.people[j].blocking is not None and cr.people[j].blocking[0] == vid for j in near]
            )
            if not moving:
                held &= ~giving | touching(0.0)
            held &= ~(aside & (after_t[0] - after_t[1] > 0.0) & (soon_t[0] - soon_t[1] > 0.0))
            held |= closing(now_t, drift_t, danger_to if moving else 0.0, -length - 0.5 if moving else -length, TOUCH_M)
            if moving:  # beside a moving car's body: stand still until it has passed
                held |= (
                    box_gap(
                        np.repeat(cxy[i][None], len(near), 0),
                        np.full(len(near), cyaw[i]),
                        np.full((len(near), 2), (length / 2, half_w)),
                        box_c[near],
                        box_yaw[near],
                        half[near],
                    )
                    < PASS_HOLD_M
                )
            held |= closing(now_t, head_t, danger_to if moving else 0.0, -length - 0.5 if moving else -length, TOUCH_M)
            hold[near[held]] = True
            if v <= MOVING_MPS:
                hold_standing[near[held]] = True
            seen.append((near, gap, front_gap, gap2, front2, line, half_w, v, length, held, larc))
        # cars: stop before anybody in the corridor ahead, or about to step into it without waiting for this car
        # (a person waiting for another car, say to cross behind it, is let across by the cars that can stop)
        blocking = [None] * len(pxy)  # (car id, path, half width) of a waiting car the person stands in front of
        # cars stopped inside a junction for somebody in their way, directly or queued behind a car that is (by this
        # coupling, not by SUMO's right of way; for people, not for the bodies of other cars below: a car that stops
        # for the body of one that entered before it would in turn stop that one, and both would wait for good), and
        # the other cars about to pass it: SUMO lets links whose lanes
        # converge or pass close by run at once, and a car queued on a long junction lane may stand where another
        # link passes, so the others stop before its body. A car that stands in the junction itself yields only to
        # the ones that stood there before it (no two wait for each other).
        n = len(tr.ids)
        index = {vid: k for k, vid in enumerate(tr.ids)}
        inside = np.array([tr.traci.vehicle.getRoadID(vid).startswith(":") for vid in tr.ids], bool)
        cause = {}

        def caused(k, depth=0):
            vid = tr.ids[k]
            if vid not in cause:
                cause[vid] = self.for_people.get(vid, 9.0) < 0.5
                if not cause[vid] and depth < 12:
                    lead = tr.traci.vehicle.getLeader(vid, QUEUE_GAP_M)
                    if lead and lead[0] in index and cv[index[lead[0]]] <= MOVING_MPS:
                        cause[vid] = caused(index[lead[0]], depth + 1)
            return cause[vid]

        blockers = [k for k in range(n) if inside[k] and cv[k] <= MOVING_MPS and caused(k)]
        self.blocker_since = {tr.ids[k]: self.blocker_since.get(tr.ids[k], cr.t) for k in blockers}

        def rank(k):
            return (self.blocker_since.get(tr.ids[k], np.inf), k)

        # first in the junction goes first: a car in a junction is in the way of the cars that entered it after it
        # (or are about to), whatever its speed. SUMO lets a car that stopped in a junction (for a person) drive on
        # without looking at the foe links again, and treats links whose lanes pass close by as clear once the foe's
        # narrower model body has passed. A car going the same way as the one ahead follows it (SUMO).
        self.entered = {tr.ids[k]: self.entered.get(tr.ids[k], cr.t) for k in range(n) if inside[k]}

        def first(k):
            return (self.entered.get(tr.ids[k], np.inf), k)

        ahead_of = [k for k in range(n) if inside[k] or k in blockers]
        corners = {
            k: car_corners(cxy[k], cyaw[k], tr.length[tr.types[k]] / 2, tr.width[tr.types[k]] / 2) for k in ahead_of
        }
        for i, (vid, row) in enumerate(zip(tr.ids, seen, strict=False)):
            limit = for_people = None
            for k in ahead_of:
                if k == i:
                    continue
                if not (
                    (k in blockers and rank(k) < rank(i))
                    or (
                        inside[k]
                        and first(k) < first(i)
                        and np.cos(cyaw[k] - cyaw[i]) < np.cos(np.radians(SAME_WAY_DEG))
                    )
                ):
                    continue
                line, larc = paths[i]
                half_w = tr.width[tr.types[i]] / 2
                a, lat, _ = project(corners[k], line, larc)
                on = (np.abs(lat) - half_w < YIELD_GAP_M) & (a > 0.0) & (a < larc[-1])
                if on.any() and k not in blockers:
                    lead = tr.traci.vehicle.getLeader(vid, larc[-1])
                    on &= not (lead and lead[0] == tr.ids[k])  # its leader on its own lane: SUMO follows it
                if on.any():
                    lim = float(np.sqrt(2 * YIELD_DECEL_MPS2 * max(a[on].min() - STOP_GAP_M, 0.0)))
                    limit = lim if limit is None else min(limit, lim)
            if row is not None:
                near, gap, front_gap, gap2, front2, line, half_w, v, length, own, larc = row
                end = larc[-1]
                # in the corridor ahead (not beyond its end, where the arc runs on along the last tangent), or touching
                # the front half of the body where the body driving on would sweep it
                ahead_in = (gap < YIELD_GAP_M) & (front_gap > -0.2) & (front_gap < end)
                beside = (gap < 0.05) & (front_gap > -length / 2) & ~ahead_in
                if beside.any():
                    k = near[beside]
                    clear = sweep_gap(line, larc, length, half_w, box_c[k], box_yaw[k], half[k], SWEEP_M) >= SWEEP_GAP_M
                    if clear.any():
                        limit = CREEP_MPS if limit is None else min(limit, CREEP_MPS)
                    beside[np.flatnonzero(beside)[clear]] = False
                now_in = ahead_in | beside
                # a waiting car: the people in its way stand aside. One held at the entry of a single-lane section until
                # its turn lets them pass in front of it: a person standing aside from it would walk back into its way
                # as soon as it is clear, again and again, and keep the car waiting once its turn has come
                if v <= MOVING_MPS and vid not in tr.sections.held:
                    for j in near[now_in]:
                        blocking[j] = (vid, line, half_w)
                soon_in = (gap2 < YIELD_GAP_M) & (front2 > -0.2) & (front2 < end) & ~own
                if (now_in | soon_in).any():
                    dist = np.minimum(
                        np.where(now_in, np.maximum(front_gap, 0.0), np.inf), np.where(soon_in, front2, np.inf)
                    ).min()
                    lim = float(np.sqrt(2 * YIELD_DECEL_MPS2 * max(dist - STOP_GAP_M, 0.0)))
                    limit = lim if limit is None else min(limit, lim)
                    for_people = lim if for_people is None else min(for_people, lim)
            self._limit(vid, limit)
            if for_people is None:
                self.for_people.pop(vid, None)
            else:
                self.for_people[vid] = for_people
        for p, hd, hs, bl in zip(cr.people, hold, hold_standing, blocking, strict=False):
            same = bl is not None and p.blocking is not None and bl[0] == p.blocking[0]
            p.blocking_since = p.blocking_since if same else (cr.t if bl is not None else None)
            p.hold, p.hold_standing, p.blocking = bool(hd), bool(hs), bl
            # waiting time that a moment's release does not reset: a person boxed in between standing cars is let
            # go for a tick now and then, steps a few centimetres and is held again
            p.held_s = p.held_s + cr.step_s if hd else max(p.held_s - HELD_DECAY * cr.step_s, 0.0)
            if p.held_s >= SQUEEZE_S:
                p.squeeze_until, p.held_s = cr.t + SQUEEZE_FOR_S, 0.0
        cr.cars = [
            (vid, cxy[i], float(cyaw[i]), tr.length[tr.types[i]] / 2, tr.width[tr.types[i]] / 2)
            for i, vid in enumerate(tr.ids)
            if cv[i] <= MOVING_MPS
        ]
        cr.corridors, cr.waiting = corridors, waiting
        cr.bodies = [
            (vid, cxy[i], float(cyaw[i]), tr.length[tr.types[i]] / 2, tr.width[tr.types[i]] / 2)
            for i, vid in enumerate(tr.ids)
        ]
        self.holds += int(hold.sum())
        self.standing_holds += int(hold_standing.sum())

    def _limit(self, vid, v):
        traci = self.traffic.traci
        old = self.limited.get(vid)
        if v is None:
            if old is not None:
                traci.vehicle.setSpeed(vid, -1)
                del self.limited[vid]
            return
        if old is None or v < old - 1e-6 or v > old + 0.05:  # a tighter limit at once (a stop is a stop)
            traci.vehicle.setSpeed(vid, v)
            self.limited[vid] = v
        self.yields += 1
