"""Courtesy of the life runtime: a person at a stay steps aside for a walker held up by it.

A travelling person who has stood still for ASK_AFTER_S asks the people at a stay (waiting to depart, queueing, in
service, deciding) whose body lies within reach of its next AHEAD_M of path to step aside. Such a person pauses its
stay (Agenda.pause: the stay is frozen, neither shortened nor faked; event `stepped_aside`) and walks in a straight
line to the nearest refuge: a spot REFUGE_M away at most, on the walking area, where its rest body is supported and
can turn round, clear of every other body (the waiting walker's too, also along the way there: its rest disc
BODY_GAP_M from their pose boxes, so that a spot beside a long car body is not refused for the car's bounding
circle) and at least the two bodies' half widths plus CLEAR_GAP_M from the walker's path; if it has not got there after OUT_TIMEOUT_S, it walks
back to its spot instead. It waits there until the walker has passed (RELEASE_M beyond the spot and walking on, or no longer
travelling) or for TIMEOUT_S, walks back to its spot once that is free, and resumes the stay (`returned_to_place`).
Nobody is moved other than by walking; a person is asked at most once per COOLDOWN_S, and a walker asks at most
every ASK_EVERY_S.

A car on a trip that has stood for ASK_AFTER_S asks the same way: the people (at a stay or walking) whose body lies
within reach of its route ahead (the car's half width plus the person's radius and REACH_GAP_M from the centre line,
from the car up to CAR_AHEAD_M beyond its front), and the person whose body sets the car's stop (its live
lookahead, awr.life.motion.vehicles.stop_blocker), step aside to a refuge clear of that route. A car follows its
measured route and cannot walk round a person; a walker held up beside the car cannot pass it either. The people
wait until the car has passed (RELEASE_M beyond the spot plus the car's half length, moving on) or is no longer on
its trip; a person at a stay then walks back to its spot and resumes, a walker on its way resumes where it stands
(also when it has not reached its refuge in OUT_TIMEOUT_S, or after TIMEOUT_S) and plans its way on from there:
walking back to the spot it was asked from would take it into the car's way again (cq3p v6, person-035 returned in
front of vehicle-049 and held it 60 s).
"""

from __future__ import annotations

import math

import numpy as np

from . import kernels as K

ASK_AFTER_S = 4.0
ASK_EVERY_S = 10.0
AHEAD_M = 6.0
REACH_GAP_M = 0.2
REFUGE_M = (0.8, 1.2, 1.6, 2.0, 2.5, 3.0)
CLEAR_GAP_M = 0.4
BODY_GAP_M = 0.15
RELEASE_M = 2.0
TIMEOUT_S = 30.0
OUT_TIMEOUT_S = 8.0
COOLDOWN_S = 20.0
ARRIVE_M = 0.05
CAR_AHEAD_M = 6.0


class Courtesy:
    def __init__(self, crowd):
        self.crowd = crowd
        self.free, self.fx0, self.fy0, self.fstep = crowd.free, crowd.fx0, crowd.fy0, crowd.fstep
        self.radius = np.asarray(crowd.radius, float)
        self.ext = crowd.ext
        self.rest_radius = np.hypot(np.abs(crowd.ext[:, 0, :2]).max(1), np.abs(crowd.ext[:, 0, 2:]).max(1))
        self.pose = None  # (yaw, speed) of every body this tick (update)
        self.half_width = np.abs(crowd.ext[:, 1, 2:]).max(1)
        self.half_length = np.abs(crowd.ext[:, 1, :2]).max(1)
        self.aside = {}  # person -> dict(requester, home, target, stage 'out' | 'wait' | 'back', since)
        self.cooldown = {}
        self.asked = {}
        self.stats = dict(
            asked=0,
            stepped_aside=0,
            no_refuge=0,
            returned=0,
            timeouts=0,
            out_timeouts=0,
            car_asked=0,
            resumed_in_place=0,
        )
        self.refusals = {}  # person -> counts of the conditions the spots of its last refuge search failed (diagnostic)
        self.resumed = []  # walkers that resumed where they stand this tick (the caller plans their way on)

    @property
    def walking(self):
        """People walking to a refuge or back to their spot (the caller walks them along the returned paths)."""
        return sorted(j for j, a in self.aside.items() if a["stage"] in ("out", "back"))

    def _walkable(self, a, b):
        n = max(int(math.ceil(float(np.linalg.norm(b - a)) / 0.1)), 1)
        return all(
            K.in_cells(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, self.free, self.fx0, self.fy0, self.fstep)
            for u in np.linspace(0.0, 1.0, n + 1)
        )

    def _fits(self, i, x, y, heading):
        c = self.crowd
        return K.in_cells(x, y, self.free, self.fx0, self.fy0, self.fstep) and K.body_supported(
            x, y, heading, 0.0, c.ext[i], c.z_ok, c.walk, c.sx0, c.sy0, c.scell
        )

    def _turns(self, i, x, y):
        return all(self._fits(i, x, y, a) for a in np.linspace(0.0, 2 * math.pi, 16, endpoint=False))

    def _clear(self, j, spot, xy, active, skip):
        """Rest body of j at `spot` (a disc: it may turn there) BODY_GAP_M clear of the pose boxes of the other active
        bodies (their bounding circles where the pose is not known)."""
        others = np.flatnonzero(active)
        others = others[(others != j) & ~np.isin(others, skip)]
        if not len(others):
            return True
        need = self.rest_radius[j] + BODY_GAP_M
        others = others[np.linalg.norm(xy[others] - spot, axis=1) < self.radius[others] + need]
        if self.pose is None:
            return not len(others)
        yaw, speed = self.pose
        for k in others:
            bx, by, bhl, bhw = K.box(xy[k, 0], xy[k, 1], yaw[k], speed[k], self.ext[k])
            c, s = math.cos(yaw[k]), math.sin(yaw[k])
            dx, dy = spot[0] - bx, spot[1] - by
            if math.hypot(max(abs(dx * c + dy * s) - bhl, 0.0), max(abs(-dx * s + dy * c) - bhw, 0.0)) < need:
                return False
        return True

    def _refuge(self, j, i, xy, active, path_pts):
        """The refuge of person j from the way of i (module docstring), or None. The spots tried and the first
        condition each failed are kept in `refusals[j]` for diagnostics."""
        need = self.half_width[i] + self.half_width[j] + CLEAR_GAP_M
        why = dict(near_way=0, unsupported=0, occupied=0, not_walkable=0, cannot_turn=0, way_occupied=0)
        self.refusals[j] = why
        for r in REFUGE_M:
            best = None
            for a in np.linspace(0.0, 2 * math.pi, 16, endpoint=False):
                spot = xy[j] + r * np.array([math.cos(a), math.sin(a)])
                gap = float(np.linalg.norm(path_pts - spot, axis=1).min())
                if gap < need:
                    why["near_way"] += 1
                    continue
                if best is not None and gap <= best[0]:
                    continue
                failed = (
                    "unsupported"
                    if not self._fits(j, spot[0], spot[1], a)
                    else "occupied"
                    if not self._clear(j, spot, xy, active, [])
                    else "not_walkable"
                    if not self._walkable(xy[j], spot)
                    else "cannot_turn"
                    if not self._turns(j, spot[0], spot[1])
                    else None
                )
                if failed is not None:
                    why[failed] += 1
                    continue
                # the straight way there passes no other body, the waiting walker's included
                ok = True
                for u in np.linspace(0.25, 0.75, 3):
                    if not self._clear(j, xy[j] + u * (spot - xy[j]), xy, active, []):
                        ok = False
                        break
                if ok:
                    best = (gap, spot)
                else:
                    why["way_occupied"] += 1
            if best is not None:
                return best[1]
        return None

    def update(
        self, t, travel, standing, xy, speed, active, stall_run, paths, agendas, dt, names=None, cars=None, yaw=None
    ):
        """Ask and release. `standing`: people at a stay who may be asked (their agendas have pause/resume);
        `cars`: {car: dict(ahead=points, longer=points, waited=ticks[, blocker=body])} for the cars on a trip (route
        ahead of the car's centre, up to CAR_AHEAD_M beyond its front, and 4 m further; ticks it has stood; the body
        that sets its stop, -1 for none); `yaw`: headings of all bodies (with `speed`, their pose boxes for
        clearance; bounding circles without it).
        Returns {person: points} for the people who start walking now (to a refuge or back to their spot)."""
        out = {}
        cars = cars or {}
        self.pose = None if yaw is None else (np.asarray(yaw, float), np.asarray(speed, float))
        travelling = set(travel)
        self.resumed = []

        def resume_here(j):
            agendas[j].resume(t)
            del self.aside[j]
            self.cooldown[j] = t + COOLDOWN_S
            self.resumed.append(j)
            self.stats["resumed_in_place"] += 1

        for j, a in list(self.aside.items()):
            if a["stage"] == "out" and t - a["since"] > OUT_TIMEOUT_S:
                # the refuge cannot be reached (the way has filled): a walker goes on from here, a person at a stay
                # back to its spot
                self.stats["out_timeouts"] += 1
                if a.get("travelling"):
                    resume_here(j)
                    continue
                a["stage"] = "back"
                a["target"] = a["home"].copy()
                out[j] = np.array([xy[j], a["home"]])
                continue
            if a["stage"] in ("out", "back") and np.linalg.norm(xy[j] - a["target"]) < ARRIVE_M and speed[j] < 0.08:
                if a["stage"] == "out":
                    a["stage"] = "wait"
                    a["since"] = t
                else:
                    agendas[j].resume(t)
                    del self.aside[j]
                    self.cooldown[j] = t + COOLDOWN_S
                    self.stats["returned"] += 1
                continue
            if a["stage"] == "wait":
                r = a["requester"]
                going = r in cars if a.get("car") else r in travelling
                beyond = RELEASE_M + (self.half_length[r] if a.get("car") else 0.0)
                passed = not going or (
                    float(np.linalg.norm(xy[r] - a["home"])) > beyond
                    and float(np.dot(xy[r] - a["home"], xy[r] - a["asked_from"])) > 0.0
                )
                timeout = t - a["since"] > TIMEOUT_S
                if (passed or timeout) and (a.get("travelling") or self._clear(j, a["home"], xy, active, [])):
                    if timeout and not passed:
                        self.stats["timeouts"] += 1
                    if a.get("travelling"):
                        resume_here(j)
                        continue
                    a["stage"] = "back"
                    a["target"] = a["home"].copy()
                    out[j] = np.array([xy[j], a["home"]])
        need_ticks = round(ASK_AFTER_S / dt)
        movable = [
            j
            for j in sorted(set(standing) | travelling)
            if j not in self.aside and t >= self.cooldown.get(j, -math.inf)
        ]
        for v in sorted(cars):
            car = cars[v]
            if car["waited"] < need_ticks or t < self.asked.get(v, -math.inf):
                continue
            blocker = car.get("blocker", -1)
            near = sorted(
                (
                    j
                    for j in movable
                    if j not in self.aside
                    and (
                        j == blocker
                        or float(np.linalg.norm(car["ahead"] - xy[j], axis=1).min())
                        <= self.half_width[v] + self.radius[j] + REACH_GAP_M
                    )
                ),
                key=lambda j: (j != blocker, float(np.linalg.norm(xy[j] - xy[v]))),
            )
            if not near:
                continue
            self.asked[v] = t + ASK_EVERY_S
            for j in near[:3]:
                self.stats["car_asked"] += 1
                spot = self._refuge(j, v, xy, active, car["longer"])
                if spot is None:
                    self.stats["no_refuge"] += 1
                    self.cooldown[j] = t + COOLDOWN_S / 2
                    continue
                if not agendas[j].pause(t, "obstructs_" + (str(names[v]) if names is not None else str(v))):
                    continue
                self.aside[j] = dict(
                    requester=v,
                    home=xy[j].copy(),
                    target=spot,
                    stage="out",
                    since=t,
                    asked_from=xy[v].copy(),
                    car=True,
                    travelling=j in travelling,
                )
                out[j] = np.array([xy[j], spot])
                self.stats["stepped_aside"] += 1
        candidates = [j for j in standing if j not in self.aside and t >= self.cooldown.get(j, -math.inf)]
        if not candidates:
            return out
        cx = xy[candidates]
        for i in travel:
            if stall_run[i] < need_ticks or i not in paths or i in self.aside or t < self.asked.get(i, -math.inf):
                continue
            near = [candidates[k] for k in np.flatnonzero(np.linalg.norm(cx - xy[i], axis=1) <= AHEAD_M + 1.0)]
            near = [j for j in near if j not in self.aside]
            if not near:
                continue
            path = paths.path(i)
            ahead = np.array(
                [path.at(s) for s in np.arange(path.s, min(path.s + AHEAD_M, path.total) + 1e-9, 0.25)]
                or [path.at(path.s)]
            )
            longer = np.array(
                [path.at(s) for s in np.arange(path.s, min(path.s + AHEAD_M + 4.0, path.total) + 1e-9, 0.25)]
                or [path.at(path.s)]
            )
            asked = 0
            for j in sorted(near, key=lambda j: float(np.linalg.norm(xy[j] - xy[i]))):
                if float(np.linalg.norm(ahead - xy[j], axis=1).min()) > self.radius[i] + self.radius[j] + REACH_GAP_M:
                    continue
                self.asked[i] = t + ASK_EVERY_S
                self.stats["asked"] += 1
                spot = self._refuge(j, i, xy, active, longer)
                if spot is None:
                    self.stats["no_refuge"] += 1
                    self.cooldown[j] = t + COOLDOWN_S / 2
                    continue
                if not agendas[j].pause(t, "obstructs_" + (str(names[i]) if names is not None else str(i))):
                    continue
                self.aside[j] = dict(
                    requester=i, home=xy[j].copy(), target=spot, stage="out", since=t, asked_from=xy[i].copy()
                )
                out[j] = np.array([xy[j], spot])
                self.stats["stepped_aside"] += 1
                asked += 1
                if asked >= 2:
                    break
        return out
