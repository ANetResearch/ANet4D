"""Vehicle motion of the life runtime on admitted driving geometry.

A driving vehicle follows the exact route of its trip (DrivingRoutes.plan: concatenated admitted edges, unwrapped
yaw, gear per segment); only its progress along the route is controlled. Each tick the live lookahead
(`stop_arc`) finds the first route position where the vehicle body, kept VEHICLE_ROOM_M from vehicles and objects
and PERSON_ROOM_M plus PERSON_ROOM_S seconds of its speed from people (it passes a pedestrian closely only at a
crawl), would meet another body: standing bodies where they stand, moving people where
they are predicted to be when the vehicle gets there. `LiveDrive` then advances the vehicle within its limits so
that it can stop before that position, before every gear change and at the destination berth. A body already
nearer than its room (a person who stepped up to the side of a standing vehicle) stops only a motion that would bring
the vehicle nearer to it still (by more than NO_CLOSER_M) or keep it within CLOSE_FLOOR_M without drawing away (a
person pressed against the back of a car does not hold it from driving off); the vehicle passes it at a crawl
(its stop position is kept CRAWL_M ahead). It brakes
comfortably at Limits.braking_mps2 and, only for a newly appeared obstruction, at up to EMERGENCY_BRAKE_MPS2
(below the tape QA limit of 6 m/s2). If even that cannot stop it in time (a person stepped in late), it brakes at
that rate and the people in the way give way in the crowd's safety pass, where an actual contact remains a
failure; such overruns are counted.

LiveDrive has the interface of awr.life.civil.driving_motion.RouteExecution (state, advance, arrived, arc_m, time_s,
brake_count), so DrivingRoutes (trip admission, corridor reservation, arrival) and DriverExecution (the activity
Director's drive step) are reused unchanged. People see a driving vehicle through `predict_route`: its predicted
poses along the route at its current speed (at a set-off speed once it has stood a while in its trip,
awr.life.motion.engine.car_intent_mps).
"""

from __future__ import annotations

import copy
import math

import numpy as np
from numba import njit

from awr.life.civil.driving_graph import route_arrays
from awr.life.civil.driving_motion import Limits, Schedule

from .kernels import box, gap

PERSON_ROOM_M = 0.25
PERSON_ROOM_S = 0.1
VEHICLE_ROOM_M = 0.3
EMERGENCY_BRAKE_MPS2 = 4.5
LOOKAHEAD_S = 2.5
LOOKAHEAD_STEP_M = 0.2
CRAWL_M = 0.5
NO_CLOSER_M = 0.005
CLOSE_FLOOR_M = 0.05


@njit(cache=True)
def route_at(pts, yaw, arc, s):
    """Pose (x, y, yaw) at arc length s of a route (linear interpolation of points and unwrapped yaw)."""
    n = len(arc)
    if s <= arc[0]:
        return pts[0, 0], pts[0, 1], yaw[0]
    if s >= arc[n - 1]:
        return pts[n - 1, 0], pts[n - 1, 1], yaw[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if arc[mid] <= s:
            lo = mid
        else:
            hi = mid
    u = (s - arc[lo]) / (arc[hi] - arc[lo])
    return (
        pts[lo, 0] + u * (pts[hi, 0] - pts[lo, 0]),
        pts[lo, 1] + u * (pts[hi, 1] - pts[lo, 1]),
        yaw[lo] + u * (yaw[hi] - yaw[lo]),
    )


@njit(cache=True)
def predict_route(pts, yaw, arc, s0, speed, times):
    """Predicted poses of a driving vehicle at `times` (constant current speed along its route)."""
    m = len(times)
    qx = np.empty(m)
    qy = np.empty(m)
    qyaw = np.empty(m)
    for k in range(m):
        qx[k], qy[k], qyaw[k] = route_at(pts, yaw, arc, s0 + speed * times[k])
    return qx, qy, qyaw


@njit(cache=True)
def _holds(g, now, d):
    """Whether a body within the vehicle's room at gap g (m) from the swept pose d metres ahead, and at gap `now` from
    the vehicle where it stands, stops the vehicle: that pose is nearer to it than the vehicle is now (by more than
    NO_CLOSER_M), or stays within CLOSE_FLOOR_M of it without drawing away. The present pose (d = 0) is no motion of
    the vehicle, and standing there keeps it from nobody: a person pressed against the back of a car, or walking up
    to it, does not hold it from driving off (cq3p-d8-s4 v1, person-086 1.3 cm behind vehicle-042; cq3p-d12-s3 v4,
    person-128 walking up behind vehicle-049; each car stood 60 s). A body in the way ahead is met at the first step."""
    return d > 0.0 and (g < now - NO_CLOSER_M or (g < CLOSE_FLOOR_M and g < now + NO_CLOSER_M))


@njit(cache=True)
def _lookahead(
    me,
    pts,
    yaw,
    arc,
    s0,
    speed,
    length,
    ext,
    px,
    py,
    pyaw,
    pspeed,
    is_person,
    active,
    ignore,
    pred_x,
    pred_y,
    pred_yaw,
    times,
    person_room,
    person_room_s,
    vehicle_room,
    step,
):
    """(stop arc, index of the body that sets it or -1): stop_arc."""
    n = len(px)
    end = arc[len(arc) - 1]
    horizon = times[len(times) - 1]
    # Present gap of each body near the vehicle where both are now (+inf farther than any room).
    x0, y0, th0 = route_at(pts, yaw, arc, s0)
    cx, cy, chl, chw = box(x0, y0, th0, 0.0, ext[me])
    near = math.hypot(chl, chw) + 3.0
    now = np.full(n, np.inf)
    for j in range(n):
        if j == me or not active[j] or ignore[j] or (px[j] - x0) ** 2 + (py[j] - y0) ** 2 > near * near:
            continue
        bx, by, bhl, bhw = box(px[j], py[j], pyaw[j], pspeed[j], ext[j])
        now[j] = gap(cx, cy, th0, chl, chw, bx, by, pyaw[j], bhl, bhw)
    crawl = math.inf
    d = 0.0
    while d <= length + 1e-9:
        s = s0 + d
        if s > end:
            break
        x, y, th = route_at(pts, yaw, arc, s)
        ax, ay, ahl, ahw = box(x, y, th, 0.0, ext[me])
        tau = d / max(speed, 1.0)
        for j in range(n):
            if j == me or not active[j] or ignore[j]:
                continue
            room = (person_room + person_room_s * speed) if is_person[j] else vehicle_room
            reach = math.hypot(ahl, ahw) + 3.0 + room + pspeed[j] * (min(tau, horizon) + 0.5)
            if (px[j] - x) ** 2 + (py[j] - y) ** 2 > reach * reach:
                continue
            if pspeed[j] <= 0.05 or not is_person[j]:
                bx, by, bhl, bhw = box(px[j], py[j], pyaw[j], pspeed[j], ext[j])
                g = gap(ax, ay, th, ahl, ahw, bx, by, pyaw[j], bhl, bhw)
                if g < room:
                    if _holds(g, now[j], d):
                        return min(max(s - step, s0), crawl), j
                    crawl = s0 + CRAWL_M
                continue
            for dt_ in (-1e9, -0.5, 0.0, 0.5):  # where it is now (it may stop at once) and around the vehicle's arrival
                tt = min(max(tau + dt_, 0.0), horizon)
                k = 0
                while k < len(times) - 1 and times[k] < tt:
                    k += 1
                if tt <= 0.0:
                    qx, qy, qth = px[j], py[j], pyaw[j]
                elif tt <= times[0]:
                    u = tt / times[0]
                    qx = px[j] + u * (pred_x[j, 0] - px[j])
                    qy = py[j] + u * (pred_y[j, 0] - py[j])
                    qth = pred_yaw[j, 0]
                else:
                    u = (tt - times[k - 1]) / (times[k] - times[k - 1])
                    qx = pred_x[j, k - 1] + u * (pred_x[j, k] - pred_x[j, k - 1])
                    qy = pred_y[j, k - 1] + u * (pred_y[j, k] - pred_y[j, k - 1])
                    qth = pred_yaw[j, k]
                bx, by, bhl, bhw = box(qx, qy, qth, pspeed[j], ext[j])
                g = gap(ax, ay, th, ahl, ahw, bx, by, qth, bhl, bhw)
                if g < room:
                    if _holds(g, now[j], d):
                        return min(max(s - step, s0), crawl), j
                    crawl = s0 + CRAWL_M
        d += step
    return crawl, -1


@njit(cache=True)
def stop_arc(
    me,
    pts,
    yaw,
    arc,
    s0,
    speed,
    length,
    ext,
    px,
    py,
    pyaw,
    pspeed,
    is_person,
    active,
    ignore,
    pred_x,
    pred_y,
    pred_yaw,
    times,
    person_room,
    person_room_s,
    vehicle_room,
    step,
):
    """First arc position (absolute) at which body `me` following the route would come within the room of
    another body, less one lookahead step, or +inf when the next `length` metres are clear.

    Standing bodies (speed <= 0.05 m/s) are tested where they stand; moving people where they are (a person can
    stop at once) and where they are predicted to be (pred_*, sampled at `times`) within 0.5 s of the moment the
    vehicle reaches the position at its present speed (at least 1 m/s, so that a stopped vehicle still sees people
    who are about to cross). Bodies with
    `ignore` set (the vehicle's own occupant, partners of a manoeuvre) are skipped. A body nearer than its room
    already stops the vehicle only where it would come nearer still (module docstring); while it passes such a body
    the stop is at most CRAWL_M ahead."""
    return _lookahead(
        me,
        pts,
        yaw,
        arc,
        s0,
        speed,
        length,
        ext,
        px,
        py,
        pyaw,
        pspeed,
        is_person,
        active,
        ignore,
        pred_x,
        pred_y,
        pred_yaw,
        times,
        person_room,
        person_room_s,
        vehicle_room,
        step,
    )[0]


@njit(cache=True)
def stop_blocker(
    me,
    pts,
    yaw,
    arc,
    s0,
    speed,
    length,
    ext,
    px,
    py,
    pyaw,
    pspeed,
    is_person,
    active,
    ignore,
    pred_x,
    pred_y,
    pred_yaw,
    times,
    person_room,
    person_room_s,
    vehicle_room,
    step,
):
    """Index of the body that sets stop_arc (same arguments), or -1 when none does."""
    return _lookahead(
        me,
        pts,
        yaw,
        arc,
        s0,
        speed,
        length,
        ext,
        px,
        py,
        pyaw,
        pspeed,
        is_person,
        active,
        ignore,
        pred_x,
        pred_y,
        pred_yaw,
        times,
        person_room,
        person_room_s,
        vehicle_room,
        step,
    )[1]


class LiveDrive:
    """Arc-based execution of one admitted trip (RouteExecution interface).

    The nominal speed at each route position is that of the rest-to-rest Schedule (curvature, reverse speed,
    rest at gear changes and at the end, acceleration and comfortable braking); the vehicle never exceeds it, and
    additionally keeps the stopping distance to the live stop position. At a gear change it rests for
    Limits.gear_dwell_s. World time advances while braking and waiting."""

    def __init__(self, route, limits=Limits()):
        self.limits = limits
        self._set_route(route)
        self.arc_m = 0.0
        self.speed_mps = 0.0
        self.gear = 0
        self.time_s = 0.0
        self.hold_s = 0.0
        self.arrived = False
        self.brake_count = 0
        self.emergency_count = 0
        self.overruns = 0
        self.mode = "scheduled"
        self._braking = False

    def _set_route(self, route):
        self.route = route_arrays(route["points"], route["yaw"], route.get("gear"))
        self.schedule = Schedule(self.route, self.limits)
        ph = self.schedule.phases
        moving = ph[ph[:, 3] > ph[:, 2] + 1e-12]
        self._s0, self._s1, self._v0, self._v1 = (np.ascontiguousarray(moving[:, k]) for k in (2, 3, 4, 5))
        r = self.route
        self.cusps = r["arc"][np.flatnonzero(np.diff(r["gear"])) + 1].astype(float)
        self.length_m = float(r["arc"][-1])

    def prepare_reroute(self, route, splice_m):
        """A copy of this execution that continues on `route`, which follows the present route up to arc `splice_m`
        (ahead of the body) and then takes another way: the position, speed, gear, clock and counters carry over; the
        nominal speed is that of the new route's schedule, which may not lie below the present speed at the body. The
        caller chooses the splice beyond the stopping distance. This execution is not changed."""
        old = self.route
        new = route_arrays(route["points"], route["yaw"], route.get("gear"))
        if self.arrived or not self.arc_m < splice_m <= min(self.length_m, float(new["arc"][-1])) + 1e-9:
            raise ValueError("A new route must follow the present one up to a splice ahead of the body")
        # Both routes are piecewise linear: comparing them at the union of their knots before the splice checks the
        # whole kept prefix, gear changes included.
        probe = np.unique(np.r_[0.0, old["arc"][old["arc"] < splice_m], new["arc"][new["arc"] < splice_m], splice_m])
        a = np.column_stack([np.interp(probe, old["arc"], old["points"][:, k]) for k in (0, 1)])
        b = np.column_stack([np.interp(probe, new["arc"], new["points"][:, k]) for k in (0, 1)])
        turn = (
            np.interp(probe, old["arc"], old["yaw"]) - np.interp(probe, new["arc"], new["yaw"]) + math.pi
        ) % math.tau - math.pi
        mid = (probe[1:] + probe[:-1]) / 2.0

        def gear(r):
            return r["gear"][np.minimum(np.searchsorted(r["arc"], mid, side="right") - 1, len(r["gear"]) - 1)]

        if (
            np.max(np.linalg.norm(a - b, axis=1)) > 1e-6
            or np.max(np.abs(turn)) > 1e-6
            or not np.array_equal(gear(old), gear(new))
        ):
            raise ValueError("A new route must follow the present one up to the splice")
        candidate = copy.copy(self)
        candidate._set_route(route)
        if candidate.nominal_speed(self.arc_m) + 1e-6 < self.speed_mps:
            raise ValueError("A new route may not demand an instantaneous speed reduction")
        return candidate

    def reroute(self, route, splice_m):
        """Continue on `route` (prepare_reroute), in place."""
        self.__dict__.update(self.prepare_reroute(route, splice_m).__dict__)

    def nominal_speed(self, s):
        """Schedule speed at arc s (piecewise constant acceleration in time, so v^2 is linear in arc)."""
        k = int(np.searchsorted(self._s1, s, side="left"))
        if k >= len(self._s1):
            return 0.0
        s0, s1, v0, v1 = self._s0[k], self._s1[k], self._v0[k], self._v1[k]
        u = min(max((s - s0) / (s1 - s0), 0.0), 1.0)
        return math.sqrt(max(v0 * v0 + (v1 * v1 - v0 * v0) * u, 0.0))

    def _gear_at(self, s):
        g = self.route["gear"]
        k = min(max(int(np.searchsorted(self.route["arc"], s + 1e-9, side="right")) - 1, 0), len(g) - 1)
        return int(g[k])

    def state(self):
        x, y, th = route_at(self.route["points"], self.route["yaw"], self.route["arc"], self.arc_m)
        return dict(
            xy=np.array([x, y]),
            yaw=float(th),
            arc_m=self.arc_m,
            speed_mps=self.speed_mps,
            gear=self.gear,
            signed_speed_mps=self.gear * self.speed_mps,
            arrived=self.arrived,
            elapsed_s=self.time_s,
            mode=self.mode,
        )

    def stopping_m(self, decel=None):
        return self.speed_mps**2 / (2.0 * (self.limits.braking_mps2 if decel is None else decel))

    def advance(self, dt, *, stop_before_arc_m=None):
        if not math.isfinite(dt) or not 0 < dt <= 0.1 + 1e-12:
            raise ValueError("Driving execution requires a finite tick in (0, 0.1] s")
        stop = math.inf if stop_before_arc_m is None else float(stop_before_arc_m)
        if stop < self.arc_m - 1e-9:
            raise ValueError("Finite stopping arc cannot be behind the measured body")
        self.time_s += dt
        if self.arrived:
            return self.state()
        if self.hold_s > 1e-9:
            self.hold_s = max(0.0, self.hold_s - dt)
            return self.state()
        lim = self.limits
        later = self.cusps[self.cusps > self.arc_m + 1e-9]
        rest = min(float(later[0]) if len(later) else self.length_m, self.length_m)
        target = min(stop, rest)
        room = max(target - self.arc_m, 0.0)
        v = self.speed_mps
        need = v * v / (2.0 * room) if room > 1e-12 else (math.inf if v > 1e-9 else 0.0)
        decel = lim.braking_mps2
        overrun = False
        if need > decel:
            decel = need
            if stop < rest:
                self.emergency_count += 1
                if need > EMERGENCY_BRAKE_MPS2:
                    # Brake as hard as allowed; the bodies in the way give way in the safety pass (an actual contact
                    # is a failure there). The stop position is then passed; it is not an executable target.
                    decel = EMERGENCY_BRAKE_MPS2
                    self.overruns += 1
                    target = rest
                    room = max(target - self.arc_m, 0.0)
                    overrun = True
        # Highest speed after this tick from which `target` can still be reached at rest braking at `decel`
        # (exact for constant deceleration within the tick: v1^2 = 2 decel (room - (v+v1) dt/2)).
        free = max(room - v * dt * 0.5, 0.0)
        envelope = (-decel * dt + math.sqrt((decel * dt) ** 2 + 8.0 * decel * free)) * 0.5
        ahead = min(self.arc_m + v * dt + 0.5 * lim.acceleration_mps2 * dt * dt, self.length_m)
        want = 0.0 if overrun else min(self.nominal_speed(ahead), envelope)
        braking = want < v - 1e-9 and stop < rest
        if braking and not self._braking:
            self.brake_count += 1
        self._braking = braking
        v1 = min(want, v + lim.acceleration_mps2 * dt) if want >= v else max(want, v - decel * dt)
        ds = (v + v1) * 0.5 * dt
        if self.arc_m + ds >= target - 1e-4 or (v1 < 0.02 and room - ds < 0.02):
            ds = room
            v1 = 0.0
        self.arc_m += ds
        self.speed_mps = max(v1, 0.0)
        if self.speed_mps <= 1e-9:
            self.speed_mps = 0.0
        self.gear = self._gear_at(self.arc_m) if self.speed_mps > 0.0 else 0
        if self.arc_m >= self.length_m - 1e-6 and self.speed_mps == 0.0:
            self.arc_m = self.length_m
            self.arrived = True
            self.mode = "arrived"
        elif self.speed_mps == 0.0 and abs(self.arc_m - rest) <= 1e-6 and rest < self.length_m:
            self.hold_s = lim.gear_dwell_s
            self.mode = "gear_change"
        elif self.speed_mps == 0.0:
            self.mode = "waiting"
        else:
            self.mode = "scheduled"
        return self.state()
