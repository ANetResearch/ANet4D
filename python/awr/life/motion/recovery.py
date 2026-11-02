"""Bounded pedestrian shuffling to regain a walkable path after a local deadlock.

A person may briefly translate sideways while retaining its body heading. This
is a separate motion authority, not a change to its activity or a pose repair.
Targets come from nearby supported space with a fully walkable connector back
to the current route. Search is local, event-triggered and budgeted; no global
A* or connectivity rebuild. Every executed step is rechecked against live bodies.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from . import kernels as K
from .crowd import DT

AFTER_S = 8.0
RETRY_S = 10.0
TIMEOUT_S = 15.0
PLANS_PER_TICK = 2
MAX_SPEED = 0.25


@njit(cache=True)
def _clear(i, x, y, heading, velocity, xy, yaw, speed, active, ext, margin):
    ax, ay, al, aw = K.box(x, y, heading, velocity, ext[i])
    for j in range(len(xy)):
        if j == i or not active[j]:
            continue
        bx, by, bl, bw = K.box(xy[j, 0], xy[j, 1], yaw[j], abs(speed[j]), ext[j])
        if abs(ax - bx) > al + aw + bl + bw + margin or abs(ay - by) > al + aw + bl + bw + margin:
            continue
        if K.overlap(ax, ay, heading, al, aw, bx, by, yaw[j], bl, bw, -margin):
            return False
    return True


@njit(cache=True)
def _target(i, here, heading, lead, xy, yaw, speed, active, ext, z_ok, walk, sx, sy, cell, free, fx, fy, step):
    best = math.inf
    target = here.copy()
    found = False
    # Fixed ordering and bounded polar candidates, independent of actor/query ids.
    for radius in (0.25, 0.5, 0.75, 1.0):
        for j in range(16):
            a = j * math.pi / 8
            point = here + radius * np.array([math.cos(a), math.sin(a)])
            next_heading = math.atan2(lead[1] - point[1], lead[0] - point[0])
            if np.linalg.norm(lead - point) < 0.15:
                continue
            if not K.route_supported(np.vstack((point, lead)), ext[i], z_ok, walk, sx, sy, cell, free, fx, fy, step):
                continue
            if not K.translated_body_supported(here, point, heading, 0.02, ext[i], z_ok, walk, sx, sy, cell):
                continue
            valid = True
            # Dynamic occupancy of the rejoin segment can clear after a car
            # passes. Only the shuffle itself is admitted here; normal walking
            # must recheck that segment against the live world after recovery.
            # The compact shuffle corridor and the turn back to normal walking.
            for k in range(1, int(math.ceil(radius / 0.05)) + 1):
                q = here + (point - here) * k / math.ceil(radius / 0.05)
                if not K.in_cells(q[0], q[1], free, fx, fy, step) or not _clear(
                    i, q[0], q[1], heading, 0.02, xy, yaw, speed, active, ext, 0.02
                ):
                    valid = False
                    break
            if not valid:
                continue
            delta = K.wrap(next_heading - heading)
            for k in range(17):
                h = heading + delta * k / 16
                if not K.body_supported(point[0], point[1], h, 0.02, ext[i], z_ok, walk, sx, sy, cell) or not _clear(
                    i, point[0], point[1], h, 0.02, xy, yaw, speed, active, ext, 0.02
                ):
                    valid = False
                    break
            if not valid or not _clear(i, point[0], point[1], next_heading, 0.5, xy, yaw, speed, active, ext, 0.02):
                continue
            cost = radius + 0.5 * np.linalg.norm(lead - point)
            if cost < best:
                best, target, found = cost, point, True
    return target, found


@njit(cache=True)
def _step(
    i,
    here,
    heading,
    v0,
    target,
    xy,
    yaw,
    speed,
    next_xy,
    next_yaw,
    active,
    ext,
    z_ok,
    walk,
    sx,
    sy,
    cell,
    free,
    fx,
    fy,
    step,
):
    distance = np.linalg.norm(target - here)
    direction = (target - here) / max(distance, 1e-12)
    upper = min(MAX_SPEED, v0 + 0.1, distance / DT)
    lower = max(v0 - 0.5, 0.0)
    heading_delta = np.empty(len(yaw))
    for j in range(len(yaw)):
        heading_delta[j] = K.wrap(next_yaw[j] - yaw[j])
    # Include a compact shuffle below the rest/stride blending threshold.
    for v in (upper, min(upper, 0.12), min(upper, 0.06), min(upper, 0.015), 0.0):
        if v < lower:
            continue
        point = here + direction * v * DT
        good = True
        sw = max(v0, v) + 0.01
        for k in range(7):
            u = k / 6
            q = here + (point - here) * u
            if not K.in_cells(q[0], q[1], free, fx, fy, step) or not K.body_supported(
                q[0], q[1], heading, sw, ext[i], z_ok, walk, sx, sy, cell
            ):
                good = False
                break
            positions = xy + (next_xy - xy) * u
            headings = yaw + heading_delta * u
            if not _clear(i, q[0], q[1], heading, sw, positions, headings, speed, active, ext, 0.01):
                good = False
                break
        if good:
            return point, v, True
    return here.copy(), 0.0, False


class Recoveries:
    def __init__(self, crowd):
        self.crowd = crowd
        self.targets = {}
        self.retry = {}
        self.events = []
        self.stats = dict(plans=0, admitted=0, no_target=0, completed=0, timed_out=0, motion_waits=0)

    def warm(self, xy, yaw, speed):
        """Compile/load kernels before the simulation clock starts, without changing state."""
        ids = np.flatnonzero(self.crowd.kinds == "person")
        if not len(ids):
            return
        i = int(ids[0])
        c = self.crowd
        active = np.zeros(len(xy), bool)
        _target(
            i,
            xy[i],
            yaw[i],
            xy[i] + np.array([0.5, 0.0]),
            xy,
            yaw,
            speed,
            active,
            c.ext,
            c.z_ok,
            c.walk,
            c.sx0,
            c.sy0,
            c.scell,
            c.free,
            c.fx0,
            c.fy0,
            c.fstep,
        )
        _step(
            i,
            xy[i],
            yaw[i],
            0.0,
            xy[i].copy(),
            xy,
            yaw,
            speed,
            xy.copy(),
            yaw.copy(),
            active,
            c.ext,
            c.z_ok,
            c.walk,
            c.sx0,
            c.sy0,
            c.scell,
            c.free,
            c.fx0,
            c.fy0,
            c.fstep,
        )

    def predictions(self, xy, yaw, times, predictions):
        """Override forward-heading predictions for actual sideways motion."""
        for i, r in self.targets.items():
            v = r["velocity"]
            for k, t in enumerate(times):
                displacement = v * t
                remaining = r["target"] - xy[i]
                if np.linalg.norm(displacement) > np.linalg.norm(remaining):
                    displacement = remaining
                predictions[0][i, k], predictions[1][i, k] = xy[i] + displacement
                predictions[2][i, k] = yaw[i]
        return predictions

    def propose(self, t, travel, xy, yaw, speed, active, paths, stall, next_xy=None, next_yaw=None, next_speed=None):
        c = self.crowd
        travel = set(travel)
        for i, r in list(self.targets.items()):
            reason = None
            if i not in travel or i not in paths or not np.array_equal(paths.path(i).p[-1], r["goal"]):
                reason = "route_changed"
            elif np.linalg.norm(xy[i] - r["target"]) < 0.025 and abs(speed[i]) < 0.03:
                reason = "completed"
                self.stats["completed"] += 1
            elif t - r["since"] > TIMEOUT_S:
                reason = "timed_out"
                self.stats["timed_out"] += 1
            if reason:
                del self.targets[i]
                self.retry[i] = t + RETRY_S
                self.events.append(dict(t_s=t, actor=i, kind="local_recovery_" + reason))
        due = sorted(
            (
                i
                for i in travel
                if i not in self.targets
                and i in paths
                and abs(speed[i]) <= 0.08
                and stall[i] * DT >= AFTER_S
                and t >= self.retry.get(i, -math.inf)
            ),
            key=lambda i: (-stall[i], i),
        )[:PLANS_PER_TICK]
        for i in due:
            self.stats["plans"] += 1
            self.retry[i] = t + RETRY_S
            p = paths.path(i)
            lead = p.at(min(p.total, p.s + 1.5))
            target, found = _target(
                i,
                xy[i],
                yaw[i],
                lead,
                xy,
                yaw,
                speed,
                active,
                c.ext,
                c.z_ok,
                c.walk,
                c.sx0,
                c.sy0,
                c.scell,
                c.free,
                c.fx0,
                c.fy0,
                c.fstep,
            )
            if found:
                self.targets[i] = dict(target=target, since=t, goal=p.p[-1].copy(), velocity=np.zeros(2))
                self.stats["admitted"] += 1
                self.events.append(dict(t_s=t, actor=i, kind="local_recovery_started", target=target.tolist()))
            else:
                self.stats["no_target"] += 1
        future = xy.copy() if next_xy is None else next_xy.copy()
        facing = yaw.copy() if next_yaw is None else next_yaw.copy()
        # Every controller shares the swept body envelope, not only positions.
        # A preceding shuffle can enlarge a resting person's stride envelope.
        # Reserve that enlarged body before proposing another immutable shuffle.
        swept_speed = np.maximum(np.abs(speed), np.abs(speed if next_speed is None else next_speed)) + 0.01
        rows = []
        for i, r in sorted(self.targets.items()):
            point, v, valid = _step(
                i,
                xy[i],
                yaw[i],
                abs(speed[i]),
                r["target"],
                xy,
                yaw,
                swept_speed,
                future,
                facing,
                active,
                c.ext,
                c.z_ok,
                c.walk,
                c.sx0,
                c.sy0,
                c.scell,
                c.free,
                c.fx0,
                c.fy0,
                c.fstep,
            )
            if not valid:
                raise ValueError(f"No safe recovery stop for body {i}")
            if v == 0.0:
                self.stats["motion_waits"] += 1
            r["velocity"] = (point - xy[i]) / DT
            future[i] = point
            swept_speed[i] = max(abs(speed[i]), abs(v)) + 0.01
            pred = np.array(
                [xy[i] + r["velocity"] * min(t_, np.linalg.norm(r["target"] - xy[i]) / max(v, 1e-12)) for t_ in c.times]
            )
            rows.append((i, point, yaw[i], v, pred[:, 0], pred[:, 1], np.full(len(c.times), yaw[i])))
        return rows
