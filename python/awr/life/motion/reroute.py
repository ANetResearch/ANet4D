"""Congestion rerouting of the life runtime: a walker held up by people standing in its way looks for another way.

A travelling person who has not moved for AFTER_S (and is not giving way in a standoff) plans its route anew from
where it stands to the same goal, with the cells near every body that stands still (people at a stay or waiting,
stalled walkers, stopped vehicles: within FOOTPRINT_M of its observed footprint, or JAM_RADIUS_M of its centre when no
body geometry is given) costing (1+JAM_FACTOR) times their static cost: the search
accepts a detour of up to about JAM_FACTOR metres for every metre of standing bodies it would otherwise pass
through. The person takes the new route only when it passes fewer standing bodies: the length of its first AHEAD_M
within JAM_RADIUS_M of another standing body is at least GAIN_M shorter than that of the present route (a way
round a knot of people as well as another street); otherwise it keeps its route. Each person replans at most every
RETRY_S, and at most PER_TICK people replan in one tick (the longest waiting first).

A new route is an ordinary static route (LifeRouter on the same snapshot, with the connector, body and turning
checks of WalkNetwork.finish_grid_path), walked by the crowd like any other: nobody is moved. A change of route is
reported as `journey_delayed` with reason `rerouted`.
"""

from __future__ import annotations

import math

import numpy as np

from .crowd import Path

AFTER_S = 8.0
RETRY_S = 20.0
STILL_MPS = 0.2
JAM_RADIUS_M = 0.8
FOOTPRINT_M = 0.45
JAM_FACTOR = 8.0
AHEAD_M = 12.0
GAIN_M = 0.5
SAMPLE_M = 0.25
PER_TICK = 4


def _samples(path, s0, length, step=SAMPLE_M):
    s = np.arange(s0, min(s0 + length, path.total) + 1e-9, step)
    return np.array([path.at(v) for v in s]) if len(s) else path.at(s0)[None]


class Reroutes:
    def __init__(self, router, ext=None):
        self.router = router
        # Body extents (n, 2, 4) of the population: a stopped car costs its whole body, not a disc round its centre
        # (cruise5 r11f v3: a walker beside a car stopped for it kept its way through the car for 110 s).
        self.ext = None if ext is None else np.asarray(ext, float)
        self.next = {}
        self.stats = dict(reroute_plans=0, rerouted=0, kept=0, reroute_failures=0)

    def update(
        self, t, travel, xy, speed, active, stall_run, paths, goal_of, busy, agendas, dt, *, yaw=None, extra=None
    ):
        """Returns {person: points} for the people who take another way now. `goal_of(i)` is the goal of person
        i's current walk; `busy` the people whose path is managed elsewhere (giving way); `extra` further router
        cells counted as standing bodies (the ways ahead of standing cars, LifeEngine.car_cells)."""
        need = round(AFTER_S / dt)
        due = [
            i
            for i in travel
            if stall_run[i] >= need and i in paths and i not in busy and t >= self.next.get(i, -math.inf)
        ]
        if not due:
            return {}
        due = sorted(due, key=lambda i: (-stall_run[i], i))[:PER_TICK]
        still = np.flatnonzero(np.asarray(active, bool) & (np.asarray(speed) < STILL_MPS))
        out = {}
        for i in due:
            self.next[i] = t + RETRY_S
            self.stats["reroute_plans"] += 1
            others = still[still != i]
            if self.ext is None:
                cells = self.router.avoid_cells(xy[others], JAM_RADIUS_M)
            else:
                if yaw is None:
                    raise ValueError("Footprint congestion requires observed headings")
                cells = self.router.avoid_bodies(
                    xy[others], np.asarray(yaw)[others], np.asarray(speed)[others], self.ext[others], FOOTPRINT_M
                )
            if extra is not None and len(extra):
                cells = np.union1d(cells, extra)
            rr = self.router.plan(np.asarray(xy[i]), np.asarray(goal_of(i), float), avoid=(cells, JAM_FACTOR))
            if rr is None:
                self.stats["reroute_failures"] += 1
                continue
            pts = np.vstack([xy[i], np.asarray(rr["points"])[1:]])
            old = paths.path(i)
            jam = np.zeros(len(self.router.cost), bool)
            jam[cells] = True
            if (
                self._exposure(_samples(Path(pts), 0.0, AHEAD_M), jam)
                > self._exposure(_samples(old, old.s, AHEAD_M), jam) - GAIN_M
            ):
                self.stats["kept"] += 1
                continue
            out[i] = pts
            self.stats["rerouted"] += 1
            agendas[i].blocked(t, "rerouted")
        return out

    def _exposure(self, points, jam):
        """Length (m) of a sampled path within the standing bodies' cells."""
        network, index = self.router.network, self.router.index
        rc = np.floor((points - network.origin) / network.step + 0.5).astype(int)[:, ::-1]
        ok = (rc >= 0).all(1) & (rc[:, 0] < index.shape[0]) & (rc[:, 1] < index.shape[1])
        k = np.full(len(points), -1)
        k[ok] = index[rc[ok, 0], rc[ok, 1]]
        return SAMPLE_M * float(jam[k[k >= 0]].sum())
