"""Standoffs of the life runtime: two people who block each other and wait.

Two travelling people who have both stood still for STANDOFF_S within REACH_M of each other, one in front of the
other, may be unable to resolve their positions by walking forward: two people face to face in a passage that
only one can use, or a person turning on the spot beside another who needs that space. One of them gives way: it
walks back along the part of its own path it has already walked (known to be walkable) to the nearest refuge
behind it (at most BACK_M[-1]), preferably offset to the path's right, where it can turn round and where the
other can pass - for people walking in opposite directions, the other's walking body fits beside the refuge;
otherwise the refuge is clear of the other's next CLEAR_PATH_M of path. It waits there until the other has passed
it (or, when not opposite, moved on), then resumes its journey along the same path. The person with the shorter
way back gives way (preferably one nobody is waiting for); on a tie, the higher index. After TIMEOUT_S the
person resumes in any case and the standoff may be resolved anew. Nobody is moved other than by walking: the
crowd step checks every step of the retreat as of any other walk.
"""

from __future__ import annotations

import math

import numpy as np

from . import kernels as K
from .crowd import Path

STANDOFF_S = 1.5
REACH_M = 2.5
CLEAR_PATH_M = 6.0
CLEAR_GAP_M = 0.3
MOVED_ON_M = 3.0
BACK_M = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
PASS_GAP_M = 0.12
TURN_SLACK_M = 0.2
PASS_REACH_M = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)  # the passing lane, along the path around the refuge
RIGHT_M = (0.6, 0.3, 0.0)
RELEASE_PASSED_M = 1.0
TIMEOUT_S = 20.0
RETRY_S = 5.0


class Standoffs:
    def __init__(self, crowd):
        self.crowd = crowd
        self.free, self.fx0, self.fy0, self.fstep = crowd.free, crowd.fx0, crowd.fy0, crowd.fstep
        self.radius = np.asarray(crowd.radius, float)
        self.half_width = np.abs(crowd.ext[:, 1, 2:]).max(1)
        self.yielding = {}  # yielder -> dict(partner, since, partner_dir, partner_from, opposing, resume)
        self.retry = {}  # frozenset pair -> time of the next attempt
        self.stats = dict(standoffs=0, gave_way=0, no_refuge=0, released_passed=0, released_partner_left=0, timeouts=0)

    def _walkable(self, a, b):
        n = max(int(math.ceil(float(np.linalg.norm(b - a)) / 0.1)), 1)
        return all(
            K.in_cells(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, self.free, self.fx0, self.fy0, self.fstep)
            for u in np.linspace(0.0, 1.0, n + 1)
        )

    def _fits(self, i, x, y, heading, speed):
        c = self.crowd
        return K.in_cells(x, y, self.free, self.fx0, self.fy0, self.fstep) and K.body_supported(
            x, y, heading, speed, c.ext[i], c.z_ok, c.walk, c.sx0, c.sy0, c.scell
        )

    def _turns(self, i, x, y, slack=TURN_SLACK_M):
        """The resting body can turn round on the spot (it arrives facing back and leaves facing forward), also
        where it stops up to `slack` short of or beside the spot."""
        spots = [(x, y)] + [
            (x + slack * math.cos(b), y + slack * math.sin(b)) for b in np.linspace(0.0, 2 * math.pi, 8, endpoint=False)
        ]
        return all(
            self._fits(i, sx, sy, a, 0.0) for sx, sy in spots for a in np.linspace(0.0, 2 * math.pi, 16, endpoint=False)
        )

    @staticmethod
    def _resume(here, points, s0, s1):
        """The way from `here` back onto the original path (`points`): to the nearest point of the stretch walked
        back (arc s0..s1), wherever the retreat ended, then along the rest of the path."""
        q = Path(points)
        ss = np.linspace(s0, s1, max(int((s1 - s0) / 0.1), 1) + 1)
        s = float(ss[np.argmin([np.linalg.norm(q.at(v) - here) for v in ss])])
        return np.array([here, q.at(s)] + [q.p[i] for i in range(len(q.arc)) if q.arc[i] > s], float)

    def _refuge(self, y, p, xy, path, others, opposing, ahead):
        """(length, points, (path points, s0, s)) of the way back for `y` giving way to `p`, or None. `ahead`
        samples the other's next path."""
        u = xy[p] - xy[y]
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        here = path.at(path.s)
        gap = self.half_width[y] + self.half_width[p] + PASS_GAP_M
        for back in BACK_M:
            s0 = max(path.s - back, 0.0)
            if path.s - s0 < min(back, 1.0):
                break
            base = path.at(s0)
            k = int(np.clip(np.searchsorted(path.arc, s0, side="right") - 1, 0, len(path.seg) - 1))
            d = path.seg[k] / max(float(path.length[k]), 1e-9)  # path direction at the base
            # The other person comes along this path in the opposite direction and keeps to its own right, the
            # path's left; the refuge is offset to the path's right.
            right = np.array([d[1], -d[0]])
            back_heading = math.atan2(-d[1], -d[0])  # the other person's heading as it passes
            for off in RIGHT_M:
                # cheapest tests first
                spot = base + off * right
                if float(np.dot(spot - xy[y], u)) > 0.3:
                    continue
                if (
                    len(others)
                    and (np.linalg.norm(xy[others] - spot, axis=1) < self.radius[others] + self.radius[y]).any()
                ):
                    continue
                if not self._fits(y, spot[0], spot[1], back_heading, 0.0):
                    continue
                if opposing:
                    lane = spot - gap * right  # the other person walks past here, facing back along the path
                    if not all(self._fits(p, *(lane + w * d), back_heading, 1.3) for w in PASS_REACH_M):
                        continue
                elif np.linalg.norm(ahead - spot, axis=1).min() < self.half_width[y] + self.half_width[p] + CLEAR_GAP_M:
                    continue
                if not (self._walkable(base, spot) and self._turns(y, spot[0], spot[1])):
                    continue
                ks = [i for i in range(len(path.arc)) if s0 < path.arc[i] < path.s]
                pts = [xy[y], here] + [path.p[i] for i in reversed(ks)] + [base] + ([spot] if off > 0.0 else [])
                # Afterwards the journey resumes along the same path (see _resume).
                return float(path.s - s0 + off), np.array(pts, float), (path.p.copy(), s0, float(path.s))
        return None

    def update(self, t, travel, xy, stall_run, paths, tangent, agendas, dt):
        """Resolve new standoffs and release finished ones. `paths` is the PathStore. Returns {person: points} to
        apply now: the way back to a refuge for a person who gives way, and for a released one the way from its
        refuge back onto its path."""
        overrides = {}
        travelling = set(travel)
        for y, st in list(self.yielding.items()):
            p = st["partner"]
            passed = (
                float(np.dot(xy[p] - xy[y], st["partner_dir"])) > RELEASE_PASSED_M
            )  # the other is beyond the refuge
            if not st["opposing"]:
                passed |= float(np.linalg.norm(xy[p] - st["partner_from"])) > MOVED_ON_M
            if p not in travelling or passed or t - st["since"] > TIMEOUT_S:
                self.stats[
                    "released_passed" if passed else "released_partner_left" if p not in travelling else "timeouts"
                ] += 1
                del self.yielding[y]
                if y in travelling:
                    overrides[y] = self._resume(xy[y], *st["resume"])
        # A person who gives way is not asked again, nor one released this tick (its stored path is still the way to
        # its refuge, which must not become the path it resumes); one being given way to may be.
        busy = set(self.yielding) | set(overrides)
        passers = {st["partner"] for st in self.yielding.values()}
        need = round(STANDOFF_S / dt)
        stuck = [i for i in travel if stall_run[i] >= need and i in paths and i not in busy]
        if len(stuck) < 2:
            return overrides
        sx = xy[stuck]
        for a, i in enumerate(stuck):
            if i in busy:
                continue
            ti = np.array([math.cos(tangent[i]), math.sin(tangent[i])])
            for b in np.argsort(np.linalg.norm(sx - xy[i], axis=1)):
                j = stuck[b]
                if j == i or j in busy or float(np.linalg.norm(xy[j] - xy[i])) > REACH_M:
                    continue
                tj = np.array([math.cos(tangent[j]), math.sin(tangent[j])])
                if float(np.dot(xy[j] - xy[i], ti)) <= 0.0 and float(np.dot(xy[i] - xy[j], tj)) <= 0.0:
                    continue  # neither is in front of the other
                opposing = float(np.dot(ti, tj)) < -0.3
                pair = frozenset((i, j))
                if t < self.retry.get(pair, -math.inf):
                    continue
                self.stats["standoffs"] += 1
                others = np.array([k for k in range(len(xy)) if k not in (i, j)], int)
                options = []
                for y, p in ((i, j), (j, i)):
                    pp = paths.path(p)
                    ahead = np.array([pp.at(pp.s + w) for w in np.arange(0.0, CLEAR_PATH_M, 0.25)])
                    r = self._refuge(y, p, xy, paths.path(y), others, opposing, ahead)
                    if r is not None:
                        options.append((y in passers, r[0], -y, y, p, r[1], r[2]))
                if not options:
                    self.stats["no_refuge"] += 1
                    self.retry[pair] = t + RETRY_S
                    continue
                _, _, _, y, p, pts, rest = min(options, key=lambda o: o[:3])
                pd = np.array([math.cos(tangent[p]), math.sin(tangent[p])])
                self.yielding[y] = dict(
                    partner=p, since=t, partner_dir=pd, partner_from=xy[p].copy(), opposing=opposing, resume=rest
                )
                overrides[y] = pts
                agendas[y].blocked(t, "gave_way")
                self.stats["gave_way"] += 1
                busy.add(y)
                passers.add(p)
                break
        return overrides
