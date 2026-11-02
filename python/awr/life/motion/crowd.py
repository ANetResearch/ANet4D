"""Crowd motion of the life runtime: path following, compiled velocity selection and the per-tick safety pass.

One authority writes every person's pose. A person moves along its heading (no sideways steps); speed changes
are limited to ACCEL_MPS2 up and COMFORT_DECEL_MPS2 down, or EMERGENCY_DECEL_MPS2 when the selected velocity
predicts contact sooner than a comfortable stop, and LAST_RESORT_DECEL_MPS2 (braking straight) only where the
safety pass cannot otherwise separate two bodies; the heading turns at most TURN_RATE_RPS. These limits stay
below the tape QA (8 m/s2, 3 rad/s). The safety pass checks the swept pose boxes of the tick against every other
body (people, vehicles, street objects) and the measured support surface; a conflicting pair is resolved by
shortening one body's step within the deceleration limit. Nothing is pushed, teleported or excused.

A walker follows its path in a kept lane (its present lateral offset, up to LANE_M) and chooses among lateral
offsets from that lane and speeds (kernels.choose_velocities). Its centre stays in the walking area
(WalkNetwork.area less LEGAL_INSET_M) and its pose box on the measured support; the planning network's extra
clearance applies to route planning only. People keep PERSONAL_M more from a person standing in or crossing
their way, less the longer they have waited (PATIENCE_S). Every offset is also tried at SHUFFLE_MPS, where the
body keeps nearly its compact rest pose, with SHUFFLE_ROOM_M more room from people. Everybody keeps
VEHICLE_ROOM_M more from a moving vehicle. A heading change of more than TURN_FIRST_RAD is made standing first.
Standing costs up to WAIT_PENALTY more after PATIENCE_S of waiting, and a waiting walker may also edge off along
its present heading, so that it takes a safe detour or a slow step instead of waiting for others who wait for it.
Before a hold, the safety pass searches the full dynamic window of the bodies concerned (Crowd._rescue).
After NUDGE_AFTER_S of waiting a walker whose standing is itself blocked may edge aside: a step of NUDGE_MPS in any
direction with its heading kept, so slow that the body keeps its rest pose (awr.life.civil.body). Two people locked
abeam a few centimetres apart, where any turn or start would swing a pose box into the other, separate this way.
"""

from __future__ import annotations

import math

import numpy as np

from ..civil.assemble import recorded_pose
from . import kernels as K

DT = 0.1
ACCEL_MPS2 = 1.0
COMFORT_DECEL_MPS2 = 1.8
EMERGENCY_DECEL_MPS2 = 5.0
LAST_RESORT_DECEL_MPS2 = 7.0
TURN_RATE_RPS = 2.0
CREEP_MPS = 0.15
LEGAL_INSET_M = 0.05
LANE_M = 1.0
PERSONAL_M = 0.1
PATIENCE_S = 3.0
SHUFFLE_MPS = 0.12
SHUFFLE_ROOM_M = 0.15
WAIT_PENALTY = 3.0
VEHICLE_ROOM_M = 0.5
TURN_FIRST_RAD = 0.6
TURN_FIRST_BELOW_MPS = 0.3
NUDGE_AFTER_S = 5.0
RESUMED_S = 1.0  # a wait ends after this long of walking on its way (see PROGRESS_MPS, PROGRESS_RAD)
PROGRESS_RAD = math.pi / 3.0  # a step further off the desired heading (a sidestep) does not end a wait
PROGRESS_MPS = 0.1  # walking on its way: at least this speed (or the requested speed, if slower)
RESCUE_HEADINGS = 41
NUDGE_MPS = 0.015  # below the rest-pose speed of awr.life.civil.body (0.03 m/s) also after the swept-check allowance


class Path:
    """A planned walking polyline with the walker's progress along it."""

    def __init__(self, points):
        p = np.asarray(points, float)
        keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
        self.p = p[keep]
        if len(self.p) < 2:
            self.p = np.vstack([self.p, self.p[-1:]])
        self.seg = np.diff(self.p, axis=0)
        self.length = np.linalg.norm(self.seg, axis=1)
        self.arc = np.r_[0.0, np.cumsum(self.length)]
        self.s = 0.0
        self.k = 0

    @property
    def total(self):
        return float(self.arc[-1])

    def update(self, xy):
        """Project the walker onto the next segments (never backwards along the path)."""
        best, bk, bs = math.inf, self.k, self.s
        for k in range(self.k, min(self.k + 8, len(self.seg))):
            L = self.length[k]
            u = 0.0 if L < 1e-9 else min(max(float(np.dot(xy - self.p[k], self.seg[k])) / (L * L), 0.0), 1.0)
            d = float(np.linalg.norm(self.p[k] + u * self.seg[k] - xy))
            if d < best - 1e-9:
                best, bk, bs = d, k, float(self.arc[k] + u * L)
        if bs >= self.s:
            self.k, self.s = bk, bs
        return best

    def at(self, s):
        s = min(max(s, 0.0), self.total)
        k = min(int(np.searchsorted(self.arc, s, side="right")) - 1, len(self.seg) - 1)
        L = self.length[k]
        return self.p[k] + (0.0 if L < 1e-9 else (s - self.arc[k]) / L) * self.seg[k]

    def desire(self, xy, speed, vmax, yaw=None):
        """Desired heading, speed and the largest speed that does not pass the goal within one tick.

        Also sets `anchor` (the walker's projection on the path), `tangent` (the path direction there) and
        `lane`: the walker keeps its present lateral offset from the path (people do not all walk on the route's
        centre line), within LANE_M and narrowing to zero over the last metres so that it arrives at the goal.
        These are the frame of the lateral-offset candidates.

        With `yaw`, the speed also stays low enough to reach the goal within the turn rate (no circling)."""
        self.update(xy)
        goal = self.p[-1]
        to_goal = float(np.linalg.norm(goal - xy))
        remaining = max(self.total - self.s, to_goal)
        look = min(max(0.6 + 0.6 * speed, 0.6), 1.5)
        anchor = self.at(self.s)
        ahead = self.at(self.s + max(look, 1.0)) - anchor
        lane, normal = 0.0, np.zeros(2)
        if np.linalg.norm(ahead) >= 1e-6:
            normal = np.array([-ahead[1], ahead[0]]) / np.linalg.norm(ahead)
            band = min(LANE_M, max(0.0, 0.5 * (remaining - 1.0)))
            lane = min(max(float(np.dot(xy - anchor, normal)), -band), band)
        target = goal if remaining <= look else self.at(self.s + look) + lane * normal
        heading = math.atan2(target[1] - xy[1], target[0] - xy[0]) if np.linalg.norm(target - xy) > 1e-6 else None
        self.anchor, self.lane = anchor, lane
        self.tangent = heading if np.linalg.norm(ahead) < 1e-6 or heading is None else math.atan2(ahead[1], ahead[0])
        want = min(vmax, math.sqrt(2.0 * 1.2 * max(remaining - 0.02, 0.0)))
        # Corners within the stopping reach: at most 1.4 rad/s of heading change through a bend.
        j = int(np.searchsorted(self.arc, self.s, side="right"))
        while 0 < j < len(self.p) - 1 and self.arc[j] - self.s <= 3.0:
            a, b = self.seg[j - 1], self.seg[j]
            turn = abs(math.atan2(a[0] * b[1] - a[1] * b[0], float(np.dot(a, b))))
            if turn > 0.05:
                bend = max(min(self.length[j - 1], self.length[j], 1.0) / turn * 1.4, CREEP_MPS * 2)
                want = min(want, math.sqrt(bend**2 + 2.0 * 1.2 * max(self.arc[j] - self.s, 0.0)))
            j += 1
        if yaw is not None and to_goal > 1e-6:
            # the circle through the goal tangent to the present heading must be within the turn rate
            err = abs((math.atan2(goal[1] - xy[1], goal[0] - xy[0]) - yaw + math.pi) % (2 * math.pi) - math.pi)
            if err > 1e-3:
                want = min(want, 0.8 * TURN_RATE_RPS * to_goal / (2.0 * math.sin(min(err, math.pi / 2))))
        return heading, want, to_goal / DT


class PathStore:
    """The current walking paths of all walkers in flat arrays, followed by the compiled `kernels.desire_all`
    (the arithmetic of Path.desire). `Path` remains the per-path view used for geometry queries."""

    def __init__(self, n, capacity=1 << 16):
        self.pts = np.zeros((capacity, 2))
        self.arc = np.zeros(capacity)
        self.used = 0
        self.first = np.full(n, -1, np.int64)
        self.count = np.zeros(n, np.int64)
        self.s = np.zeros(n)
        self.k = np.zeros(n, np.int64)

    def __contains__(self, i):
        return self.first[i] >= 0

    def __len__(self):
        return int((self.first >= 0).sum())

    def pop(self, i):
        self.first[i] = -1

    def set(self, i, points):
        q = Path(points)  # same de-duplication as Path
        m = len(q.p)
        if self.used + m > len(self.pts):
            self._compact(m)
        f = self.used
        self.pts[f : f + m] = q.p
        self.arc[f : f + m] = q.arc
        self.first[i], self.count[i], self.s[i], self.k[i] = f, m, 0.0, 0
        self.used += m

    def _compact(self, extra):
        live = np.flatnonzero(self.first >= 0)
        need = int(self.count[live].sum()) + extra
        cap = len(self.pts)
        while cap < 2 * need:
            cap *= 2
        pts, arc = np.zeros((cap, 2)), np.zeros(cap)
        u = 0
        for i in live:
            f, m = self.first[i], self.count[i]
            pts[u : u + m] = self.pts[f : f + m]
            arc[u : u + m] = self.arc[f : f + m]
            self.first[i] = u
            u += m
        self.pts, self.arc, self.used = pts, arc, u

    def remaining(self, i):
        """Length of walker i's path ahead of its progress."""
        f, m = self.first[i], self.count[i]
        return float(self.arc[f + m - 1] - self.s[i])

    def end(self, i):
        f, m = self.first[i], self.count[i]
        return self.pts[f + m - 1]

    def path(self, i):
        """A Path view of walker i's path with its progress."""
        f, m = self.first[i], self.count[i]
        q = Path(self.pts[f : f + m].copy())
        q.s, q.k = float(self.s[i]), int(self.k[i])
        return q

    def desire(self, ids, xy, speed, vmax, yaw):
        """(heading (NaN: none), wanted speed, goal cap, anchor (n, 2), tangent, lane) for `ids`; other rows unset."""
        n = len(self.first)
        heading, want, cap, ax, ay, tangent, lane = (np.full(n, np.nan) for _ in range(7))
        K.desire_all(
            np.asarray(ids, np.int64),
            self.pts,
            self.arc,
            self.first,
            self.count,
            self.s,
            self.k,
            np.ascontiguousarray(xy[:, 0]),
            np.ascontiguousarray(xy[:, 1]),
            np.asarray(speed, float),
            np.asarray(vmax, float),
            np.asarray(yaw, float),
            DT,
            CREEP_MPS,
            LANE_M,
            TURN_RATE_RPS,
            heading,
            want,
            cap,
            ax,
            ay,
            tangent,
            lane,
        )
        return heading, want, cap, np.stack([ax, ay], 1), tangent, lane


def legal_cells(area, surface, inset, step=0.25):
    """Cells (step m) whose centre lies in the walking area shrunk by `inset`: where a walker's centre may be.

    The planning network (WalkNetwork.free) keeps the centre a body radius plus a grid allowance from the area
    boundary; local motion instead checks the support of the body box at every step, so its centres need only
    lie in the area itself."""
    import shapely

    inner = area.buffer(-inset)
    shapely.prepare(inner)
    nx, ny = int(math.ceil(surface.width_m / step)), int(math.ceil(surface.height_m / step))
    x0, y0 = float(surface.x0) + step / 2, float(surface.y0) + step / 2
    xs = x0 + np.arange(nx) * step
    free = np.zeros((ny, nx), bool)
    for r in range(ny):
        free[r] = shapely.contains_xy(inner, xs, np.full(nx, y0 + r * step))
    return free, (x0, y0), step


class Crowd:
    def __init__(self, ext, kinds, surface, walking, *, grid_cell=4.0):
        self.ext = np.ascontiguousarray(ext, np.float64)
        corner = np.maximum(np.abs(self.ext[:, :, :2]).max(-1), 0.0)
        side = np.abs(self.ext[:, :, 2:]).max(-1)
        self.radius = np.hypot(corner, side).max(1)
        self.kinds = np.asarray(kinds)
        self.z_ok = np.isfinite(surface.z)
        self.walk = np.asarray(surface.walk["person"], bool)
        from scipy.ndimage import distance_transform_edt

        ok = np.pad(self.walk & self.z_ok, 1)
        self.sdist = np.minimum(distance_transform_edt(ok)[1:-1, 1:-1], 255).astype(np.uint8)  # cells to unsupported
        self.sx0, self.sy0, self.scell = float(surface.x0), float(surface.y0), float(surface.cell_m)
        area = getattr(walking, "area", None)
        if area is None:
            self.free = np.asarray(walking.free, bool)
            self.fx0, self.fy0 = (float(v) for v in walking.origin)
            self.fstep = float(walking.step)
        else:
            self.free, (self.fx0, self.fy0), self.fstep = legal_cells(area, surface, LEGAL_INSET_M)
        self.grid = (
            float(surface.x0) - 50.0,
            float(surface.y0) - 50.0,
            float(grid_cell),
            int(math.ceil((surface.width_m + 100.0) / grid_cell)),
            int(math.ceil((surface.height_m + 100.0) / grid_cell)),
        )
        self.offsets = np.array(
            [0.0, -0.12, 0.12, -0.25, 0.25, -0.4, 0.4, -0.6, 0.6, -0.9, 0.9, -1.4, 1.4]
        )  # from the lane (m), + = left
        self.speed_fracs = np.array([1.0, 0.6, 0.3])
        self.times = np.r_[
            np.arange(1, 11) * 0.1, 1.0 + np.arange(1, 7) * 0.25
        ]  # prediction samples (s): 0.1 s to 1 s, then 0.25 s to 2.5 s
        self.horizon = float(self.times[-1])
        self.margin, self.margin_growth = 0.03, 0.02
        self.waited = np.zeros(len(self.ext))  # seconds a walker has wanted to walk but stood
        self.walked = np.zeros(len(self.ext))  # seconds a walker has walked without stopping since it last stood
        self.motion = np.full(len(self.ext), np.nan)  # direction of an edging step (NaN: along the heading)
        self.aim = np.zeros(len(self.ext))  # heading each walker chose in the last step
        self.weights = (2.0, 0.75, 0.75, 2.5)  # desired, current, side, time to contact (DetourCrowd defaults)
        self.stats = dict(
            emergency_brakes=0,
            conflict_passes=0,
            shortened=0,
            turn_cancelled=0,
            last_resort=0,
            holds=0,
            unresolved=0,
            rescued=0,
            support_shortened=0,
            support_rescued=0,
            support_holds=0,
            budget_spent=0,
            yield_unsupported=0,
            restored=0,
        )

    def step(
        self,
        xy,
        yaw,
        speed,
        active,
        mover,
        des_yaw,
        des_speed,
        vmax,
        goal_cap,
        anchor=None,
        tangent=None,
        lane=None,
        driven=None,
        seen_standing=None,
    ):
        """New (xy, yaw, speed) for the movers; others keep their pose. Returns also the bodies held beyond limits.

        `anchor` (n, 2), `tangent` and `lane` (n) give each walker's path frame and kept lateral offset (Path.anchor,
        Path.tangent, Path.lane); without a path the frame is the walker's position and desired heading.

        `driven` = (ids, new_xy, new_yaw, new_speed, pred_x, pred_y, pred_yaw) gives bodies moved by another
        authority this tick (driving vehicles: awr.life.motion.vehicles): their new poses are fixed (walkers give way to
        them in the safety pass) and walkers predict them along their routes (rows of pred_*, at `self.times`).
        `seen_standing` (n, -1 for none): the body each walker predicts where it stands instead (a car about to set
        off whose way the walker is already in: it leaves that way as from a standing car)."""
        n = len(xy)
        px, py = np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1])
        vx, vy = speed * np.cos(yaw), speed * np.sin(yaw)
        # A walker whose desired heading is far off turns on the spot first, unless it has waited (a blocked turn):
        # then its candidates include edging off along its present heading.
        turn_first = (
            mover
            & (des_speed > 0.0)
            & (np.abs(K_wrap(des_yaw - yaw)) > TURN_FIRST_RAD)
            & (speed < TURN_FIRST_BELOW_MPS)
            & (self.waited < PATIENCE_S)
        )
        wanted = des_speed
        des_speed = np.where(turn_first, 0.0, des_speed)
        gx0, gy0, gc, gnx, gny = self.grid
        order, start = K.build_grid(px, py, active, gx0, gy0, gc, gnx, gny)
        walkers = np.flatnonzero(
            mover & active & ((des_speed > 0.0) | (speed > 0.0) | (np.abs(K_wrap(des_yaw - yaw)) > 1e-3))
        )
        new_xy, new_yaw, new_speed = xy.copy(), yaw.copy(), speed.copy()
        if driven is not None:
            ids = np.asarray(driven[0], np.int64)
            if np.any(mover[ids]):
                raise ValueError("A driven body cannot also be a walker")
            new_xy[ids], new_yaw[ids], new_speed[ids] = driven[1], driven[2], driven[3]
        if not len(walkers):
            self.waited[:] = 0.0
            self.walked[:] = 0.0
            self.choice = (walkers, np.zeros(0), np.zeros(0), np.zeros(0))
            return new_xy, new_yaw, new_speed, []
        reach = float(vmax.max()) * self.horizon + 2.0 * float(self.radius.max()) + 1.0
        w_des, w_cur, w_side, w_toi = self.weights
        anchor = xy if anchor is None else anchor
        tangent = des_yaw if tangent is None else tangent
        lane = np.zeros(n) if lane is None else np.nan_to_num(np.asarray(lane, np.float64))
        # Neighbours keep turning towards the heading they chose last tick (a turn or sidestep in progress).
        pred = K.predict_bodies(px, py, yaw, speed, self.aim, mover & active & (speed > 0.0), self.times, TURN_RATE_RPS)
        if driven is not None and len(driven[0]):
            for k, arr in enumerate(pred):
                arr[np.asarray(driven[0], np.int64)] = driven[4 + k]
        h, s, tt, md = K.choose_velocities(
            walkers,
            px,
            py,
            yaw,
            speed,
            vx,
            vy,
            self.ext,
            self.radius,
            active,
            mover & (self.kinds == "person"),
            des_yaw,
            des_speed,
            vmax,
            np.ascontiguousarray(anchor[:, 0], np.float64),
            np.ascontiguousarray(anchor[:, 1], np.float64),
            np.asarray(tangent, np.float64),
            lane,
            order,
            start,
            gx0,
            gy0,
            gc,
            gnx,
            gny,
            self.z_ok,
            self.walk,
            self.sx0,
            self.sy0,
            self.scell,
            self.sdist,
            self.free,
            self.fx0,
            self.fy0,
            self.fstep,
            self.offsets,
            self.speed_fracs,
            SHUFFLE_MPS,
            SHUFFLE_ROOM_M,
            VEHICLE_ROOM_M,
            self.times,
            TURN_RATE_RPS,
            ACCEL_MPS2,
            2.5,
            self.margin,
            self.margin_growth,
            w_des,
            w_cur,
            w_side,
            w_toi,
            reach,
            np.asarray(goal_cap, np.float64) * DT,
            PERSONAL_M * np.clip(1.0 - self.waited / PATIENCE_S, 0.0, 1.0),
            WAIT_PENALTY * np.clip(self.waited / PATIENCE_S, 0.0, 1.0),
            np.where(self.waited >= NUDGE_AFTER_S, NUDGE_MPS, 0.0),
            self.waited >= PATIENCE_S,
            self.aim,
            *pred,
            np.full(n, -1, np.int64) if seen_standing is None else np.asarray(seen_standing, np.int64),
        )
        self.aim = np.array(yaw, np.float64)
        self.aim[walkers] = h
        self.choice = (walkers, h, s, tt)  # the selected heading, speed and predicted time to contact (diagnostics)
        self.motion[:] = np.nan
        self.motion[walkers] = md
        floor = np.maximum(speed - EMERGENCY_DECEL_MPS2 * DT, 0.0)
        last = np.maximum(speed - LAST_RESORT_DECEL_MPS2 * DT, 0.0)
        for w, i in enumerate(walkers):
            turn = K.wrap(h[w] - yaw[i])
            new_yaw[i] = yaw[i] + min(max(turn, -TURN_RATE_RPS * DT), TURN_RATE_RPS * DT)
            decel = COMFORT_DECEL_MPS2
            if s[w] < speed[i] and tt[w] < self.horizon and tt[w] < speed[i] / COMFORT_DECEL_MPS2 + 0.5:
                decel = EMERGENCY_DECEL_MPS2
                self.stats["emergency_brakes"] += 1
            v = min(max(s[w], speed[i] - decel * DT), speed[i] + ACCEL_MPS2 * DT)
            new_speed[i] = max(min(v, goal_cap[i]), floor[i])
            if md[w] == md[w]:  # edging aside: the step's speed is the edging speed, the heading is kept
                new_yaw[i] = yaw[i]
                new_speed[i] = min(new_speed[i], NUDGE_MPS)
            self._place(i, xy, new_xy, new_yaw, new_speed)
        sw = np.maximum(speed, new_speed) + 0.01
        held = []
        for i in walkers:
            if self._supported(i, xy, yaw, new_xy, new_yaw, sw):
                continue
            ok = False
            for yw in (new_yaw[i], yaw[i]):
                for v in np.linspace(new_speed[i], floor[i], 5):
                    new_yaw[i], new_speed[i] = yw, v
                    self._place(i, xy, new_xy, new_yaw, new_speed)
                    sw[i] = max(speed[i], v) + 0.01
                    if self._supported(i, xy, yaw, new_xy, new_yaw, sw):
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                ok = self._rescue(
                    (i,), xy, yaw, speed, new_xy, new_yaw, new_speed, sw, mover, active, check_others=False
                )
                if ok:
                    self.stats["support_rescued"] += 1
            self.stats["support_shortened" if ok else "support_holds"] += 1
            if not ok:
                new_yaw[i], new_speed[i] = yaw[i], 0.0
                self._place(i, xy, new_xy, new_yaw, new_speed)
                sw[i] = speed[i] + 0.01
                if speed[i] - EMERGENCY_DECEL_MPS2 * DT > 1e-9:
                    held.append(int(i))
        changed = walkers
        planned = {int(i): (float(new_yaw[i]), float(new_speed[i]), float(self.motion[i])) for i in walkers}
        reach = 2.0 * float(self.radius.max()) + 2.0 * float(max(vmax.max(), 1.0)) * DT + 0.2
        for _ in range(6 * len(walkers) + 24):
            order, start = K.build_grid(
                np.ascontiguousarray(new_xy[:, 0]), np.ascontiguousarray(new_xy[:, 1]), active, gx0, gy0, gc, gnx, gny
            )
            pairs = K.swept_conflicts(
                np.asarray(changed, np.int64),
                px,
                py,
                yaw,
                np.ascontiguousarray(new_xy[:, 0]),
                np.ascontiguousarray(new_xy[:, 1]),
                new_yaw,
                sw,
                self.ext,
                self.radius,
                active,
                order,
                start,
                gx0,
                gy0,
                gc,
                gnx,
                gny,
                reach,
                -K.CONTACT_CLEARANCE_M,
                6,
            )
            if not len(pairs):
                break
            self.stats["conflict_passes"] += 1
            nxt = set()
            for i, j in sorted({(min(a, b), max(a, b)) for a, b in pairs.tolist()}):
                y = self._yielder(i, j, xy, yaw, speed, new_xy, new_yaw, new_speed, mover, floor, self.motion)
                if y is not None:
                    # The yielder's step: shorter, else without its turn; it must stay supported (the shorter step
                    # at either heading, down to the floor speed). Without a supported yield the pair is resolved
                    # as one without a yielder.
                    before = (new_yaw[y], new_speed[y])
                    if new_speed[y] > floor[y] + 1e-12:
                        new_speed[y] = max(new_speed[y] - COMFORT_DECEL_MPS2 * DT, floor[y])
                        kind = "shortened"
                    else:
                        new_yaw[y] = yaw[y]
                        kind = "turn_cancelled"
                    if self._keep_supported(
                        y,
                        xy,
                        yaw,
                        speed,
                        new_xy,
                        new_yaw,
                        new_speed,
                        sw,
                        np.linspace(new_speed[y], floor[y], 5),
                        (new_yaw[y], yaw[y]),
                    ):
                        self.stats[kind] += 1
                        nxt.add(y)
                        continue
                    new_yaw[y], new_speed[y] = before
                    self._place(y, xy, new_xy, new_yaw, new_speed)
                    sw[y] = max(speed[y], new_speed[y]) + 0.01
                    self.stats["yield_unsupported"] += 1
                # Last resort before a hold: brake straight (no turn) at up to LAST_RESORT_DECEL_MPS2; the turn
                # is kept where the straight step is not supported.
                lowered = [b for b in (i, j) if mover[b] and floor[b] > last[b] + 1e-12]
                if lowered:
                    for b in lowered:
                        floor[b] = last[b]
                        self._keep_supported(
                            b, xy, yaw, speed, new_xy, new_yaw, new_speed, sw, (new_speed[b],), (yaw[b], new_yaw[b])
                        )
                        nxt.add(b)
                    self.stats["last_resort"] += len(lowered)
                    continue
                if self._rescue((i, j), xy, yaw, speed, new_xy, new_yaw, new_speed, sw, mover, active):
                    self.stats["rescued"] += 1
                    nxt.update((i, j))
                    continue
                self.stats["unresolved"] += 1
                for b in (i, j):
                    if mover[b] and (new_speed[b] > 0.0 or abs(K.wrap(new_yaw[b] - yaw[b])) > 1e-12):
                        new_yaw[b], new_speed[b] = yaw[b], 0.0
                        self._place(b, xy, new_xy, new_yaw, new_speed)
                        sw[b] = speed[b] + 0.01
                        held.append(int(b))
                        nxt.add(b)
            changed = sorted(nxt)
            if not changed:
                break
        else:
            planned = {}
            # The pass budget is spent with conflicts left: resolve them as unresolved pairs (rescue, else hold),
            # never accept a conflicting step.
            order, start = K.build_grid(
                np.ascontiguousarray(new_xy[:, 0]), np.ascontiguousarray(new_xy[:, 1]), active, gx0, gy0, gc, gnx, gny
            )
            pairs = K.swept_conflicts(
                np.asarray(walkers, np.int64),
                px,
                py,
                yaw,
                np.ascontiguousarray(new_xy[:, 0]),
                np.ascontiguousarray(new_xy[:, 1]),
                new_yaw,
                sw,
                self.ext,
                self.radius,
                active,
                order,
                start,
                gx0,
                gy0,
                gc,
                gnx,
                gny,
                reach,
                -0.01,
                6,
            )
            for i, j in sorted({(min(a, b), max(a, b)) for a, b in pairs.tolist()}):
                self.stats["budget_spent"] += 1
                if self._rescue((i, j), xy, yaw, speed, new_xy, new_yaw, new_speed, sw, mover, active):
                    self.stats["rescued"] += 1
                    continue
                for b in (i, j):
                    if mover[b] and (new_speed[b] > 0.0 or abs(K.wrap(new_yaw[b] - yaw[b])) > 1e-12):
                        new_yaw[b], new_speed[b] = yaw[b], 0.0
                        self._place(b, xy, new_xy, new_yaw, new_speed)
                        sw[b] = speed[b] + 0.01
                        if speed[b] - EMERGENCY_DECEL_MPS2 * DT > 1e-9:
                            held.append(int(b))
        if planned:
            self._restore(planned, held, xy, yaw, speed, new_xy, new_yaw, new_speed, sw, active, reach)
        self.stats["holds"] += len(held)
        # A walker that edges forward a centimetre and stands again, or steps aside from an obstacle, is still
        # waiting: its wait ends only after RESUMED_S of walking on its way (otherwise the waiting cost never builds
        # up in front of an obstacle it must walk round, and a sidestep is undone by turning back to the way).
        # A requested slow approach is progress at that speed, not a wait.
        progress = np.minimum(PROGRESS_MPS, wanted)
        on_way = (new_speed >= progress - 1e-9) & (np.abs(K_wrap(new_yaw - des_yaw)) <= PROGRESS_RAD)
        self.walked = np.where(on_way, self.walked + DT, 0.0)
        waiting = (wanted > 0.0) & ~on_way
        self.waited = np.where(
            waiting, self.waited + DT, np.where((wanted > 0.0) & (self.walked < RESUMED_S - 1e-9), self.waited, 0.0)
        )
        return new_xy, K_wrap(new_yaw), new_speed, sorted(set(held))

    def _restore(self, planned, held, xy, yaw, speed, new_xy, new_yaw, new_speed, sw, active, reach):
        """After the safety pass: give back, in index order, each reduced step that is clear of every body's final
        pose and supported. Yielding is greedy (of a conflicting pair one body gives way, then the other may too); a
        step that had to give way only to a move that was later cut is restored, so that two walkers do not both
        stand where one of them could have gone."""
        gx0, gy0, gc, gnx, gny = self.grid
        pxa, pya = np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1])
        for b in sorted(planned):
            h, v, m = planned[b]
            if b in held or (new_speed[b] == v and new_yaw[b] == h):
                continue
            keep = (new_xy[b].copy(), float(new_yaw[b]), float(new_speed[b]), float(sw[b]), float(self.motion[b]))
            new_yaw[b], new_speed[b], self.motion[b] = h, v, m
            self._place(b, xy, new_xy, new_yaw, new_speed)
            sw[b] = max(speed[b], v) + 0.01
            order, start = K.build_grid(
                np.ascontiguousarray(new_xy[:, 0]), np.ascontiguousarray(new_xy[:, 1]), active, gx0, gy0, gc, gnx, gny
            )
            if self._supported(b, xy, yaw, new_xy, new_yaw, sw) and not len(
                K.swept_conflicts(
                    np.array([b], np.int64),
                    pxa,
                    pya,
                    yaw,
                    np.ascontiguousarray(new_xy[:, 0]),
                    np.ascontiguousarray(new_xy[:, 1]),
                    new_yaw,
                    sw,
                    self.ext,
                    self.radius,
                    active,
                    order,
                    start,
                    gx0,
                    gy0,
                    gc,
                    gnx,
                    gny,
                    reach,
                    -0.01,
                    6,
                )
            ):
                self.stats["restored"] += 1
                continue
            new_xy[b], new_yaw[b], new_speed[b], sw[b], self.motion[b] = keep

    def _rescue(self, bodies, xy, yaw, speed, new_xy, new_yaw, new_speed, sw, mover, active, check_others=True):
        """Before a hold: search the full dynamic window of the given walkers (one, or a conflicting pair) for a
        joint step that keeps each supported, clear of the others and of each other: speeds from the last-resort
        floor up to the acceleration limit, headings within the turn rate (6 x RESCUE_HEADINGS steps each, 0.01 rad
        apart: beside an edge of the support the supported turns may span only a few degrees). The step closest to
        keeping speed and heading wins (ties: the first in a fixed order). Applies it and returns True, or leaves
        the poses unchanged and returns False."""
        keep = {b: (new_xy[b].copy(), float(new_yaw[b]), float(new_speed[b]), float(sw[b])) for b in bodies}
        motion = {b: float(self.motion[b]) for b in bodies}
        for b in bodies:
            if mover[b]:
                self.motion[b] = np.nan  # the window is searched for ordinary steps along the heading
        pxa, pya = np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1])
        if check_others:
            gx0, gy0, gc, gnx, gny = self.grid
            order, start = K.build_grid(
                np.ascontiguousarray(new_xy[:, 0]), np.ascontiguousarray(new_xy[:, 1]), active, gx0, gy0, gc, gnx, gny
            )
            reach = 2.0 * float(self.radius.max()) + 2.0 * max(float(speed.max()) + ACCEL_MPS2 * DT, 1.0) * DT + 0.3
        options = {}
        for b in bodies:
            if not mover[b]:
                options[b] = [(0.0, float(new_yaw[b]), float(new_speed[b]))]
                continue
            found = []
            for v in np.linspace(max(speed[b] - LAST_RESORT_DECEL_MPS2 * DT, 0.0), speed[b] + ACCEL_MPS2 * DT, 6):
                for dh in np.linspace(-TURN_RATE_RPS * DT, TURN_RATE_RPS * DT, RESCUE_HEADINGS):
                    new_yaw[b], new_speed[b] = yaw[b] + dh, v
                    self._place(b, xy, new_xy, new_yaw, new_speed)
                    sw[b] = max(speed[b], v) + 0.01
                    if not self._supported(b, xy, yaw, new_xy, new_yaw, sw):
                        continue
                    if check_others:
                        pairs = K.swept_conflicts(
                            np.array([b], np.int64),
                            pxa,
                            pya,
                            yaw,
                            np.ascontiguousarray(new_xy[:, 0]),
                            np.ascontiguousarray(new_xy[:, 1]),
                            new_yaw,
                            sw,
                            self.ext,
                            self.radius,
                            active,
                            order,
                            start,
                            gx0,
                            gy0,
                            gc,
                            gnx,
                            gny,
                            reach,
                            -K.CONTACT_CLEARANCE_M,
                            6,
                        )
                        if any(int(q) not in bodies for q in pairs[:, 1]):
                            continue
                    found.append((abs(v - speed[b]) + 0.5 * abs(dh), float(new_yaw[b]), float(v)))
            new_xy[b], new_yaw[b], new_speed[b], sw[b] = keep[b]
            options[b] = sorted(found)
        best = None
        if len(bodies) == 1:
            best = options[bodies[0]][:1] and [options[bodies[0]][0]]
        else:
            i, j = bodies
            for ci, hi, vi in options[i]:
                for cj, hj, vj in options[j]:
                    if best is not None and ci + cj >= best[0]:
                        break
                    for b, h, v in ((i, hi, vi), (j, hj, vj)):
                        new_yaw[b], new_speed[b] = h, v
                        self._place(b, xy, new_xy, new_yaw, new_speed)
                        sw[b] = max(speed[b], v) + 0.01
                    if not K.pair_swept(
                        pxa,
                        pya,
                        yaw,
                        np.ascontiguousarray(new_xy[:, 0]),
                        np.ascontiguousarray(new_xy[:, 1]),
                        new_yaw,
                        sw,
                        self.ext,
                        i,
                        j,
                        -K.CONTACT_CLEARANCE_M,
                        6,
                    ):
                        best = (ci + cj, (ci, hi, vi), (cj, hj, vj))
            best = None if best is None else list(best[1:])
        if not best:
            for b in bodies:
                new_xy[b], new_yaw[b], new_speed[b], sw[b] = keep[b]
                self.motion[b] = motion[b]
            return False
        for b, (_, h, v) in zip(bodies, best, strict=False):
            new_yaw[b], new_speed[b] = h, v
            self._place(b, xy, new_xy, new_yaw, new_speed)
            sw[b] = max(speed[b], v) + 0.01
        return True

    def _place(self, i, xy, new_xy, new_yaw, new_speed):
        a = self.motion[i] if self.motion[i] == self.motion[i] else new_yaw[i]
        new_xy[i] = xy[i] + new_speed[i] * DT * np.array([math.cos(a), math.sin(a)])

    def _keep_supported(self, b, xy, yaw, speed, new_xy, new_yaw, new_speed, sw, speeds, yaws):
        """Give `b` the first supported step of `yaws` x `speeds` (in that order); unchanged and False if none is."""
        before = (new_yaw[b], new_speed[b])
        for yw in yaws:
            for v in speeds:
                new_yaw[b], new_speed[b] = yw, v
                self._place(b, xy, new_xy, new_yaw, new_speed)
                sw[b] = max(speed[b], v) + 0.01
                if self._supported(b, xy, yaw, new_xy, new_yaw, sw):
                    return True
        new_yaw[b], new_speed[b] = before
        self._place(b, xy, new_xy, new_yaw, new_speed)
        sw[b] = max(speed[b], new_speed[b]) + 0.01
        return False

    def _supported(self, i, xy, yaw, new_xy, new_yaw, sw):
        """The step's swept poses with the box of speed `sw`, and the new pose as the tape records it (mm, centidegrees
        and the velocity in cm/s, awr.life.civil.assemble.recorded_pose) with the box of that speed: the corner test is
        not monotone in the box (a corner of a nested box can lie within a cell of a hole that the larger box's corners
        clear), and a corner within a millimetre of the support's edge can leave it when the pose is recorded, so the
        box the tape QA draws is tested as well."""
        rx, ry, ryaw, rv = recorded_pose(
            new_xy[i, 0], new_xy[i, 1], new_yaw[i], (new_xy[i, 0] - xy[i, 0]) / DT, (new_xy[i, 1] - xy[i, 1]) / DT
        )
        return K.swept_supported(
            i,
            float(xy[i, 0]),
            float(xy[i, 1]),
            float(yaw[i]),
            float(new_xy[i, 0]),
            float(new_xy[i, 1]),
            float(new_yaw[i]),
            float(sw[i]),
            self.ext,
            self.z_ok,
            self.walk,
            self.sx0,
            self.sy0,
            self.scell,
            self.free,
            self.fx0,
            self.fy0,
            self.fstep,
            6,
        ) and K.body_supported(rx, ry, ryaw, rv, self.ext[i], self.z_ok, self.walk, self.sx0, self.sy0, self.scell)

    @staticmethod
    def _yielder(i, j, xy, yaw, speed, new_xy, new_yaw, new_speed, mover, floor, motion):
        """The body that gives way: one that can still shorten its step (or undo its turn) within the limits;
        of two, not one edging aside (its step opens a gap), then the one closing the gap; head-on, the slower; then
        the higher index."""

        def can(b):
            return bool(mover[b]) and (new_speed[b] > floor[b] + 1e-12 or abs(K.wrap(new_yaw[b] - yaw[b])) > 1e-12)

        ci, cj = can(i), can(j)
        if ci != cj:
            return i if ci else j
        if not ci:
            return None
        ei, ej = motion[i] == motion[i], motion[j] == motion[j]
        if ei != ej:
            return j if ei else i

        def closing(a, b):
            d = xy[b] - xy[a]
            dn = float(np.linalg.norm(d))
            return float(np.dot(new_xy[a] - xy[a], d / dn)) if dn > 1e-9 else float(new_speed[a])

        a, b = closing(i, j), closing(j, i)
        if abs(a - b) > 0.005:
            return i if a > b else j
        if abs(speed[i] - speed[j]) > 1e-9:
            return i if speed[i] < speed[j] else j
        return max(i, j)


def K_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
