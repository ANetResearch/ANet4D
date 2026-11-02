"""Social-force crowd (Helbing-style) on the pedestrian grid, vectorised over agents.

Forces: goal attraction (v0·e − v)/τ, agent–agent repulsion A·exp((r_i + r_j − d)/B) with anisotropy, obstacle
repulsion from the grid distance field, and repulsion from moving vehicles. Positions that would leave the walkable
area are projected back (agents slide along walls). Standing agents keep their anchor point as goal with v0 = 0.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

TAU = 0.5
A_AG, B_AG, LAMBDA = 2.2, 0.32, 0.35
A_VEH, B_VEH = 4.0, 1.0
V_MAX_FACTOR = 1.15


class Crowd:
    def __init__(self, grid, n: int) -> None:
        self.g = grid
        self.n = n
        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.goal = np.zeros((n, 2))
        self.v0 = np.zeros(n)  # desired speed (0 = standing)
        self.r = np.full(n, 0.28)
        self.active = np.zeros(n, bool)  # simulated (on foot, visible)
        self.yaw = np.zeros(n)
        self.reached = np.zeros(n, bool)  # within tolerance of goal this step
        self.tol = np.full(n, 0.7)
        self.road_ok = np.zeros(n, bool)  # True while an agent is legitimately crossing / riding on the carriageway

    def place(self, i: int, xy, goal=None, v0: float = 0.0) -> None:
        x, y = self.g.snap(float(xy[0]), float(xy[1]), r=0.75)
        other = np.flatnonzero(self.active & (np.arange(self.n) != i))
        if len(other) and np.any(np.linalg.norm(self.pos[other] - [x, y], axis=1) < self.r[other] + self.r[i]):
            raise ValueError(f"Spawn overlaps an existing person: {i} at {(x, y)}")
        self.pos[i] = (x, y)
        self.vel[i] = 0.0
        self.goal[i] = (x, y) if goal is None else goal
        self.v0[i] = v0
        self.active[i] = True

    def deactivate(self, i: int) -> None:
        self.active[i] = False
        self.vel[i] = 0.0
        self.v0[i] = 0.0

    def step(
        self,
        dt: float,
        veh_xy: np.ndarray | None = None,
        veh_v: np.ndarray | None = None,
        veh_r: np.ndarray | None = None,
        veh_h: np.ndarray | None = None,
    ) -> None:
        act = np.flatnonzero(self.active)
        if len(act) == 0:
            return
        P, V = self.pos[act].copy(), self.vel[act].copy()
        P0 = P.copy()  # positions before any push (always walkable)
        d = self.goal[act] - P
        dist = np.linalg.norm(d, axis=1)
        e = d / (dist[:, None] + 1e-9)
        v0 = self.v0[act].copy()
        slow = dist < 1.5  # ease into the goal
        v0 = np.where(slow, np.minimum(v0, np.maximum(dist / 1.2, 0.0) + 0.15), v0)
        F = (v0[:, None] * e - V) / TAU
        # agent–agent
        if len(act) > 1:
            tree = cKDTree(P)
            pairs = tree.query_pairs(r=2.2, output_type="ndarray")
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                dij = P[i] - P[j]
                dn = np.linalg.norm(dij, axis=1) + 1e-9
                nij = dij / dn[:, None]
                rr = self.r[act][i] + self.r[act][j]
                mag = A_AG * np.exp((rr - dn) / B_AG)
                # anisotropy: weaker response to agents behind
                ei = V[i] / (np.linalg.norm(V[i], axis=1, keepdims=True) + 1e-9)
                ej = V[j] / (np.linalg.norm(V[j], axis=1, keepdims=True) + 1e-9)
                ci = np.einsum("ij,ij->i", ei, -nij)
                cj = np.einsum("ij,ij->i", ej, nij)
                wi = LAMBDA + (1 - LAMBDA) * (1 + ci) / 2
                wj = LAMBDA + (1 - LAMBDA) * (1 + cj) / 2
                fi = nij * (mag * wi)[:, None]
                fj = -nij * (mag * wj)[:, None]
                np.add.at(F, i, fi)
                np.add.at(F, j, fj)
        # obstacles, and the carriageway edge for agents that are not crossing
        F += self.g.obstacle_force(P, self.r[act].mean())
        nr = ~self.road_ok[act]
        if nr.any():
            F[nr] += self.g.road_force(P[nr])
        # vehicles
        if veh_xy is not None and len(veh_xy):
            # repulsion from the vehicle body and from the point it reaches in 1 s (moving vehicles)
            pts = [veh_xy]
            rad = [veh_r]
            if veh_h is not None and veh_v is not None:
                pts.append(veh_xy + veh_h * veh_v[:, None] * 1.0)
                rad.append(veh_r)
            Q = np.concatenate(pts)
            R = np.concatenate(rad)
            vt = cKDTree(Q)
            lists = vt.query_ball_point(P, r=10.0)
            for k, lst in enumerate(lists):
                for vi in lst:
                    dv = P[k] - Q[vi]
                    dn = np.linalg.norm(dv) + 1e-9
                    gap = dn - R[vi] - self.r[act][k]
                    if gap < 4.0:
                        F[k] += dv / dn * A_VEH * np.exp(-gap / B_VEH)
        V = V + F * dt
        sp = np.linalg.norm(V, axis=1)
        vmax = np.maximum(self.v0[act] * V_MAX_FACTOR, 0.6)
        over = sp > vmax
        V[over] *= (vmax[over] / sp[over])[:, None]
        # standing agents damp out
        stand = self.v0[act] <= 0.0
        V[stand] *= 0.5
        newP = P + V * dt
        # keep inside the walkable area: try full step, then axis-projected steps, else stay
        ok = self._walk_mask(newP)
        if not ok.all():
            bad = np.flatnonzero(~ok)
            for b in bad:
                cand = [np.array([newP[b, 0], P0[b, 1]]), np.array([P0[b, 0], newP[b, 1]])]
                moved = False
                for c in cand:
                    if self.g.walkable(c[0], c[1]):
                        newP[b] = c
                        moved = True
                        break
                if not moved:
                    newP[b] = P0[b]
                    V[b] = 0.0
        # Continuous segment checks: reject a step instead of projecting agents
        # through bodies or moving them an unbounded distance after collision.
        for k in range(len(act)):
            if not self.g.los(tuple(P0[k]), tuple(newP[k]), cost_aware=False, margin=0.0):
                newP[k] = P0[k]
        # Deterministic symmetric rejection. Repeat because stopping one actor
        # can invalidate another actor's previously free segment.
        for _ in range(len(act) + 1):
            reject = set()
            tree = cKDTree(P0)
            reach = float(np.linalg.norm(newP - P0, axis=1).max(initial=0))
            for i, j in tree.query_pairs(2 * self.r[act].max() + 2 * reach):
                r0 = P0[i] - P0[j]
                dv = (newP[i] - P0[i]) - (newP[j] - P0[j])
                u = np.clip(-np.dot(r0, dv) / max(np.dot(dv, dv), 1e-12), 0, 1)
                if np.linalg.norm(r0 + u * dv) < self.r[act[i]] + self.r[act[j]] - 1e-6:
                    reject.update((i, j))
            changed = False
            for k in reject:
                if np.any(newP[k] != P0[k]):
                    newP[k] = P0[k]
                    changed = True
            if not changed:
                break
        self.pos[act] = newP
        V = (newP - P0) / dt
        self.vel[act] = V
        sp = np.linalg.norm(V, axis=1)
        moving = sp > 0.15
        desired = np.arctan2(V[moving, 1], V[moving, 0])
        delta = (desired - self.yaw[act[moving]] + np.pi) % (2 * np.pi) - np.pi
        self.yaw[act[moving]] += np.clip(delta, -3.0 * dt, 3.0 * dt)
        self.reached[act] = np.linalg.norm(self.goal[act] - newP, axis=1) < self.tol[act]

    def _walk_mask(self, P: np.ndarray) -> np.ndarray:
        g = self.g
        ix = np.floor(P[:, 0] - g.x0).astype(int)
        iy = np.floor(P[:, 1] - g.y0).astype(int)
        ok = (ix >= 0) & (iy >= 0) & (ix < g.nx) & (iy < g.ny)
        res = np.zeros(len(P), bool)
        res[ok] = g.walk[iy[ok], ix[ok]]
        return res
