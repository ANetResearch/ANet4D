"""Local failure scenes of the life runtime (`<out>.scene.npz`, written by awr.life.motion.runtime on a failure) and
their offline replay on the crowd alone.

A scene holds the support and walking masks within 20 m of the failing walkers, every body there and the paths and
waiting state of its walkers. `Scene.step` advances only the crowd (awr.life.motion.crowd) on the real masks: walkers
follow their recorded paths, every other body stands where it stood. Standoffs, rerouting, courtesy and the
activity director are not replayed, so a replay shows whether the motion layer alone resolves the situation.
Outside the window there is no support: a walker reaching its edge stops there. A scene that holds the static walking
network there (`net_*`) also answers route queries offline (`Scene.router`: awr.life.motion.route.LifeRouter on that
window, with the scene's support for off-network connectors and without the exact static bodies).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .crowd import Crowd, Path


class WindowSurface(SimpleNamespace):
    """The support of a scene window with awr.life.civil.surface.SurfaceMap.supported (unknown height = no support)."""

    def supported(self, xy, profile="person"):
        p = np.asarray(xy, float).reshape(-1, 2)
        finite = np.isfinite(p).all(1)
        p = np.where(np.isfinite(p), p, 0.0)
        ix = np.floor((p[:, 0] - self.x0) / self.cell_m - 0.5).astype(int)
        iy = np.floor((p[:, 1] - self.y0) / self.cell_m - 0.5).astype(int)
        inside = finite & (ix >= 0) & (iy >= 0) & (ix + 1 < self.nx) & (iy + 1 < self.ny)
        ix = np.clip(ix, 0, self.nx - 2)
        iy = np.clip(iy, 0, self.ny - 2)
        valid = np.isfinite(
            np.stack([self.z[iy, ix], self.z[iy, ix + 1], self.z[iy + 1, ix], self.z[iy + 1, ix + 1]], 1)
        ).all(1)
        cx = np.clip(np.floor((p[:, 0] - self.x0) / self.cell_m).astype(int), 0, self.nx - 1)
        cy = np.clip(np.floor((p[:, 1] - self.y0) / self.cell_m).astype(int), 0, self.ny - 1)
        return inside & valid & self.walk[profile][cy, cx]


class Scene:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.z = z
        self.t = float(z["t"])
        self.ids = [str(s) for s in z["ids"]]
        n = self.n = len(self.ids)
        walk = z["walk"]
        ny, nx = walk.shape
        cell = float(z["support_cell"])
        sx0, sy0 = z["support_origin"]
        surface = self.surface = WindowSurface(
            x0=float(sx0),
            y0=float(sy0),
            cell_m=cell,
            nx=nx,
            ny=ny,
            width_m=nx * cell,
            height_m=ny * cell,
            z=np.where(z["z_ok"], 0.0, np.nan),
            walk={"person": walk, "vehicle": np.zeros_like(walk)},
        )
        walking = SimpleNamespace(free=z["free"], origin=z["free_origin"], step=float(z["free_step"]), area=None)
        self.ext = z["ext"]
        self.crowd = Crowd(z["ext"], z["kinds"], surface, walking)
        self.crowd.waited[:] = z["waited"]
        self.crowd.motion[:] = z["motion"]
        self.crowd.aim[:] = z["aim"]
        self.xy, self.yaw, self.speed = z["xy"].astype(float), z["yaw"].astype(float), z["speed"].astype(float)
        self.active = z["active"].astype(bool)
        self.vmax = z["vmax"].astype(float)
        self.paths = {}
        for i in range(n):
            f, m = int(z["path_first"][i]), int(z["path_count"][i])
            if m:
                p = Path(z["path_points"][f : f + m].copy())
                p.s, p.k = float(z["path_progress"][i][0]), int(z["path_progress"][i][1])
                self.paths[i] = p
        self.mover = np.array([i in self.paths for i in range(n)])

    def router(self):
        """LifeRouter on the scene's static walking network window."""
        from awr.life.civil.walking import WalkNetwork

        from .route import LifeRouter

        z = self.z
        if "net_free" not in z:
            raise ValueError("This scene holds no walking network window")
        network = WalkNetwork(
            np.asarray(z["net_free"], bool),
            np.asarray(z["net_origin"], float),
            float(z["net_step"]),
            weight=z["net_weight"],
        )
        network.shortcut_tolerance = float(z["net_shortcut_tolerance"])
        network.surface = self.surface
        return LifeRouter(network, network.free, landmarks=2)

    def step(self):
        """One crowd tick; returns (new_xy, new_yaw, new_speed, held) and adopts the new poses."""
        n = self.n
        des_yaw, des_speed, cap = self.yaw.copy(), np.zeros(n), np.full(n, np.inf)
        anchor, tangent, lane = self.xy.copy(), self.yaw.copy(), np.zeros(n)
        for i, p in self.paths.items():
            h, want, cc = p.desire(self.xy[i], self.speed[i], self.vmax[i], self.yaw[i])
            if h is not None:
                des_yaw[i] = h
                des_speed[i] = want
                anchor[i] = p.anchor
                tangent[i] = p.tangent
                lane[i] = p.lane
            cap[i] = cc
        self.des_yaw, self.des_speed = des_yaw, des_speed
        out = self.crowd.step(
            self.xy,
            self.yaw,
            self.speed,
            self.active,
            self.mover,
            des_yaw,
            des_speed,
            self.vmax,
            cap,
            anchor,
            tangent,
            lane,
        )
        self.xy, self.yaw, self.speed = out[0], out[1], out[2]
        return out
