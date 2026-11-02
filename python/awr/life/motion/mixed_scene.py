"""Local motion-only replay of a rejected mixed activity frame.

Adapted from upstream bf1d5d7's failure scenes. Saves measured masks, complete
nearby bodies and observed path progress. Replay freezes rigid bodies and does
not run the Director, native physics, standoffs or rerouting: diagnostic only,
never a resumable simulation checkpoint or release evidence.
"""

from types import SimpleNamespace

import numpy as np

from .crowd import Crowd, PathStore


def save(sim, actor, output, half=20.0):
    if actor not in sim.index or not 5.0 <= half <= 100.0:
        raise ValueError("Known actor and bounded diagnostic window required")
    c = sim.crowd
    centre = sim.xy[sim.index[actor]]
    lo = centre - half
    hi = centre + half

    def window(x0, y0, step, shape):
        a = np.maximum(np.floor((lo - [x0, y0]) / step).astype(int), 0)
        b = np.minimum(np.ceil((hi - [x0, y0]) / step).astype(int) + 1, shape[::-1])
        return (slice(a[1], b[1]), slice(a[0], b[0])), np.array([x0, y0]) + a * step

    sw, so = window(c.sx0, c.sy0, c.scell, c.walk.shape)
    fw, fo = window(c.fx0, c.fy0, c.fstep, c.free.shape)
    # Include whole bodies which touch the window, not only their centres.
    inside = np.flatnonzero(((sim.xy + c.radius[:, None] >= lo) & (sim.xy - c.radius[:, None] <= hi)).all(1))
    points = []
    first = []
    count = []
    progress = []
    used = 0
    for i in inside:
        q = sim.paths.path(i) if i in sim.paths else None
        first.append(used)
        count.append(0 if q is None else len(q.p))
        progress.append((0.0, 0) if q is None else (q.s, q.k))
        if q is not None:
            points.append(q.p)
            used += len(q.p)
    np.savez_compressed(
        output,
        ids=np.array(sim.ids)[inside],
        index=inside,
        kinds=sim.kinds[inside],
        ext=c.ext[inside],
        xy=sim.xy[inside],
        yaw=sim.yaw[inside],
        speed=sim.speed[inside],
        active=sim.container[inside] < 0,
        vmax=sim.vmax[inside],
        waited=c.waited[inside],
        aim=c.aim[inside],
        path_points=np.concatenate(points) if points else np.zeros((0, 2)),
        path_first=np.array(first),
        path_count=np.array(count),
        path_progress=np.array(progress),
        walk=c.walk[sw],
        z_ok=c.z_ok[sw],
        support_origin=so,
        support_cell=c.scell,
        free=c.free[fw],
        free_origin=fo,
        free_step=c.fstep,
        t=sim.time_s,
        focus=np.array(actor),
        diagnostic_only=True,
    )


class Scene:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.ids = z["ids"].tolist()
            self.focus = str(z["focus"])
            n = len(self.ids)
            walk = z["walk"]
            ny, nx = walk.shape
            cell = float(z["support_cell"])
            sx, sy = z["support_origin"]
            surface = SimpleNamespace(
                x0=sx,
                y0=sy,
                cell_m=cell,
                nx=nx,
                ny=ny,
                width_m=nx * cell,
                height_m=ny * cell,
                z=np.where(z["z_ok"], 0.0, np.nan),
                walk={"person": walk},
            )
            network = SimpleNamespace(free=z["free"], origin=z["free_origin"], step=float(z["free_step"]), area=None)
            self.crowd = Crowd(z["ext"], z["kinds"], surface, network)
            self.crowd.waited[:] = z["waited"]
            self.crowd.aim[:] = z["aim"]
            self.xy = z["xy"]
            self.yaw = z["yaw"]
            self.speed = z["speed"]
            self.active = z["active"]
            self.vmax = z["vmax"]
            self.paths = PathStore(n)
            for i in range(n):
                f, m = int(z["path_first"][i]), int(z["path_count"][i])
                if m:
                    self.paths.set(i, z["path_points"][f : f + m])
                    self.paths.s[i], self.paths.k[i] = z["path_progress"][i]
            self.travel = np.array([i in self.paths for i in range(n)])
            # MixedLife grants Crowd ownership to every visible person, even
            # during a standing activity. Path presence only supplies intent;
            # it must not change how neighbours classify a pedestrian's margin.
            self.mover = self.active & (self.crowd.kinds == "person")

    def step(self):
        n = len(self.ids)
        h, want, cap, anchor, tangent, lane = self.paths.desire(
            np.flatnonzero(self.mover & self.travel), self.xy, np.abs(self.speed), self.vmax, self.yaw
        )
        dy = self.yaw.copy()
        ds = np.zeros(n)
        ok = self.mover & ~np.isnan(h)
        dy[ok] = h[ok]
        ds[ok] = want[ok]
        out = self.crowd.step(
            self.xy, self.yaw, self.speed, self.active, self.mover, dy, ds, self.vmax, cap, anchor, tangent, lane
        )
        self.xy, self.yaw, self.speed = out[:3]
        return out
