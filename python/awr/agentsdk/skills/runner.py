"""单机技能执行器：把一条技能展开成逐 tick 的 `bench/act.v1` 导航与云台指令。

技能 → 动作：

    goto     {x, y, agl}                       fly to a point at the given height above ground
    sweep    {polygon|center+size, agl, lane}  lawnmower coverage of an area; lane spacing defaults to the
                                               ground footprint width at that height (60° HFOV)
    orbit    {x, y, radius, agl, period}       circle a point, camera on the centre
    follow   {x, y, agl, standoff}             fly over a (moving) ground point and keep the camera on it; the
                                               commander refreshes x, y every decision step; ``standoff`` (m, default
                                               0) stops that far short of the point on the approach side for an
                                               oblique view
    look_at  {x, y}                            point the gimbal at a ground position (keeps current motion)
    hold     {}                                hover in place
    rtl      {}                                return to the launch pad and land (ends the drone's cost clock)
    land     {}                                harness only: emergency landing in place (battery exhausted)

高度由指挥者决定：技能只保留地形以上 2 m 的物理下限，给出速度不超过巡航的 `goto`；任务的高度包络由评分判定，不在这里强制。
每条航段都贴地形：目标点取沿航段最多 25 m 之前、该处地形之上 `agl` 的点（`_terrain_following`），长航段对着远端航点的高度
直飞会在上坡处跌出高度包络。`terrain(x, y) -> z` 为公开静态地形（`terrain.GridTerrain`），缺省用观测里本机脚下的 DTM。

2026-11-05 自 CHIMERA 基准移入（`chimera.commander.skills`），算法与数值不变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from awr.agentsdk.skills.actions import gimbal_look_at, nav_goto, nav_hover, nav_land

__all__ = ["ASPECT", "CRUISE_MPS", "HFOV_DEG", "SkillRunner", "SkillState", "footprint_w", "lawnmower", "point_in_polygon"]

CRUISE_MPS = 8.0
HFOV_DEG = 60.0
ASPECT = 16 / 9


def footprint_w(agl: float, hfov_deg: float = HFOV_DEG) -> float:
    return 2.0 * agl * math.tan(math.radians(hfov_deg / 2.0))


def point_in_polygon(poly: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorised over points. ``poly`` is (K, 2)."""
    P = np.asarray(poly, np.float64)
    inside = np.zeros(len(x), bool)
    j = len(P) - 1
    for i in range(len(P)):
        xi, yi, xj, yj = P[i, 0], P[i, 1], P[j, 0], P[j, 1]
        c = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
        inside ^= c
        j = i
    return inside


def lawnmower(poly: np.ndarray, lane: float) -> list[tuple[float, float]]:
    """Boustrophedon lanes along x inside the polygon bounding box, clipped to the polygon."""
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    pts: list[tuple[float, float]] = []
    ys = np.arange(y0 + lane / 2, y1, lane)
    for i, y in enumerate(ys):
        xs = np.arange(x0 + lane / 4, x1, lane / 4)
        inside = point_in_polygon(poly, xs, np.full_like(xs, y))
        if not inside.any():
            continue
        a, b = float(xs[inside].min()), float(xs[inside].max())
        pts += [(a, y), (b, y)] if i % 2 == 0 else [(b, y), (a, y)]
    return pts


@dataclass
class SkillState:
    kind: str = "hold"
    params: dict = field(default_factory=dict)
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    wp_i: int = 0
    t0_s: float = 0.0
    done: bool = False


class SkillRunner:
    """Per-UAV skill executor. ``tick(obs, ground_z, t_s)`` returns (nav_cmd | None, gimbal_cmd | None)."""

    def __init__(self, uav: str, cam: str, alt_range: tuple[float, float], home_xy: tuple[float, float], terrain=None) -> None:
        self.uav, self.cam = uav, cam
        self.lo, self.hi = alt_range
        self.home = home_xy
        self.terrain = terrain
        self.state = SkillState()
        self.look: tuple[float, float] | None = None
        self.landed = False
        self.land_sent = False
        self.land_here_sent = False

    def _agl(self, v) -> float:
        return max(float(v), 2.0)

    def set(self, kind: str, params: dict, t_s: float, pos_xy: tuple[float, float]) -> str | None:
        """Install a new skill; returns an error string for a malformed request."""
        p = dict(params or {})
        if self.terrain is not None and "x" in p and "y" in p:
            try:
                self.terrain(float(p["x"]), float(p["y"]))
            except ValueError as e:
                return str(e)
        st = SkillState(kind, p, t0_s=t_s)
        if "speed" in p:
            try:
                p["speed"] = float(min(max(float(p["speed"]), 1.0), CRUISE_MPS))
            except (TypeError, ValueError):
                return "speed must be a number (m/s)"
        if kind == "goto" or kind == "follow":
            if "x" not in p or "y" not in p:
                return f"{kind} needs x, y"
            # Over the point of interest, or (follow with "standoff") that many metres short of it on the approach
            # side, where the camera sees it at an oblique angle instead of straight down. The choice is the
            # commander's: the executor does not impose a viewing geometry.
            tx, ty = float(p["x"]), float(p["y"])
            agl = self._agl(p.get("agl", 60))
            d = math.dist(pos_xy, (tx, ty))
            so = max(0.0, float(p.get("standoff", 0.0))) if kind == "follow" else 0.0
            if d > so:
                hx, hy = tx + (pos_xy[0] - tx) * so / d, ty + (pos_xy[1] - ty) * so / d
            else:
                hx, hy = pos_xy
            st.waypoints = [(hx, hy)]
            st.params["agl"] = agl
            self.look = (tx, ty)
        elif kind == "sweep":
            agl = self._agl(p.get("agl", 60))
            lane = float(p.get("lane", 0.8 * footprint_w(agl) / ASPECT))
            if p.get("polygon"):
                poly = np.asarray(p["polygon"], float)
            elif "x" in p and "y" in p:
                s = float(p.get("size", 150)) / 2
                poly = np.array([[p["x"] - s, p["y"] - s], [p["x"] + s, p["y"] - s], [p["x"] + s, p["y"] + s], [p["x"] - s, p["y"] + s]], float)
            else:
                return "sweep needs polygon or x, y, size"
            st.waypoints = lawnmower(poly, max(lane, 10.0))
            if not st.waypoints:
                return "sweep polygon too small"
            # start from the nearest end
            if math.dist(pos_xy, st.waypoints[-1]) < math.dist(pos_xy, st.waypoints[0]):
                st.waypoints.reverse()
            p["agl"] = agl
            self.look = None
        elif kind == "orbit":
            if "x" not in p or "y" not in p:
                return "orbit needs x, y"
            p.setdefault("radius", 50.0)
            p.setdefault("period", 90.0)
            p["start_angle_rad"] = math.atan2(pos_xy[1] - float(p["y"]), pos_xy[0] - float(p["x"]))
            self.look = (float(p["x"]), float(p["y"]))
        elif kind == "look_at":
            if "x" not in p or "y" not in p:
                return "look_at needs x, y"
            self.look = (float(p["x"]), float(p["y"]))
            return None                                           # does not replace the motion skill
        elif kind == "hold":
            self.look = None
        elif kind == "rtl":
            st.waypoints = [self.home]
        elif kind == "land":
            st.waypoints, st.done = [], True                   # harness only: emergency landing in place
            self.land_here_sent = False
        else:
            return f"unknown skill {kind!r}"
        self.state = st
        return None

    def tick(self, obs: dict, ground_z: float, t_s: float) -> tuple[dict | None, dict | None]:
        st = self.state
        pos = np.asarray(obs["nav"]["pos_enu_m"], float)
        height = self.terrain if self.terrain is not None else (lambda x, y: ground_z)
        gim = gimbal_look_at(self.cam, (self.look[0], self.look[1], height(*self.look))) if self.look else None
        if st.kind == "land":
            if self.land_here_sent:
                return None, None
            self.land_here_sent = True
            return nav_land(), None
        if st.kind == "rtl" and st.done:
            if self.land_sent:
                return None, None                                 # landing in progress: do not re-issue the command
            self.land_sent = True
            return nav_land(), None
        if st.kind == "hold" or st.done:
            return nav_hover(), gim
        agl = self._agl(st.params.get("agl", 60.0))
        if st.kind == "orbit":
            cx, cy, r, T = float(st.params["x"]), float(st.params["y"]), float(st.params["radius"]), float(st.params["period"])
            ang = 2 * math.pi * ((t_s - st.t0_s) / T) + st.params["start_angle_rad"]
            ang_next = ang + 2 * math.pi * (2.0 / T)                 # aim 2 s ahead on the circle
            tgt = (cx + r * math.cos(ang_next), cy + r * math.sin(ang_next))
            return nav_goto(self._terrain_following(pos, tgt, agl, height), float(st.params.get("speed", CRUISE_MPS))), gim
        if st.kind == "rtl":
            tgt = st.waypoints[0]
            if math.dist((pos[0], pos[1]), tgt) < 2.0:
                st.done = True
                self.land_sent = True
                return nav_land(), None
            # Terrain-following like every other leg: a straight line aimed at the pad height descends below the
            # envelope floor over ground that rises between the drone and the pad (seen at 16 m AGL, 158 m out).
            return nav_goto(self._terrain_following(pos, tgt, max(self.lo + 2.0, 30.0), height), CRUISE_MPS), None
        # goto / follow / sweep: advance along waypoints
        tgt = st.waypoints[st.wp_i]
        if math.dist((pos[0], pos[1]), tgt) < 3.0 and abs(pos[2] - (height(*tgt) + agl)) < 2.0:
            if st.wp_i + 1 < len(st.waypoints):
                st.wp_i += 1
                tgt = st.waypoints[st.wp_i]
            else:
                st.done = st.kind != "follow"
        if st.kind == "sweep" and self.look is None:
            gim = gimbal_look_at(self.cam, (pos[0], pos[1], ground_z))   # nadir while sweeping
        return nav_goto(self._terrain_following(pos, tgt, agl, height), float(st.params.get("speed", CRUISE_MPS))), gim

    @staticmethod
    def _terrain_following(pos, tgt, agl, height, step: float = 25.0):
        """Aim at most ``step`` metres ahead along the leg, at ``agl`` above the terrain there: a long leg aimed at the
        far waypoint's height descends below the altitude floor over rising ground before that height applies."""
        d = math.dist((pos[0], pos[1]), tgt)
        if d > step:
            u = step / d
            tgt = (pos[0] + (tgt[0] - pos[0]) * u, pos[1] + (tgt[1] - pos[1]) * u)
        return (tgt[0], tgt[1], height(*tgt) + agl)

    def status(self) -> dict:
        st = self.state
        return {"skill": st.kind, "done": st.done, "progress": f"{st.wp_i + 1}/{len(st.waypoints)}" if st.waypoints else "", "look_at": self.look}
