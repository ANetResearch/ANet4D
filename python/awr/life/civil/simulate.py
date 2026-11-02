"""Lockstep simulation of plans: pedestrians (social force on the nav grid), vehicles (SUMO), dogs, objects and
machines at 10 Hz. Produces ``tracks.npz`` (per-actor tick records from the end of the warm-up), ``actors.json``
and ``sim_events.jsonl`` (what actually happened, with tape times).

    python -m awr.life.civil.simulate --semantics … --population … --plans … --nav … --sumo … --out <dir> [--duration]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np

from awr.life.civil.navgrid import PedGrid
from awr.life.civil.pedestrians import Crowd
from awr.life.civil.population import VEHICLE_TYPES
from awr.life.civil.traffic import Traffic

DT = 0.1
ST_ABSENT, ST_MOVING, ST_WAITING, ST_IN_VEHICLE, ST_INSIDE, ST_GONE = 0, 1, 2, 3, 4, 5


class Sim:
    def __init__(
        self,
        sem: dict,
        pop: dict,
        plans: dict,
        nav_dir: Path,
        sumo_dir: Path,
        out: Path,
        seed: int,
        duration_s: float | None = None,
    ) -> None:
        self.sem, self.pop, self.plans, self.out = sem, pop, plans, out
        self.rng = np.random.default_rng(seed + 77)
        self.W = float(plans["warmup_s"])
        self.D = float(duration_s if duration_s is not None else plans["duration_s"])
        self.T = self.W + self.D
        self.grid = PedGrid(nav_dir, sem)
        from awr.life.civil.surface import SurfaceMap

        self.surface = SurfaceMap(nav_dir)
        self.traffic = Traffic(sumo_dir, seed)
        self.slots = {s["id"]: s for s in sem["curb_slots"]} | {
            s["id"]: {**s, "lot": l["id"], "sumo": l.get("sumo")} for l in sem["parking_lots"] for s in l["slots"]
        }
        self.stops = {s["id"]: s for s in sem["bus_stops"]}
        self.bld = {b["id"]: b for b in sem["buildings"]}
        self.poi = {p["id"]: p for p in sem["pois"]}
        x0, y0, x1, y1 = sem["stage"]
        self.stage = (x0, y0, x1, y1)
        self.events: list[dict] = []
        self.warnings: list[str] = []
        self._build_actors()
        self._boundary_cells()

    # ------------------------------------------------------------------ actors
    def _build_actors(self) -> None:
        pop, plans = self.pop, self.plans
        self.actors: list[dict] = []
        self.P: dict[str, dict] = {}
        self.Vh: dict[str, dict] = {}
        self.Dg: dict[str, dict] = {}
        self.Ob: dict[str, dict] = {}
        persons = {p["id"]: p for p in pop["persons"]}
        vehicles = {v["id"]: v for v in pop["vehicles"]}
        home_slots = plans.get("vehicle_home_slots", {})
        # persons with plans
        for pid, legs in plans["persons"].items():
            p = persons[pid]
            idx = len(self.actors)
            size = [500, 300, int(p["height_m"] * 1000)]
            ap = {
                "asset": "person-proxy@1",
                "top_rgb": list(p["top_rgb"]),
                "bottom_rgb": list(p["bottom_rgb"]),
                "height_mm": size[2],
            }
            self.actors.append(
                {
                    "idx": idx,
                    "id": pid,
                    "class": "person",
                    "scripted": True,
                    "profile": "person",
                    "size_mm": size,
                    "appearance": ap,
                    "role": p["role"],
                    "description": p["description"],
                    "home": p.get("home"),
                }
            )
            self.P[pid] = {
                "idx": idx,
                "legs": legs,
                "li": -1,
                "status": "absent",
                "t_until": math.inf,
                "speed": p["speed_mps"],
                "wps": [],
                "wi": 0,
                "vehicle": None,
                "carry": None,
                "dog": p.get("dog"),
                "walks_done": 0,
                "leg": None,
                "hold_t": 0.0,
                "role": p["role"],
            }
        # vehicles with trips or a home slot (parked scenery); bicycles only when someone rides them
        ridden_bikes = {
            leg.get("vehicle") for legs in plans["persons"].values() for leg in legs if leg.get("kind") == "bike"
        }
        for vid, v in vehicles.items():
            trips = plans["vehicles"].get(vid, [])
            slot = home_slots.get(vid) or v.get("home_slot")
            if v["vtype"] == "bicycle":
                if vid not in ridden_bikes:
                    continue
            elif not trips and not slot:
                continue
            idx = len(self.actors)
            size = [int(s * 1000) for s in v["size_m"]]
            self.actors.append(
                {
                    "idx": idx,
                    "id": vid,
                    "class": "vehicle",
                    "scripted": True,
                    "profile": "vehicle",
                    "size_mm": size,
                    "appearance": {"asset": "vehicle-proxy@1", "body_rgb": list(v["body_rgb"])},
                    "role": v["role"],
                    "vtype": v["vtype"],
                    "description": v["description"],
                }
            )
            st = {
                "id": vid,
                "idx": idx,
                "v": v,
                "trips": trips,
                "ti": -1,
                "status": "parked" if slot else "absent",
                "slot": slot,
                "xy": None,
                "yaw": 0.0,
                "speed": 0.0,
                "boarded": set(),
                "riders": [],
                "stop_until": math.inf,
                "trip": None,
                "sumo": False,
                "pending": False,
                "bd_done": False,
                "driver_out": None,
            }
            if slot:
                s = self.slots[slot]
                st["xy"] = list(s["xy"])
                st["yaw"] = float(s.get("yaw_deg", 0.0))
            self.Vh[vid] = st
        for d in pop["dogs"]:
            if d["owner"] not in self.P:
                continue
            idx = len(self.actors)
            self.actors.append(
                {
                    "idx": idx,
                    "id": d["id"],
                    "class": "dog",
                    "scripted": True,
                    "profile": "person",
                    "size_mm": [int(s * 1000) for s in d["size_m"]],
                    "appearance": {"asset": "dog-proxy@1", "body_rgb": list(d["body_rgb"])},
                    "role": "dog",
                    "description": f"{d['colour_name']}的狗",
                }
            )
            self.Dg[d["id"]] = {
                "idx": idx,
                "owner": d["owner"],
                "xy": None,
                "vel": np.zeros(2),
                "status": "absent",
                "yaw": 0.0,
                "roam": None,
            }
        for o in plans["objects"]:
            idx = len(self.actors)
            self.actors.append(
                {
                    "idx": idx,
                    "id": o["id"],
                    "class": "object",
                    "scripted": True,
                    "profile": None,
                    "size_mm": [int(s * 1000) for s in o["size_m"]],
                    "appearance": {"asset": "object-proxy@1", "body_rgb": list(o["body_rgb"])},
                    "role": o["kind"],
                    "description": o["kind"],
                }
            )
            self.Ob[o["id"]] = {
                "idx": idx,
                "o": o,
                "xy": list(o["xy"]) if o.get("xy") else None,
                "status": "static" if o.get("static") else "stowed",
            }
        for m in plans["machines"]:
            idx = len(self.actors)
            self.actors.append(
                {
                    "idx": idx,
                    "id": m["id"],
                    "class": "machine",
                    "scripted": True,
                    "profile": None,
                    "size_mm": [int(s * 1000) for s in m["size_m"]],
                    "appearance": {"asset": "machine-proxy@1", "body_rgb": list(m["body_rgb"])},
                    "role": m["kind"],
                    "description": m["kind"],
                }
            )
            self.Ob[m["id"]] = {
                "idx": idx,
                "o": m,
                "xy": list(m["xy"]),
                "status": "static",
                "yaw": m.get("yaw_deg", 0.0),
            }
        n = len(self.actors)
        self.n = n
        self.crowd = Crowd(self.grid, len(self.P))
        self.pid_of = list(self.P.keys())
        for i, pid in enumerate(self.pid_of):
            self.P[pid]["ci"] = i
        self.t_until = np.full(len(self.P), math.inf)  # timed transitions (vectorised check)
        nt = int(round(self.D / DT))
        self.rec_xy = np.zeros((nt, n, 2), np.float32)
        self.rec_v = np.zeros((nt, n, 2), np.float32)
        self.rec_yaw = np.zeros((nt, n), np.float32)
        self.rec_state = np.zeros((nt, n), np.uint8)
        self.rec_cont = np.full((nt, n), -1, np.int16)
        self.cur_xy = np.zeros((n, 2))
        self.cur_v = np.zeros((n, 2))
        self.cur_yaw = np.zeros(n)
        self.cur_state = np.zeros(n, np.uint8)
        self.cur_cont = np.full(n, -1, np.int16)

    def _boundary_cells(self) -> None:
        g = self.grid
        x0, y0, x1, y1 = self.stage
        iy, ix = np.nonzero(g.walk)
        xs, ys = ix + g.x0 + 0.5, iy + g.y0 + 0.5
        m = (xs < x0 + 4) | (xs > x1 - 4) | (ys < y0 + 4) | (ys > y1 - 4)
        self.boundary = np.stack([xs[m], ys[m]], 1)

    def log(self, kind: str, t: float, actors: list, xy, **extra) -> None:
        self.events.append(
            {
                "kind": kind,
                "t_s": round(t - self.W, 1),
                "actors": actors,
                "xy": [round(float(xy[0]), 2), round(float(xy[1]), 2)],
                **extra,
            }
        )

    # ------------------------------------------------------------------ person legs
    def start_leg(self, pid: str, t: float) -> None:
        p = self.P[pid]
        p["li"] += 1
        p["hold_t"] = 0.0
        ci = p["ci"]
        if p["li"] >= len(p["legs"]):
            p["status"] = "done"
            p["leg"] = None
            self.t_until[ci] = math.inf
            if self.crowd.active[ci]:
                self.crowd.v0[ci] = 0.0
                self.crowd.goal[ci] = self.crowd.pos[ci]
            return
        leg = p["legs"][p["li"]]
        p["leg"] = leg
        k = leg["kind"]
        if k == "inside":
            place = self.bld.get(leg["place"]) or self.poi.get(leg["place"])
            xy = place["door"] if place and "door" in place else (place["xy"] if place else self.cur_xy[p["idx"]])
            p["status"] = "inside"
            p["xy"] = list(xy)
            self.crowd.deactivate(ci)
            self.t_until[ci] = float(leg["until"])
        elif k == "appear":
            p["status"] = "absent"
            p["xy"] = list(leg["xy"])
            self.crowd.deactivate(ci)
            self.t_until[ci] = float(leg["at"])
        elif k in ("walk", "bike"):
            self._begin_walk(pid, leg, t)
        elif k == "loop":
            pts = leg.get("points") or []
            if not pts:
                self.start_leg(pid, t)
                return
            p["loop_pts"] = list(pts)
            p["loop_i"] = 0
            p["loop_speed"] = float(leg.get("speed", 1.2))
            p["loop_pauses"] = int(leg.get("pauses", 0))
            self._begin_walk(
                pid, {"kind": "walk", "to": pts[0], "speed": p["loop_speed"], "dog": leg.get("dog")}, t, sub=True
            )
        elif k in ("stand", "meet"):
            self._stand(pid, t, until=leg.get("until"), dur=leg.get("dur"))
        elif k == "enter":
            place = self.bld.get(leg["place"]) or self.poi.get(leg["place"])
            if place is None:
                self._block_action(pid, t, "unknown_entrance")
                return
            xy = place["door"] if "door" in place else place["xy"]
            if math.dist(self.crowd.pos[ci], xy) > 1.25:
                self._block_action(pid, t, "entrance_not_reached", entrance=xy)
                return
            p["status"] = "inside"
            # Preserve the actual doorway position across visibility changes.
            p["xy"] = self.crowd.pos[ci].tolist()
            self.crowd.deactivate(ci)
            dur = float(leg.get("dur", 60.0))
            self.t_until[ci] = t + dur if dur < 9000 else math.inf
            self.log("enter", t, [pid], xy, place=leg["place"], dur_s=round(dur, 0) if dur < 9000 else None)
            # carried parcel goes inside too
            if p.get("carry"):
                self.Ob[p["carry"]]["status"] = "inside"
                self.log("delivered", t, [pid, p["carry"]], xy, place=leg["place"])
                p["carry"] = None
        elif k == "board":
            v = self.Vh.get(leg["vehicle"])
            if v is None:
                self.start_leg(pid, t)
                return
            door = self._vehicle_door(v, pid)
            if door is not None and math.dist(self.crowd.pos[ci], door) > 0.65 and v["status"] == "parked":
                self._begin_walk(
                    pid, {"kind": "walk", "to": door, "speed": p["speed"], "place": "curb-door"}, t, sub=True
                )
                p["li"] -= 1
                return
            if v["status"] not in ("parked", "pending", "stopped"):
                p["status"] = "waitveh"
                p["vehicle"] = leg["vehicle"]
                self.crowd.v0[ci] = 0.0
                self.crowd.goal[ci] = self.crowd.pos[ci]
                self.t_until[ci] = t + 1.0
                return
            if self._board(pid, leg["vehicle"], t):
                self.start_leg(pid, t)
        elif k == "ride":
            p["status"] = "ride"
            p["vehicle"] = leg["vehicle"]
            p["alight_at"] = leg.get("alight_at")
            v = self.Vh.get(leg["vehicle"])
            if v is None or pid not in v["riders"]:
                self._block_action(pid, t, "ride_without_boarding", vehicle=leg["vehicle"])
                return
            self.crowd.deactivate(ci)
            self.t_until[ci] = math.inf
        elif k == "wait_vehicle":
            p["status"] = "waitveh"
            p["vehicle"] = leg["vehicle"]
            self.crowd.v0[ci] = 0.0
            self.crowd.goal[ci] = self.crowd.pos[ci]
            self.t_until[ci] = t + 1.0
        elif k == "wander":
            p["status"] = "wander"
            c = leg["center"]
            if not self.crowd.active[ci]:
                self.crowd.place(ci, c)
            p["wander"] = leg
            self.t_until[ci] = t + 0.1
        elif k == "drop":
            o = self.Ob.get(leg["object"])
            if o:
                pos = self.crowd.pos[ci]
                o["xy"] = [float(pos[0]) + 0.4, float(pos[1]) + 0.3]
                o["status"] = "static"
                self.log("drop_object", t, [pid, leg["object"]], pos)
            p["carry"] = None
            self.start_leg(pid, t)
        elif k == "leave":
            pos = self.crowd.pos[ci] if self.crowd.active[ci] else np.asarray(p.get("xy") or [0, 0])
            d = np.hypot(self.boundary[:, 0] - pos[0], self.boundary[:, 1] - pos[1])
            k_ = int(np.argmin(d)) if len(d) else None
            if k_ is None or d[k_] > 900:
                self._gone(pid, t)
                return
            self._begin_walk(
                pid,
                {
                    "kind": "walk",
                    "to": [float(self.boundary[k_, 0]), float(self.boundary[k_, 1])],
                    "speed": p["speed"],
                    "then_gone": True,
                },
                t,
                sub=True,
            )
        elif k == "gone":
            self._gone(pid, t)
        else:
            self.warnings.append(f"{pid}: unknown leg {k}")
            self.start_leg(pid, t)

    def _gone(self, pid: str, t: float) -> None:
        p = self.P[pid]
        p["status"] = "gone"
        self.crowd.deactivate(p["ci"])
        self.t_until[p["ci"]] = math.inf
        self.log("leave_stage", t, [pid], self.crowd.pos[p["ci"]])

    def _stand(self, pid: str, t: float, until=None, dur=None, label=None) -> None:
        p = self.P[pid]
        ci = p["ci"]
        if not self.crowd.active[ci]:
            self.crowd.place(ci, p.get("xy") or self.crowd.pos[ci])
        self.crowd.v0[ci] = 0.0
        pos = self.crowd.pos[ci]
        safe = self.grid.safe_point(float(pos[0]), float(pos[1]))
        if math.dist(safe, pos) > 0.3:
            self.crowd.goal[ci] = np.asarray(safe)
            self.crowd.v0[ci] = 0.6  # step off the carriageway edge before standing
        else:
            self.crowd.goal[ci] = pos.copy()
        p["status"] = "stand"
        tu = float(until) if until is not None else t + float(dur or 0.0)
        self.t_until[ci] = max(tu, t + 0.1)

    def _begin_walk(self, pid: str, leg: dict, t: float, sub: bool = False) -> None:
        p = self.P[pid]
        ci = p["ci"]
        if not self.crowd.active[ci]:
            self.crowd.place(ci, p.get("xy") or leg["to"])
        start = (float(self.crowd.pos[ci][0]), float(self.crowd.pos[ci][1]))
        to = (float(leg["to"][0]), float(leg["to"][1]))
        try:
            if leg.get("spread"):
                to = self.grid.random_point_near(to[0], to[1], float(leg["spread"]), self.rng, 1.5)
            elif leg["kind"] != "bike" and not (leg.get("place") or "").startswith(("curb-", "lot-")):
                to = self.grid.safe_point(to[0], to[1], 1.5, 5.0)
        except ValueError:
            self._block_action(pid, t, "unsupported_destination", destination=list(to))
            return
        if leg["kind"] == "bike" and not sub:
            bv = self.Vh.get(leg.get("vehicle"))
            if (
                bv is not None
                and bv.get("trip") is None
                and bv["status"] in ("parked", "absent")
                and math.dist(start, to) > 80.0
            ):
                f_ep = self.traffic.nearest_edge(start, "bicycle", 60.0)
                t_ep = self.traffic.nearest_edge(to, "bicycle", 60.0)
                if f_ep is not None and t_ep is not None and f_ep[0] != t_ep[0]:
                    if bv["xy"] is None:
                        bv["xy"] = [start[0] + 0.2, start[1] + 0.2]
                        bv["yaw"] = math.degrees(self.crowd.yaw[ci])
                    bv["status"] = "parked"
                    bv["slot"] = None
                    walk = {"kind": "walk", "to": [to[0], to[1]], "place": leg.get("place"), "speed": p["speed"]}
                    bv["trips"].insert(
                        bv["ti"] + 1,
                        {
                            "from": {"xy": [start[0], start[1]], "vclass": "bicycle"},
                            "to": {"xy": [to[0], to[1]], "vclass": "bicycle"},
                            "depart": t,
                            "driver": pid,
                            "purpose": "bike",
                            "push_speed": 1.5,
                            "walk_to": walk,
                        },
                    )
                    p["bike_v"] = None
                    p["sub"] = False
                    p["walk_leg"] = leg
                    self._board(pid, leg["vehicle"], t)
                    return
        follow = leg.get("follow")
        p["sub"] = sub
        p["walk_leg"] = leg
        p["then_gone"] = bool(leg.get("then_gone"))
        if leg.get("carry"):
            p["carry"] = leg["carry"]
            self.Ob[leg["carry"]]["status"] = "carried"
            self.Ob[leg["carry"]]["carrier"] = pid
        if leg["kind"] == "bike":
            p["bike_v"] = leg["vehicle"]
            bv = self.Vh.get(leg["vehicle"])
            if bv is not None:
                bv["status"] = "ridden"
                bv["rider"] = pid
        if follow and follow in self.P:
            p["status"] = "follow"
            p["leader"] = follow
            p["follow_off"] = float(self.rng.choice([-0.9, 0.9]))
            p["to"] = to
            self.crowd.v0[ci] = p["speed"]
            self.t_until[ci] = t + 0.2
            return
        jay = bool(leg.get("jaywalk")) or leg["kind"] == "bike"
        try:
            wps = self.grid.path(start, to, jaywalk=jay)
        except ValueError:
            wps = None
        if wps is None:
            self._block_action(pid, t, "no_path", destination=list(to))
            return
        if len(wps) < 2:
            self._end_walk(pid, t)
            return
        p["wps"] = wps
        p["wi"] = 1
        p["status"] = "walk"
        spd = float(leg.get("speed") or p["speed"])
        if leg["kind"] == "bike" and p.get("bike_v") and p["bike_v"] in self.Vh:
            spd = 4.5
        self.crowd.v0[ci] = spd
        self.crowd.goal[ci] = wps[1]
        self.crowd.tol[ci] = 0.9 if len(wps) > 2 else 0.5
        self.t_until[ci] = math.inf

    def _block_action(self, pid: str, t: float, reason: str, **detail) -> None:
        """A failed action remains at its real position; it never completes a plan."""
        p = self.P[pid]
        ci = p["ci"]
        p["status"] = "blocked"
        self.crowd.v0[ci] = 0.0
        self.crowd.vel[ci] = 0.0
        self.crowd.goal[ci] = self.crowd.pos[ci].copy()
        self.t_until[ci] = math.inf
        self.warnings.append(f"{pid}: {reason}")
        self.log("action_failed", t, [pid], self.crowd.pos[ci], reason=reason, **detail)

    def _end_walk(self, pid: str, t: float) -> None:
        p = self.P[pid]
        ci = p["ci"]
        p["walks_done"] += 1
        self.crowd.road_ok[ci] = False
        self.crowd.v0[ci] = 0.0
        self.crowd.goal[ci] = self.crowd.pos[ci].copy()
        p["status"] = "stand"
        if p.get("bike_v"):
            bv = self.Vh.get(p["bike_v"])
            if bv is not None:
                bv["status"] = "parked"
                bv["slot"] = None
                pos = self.crowd.pos[ci]
                bv["xy"] = [float(pos[0]) + 0.2, float(pos[1]) + 0.2]  # leant next to the rider (no visible jump)
                bv["yaw"] = math.degrees(self.crowd.yaw[ci])
            p["bike_v"] = None
        if p.get("then_gone"):
            p["then_gone"] = False
            self._gone(pid, t)
            return
        wl = p.get("walk_leg") or {}
        if p.get("loop_pts") is not None and p.get("sub"):
            p["loop_i"] += 1
            if p["loop_i"] < len(p["loop_pts"]):
                nxt = {"kind": "walk", "to": p["loop_pts"][p["loop_i"]], "speed": p["loop_speed"], "dog": wl.get("dog")}
                if p["loop_pauses"] > 0 and self.rng.random() < 0.35:
                    p["loop_pauses"] -= 1
                    p["pending_walk"] = nxt
                    self._stand(pid, t, dur=float(self.rng.uniform(20, 60)))
                    return
                self._begin_walk(pid, nxt, t, sub=True)
                return
            p["loop_pts"] = None
        until = wl.get("until_t")
        if until is not None and t < float(until):
            self._stand(pid, t, until=float(until))
            p["after_stand_next"] = True
            return
        self.start_leg(pid, t)

    def _vehicle_door(self, v: dict, pid: str):
        if v.get("xy") is None:
            return None
        yaw = math.radians(v["yaw"])
        width = self.actors[v["idx"]]["size_mm"][1] / 2000.0
        for side in (-1, 1):
            q = np.asarray(v["xy"]) + side * (width + 0.45) * np.array([-math.sin(yaw), math.cos(yaw)])
            if not self.grid.walkable(*q):
                continue
            other = [p["ci"] for key, p in self.P.items() if key != pid and self.crowd.active[p["ci"]]]
            if other and np.any(np.linalg.norm(self.crowd.pos[other] - q, axis=1) < 0.6):
                continue
            return q.tolist()
        return None

    def _board(self, pid: str, vid: str, t: float) -> bool:
        p = self.P[pid]
        v = self.Vh[vid]
        ci = p["ci"]
        if (
            v.get("xy") is None
            or v.get("motion")
            or v.get("speed", 0.0) > 0.1
            or v["status"] not in ("parked", "stopped")
        ):
            self._block_action(pid, t, "vehicle_not_stationary", vehicle=vid)
            return False
        yaw = math.radians(v["yaw"])
        half_width = self.actors[v["idx"]]["size_mm"][1] / 2000.0
        rel = self.crowd.pos[ci] - np.asarray(v["xy"])
        along = abs(rel[0] * math.cos(yaw) + rel[1] * math.sin(yaw))
        side = abs(-rel[0] * math.sin(yaw) + rel[1] * math.cos(yaw))
        half_length = self.actors[v["idx"]]["size_mm"][0] / 2000.0
        if along > half_length * 0.8 or not half_width + 0.05 <= side <= half_width + 0.85:
            self._block_action(pid, t, "vehicle_door_not_reached", vehicle=vid)
            return False
        capacity = 40 if v["v"].get("vtype") == "bus" else (2 if v["v"].get("vtype") == "truck" else 5)
        if len(v["riders"]) >= capacity and pid not in v["riders"]:
            self._block_action(pid, t, "vehicle_full", vehicle=vid)
            return False
        p["status"] = "ride"
        p["vehicle"] = vid
        v["boarded"].add(pid)
        if pid not in v["riders"]:
            v["riders"].append(pid)
        self.crowd.deactivate(ci)
        self.t_until[ci] = math.inf
        self.log("board", t, [pid, vid], v["xy"] or self.crowd.pos[ci])
        return True

    def _alight(self, pid: str, vid: str, t: float, walk: dict | None = None) -> None:
        p = self.P[pid]
        v = self.Vh[vid]
        ci = p["ci"]
        if v.get("motion") or v.get("speed", 0.0) > 0.1:
            self.warnings.append(f"{pid}: refused alighting from moving vehicle")
            return
        door = self._vehicle_door(v, pid)
        if door is None:
            self.warnings.append(f"{pid}: no safe alighting surface")
            self.log("action_failed", t, [pid, vid], v["xy"], reason="no_safe_alighting_surface")
            return
        x, y = door
        if pid in v["riders"]:
            v["riders"].remove(pid)
        v["boarded"].discard(pid)
        p["xy"] = [x, y]
        self.crowd.place(ci, (x, y))
        self.log("alight", t, [pid, vid], (x, y))
        if walk:
            self._begin_walk(pid, dict(walk), t, sub=True)
        else:
            self.start_leg(pid, t)

    # ------------------------------------------------------------------ vehicles
    def _slot_sumo(self, slot_id: str):
        s = self.slots[slot_id]
        su = s.get("sumo")
        if su:
            return su["lane"].rsplit("_", 1)[0], float(su["pos"])
        ne = self.traffic.nearest_edge(s["xy"])
        return ne

    def _endpoint(self, spec: dict, first: bool):
        if "slot" in spec:
            return self._slot_sumo(spec["slot"])
        if "stop" in spec:
            su = self.stops[spec["stop"]]["sumo"]
            return su["lane"].rsplit("_", 1)[0], float(su["pos"])
        if "entry" in spec:
            return spec["entry"], 3.0
        if "exit" in spec:
            return spec["exit"], max(3.0, self.traffic.edge_len(spec["exit"]) - 3.0)
        if "xy" in spec:
            return self.traffic.nearest_edge(spec["xy"], spec.get("vclass", "passenger"), 60.0)
        return None

    def _lane_xy(self, ep, vclass: str = "passenger") -> list | None:
        try:
            import sumolib

            e = self.traffic.net.getEdge(ep[0])
            L = e.getLanes()[self.traffic.lane_index(ep[0], vclass)]
            x, y = sumolib.geomhelper.positionAtShapeOffset(L.getShape(), float(ep[1]))
            return [float(x), float(y)]
        except Exception:
            return None

    def _motion(self, v: dict, t: float) -> None:
        m = v.get("motion")
        if not m:
            return
        if "path" not in m:
            from awr.life.civil.maneuver import plan

            size = np.asarray(self.actors[v["idx"]]["size_mm"]) * 0.001
            pp, angles, ss = plan(
                m["from"], m["to"], math.radians(v["yaw"]), math.radians(m["yaw_to"]), size, self.surface
            )
            m["path"], m["angles"], m["arc"] = pp, angles, ss
            # Ease from/to rest, keeping peak speed <= 1.5 m/s.
            m["t1"] = m["t0"] + max(2.0, ss[-1])
        u = min(1.0, max(0.0, (t - m["t0"]) / (m["t1"] - m["t0"])))
        distance = (3 * u * u - 2 * u * u * u) * m["arc"][-1]
        v["xy"] = [float(np.interp(distance, m["arc"], m["path"][:, j])) for j in (0, 1)]
        v["yaw"] = math.degrees(float(np.interp(distance, m["arc"], m["angles"])))
        v["speed"] = 6 * u * (1 - u) * m["arc"][-1] / (m["t1"] - m["t0"])
        if u >= 1.0:
            v["yaw"] = m["yaw_to"]
            v["motion"] = None
            if v.get("alight_pending"):
                self._finish_alighting(v["id"], v, v.pop("alight_pending"), t)
            if v.get("exchange_pending"):
                self._bus_exchange(v["id"], v, v.pop("exchange_pending"), t)

    def _depart_ready(self, v: dict, trip: dict, t: float) -> bool:
        d = trip["depart"]
        if isinstance(d, (int, float)):
            return t >= float(d)
        if "after_board" in d:
            return all(pid in v["boarded"] for pid in d["after_board"]) and t >= v["stop_until"]
        if "after_stop" in d:
            return t >= v["stop_until"]
        return False

    def _start_trip(self, vid: str, v: dict, trip: dict, t: float) -> None:
        tr = self.traffic
        vv = v["v"]
        vclass, size = VEHICLE_TYPES[vv["vtype"]][0], vv["size_m"]
        sf = float(trip.get("speed_factor", 1.0))
        tid = tr.ensure_type(vv["vtype"], vclass, size, vv["body_rgb"], sf)
        frm = self._endpoint(trip["from"], True)
        to = self._endpoint(trip["to"], False)
        if frm is None or to is None:
            self.warnings.append(f"{vid}: trip endpoints unresolved")
            v["ti"] += 1
            return
        stops = []
        if trip.get("breakdown"):
            stops.append(("@route", 0.4, float(trip["breakdown"]["dur"])))
        if trip.get("double_park"):
            stops.append(("@dest", -18.0, float(trip["double_park"]["dur"])))
        v["pullout"] = (vid, tid, vclass, size[0], frm, to, self._resolve_stops(tr, tid, frm, to, stops))
        v["trip"] = trip
        v["status"] = "pending"
        v["pending"] = True
        if v["xy"] is not None and ("slot" in trip["from"] or "stop" in trip["from"] or "xy" in trip["from"]):
            lane_xy = self._lane_xy(frm, vclass)
            if lane_xy is not None:
                lane_next = self._lane_xy((frm[0], min(frm[1] + 0.5, tr.edge_len(frm[0]) - 0.1)), vclass)
                lane_yaw = (
                    math.degrees(math.atan2(lane_next[1] - lane_xy[1], lane_next[0] - lane_xy[0]))
                    if lane_next is not None
                    else v["yaw"]
                )
                lane_xy = (
                    np.asarray(lane_xy)
                    - np.array([math.cos(math.radians(lane_yaw)), math.sin(math.radians(lane_yaw))]) * size[0] / 2
                ).tolist()
                dist = math.dist(v["xy"], lane_xy)
                ps = float(trip.get("push_speed", 2.5))
                v["motion"] = {
                    "from": list(v["xy"]),
                    "to": list(lane_xy),
                    "yaw_to": lane_yaw,
                    "t0": t,
                    "t1": t + max(0.5, dist / ps),
                    "then": "pending",
                }
        self._insert_if_ready(vid, v)
        v["sumo"] = False
        v["bd_done"] = False
        v["dp_done"] = False
        if "slot" in trip["from"]:
            v["slot"] = None
        self.log("depart", t, [vid], v["xy"] or [0, 0], purpose=trip.get("purpose"))

    def _insert_if_ready(self, vid: str, v: dict) -> None:
        """Hand the vehicle to SUMO once its pull-out motion (if any) has finished."""
        if not v.get("pullout") or v.get("motion"):
            return
        args = v.pop("pullout")
        ok = self.traffic.add_trip(*args[:6], stops=args[6])
        if not ok:
            self.warnings.append(f"{vid}: no route {args[4]} → {args[5]}")
            v["trip"] = None
            v["status"] = "parked" if v.get("xy") is not None else "absent"
            v["pending"] = False

    def _resolve_stops(self, tr, tid, frm, to, stops):
        if not stops:
            return []
        try:
            edges = list(tr.ls.simulation.findRoute(frm[0], to[0], tid).edges)
        except Exception:
            return []
        out = []
        for kind, a, dur in stops:
            if kind == "@route" and len(edges) >= 2:
                k = min(len(edges) - 1, max(1, int(round(a * len(edges)))))
                e = edges[k]
                out.append((e, max(5.0, 0.5 * tr.edge_len(e)), dur))
            elif kind == "@dest":
                e = edges[-1]
                pos = to[1] + a
                if pos > 8.0:
                    out.append((e, pos, dur))
        return out

    def _complete_trip(self, vid: str, v: dict, t: float) -> None:
        trip = v["trip"]
        v["sumo"] = False
        v["pending"] = False
        v["trip"] = None
        v["speed"] = 0.0
        if trip is None:
            return
        to = trip["to"]
        if "slot" in to:
            s = self.slots[to["slot"]]
            start = list(v["xy"]) if v["xy"] is not None else list(s["xy"])
            dist = math.dist(start, s["xy"])
            v["motion"] = {
                "from": start,
                "to": list(s["xy"]),
                "yaw_to": float(s.get("yaw_deg", v["yaw"])),
                "t0": t,
                "t1": t + max(0.5, dist / 2.5),
                "then": "parked",
            }
            v["slot"] = to["slot"]
            v["status"] = "parked"
            sd = trip.get("stop_dur")
            v["stop_until"] = (t + float(sd)) if isinstance(sd, (int, float)) else (t if sd is None else math.inf)
            self.log(
                "park",
                t,
                [vid],
                s["xy"],
                slot=to["slot"],
                legal=bool(s.get("legal", True)),
                reasons=s.get("reasons", []),
                where=s.get("where", "lot"),
                purpose=trip.get("purpose"),
            )
            if isinstance(sd, dict) and "until_board" in sd:
                v["wait_for"] = sd["until_board"]
                v["stop_until"] = math.inf
        elif "stop" in to:
            s = self.stops[to["stop"]]
            start = list(v["xy"]) if v["xy"] is not None else list(s["xy"])
            v["motion"] = {
                "from": start,
                "to": list(s["xy"]),
                "yaw_to": v["yaw"],
                "t0": t,
                "t1": t + max(0.5, math.dist(start, s["xy"]) / 2.5),
                "then": "parked",
            }
            v["status"] = "parked"
            v["slot"] = None
            v["stop_until"] = t + float(trip.get("stop_dur") or 30.0)
            self.log("bus_stop", t, [vid], s["xy"], stop=to["stop"])
            v["exchange_pending"] = to["stop"]
        elif "xy" in to:
            start = list(v["xy"]) if v["xy"] is not None else list(to["xy"])
            sp = self.grid.safe_point(start[0], start[1], 1.0, 6.0)
            v["motion"] = {
                "from": start,
                "to": [float(sp[0]), float(sp[1])],
                "yaw_to": v["yaw"],
                "t0": t,
                "t1": t + max(0.5, math.dist(start, sp) / 1.5),
                "then": "parked",
            }
            v["status"] = "parked"
            v["slot"] = None
            v["stop_until"] = t
            self.log("bike_park", t, [vid], sp)
        else:
            v["status"] = "gone"
            self.log("leave_stage", t, [vid], v["xy"] or [0, 0])
        # A parking manoeuvre must finish before any passenger can alight.
        if v.get("motion"):
            v["alight_pending"] = trip
            return
        self._finish_alighting(vid, v, trip, t)

    def _finish_alighting(self, vid, v, trip, t):
        to = trip["to"]
        for pid in list(v["riders"]):
            p = self.P[pid]
            if p["status"] != "ride":
                continue
            if "stop" in to and p.get("alight_at") and p["alight_at"] != to["stop"] and v["status"] != "gone":
                continue
            if v["status"] == "gone":
                p["status"] = "gone"
                v["riders"].remove(pid)
                continue
            self._alight(pid, vid, t, walk=trip.get("walk_to"))
        v["boarded"] = set()
        v["ti"] += 0

    def _bus_exchange(self, vid: str, v: dict, stop_id: str, t: float) -> None:
        sxy = self.stops[stop_id]["xy"]
        for pid, p in self.P.items():
            if (
                p["status"] == "waitveh"
                and p["vehicle"] == vid
                and self.crowd.active[p["ci"]]
                and math.dist(self.crowd.pos[p["ci"]], sxy) < 20.0
            ):
                if not self._board(pid, vid, t):
                    continue
                self.start_leg(pid, t)  # board leg
                if p["leg"] and p["leg"]["kind"] == "board":
                    self.start_leg(pid, t)  # → ride

    def step_vehicles(self, t: float) -> None:
        tr = self.traffic
        # departures
        for vid, v in self.Vh.items():
            if v["status"] in ("gone", "ridden"):
                continue
            if v["trip"] is None and v["ti"] + 1 < len(v["trips"]):
                nxt = v["trips"][v["ti"] + 1]
                if v.get("wait_for"):
                    if all(pid in v["boarded"] for pid in v["wait_for"]):
                        v["wait_for"] = None
                        v["stop_until"] = t
                    else:
                        continue
                if self._depart_ready(v, nxt, t):
                    v["ti"] += 1
                    self._start_trip(vid, v, nxt, t)
        res, departed, arrived = tr.step()
        for vid in departed:
            v = self.Vh.get(vid)
            if v:
                v["sumo"] = True
                v["pending"] = False
                v["status"] = "driving"
                v["blend_from"] = list(v["xy"]) if v.get("xy") is not None else None
                v["blend_t0"] = t
        for vid, v in self.Vh.items():
            if v.get("motion") and not v["sumo"]:
                self._motion(v, t)
                self._insert_if_ready(vid, v)
        for vid, (x, y, yaw, sp, road, lp, stopped) in res.items():
            v = self.Vh.get(vid)
            if v is None:
                continue
            v["motion"] = None
            if v.get("blend_from") is not None:
                u = min(1.0, (t - v["blend_t0"]) / 1.5)
                if u >= 1.0 or math.dist(v["blend_from"], (x, y)) > 12.0:
                    v["blend_from"] = None
                else:
                    x = v["blend_from"][0] + (x - v["blend_from"][0]) * u
                    y = v["blend_from"][1] + (y - v["blend_from"][1]) * u
                    if v.get("xy") is not None:
                        sp = max(sp, math.dist(v["xy"], (x, y)) / 0.1)  # report the blended displacement as speed
            v["xy"] = [x, y]
            v["yaw"] = yaw
            v["speed"] = sp
            trip = v["trip"] or {}
            if stopped and trip.get("breakdown") and not v["bd_done"]:
                v["bd_done"] = True
                v["status"] = "stopped"
                drv = trip.get("driver")
                if drv in self.P and self.P[drv]["status"] == "ride":
                    self._alight(drv, vid, t)
                    self.P[drv]["li"] -= 1  # stay on the ride leg logically
                    self._stand(drv, t, dur=float(trip["breakdown"]["dur"]) - 5.0)
                    self.P[drv]["reboard"] = vid
                    v["driver_out"] = drv
                self.log("breakdown", t, [vid, drv], (x, y), dur_s=trip["breakdown"]["dur"])
            elif not stopped and v["status"] == "stopped":
                v["status"] = "driving"
                drv = v.get("driver_out")
                if drv:
                    p = self.P[drv]
                    p["status"] = "ride"
                    self.crowd.deactivate(p["ci"])
                    self.t_until[p["ci"]] = math.inf
                    v["driver_out"] = None
                    self.log("breakdown_end", t, [vid, drv], (x, y))
            if stopped and trip.get("double_park") and not v.get("dp_done"):
                v["dp_done"] = True
                self.log("double_park", t, [vid], (x, y), dur_s=trip["double_park"]["dur"])
        for vid in arrived:
            v = self.Vh.get(vid)
            if v is not None:
                self._complete_trip(vid, v, t)
        self._veh_arrays(res)
        self._yield_to_pedestrians(res)

    def _yield_to_pedestrians(self, res: dict) -> None:
        """Vehicles brake for walkers on the carriageway within 14 m ahead (lateral < 3 m) and resume when clear."""
        cr = self.crowd
        act = np.flatnonzero(cr.active)
        if not hasattr(self, "_yielding"):
            self._yielding: set[str] = set()
        if len(act) == 0 or not res:
            return
        P = cr.pos[act]
        onroad = self.grid.on_road(P)
        Q = P[onroad]
        ls = self.traffic.ls
        for vid, (x, y, yaw, sp, *_r) in res.items():
            hx, hy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            L = self.actors[self.Vh[vid]["idx"]]["size_mm"][0] / 1000.0 if vid in self.Vh else 4.5
            fx, fy = x + hx * L / 2.0, y + hy * L / 2.0  # front bumper
            block = False
            if len(Q):
                rel = Q - np.array([fx, fy])
                along = rel[:, 0] * hx + rel[:, 1] * hy
                cross = np.abs(rel[:, 0] * hy - rel[:, 1] * hx)
                block = bool(((along > -1.0) & (along < 14.0) & (cross < 3.0)).any())
            if block and vid not in self._yielding:
                with contextlib.suppress(Exception):
                    ls.vehicle.slowDown(vid, 0.0, 1.2)
                self._yielding.add(vid)
            elif not block and vid in self._yielding:
                with contextlib.suppress(Exception):
                    ls.vehicle.setSpeed(vid, -1.0)
                self._yielding.discard(vid)

    def _veh_arrays(self, res: dict) -> None:
        xs, vs, rs, hs = [], [], [], []
        for vid, (x, y, yaw, sp, *_r) in res.items():
            v = self.Vh.get(vid)
            if v is None:
                continue
            L = self.actors[v["idx"]]["size_mm"][0] / 1000.0
            xs.append((x, y))
            hs.append((math.cos(math.radians(yaw)), math.sin(math.radians(yaw))))
            vs.append(sp)
            rs.append(max(1.0, L / 2.0))
        for vid, v in self.Vh.items():
            if v.get("motion") and v["xy"] is not None and v.get("speed", 0.0) > 0.1:
                L = self.actors[v["idx"]]["size_mm"][0] / 1000.0
                xs.append(tuple(v["xy"]))
                hs.append((math.cos(math.radians(v["yaw"])), math.sin(math.radians(v["yaw"]))))
                vs.append(v["speed"])
                rs.append(max(1.0, L / 2.0))
        self.veh_xy = np.array(xs) if xs else np.zeros((0, 2))
        self.veh_h = np.array(hs) if hs else np.zeros((0, 2))
        self.veh_sp = np.array(vs) if vs else np.zeros(0)
        self.veh_r = np.array(rs) if rs else np.zeros(0)

    def _vehicle_coming(self, xy) -> bool:
        if len(self.veh_xy) == 0:
            return False
        rel = np.asarray(xy) - self.veh_xy
        d = np.linalg.norm(rel, axis=1)
        along = np.einsum("ij,ij->i", rel, self.veh_h)
        cross = np.abs(rel[:, 0] * self.veh_h[:, 1] - rel[:, 1] * self.veh_h[:, 0])
        ttc = along / np.maximum(self.veh_sp, 0.3)
        m = (self.veh_sp > 0.8) & (d < 60.0) & (along > -2.0) & (cross < 4.5) & (ttc < 6.0)
        return bool(m.any())

    # ------------------------------------------------------------------ persons per tick
    def step_persons(self, t: float) -> None:
        cr = self.crowd
        # timed transitions
        due = np.flatnonzero(t >= self.t_until)
        for ci in due:
            pid = self.pid_of[ci]
            p = self.P[pid]
            st = p["status"]
            if st == "waitveh":
                v = self.Vh.get(p["vehicle"])
                if (
                    v is not None
                    and v["status"] in ("parked", "stopped")
                    and v["xy"] is not None
                    and math.dist(cr.pos[ci], v["xy"]) < 20.0
                ):
                    self.start_leg(pid, t)
                    if p["leg"] and p["leg"]["kind"] == "board":
                        self.start_leg(pid, t)
                elif v is not None and v["status"] == "gone":
                    self.start_leg(pid, t)
                else:
                    self.t_until[ci] = t + 1.0
            elif st == "follow":
                L = self.P[p["leader"]]
                lci = L["ci"]
                if not cr.active[lci] or L["status"] not in ("walk", "follow") or math.dist(cr.pos[ci], p["to"]) < 2.0:
                    if L["status"] not in ("walk", "follow") or math.dist(cr.pos[ci], p["to"]) < 2.0:
                        self._end_walk(pid, t)
                        continue
                hv = cr.vel[lci]
                hn = np.linalg.norm(hv)
                h = hv / hn if hn > 0.1 else np.array([math.cos(cr.yaw[lci]), math.sin(cr.yaw[lci])])
                perp = np.array([-h[1], h[0]])
                goal = cr.pos[lci] - h * 1.0 + perp * p["follow_off"]
                cr.goal[ci] = goal
                cr.v0[ci] = min(1.9, max(0.0, hn + 0.3)) if hn > 0.1 else 0.0
                self.t_until[ci] = t + 0.2
            elif st == "wander":
                w = p["wander"]
                if t >= float(w["until"]):
                    self.start_leg(pid, t)
                    continue
                if cr.reached[ci] or cr.v0[ci] == 0.0:
                    if self.rng.random() < float(w.get("stand_frac", 0.4)):
                        cr.v0[ci] = 0.0
                        cr.goal[ci] = cr.pos[ci].copy()
                        self.t_until[ci] = t + float(self.rng.uniform(5, 25))
                    else:
                        gx, gy = self.grid.random_point_near(
                            w["center"][0], w["center"][1], float(w["radius"]), self.rng, 0.0
                        )
                        cr.goal[ci] = (gx, gy)
                        cr.v0[ci] = float(w.get("speed", 0.8))
                        self.t_until[ci] = t + 15.0
                else:
                    self.t_until[ci] = t + 1.0
            elif st == "stand" and p.get("pending_walk"):
                nxt = p.pop("pending_walk")
                self._begin_walk(pid, nxt, t, sub=True)
            elif st == "stand" and p.get("after_stand_next"):
                p["after_stand_next"] = False
                self.start_leg(pid, t)
            elif st in ("inside", "absent"):
                xy = p.get("xy")
                cr.place(ci, xy)
                if st == "inside":
                    self.log("exit", t, [pid], xy, place=(p["leg"] or {}).get("place"))
                self.start_leg(pid, t)
            elif st == "xwait":
                self._try_cross(pid, t)
            else:
                self.start_leg(pid, t)
        # walkers reaching waypoints
        walk = [ci for ci in np.flatnonzero(cr.reached & cr.active) if self.P[self.pid_of[ci]]["status"] == "walk"]
        for ci in walk:
            pid = self.pid_of[ci]
            p = self.P[pid]
            p["wi"] += 1
            if p["wi"] >= len(p["wps"]):
                self._end_walk(pid, t)
                continue
            cr.reached[ci] = False
            self._advance_wp(pid, t)

    def _advance_wp(self, pid: str, t: float) -> None:
        p = self.P[pid]
        ci = p["ci"]
        nxt = p["wps"][p["wi"]]
        pos = self.crowd.pos[ci]
        entering = (not self.grid.on_road(pos[None, :])[0]) and self._segment_on_road(pos, nxt)
        if entering and not (p.get("walk_leg") or {}).get("kind") == "bike":
            if self._vehicle_coming(pos):
                p["status"] = "xwait"
                p["hold_t"] = t
                self.crowd.v0[ci] = 0.0
                self.crowd.goal[ci] = pos.copy()
                self.t_until[ci] = t + 0.5
                return
        self.crowd.goal[ci] = nxt
        self.crowd.tol[ci] = 0.9 if p["wi"] < len(p["wps"]) - 1 else 0.5
        is_bike = (p.get("walk_leg") or {}).get("kind") == "bike" and bool(p.get("bike_v")) and p["bike_v"] in self.Vh
        self.crowd.road_ok[ci] = (
            is_bike or entering or self.grid.on_road(pos[None, :])[0] or self._segment_on_road(pos, nxt)
        )
        self.crowd.v0[ci] = float((p.get("walk_leg") or {}).get("speed") or p["speed"]) if not is_bike else 4.5

    def _try_cross(self, pid: str, t: float) -> None:
        p = self.P[pid]
        ci = p["ci"]
        pos = self.crowd.pos[ci]
        if self._vehicle_coming(pos) and t - p["hold_t"] < 25.0:
            self.t_until[ci] = t + 0.5
            return
        if t - p["hold_t"] > 3.0:
            self.log("crossing_wait", t, [pid], pos, wait_s=round(t - p["hold_t"], 1))
        p["status"] = "walk"
        self.t_until[ci] = math.inf
        self.crowd.goal[ci] = p["wps"][p["wi"]]
        self.crowd.road_ok[ci] = True
        self.crowd.v0[ci] = float((p.get("walk_leg") or {}).get("speed") or p["speed"]) * 1.15

    def _segment_on_road(self, a, b) -> bool:
        n = int(math.dist(a, b)) + 2
        xs = np.linspace(a[0], b[0], n)
        ys = np.linspace(a[1], b[1], n)
        return bool(self.grid.on_road(np.stack([xs, ys], 1)).any())

    # ------------------------------------------------------------------ dogs, objects
    def step_dogs(self, t: float) -> None:
        cr = self.crowd
        for did, d in self.Dg.items():
            o = self.P[d["owner"]]
            ci = o["ci"]
            out = cr.active[ci] and o["status"] in ("walk", "stand", "xwait", "follow") and (o["role"] == "dogwalker")
            if not out:
                d["status"] = "absent" if o["status"] in ("absent", "gone") else "inside"
                d["xy"] = list(o.get("xy") or cr.pos[ci])
                continue
            opos = cr.pos[ci]
            if d["xy"] is None or d["status"] in ("absent", "inside"):
                d["xy"] = [float(opos[0]) - 1.0, float(opos[1])]
                d["vel"] = np.zeros(2)
            d["status"] = "out"
            ov = cr.vel[ci]
            on = np.linalg.norm(ov)
            h = ov / on if on > 0.1 else np.array([math.cos(cr.yaw[ci]), math.sin(cr.yaw[ci])])
            perp = np.array([-h[1], h[0]])
            if d["roam"] is not None:
                target, t_end = d["roam"]
                if t > t_end or math.dist(d["xy"], target) < 1.0:
                    d["roam"] = None
                    target = opos - h * 1.5
            else:
                if o["status"] == "stand" and self.rng.random() < 0.004:
                    a = self.rng.uniform(0, 2 * math.pi)
                    r = self.rng.uniform(6, 14)
                    tx, ty = self.grid.snap(opos[0] + r * math.cos(a), opos[1] + r * math.sin(a), 6.0)
                    d["roam"] = ((tx, ty), t + 12.0)
                    self.log("dog_off_leash", t, [did, d["owner"]], opos)
                target = opos - h * 1.6 + perp * 0.6
            dv = np.asarray(target) - np.asarray(d["xy"])
            dn = np.linalg.norm(dv)
            des = dv / dn * min(3.0, max(0.0, dn * 1.5)) if dn > 0.05 else np.zeros(2)
            d["vel"] = d["vel"] + (des - d["vel"]) * 0.4
            nxy = np.asarray(d["xy"]) + d["vel"] * DT
            if not self.grid.walkable(nxy[0], nxy[1]):
                nxy = np.asarray(d["xy"])
                d["vel"] *= 0.0
            d["xy"] = [float(nxy[0]), float(nxy[1])]
            if np.linalg.norm(d["vel"]) > 0.2:
                d["yaw"] = math.degrees(math.atan2(d["vel"][1], d["vel"][0]))

    # ------------------------------------------------------------------ state assembly per tick
    def collect(self, t: float) -> None:
        cr = self.crowd
        X, V, Y, S, C = self.cur_xy, self.cur_v, self.cur_yaw, self.cur_state, self.cur_cont
        S[:] = ST_ABSENT
        C[:] = -1
        V[:] = 0.0
        for p in self.P.values():
            i, ci = p["idx"], p["ci"]
            st = p["status"]
            if st == "walk" and p.get("bike_v") and self.Vh.get(p["bike_v"], {}).get("status") == "ridden":
                X[i] = cr.pos[ci]
                Y[i] = math.degrees(cr.yaw[ci])
                S[i] = ST_IN_VEHICLE
                C[i] = self.Vh[p["bike_v"]]["idx"]
            elif st in ("walk", "stand", "xwait", "follow", "wander", "waitveh", "blocked"):
                X[i] = cr.pos[ci]
                V[i] = cr.vel[ci]
                Y[i] = math.degrees(cr.yaw[ci])
                S[i] = ST_MOVING if np.linalg.norm(cr.vel[ci]) > 0.15 else ST_WAITING
            elif st == "walk" and p.get("bike_v") and self.Vh.get(p["bike_v"], {}).get("status") == "ridden":
                X[i] = cr.pos[ci]
                Y[i] = math.degrees(cr.yaw[ci])
                S[i] = ST_IN_VEHICLE
                C[i] = self.Vh[p["bike_v"]]["idx"]
            elif st == "ride":
                v = self.Vh.get(p["vehicle"])
                if v is not None and v["xy"] is not None and v["status"] != "gone":
                    X[i] = v["xy"]
                    Y[i] = v["yaw"]
                    S[i] = ST_IN_VEHICLE
                    C[i] = v["idx"]
                else:
                    S[i] = ST_ABSENT if (v is None or v["status"] in ("absent", "pending")) else ST_GONE
            elif st == "inside":
                X[i] = p.get("xy") or X[i]
                S[i] = ST_INSIDE
            elif st == "gone":
                S[i] = ST_GONE
            elif st == "done":
                X[i] = cr.pos[ci] if cr.active[ci] else (p.get("xy") or X[i])
                S[i] = ST_WAITING if cr.active[ci] else ST_INSIDE
            else:
                S[i] = ST_ABSENT
        for v in self.Vh.values():
            i = v["idx"]
            st = v["status"]
            if st == "driving" or st == "stopped":
                X[i] = v["xy"]
                Y[i] = v["yaw"]
                yaw = math.radians(v["yaw"])
                V[i] = (v["speed"] * math.cos(yaw), v["speed"] * math.sin(yaw))
                S[i] = ST_MOVING if v["speed"] > 0.1 else ST_WAITING
            elif st in ("parked", "pending") and v["xy"] is not None:
                X[i] = v["xy"]
                Y[i] = v["yaw"]
                if v.get("motion"):
                    yaw = math.radians(v["yaw"])
                    V[i] = (v["speed"] * math.cos(yaw), v["speed"] * math.sin(yaw))
                    S[i] = ST_MOVING if v["speed"] > 0.1 else ST_WAITING
                else:
                    S[i] = ST_WAITING
            elif st == "ridden":
                p = self.P.get(v.get("rider"))
                if p is not None and cr.active[p["ci"]]:
                    X[i] = cr.pos[p["ci"]]
                    V[i] = cr.vel[p["ci"]]
                    Y[i] = math.degrees(cr.yaw[p["ci"]])
                    S[i] = ST_MOVING
            elif st == "gone":
                S[i] = ST_GONE
        for d in self.Dg.values():
            i = d["idx"]
            if d["status"] == "out":
                X[i] = d["xy"]
                V[i] = d["vel"]
                Y[i] = d["yaw"]
                S[i] = ST_MOVING if np.linalg.norm(d["vel"]) > 0.2 else ST_WAITING
            elif d["status"] == "inside":
                X[i] = d["xy"] or X[i]
                S[i] = ST_INSIDE
        # Stage bounds select evaluation geography, not optical visibility.
        # Entities leave only through an explicit exit action/event.
        for o in self.Ob.values():
            i = o["idx"]
            st = o["status"]
            if st == "static":
                X[i] = o["xy"]
                Y[i] = o.get("yaw", 0.0)
                S[i] = ST_WAITING
            elif st == "carried":
                p = self.P[o["carrier"]]
                ci = p["ci"]
                if cr.active[ci]:
                    X[i] = cr.pos[ci] + 0.45 * np.array([math.cos(cr.yaw[ci]), math.sin(cr.yaw[ci])])
                    V[i] = cr.vel[ci]
                    Y[i] = math.degrees(cr.yaw[ci])
                    S[i] = ST_MOVING if np.linalg.norm(cr.vel[ci]) > 0.15 else ST_WAITING
                else:
                    S[i] = ST_ABSENT
            elif st == "stowed":
                fv = o["o"].get("from_vehicle")
                v = self.Vh.get(fv) if fv else None
                if v is not None and v["xy"] is not None and v["status"] not in ("absent", "gone", "pending"):
                    X[i] = v["xy"]
                    S[i] = ST_IN_VEHICLE
                    C[i] = v["idx"]
                else:
                    S[i] = ST_ABSENT
            elif st == "inside":
                S[i] = ST_INSIDE

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        t0 = time.time()
        nt_total = int(round(self.T / DT))
        wt = int(round(self.W / DT))
        self.veh_xy = np.zeros((0, 2))
        self.veh_h = np.zeros((0, 2))
        self.veh_sp = np.zeros(0)
        self.veh_r = np.zeros(0)
        for pid in self.pid_of:
            self.start_leg(pid, 0.0)
        for k in range(nt_total):
            t = k * DT
            self.step_vehicles(t)
            self.step_persons(t)
            self.crowd.step(DT, self.veh_xy, self.veh_sp, self.veh_r, self.veh_h)
            self.step_dogs(t)
            if k >= wt:
                self.collect(t)
                j = k - wt
                self.rec_xy[j] = self.cur_xy
                self.rec_v[j] = self.cur_v
                self.rec_yaw[j] = self.cur_yaw
                self.rec_state[j] = self.cur_state
                self.rec_cont[j] = self.cur_cont
            if k % 3000 == 0:
                act = int(self.crowd.active.sum())
                print(
                    f"t={t:7.1f}s  walkers {act:4d}  vehicles in SUMO {len(self.veh_xy):3d}  elapsed {time.time() - t0:6.1f}s",
                    flush=True,
                )
        self.traffic.close()
        print(f"done in {time.time() - t0:.1f}s; events {len(self.events)}, warnings {len(self.warnings)}")

    def save(self) -> None:
        out = self.out
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "tracks.npz",
            xy=self.rec_xy,
            v=self.rec_v,
            yaw=self.rec_yaw,
            state=self.rec_state,
            container=self.rec_cont,
        )
        (out / "actors.json").write_text(json.dumps(self.actors, ensure_ascii=False), encoding="utf-8")
        with (out / "sim_events.jsonl").open("w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        (out / "sim_warnings.json").write_text(
            json.dumps(self.warnings[:500], ensure_ascii=False, indent=0), encoding="utf-8"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantics", required=True, type=Path)
    ap.add_argument("--population", required=True, type=Path)
    ap.add_argument("--plans", required=True, type=Path)
    ap.add_argument("--nav", required=True, type=Path)
    ap.add_argument("--sumo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--duration", type=float, default=None, help="override recorded duration (s) for quick tests")
    a = ap.parse_args()
    sem = json.loads(a.semantics.read_text(encoding="utf-8"))
    pop = json.loads(a.population.read_text(encoding="utf-8"))
    plans = json.loads(a.plans.read_text(encoding="utf-8"))
    sim = Sim(sem, pop, plans, a.nav, a.sumo, a.out, a.seed, a.duration)
    print(
        f"actors {sim.n}: persons {len(sim.P)}, vehicles {len(sim.Vh)}, dogs {len(sim.Dg)}, objects/machines {len(sim.Ob)}",
        flush=True,
    )
    sim.run()
    sim.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
