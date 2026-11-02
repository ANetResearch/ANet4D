"""Vehicle layer: an in-process SUMO (libsumo) stepped at 10 Hz in lockstep with the pedestrian engine.

Vehicles enter SUMO when a trip departs and leave it when they arrive (SUMO removes arrived vehicles); between trips
they are "parked" by the engine at a slot off the carriageway. Short in-lane stops (breakdown, double parking,
bus stop) are SUMO stops on the route. Positions returned are the vehicle centre in the world frame with yaw in
degrees counter-clockwise from +x.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path

MAX_SPEED = {
    "passenger": 50.0,
    "delivery": 40.0,
    "truck": 30.0,
    "bus": 25.0,
    "taxi": 50.0,
    "emergency": 55.0,
    "motorcycle": 50.0,
    "bicycle": 7.0,
}
ACCEL = {
    "passenger": 2.6,
    "delivery": 2.0,
    "truck": 1.3,
    "bus": 1.2,
    "taxi": 2.6,
    "emergency": 3.0,
    "motorcycle": 3.5,
    "bicycle": 1.2,
}


class Traffic:
    def __init__(self, sumo_dir: Path, seed: int, step_s: float = 0.1) -> None:
        import libsumo as ls
        import sumolib

        self.ls = ls
        self.tc = ls.constants
        self.step_s = step_s
        self.net = sumolib.net.readNet(str(sumo_dir / "net.net.xml"))
        ls.start(
            [
                "sumo",
                "-n",
                str(sumo_dir / "net.net.xml"),
                "--step-length",
                f"{step_s}",
                "--no-step-log",
                "true",
                "--no-warnings",
                "true",
                "--seed",
                str(seed),
                "--time-to-teleport",
                "-1",
                "--collision.action",
                "warn",
                "--default.speeddev",
                "0.1",
                "--ignore-route-errors",
                "true",
                "--max-depart-delay",
                "-1",
                "--eager-insert",
                "true",
            ]
        )
        self.types: set[str] = set()
        self.info: dict[str, dict] = {}  # vid → {length, vclass, lane_idx cache}
        self.n_routes = 0
        self._lane_idx: dict[str, int] = {}

    def ensure_type(self, vtype: str, vclass: str, size, rgb, speed_factor: float = 1.0) -> str:
        tid = f"{vtype}_{int(speed_factor * 100)}"
        if tid in self.types:
            return tid
        ls = self.ls
        ls.vehicletype.copy("DEFAULT_VEHTYPE", tid)
        ls.vehicletype.setVehicleClass(tid, vclass)
        ls.vehicletype.setLength(tid, float(size[0]))
        ls.vehicletype.setWidth(tid, float(size[1]))
        ls.vehicletype.setHeight(tid, float(size[2]))
        ls.vehicletype.setMaxSpeed(tid, MAX_SPEED.get(vclass, 40.0))
        ls.vehicletype.setAccel(tid, ACCEL.get(vclass, 2.5))
        ls.vehicletype.setDecel(tid, 4.5)
        ls.vehicletype.setSpeedFactor(tid, speed_factor)
        ls.vehicletype.setMinGap(tid, 2.0 if vclass != "bicycle" else 0.8)
        ls.vehicletype.setColor(tid, (int(rgb[0]), int(rgb[1]), int(rgb[2]), 255))
        self.types.add(tid)
        return tid

    def lane_index(self, edge_id: str, vclass: str) -> int:
        key = f"{edge_id}|{vclass}"
        if key in self._lane_idx:
            return self._lane_idx[key]
        e = self.net.getEdge(edge_id)
        idx = 0
        for i, L in enumerate(e.getLanes()):
            if L.allows(vclass):
                idx = i
                break
        self._lane_idx[key] = idx
        return idx

    def edge_len(self, edge_id: str) -> float:
        return float(self.net.getEdge(edge_id).getLength())

    def nearest_edge(self, xy, vclass: str = "passenger", r: float = 40.0):
        """(edge_id, pos) of the closest driving lane to xy."""
        best = None
        for e, d in self.net.getNeighboringEdges(xy[0], xy[1], r):
            if e.getFunction() or not e.allows(vclass):
                continue
            if best is None or d < best[1]:
                best = (e, d)
        if best is None:
            return None
        e = best[0]
        for L in e.getLanes():
            if L.allows(vclass):
                pos, _ = L.getClosestLanePosAndDist(xy)
                return e.getID(), float(min(max(pos, 1.0), L.getLength() - 1.0))
        return None

    def add_trip(
        self,
        vid: str,
        tid: str,
        vclass: str,
        length: float,
        frm: tuple[str, float],
        to: tuple[str, float],
        stops: list | None = None,
    ) -> bool:
        ls = self.ls
        try:
            st = ls.simulation.findRoute(frm[0], to[0], tid)
        except Exception:
            return False
        edges = list(st.edges)
        if not edges:
            return False
        rid = f"r{self.n_routes}"
        self.n_routes += 1
        ls.route.add(rid, edges)
        dpos = min(max(frm[1], 1.0), self.edge_len(edges[0]) - 2.0)
        apos = min(max(to[1], 2.0), self.edge_len(edges[-1]) - 1.0)
        if len(edges) == 1 and apos <= dpos + 5.0:
            apos = min(dpos + 5.0, self.edge_len(edges[0]) - 1.0)
        try:
            ls.vehicle.add(
                vid,
                rid,
                typeID=tid,
                depart="now",
                departLane=str(self.lane_index(edges[0], vclass)),
                departPos=f"{dpos:.1f}",
                departSpeed="0",
                arrivalPos=f"{apos:.1f}",
            )
        except Exception as e:  # pragma: no cover
            return False
        for s in stops or []:
            edge, pos, dur = s
            if edge not in edges:
                continue
            if edge == edges[-1] and pos >= apos - 3.0:
                pos = max(2.0, apos - 10.0)
            if edge == edges[0] and pos <= dpos + 3.0:
                continue
            with contextlib.suppress(Exception):
                ls.vehicle.setStop(
                    vid, edge, pos=float(pos), laneIndex=self.lane_index(edge, vclass), duration=float(dur)
                )
        ls.vehicle.subscribe(
            vid,
            varIDs=[
                self.tc.VAR_POSITION,
                self.tc.VAR_ANGLE,
                self.tc.VAR_SPEED,
                self.tc.VAR_ROAD_ID,
                self.tc.VAR_LANEPOSITION,
                self.tc.VAR_STOPSTATE,
            ],
        )
        self.info[vid] = {"length": length, "vclass": vclass, "edges": edges}
        return True

    def step(self) -> tuple[dict, list, list]:
        """Advance one step; returns ({vid: (x, y, yaw_deg, speed, road, lanepos, stopped)}, departed, arrived)."""
        ls, tc = self.ls, self.tc
        ls.simulationStep()
        teleports = list(ls.simulation.getStartingTeleportIDList())
        collisions = list(ls.simulation.getCollidingVehiclesIDList())
        if teleports or collisions:
            raise RuntimeError(f"Invalid traffic state: teleports={teleports}, collisions={collisions}")
        res = ls.vehicle.getAllSubscriptionResults()
        out = {}
        for vid, r in res.items():
            x, y = r[tc.VAR_POSITION]
            ang = r[tc.VAR_ANGLE]
            yaw = 90.0 - ang
            L = self.info.get(vid, {}).get("length", 4.5)
            hx, hy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            out[vid] = (
                x - hx * L / 2.0,
                y - hy * L / 2.0,
                yaw,
                r[tc.VAR_SPEED],
                r[tc.VAR_ROAD_ID],
                r[tc.VAR_LANEPOSITION],
                bool(r[tc.VAR_STOPSTATE] & 1),
            )
        return out, list(ls.simulation.getDepartedIDList()), list(ls.simulation.getArrivedIDList())

    def remove(self, vid: str) -> None:
        with contextlib.suppress(Exception):
            self.ls.vehicle.remove(vid)

    def time(self) -> float:
        return float(self.ls.simulation.getTime())

    def close(self) -> None:
        self.ls.close()
