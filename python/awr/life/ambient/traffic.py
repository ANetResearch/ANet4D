"""Ambient vehicle traffic on the stage network: Eclipse SUMO (libsumo) drives the cars, this module only places
them, keeps their routes open and lets one direction at a time use a single-lane road.

The cars circulate for the whole scene: they are inserted before the warm-up, never leave the network (each route is
a random walk over the junction connections, extended before it runs out) and are never teleported. SUMO computes
car following (IDM), junction right of way and the speed limits of bends; a road wide enough for one car only is a
section that a car enters only when no car holds it in the opposite direction (requests are served in arrival
order), otherwise it stops at the end of the lane before it.

Positions are SUMO's lane positions mapped onto the planned geometry (stage-geometry.json); the network's simulated
lengths equal the planned ones, so the mapping is exact. A car's recorded pose is the middle of its body: half its
length behind its front along the lanes it has driven.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VEHICLE_ASSETS = ("sedan", "hatchback", "suv")
LOOKAHEAD_M = 45.0  # a car asks for a single-lane section this far ahead of it
STOP_BEFORE_M = 0.3  # a car held for a section stops this far before the end of its lane
ROUTE_AHEAD = 40  # route edges kept ahead of a car
STORAGE_M = 16.0  # an edge shorter than this next to a section cannot hold a waiting car: it joins the section
SWITCH_AFTER_S = (
    20.0  # cars keep entering a section behind their own direction until the other one has waited this long
)
LONG_SECTION_M = 30.0  # a section with more single-lane road than this is driven only by the through cars
THROUGH_SHARE = 0.25  # share of the cars that may use the long sections
IMPATIENCE_S = 20.0  # a car waiting this long at a minor link takes gaps that make the cars with priority brake


def vtypes_xml(catalog, assets=VEHICLE_ASSETS):
    """SUMO vehicle types for the catalogue's car assets (IDM; decelerations within the tape QA's 6 m/s^2)."""
    sizes = {
        a["id"]: a["dimensions_m"] for a in json.loads(Path(catalog).read_text())["assets"] if a["kind"] == "vehicle"
    }
    rows = [
        f'  <vType id="{key}" vClass="passenger" length="{sizes[key][0]:.3f}" width="{sizes[key][1]:.3f}" '
        f'height="{sizes[key][2]:.3f}" minGap="2.5" accel="1.6" decel="2.6" emergencyDecel="4.0" apparentDecel="2.6" '
        f'maxSpeed="8.33" speedFactor="normc(1,0.08,0.85,1.15)" carFollowModel="IDM" tau="1.4" delta="4" '
        f'stepping="1" jmIgnoreKeepClearTime="-1"/>'
        for key in assets
    ]
    return "<additional>\n" + "\n".join(rows) + "\n</additional>\n"


@dataclass
class Stage:
    """The compiled stage network (awr.life.ambient.roads.compile_stage) with its lane geometry."""

    root: Path
    lanes: dict = field(default_factory=dict)  # SUMO lane id -> (polyline, cumulative arc, offset into it)
    lengths: dict = field(default_factory=dict)  # SUMO lane id -> simulated length
    successors: dict = field(default_factory=dict)  # edge -> next edges
    next_lanes: dict = field(default_factory=dict)  # SUMO lane -> lanes a car can drive onto from it
    sections: dict = field(default_factory=dict)  # section edge -> section id
    alternating: set = field(default_factory=set)  # single-lane roads driven in both directions
    edge_lengths: dict = field(default_factory=dict)

    @classmethod
    def load(cls, root):
        import sumolib

        root = Path(root)
        st = cls(root)
        geo = json.loads((root / "stage-geometry.json").read_text())
        net = sumolib.net.readNet(str(root / "stage.net.xml"), withInternal=True)
        curves = {(c["source"], c["target"]): np.asarray(c["points"], float) for c in geo["connections"]}

        def polyline(pts):
            pts = np.asarray(pts, float)
            return pts, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]

        for e in net.getEdges(withInternal=False):
            pts, arc = polyline(geo["edges"][e.getID()])
            lane = e.getLanes()[0]
            st.lanes[lane.getID()] = (pts, arc, 0.0)
            st.lengths[lane.getID()] = lane.getLength()
            st.edge_lengths[e.getID()] = lane.getLength()
            for conn in lane.getOutgoing():
                st.successors.setdefault(e.getID(), []).append(conn.getTo().getID())
                pts, arc = polyline(curves[(e.getID(), conn.getTo().getID())])
                via, offset, prev = conn.getViaLaneID(), 0.0, lane.getID()
                while via:
                    inner = net.getLane(via)
                    st.lanes[via] = (pts, arc, offset)
                    st.lengths[via] = inner.getLength()
                    st.next_lanes.setdefault(prev, []).append(via)
                    offset += inner.getLength()
                    nxt = [c for c in inner.getOutgoing() if c.getViaLaneID()]
                    prev, via = via, nxt[0].getViaLaneID() if nxt else ""
                st.next_lanes.setdefault(prev, []).append(conn.getToLane().getID())
        doc = json.loads((root / "stage-network.json").read_text())
        st.alternating = set(doc["alternating"])
        single = {e: e[:-2] for b in doc["alternating"] for e in doc["roads"][b]["edges"]}
        links = [(a, b) for a, nexts in st.successors.items() for b in nexts]
        while True:
            grow = {
                e: e[:-2]
                for a, b in links
                for e in (a, b)
                if e not in single and st.edge_lengths[e] < STORAGE_M and ({a, b} - {e}) & set(single)
            }
            if not grow:
                break
            single.update(grow)
        parent = {b: b for b in set(single.values())}

        def find(b):
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            return b

        for a, b in links:
            if a in single and b in single:
                parent[find(single[a])] = find(single[b])
        for e in single:  # both lanes of a road belong to one section
            if e[:-2] in parent:
                parent[find(e[:-2])] = find(single[e])
        st.sections = {e: find(b) for e, b in single.items()}
        return st

    def between(self, a, b, depth=4):
        """The lanes a car passes from lane `a` to lane `b` when it is seen on both in consecutive steps (a lane
        shorter than a step is passed unseen); empty if `b` follows `a` directly or no short path exists."""
        paths = [[a]]
        for _ in range(depth):
            longer = []
            for path in paths:
                for nxt in self.next_lanes.get(path[-1], ()):
                    if nxt == b:
                        return path[1:]
                    longer.append(path + [nxt])
            paths = longer
        return []

    def pose(self, lane, pos):
        """World position and heading at `pos` metres along the SUMO lane `lane`."""
        pts, arc, offset = self.lanes[lane]
        s = float(np.clip(offset + pos, 0.0, arc[-1]))
        i = int(np.clip(np.searchsorted(arc, s, side="right") - 1, 0, len(arc) - 2))
        u = (s - arc[i]) / max(arc[i + 1] - arc[i], 1e-9)
        p = pts[i] + (pts[i + 1] - pts[i]) * u
        a = np.interp(max(s - 0.5, 0.0), arc, pts[:, 0]), np.interp(max(s - 0.5, 0.0), arc, pts[:, 1])
        b = np.interp(min(s + 0.5, arc[-1]), arc, pts[:, 0]), np.interp(min(s + 0.5, arc[-1]), arc, pts[:, 1])
        return p, float(np.arctan2(b[1] - a[1], b[0] - a[0]))


class Sections:
    """Single-lane sections: a car holds the section edges of its route from the entry onwards and gives up each road
    of it once its body has left the road and the junction after it. A request is granted when no holder still has one of its single-lane roads
    ahead in the opposite direction; cars of the holders' direction keep entering until a car of the other direction
    has waited SWITCH_AFTER_S, and a free section goes to the longest-waiting direction."""

    def __init__(self, stage):
        self.stage = stage
        self.holders = {}  # section -> {vid: set of (road, direction)}
        self.requests = {}  # section -> {vid: (time, traversal)}
        self.held = {}  # vid -> (section, stop edge)
        self.release = {}  # (vid, section) -> route position of the first edge after the section
        self.parts = {}  # (vid, section) -> [(route position, road, direction)] of the traversal
        self.left = {}  # (vid, section) -> {route position: odometer when the front was first seen past it}
        self.base = {}  # vid -> route positions dropped by route replacements
        self.holds = 0

    def rebase(self, vid, dropped):
        """The car's route was replaced by one starting `dropped` edges further on."""
        self.base[vid] = self.base.get(vid, 0) + dropped

    def traversal(self, route, i):
        """Roads and directions of the section entered at route[i], each with its route position, and the route
        position after the section."""
        sec = self.stage.sections[route[i]]
        out = []
        while i < len(route) and self.stage.sections.get(route[i]) == sec:
            out.append((i, route[i][:-2], route[i][-1]))
            i += 1
        return out, i

    def conflict(self, a, b):
        return any((road, "b" if d == "f" else "f") in b for road, d in a if road in self.stage.alternating)

    def step(self, traci, t):
        for vid in traci.vehicle.getIDList():
            route, k = traci.vehicle.getRoute(vid), traci.vehicle.getRouteIndex(vid)
            road = traci.vehicle.getRoadID(vid)
            # release the sections the car has driven out of (by route position: a very short edge can be
            # passed within one step)
            at = self.base.get(vid, 0) + k
            odo, length = traci.vehicle.getDistance(vid), traci.vehicle.getLength(vid)
            for sec, holders in self.holders.items():
                if vid not in holders:
                    continue
                if at >= self.release[(vid, sec)]:
                    del holders[vid]
                    continue
                # a road is left once the car has driven its length (and a metre) beyond the point where its front
                # was first seen on a later edge: past the junction after the road too, where the lanes of both
                # directions converge on the shared one (a junction within a section may also be where the way into
                # the section from outside it passes)
                left = self.left.setdefault((vid, sec), {})
                for p, _, _ in self.parts[(vid, sec)]:
                    if p not in left and at > p:
                        left[p] = odo
                parts = [q for q in self.parts[(vid, sec)] if not (q[0] in left and odo - left[q[0]] > length + 1.0)]
                if len(parts) < len(self.parts[(vid, sec)]):
                    self.parts[(vid, sec)] = parts
                    holders[vid] = {(r, d) for _, r, d in parts}
            # the next section entry ahead
            dist = (
                self.stage.edge_lengths.get(road, 0.0) - traci.vehicle.getLanePosition(vid)
                if not road.startswith(":")
                else 0.0
            )
            for j in range(k + 1, len(route)):
                sec = self.stage.sections.get(route[j])
                if sec is not None and self.stage.sections.get(route[j - 1]) != sec:
                    if vid in self.holders.get(sec, {}):
                        break
                    if dist <= LOOKAHEAD_M:
                        self.request(traci, t, vid, sec, route, j)
                    break
                if dist > LOOKAHEAD_M:
                    break
                dist += self.stage.edge_lengths.get(route[j], 0.0) + 5.0

    @staticmethod
    def behind_waiting(traci, vid, queue, holders):
        """Whether a car ahead of `vid` (through the granted cars ahead of it) still waits for the section."""
        cur = vid
        for _ in range(12):
            lead = traci.vehicle.getLeader(cur, LOOKAHEAD_M)
            if not lead or not lead[0]:
                return False
            if lead[0] in queue:
                return True
            if lead[0] not in holders:
                return False
            cur = lead[0]
        return False

    def request(self, traci, t, vid, sec, route, j):
        parts, after = self.traversal(route, j)
        base = self.base.get(vid, 0)
        want = {(r, d) for _, r, d in parts}
        queue = self.requests.setdefault(sec, {})
        since = queue.setdefault(vid, (t, want))[0]
        holders = self.holders.setdefault(sec, {})

        def alternating(use):
            return any(road in self.stage.alternating for road, _ in use)

        active = [use for use in holders.values() if alternating(use)]
        opposed = [s for other, (s, use) in queue.items() if other != vid and self.conflict(want, use)]
        if self.behind_waiting(traci, vid, queue, holders):
            blocked = True  # requests are served in the order of each approach lane
        elif not alternating(want):
            blocked = False  # only the storage-less edges of the section
        elif active:
            blocked = any(self.conflict(want, use) for use in active) or any(t - s >= SWITCH_AFTER_S for s in opposed)
        else:
            blocked = any(s < since for s in opposed)
        if not blocked:
            holders[vid] = want
            del queue[vid]
            self.release[(vid, sec)] = base + after
            self.parts[(vid, sec)] = [(base + i, r, d) for i, r, d in parts]
            self.left[(vid, sec)] = {}
            if vid in self.held:
                _, edge = self.held.pop(vid)
                try:
                    if traci.vehicle.isStopped(vid):
                        traci.vehicle.resume(vid)
                    else:
                        traci.vehicle.replaceStop(vid, 0, "")
                except Exception:
                    pass
            return
        if vid not in self.held:
            edge = route[j - 1]
            traci.vehicle.setStop(
                vid, edge, pos=max(self.stage.edge_lengths[edge] - STOP_BEFORE_M, 0.1), laneIndex=0, duration=1e6
            )
            self.held[vid] = (sec, edge)
            self.holds += 1


RECENT_EDGES = 8  # a random walk avoids the edges of its last few steps (no circling a small block)


def random_route(stage, start, rng, n, allowed=None, recent=()):
    """A random walk of `n` edges from `start` over the junction connections, within `allowed` edges if given,
    avoiding the last RECENT_EDGES edges driven when it can."""
    route = [start]
    while len(route) < n:
        nxt = [e for e in stage.successors.get(route[-1], ()) if allowed is None or e in allowed]
        if not nxt:
            break
        fresh = [e for e in nxt if e not in (list(recent) + route)[-RECENT_EDGES:]]
        nxt = fresh or nxt
        route.append(nxt[int(rng.integers(len(nxt)))])
    return route


def local_network(stage):
    """Edges of the largest strongly connected network without the single-lane roads of the long sections: the
    territory of the local cars. A queue in front of a long section can then hold at most the through cars, so it never spills
    back into the lanes the cars leaving the section need."""
    from awr.life.civil.activity.driving import _components

    length = {}
    for e, sec in stage.sections.items():
        if e[:-2] in stage.alternating:
            length[sec] = length.get(sec, 0.0) + stage.edge_lengths[e] / 2
    long = {
        e for e, sec in stage.sections.items() if length.get(sec, 0.0) > LONG_SECTION_M and e[:-2] in stage.alternating
    }
    edges = sorted(e for e in stage.edge_lengths if e not in long)
    links = {a: [b for b in stage.successors.get(a, ()) if b not in long] for a in edges}
    return set(max(_components(edges, links), key=len))


class Traffic:
    """The ambient cars of a stage in SUMO, stepped by the caller (step_s per step)."""

    def __init__(self, stage_dir, out_dir, catalog, *, vehicles=16, seed=7, step_s=0.1):
        import libsumo as traci

        self.traci, self.step_s = traci, step_s
        self.stage = stage = Stage.load(stage_dir)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "vtypes.add.xml").write_text(vtypes_xml(catalog))
        self.rng = rng = np.random.default_rng(seed)
        traci.start(
            [
                "sumo",
                "-n",
                str(Path(stage_dir) / "stage.net.xml"),
                "-a",
                str(out / "vtypes.add.xml"),
                "--step-length",
                str(step_s),
                "--no-step-log",
                "true",
                "--time-to-teleport",
                "-1",
                "--collision.action",
                "warn",
                "--collision.check-junctions",
                "true",
                "--seed",
                str(seed),
                "--default.action-step-length",
                str(step_s),
                "--time-to-impatience",
                str(IMPATIENCE_S),
                "--no-warnings",
                "true",
            ]
        )
        self.local = local = local_network(stage)
        through = max(int(round(vehicles * THROUGH_SHARE)), 1) if len(local) < len(stage.edge_lengths) else vehicles
        start = sorted(e for e in local if e not in stage.sections and stage.edge_lengths[e] > 12.0)
        slots = []
        for e in start:
            slots += [(e, p) for p in np.arange(6.0, stage.edge_lengths[e] - 1.0, 16.0)]
        order = rng.permutation(len(slots))[:vehicles]
        self.ids, self.types, self.allowed = [], [], {}
        for n, k in enumerate(sorted(order)):
            e, p = slots[k]
            vid, vt = f"car-{n:03d}", VEHICLE_ASSETS[n % len(VEHICLE_ASSETS)]
            self.allowed[vid] = (
                None
                if n % max(vehicles // through, 1) == 0 and sum(a is None for a in self.allowed.values()) < through
                else local
            )
            traci.route.add("r-" + vid, random_route(stage, e, rng, ROUTE_AHEAD, self.allowed[vid]))
            traci.vehicle.add(vid, "r-" + vid, typeID=vt, depart="0", departPos=str(p), departSpeed="0", departLane="0")
            self.ids.append(vid)
            self.types.append(vt)
        self.sections = Sections(stage)
        self.length = {vt: float(traci.vehicletype.getLength(vt)) for vt in VEHICLE_ASSETS}
        self.width = {vt: float(traci.vehicletype.getWidth(vt)) for vt in VEHICLE_ASSETS}
        self.history = {vid: [] for vid in self.ids}
        self.collisions, self.t = 0, 0.0
        self.step()  # the cars enter the network with the first step

    def step(self):
        """Advance one step; keep the routes open and the single-lane sections managed."""
        traci, stage = self.traci, self.stage
        traci.simulationStep()
        self.t += self.step_s
        if len(traci.vehicle.getIDList()) != len(self.ids):
            raise RuntimeError(f"cars missing at {self.t:.1f} s")
        self.collisions += traci.simulation.getCollidingVehiclesNumber()
        self.sections.step(traci, self.t)
        for vid in self.ids:
            route, r = traci.vehicle.getRoute(vid), traci.vehicle.getRouteIndex(vid)
            if len(route) - r < ROUTE_AHEAD // 2:
                traci.vehicle.setRoute(
                    vid,
                    list(route[r:])
                    + random_route(stage, route[-1], self.rng, ROUTE_AHEAD, self.allowed[vid], route[:-1])[1:],
                )
                # SUMO may keep the edges already driven in front of the new route (the route index stays): the
                # positions dropped are those by which the index moved back
                self.sections.rebase(vid, r - traci.vehicle.getRouteIndex(vid))
            lane = traci.vehicle.getLaneID(vid)
            h = self.history[vid]
            if not h or h[-1] != lane:
                if h:
                    h.extend(stage.between(h[-1], lane))
                h.append(lane)
            if len(h) > 64:
                del h[:-32]

    def poses(self):
        """Per car: middle of the body (x, y), heading (rad) and speed (m/s). The middle lies half a length behind
        the front along the lanes driven."""
        traci, stage = self.traci, self.stage
        xy, yaw, v = np.zeros((len(self.ids), 2)), np.zeros(len(self.ids)), np.zeros(len(self.ids))
        for i, vid in enumerate(self.ids):
            h, pos = self.history[vid], traci.vehicle.getLanePosition(vid)
            back, j, p = self.length[self.types[i]] / 2, len(h) - 1, pos
            while p < back and j > 0:
                back -= p
                j -= 1
                p = stage.lengths[h[j]]
            xy[i], yaw[i] = stage.pose(h[j], max(p - back, 0.0))
            v[i] = traci.vehicle.getSpeed(vid)
        return xy, yaw, v

    def ahead(self, vid, distance, spacing=1.0):
        """Points of the car's path from its front onwards, `distance` metres, every `spacing`: (n, 2) and the arc
        of each from the front."""
        traci, stage = self.traci, self.stage
        lane, pos = traci.vehicle.getLaneID(vid), traci.vehicle.getLanePosition(vid)
        route, k = traci.vehicle.getRoute(vid), traci.vehicle.getRouteIndex(vid)
        lanes, start = [lane], pos
        # the lanes ahead: the rest of a junction's internal lanes, then edge lane, via lanes, edge lane, ...
        cur = lane
        edges = list(route[k + 1 :])
        while sum(stage.lengths[l] for l in lanes) - start < distance and (stage.next_lanes.get(cur) or edges):
            options = stage.next_lanes.get(cur, [])
            if cur.startswith(":"):
                nxt = options[0] if options else None
                if nxt is not None and edges and nxt == edges[0] + "_0":
                    edges.pop(0)  # (in a junction the route index is still that of the edge before it)
            else:
                if not edges:
                    break
                target = edges.pop(0) + "_0"
                nxt = None
                for o in options:  # the via lane chain towards the next route edge
                    probe = o
                    while probe.startswith(":"):
                        probe = stage.next_lanes[probe][0]
                    if probe == target:
                        nxt = o
                        break
            if nxt is None:
                break
            lanes.append(nxt)
            cur = nxt
        pts, arcs, base = [], [], -start
        for l in lanes:
            L = stage.lengths[l]
            for q in np.arange(max(0.0, -base) if l == lanes[0] else 0.0, L, spacing):
                a = base + q
                if a > distance:
                    break
                pts.append(stage.pose(l, q)[0])
                arcs.append(a)
            base += L
        return np.array(pts).reshape(-1, 2), np.array(arcs)

    def close(self):
        self.traci.close()


def run(stage_dir, out_dir, catalog, *, vehicles=16, warmup_s=120.0, duration_s=900.0, seed=7, step_s=0.1):
    """Run the ambient traffic alone and write `out_dir`/traffic.npz: per recorded tick (step_s) and car the middle
    of the body (x, y), heading (rad) and SUMO speed, plus the cars' ids and vehicle types."""
    tr = Traffic(stage_dir, out_dir, catalog, vehicles=vehicles, seed=seed, step_s=step_s)
    steps, first = int(round((warmup_s + duration_s) / step_s)), int(round(warmup_s / step_s))
    n = len(tr.ids)
    xy = np.full((steps - first + 1, n, 2), np.nan)
    yaw = np.full((steps - first + 1, n), np.nan)
    speed = np.zeros((steps - first + 1, n))
    waits = np.zeros(n)
    for k in range(steps + 1):
        if k:
            tr.step()
        if k >= first:
            xy[k - first], yaw[k - first], speed[k - first] = tr.poses()
            waits += [step_s if tr.traci.vehicle.isStopped(v) else 0.0 for v in tr.ids]
    tr.close()
    out = Path(out_dir)
    np.savez_compressed(
        out / "traffic.npz", xy=xy, yaw=yaw, speed=speed, ids=np.array(tr.ids), types=np.array(tr.types), step_s=step_s
    )
    report = dict(
        vehicles=n,
        through=[v for v in tr.ids if tr.allowed[v] is None],
        local_edges=len(tr.local),
        collisions_reported=int(tr.collisions),
        section_holds=tr.sections.holds,
        section_wait_s=np.round(waits, 1).tolist(),
    )
    (out / "traffic-report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report
