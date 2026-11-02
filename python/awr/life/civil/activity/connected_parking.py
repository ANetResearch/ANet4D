"""Parked cars that never move keep the driving map's roads passable.

A car parked for the whole recording blocks the edges of the driving map whose swept corridor its body intersects
(the occupancy test of awr.life.civil.activity.driving.DrivingRoutes: the body with its clearance against the corridor
of the map's car). ConnectedRoads admits such a car only where it cuts no road: any two lane nodes of the map that
are mutually reachable before (in the strongly connected parts holding at least two berths, the parts drivers can
use) stay mutually reachable after, except the lane nodes of the stretch the car itself stands across: a car beside
the kerb may close the stretch of one lane it stands on where another lane passes it. That stretch is the lane nodes
of the lanes the body blocks (the lane ends of the edges it blocks, and the nodes of those lanes within
STRETCH_M of it) from which every way on along their part is blocked or leads only into the stretch, or to which
every way in is blocked or comes only from it: the map joins a lane to the lanes beside it at waypoints about 10 m
apart over joins up to about 30 m long, so the lane nodes just before and after a parked car keep a join or two into
the closed stretch and none out of it. Edges to berths do not count as ways on (a berth is a terminal). Any other
lane whose ways all pass the car is on a road the car cuts: a whole street closed, a part of the network one can
enter and not leave (cruise5 r11f v6: with this stretch 127 of the 2523 berths of r11f cut a road with the cars of
the start in place, against 276 when only the nodes all of whose ways the body blocks were its stretch). Berths next to the car may become
unusable, as at a full kerb; the road itself stays open, and the drivers' round trips
(awr.life.civil.activity.parking_admission) are admitted on what remains.

The same rule holds for a car parking during the run (`keeps`, used by DrivingRoutes.keeps_roads): with the bodies
standing then, its body may cut no road those bodies leave. A car that parks across the only way along a street
sends every car behind it round by a turn back that holds the street both ways (cruise5 r11f v5, vehicle-046).

The corridors are shared with the drivers' admission (`edge_corridor`); nothing is moved and no check is relaxed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely import points
from shapely.strtree import STRtree

from ..curb_allocation import swept

STRETCH_M = 40.0  # reach of the stretch a parked car closes along a lane it blocks: its length and one long join


def edge_corridor(graph, size, edge):
    """The swept corridor of `edge` for a car of `size` (cached on the graph: the map is frozen)."""
    cache = graph.__dict__.setdefault("_corridors", {})
    key = (edge, tuple(float(v) for v in size))
    if key not in cache:
        cache[key] = swept(graph.edges[edge]["route"], size)
    return cache[key]


class ConnectedRoads:
    def __init__(self, graph, size, bodies=()):
        """`graph`: the driving map; `size`: its car's body; `bodies`: footprints (with clearance) of the bodies
        already standing (they may already block edges; the roads they leave are the ones kept)."""
        self.graph, self.size = graph, tuple(float(v) for v in size)
        initial = {n: k for k, component in enumerate(graph.components()) for n in component}
        berths = Counter(initial[n] for n, d in graph.nodes.items() if d["kind"] == "berth")
        used = {k for k, count in berths.items() if count >= 2}
        self.edge_ids = sorted(
            e
            for e, d in graph.edges.items()
            if initial[d["source"]] in used and initial[d["target"]] == initial[d["source"]]
        )
        self.tree = STRtree([edge_corridor(graph, self.size, e) for e in self.edge_ids])
        self.lanes = sorted(n for n, d in graph.nodes.items() if d["kind"] != "berth" and initial[n] in used)
        # The lane parts are the strongly connected parts of the edges between these lanes: berths are terminals
        # (DrivingGraph.components), and a cycle through two lanes of one part stays inside it.
        index = {n: i for i, n in enumerate(self.lanes)}
        links = [
            (e, index[graph.edges[e]["source"]], index[graph.edges[e]["target"]])
            for e in self.edge_ids
            if graph.edges[e]["source"] in index and graph.edges[e]["target"] in index
        ]
        self._link = {e: k for k, (e, _, _) in enumerate(links)}
        self._ends = (np.array([a for _, a, _ in links], np.int64), np.array([b for _, _, b in links], np.int64))
        self._ahead, self._behind = {}, {}  # lane -> its edges to / from lanes
        for e, a, b in links:
            self._ahead.setdefault(self.lanes[a], []).append(e)
            self._behind.setdefault(self.lanes[b], []).append(e)
        self._lane_set = set(self.lanes)
        self._lane_tree = STRtree(points([graph.nodes[n]["pose"][:2] for n in self.lanes])) if self.lanes else None
        self.blocked = set()
        for body in bodies:
            self.blocked |= self._hits(body)
        self.labels = self._labels(self.blocked)
        self._last = None
        self.report = dict(edges=len(self.edge_ids), lanes=len(self.lanes), checked=0, refused=0, searches=0)

    def _hits(self, body):
        return {self.edge_ids[int(k)] for k in self.tree.query(body, predicate="intersects")}

    def _labels(self, blocked):
        """The lane part of each lane with the edges `blocked` removed."""
        keep = np.ones(len(self._link), bool)
        keep[[self._link[e] for e in blocked if e in self._link]] = False
        n = len(self.lanes)
        rows, cols = self._ends[0][keep], self._ends[1][keep]
        _, labels = connected_components(
            csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=True, connection="strong"
        )
        return dict(zip(self.lanes, labels.tolist(), strict=False))

    def _after(self, hit):
        key = frozenset(hit)
        if self._last is None or self._last[0] != key:
            self.report["searches"] += 1
            self._last = (key, self._labels(self.blocked | hit))
        return self._last[1]

    def allows(self, body):
        """Whether a car parked with this footprint (with clearance) keeps every pair of mutually reachable lanes."""
        self.report["checked"] += 1
        hit = self._hits(body) - self.blocked
        if not hit:
            return True
        if self._splits(body, self.blocked, self.labels, hit, self._after(hit)):
            self.report["refused"] += 1
            return False
        return True

    def standing(self, bodies):
        """The roads the footprints `bodies` (with clearance) leave: the edges they block and the lane parts (`keeps`).
        Independent of the bodies recorded at construction and by `reserve`."""
        blocked = frozenset().union(*(self._hits(b) for b in bodies))
        return blocked, self._labels(blocked)

    def keeps(self, body, standing):
        """Whether a car parked with footprint `body` keeps every pair of lanes that are mutually reachable with the
        bodies of `standing` (`standing`) in place, as `allows` does for the bodies recorded here."""
        blocked, labels = standing
        hit = self._hits(body) - blocked
        return not hit or not self._splits(body, blocked, labels, hit, self._labels(blocked | hit))

    def _splits(self, body, blocked, labels, hit, after):
        """Whether two lanes of one part (`labels`) fall into different parts (`after`) once the edges `hit` are
        blocked too, other than the lanes of the stretch the body closes (`_stretch`), or a part of two lanes or more
        lies wholly in that stretch."""
        stretch = self._stretch(body, blocked | hit, labels, hit)
        groups = {}
        for n in self.lanes:
            if n not in stretch:
                groups.setdefault(labels[n], set()).add(after[n])
        if any(len(s) > 1 for s in groups.values()):
            return True
        # A part every lane of which lies in the stretch is closed as a whole, though no two lanes outside the stretch
        # are left to fall apart (codex DK: a two-node road between the ends of the edges a body blocks; a body over a
        # whole part).
        closed = Counter(labels[n] for n in stretch if labels[n] not in groups)
        return any(count > 1 for count in closed.values())

    def _stretch(self, body, blocked, labels, hit):
        """The lane nodes of the lanes `body` blocks (lane ends of the edges `hit`, and the nodes of their lanes
        within STRETCH_M of it) whose every edge on to a lane of their part (`labels`), or every edge in from one, is
        `blocked` or meets another such node from the same side: the stretch of a lane the body closes (module
        docstring)."""
        nodes = self.graph.nodes
        near = {self.graph.edges[e][end] for e in hit for end in ("source", "target")} & self._lane_set
        lanes = {
            (nodes[n].get("fragment"), nodes[n].get("direction")) for n in near if nodes[n].get("fragment") is not None
        }
        if lanes and self._lane_tree is not None:
            near |= {
                n
                for n in (self.lanes[int(k)] for k in self._lane_tree.query(body.buffer(STRETCH_M)))
                if (nodes[n].get("fragment"), nodes[n].get("direction")) in lanes
            }
        ahead, behind = set(), set()  # nodes with no way on / no way in outside the stretch
        changed = True
        while changed:
            changed = False
            for n in near:
                if n not in ahead and all(
                    e in blocked
                    or self.graph.edges[e]["target"] in ahead
                    or labels[self.graph.edges[e]["target"]] != labels[n]
                    for e in self._ahead.get(n, ())
                ):
                    ahead.add(n)
                    changed = True
                if n not in behind and all(
                    e in blocked
                    or self.graph.edges[e]["source"] in behind
                    or labels[self.graph.edges[e]["source"]] != labels[n]
                    for e in self._behind.get(n, ())
                ):
                    behind.add(n)
                    changed = True
        return ahead | behind

    def reserve(self, body):
        """Record a parked car (after `allows`)."""
        hit = self._hits(body) - self.blocked
        if hit:
            after = self._after(hit)
            self.blocked |= hit
            self.labels = after
