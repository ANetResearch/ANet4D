"""Joint initial parking: every admitted owner retains a usable round trip.

This is a static admission check, not traffic execution or a capacity maximum.
It checks the same physical edges as DrivingRoutes, then recomputes directed
connectivity under the current collection of parked bodies. A candidate may
not strand an earlier owner. Actual departure still needs live admission.
"""

import time
from collections import Counter

from shapely.strtree import STRtree

from ..curb_allocation import footprint
from ..vehicle_connections import admit
from .connected_parking import edge_corridor


class JointParkingAccess:
    def __init__(self, graph, surface, size, area, bodies, homes, targets, destination_access):
        self.graph, self.size = graph, size
        self.targets = set(targets)
        self.destination_access = destination_access
        self.access_cache = {}
        self.selected = {}
        self.report = dict(requests=0, accepted=0, no_candidate_round_trip=0, would_strand_owner=0)
        initial = {n: i for i, group in enumerate(graph.components()) for n in group}
        usable_components = {initial[n] for n in homes if any(initial[n] == initial[b] and b != n for b in targets)}
        self.home_components = initial
        started = time.perf_counter()
        invalid = {}
        edge_ids = []
        corridors = []
        for eid, e in sorted(graph.edges.items()):
            if initial[e["source"]] not in usable_components or initial[e["source"]] != initial[e["target"]]:
                continue
            ok, reason = admit(surface, e["route"], size, allowed_area=area)
            if not ok:
                invalid[eid] = reason
            else:
                edge_ids.append(eid)
                corridors.append(edge_corridor(graph, size, eid))
        self.edge_ids = edge_ids
        self.tree = STRtree(corridors)
        self.base_blocked = set(invalid)
        for body in bodies:
            self.base_blocked.update(self._blocked_by(body))
        self.report.update(
            edge_checks=len(invalid) + len(edge_ids),
            invalid_edges=dict(Counter(invalid.values())),
            background_blocked_edges=len(self.base_blocked) - len(invalid),
            prepare_wall_s=time.perf_counter() - started,
        )

    def _blocked_by(self, body):
        return {self.edge_ids[int(i)] for i in self.tree.query(body, predicate="intersects")}

    def _access(self, actor, target):
        key = (actor, target)
        if key not in self.access_cache:
            self.access_cache[key] = bool(self.destination_access(actor, target))
        return self.access_cache[key]

    def accept(self, actor, home):
        if actor in self.selected:
            raise ValueError("Owner already admitted")
        self.report["requests"] += 1
        pose = self.graph.nodes[home]["pose"]
        candidate = dict(home=home, blocked=self._blocked_by(footprint(dict(xy=pose[:2], yaw=pose[2]), self.size)))
        proposed = {**self.selected, actor: candidate}
        # Check the new owner first, then preserve every earlier owner's access.
        for owner in [actor] + sorted(self.selected):
            source = proposed[owner]["home"]
            goals = [
                b
                for b in sorted(self.targets)
                if b != source and self.home_components[b] == self.home_components[source]
            ]
            blocked = self.base_blocked | set().union(
                *(row["blocked"] for who, row in proposed.items() if who != owner)
            )
            component = next(group for group in self.graph.components(blocked_edges=blocked) if source in group)
            if not any(b in component and self._access(owner, b) for b in goals):
                self.report["no_candidate_round_trip" if owner == actor else "would_strand_owner"] += 1
                return False
        self.selected = proposed
        self.report["accepted"] += 1
        return True
