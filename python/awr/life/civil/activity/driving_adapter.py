"""Resident drivers on a verified driving map: berths with car-side access, home berths and owned cars.

An environment adapter for the shared Director (awr.life.civil.activity.director). Its inputs are scene features
only: a driving map compiled on the scene's registered layers (berth poses with measured entry and exit curves,
awr.life.civil.driving_graph), the walking network with its static router, and the bodies already admitted. It does
not move an admitted body, invent a berth, shorten a measured walk or relax a driving check. A resident whose home
has no free berth with a measured walk to it remains a pedestrian; the number of drivers is what the scene allows,
up to the number requested.

- A berth is usable when its body is clear of every admitted body and keeps the walking passages open, and one of
  its car-side anchors (awr.life.civil.activity.access.side_access_candidates) stands on free, supported walking
  ground with room to turn. The anchor off the mapped carriageway is preferred (the kerb side of a curb berth).
- Only berths in a strongly connected part of the driving map with at least two berths are used: a car parked
  there can leave for another berth and come back (not a directed dead end).
- A driver's home berth is the nearest usable berth whose anchor the resident reaches on foot, there and back,
  within `home_walk_m`. Its home parking is that berth and the nearest other usable berths within `home_walk_m`:
  a home berth is public, so a returning driver whose berth was taken parks nearby and walks home.
- A facility's parking candidates are the nearest usable berths in each strongly connected road component,
  within `facility_walk_m` of it in a straight line. A closer disconnected lot must not hide reachable parking.
  They are candidates only: the Director plans the walk from the berth and the drive to it before every journey.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from shapely.geometry import Point

from ..curb_allocation import footprint
from .access import side_access_candidates
from .director import Place, Vehicle


def strongly_connected_berths(graph):
    """Berths in strongly connected components of the driving map holding at least two berths."""
    keep = set()
    for component in graph.components():
        berths = [n for n in component if graph.nodes[n]["kind"] == "berth"]
        if len(berths) >= 2:
            keep.update(berths)
    return keep


class DrivingAdapter:
    def __init__(
        self,
        graph,
        size,
        asset,
        domain,
        network,
        surface,
        turn_radius,
        *,
        home_walk_m=150.0,
        facility_walk_m=200.0,
        facility_berths=4,
    ):
        self.graph = graph
        self.size = np.asarray(size, float)
        self.asset = asset
        self.domain = domain
        self.network = network
        self.surface = surface
        self.turn_radius = float(turn_radius)
        self.home_walk_m = float(home_walk_m)
        self.facility_walk_m = float(facility_walk_m)
        self.facility_berths = int(facility_berths)
        # The adapter consumes a compiled, fixed topology; live occupied bodies
        # are checked by DrivingRoutes, not by relabelling the whole graph.
        self.component_of = {name: label for label, component in enumerate(graph.components()) for name in component}
        self.report = dict(berths=0, connected=0, body_clear=0, passage_kept=0, with_access=0)

    # -- berths ------------------------------------------------------------------------------------------------
    def usable_berths(self, bodies, free, turn_clear, keeps_passage, parking_check=None):
        """{berth: (pose, access_xy)} for berths whose body fits now and whose car side is walkable.

        `bodies`: footprints (with clearance) of every admitted body; `free`/`turn_clear`: the walking network's
        static masks with those bodies; `keeps_passage(xy, yaw, size)`: the builder's pavement passage check."""
        connected = strongly_connected_berths(self.graph)
        self.report["parking_rule_rejected"] = 0
        self.report["parking_rule_checked"] = parking_check is not None
        from shapely.strtree import STRtree

        tree = STRtree(bodies) if bodies else None
        usable = {}
        for name, node in sorted(self.graph.nodes.items()):
            if node["kind"] != "berth":
                continue
            self.report["berths"] += 1
            if name not in connected:
                continue
            self.report["connected"] += 1
            pose = np.asarray(node["pose"], float)
            if parking_check is not None:
                parking = parking_check(pose[:2], float(pose[2]), self.size)
                if parking.get("legal") is not True:
                    self.report["parking_rule_rejected"] += 1
                    continue
                # The execution recorder reads this measured rule result when
                # an actual journey ends. This annotates the in-memory graph,
                # never rewrites the compiler's source document.
                node["parking_legality"] = dict(parking, source="scene_parking_rules")
            body = footprint(dict(xy=pose[:2], yaw=pose[2]), self.size)
            if tree is not None and len(tree.query(body, predicate="intersects")):
                continue
            self.report["body_clear"] += 1
            if not keeps_passage(pose[:2], float(pose[2]), self.size):
                continue
            self.report["passage_kept"] += 1
            anchors = []
            for candidate in side_access_candidates(pose, self.size, self.turn_radius):
                xy = np.asarray(candidate["xy"], float)
                cell = self.network.cell(xy)
                if np.any(cell < 0) or np.any(cell >= np.asarray(free.shape)) or not free[tuple(cell)]:
                    continue
                if turn_clear is not None and not turn_clear[tuple(cell)]:
                    continue
                ring = xy + self.turn_radius * np.column_stack(
                    [np.cos(np.linspace(0.0, 2 * math.pi, 49)), np.sin(np.linspace(0.0, 2 * math.pi, 49))]
                )
                if not self.surface.supported(np.vstack([xy, ring]), "person").all():
                    continue
                anchors.append((bool(self.domain.covers(Point(xy))), candidate["side"] != "left", tuple(xy)))
            if anchors:
                self.report["with_access"] += 1
                usable[name] = (pose, min(anchors)[2])
        return usable

    # -- drivers -----------------------------------------------------------------------------------------------
    def assign(self, residents, usable, walk, count, admit):
        """Home berths for up to `count` residents: [(actor, home_place, berth)].

        `residents`: [(actor_id, home Place)] in a deterministic order; `walk(start, end)`: measured walking route
        length in metres or None; `admit(actor, berth)`: places the owned car at the berth if it still fits beside
        the cars already placed (bodies and walking passages), returning whether it did. Each berth serves one car."""
        if count <= 0 or not usable:
            return []
        names = sorted(usable)
        xy = np.array([usable[n][1] for n in names])
        taken, chosen = set(), []
        rows = dict(residents=len(residents), without_berth_in_reach=0, without_walk=0, car_not_admitted=0, assigned=0)
        for actor, home in residents:
            if len(chosen) >= count:
                break
            d = np.linalg.norm(xy - np.asarray(home.xy, float), axis=1)
            order = [k for k in np.argsort(d, kind="stable") if d[k] <= self.home_walk_m and names[k] not in taken]
            if not order:
                rows["without_berth_in_reach"] += 1
                continue
            found, walked = None, False
            for k in order[:6]:
                name = names[k]
                there = walk(home.xy, usable[name][1])
                back = walk(usable[name][1], home.xy) if there is not None else None
                if there is None or back is None or max(there, back) > self.home_walk_m:
                    continue
                walked = True
                if admit(actor, name):
                    found = name
                    break
            if found is None:
                rows["without_walk" if not walked else "car_not_admitted"] += 1
                continue
            taken.add(found)
            chosen.append((actor, home, found))
        rows["assigned"] = len(chosen)
        self.report["assignment"] = rows
        return chosen

    def parking_candidates(self, place, usable):
        """Nearby candidates per connected component, shared by admission and binding."""
        if place.kind in ("home", "berth"):
            return ()
        return self._nearest(place.xy, usable, self.facility_walk_m)

    def _nearest(self, xy0, usable, radius):
        """The nearest usable berths within `radius` (straight line), at most facility_berths per component."""
        names = sorted(usable)
        if not names:
            return ()
        xy = np.array([usable[n][1] for n in names])
        d = np.linalg.norm(xy - np.asarray(xy0, float), axis=1)
        selected, counts = [], {}
        for k in np.argsort(d, kind="stable"):
            part = self.component_of[names[k]]
            if d[k] <= radius and counts.get(part, 0) < self.facility_berths:
                selected.append(names[k])
                counts[part] = counts.get(part, 0) + 1
        return tuple(sorted(selected))

    def home_parking(self, home, berth, usable):
        """Keep the admitted home berth and nearby alternatives, without reserving them.

        As for facilities, the radius is a candidate search, not evidence of a
        traversable route. The Director checks live capacity, driving access and
        the final walk before it reserves a destination. This adapter consumes
        shared parking; reserved/private berths need explicit access policy.
        """
        return (berth,) + tuple(n for n in self._nearest(home.xy, usable, self.home_walk_m) if n != berth)

    def bind(self, director, usable, chosen, vehicle_ids):
        """Add berth places, parking candidates and the owned cars to the Director."""
        names = sorted(usable)
        for name in names:
            pose, access = usable[name]
            director.places[name] = Place(name, (float(pose[0]), float(pose[1])), "berth", (name,), access_xy=access)
        home_berth = {home.id: berth for _, home, berth in chosen}
        linked = 0
        for pid, place in list(director.places.items()):
            if place.kind == "berth":
                continue
            if place.kind == "home":
                parking = self.home_parking(place, home_berth[pid], usable) if pid in home_berth else ()
            else:
                parking = self.parking_candidates(place, usable)
            linked += bool(parking)
            director.places[pid] = replace(place, parking=parking)
        for (actor, _, berth), vid in zip(chosen, vehicle_ids, strict=False):
            director.actors[actor].vehicle = vid
            director.add_vehicle(Vehicle(vid, actor, berth))
        self.report.update(places_with_parking=linked, drivers=len(chosen))
