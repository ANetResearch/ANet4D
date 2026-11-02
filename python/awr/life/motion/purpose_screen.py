"""Offline home/service/mode screening before costly road compilation.

Consumes bounded, historical car-side walking witnesses, not mesh doors. A cached
directed graph joins exact witnessed endpoints without any new grid searches. These
comparisons nominate candidates; neither missing walking evidence nor a topology
lower bound certifies a journey. No actor schedule or pose is generated here.
"""

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from awr.life.civil.activity.access import bind_parking_access


@dataclass(frozen=True)
class PurposePolicy:
    walk_mps: float = 1.2
    max_drive_mps: float = 2.0
    transfer_s: float = 8.0  # Existing director: board 4 s + alight 4 s per trip.
    min_home_service_separation_m: float = 50.0

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Positive finite purpose screening policy required")


class PurposeScreen:
    """Select a consistent witnessed side per berth; never mix opposite sides.

    Caller binds the access/places source files. It may provide a subset of the
    historical berth roster, but every selected pose/anchor/body/facility/path
    must match exactly. Changed road geometry still needs fresh physical proof.
    """

    def __init__(self, graph, places, report, *, asset, size, policy=PurposePolicy()):
        if not isinstance(policy, PurposePolicy):
            raise ValueError("Explicit purpose policy required")
        places = list(places)
        self.policy = policy
        self.graph = graph
        self.places = {p.id: p for p in places}
        rows = report["rows"]
        by_id = {r["berth"]: r for r in rows}
        if len(by_id) != len(rows):
            raise ValueError("Duplicate historical berth")
        names = {n for n, v in graph.nodes.items() if v["kind"] == "berth"}
        if names - set(by_id):
            raise ValueError("Missing historical berth access")
        subset = {**report, "rows": [by_id[n] for n in sorted(names)]}
        _, choices = bind_parking_access(places, graph, subset, asset=asset, size=size)
        self.connections = {}
        for choice in choices:
            a = next(a for a in by_id[choice["berth"]]["access"] if a["side"] == choice["side"])
            self.connections[choice["berth"]] = a["connections"]
        self.choices = choices
        self.walk = defaultdict(list)
        self.reverse_walk = defaultdict(list)
        self.trees = {}
        self.services = defaultdict(set)
        for berth, connections in self.connections.items():
            anchor = ("anchor", berth)
            for c in connections:
                place = ("place", c["place"])
                if c["kind"] != "home":
                    self.services[(c["kind"], c["group"])].add(place)
                for start, end, path in ((anchor, place, c["outward"]), (place, anchor, c["returning"])):
                    length = float(path["arc"][-1])
                    edge = (start, end)
                    self.walk[start].append((end, length, edge))
                    self.reverse_walk[end].append((start, length, edge))

    def _walk_tree(self, home, reverse):
        key = (home, reverse)
        if key not in self.trees:
            source = ("place", home)
            adj = self.reverse_walk if reverse else self.walk
            distances = {source: 0.0}
            previous = {}
            q = [(0.0, source)]
            while q:
                cost, node = heapq.heappop(q)
                if cost != distances[node]:
                    continue
                for target, length, edge in sorted(adj[node]):
                    total = cost + length
                    if total < distances.get(target, math.inf):
                        distances[target] = total
                        previous[target] = (node, edge)
                        heapq.heappush(q, (total, target))
            self.trees[key] = (distances, previous)
        return self.trees[key]

    def walking_alternative(self, home, kind, group):
        outward, op = self._walk_tree(home, False)
        back, bp = self._walk_tree(home, True)
        candidates = [(outward[p] + back[p], p) for p in self.services[(kind, group)] if p in outward and p in back]
        if not candidates:
            return None
        length, target = min(candidates)
        source = ("place", home)

        def links(previous):
            chain = []
            node = target
            while node != source:
                node, edge = previous[node]
                chain.append(edge)
            return chain[::-1]

        return dict(
            roundtrip_s=length / self.policy.walk_mps,
            service_place=target[1],
            outward_m=outward[target],
            return_m=back[target],
            outward_links=links(op),
            return_links=links(bp)[::-1],
        )

    def evaluate(self, a, b, lengths):
        if len(lengths) != 2 or any(not math.isfinite(v) or v < 0 for v in lengths):
            raise ValueError("Finite outward/return topology bounds required")
        options = []
        rejected = Counter()
        p = self.policy
        for origin, dest, out_m, back_m in ((a, b, *lengths), (b, a, *lengths[::-1])):
            start = self.connections.get(origin, [])
            end = self.connections.get(dest, [])
            homes = [x for x in start if x["kind"] == "home"]
            services = [x for x in end if x["kind"] not in ("home", "berth", "parking", "parking_lot")]
            if not homes or not services:
                rejected["missing_home_or_service"] += 1
                continue
            for home in homes:
                for service in services:
                    source_place = self.places[home["place"]]
                    target = self.places[service["place"]]
                    separation = math.dist(source_place.xy, target.xy)
                    if separation < p.min_home_service_separation_m:
                        rejected["home_service_too_close"] += 1
                        continue
                    approach = home["returning"]["arc"][-1]
                    homeward = home["outward"]["arc"][-1]
                    egress = service["outward"]["arc"][-1]
                    ingress = service["returning"]["arc"][-1]
                    car_out = (approach + egress) / p.walk_mps + out_m / p.max_drive_mps + p.transfer_s
                    car_back = (homeward + ingress) / p.walk_mps + back_m / p.max_drive_mps + p.transfer_s
                    # Same facility, including another standing slot, is a valid
                    # alternative service visit. Preserve the actual witness ID.
                    witness = self.walking_alternative(home["place"], service["kind"], service["group"])
                    # Driving estimates omit manoeuvre delays and route curvature
                    # still to be compiled. If even this optimistic full journey
                    # loses to a known walk, exclude this chosen topology.
                    if witness and car_out + car_back >= witness["roundtrip_s"] - 1e-8:
                        rejected["known_walk_dominates_optimistic_drive"] += 1
                        continue
                    options.append(
                        dict(
                            home_berth=origin,
                            destination_berth=dest,
                            home_place=home["place"],
                            service_place=service["place"],
                            service_group=service["group"],
                            service_kind=service["kind"],
                            home_service_separation_m=separation,
                            car_outward_lower_s=car_out,
                            car_return_lower_s=car_back,
                            known_walk_roundtrip_s=None if witness is None else witness["roundtrip_s"],
                            known_walk_service_place=None if witness is None else witness["service_place"],
                            mode_evidence="unknown_walking_alternative"
                            if witness is None
                            else "optimistic_car_may_improve_known_walk",
                            walking_witness=witness,
                            requires_direct_walk_comparison=True,
                        )
                    )
        options.sort(
            key=lambda x: (
                x["known_walk_roundtrip_s"] is None,
                x["car_outward_lower_s"] + x["car_return_lower_s"],
                x["home_place"],
                x["service_group"],
                x["service_place"],
            )
        )
        # Multiple slots at the same service are not multiple activity purposes.
        unique = {}
        for x in options:
            unique.setdefault((x["home_berth"], x["home_place"], x["service_group"]), x)
        return dict(
            eligible=bool(unique),
            options=list(unique.values())[:4],
            possible_options=len(unique),
            rejected=dict(rejected),
            candidate_only=True,
            admitted=False,
            limitations=[
                "Historical body-side proxies; current doors, support, occupancy and transfers unverified.",
                "Car time is a topology-based optimistic estimate; direct walking and actual drive may change mode choice.",
                "One service option is not two distinct recurring destinations or population capacity.",
            ],
        )
