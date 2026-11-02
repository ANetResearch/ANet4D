"""Directed interface reachability, independent of candidate-waypoint survival.

This is a diagnostic over an already compiled graph and an explicit interface
contract. It does not establish the truth of interface geometry/semantics,
grant parking permission, or replace physical and live-driver admission.
An interface is a set of alternative poses, never an implicit connecting edge.
"""

import hashlib
import json
from collections import deque

from awr.life.common.schema import schema_id


class DirectedServices:
    def __init__(self, graph, interfaces, connections):
        # Snapshot topology, not the large geometric arrays or mutable occupancy.
        self.kinds = {n: d["kind"] for n, d in graph.nodes.items()}
        self.edges = {e: (d["source"], d["target"]) for e, d in graph.edges.items()}
        self.links = {n: [] for n in self.kinds}
        for e, (a, b) in sorted(self.edges.items()):
            self.links[a].append((e, b))
        self.interfaces = {}
        if not interfaces or not connections:
            raise ValueError("Nonempty explicit interfaces and directed services required")
        for name, nodes in interfaces.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(nodes, (list, tuple))
                or not nodes
                or any(not isinstance(n, str) for n in nodes)
                or len(set(nodes)) != len(nodes)
                or set(nodes) - self.kinds.keys()
            ):
                raise ValueError("An interface needs distinct known node identities")
            self.interfaces[name] = tuple(sorted(nodes))
        self.connections = []
        seen = set()
        for c in connections:
            name, source, target = c["id"], c["source"], c["target"]
            if (
                not isinstance(name, str)
                or not name
                or name in seen
                or source not in self.interfaces
                or target not in self.interfaces
                or set(self.interfaces[source]) & set(self.interfaces[target])
            ):
                raise ValueError("Distinct service identities and disjoint endpoint interfaces required")
            seen.add(name)
            # An optional corridor/turn restriction prevents a local service
            # from being declared intact using an unrelated city-wide detour.
            allowed = c.get("allowed_edges")
            if allowed is not None:
                if (
                    not isinstance(allowed, (list, tuple))
                    or not allowed
                    or any(not isinstance(e, str) for e in allowed)
                    or len(set(allowed)) != len(allowed)
                    or set(allowed) - self.edges.keys()
                ):
                    raise ValueError("A service edge scope must name distinct existing edges")
                allowed = frozenset(allowed)
            self.connections.append((name, source, target, allowed))
        self.connections.sort(key=lambda row: row[0])

    def _search(self, source, blocked, allowed):
        starts = self.interfaces[source]
        previous = {n: None for n in starts}
        queue = deque(starts)
        while queue:
            node = queue.popleft()
            # Each starting pose is an alternative origin. A reached berth is
            # a terminal; it never becomes a road shortcut or a new origin.
            if self.kinds[node] == "berth" and previous[node] is not None:
                continue
            for edge, end in self.links[node]:
                if edge in blocked or (allowed is not None and edge not in allowed) or end in previous:
                    continue
                previous[end] = (node, edge)
                queue.append(end)
        return previous

    def _paths(self, blocked):
        blocked = frozenset(blocked)
        if blocked - self.edges.keys():
            raise ValueError("Unknown blocked edge; occupancy belongs to a different graph")
        cache, paths = {}, {}
        for name, source, target, allowed in self.connections:
            key = (source, allowed)
            if key not in cache:
                cache[key] = self._search(source, blocked, allowed)
            previous = cache[key]
            end = next((n for n in self.interfaces[target] if n in previous), None)
            if end is None:
                paths[name] = None
                continue
            node, edges = end, []
            while previous[node] is not None:
                node, edge = previous[node]
                edges.append(edge)
            paths[name] = dict(source_node=node, target_node=end, edges=edges[::-1])
        return paths, len(cache)

    def compare(self, *, before=(), after=()):
        """Compare complete blocked-edge snapshots; never mutate or accumulate.

        Missing baseline service remains unavailable, not a new parking cut.
        All alternatives lost cannot pass vacuously. Paths through different
        interface representatives are not composed into a fictional journey.
        Actual driver origins/destinations should be separate singleton services.
        """
        old, old_searches = self._paths(before)
        new, new_searches = self._paths(after)
        rows = []
        for name, source, target, _ in self.connections:
            status = (
                ("preserved" if new[name] else "cut") if old[name] else ("restored" if new[name] else "unavailable")
            )
            rows.append(dict(id=name, source=source, target=target, status=status, before=old[name], after=new[name]))
        return dict(
            schema="awr.life.directed_service_audit.v1",
            rows=rows,
            cut_services=sum(r["status"] == "cut" for r in rows),
            unavailable_before=sum(r["before"] is None for r in rows),
            all_declared_services_available_after=all(r["after"] is not None for r in rows),
            searches=old_searches + new_searches,
            admitted=False,
            scope="Directed topology only; interface source/geometry, physical edge validity, "
            "parking legality and joint dynamic capacity require separate verification.",
        )


def load_bound_services(map_bytes, contract_bytes):
    """Bind declared interfaces to exact map bytes; missing metadata is not guessed.

    The contract retains its feature evidence references for independent review.
    Hash agreement is provenance binding, not semantic certification.
    """
    from .driving_graph import load_candidate

    contract = json.loads(contract_bytes)
    if (
        schema_id(contract.get("schema")) != "awr.life.directed_service_contract.v1"
        or contract.get("driving_map_sha256") != hashlib.sha256(map_bytes).hexdigest()
        or not isinstance(contract.get("feature_evidence"), dict)
        or not contract["feature_evidence"]
    ):
        raise ValueError("Exact map binding and explicit feature evidence required")
    graph = load_candidate(json.loads(map_bytes))
    return DirectedServices(graph, contract["interfaces"], contract["connections"])
