"""Recover missing source-authorized turns at directed reachability cuts.

The topology diagnostic is not a driving route. Only bounded, permitted,
measured rear-axle searches add edges; native physical admission still follows.
No geometric proximity, U-turn, reversed road, or access-through-berth shortcut
creates connectivity. Work is offline, never per actor or simulation tick.
"""

import hashlib
from collections import defaultdict

import numpy as np

from awr.life.civil.driving_graph import load_candidate

from .local_road_search import search
from .maneuver import Maneuver
from .road_guides import body_route


def _reachable(start, adjacency):
    seen = {start}
    queue = [start]
    for node in queue:
        for target in sorted(adjacency[node] - seen):
            seen.add(target)
            queue.append(target)
    return seen


def source_turn_pairs(graph, roads, *, max_pairs=8192):
    """Enumerate missing source-authorized pairs, without inventing a route."""
    if type(max_pairs) is not int or not 1 <= max_pairs <= 8192:
        raise ValueError("Bounded source-turn opportunity count required")
    incoming = defaultdict(set)
    outgoing = defaultdict(set)
    existing = set()
    for key, e in graph.edges.items():
        a, b = e["source"], e["target"]
        if e["kind"] == "travel":
            incoming[b].add(key)
            outgoing[a].add(key)
        if e["kind"] == "join":
            existing.add((a, b))

    def endpoint(node, arrival):
        n = graph.nodes[node]
        road = n.get("road")
        direction = n.get("direction")
        junction = n.get("source_junction")
        if n.get("kind") != "lane" or road not in roads or direction not in ("forward", "reverse") or not junction:
            return None
        r = roads[road]
        if direction == "reverse" and r["oneway"]:
            return None
        expected = r["source_to" if arrival == (direction == "forward") else "source_from"]
        if expected != junction:
            return None
        links = incoming[node] if arrival else outgoing[node]
        # Original endpoint must connect to an actual travel edge on this road
        # and direction. Interpolated berth ports are not source junctions.
        if not any(
            graph.nodes[graph.edges[k]["source" if arrival else "target"]].get("road") == road
            and graph.nodes[graph.edges[k]["source" if arrival else "target"]].get("direction") == direction
            for k in links
        ):
            return None
        return junction, road

    ends = defaultdict(list)
    for node in sorted(graph.nodes):
        meta = endpoint(node, False)
        if meta:
            ends[meta[0]].append((node, meta[1]))
    rows = []
    for a in sorted(graph.nodes):
        meta = endpoint(a, True)
        if not meta:
            continue
        junction, ra = meta
        for b, rb in ends[junction]:
            if ra == rb or (a, b) in existing:
                continue
            rows.append(dict(source=a, target=b, source_junction=junction, roads=[ra, rb]))
            if len(rows) > max_pairs:
                raise ValueError("Source-turn opportunity budget exceeded")
    return rows


def missing_cut_turns(graph, roads, source, target):
    if source == target or any(n not in graph.nodes or graph.nodes[n]["kind"] != "berth" for n in (source, target)):
        raise ValueError("Two distinct mapped berths required")
    adj = defaultdict(set)
    reverse = defaultdict(set)
    for e in graph.edges.values():
        a, b = e["source"], e["target"]
        if (e["kind"] == "exit" and a != source) or (e["kind"] == "entry" and b != target):
            continue
        adj[a].add(b)
        reverse[b].add(a)
    reached = _reachable(source, adj)
    ancestors = _reachable(target, reverse)
    if target in reached:
        return [], dict(already_connected=True, reachable_nodes=len(reached), ancestor_nodes=len(ancestors))
    rows = [r for r in source_turn_pairs(graph, roads) if r["source"] in reached and r["target"] in ancestors]
    return rows, dict(
        already_connected=False,
        reachable_nodes=len(reached),
        ancestor_nodes=len(ancestors),
        missing_cut_pairs=len(rows),
        multiple_cut_search=False,
    )


def compile_missing(
    document,
    roads,
    profile,
    support,
    source,
    target,
    domain,
    *,
    max_searches=8,
    max_expansions=1024,
    max_length_m=80.0,
    selected_pairs=None,
):
    if (
        type(max_searches) is not int
        or not 1 <= max_searches <= 128
        or type(max_expansions) is not int
        or not 1 <= max_expansions <= 2000
        or not np.isfinite(max_length_m)
        or not 1 <= max_length_m <= 150
    ):
        raise ValueError("Bounded missing-turn search required")
    graph = load_candidate(document)
    templates = {k: Maneuver(**v) for k, v in document["rear_axle_templates"].items()}
    pairs, topology = missing_cut_turns(graph, roads, source, target)
    if selected_pairs is not None:
        allowed = {(p["source"], p["target"]): p for p in source_turn_pairs(graph, roads)}
        keys = [(p["source"], p["target"]) for p in selected_pairs]
        if not 1 <= len(keys) <= 128 or len(set(keys)) != len(keys) or any(k not in allowed for k in keys):
            raise ValueError("Bounded distinct source-authorized selected turns required")
        # Re-derive identities and resources from source data; selection is not authority.
        pairs = [allowed[k] for k in sorted(keys)]
        topology = dict(topology, multiple_cut_search=True, selected_turns=len(pairs))
    rows = []
    added = []
    attempts = 0
    for pair in pairs:
        row = dict(pair, candidate_found=False)
        rows.append(row)
        if attempts >= max_searches:
            row["reason"] = "global_search_budget"
            continue
        a, b = pair["source"], pair["target"]
        p, q = graph.nodes[a]["pose"], graph.nodes[b]["pose"]
        area = domain(*pair["roads"], pair["source_junction"])
        if not support.supported(np.array([p[:2], q[:2]]), "vehicle").all():
            row["reason"] = "unsupported_port"
            continue
        attempts += 1
        m, r = search(profile, support, p, q, area, max_expansions=max_expansions, max_length_m=max_length_m)
        row.update(r)
        if m is None:
            continue
        # Same forward-only approach convention as the existing port compiler.
        body = body_route(m)["points"]
        backtracks = False
        for pose, entering in ((p, True), (q, False)):
            axis = np.array([np.cos(pose[2]), np.sin(pose[2])])
            station = (body - pose[:2]) @ axis
            if ((station < -1e-8) if entering else (station > 1e-8)).any():
                backtracks = True
                break
        if backtracks:
            row.update(candidate_found=False, reason="approach_backtracking")
            continue
        key = "source-missing-turn:" + hashlib.sha256((a + "|" + b).encode()).hexdigest()[:24]
        graph.add_edge(
            key,
            a,
            b,
            body_route(m),
            kind="join",
            source_junction=pair["source_junction"],
            traffic_resources=["junction:" + pair["source_junction"], *["road:" + r for r in pair["roads"]]],
        )
        templates[key] = m
        added.append(key)
        row.update(candidate_found=True, new_edge=key, length_m=float(m.arc[-1]))
    return (
        graph,
        templates,
        dict(
            candidate_only=True,
            release_evidence=False,
            topology=topology,
            rows=rows,
            source=source,
            target=target,
            added_joins=added,
            search_count=attempts,
            limits=dict(max_searches=max_searches, max_expansions=max_expansions, max_length_m=max_length_m),
            limitations=[
                "Exact existing source-road endpoints only; no inferred junctions or dead-end U-turns.",
                "Candidates require native terrain, body, traffic and activity admission.",
            ],
        ),
    )


def compile_missing_ports(
    document,
    roads,
    profile,
    support,
    selected_pairs,
    domain,
    *,
    max_searches=48,
    max_expansions=1024,
    max_length_m=80.0,
):
    """Jointly choose exact approach ports around missing source connections.

    Intent links stay outside DrivingGraph. Only searched maneuvers and exact
    slices of existing approaches are added; fixed global budgets apply.
    """
    from .junction_access import extend

    graph = load_candidate(document)
    allowed = {(p["source"], p["target"]): p for p in source_turn_pairs(graph, roads)}
    keys = [(p["source"], p["target"]) for p in selected_pairs]
    if not 1 <= len(keys) <= 32 or len(set(keys)) != len(keys) or any(k not in allowed for k in keys):
        raise ValueError("Bounded distinct source-authorized selected turns required")
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for key, e in sorted(graph.edges.items()):
        if e["kind"] == "travel":
            incoming[e["target"]].append(key)
            outgoing[e["source"]].append(key)
    intents = {}
    chains = []
    for a, b in sorted(keys):
        name = "source-turn-intent:" + hashlib.sha256((a + "|" + b).encode()).hexdigest()[:24]
        intents[name] = allowed[(a, b)]
        for left in incoming[a]:
            for right in outgoing[b]:
                chains.append((left, name, right))
                if len(chains) > 32:
                    raise ValueError("Source approach chain budget exceeded")

    def area_for_join(g, key):
        p = intents[key]
        return domain(*p["roads"], p["source_junction"])

    g, t, r = extend(
        document,
        profile,
        support,
        chains,
        area_for_join,
        include_boundary_ports=True,
        offsets_m=(0.0, 3.0, 6.0, 10.0),
        max_searches=max_searches,
        max_expansions=max_expansions,
        max_length_m=max_length_m,
        connection_intents=intents,
        source_roads=roads,
    )
    r.update(
        topology=dict(multiple_cut_search=True, selected_turns=len(keys), approach_port_search=True),
        required_turns=[allowed[k] for k in sorted(keys)],
    )
    return g, t, r
