"""Offline joint selection of existing approach samples and junction paths.

Source-authorized travel/join/travel chains can be extended. Missing joins
may be supplied as separately authorized intents, never executable placeholders.
The old graph remains intact. New ports are exact existing samples, retain
source direction and exclusive road resources, and never certify physics.
"""

import hashlib
from collections import defaultdict
from copy import deepcopy

import numpy as np

from awr.life.civil.driving_graph import load_candidate

from .local_road_search import search
from .maneuver import Maneuver
from .road_guides import body_route


def extend(
    document,
    profile,
    support,
    chains,
    domain_for_join,
    *,
    offsets_m=(3.0, 6.0, 10.0),
    max_expansions=256,
    max_searches=24,
    max_length_m=80.0,
    include_boundary_ports=False,
    connection_intents=None,
    source_roads=None,
    body_clearance=None,
):
    offsets = np.asarray(offsets_m, float)
    chains = tuple(tuple(c) for c in chains)
    if (
        not 1 <= len(chains) <= 32
        or len(set(chains)) != len(chains)
        or offsets.ndim != 1
        or not 1 <= len(offsets) <= 4
        or not np.isfinite(offsets).all()
        or (offsets < 0).any()
        or (offsets > 20).any()
        or len(set(offsets)) != len(offsets)
        or type(include_boundary_ports) is not bool
        or type(max_searches) is not int
        or not 1 <= max_searches <= 128
        or type(max_expansions) is not int
        or not 1 <= max_expansions <= 2000
        or not np.isfinite(max_length_m)
        or not 1 <= max_length_m <= 150
    ):
        raise ValueError("Bounded distinct junction chains and search budgets required")
    graph = load_candidate(document)
    templates = {k: Maneuver(**v) for k, v in document["rear_axle_templates"].items()}
    # Intent metadata is re-derived from source authority. It deliberately has
    # no route/template and cannot enter the executable graph.
    intents = {}
    if connection_intents:
        from .missing_turns import source_turn_pairs

        if source_roads is None or len(connection_intents) > 32:
            raise ValueError("Bounded source-authorized connection intents required")
        allowed = {(p["source"], p["target"]): p for p in source_turn_pairs(graph, source_roads)}
        for key, request in sorted(connection_intents.items()):
            pair = (request["source"], request["target"])
            if key in graph.edges or pair not in allowed:
                raise ValueError("Intent must be missing and source-authorized")
            actual = allowed[pair]
            intents[key] = dict(
                source=pair[0],
                target=pair[1],
                kind="join",
                source_junction=actual["source_junction"],
                blockers=[],
                traffic_resources=["junction:" + actual["source_junction"], *["road:" + r for r in actual["roads"]]],
            )

    def resolve_edge(key):
        return intents[key] if key in intents else graph.edges[key]

    # A chain may cross several subdivisions, but exactly one authorized join.
    # References are temporary original geometry, removed before returning.
    from .junction_approaches import reference

    references = {}
    reference_ports = {}
    normalized = []
    for chain in sorted(chains):
        if not 3 <= len(chain) <= 17 or any(k not in graph.edges and k not in intents for k in chain):
            raise ValueError("Known bounded travel/join/travel chain required")
        joins = [i for i, k in enumerate(chain) if resolve_edge(k)["kind"] == "join"]
        if len(joins) != 1 or joins[0] in (0, len(chain) - 1):
            raise ValueError("Connected directed travel/join/travel chain required")
        middle = joins[0]
        pair = []
        for side in (chain[:middle], chain[middle + 1 :]):
            key, ports = reference(graph, templates, side)
            pair.append(key)
            if len(side) > 1:
                references[key] = list(side)
                reference_ports[key] = ports
        normalized.append((pair[0], chain[middle], pair[1]))
    chains = tuple(normalized)
    jobs = []
    rows = []
    for chain in sorted(chains):
        if (
            len(chain) != 3
            or chain[0] not in graph.edges
            or chain[2] not in graph.edges
            or (chain[1] not in graph.edges and chain[1] not in intents)
        ):
            raise ValueError("Three known source edges required")
        a, j, b = (resolve_edge(k) for k in chain)
        if (
            [e["kind"] for e in (a, j, b)] != ["travel", "join", "travel"]
            or a["target"] != j["source"]
            or j["target"] != b["source"]
        ):
            raise ValueError("Connected directed travel/join/travel chain required")
        start, end = graph.nodes[j["source"]], graph.nodes[j["target"]]
        junction = j.get("source_junction", start.get("source_junction"))
        if not junction or ("source_junction" not in j and junction != end.get("source_junction")):
            raise ValueError("Shared explicit source junction required")
        if start.get("road") == end.get("road"):
            raise ValueError("Distinct source roads required")
        domain = domain_for_join(graph, chain[1])
        left, right = templates[chain[0]], templates[chain[2]]
        if (left.gear != 1).any() or (right.gear != 1).any():
            raise ValueError("Forward approach templates required")
        ports = []
        for m, back in ((left, True), (right, False)):
            selected = []
            # A nearby berth may split an approach BEFORE the first configured
            # extension. Preserve both exact boundary samples of short edges,
            # never extrapolate a missing station or enlarge the 20 m domain.
            # Prioritize those ports under the same global search budget.
            if include_boundary_ports:
                selected.append(len(m.points) - 1 if back else 0)
                if m.arc[-1] <= 20:
                    selected.append(0 if back else len(m.points) - 1)
            for distance in sorted(offsets):
                station = m.arc[-1] - distance if back else distance
                if not -1e-8 <= station <= m.arc[-1] + 1e-8:
                    continue
                i = int(np.argmin(abs(m.arc - station)))
                if i not in selected:
                    selected.append(i)
            ports.append(selected)
        pending = []
        for i in ports[0]:
            for k in ports[1]:
                p, q = left.body_pose(i), right.body_pose(k)
                if not support.supported(np.array([p[:2], q[:2]]), "vehicle").all():
                    rows.append(dict(chain=chain, samples=[i, k], candidate_found=False, reason="unsupported_port"))
                    continue
                pending.append((chain, i, k, p, q, domain))
        jobs.append(pending)
    # Round robin by source chain: early junctions cannot consume every search.
    ordered = [group[i] for i in range(max(map(len, jobs), default=0)) for group in jobs if i < len(group)]
    additions = []
    road_ports = defaultdict(list)
    search_count = 0
    for job_index, (chain, i, k, p, q, domain) in enumerate(ordered):
        row = dict(chain=chain, samples=[i, k], candidate_found=False)
        identity = hashlib.sha256(("|".join(chain) + f"|{i}|{k}").encode()).hexdigest()[:20]
        new = "junction-access:" + identity
        if new in graph.edges:
            row.update(reason="existing_candidate", existing_edge=new)
            rows.append(row)
            continue
        if search_count >= max_searches:
            row["reason"] = "global_search_budget"
            rows.append(row)
            continue
        search_count += 1
        m, report = search(
            profile,
            support,
            p,
            q,
            domain,
            max_expansions=max_expansions,
            max_length_m=max_length_m,
            body_clearance=body_clearance,
        )
        row.update(report)
        if m is not None:
            # Reject loops/backtracking on the incoming and outgoing legs.
            # End headings alone do not prove compliance with one-way access.
            body = body_route(m)["points"]
            head = np.column_stack([np.cos(m.yaw), np.sin(m.yaw)])
            valid = True
            for pose, incoming in ((p, True), (q, False)):
                axis = np.array([np.cos(pose[2]), np.sin(pose[2])])
                station = (body - pose[:2]) @ axis
                side = (station < -1e-8) if incoming else (station > 1e-8)
                if side.any() or np.dot(head[0 if incoming else -1], axis) < 0.99:
                    valid = False
            if not valid:
                row["reason"] = "approach_backtracking"
                m = None
        if m is not None:
            ports = []
            for edge_id, index, incoming in ((chain[0], i, True), (chain[2], k, False)):
                e = graph.edges[edge_id]
                old = templates[edge_id]
                label = new + (":in" if incoming else ":out")
                node = e["source"] if index == 0 else e["target"] if index == len(old.points) - 1 else label
                metadata = deepcopy(graph.nodes[e["target" if incoming else "source"]])
                metadata.pop("pose")
                metadata.pop("source_sample", None)
                graph.add_node(node, old.body_pose(index), **metadata, source_sample=index)
                road_ports[edge_id].append((index, node))
                lo, hi = (0, index) if incoming else (index, len(old.points) - 1)
                if hi == lo:
                    ports.append(node)
                    continue
                piece = Maneuver(old.points[lo : hi + 1], old.yaw[lo : hi + 1], old.gear[lo:hi], old.rear_x)
                key = label + ":approach"
                graph.add_edge(
                    key,
                    e["source"] if incoming else node,
                    node if incoming else e["target"],
                    body_route(piece),
                    kind="travel",
                    blockers=e["blockers"],
                    traffic_resources=e["traffic_resources"],
                )
                templates[key] = piece
                ports.append(node)
            graph.add_edge(
                new,
                *ports,
                body_route(m),
                kind="join",
                source_junction=resolve_edge(chain[1]).get(
                    "source_junction", graph.nodes[resolve_edge(chain[1])["source"]].get("source_junction")
                ),
                blockers=sorted({x for key in chain for x in resolve_edge(key)["blockers"]}),
                traffic_resources=sorted({x for key in chain for x in resolve_edge(key)["traffic_resources"]}),
            )
            templates[new] = m
            row.update(candidate_found=True, new_edge=new, length_m=float(m.arc[-1]))
            additions.append(new)
        rows.append(row)
    # Adjacent repaired junctions must connect through their shared approach
    # road. Retaining only whole prefixes/suffixes would strand every new port.
    # Never connect decreasing/equal stations or bridge different source edges.
    interports = []
    # Recover ports introduced by earlier compilation passes by BOTH exact
    # source sample and physical pose. Local sample numbers alone are not
    # globally unique after berth splitting. Include original endpoints so a
    # new zero-extension junction shares the real node, not an isolated clone.
    for chain in chains:
        for edge_id in (chain[0], chain[-1]):
            e = graph.edges[edge_id]
            old = templates[edge_id]
            road = graph.nodes[e["source"]].get("road")
            for node, v in graph.nodes.items():
                index = v.get("source_sample")
                if (
                    v.get("road") != road
                    or not isinstance(index, (int, np.integer))
                    or not 0 <= index < len(old.points)
                ):
                    continue
                actual = old.body_pose(index)
                expected = v["pose"]
                if (
                    np.linalg.norm(actual[:2] - expected[:2]) <= 1e-8
                    and abs(np.arctan2(np.sin(actual[2] - expected[2]), np.cos(actual[2] - expected[2]))) <= 1e-8
                ):
                    road_ports[edge_id].append((index, node))
            road_ports[edge_id].extend([(0, e["source"]), (len(old.points) - 1, e["target"])])
            road_ports[edge_id].extend(reference_ports.get(edge_id, ()))
    for edge_id, ports in sorted(road_ports.items()):
        old = templates[edge_id]
        e = graph.edges[edge_id]
        for a, node_a in sorted(set(ports)):
            for b, node_b in sorted(set(ports)):
                if b <= a:
                    continue
                if node_a == e["source"] and node_b == e["target"]:
                    continue
                piece = Maneuver(old.points[a : b + 1], old.yaw[a : b + 1], old.gear[a:b], old.rear_x)
                key = "junction-access-link:" + hashlib.sha256((node_a + "|" + node_b).encode()).hexdigest()[:20]
                if key in graph.edges:
                    continue
                if len(interports) >= 4096:
                    raise ValueError("Interport connection budget exceeded")
                graph.add_edge(
                    key,
                    node_a,
                    node_b,
                    body_route(piece),
                    kind="travel",
                    blockers=e["blockers"],
                    traffic_resources=e["traffic_resources"],
                )
                templates[key] = piece
                interports.append(key)
    # Different previously compiled curves may have separate node identities
    # for the SAME exact source sample/pose. A new turn must be reachable from
    # each such identity; a zero-length motion edge would be invalid. Reuse the
    # exact same turn geometry with those verified endpoints instead. This
    # changes graph connectivity, never a position, trajectory or road domain.
    aliases = {}
    for row in rows:
        if not row["candidate_found"]:
            continue
        key = row["new_edge"]
        edge = graph.edges[key]
        choices = []
        for edge_id, index, side in (
            (row["chain"][0], row["samples"][0], "source"),
            (row["chain"][-1], row["samples"][1], "target"),
        ):
            base = graph.nodes[edge[side]]
            choices.append(
                sorted(
                    {
                        node
                        for station, node in road_ports[edge_id]
                        if station == index
                        and np.array_equal(graph.nodes[node]["pose"], base["pose"])
                        and graph.nodes[node].get("source_junction") == base.get("source_junction")
                    }
                    | {edge[side]}
                )
            )
        for a in choices[0]:
            for b in choices[1]:
                if (a, b) == (edge["source"], edge["target"]):
                    continue
                name = key + ":alias:" + hashlib.sha256((a + "|" + b).encode()).hexdigest()[:16]
                if name in graph.edges:
                    continue
                if len(aliases) >= 1024:
                    raise ValueError("Exact-port alias budget exceeded")
                graph.add_edge(
                    name,
                    a,
                    b,
                    body_route(templates[key]),
                    kind="join",
                    blockers=edge["blockers"],
                    traffic_resources=edge["traffic_resources"],
                    source_junction=edge.get("source_junction"),
                )
                templates[name] = templates[key]
                aliases[name] = key
    for key in references:
        edge = graph.edges.pop(key)
        graph.links[edge["source"]].remove(key)
        templates.pop(key)
    return (
        graph,
        templates,
        dict(
            candidate_only=True,
            release_evidence=False,
            rows=rows,
            added_joins=additions,
            approach_references=references,
            mesh_body_screened=body_clearance is not None,
            added_interport_links=interports,
            added_join_aliases=aliases,
            search_count=search_count,
            connection_intents=len(intents),
            limits=dict(
                offsets_m=offsets.tolist(),
                max_searches=max_searches,
                max_expansions=max_expansions,
                max_length_m=max_length_m,
                max_interport_links=4096,
                max_join_aliases=1024,
                include_boundary_ports=include_boundary_ports,
                max_ports_per_approach=len(offsets) + 2,
            ),
            limitations=[
                "Exact source-connected candidates only; native terrain, body and traffic admission still required."
            ],
        ),
    )
