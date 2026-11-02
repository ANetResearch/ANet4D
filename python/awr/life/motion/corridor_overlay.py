"""Connect exact lane ports introduced by independent corridor compilations.

An older travel slice can pass straight through a newer port without sharing a
graph vertex. Add exact sub-slices at those ports; never snap positions, connect
different lanes, create junction turns, or inherit physical admission.
"""

import hashlib
import json
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from awr.life.civil.driving_graph import load_candidate

from .maneuver import Maneuver
from .road_guides import body_route


def corridor(node):
    fields = ("road", "direction", "guide_offset_m", "traffic_mode", "traffic_resource")
    if node.get("kind") != "lane" or any(node.get(k) is None for k in fields):
        return None
    return tuple(node[k] for k in fields)


def connect(document, *, max_added_edges=50000):
    if type(max_added_edges) is not int or not 1 <= max_added_edges <= 100000:
        raise ValueError("Bounded corridor overlay required")
    graph = load_candidate(document)
    templates = {k: Maneuver(**v) for k, v in document["rear_axle_templates"].items()}
    if set(graph.edges) != set(templates):
        raise ValueError("Every graph edge needs its physical axle template")
    groups = defaultdict(list)
    for name, node in sorted(graph.nodes.items()):
        key = corridor(node)
        if key is not None:
            groups[key].append(name)
    trees = {key: cKDTree([graph.nodes[n]["pose"][:2] for n in names]) for key, names in groups.items()}
    additions = []

    # Signatures prevent repeated composition from accumulating the same slice.
    def signature(a, b, m, e):
        h = hashlib.sha256()
        for array in (m.points, m.yaw, m.gear):
            h.update(array.tobytes())
        return (
            a,
            b,
            m.rear_x,
            h.hexdigest(),
            tuple(e["blockers"]),
            e["traffic_resource"],
            tuple(e["traffic_resources"]),
        )

    seen = {
        signature(e["source"], e["target"], templates[k], e) for k, e in graph.edges.items() if e["kind"] == "travel"
    }
    for parent, edge in sorted(list(graph.edges.items())):
        if edge["kind"] != "travel":
            continue
        key = corridor(graph.nodes[edge["source"]])
        if key is None or key != corridor(graph.nodes[edge["target"]]):
            continue
        m = templates[parent]
        route = body_route(m)
        if (
            route["points"].shape != edge["route"]["points"].shape
            or not np.allclose(route["points"], edge["route"]["points"], atol=1e-8, rtol=0)
            or not np.allclose(np.exp(1j * route["yaw"]), np.exp(1j * edge["route"]["yaw"]), atol=1e-8, rtol=0)
            or not np.array_equal(route["gear"], edge["route"]["gear"])
        ):
            raise ValueError("Body route and physical axle template differ")
        cuts = {}
        for index, nearby in enumerate(trees[key].query_ball_point(route["points"], r=1e-8)):
            names = [
                groups[key][j]
                for j in nearby
                if abs(np.angle(np.exp(1j * (graph.nodes[groups[key][j]]["pose"][2] - m.yaw[index])))) <= 1e-8
            ]
            if names:
                cuts[index] = sorted(names)
        # Endpoint identities must always be preserved even when numerical
        # endpoint noise is below graph tolerance but above overlay tolerance.
        cuts[0] = sorted(set(cuts.get(0, ())) | {edge["source"]})
        last = len(m.points) - 1
        cuts[last] = sorted(set(cuts.get(last, ())) | {edge["target"]})
        indices = sorted(cuts)
        if len(indices) == 2 and cuts[0] == [edge["source"]] and cuts[last] == [edge["target"]]:
            continue
        for a, b in zip(indices, indices[1:], strict=False):
            section = Maneuver(m.points[a : b + 1], m.yaw[a : b + 1], m.gear[a:b], m.rear_x)
            for start in cuts[a]:
                for end in cuts[b]:
                    sig = signature(start, end, section, edge)
                    if sig in seen:
                        continue
                    if len(additions) >= max_added_edges:
                        raise ValueError("Corridor overlay exceeded explicit edge budget")
                    name = (
                        "corridor-overlay:"
                        + hashlib.sha256(
                            json.dumps([parent, a, b, start, end], separators=(",", ":")).encode()
                        ).hexdigest()[:24]
                    )
                    graph.add_edge(
                        name,
                        start,
                        end,
                        body_route(section),
                        kind="travel",
                        blockers=edge["blockers"],
                        traffic_resource=edge["traffic_resource"],
                        traffic_resources=edge["traffic_resources"],
                    )
                    templates[name] = section
                    additions.append(
                        dict(edge=name, parent=parent, source=start, target=end, start_sample=a, end_sample=b)
                    )
                    seen.add(sig)
    return (
        graph,
        templates,
        dict(
            added_edges=additions,
            original_edges_removed=0,
            original_nodes_moved=0,
            candidate_only=True,
            release_evidence=False,
            limitations=[
                "Exact existing travel sub-slices only; occupancy/support must be screened again.",
                "No interpolation, cross-lane snapping, new road area or inferred source junction.",
            ],
        ),
    )
