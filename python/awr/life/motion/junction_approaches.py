"""Bounded approach references across existing subdivisions of one corridor.

References concatenate original samples; they are never admitted motion and
must not remain in the executable graph. Junction compilation uses them to
select ports without making a berth subdivision an artificial turning limit.
"""

import hashlib

import numpy as np

from .maneuver import Maneuver
from .road_guides import body_route
from .support_chains import corridor, validate_chain


def reference(graph, templates, chain):
    chain = tuple(chain)
    validate_chain(graph, chain)
    if len(chain) == 1:
        return chain[0], []
    parts = [templates[k] for k in chain]
    for a, b in zip(parts, parts[1:], strict=False):
        if (
            a.rear_x != b.rear_x
            or np.linalg.norm(a.points[-1] - b.points[0]) > 1e-8
            or abs(np.arctan2(np.sin(a.yaw[-1] - b.yaw[0]), np.cos(a.yaw[-1] - b.yaw[0]))) > 1e-8
        ):
            raise ValueError("Exact same-axle approach seams required")
    m = Maneuver(
        np.vstack([p.points[1 if i else 0 :] for i, p in enumerate(parts)]),
        np.concatenate([p.yaw[1 if i else 0 :] for i, p in enumerate(parts)]),
        np.concatenate([p.gear for p in parts]),
        parts[0].rear_x,
    )
    key = "approach-reference:" + hashlib.sha256("|".join(chain).encode()).hexdigest()[:24]
    if key not in graph.edges:
        graph.add_edge(
            key,
            graph.edges[chain[0]]["source"],
            graph.edges[chain[-1]]["target"],
            body_route(m),
            kind="travel",
            blockers=sorted({x for k in chain for x in graph.edges[k]["blockers"]}),
            traffic_resources=sorted({x for k in chain for x in graph.edges[k]["traffic_resources"]}),
        )
        templates[key] = m
    index = 0
    ports = [(0, graph.edges[chain[0]]["source"])]
    for k, p in zip(chain, parts, strict=False):
        index += len(p.points) - 1
        ports.append((index, graph.edges[k]["target"]))
    return key, ports


def select(graph, paths, *, distance_m=20.0, max_edges=8):
    """Select every existing join on diagnostic paths, with same-road approaches.

    No shortest-path substitution or geometric shortcut: each returned window
    is a contiguous slice of its supplied path. Actual support and body checks
    remain the port compiler's responsibility.
    """
    if (
        not np.isfinite(distance_m)
        or not 0 < distance_m <= 20
        or type(max_edges) is not int
        or not 1 <= max_edges <= 8
        or not 1 <= len(paths) <= 32
    ):
        raise ValueError("Bounded junction approach selection required")
    found = set()
    for path in paths:
        if not path or len(path) > 10000 or set(path) - set(graph.edges):
            raise ValueError("Known bounded path required")
        if any(graph.edges[a]["target"] != graph.edges[b]["source"] for a, b in zip(path, path[1:], strict=False)):
            raise ValueError("Connected diagnostic path required")
        for i, k in enumerate(path):
            if graph.edges[k]["kind"] != "join" or i == 0 or i == len(path) - 1:
                continue
            if any(graph.edges[path[j]]["kind"] != "travel" for j in (i - 1, i + 1)):
                continue
            sides = []
            for first, step in ((i - 1, -1), (i + 1, 1)):
                chain = [path[first]]
                identity = corridor(graph, path[first])
                j = first
                length = float(graph.edges[path[first]]["route"]["arc"][-1])
                while (
                    identity is not None
                    and length < distance_m
                    and len(chain) < max_edges
                    and 0 <= j + step < len(path)
                    and corridor(graph, path[j + step]) == identity
                ):
                    j += step
                    chain.append(path[j])
                    length += float(graph.edges[path[j]]["route"]["arc"][-1])
                if step < 0:
                    chain.reverse()
                validate_chain(graph, chain)
                sides.append(chain)
            found.add(tuple([*sides[0], k, *sides[1]]))
    if len(found) > 32:
        raise ValueError("Junction approach window budget exceeded")
    return sorted(found)
