"""Repair a blocked junction-to-berth transition without moving the berth.

Choose an existing forward entry sample with a clear remaining suffix. Search
only inside the original junction domain, then retain that exact suffix. This
is an offline bounded candidate compiler, not a native motion certificate.
"""

import hashlib
from copy import deepcopy

import numpy as np
import shapely

from awr.life.civil.driving_graph import load_candidate

from .local_road_search import search
from .maneuver import Maneuver
from .road_guides import body_envelope, body_route


def repair(
    document,
    profile,
    support,
    paths,
    bodies,
    domain_for_edge,
    *,
    max_searches=8,
    max_expansions=256,
    body_clearance=None,
):
    if (
        type(max_searches) is not int
        or not 1 <= max_searches <= 32
        or type(max_expansions) is not int
        or not 1 <= max_expansions <= 2000
        or not 1 <= len(paths) <= 32
    ):
        raise ValueError("Bounded occupied access repair required")
    graph = load_candidate(document)
    templates = {k: Maneuver(**v) for k, v in document["rear_axle_templates"].items()}
    obstacles = tuple(bodies[n].polygon() for n in sorted(bodies))
    tree = shapely.STRtree(obstacles)

    def blocked(m):
        return bool(len(tree.query(body_envelope(m, profile).buffer(0.4), predicate="intersects")))

    transitions = {}
    for path in paths:
        if not path or len(path) > 10000 or set(path) - set(graph.edges):
            raise ValueError("Known connected path required")
        for index, (a, b) in enumerate(zip(path, path[1:], strict=False)):
            first, last = graph.edges[a], graph.edges[b]
            if first["target"] != last["source"]:
                raise ValueError("Disconnected path")
            if first["kind"] == "join" and last["kind"] == "entry" and (blocked(templates[a]) or blocked(templates[b])):
                incoming = path[index - 1] if index and graph.edges[path[index - 1]]["kind"] == "travel" else None
                transitions.setdefault((a, b), set()).add(incoming)
    rows = []
    count = 0
    added = []
    for a, b in sorted(transitions):
        first, last = graph.edges[a], graph.edges[b]
        entry = templates[b]
        if (entry.gear != 1).any() or (templates[a].gear != 1).any():
            rows.append(dict(chain=[a, b], reason="forward_entry_required"))
            continue
        junction = first.get("source_junction", graph.nodes[first["source"]].get("source_junction"))
        if not junction:
            raise ValueError("Source junction required")
        domain = domain_for_edge(graph, a)
        entry_domain = domain_for_edge(graph, b)
        starts = []
        for previous in sorted(k for k in transitions[(a, b)] if k is not None):
            e = graph.edges[previous]
            old = templates[previous]
            road = graph.nodes[first["source"]].get("road")
            if e["target"] != first["source"] or (old.gear != 1).any() or graph.nodes[e["source"]].get("road") != road:
                continue
            indices = set()
            for distance in (20.0, 10.0, 6.0, 3.0):
                station = max(0.0, float(old.arc[-1]) - distance)
                j = int(np.argmin(abs(old.arc - station)))
                if j == len(old.points) - 1 or j in indices:
                    continue
                indices.add(j)
                prefix = None if j == 0 else Maneuver(old.points[: j + 1], old.yaw[: j + 1], old.gear[:j], old.rear_x)
                if prefix is not None and (
                    blocked(prefix) or not domain_for_edge(graph, previous).covers(body_envelope(prefix, profile))
                ):
                    continue
                starts.append((previous, j, prefix, old.body_pose(j)))
        starts.append((None, None, None, graph.nodes[first["source"]]["pose"]))
        goals = []
        previous = -float("inf")
        for i in range(1, len(entry.points) - 1):
            # Spatially separate trials, preserving original analytic samples.
            if entry.arc[i] - previous < 1.0 or entry.arc[i] > 20.0:
                continue
            tail = Maneuver(entry.points[i:], entry.yaw[i:], entry.gear[i:], entry.rear_x)
            if blocked(tail) or not entry_domain.covers(body_envelope(tail, profile)):
                continue
            goal = entry.body_pose(i)
            if not support.supported([goal[:2]], "vehicle")[0]:
                continue
            previous = float(entry.arc[i])
            goals.append((i, tail, goal))
        # Near-identical arrival samples must not exhaust a finite search
        # budget. Cover the available suffix range before filling small gaps.
        if goals:
            remaining = list(goals[1:])
            spread = [goals[0]]
            while remaining:
                chosen = max(
                    remaining, key=lambda g: (min(abs(float(entry.arc[g[0]] - entry.arc[h[0]])) for h in spread), -g[0])
                )
                remaining.remove(chosen)
                spread.append(chosen)
            goals = spread
        # Diagonal ordering spreads the fixed search budget over departure and
        # arrival samples; one blocked incoming heading cannot consume it all.
        jobs = sorted(
            ((s, t) for s in range(len(starts)) for t in range(len(goals))), key=lambda st: (sum(st), st[1], st[0])
        )
        for s, t in jobs:
            if count >= max_searches:
                break
            previous, j, prefix, start = starts[s]
            i, tail, goal = goals[t]
            count += 1
            m, stats = search(
                profile,
                support,
                start,
                goal,
                domain,
                obstacles=obstacles,
                body_clearance=body_clearance,
                max_expansions=max_expansions,
                max_length_m=80.0,
            )
            row = dict(
                chain=[a, b],
                incoming_edge=previous,
                incoming_sample=j,
                entry_sample=i,
                entry_station_m=float(entry.arc[i]),
                candidate_found=False,
                **stats,
            )
            if m is not None:
                # The new turn must progress from its incoming pose to the
                # outgoing port; no forward-driven loop around the obstacle.
                points = body_route(m)["points"]
                for pose, incoming in ((start, True), (goal, False)):
                    station = (points - pose[:2]) @ np.array([np.cos(pose[2]), np.sin(pose[2])])
                    backwards = np.any(station < -1e-8) if incoming else np.any(station > 1e-8)
                    if backwards:
                        m = None
                        row["reason"] = "approach_backtracking"
                        break
            if m is not None:
                identity = "|".join((a, b, str(i), str(previous), str(j)))
                key = "occupied-access:" + hashlib.sha256(identity.encode()).hexdigest()[:24]
                if key in graph.edges:
                    raise ValueError("Repair already exists")
                node = key + ":port"
                meta = deepcopy(graph.nodes[last["source"]])
                meta.pop("pose")
                meta.pop("source_sample", None)
                graph.add_node(node, goal, **meta, source_sample=i)
                origin = first["source"]
                new_edges = []
                if previous is not None:
                    upstream = graph.edges[previous]
                    origin = upstream["source"] if j == 0 else key + ":incoming"
                    if prefix is not None:
                        meta = deepcopy(graph.nodes[first["source"]])
                        meta.pop("pose")
                        meta.pop("source_sample", None)
                        graph.add_node(origin, start, **meta, source_sample=j)
                        prefix_key = key + ":approach"
                        graph.add_edge(
                            prefix_key,
                            upstream["source"],
                            origin,
                            body_route(prefix),
                            kind="travel",
                            blockers=upstream["blockers"],
                            traffic_resources=upstream["traffic_resources"],
                        )
                        templates[prefix_key] = prefix
                        new_edges.append(prefix_key)
                graph.add_edge(
                    key,
                    origin,
                    node,
                    body_route(m),
                    kind="join",
                    source_junction=junction,
                    blockers=sorted(
                        set(first["blockers"])
                        | set(last["blockers"])
                        | (set(graph.edges[previous]["blockers"]) if previous is not None else set())
                    ),
                    traffic_resources=sorted(
                        set(first["traffic_resources"])
                        | set(last["traffic_resources"])
                        | (set(graph.edges[previous]["traffic_resources"]) if previous is not None else set())
                    ),
                )
                suffix = key + ":entry"
                graph.add_edge(
                    suffix,
                    node,
                    last["target"],
                    body_route(tail),
                    kind="entry",
                    blockers=last["blockers"],
                    traffic_resources=last["traffic_resources"],
                )
                templates[key] = m
                templates[suffix] = tail
                new_edges.extend([key, suffix])
                added.extend(new_edges)
                row.update(candidate_found=True, new_edges=new_edges)
                rows.append(row)
                break
            rows.append(row)
    return (
        graph,
        templates,
        dict(
            candidate_only=True,
            release_evidence=False,
            rows=rows,
            search_count=count,
            added_edges=added,
            transition_count=len(transitions),
            limits=dict(
                max_searches=max_searches,
                max_expansions=max_expansions,
                entry_extension_m=20.0,
                incoming_extension_m=20.0,
            ),
            limitations=[
                "Original junction domain and unchanged entry suffix only; every complete journey requires native revalidation."
            ],
        ),
    )
