"""Compose exact source-corridor access additions without deleting old repairs.

Compile new berths against the unsplit source corridors, then merge only new
ports/access/slices. Each retained parent edge and original pose stays intact.
All new geometry remains candidate-only and needs refreshed occupancy/support.
"""

from copy import deepcopy

import numpy as np

from awr.life.civil.driving_graph import load_candidate

from .maneuver import Maneuver


def compose(parent, source, compiled, templates):
    graph = load_candidate(parent)
    result = {k: Maneuver(**v) for k, v in parent["rear_axle_templates"].items()}
    added = [k for k in compiled.edges if k not in source["edges"]]
    node_ids = {compiled.edges[k][end] for k in added for end in ("source", "target")}
    pose_fields = (
        "kind",
        "road",
        "direction",
        "source_junction",
        "traffic_mode",
        "traffic_resource",
        "guide_offset_m",
        "region",
        "source_actor",
        "parking_legality",
        "curb_maneuver_policy",
    )
    for name in sorted(node_ids):
        value = compiled.nodes[name]
        if name in graph.nodes:
            old = graph.nodes[name]
            a, b = np.asarray(old["pose"]), np.asarray(value["pose"])
            if (
                np.linalg.norm(a[:2] - b[:2]) > 1e-8
                or abs(np.arctan2(np.sin(a[2] - b[2]), np.cos(a[2] - b[2]))) > 1e-8
                or any(old.get(k) != value.get(k) for k in pose_fields)
            ):
                raise ValueError("Source port identity, authority or pose changed: " + name)
        else:
            graph.add_node(name, value["pose"], **{k: deepcopy(v) for k, v in value.items() if k != "pose"})
    new = []
    reused = []
    for key in sorted(added):
        edge = compiled.edges[key]
        m = templates[key]
        if key in graph.edges:
            old = graph.edges[key]
            n = result[key]
            if (
                any(
                    old.get(k) != edge.get(k)
                    for k in (
                        "source",
                        "target",
                        "kind",
                        "traffic_resource",
                        "traffic_resources",
                        "blockers",
                        "source_junction",
                    )
                )
                or m.rear_x != n.rear_x
                or m.points.shape != n.points.shape
                or not np.allclose(m.points, n.points, atol=1e-8, rtol=0)
                or not np.allclose(np.exp(1j * m.yaw), np.exp(1j * n.yaw), atol=1e-8, rtol=0)
                or not np.array_equal(m.gear, n.gear)
            ):
                raise ValueError("Existing edge geometry or authority changed: " + key)
            reused.append(key)
            continue
        graph.add_edge(
            key,
            edge["source"],
            edge["target"],
            edge["route"],
            kind=edge["kind"],
            blockers=edge["blockers"],
            traffic_resource=edge["traffic_resource"],
            traffic_resources=edge["traffic_resources"],
            source_junction=edge.get("source_junction"),
        )
        result[key] = m
        new.append(key)
    return (
        graph,
        result,
        dict(
            added_edges=new,
            reused_edges=reused,
            parent_edges_removed=0,
            parent_nodes_removed=0,
            source_corridor_geometry_changed=False,
        ),
    )
