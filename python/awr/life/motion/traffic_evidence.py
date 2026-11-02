"""Failure-only traffic evidence; no refresh, admission or motion side effects.

Preserve the exact held/waiting resource geometry alongside route and body
state. This is diagnostic evidence, not a restart checkpoint: pedestrian and
activity state are not restored here. No work is added to successful ticks.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def snapshot(routes):
    traffic = routes.traffic
    runs = []

    def run_ref(run):
        if run is None:
            return None
        for i, old in enumerate(runs):
            if old is run:
                return i
        runs.append(run)
        return len(runs) - 1

    trips = {}
    for vehicle, trip in sorted(traffic.trips.items()):
        waiting = traffic.waiting.get(vehicle)
        trips[vehicle] = dict(
            evidence=trip["evidence"],
            arc_m=trip["arc"],
            speed_mps=trip["speed"],
            runs=[run_ref(r) for r in trip["runs"]],
            held=[run_ref(r) for r in trip["held"]],
            waiting=None
            if waiting is None
            else dict(order=waiting["order"], reason=waiting["reason"], run=run_ref(waiting["run"])),
        )
    resources = []
    for run in runs:
        room = run.room
        resources.append(
            dict(
                stop_m=run.stop,
                start_m=run.s0,
                end_m=run.s1,
                parts=[dict(edge=edge, wkb_hex=polygon.wkb_hex) for edge, polygon in run.parts],
                room=None
                if room is None
                else dict(start_m=room[0], end_m=room[1], capacity=room[2], wkb_hex=room[3].wkb_hex),
                exit_pose=run.exit_pose,
            )
        )
    courses = {}
    edges = set()
    for evidence in sorted({t["evidence"] for t in traffic.trips.values()}):
        # Reading through course()/layout() would refresh caches or reservations.
        course = traffic.courses.get(evidence)
        plan = routes.plans[evidence]
        courses[evidence] = dict(source=plan["source"], target=plan["target"], cached=course is not None)
        if course is not None:
            edges.update(course.edges)
            courses[evidence].update(edges=course.edges, ends=course.ends, route=course.route)
    edge_rows = {e: routes.graph.edges[e] for e in sorted(edges)}
    nodes = {n for e in edge_rows.values() for n in (e["source"], e["target"])}
    root = Path(__file__).resolve().parents[1]
    data = dict(
        schema="awr.life.traffic_failure.v1",
        admitted=False,
        scope="Failure-time traffic state, not a full-world replay or a causal counterfactual.",
        trips=trips,
        resources=resources,
        courses=courses,
        edges=edge_rows,
        nodes={n: routes.graph.nodes[n] for n in sorted(nodes)},
        occupancy={
            v: None if b is None else dict(pose=b.pose, size=b.size) for v, b in sorted(routes.occupancy.items())
        },
        executions={v: e.state() for v, e in sorted(routes.executions.items())},
        yields={v: dict(y) for v, y in sorted(getattr(traffic, "yields", {}).items())},
        path_failures={v: dict(f) for v, f in sorted(getattr(routes, "path_failures", {}).items())},
        planning_snapshot=dict(
            active=dict(getattr(routes, "active", {})), pending=dict(getattr(routes, "_pending", {}))
        ),
        stats=dict(traffic.stats),
        source_sha256={
            n: hashlib.sha256((root / n).read_bytes()).hexdigest()
            for n in (
                "civil/activity/traffic_runs.py",
                "civil/activity/driving.py",
                "civil/driving_graph.py",
                "motion/traffic_evidence.py",
            )
        },
    )

    def encode(value):
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported traffic evidence value: {type(value)}")

    # Detach all returned containers and preserve full float precision.
    return json.loads(json.dumps(data, default=encode, allow_nan=False))


def record_traffic(out, engine):
    if engine.driving is None:
        return None
    out = Path(out)
    path = out.parent / (out.name + ".traffic-failure.json")
    path.write_text(json.dumps(snapshot(engine.driving), indent=2) + "\n")
    return path.name
