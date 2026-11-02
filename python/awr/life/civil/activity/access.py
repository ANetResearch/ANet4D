"""Bind actual facility places to exact, statically witnessed car-side access.

This compiles metadata for the shared Director; it does not create destinations,
assign drivers, move actors or bypass live routing/transfer checks. Each berth
uses one explicit side anchor, so opposite-side walk proofs cannot be mixed into
an invented passage through the vehicle. Anchors remain body-box proxies.
"""

import math
from dataclasses import replace

import numpy as np

from awr.life.common.schema import schema_id

from .director import Place


def side_access_candidates(pose, size, radius):
    """Declared front-side body-box anchors, not authored mesh door metadata."""
    pose, size = np.asarray(pose, float), np.asarray(size, float)
    if (
        pose.shape != (3,)
        or size.shape != (3,)
        or np.any(size <= 0)
        or not np.isfinite(np.r_[pose, size, radius]).all()
        or radius <= 0
    ):
        raise ValueError("Finite body and walking clearance required")
    forward = np.array([math.cos(pose[2]), math.sin(pose[2])])
    left = np.array([-forward[1], forward[0]])
    for side in (1, -1):
        centre = pose[:2] + forward * (0.15 * size[0]) + side * left * (size[1] / 2 + radius + 0.15)
        yield dict(
            xy=centre,
            yaw=pose[2] + side * math.pi / 2,
            side="left" if side == 1 else "right",
            anchor_semantics="front-side body-box proxy; mesh door not authored",
        )


def _same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (
        a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, rtol=0.0, atol=1e-8)
    )


def _walk_path(route, start, end):
    points = np.asarray(route["points"], float)
    arc = np.asarray(route["arc"], float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or arc.shape != (len(points),)
        or not np.isfinite(points).all()
        or not np.isfinite(arc).all()
        or not _same(points[0], start)
        or not _same(points[-1], end)
        or not _same(arc, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    ):
        raise ValueError("Access path does not join the exact declared endpoints")


def bind_parking_access(places, graph, report, *, asset, size, choices=None):
    """Return new Place records and auditable side choices, without mutation.

    `choices` can select a witnessed side for a berth. Otherwise a deterministic
    affordance order prefers an anchor serving both homes and facilities, then
    distinct facilities, homes, and the left side. Population assignment must
    still verify its actual selected anchor. This is not a capacity certificate.
    The caller must verify source-file hashes before consuming external reports.
    """
    if (
        schema_id(report.get("schema")) != "awr.life.berth_activity_access_candidate.v1"
        or report["vehicle_asset"] != asset
        or not _same(report["size_m"], size)
    ):
        raise ValueError("Access evidence uses a different vehicle body/schema")
    places = list(places)
    base = {p.id: p for p in places}
    if len(base) != len(places) or any(p.kind == "berth" or p.parking for p in places):
        raise ValueError("Distinct unbound activity places required")
    berths = {n: d for n, d in graph.nodes.items() if d["kind"] == "berth"}
    rows = report["rows"]
    choices = choices or {}
    if (
        len(rows) != len(berths)
        or {r["berth"] for r in rows} != set(berths)
        or set(choices) - set(berths)
        or set(base) & set(berths)
    ):
        raise ValueError("Exact non-conflicting berth roster required")
    slots = {s for p in places for s in p.slots}
    links = {k: [] for k in base}
    bound = []
    evidence = []
    for row in sorted(rows, key=lambda r: r["berth"]):
        name = row["berth"]
        node = berths[name]
        if not _same(row["pose"], node["pose"]):
            raise ValueError("Berth pose changed")
        anchors = {a["side"]: a for a in side_access_candidates(node["pose"], size, report["human_turn_radius_m"])}
        candidates = {}
        for access in row["access"]:
            side = access["side"]
            expected = anchors.get(side)
            if (
                expected is None
                or side in candidates
                or not _same(access["xy"], expected["xy"])
                or not math.isfinite(access["yaw"])
                or abs(math.remainder(access["yaw"] - expected["yaw"], 2 * math.pi)) > 1e-8
            ):
                raise ValueError("Car-side anchor changed or duplicated")
            verified = {}
            for connection in access["connections"]:
                place = base.get(connection["place"])
                if (
                    place is None
                    or place.id in verified
                    or place.kind != connection["kind"]
                    or (place.group or place.id) != connection["group"]
                    or not _same(place.xy, connection["xy"])
                ):
                    raise ValueError("Access proof refers to a different real facility")
                _walk_path(connection["outward"], access["xy"], place.xy)
                _walk_path(connection["returning"], place.xy, access["xy"])
                verified[place.id] = place
            candidates[side] = (access, verified)
        if name in choices:
            if choices[name] not in candidates:
                raise ValueError("Requested side has no witnessed access")
            side = choices[name]
        elif not candidates:
            continue
        else:

            def score(side):
                values = candidates[side][1].values()
                homes = {p.id for p in values if p.kind == "home"}
                facilities = {p.group or p.id for p in values if p.kind != "home"}
                return (bool(homes and facilities), len(facilities), len(homes), side == "left")

            side = max(sorted(candidates), key=score)
        access, verified = candidates[side]
        if not verified:
            continue
        if name in slots:
            raise ValueError("Berth slot collides with facility slot")
        bound.append(Place(name, tuple(node["pose"][:2]), "berth", (name,), access_xy=tuple(access["xy"])))
        for place in verified:
            links[place].append(name)
        evidence.append(
            dict(
                berth=name,
                side=side,
                access_xy=list(access["xy"]),
                places=sorted(verified),
                anchor_semantics=anchors[side]["anchor_semantics"],
            )
        )
    return [replace(p, parking=tuple(sorted(links[p.id]))) for p in places] + bound, evidence
