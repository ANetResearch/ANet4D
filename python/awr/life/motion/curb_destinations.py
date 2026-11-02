"""Bounded vacant curb proposals from named roads and parking conventions.

The source direction, full body, remaining carriageway, measured support and
original occupied scene constrain proposals. No native parking/motion admission
is inferred. Population and behavior tick code do not run this sampler.
"""

import hashlib
import json
from collections import Counter

import numpy as np
import shapely
from shapely.geometry import LineString

from awr.life.civil.curb import poses_at, road_geometry, road_layout
from awr.life.civil.curb_allocation import footprint


def propose(roads, profile, permission, support, rules, obstacles, *, max_berths=200, spacing_m=16.0):
    if type(max_berths) is not int or not 1 <= max_berths <= 300 or not 8 <= spacing_m <= 50:
        raise ValueError("Bounded curb nomination required")
    if len({r["id"] for r in roads}) != len(roads):
        raise ValueError("Unique named source roads required")
    if permission.is_empty or not permission.is_valid:
        raise ValueError("Valid road permission required")
    tree = shapely.STRtree(obstacles)
    size = profile.body_hi - profile.body_lo
    middle = (profile.body_lo + profile.body_hi) / 2
    groups = []
    rejected = Counter()
    samples = 0
    for road in sorted(roads, key=lambda r: r["id"]):
        if type(road.get("oneway")) is not bool:
            raise ValueError("Explicit source direction required")
        layout = road_layout(road["width_m"], size)
        if layout is None:
            rejected["insufficient_carriageway"] += 1
            continue
        domain = LineString(road["line"]).buffer(road["width_m"] / 2, cap_style=2).intersection(permission)
        choices = []
        for reverse in (False,) if road["oneway"] else (False, True):
            oriented = {**road, "line": road["line"][::-1] if reverse else road["line"]}
            geometry = road_geometry(oriented)
            # Keep complete car lengths away from raw road endpoints; the
            # parking convention additionally rejects junction/crossing areas.
            stations = np.arange(float(size[0]) + 5.0, geometry[0].length - float(size[0]) - 5.0, spacing_m)
            if len(stations) > 200:
                raise ValueError("Per-road nomination budget exceeded")
            centres, headings = poses_at(geometry, stations, layout["parking_offset_m"])
            for station, centre, yaw in zip(stations, centres, headings, strict=False):
                samples += 1
                body = footprint(dict(xy=centre, yaw=yaw), size, margin_m=0.0)
                if not domain.covers(body):
                    rejected["outside_source_road"] += 1
                    continue
                if tree.query(body.buffer(0.4), predicate="intersects").size:
                    rejected["original_occupancy"] += 1
                    continue
                legality = rules.evaluate(centre, float(yaw), size)
                if legality.get("legal") is not True:
                    rejected["parking_rule"] += 1
                    continue
                c, s = np.cos(yaw), np.sin(yaw)
                pivot = centre - np.array([c * middle[0] - s * middle[1], s * middle[0] + c * middle[1]])
                probes = np.vstack([pivot, shapely.get_coordinates(body)])
                if not support.supported(probes, "vehicle").all():
                    rejected["measured_support"] += 1
                    continue
                identity = json.dumps(
                    [road["id"], reverse, float(station), pivot.tolist(), float(yaw)], separators=(",", ":")
                )
                key = "berth:source-curb:" + hashlib.sha256(identity.encode()).hexdigest()[:24]
                choices.append(
                    dict(
                        berth=key,
                        mapped_node=dict(
                            kind="berth",
                            pose=[*pivot.tolist(), float(yaw)],
                            region="curb:" + road["id"],
                            parking_legality=legality,
                            source_station_m=float(station),
                            source_direction="reverse" if reverse else "forward",
                        ),
                        source_road=road["id"],
                    )
                )
        if choices:
            groups.append(choices)
    # Spread finite inventory across source roads, not the first long street.
    ordered = [g[i] for i in range(max((len(g) for g in groups), default=0)) for g in groups if i < len(g)]
    selected = []
    selected_bodies = []
    for row in ordered:
        n = row["mapped_node"]
        pose = np.asarray(n["pose"])
        c, s = np.cos(pose[2]), np.sin(pose[2])
        centre = pose[:2] + [c * middle[0] - s * middle[1], s * middle[0] + c * middle[1]]
        body = footprint(dict(xy=centre, yaw=pose[2]), size, margin_m=0.4)
        if any(body.intersects(other) for other in selected_bodies):
            rejected["overlapping_proposal"] += 1
            continue
        selected.append(row)
        selected_bodies.append(body)
        if len(selected) >= max_berths:
            break
    return dict(
        rows=selected,
        samples=samples,
        eligible_before_cap=len(ordered),
        rejected=dict(rejected),
        selected=len(selected),
        selection_policy="source-road round robin, fixed station spacing",
        limits=dict(max_berths=max_berths, spacing_m=spacing_m),
        candidate_only=True,
        release_evidence=False,
    )
