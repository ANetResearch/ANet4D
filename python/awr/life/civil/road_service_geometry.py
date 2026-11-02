"""Measure candidate road interfaces against their registered source polyline.

This audits the *meaning* of the sets used by DirectedServices. A graph search
cannot certify physical capacity when its endpoint sets are wrong. Distances
and station order here are diagnostics, never permissions or new road edges.
"""

import json
import math
from collections import Counter

import numpy as np


class Stationing:
    def __init__(self, coordinates):
        xy = np.asarray(coordinates, dtype=float)
        if xy.ndim != 2 or len(xy) < 2 or xy.shape[1] < 2 or not np.isfinite(xy).all():
            raise ValueError("Finite source polyline required")
        xy = xy[:, :2]
        delta = np.diff(xy, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        keep = lengths > 1e-10
        if not keep.any():
            raise ValueError("Source road has no length")
        self.start = xy[:-1][keep]
        self.delta = delta[keep]
        self.lengths = lengths[keep]
        self.stations = np.r_[0.0, np.cumsum(lengths)[:-1]][keep]
        self.length = float(lengths.sum())

    def project(self, pose, direction):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all() or direction not in ("forward", "reverse"):
            raise ValueError("Finite XY/yaw pose and explicit source direction required")
        fraction = np.clip(np.sum((pose[:2] - self.start) * self.delta, axis=1) / self.lengths**2, 0.0, 1.0)
        distance = np.linalg.norm(self.start + fraction[:, None] * self.delta - pose[:2], axis=1)
        # Retain every equally near projection: folds/self-intersections must not
        # be assigned an arbitrary station by argmin or map iteration order.
        near = np.flatnonzero(distance <= distance.min() + 1e-7)
        stations = self.stations[near] + fraction[near] * self.lengths[near]
        angles = np.arctan2(self.delta[near, 1], self.delta[near, 0])
        angles += math.pi if direction == "reverse" else 0.0
        error = np.abs((pose[2] - angles + math.pi) % (2 * math.pi) - math.pi)
        return dict(
            lateral_distance_m=float(distance.min()),
            station_range_m=[float(stations.min()), float(stations.max())],
            station_ambiguous=bool(np.ptp(stations) > 1e-6),
            heading_error_range_deg=[float(np.degrees(error.min())), float(np.degrees(error.max()))],
        )


def audit(document, source_graph, contract, baseline):
    roads = {}
    for feature in source_graph["features"]:
        p = feature["properties"]
        if p.get("feature") != "edge" or "vehicle" not in p.get("agents", []):
            continue
        rid = p["edge_id"]
        if rid in roads or type(p.get("oneway")) is not bool:
            raise ValueError("Unique roads and explicit direction required")
        roads[rid] = (p, Stationing(feature["geometry"]["coordinates"]))
    ports = {}
    for key, names in sorted(contract["interfaces"].items()):
        rid, junction, side, direction = json.loads(key)
        if side not in ("incoming", "outgoing") or direction not in ("forward", "reverse"):
            raise ValueError("Invalid interface side/direction")
        p, line = roads[rid]
        at_start = (side == "outgoing") == (direction == "forward")
        endpoint = str(p["from_node"] if at_start else p["to_node"])
        if junction != endpoint:
            raise ValueError("Interface junction disagrees with source direction")
        if not names:
            raise ValueError("Empty candidate interface")
        samples = []
        for name in names:
            n = document["nodes"][name]
            if n.get("traffic_resource") != "road:" + rid or n.get("direction") != direction:
                raise ValueError("Interface member has different source road/direction")
            s = line.project(n["pose"], direction)
            lo, hi = s["station_range_m"]
            along = [lo, hi] if at_start else [line.length - hi, line.length - lo]
            samples.append(
                dict(
                    node=name,
                    **s,
                    distance_from_endpoint_range_m=along,
                    entirely_in_opposite_half=along[0] > line.length / 2 + 1e-6,
                    against_source_heading=s["heading_error_range_deg"][0] > 90.0 + 1e-6,
                )
            )
        ports[key] = dict(
            road=rid,
            junction=junction,
            side=side,
            direction=direction,
            source_length_m=line.length,
            source_width_m=p.get("width_m"),
            against_source_oneway=p["oneway"] and direction == "reverse",
            samples=samples,
        )
    statuses = {r["id"]: r["status"] for r in baseline["rows"]}
    rows = []
    for c in contract["connections"]:
        a, b = ports[c["source"]], ports[c["target"]]
        if not (
            a["road"] == b["road"]
            and a["direction"] == b["direction"]
            and a["side"] == "outgoing"
            and b["side"] == "incoming"
        ):
            continue
        # Station order is only a diagnostic envelope; no alternative nodes are
        # concatenated into an imaginary path and no graph witness is changed.
        sa = [s["station_range_m"] for s in a["samples"]]
        sb = [s["station_range_m"] for s in b["samples"]]
        delta = [min(s[0] for s in sb) - max(s[1] for s in sa), max(s[1] for s in sb) - min(s[0] for s in sa)]
        if a["direction"] == "reverse":
            delta = [-delta[1], -delta[0]]
        rows.append(
            dict(
                id=c["id"],
                baseline_status=statuses[c["id"]],
                road=a["road"],
                source=c["source"],
                target=c["target"],
                directed_station_delta_range_m=delta,
                has_positive_station_order=delta[1] > 1e-6,
                opposite_half_members=sum(s["entirely_in_opposite_half"] for p in (a, b) for s in p["samples"]),
                ambiguous_station_members=sum(s["station_ambiguous"] for p in (a, b) for s in p["samples"]),
            )
        )
    samples = [s for p in ports.values() for s in p["samples"]]
    return dict(
        schema="awr.life.road_service_geometry.v1",
        interfaces=ports,
        corridors=rows,
        summary=dict(
            interfaces=len(ports),
            memberships=len(samples),
            opposite_half_memberships=sum(s["entirely_in_opposite_half"] for s in samples),
            ambiguous_station_memberships=sum(s["station_ambiguous"] for s in samples),
            against_heading_memberships=sum(s["against_source_heading"] for s in samples),
            corridors_by_baseline=dict(Counter(r["baseline_status"] for r in rows)),
            no_positive_order_by_baseline=dict(
                Counter(r["baseline_status"] for r in rows if not r["has_positive_station_order"])
            ),
        ),
        admitted=False,
        scope="Source-polyline stationing only; source width is annotation, not measured "
        "ground support. Opposite-half membership is an interface-review signal, not proof of an illegal "
        "turn. No geometry, topology, runtime parking rule or task changed.",
    )
