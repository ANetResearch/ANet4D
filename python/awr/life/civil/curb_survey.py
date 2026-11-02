"""Actor-independent curb berth candidates from verified source roads.

These are alternative physical poses, not an allocation: opposite-side bays
may not be occupied simultaneously on a narrow street. Walking passages,
combined parking legality and entry/exit routes still need admission.
"""

import hashlib
import math
from collections import defaultdict

import numpy as np

from .curb import berth, road_geometry, road_layout
from .road_corridors import source_roads


def pose_id(pose):
    pose = np.asarray(pose, float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("Invalid curb pose")
    signature = np.round(np.r_[pose[:2], np.cos(pose[2]), np.sin(pose[2])], 6)
    return hashlib.sha256(np.asarray(signature, dtype="<f8").tobytes()).hexdigest()[:20]


def survey(
    surface, semantics, source_graph, size, rules, *, existing_ids=(), activity_places=(), max_berths=240, spacing_m=6.0
):
    size = np.asarray(size, float)
    if (
        size.shape != (3,)
        or not np.isfinite(size).all()
        or np.any(size <= 0)
        or type(max_berths) is not int
        or max_berths < 1
        or not math.isfinite(spacing_m)
        or spacing_m < float(size[0]) + 1.0
    ):
        raise ValueError("Finite physical size, berth spacing and budget required")
    facilities = np.array(
        [p["xy"] for p in activity_places if p["kind"] not in ("home", "berth", "parking", "parking_lot")], float
    ).reshape(-1, 2)
    seen = set(existing_ids)
    groups = defaultdict(list)
    counts = dict(source_roads=0, tested_poses=0, supported_legal_poses=0, existing_poses=0)
    for road in source_roads(semantics, source_graph):
        counts["source_roads"] += 1
        layout = road_layout(road["width_m"], size)
        if layout is None:
            continue
        # Preserve source direction; the opposite parking orientation is only
        # considered on an explicitly verified two-way street.
        for reverse in (False,) if road["oneway"] else (False, True):
            directed = dict(road, line=road["line"][::-1] if reverse else road["line"])
            geometry = road_geometry(directed)
            margin = float(size[0]) / 2 + 2.0
            for station in np.arange(margin, geometry[0].length - margin, spacing_m):
                counts["tested_poses"] += 1
                slot = berth(geometry, station, layout, size, surface, rules)
                if slot is None:
                    continue
                counts["supported_legal_poses"] += 1
                pose = np.r_[slot["xy"], slot["yaw"]]
                ident = pose_id(pose)
                if ident in seen:
                    counts["existing_poses"] += 1
                    continue
                seen.add(ident)
                distance = float(np.linalg.norm(facilities - pose[:2], axis=1).min()) if len(facilities) else None
                groups[road["id"]].append(
                    dict(
                        id=ident,
                        road=road["id"],
                        pose=pose,
                        slot=slot,
                        source_oneway=road["oneway"],
                        reverse_source_orientation=reverse,
                        nearest_facility_m=distance,
                    )
                )
    ordered = []
    for road, rows in groups.items():
        rows.sort(key=lambda r: (r["nearest_facility_m"] if r["nearest_facility_m"] is not None else 0.0, r["id"]))
        ordered.append((road, rows))
    ordered.sort(
        key=lambda pair: (
            pair[1][0]["nearest_facility_m"] if pair[1][0]["nearest_facility_m"] is not None else 0.0,
            pair[0],
        )
    )
    # Broad road coverage precedes adding many adjacent bays beside one facility.
    # This is a finite diagnostic search order, not a population assignment.
    selected = [
        rows[offset]
        for offset in range(max((len(rows) for _, rows in ordered), default=0))
        for _, rows in ordered
        if offset < len(rows)
    ][:max_berths]
    counts.update(
        candidate_roads=len(groups),
        selected=len(selected),
        budget_truncated=sum(map(len, groups.values())) > len(selected),
    )
    return selected, counts
