"""Compile candidate outdoor approaches from mapped POIs and a measured walk network.

No inferred entrance/interior capability or actor-specific destinations. This
generalizes the former town-specific export script; runtime still admits actual
standing slots, occupancy and paths against the complete scene population.
"""

import hashlib

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, Polygon


def compile_places(
    network,
    semantics,
    *,
    seed=20261022,
    clearance_m=1.5,
    approach_min_m=0.8,
    approach_max_m=15.0,
    spacing_m=3.0,
    open_spacing_m=6.0,
    approach_limit=24,
    open_limit=256,
):
    if (
        not 0 <= approach_min_m < approach_max_m
        or clearance_m <= 0
        or spacing_m <= 0
        or open_spacing_m <= 0
        or approach_limit < 1
        or open_limit < 1
    ):
        raise ValueError("Positive place limits, clearance and valid approach range required")
    pois = sorted(semantics.get("pois", []), key=lambda p: p["id"])
    if len({p["id"] for p in pois}) != len(pois):
        raise ValueError("Duplicate POI identity")
    buildings = {b["id"]: b for b in semantics.get("buildings", [])}
    clear = distance_transform_edt(np.pad(network.free, 1))[1:-1, 1:-1] * network.step
    room = clear[network.nodes[:, 0], network.nodes[:, 1]] >= clearance_m
    if getattr(network, "standing_free", None) is not None:
        room &= network.standing_free[network.nodes[:, 0], network.nodes[:, 1]]
    nodes = shapely.points(network.xy)
    places, rejected = [], []
    for poi in pois:
        polygon = poi.get("polygon") or buildings.get(poi.get("building"), {}).get("polygon")
        geom = Polygon(polygon) if polygon else Point(poi["xy"])
        if geom.is_empty or not geom.is_valid:
            raise ValueError(f"Invalid POI geometry: {poi['id']}")
        dist = shapely.distance(geom, nodes)
        public = poi["kind"] in ("park", "square") and polygon is not None
        domain = shapely.contains(geom, nodes) if public else ((dist >= approach_min_m) & (dist <= approach_max_m))
        choices = np.flatnonzero(room & domain)
        if public:
            # Per-POI randomness stays stable when an unrelated POI is added.
            digest = hashlib.sha256(f"{seed}:{poi['id']}".encode()).digest()
            choices = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(choices)
        else:
            choices = choices[np.argsort(dist[choices], kind="stable")]
        accepted = []
        for i in choices:
            point = network.xy[i]
            if any(np.linalg.norm(point - p) < (open_spacing_m if public else spacing_m) for p in accepted):
                continue
            if any(np.linalg.norm(point - p["xy"]) < spacing_m for p in places):
                continue
            accepted.append(point)
            places.append(
                dict(
                    id=f"{poi['id']}-approach-{len(accepted)}",
                    poi=poi["id"],
                    kind=poi["kind"],
                    xy=point.tolist(),
                    component=int(network.labels[tuple(network.nodes[i])]),
                    source=poi.get("osm"),
                    distance_to_footprint_m=round(float(dist[i]), 2),
                    function="public_open_space" if public else "outdoor_approach",
                    verified_entrance=False,
                )
            )
            if len(accepted) >= (open_limit if public else approach_limit):
                break
        if not accepted:
            rejected.append(
                dict(
                    poi=poi["id"],
                    kind=poi["kind"],
                    candidate_nodes=int(len(choices)),
                    nearest_walkable_m=None if not len(dist) else round(float(dist.min()), 2),
                    reason="no candidate after support, mapped geometry, clearance and spacing checks",
                )
            )
    counts = np.bincount(network.labels.ravel())
    components = []
    for component in sorted(set(p["component"] for p in places)):
        points = network.xy[network.labels[network.free] == component]
        components.append(
            dict(
                component=component,
                area=float(counts[component] * network.step**2),
                extent=np.ptp(points, axis=0).tolist(),
                places=sum(p["component"] == component for p in places),
            )
        )
    return dict(
        schema="awr.life.outdoor_places.v1",
        network=network.summary(),
        places=places,
        components=components,
        rejected_pois=rejected,
        admitted=False,
        standing_policy=getattr(network, "standing_policy", None),
        compiler=dict(
            seed=seed,
            clearance_m=clearance_m,
            approach_min_m=approach_min_m,
            approach_max_m=approach_max_m,
            spacing_m=spacing_m,
            open_spacing_m=open_spacing_m,
            approach_limit=approach_limit,
            open_limit=open_limit,
        ),
        limitations=[
            "Outdoor candidate points, not surveyed doors, interiors or population capacity.",
            "Runtime must recheck support, actual bodies, resource slots and purposeful access.",
        ],
    )
