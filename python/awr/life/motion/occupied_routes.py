"""Offline rejection of road candidates intersecting a frozen actor layout.

This does not certify motion. It removes unusable candidates before expensive
native trials, using the same physical chassis envelope and clearance as the
local axle planner. Occupants are never removed or moved to open a route.
"""

import math

import numpy as np
import shapely

from .maneuver import Maneuver
from .road_guides import body_envelope, body_route


def screen(graph, templates, profile, bodies, *, margin_m=0.4):
    if not math.isfinite(margin_m) or margin_m < 0.4:
        raise ValueError("Static route clearance cannot weaken the axle planner")
    if set(templates) != set(graph.edges):
        raise ValueError("Complete source template roster required")
    names = sorted(bodies)
    polygons = [bodies[n].polygon() for n in names]
    if any(p.is_empty or not p.is_valid for p in polygons):
        raise ValueError("Valid fixed actor footprints required")
    tree = shapely.STRtree(polygons)
    rows = []
    cache = {}
    hits = 0
    for key in sorted(graph.edges):
        m = Maneuver(**templates[key])
        route = body_route(m)
        original = graph.edges[key]["route"]
        if (
            route["points"].shape != original["points"].shape
            or not np.allclose(route["points"], original["points"], rtol=0, atol=1e-7)
            or not np.allclose(route["yaw"], original["yaw"], rtol=0, atol=1e-7)
            or not np.array_equal(route["gear"], original["gear"])
        ):
            raise ValueError("Rear axle template differs from graph route: " + key)
        # Alias edges often share geometry. Exact bytes only: no rounded pose
        # key can incorrectly reuse clearance for a nearby but blocked guide.
        identity = (m.points.tobytes(), m.yaw.tobytes(), m.gear.tobytes(), m.rear_x)
        if identity in cache:
            blocked = cache[identity]
            hits += 1
        else:
            area = body_envelope(m, profile).buffer(margin_m)
            blocked = tuple(sorted(names[int(i)] for i in tree.query(area, predicate="intersects")))
            cache[identity] = blocked
        if blocked:
            rows.append(dict(edge=key, actors=list(blocked)))
    return dict(
        schema="awr.life.occupied_routes.v1",
        candidate_only=True,
        release_evidence=False,
        actor_count=len(names),
        edge_count=len(graph.edges),
        margin_m=margin_m,
        blocked_edges=[r["edge"] for r in rows],
        conflicts=rows,
        unique_geometry=len(cache),
        geometry_cache_hits=hits,
        limitations=[
            "Planar fixed occupancy screening only; native continuous motion and dynamic traffic remain mandatory."
        ],
    )
