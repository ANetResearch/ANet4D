"""Bounded physical connections between continuous poses and route nodes.

Node erosion includes half a grid diagonal; that discretization allowance is
not a physical wall. An adapter may certify the un-eroded semantic domain and
the entire swept planning footprint. Missing evidence remains a refusal.
This does not add graph nodes, join components, or change actor positions.
"""

import math

import numpy as np
from shapely.geometry import LineString, Point


def continuous_connection(network, a, b, free):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all() or np.linalg.norm(b - a) > 1.5 + 1e-9:
        return False
    area = getattr(network, "area", None)
    surface = getattr(network, "surface", None)
    certify = getattr(surface, "footprint_supported", None)
    radius = max(getattr(network, "radius", 0.0), getattr(free, "radius", 0.0))
    if area is None or certify is None or not math.isfinite(radius) or radius <= 0:
        return False
    # Preserve the physical .08 m ground margin; omit only the node's half
    # grid diagonal. Circumscribe the disk rather than inscribe its polygon.
    radius = (radius + 0.08) / math.cos(math.pi / 64)
    line = Point(a) if np.array_equal(a, b) else LineString([a, b])
    footprint = line.buffer(radius, quad_segs=16)
    return bool(area.covers(footprint) and certify(footprint, "person"))
