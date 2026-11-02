"""Driving permission from roads, surveyed parking and source junctions.

No support heights or driving directions are created here. Rounding every road
end would permit turns on unsurveyed dead ends. Reviewed junction patches must
name genuinely incident source roads and stay within their existing half-width
of the shared source point. The scene's lane network (SUMO, compiled from the
same registered source roads) may also supply its motor lanes, including the
turning lanes inside junctions, and its junction areas; dead-end and internal
junctions are not junction areas (awr.life.civil.lane_network). Routes still need
terrain and swept-body admission.
"""

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .road_corridors import source_roads


def driving_area(semantics, source_graph=None, lane_network=None):
    areas = [LineString(r["line"]).buffer(r["width_m"] / 2, cap_style=2) for r in semantics["roads"]]
    if lane_network is not None:
        areas += [lane_network.driving] + [j["polygon"] for j in lane_network.junctions]
    areas += [Polygon(r["polygon"]) for r in semantics.get("verified_parking_regions", []) if r.get("verified") is True]
    regions = [r for r in semantics.get("verified_junction_regions", []) if r.get("verified") is True]
    if regions and source_graph is None:
        raise ValueError("Reviewed junctions require their source road graph")
    roads = {r["id"]: r for r in source_roads(semantics, source_graph)} if regions else {}
    seen = set()
    for region in regions:
        key = region["source_junction"]
        named = region["source_roads"]
        if key in seen or len(set(named)) != len(named) or len(named) < 2 or not region.get("evidence"):
            raise ValueError("Junction needs unique incident roads and review evidence")
        seen.add(key)
        centres = []
        widths = []
        for rid in named:
            if rid not in roads:
                raise ValueError("Unknown junction source road")
            road = roads[rid]
            if key == str(road["source_from"]):
                centres.append(road["line"][0])
            elif key == str(road["source_to"]):
                centres.append(road["line"][-1])
            else:
                raise ValueError("Junction patch cannot join nearby unrelated roads")
            width = float(road["width_m"])
            if not math.isfinite(width) or width <= 0:
                raise ValueError("Invalid incident road width")
            widths.append(width)
        centre = np.asarray(centres[0], float)
        if (
            centre.shape != (2,)
            or not np.isfinite(centres).all()
            or not np.allclose(centres, centre, rtol=0.0, atol=1e-6)
        ):
            raise ValueError("Source junction endpoint coordinates disagree")
        polygon = Polygon(region["polygon"])
        vertices = np.asarray(polygon.exterior.coords)
        if (
            polygon.is_empty
            or not polygon.is_valid
            or polygon.interiors
            or not polygon.covers(Point(centre))
            or not np.isfinite(vertices).all()
            or np.max(np.linalg.norm(vertices - centre, axis=1)) > max(widths) / 2 + 1e-6
        ):
            raise ValueError("Reviewed junction exceeds the source road half-width envelope")
        areas.append(polygon)
    return unary_union(areas)
