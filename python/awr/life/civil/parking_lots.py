"""Generate long-stay bays only inside explicitly surveyed parking regions.

No OSM park/open area is inferred to be a car park. Rows parameterize bays;
support, clear aisles, straight bay access and pedestrian passage are verified.
This adapter does not claim to simulate lot ingress, turning or driver actions.
"""

import math

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from awr.life.civil.collision import rectangle
from awr.life.civil.curb_allocation import footprint, swept


def _polygon(coords, description):
    polygon = Polygon(coords)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"Invalid {description} polygon")
    return polygon


def candidates(surface, semantics, size, rules, report=None):
    """Each lot supplies polygon, aisles, entrances, rows and survey evidence.

    A row supplies start/end bay centres, heading_rad, spacing_m and access_m
    (signed straight distance to its aisle). Access footprints must fit inside
    the lot and end entirely within an aisle. Reserved access corridors cannot
    be filled by later parked cars. Dynamic lane admission remains separate.
    """
    report = report if report is not None else {}
    report.update(lots=0, proposed=0, admitted=0, rejected={})

    def reject(reason):
        report["rejected"][reason] = report["rejected"].get(reason, 0) + 1

    result = []
    ids = set()
    # Keep this surveyed schema separate from legacy synthetic parking_lots,
    # whose inferred slots do not prove support or an entrance connection.
    for lot in semantics.get("verified_parking_regions", []):
        ident = lot.get("id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("Parking lots require unique nonempty IDs")
        ids.add(ident)
        if lot.get("verified") is not True or not isinstance(lot.get("evidence"), str) or not lot["evidence"].strip():
            raise ValueError(f"Parking lot {ident} requires survey evidence")
        boundary = _polygon(lot["polygon"], f"{ident} boundary")
        aisle_parts = [_polygon(x, f"{ident} aisle") for x in lot["aisles"]]
        entries = [_polygon(x, f"{ident} entrance") for x in lot["entrances"]]
        if not aisle_parts or not entries or not lot.get("rows"):
            raise ValueError(f"Parking lot {ident} needs aisles, entrances and rows")
        aisles = unary_union(aisle_parts)
        access = unary_union([aisles, *entries])
        # Polygon clipping/JSON round-trips can leave a boundary endpoint
        # 1e-15 m from the source edge, or covers=False on zero-area slivers.
        # This sub-micrometre topology tolerance is not a physical clearance.
        topology_epsilon = 1e-7
        if (
            not boundary.buffer(topology_epsilon).covers(access)
            or access.geom_type != "Polygon"
            or any(not e.intersection(aisles).area > 0 for e in entries)
            or not any(e.distance(boundary.boundary) <= topology_epsilon for e in entries)
        ):
            raise ValueError(f"Parking lot {ident} has disconnected or unbounded entrance access")
        report["lots"] += 1
        for row_index, row in enumerate(lot["rows"]):
            start, end = np.asarray(row["start"], float), np.asarray(row["end"], float)
            yaw, pitch, distance = map(float, (row["heading_rad"], row["spacing_m"], row["access_m"]))
            if (
                start.shape != (2,)
                or end.shape != (2,)
                or not np.isfinite(np.r_[start, end, yaw, pitch, distance]).all()
                or pitch <= 0
                or distance == 0
            ):
                raise ValueError(f"Invalid row in parking lot {ident}")
            length = float(np.linalg.norm(end - start))
            direction = (end - start) / length if length else np.array([0.0, 1.0])
            for bay in range(math.floor(length / pitch) + 1):
                report["proposed"] += 1
                xy = start + bay * pitch * direction
                slot = dict(xy=xy.tolist(), yaw=yaw, parking_lot=ident, bay=f"{ident}/row-{row_index}/bay-{bay}")
                body = footprint(slot, size)
                if not boundary.covers(body) or body.intersects(access):
                    reject("outside_bay_area_or_blocks_aisle")
                    continue
                exit_xy = xy + distance * np.array([math.cos(yaw), math.sin(yaw)])
                # Sample translation independently of path tangent: reversing
                # out of a bay must not turn the parked body through 180 degrees.
                n = max(2, math.ceil(abs(distance) / 0.35) + 1)
                points = np.linspace(xy, exit_xy, n)
                route = dict(points=points, yaw=np.full(n, yaw), arc=np.linspace(0.0, abs(distance), n), road=ident)
                corridor = swept(route, size)
                exit_body = Polygon(rectangle(exit_xy, yaw, size, margin=0.4))
                if not boundary.covers(corridor) or not aisles.covers(exit_body):
                    reject("no_clear_bay_to_aisle_access")
                    continue
                from awr.life.civil.grounded import support_mask

                if not support_mask(surface, points, route["yaw"], size, "vehicle").all():
                    reject("unsupported_bay_or_access")
                    continue
                legality = rules.evaluate(xy, yaw, size)
                if not legality["legal"]:
                    reject("parking_rule_violation")
                    continue
                slot["legality"] = dict(legality, where="parking_lot", slot=slot["bay"])
                slot["access_evidence"] = dict(
                    kind="supported_straight_bay_to_aisle",
                    end_xy=exit_xy.tolist(),
                    survey=lot["evidence"],
                    road_connection_verified=False,
                )
                result.append(
                    dict(slot=slot, road=ident, access_route=route, layout=dict(mode="verified_parking_lot_long_stay"))
                )
                report["admitted"] += 1
    return result


def allocate(allocator, options, size, *, region=None):
    for option in options:
        if region is not None and not region.covers(footprint(option["slot"], size)):
            continue
        route = option["access_route"]
        corridor = swept(route, size)
        if allocator.free([option["slot"]], size, corridor):
            allocator.reserve([option["slot"]], size)
            # Keep bay access free without pretending it is a driven route.
            allocator.paths.append(corridor)
            return option
    return None
