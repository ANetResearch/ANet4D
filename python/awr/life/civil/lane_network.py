"""SUMO geometry as a candidate network, never as a ground-support oracle.

All coordinates are local ENU. Directed connections include their internal lane
geometry; a missing connector is rejected rather than replaced by a straight line.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from awr.life.common.schema import schema_id


def shape(text):
    return np.array([[float(v) for v in p.split(",")[:2]] for p in text.split()])


class LaneNetwork:
    def __init__(self, path, registration=None):
        """``registration``: path of a ``awr.life.lane_registration.v1`` document whose lane
        geometry (laterally registered to the measured carriageway) replaces the SUMO
        shapes of the motor lanes. Topology, widths and speeds are unchanged."""
        self.path = Path(path)
        self.registration = None
        registered = {}
        if registration:
            import json

            doc = json.loads(Path(registration).read_text())
            if schema_id(doc.get("schema")) != "awr.life.lane_registration.v1":
                raise ValueError("Unsupported lane registration document")
            self.registration = Path(registration)
            registered = {k: np.asarray(v, float) for k, v in doc["lanes"].items()}
            # Lanes the registration excluded (single-track streets keep one direction)
            # leave the network entirely; connections through them disappear with them.
            self.excluded_lanes = dict(doc.get("excluded_lanes", {}))
        else:
            self.excluded_lanes = {}
        root = ET.parse(path).getroot()
        location = root.find("location")
        if location is not None and any(abs(float(v)) > 1e-6 for v in location.get("netOffset", "0,0").split(",")):
            raise ValueError("Expected unshifted ENU SUMO network")
        self.lanes = {}
        self.walking_shapes = []
        self.driving_shapes = []
        for edge in root.findall("edge"):
            function = edge.get("function", "")
            for lane in edge.findall("lane"):
                p = shape(lane.get("shape", ""))
                if len(p) < 2:
                    continue
                if lane.get("id") in self.excluded_lanes:
                    continue
                if lane.get("id") in registered and len(registered[lane.get("id")]) >= 2:
                    p = registered[lane.get("id")]
                allow = lane.get("allow", "").split()
                deny = lane.get("disallow", "").split()
                width = float(lane.get("width", 3.2))
                pedestrian = allow == ["pedestrian"] or function in ("walkingarea", "crossing")
                motor = not pedestrian and ("passenger" in allow or not allow) and "passenger" not in deny
                if pedestrian:
                    geom = Polygon(p) if function == "walkingarea" and len(p) >= 3 else LineString(p).buffer(width / 2)
                    self.walking_shapes.append(geom.buffer(0))
                if motor:
                    self.driving_shapes.append(LineString(p).buffer(width / 2))
                    self.lanes[lane.get("id")] = dict(
                        id=lane.get("id"),
                        edge=edge.get("id"),
                        internal=function == "internal",
                        points=p,
                        width=width,
                        speed=float(lane.get("speed", 13.89)),
                    )
        # Junction polygons for parking rules; internal junctions and degenerate
        # shapes are not junction areas.
        self.junctions = []
        for junction in root.findall("junction"):
            if junction.get("type") in ("internal", "dead_end") or not junction.get("shape"):
                continue
            p = shape(junction.get("shape"))
            if len(p) < 3:
                continue
            geom = Polygon(p).buffer(0)
            if geom.is_empty or geom.area < 1.0:
                continue
            self.junctions.append(dict(id=junction.get("id"), type=junction.get("type"), polygon=geom))
        self.links = {k: [] for k in self.lanes}
        for conn in root.findall("connection"):
            src = conn.get("from") + "_" + conn.get("fromLane")
            dst = conn.get("to") + "_" + conn.get("toLane")
            via = conn.get("via")
            next_lane = via or dst
            if src in self.lanes and next_lane in self.lanes:
                self.links[src].append(next_lane)
        self.walking = unary_union(self.walking_shapes)
        self.driving = unary_union(self.driving_shapes)

    def supported_routes(self, surface, size, min_length=60.0, max_chains=4, max_hops=12):
        """Validated finite directed chains, including junction curves.

        Graph traversal is allowed only across geometrically continuous joins.
        Dense samples check the complete footprint and curvature, not just centres.
        The bounded search is a candidate sampler, not a proof that every usable
        route has been found. Support predicates are unchanged by these limits.
        """
        from awr.life.civil.grounded import line_samples, segments, support_mask

        result = []
        # Probe the full lane, including interiors whose endpoints are outside
        # the crop. Endpoint-only pruning silently removed through streets.
        usable = {}
        for name, lane in self.lanes.items():
            line = LineString(lane["points"])
            import shapely

            pp = shapely.get_coordinates(shapely.line_interpolate_point(line, np.arange(0.0, line.length + 0.01, 1.0)))
            usable[name] = len(pp) > 0 and surface.supported(pp, "vehicle").any()
        # Finite simple chains. Never replay a fragment as a synthetic loop.
        for start in sorted(self.lanes):
            if self.lanes[start]["internal"]:
                continue
            if not usable[start]:
                continue
            frontier = [(start, [start])]
            count = 0
            while frontier and count < max_chains:
                node, chain = frontier.pop()
                successors = sorted(set(self.links[node]))
                choices = [v for v in successors if v not in chain and usable[v]]
                if choices and len(chain) < max_hops:
                    for v in choices[:3]:
                        frontier.append((v, chain + [v]))
                    continue
                count += 1
                pp = self.lanes[chain[0]]["points"].copy()
                joined = True
                for name in chain[1:]:
                    q = self.lanes[name]["points"]
                    if np.linalg.norm(pp[-1] - q[0]) > 0.25:
                        joined = False
                        break
                    pp = np.vstack([pp, q[1:] if np.linalg.norm(pp[-1] - q[0]) < 1e-5 else q])
                if not joined:
                    continue
                sampled = line_samples(LineString(pp))
                if sampled is None:
                    continue
                points, yaw = sampled
                # Refuse any sharp lane joins. Vehicle headings follow the
                # continuous sampled curve; there is no rotate-in-place fallback.
                valid = support_mask(surface, points, yaw, size, "vehicle")
                for a, b in segments(valid):
                    if (b - a - 1) * 0.35 < min_length:
                        continue
                    result.append(
                        dict(
                            road="sumo:" + start,
                            points=points[a:b],
                            yaw=yaw[a:b],
                            lane_chain=chain,
                            dense_validated=True,
                        )
                    )
        # Deduplicate identical fragments reached through different graph walks.
        unique = {}
        for r in result:
            key = tuple(np.round(r["points"][[0, -1]].ravel(), 1))
            if key not in unique or len(r["points"]) > len(unique[key]["points"]):
                unique[key] = r
        return sorted(unique.values(), key=lambda r: -len(r["points"]))


def extend_support(source, destination, net_path):
    """Recompute driving permission over the SAME measured ground samples."""
    import hashlib
    import json

    import shapely
    from scipy.ndimage import binary_erosion

    from awr.life.civil.surface import SurfaceMap

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError("Support output must be a fresh immutable directory")
    surface = SurfaceMap(source)
    net = LaneNetwork(net_path)
    y, x = np.mgrid[: surface.ny, : surface.nx]
    points = np.stack([surface.x0 + (x + 0.5) * surface.cell_m, surface.y0 + (y + 0.5) * surface.cell_m], -1)
    dy, dx = np.gradient(surface.z, surface.cell_m)
    physical = binary_erosion(np.isfinite(surface.z) & (np.hypot(dx, dy) <= 0.25), iterations=1, border_value=0)
    # Person support already excludes building footprints. It is not enough
    # alone: retain the stricter vehicle slope and measured height conditions.
    legal = shapely.intersects(net.driving, shapely.points(points.reshape(-1, 2))).reshape(x.shape)
    mask = physical & surface.walk["person"] & (legal | surface.walk["vehicle"])
    destination.mkdir(parents=True)
    for name in ("ground.npy", "z_person.npy", "z_vehicle.npy", "walk_person.npy"):
        import shutil

        shutil.copy2(source / name, destination / name)
    np.save(destination / "walk_vehicle.npy", mask)
    meta = dict(
        surface.meta,
        vehicle_cells=int(mask.sum()),
        parent_support=str(source),
        lane_network_sha256=hashlib.sha256(Path(net_path).read_bytes()).hexdigest(),
        driving_permission="supported SUMO lanes including internal connectors; buildings excluded",
    )
    (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
