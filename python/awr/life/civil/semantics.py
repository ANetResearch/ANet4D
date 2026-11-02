"""World semantics for the civil engine: blocks, buildings with doors, POIs, parking (lots, curb, illegal spots), bus
stops, crossings and open areas, derived from the world lane graph, the OSM extract and the navigation grids.

    python -m awr.life.civil.semantics --world <dir with graph.geojson> --osm <osm_world.json> --nav <navgrid dir>
                                    --sumo <sumo dir with net.net.xml + edge_map.json> --stage x0 y0 x1 y1 --out <dir>

Writes ``semantics.json`` and ``anchors.overlay.json`` (awr.anchors.v1). Everything is deterministic given the inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from awr.life.civil.structures import load_terrain, synthetic_buildings, synthetic_lots

BLOCKS = [
    ("NW", "西北街区"),
    ("N", "北街区"),
    ("NE", "东北街区"),
    ("W", "西街区"),
    ("C", "中心街区"),
    ("E", "东街区"),
    ("SW", "西南街区"),
    ("S", "南街区"),
    ("SE", "东南街区"),
]
BUILDING_KIND = {
    "apartments": "apartments",
    "civic": "civic",
    "school": "school",
    "kindergarten": "kindergarten",
    "church": "church",
    "retail": "shop",
    "commercial": "shop",
    "industrial": "industrial",
    "warehouse": "industrial",
    "office": "office",
    "house": "house",
    "residential": "house",
    "detached": "house",
    "yes": "house",
    "roof": "shelter",
    "ruins": "ruin",
    "no": "ruin",
}
POI_WEIGHT = {
    "supermarket": 5.0,
    "cafe": 3.0,
    "bar": 1.5,
    "restaurant": 2.0,
    "bank": 2.0,
    "post_office": 2.0,
    "school": 4.0,
    "kindergarten": 2.5,
    "place_of_worship": 1.0,
    "park": 3.0,
    "pitch": 2.0,
    "stadium": 1.5,
    "townhall": 1.0,
    "courthouse": 1.0,
    "community_centre": 1.5,
    "bus_station": 2.0,
    "kiosk": 2.0,
    "fuel": 1.0,
    "police": 0.5,
    "fire_station": 0.3,
    "hospital": 1.5,
    "pharmacy": 2.0,
    "ferry_terminal": 1.0,
    "square": 3.0,
    "shelter": 0.5,
    "bench": 0.5,
}
SENSITIVE = {"school", "kindergarten", "fire_station", "police", "hospital"}


@dataclass
class Nav:
    x0: float
    y0: float
    nx: int
    ny: int
    walk_p: np.ndarray
    walk_v: np.ndarray
    z: np.ndarray

    @classmethod
    def load(cls, d: Path) -> Nav:
        m = json.loads((d / "meta.json").read_text())
        return cls(
            m["x0"],
            m["y0"],
            m["nx"],
            m["ny"],
            np.load(d / "walk_person.npy"),
            np.load(d / "walk_vehicle.npy"),
            np.load(d / "z_person.npy"),
        )

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        ix, iy = int(math.floor(x - self.x0)), int(math.floor(y - self.y0))
        return (ix, iy) if 0 <= ix < self.nx and 0 <= iy < self.ny else None

    def walkable(self, x: float, y: float, profile: str = "person") -> bool:
        c = self.cell(x, y)
        if c is None:
            return False
        return bool((self.walk_p if profile == "person" else self.walk_v)[c[1], c[0]])

    def z_at(self, x: float, y: float) -> float:
        c = self.cell(x, y)
        if c is None:
            return 0.0
        v = self.z[c[1], c[0]]
        return float(v) if np.isfinite(v) else 0.0

    def nearest_walkable(
        self, x: float, y: float, r: float = 12.0, profile: str = "person"
    ) -> tuple[float, float] | None:
        W = self.walk_p if profile == "person" else self.walk_v
        c = self.cell(x, y)
        if c is None:
            return None
        R = int(r)
        best, bd = None, 1e9
        for dy in range(-R, R + 1):
            iy = c[1] + dy
            if not 0 <= iy < self.ny:
                continue
            for dx in range(-R, R + 1):
                ix = c[0] + dx
                if 0 <= ix < self.nx and W[iy, ix]:
                    d = dx * dx + dy * dy
                    if d < bd:
                        bd, best = d, (self.x0 + ix + 0.5, self.y0 + iy + 0.5)
        return best


def _centroid(P: list) -> tuple[float, float]:
    c = Polygon(P).centroid
    return float(c.x), float(c.y)


def _load_edges(graph: dict, stage: tuple) -> list[dict]:
    x0, y0, x1, y1 = stage
    out = []
    for f in graph["features"]:
        p = f["properties"]
        if p.get("feature") != "edge":
            continue
        line = LineString([(c[0], c[1]) for c in f["geometry"]["coordinates"]])
        if not line.intersects(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])):
            continue
        out.append(
            {
                "id": p["edge_id"],
                "kind": p.get("kind"),
                "width": float(p.get("width_m", 6.4)),
                "speed": float(p.get("speed_mps", 13.9)),
                "highway": p.get("highway"),
                "name": p.get("name"),
                "from": p["from_node"],
                "to": p["to_node"],
                "line": line,
                "oneway": bool(p.get("oneway", False)),
                "rank": p.get("rank", 1),
            }
        )
    return out


def _junction_points(graph: dict) -> list[Point]:
    return [
        Point(f["geometry"]["coordinates"][:2])
        for f in graph["features"]
        if f["properties"].get("feature") == "junction"
    ]


def _sumo_crossings(net_path: Path, stage: tuple) -> list[dict]:
    import sumolib

    net = sumolib.net.readNet(str(net_path), withInternal=True)
    x0, y0, x1, y1 = stage
    out = []
    for e in net.getEdges(withInternal=True):
        if e.getFunction() != "crossing":
            continue
        sh = e.getShape()
        if not sh:
            continue
        mx, my = float(np.mean([p[0] for p in sh])), float(np.mean([p[1] for p in sh]))
        if x0 <= mx <= x1 and y0 <= my <= y1:
            out.append(
                {
                    "id": f"xing-{len(out):03d}",
                    "sumo_edge": e.getID(),
                    "xy": [round(mx, 2), round(my, 2)],
                    "ends": [[round(sh[0][0], 2), round(sh[0][1], 2)], [round(sh[-1][0], 2), round(sh[-1][1], 2)]],
                    "width_m": 4.0,
                    "junction": e.getFromNode().getID(),
                }
            )
    return out


def _sumo_edge_index(sumo_dir: Path) -> dict:
    import sumolib

    net = sumolib.net.readNet(str(sumo_dir / "net.net.xml"))
    emap = json.loads((sumo_dir / "edge_map.json").read_text())
    return {"net": net, "map": emap}


def _sumo_pos(idx: dict, edge_id: str, direction: str, xy: tuple[float, float]) -> dict | None:
    sid = idx["map"].get(edge_id, {}).get(direction)
    if not sid:
        return None
    e = idx["net"].getEdge(sid)
    lane = e.getLanes()[-1]  # rightmost (driving lanes are after sidewalk lane 0 if present)
    for L in e.getLanes():
        if L.allows("passenger"):
            lane = L
            break
    pos, _ = lane.getClosestLanePosAndDist(xy)
    return {"edge": sid, "lane": lane.getID(), "pos": round(float(pos), 2), "length": round(float(lane.getLength()), 2)}


def build(
    graph: dict,
    osm: dict,
    nav: Nav,
    sumo_dir: Path,
    stage: tuple[float, float, float, float],
    terrain: Path | None = None,
) -> tuple[dict, dict]:
    x0, y0, x1, y1 = stage
    stage_poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    # ---------------------------------------------------------------- blocks (3 × 3 sectors)
    blocks = []
    w, h = (x1 - x0) / 3.0, (y1 - y0) / 3.0
    for k, (bid, zh) in enumerate(BLOCKS):
        col, row = k % 3, 2 - k // 3
        bx, by = x0 + col * w, y0 + row * h
        blocks.append(
            {
                "id": f"block-{bid}",
                "label": zh,
                "label_en": f"{bid} block",
                "polygon": [[bx, by], [bx + w, by], [bx + w, by + h], [bx, by + h]],
            }
        )

    def block_of(x: float, y: float) -> str | None:
        for b in blocks:
            P = b["polygon"]
            if P[0][0] <= x < P[2][0] and P[0][1] <= y < P[2][1]:
                return b["id"]
        return None

    # ---------------------------------------------------------------- roads
    edges = _load_edges(graph, stage)
    roads = [e for e in edges if e["kind"] in ("road", "service")]
    road_lines = [e["line"] for e in roads]

    def nearest_road(pt: Point) -> tuple[dict | None, float]:
        best, bd = None, 1e9
        for e in roads:
            d = e["line"].distance(pt)
            if d < bd:
                bd, best = d, e
        return best, bd

    # ---------------------------------------------------------------- buildings and doors
    feats = osm["features"]
    pts_tagged = [f for f in feats if f["geom"] == "Point"]
    buildings, doors_anchor = [], []
    for f in feats:
        if f["geom"] != "Polygon" or "building" not in f["tags"]:
            continue
        P = Polygon(f["xy"])
        if not P.is_valid:
            P = P.buffer(0)
        c = P.centroid
        if not stage_poly.contains(c):
            continue
        t = f["tags"]
        kind = BUILDING_KIND.get(t.get("building"), "house")
        use = t.get("amenity") or t.get("shop") or t.get("leisure")
        for q in pts_tagged:  # a tagged node inside the footprint names its use
            if P.contains(Point(q["xy"][0])):
                use = use or q["tags"].get("amenity") or q["tags"].get("shop")
                if not t.get("name") and q["tags"].get("name"):
                    t = {**t, "name": q["tags"]["name"]}
        if use in (
            "school",
            "kindergarten",
            "place_of_worship",
            "bank",
            "post_office",
            "cafe",
            "supermarket",
            "townhall",
            "courthouse",
            "community_centre",
            "hospital",
            "fire_station",
            "police",
            "bar",
            "restaurant",
            "kiosk",
        ):
            kind = {"place_of_worship": "church"}.get(use, use)
        area = float(P.area)
        floors = {"apartments": 4, "civic": 2, "school": 3, "office": 2, "hospital": 3}.get(
            kind, 2 if area > 150 else 1
        )
        # door: boundary samples pushed outward until on walkable ground; prefer the side facing the nearest road
        ring = P.exterior
        ccw = ring.is_ccw
        n = max(8, int(ring.length / 1.5))
        cands = []
        for i in range(n):
            s = ring.interpolate(i / n, normalized=True)
            s2 = ring.interpolate((i + 0.5) / n, normalized=True)
            tx, ty = s2.x - s.x, s2.y - s.y
            ln = math.hypot(tx, ty) or 1.0
            nx_, ny_ = (ty / ln, -tx / ln) if ccw else (-ty / ln, tx / ln)
            for off in (0.8, 1.5, 2.5):
                qx, qy = s.x + nx_ * off, s.y + ny_ * off
                if nav.walkable(qx, qy):
                    r, d = nearest_road(Point(qx, qy))
                    cands.append((d, qx, qy, i))
                    break
        if cands:
            cands.sort()
            d, dx_, dy_, _ = cands[0]
            door, approx = [round(dx_, 2), round(dy_, 2)], False
        else:
            nw = nav.nearest_walkable(c.x, c.y, 20.0)
            door, approx = ([round(nw[0], 2), round(nw[1], 2)] if nw else [round(c.x, 2), round(c.y, 2)]), True
        bid = f"bld-{len(buildings):03d}"
        households = 1 if kind == "house" else (max(2, int(area / 70.0)) * floors if kind == "apartments" else 0)
        buildings.append(
            {
                "id": bid,
                "osm": f["id"],
                "kind": kind,
                "use": use,
                "name": t.get("name"),
                "polygon": [[round(a, 2), round(b, 2)] for a, b in P.exterior.coords[:-1]],
                "centroid": [round(c.x, 2), round(c.y, 2)],
                "area_m2": round(area, 1),
                "floors": floors,
                "households": households,
                "door": door,
                "door_approx": approx,
                "block": block_of(*door),
            }
        )
    if terrain is not None:
        ter = load_terrain(terrain)
        for b in synthetic_buildings(nav, ter, stage, buildings, roads, block_of):
            b["id"] = f"bld-{len(buildings):03d}"
            buildings.append(b)
    # ---------------------------------------------------------------- POIs (places people go to)
    pois = []

    def add_poi(
        kind: str, name: str | None, xy: list, src: str, building: str | None = None, polygon: list | None = None
    ) -> None:
        wgt = POI_WEIGHT.get(kind, 1.0)
        pois.append(
            {
                "id": f"poi-{len(pois):03d}",
                "kind": kind,
                "name": name,
                "xy": [round(xy[0], 2), round(xy[1], 2)],
                "weight": wgt,
                "building": building,
                "polygon": polygon,
                "block": block_of(xy[0], xy[1]),
                "osm": src,
            }
        )

    for b in buildings:
        if b["kind"] in POI_WEIGHT or b["use"] in POI_WEIGHT:
            add_poi(b["use"] if b["use"] in POI_WEIGHT else b["kind"], b["name"], b["door"], b["osm"], b["id"])
    open_areas = []
    for f in feats:
        t = f["tags"]
        kind = (
            t.get("amenity")
            or t.get("shop")
            or t.get("leisure")
            or ("square" if t.get("highway") == "pedestrian" and f["geom"] == "Polygon" else None)
        )
        if not kind or "building" in t:
            continue
        if f["geom"] == "Point":
            x, y = f["xy"][0]
            if not stage_poly.contains(Point(x, y)):
                continue
            if any(Polygon(b["polygon"]).contains(Point(x, y)) for b in buildings):
                continue  # already represented by the building
            nw = nav.nearest_walkable(x, y, 10.0)
            if nw and kind in POI_WEIGHT:
                add_poi(kind, t.get("name"), list(nw), f["id"])
        elif f["geom"] == "Polygon":
            P = Polygon(f["xy"])
            if not P.is_valid:
                P = P.buffer(0)
            if not stage_poly.intersects(P):
                continue
            if kind == "parking":
                continue
            P = P.intersection(stage_poly)
            if P.is_empty or P.geom_type != "Polygon":
                continue
            c = P.centroid
            nw = nav.nearest_walkable(c.x, c.y, 25.0)
            poly = [[round(a, 2), round(b, 2)] for a, b in P.exterior.coords[:-1]]
            if kind in ("park", "pitch", "square", "stadium", "garden", "playground"):
                open_areas.append(
                    {
                        "id": f"area-{len(open_areas):03d}",
                        "kind": kind,
                        "name": t.get("name"),
                        "polygon": poly,
                        "block": block_of(c.x, c.y),
                    }
                )
            if nw and kind in POI_WEIGHT:
                add_poi(kind, t.get("name"), list(nw), f["id"], polygon=poly)
    # ---------------------------------------------------------------- parking lots
    sidx = _sumo_edge_index(sumo_dir)
    lots = []
    for f in feats:
        if f["geom"] != "Polygon" or f["tags"].get("amenity") != "parking":
            continue
        P = Polygon(f["xy"])
        if not P.is_valid:
            P = P.buffer(0)
        if not stage_poly.contains(P.centroid):
            continue
        mrr = P.minimum_rotated_rectangle
        cs = list(mrr.exterior.coords)
        ex = [(cs[i], cs[i + 1]) for i in range(4)]
        (ax, ay), (bx, by) = max(ex, key=lambda s: math.dist(s[0], s[1]))
        L = math.dist((ax, ay), (bx, by))
        ux, uy = (bx - ax) / L, (by - ay) / L
        vx, vy = -uy, ux
        # rows perpendicular to the long axis every 11 m, slots every 2.7 m along the axis
        slots = []
        W = max(math.dist(ex[1][0], ex[1][1]), math.dist(ex[0][0], ex[0][1]))
        short = min(math.dist(ex[0][0], ex[0][1]), math.dist(ex[1][0], ex[1][1]))
        for r in range(int(short // 11.0) + 1):
            off = 3.0 + r * 11.0
            if off > short - 2.5:
                break
            for k in range(int(L // 2.7)):
                s = 1.5 + k * 2.7
                px, py = ax + ux * s + vx * off, ay + uy * s + vy * off
                if P.contains(Point(px, py)) and nav.walkable(px, py, "vehicle"):
                    slots.append(
                        {
                            "id": f"lot{len(lots)}-{len(slots):02d}",
                            "xy": [round(px, 2), round(py, 2)],
                            "yaw_deg": round(math.degrees(math.atan2(vy, vx)), 1),
                        }
                    )
        if not slots:
            continue
        r, d = nearest_road(P.centroid)
        ent = nearest_points(r["line"], P.exterior)[0] if r else P.centroid
        lots.append(
            {
                "id": f"lot-{len(lots):02d}",
                "name": f["tags"].get("name"),
                "polygon": [[round(a, 2), round(b, 2)] for a, b in P.exterior.coords[:-1]],
                "slots": slots,
                "entrance": [round(ent.x, 2), round(ent.y, 2)],
                "road": r["id"] if r else None,
                "sumo": _sumo_pos(sidx, r["id"], "f", (ent.x, ent.y)) if r else None,
                "block": block_of(P.centroid.x, P.centroid.y),
            }
        )
    excl = [Polygon(b["polygon"]) for b in buildings if b["kind"] in ("school", "kindergarten")] + [
        Polygon(a["polygon"]) for a in open_areas
    ]
    for l in synthetic_lots(nav, roads, stage, excl, block_of, lambda e, d, xy: _sumo_pos(sidx, e, d, xy)):
        l["id"] = f"lot-{len(lots):02d}"
        for k, sl in enumerate(l["slots"]):
            sl["id"] = f"{l['id']}-{k:02d}"
        lots.append(l)
    # ---------------------------------------------------------------- curb slots (legal / illegal) and bus stops
    junctions = _junction_points(graph)
    crossings = _sumo_crossings(sumo_dir / "net.net.xml", stage)
    xing_pts = [Point(c["xy"]) for c in crossings]
    sens_doors = [Point(b["door"]) for b in buildings if b["kind"] in SENSITIVE or b["use"] in SENSITIVE]
    curb = []
    for e in roads:
        if e["kind"] == "service" and e["line"].length < 20:
            continue
        for direction in (("f", 1.0), ("b", -1.0)) if not e["oneway"] else (("f", 1.0),):
            line = e["line"] if direction[0] == "f" else LineString(e["line"].coords[::-1])
            half = e["width"] / 2.0
            s = 12.0
            while s < line.length - 12.0:
                p0, p1 = line.interpolate(s), line.interpolate(min(s + 1.0, line.length))
                tx, ty = p1.x - p0.x, p1.y - p0.y
                ln = math.hypot(tx, ty) or 1.0
                tx, ty = tx / ln, ty / ln
                rx, ry = ty, -tx  # right-hand side in travel direction
                for off, where in ((half + 1.0, "curb"), (half + 2.8, "sidewalk")):
                    qx, qy = p0.x + rx * off, p0.y + ry * off
                    if not stage_poly.contains(Point(qx, qy)):
                        continue
                    on_v = nav.walkable(qx, qy, "vehicle")
                    on_p = nav.walkable(qx, qy, "person")
                    if where == "curb" and not on_v:
                        continue
                    if where == "sidewalk" and not on_p:
                        continue
                    q = Point(qx, qy)
                    reasons = []
                    if where == "sidewalk":
                        reasons.append("on_sidewalk")
                    if min((j.distance(q) for j in junctions), default=99) < 6.0:
                        reasons.append("near_junction")
                    if min((x.distance(q) for x in xing_pts), default=99) < 4.0:
                        reasons.append("on_crossing")
                    if min((d.distance(q) for d in sens_doors), default=99) < 8.0:
                        reasons.append("in_front_of_facility")
                    if where == "curb" and e["kind"] == "road" and e["speed"] > 15.0:
                        reasons.append("main_road")
                    curb.append(
                        {
                            "id": f"curb-{len(curb):04d}",
                            "xy": [round(qx, 2), round(qy, 2)],
                            "yaw_deg": round(math.degrees(math.atan2(ty, tx)), 1),
                            "road": e["id"],
                            "dir": direction[0],
                            "where": where,
                            "legal": not reasons,
                            "reasons": reasons,
                            "sumo": _sumo_pos(sidx, e["id"], direction[0], (p0.x, p0.y)),
                            "block": block_of(qx, qy),
                        }
                    )
                s += 6.0
    bus_stops = []
    main = sorted(
        [e for e in roads if e["kind"] == "road" and e["rank"] >= 3 and e["line"].length > 60],
        key=lambda e: -e["line"].length,
    )
    for e in main[:8]:
        for direction in (("f",), ("b",)) if not e["oneway"] else (("f",),):
            line = e["line"] if direction[0] == "f" else LineString(e["line"].coords[::-1])
            s = line.length * 0.5
            p0, p1 = line.interpolate(s), line.interpolate(min(s + 1.0, line.length))
            tx, ty = p1.x - p0.x, p1.y - p0.y
            ln = math.hypot(tx, ty) or 1.0
            qx, qy = p0.x + ty / ln * (e["width"] / 2 + 2.0), p0.y - tx / ln * (e["width"] / 2 + 2.0)
            if not nav.walkable(qx, qy) or not stage_poly.contains(Point(qx, qy)):
                continue
            if any(math.dist((qx, qy), b["xy"]) < 150 for b in bus_stops):
                continue
            bus_stops.append(
                {
                    "id": f"bus-{len(bus_stops):02d}",
                    "xy": [round(qx, 2), round(qy, 2)],
                    "road": e["id"],
                    "dir": direction[0],
                    "sumo": _sumo_pos(sidx, e["id"], direction[0], (p0.x, p0.y)),
                    "block": block_of(qx, qy),
                }
            )
    for st in bus_stops:  # curb slots near bus stops become illegal
        for c in curb:
            if math.dist(c["xy"], st["xy"]) < 15.0 and "near_bus_stop" not in c["reasons"]:
                c["reasons"].append("near_bus_stop")
                c["legal"] = False
    sem = {
        "schema": "awr.life.civil.semantics.v1",
        "world_id": graph.get("awr", {}).get("world_id"),
        "stage": list(stage),
        "blocks": blocks,
        "buildings": buildings,
        "pois": pois,
        "open_areas": open_areas,
        "parking_lots": lots,
        "curb_slots": curb,
        "bus_stops": bus_stops,
        "crossings": crossings,
        "roads": [
            {
                "id": e["id"],
                "kind": e["kind"],
                "width_m": e["width"],
                "speed_mps": e["speed"],
                "highway": e["highway"],
                "oneway": e["oneway"],
                "line": [[round(a, 2), round(b, 2)] for a, b in e["line"].coords],
            }
            for e in roads
        ],
        "paths": [
            {"id": e["id"], "kind": e["kind"], "line": [[round(a, 2), round(b, 2)] for a, b in e["line"].coords]}
            for e in edges
            if e["kind"] in ("path", "footway")
        ],
    }
    anchors = {"schema": "awr.anchors.v1", "world_id": sem["world_id"], "extends": "world", "anchors": []}
    for b in blocks:
        anchors["anchors"].append(
            {"id": b["id"], "kind": "zone", "polygon": b["polygon"], "label": b["label"], "public": True}
        )
    for b in buildings:
        anchors["anchors"].append(
            {
                "id": f"door-{b['id']}",
                "kind": "door",
                "xy": b["door"],
                "building": b["id"],
                "label": b["name"] or b["kind"],
                "public": False,
            }
        )
    for p in pois:
        anchors["anchors"].append(
            {"id": p["id"], "kind": "point", "xy": p["xy"], "label": p["name"] or p["kind"], "public": True}
        )
    for l in lots:
        anchors["anchors"].append(
            {"id": l["id"], "kind": "zone", "polygon": l["polygon"], "label": l["name"] or "停车场", "public": True}
        )
    for s in bus_stops:
        anchors["anchors"].append({"id": s["id"], "kind": "point", "xy": s["xy"], "label": "公交站", "public": True})
    return sem, anchors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, type=Path)
    ap.add_argument("--osm", required=True, type=Path)
    ap.add_argument("--nav", required=True, type=Path)
    ap.add_argument("--sumo", required=True, type=Path)
    ap.add_argument("--stage", nargs=4, type=float, default=[-300, -260, 320, 260])
    ap.add_argument(
        "--terrain",
        type=Path,
        default=None,
        help="dir with dsm_2m/dtm_10m/quality_2m (default: <world>/geometry/terrain)",
    )
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    gp = a.world / "geometry" / "nav" / "graph.geojson"
    graph = json.loads((gp if gp.exists() else a.world / "graph.geojson").read_text(encoding="utf-8"))
    osm = json.loads(a.osm.read_text(encoding="utf-8"))
    ter = a.terrain or (a.world / "geometry" / "terrain")
    sem, anchors = build(graph, osm, Nav.load(a.nav), a.sumo, tuple(a.stage), ter if ter.exists() else None)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "semantics.json").write_text(json.dumps(sem, ensure_ascii=False), encoding="utf-8")
    (a.out / "anchors.overlay.json").write_text(json.dumps(anchors, ensure_ascii=False, indent=0), encoding="utf-8")
    kinds = {}
    for b in sem["buildings"]:
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    print(
        f"buildings {len(sem['buildings'])} (synthetic {sum(1 for b in sem['buildings'] if b.get('synthetic'))}) {kinds}; households {sum(b['households'] for b in sem['buildings'])}; approx doors {sum(b['door_approx'] for b in sem['buildings'])}"
    )
    print(f"pois {len(sem['pois'])}: {sorted(set(p['kind'] for p in sem['pois']))}")
    print(
        f"open areas {len(sem['open_areas'])}, lots {len(sem['parking_lots'])} slots {sum(len(l['slots']) for l in sem['parking_lots'])}, "
        f"curb slots {len(sem['curb_slots'])} legal {sum(c['legal'] for c in sem['curb_slots'])}, bus stops {len(sem['bus_stops'])}, crossings {len(sem['crossings'])}, roads {len(sem['roads'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
