"""Structures that OSM does not carry: residences inferred from the photogrammetry (height above ground next to the
walkable network) and off-road drivable areas used as parking lots.

The DSM/DTM rasters of the world package give height above ground at 2 m; the quality raster's ``multilayer`` bit
marks canopy-like surfaces. A walkable cell whose unwalkable neighbour is ≥ 3 m high and not canopy is a candidate
house front; candidates are clustered every ~14 m along the walkable boundary into synthetic buildings with a door on
the walkable side. Vehicle-walkable cells outside road corridors form yards/lots; slots are laid on a lattice inside.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon


def load_terrain(terrain: Path) -> dict:
    dj = json.loads((terrain / "dsm_2m.json").read_text())
    tj = json.loads((terrain / "dtm_10m.json").read_text())
    qj = json.loads((terrain / "quality_2m.json").read_text())
    dsm = np.fromfile(terrain / dj["href"], np.float32).reshape(dj["height"], dj["width"])
    dtm = np.fromfile(terrain / tj["href"], np.float32).reshape(tj["height"], tj["width"])
    ql = np.fromfile(terrain / qj["href"], np.uint8).reshape(qj["height"], qj["width"])
    dtm2 = ndimage.zoom(dtm, (dj["height"] / tj["height"], dj["width"] / tj["width"]), order=1)
    return {
        "ndsm": dsm - dtm2,
        "multi": ((ql >> 4) & 1).astype(bool),
        "origin": dj["originXY"],
        "cell": float(dj["cellM"]),
    }


def _sample(ter: dict, key: str, x: float, y: float):
    c = int((x - ter["origin"][0]) / ter["cell"]), int((y - ter["origin"][1]) / ter["cell"])
    A = ter[key]
    if 0 <= c[1] < A.shape[0] and 0 <= c[0] < A.shape[1]:
        return A[c[1], c[0]]
    return 0


def synthetic_buildings(
    nav,
    ter: dict,
    stage: tuple,
    known: list[dict],
    roads: list[dict],
    block_of,
    *,
    spacing_m: float = 14.0,
    seed: int = 1,
) -> list[dict]:
    x0, y0, x1, y1 = stage
    W = nav.walk_p
    ny, nx = W.shape
    known_polys = [Polygon(b["polygon"]).buffer(3.0) for b in known]
    # boundary walkable cells with a tall, non-canopy unwalkable 4-neighbour
    cands = []
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            if not W[iy, ix]:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jx, jy = ix + dx, iy + dy
                if W[jy, jx]:
                    continue
                qx, qy = nav.x0 + jx + 0.5 + dx * 2.0, nav.y0 + jy + 0.5 + dy * 2.0
                h = float(_sample(ter, "ndsm", qx, qy))
                if h >= 3.0 and h <= 20.0 and not _sample(ter, "multi", qx, qy):
                    px, py = nav.x0 + ix + 0.5, nav.y0 + iy + 0.5
                    if x0 + 5 <= px <= x1 - 5 and y0 + 5 <= py <= y1 - 5:
                        cands.append((px, py, dx, dy, h))
                    break
    # drop candidates inside/near known footprints or on the carriageway side (distance to road centreline < width/2 + 1)
    out = []
    for px, py, dx, dy, h in cands:
        pt = Point(px, py)
        if any(P.contains(pt) for P in known_polys):
            continue
        near = min(((LineString(r["line"]).distance(pt), r) for r in roads), key=lambda t: t[0], default=(99.0, None))
        if near[1] is not None and near[0] < near[1]["width"] / 2.0 + 0.8:
            continue
        out.append((px, py, dx, dy, h))
    # greedy clustering along the boundary
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(out))
    taken = np.zeros(len(out), bool)
    pts = np.array([(c[0], c[1]) for c in out]) if out else np.zeros((0, 2))
    blds = []
    for i in order:
        if taken[i]:
            continue
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        grp = np.flatnonzero((d < spacing_m) & ~taken)
        taken[grp] = True
        px, py, dx, dy, h = out[i]
        hs = [out[j][4] for j in grp]
        hgt = float(np.median(hs))
        depth = 8.0 if hgt < 7.0 else 12.0
        half = min(spacing_m / 2.0, 7.0)
        # footprint: a rectangle starting 1 m beyond the door, extending `depth` m into the structure, `2·half` m wide
        ux, uy = dx, dy
        vx, vy = -uy, ux
        bx, by = px + ux * 1.0, py + uy * 1.0
        poly = [
            [bx + vx * half, by + vy * half],
            [bx + vx * half + ux * depth, by + vy * half + uy * depth],
            [bx - vx * half + ux * depth, by - vy * half + uy * depth],
            [bx - vx * half, by - vy * half],
        ]
        kind = "apartments" if hgt >= 9.0 else "house"
        blds.append(
            {
                "id": "",
                "osm": None,
                "kind": kind,
                "use": None,
                "name": None,
                "polygon": [[round(a, 2), round(b, 2)] for a, b in poly],
                "centroid": [round(bx + ux * depth / 2, 2), round(by + uy * depth / 2, 2)],
                "area_m2": round(2 * half * depth, 1),
                "floors": 1 if hgt < 5.0 else (2 if hgt < 9.0 else 3),
                "households": 1 if hgt < 6.0 else (2 if hgt < 9.0 else 6),
                "door": [round(px, 2), round(py, 2)],
                "door_approx": False,
                "block": block_of(px, py),
                "height_m": round(hgt, 1),
                "synthetic": True,
            }
        )
    blds.sort(key=lambda b: (b["door"][1], b["door"][0]))
    return blds


def synthetic_lots(
    nav, roads: list[dict], stage: tuple, exclude: list[Polygon], block_of, sumo_pos, *, min_area_m2: float = 150.0
) -> list[dict]:
    import shapely

    x0, y0, x1, y1 = stage
    Wv = nav.walk_v
    ny, nx = Wv.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    px, py = xs + nav.x0 + 0.5, ys + nav.y0 + 0.5
    pts = shapely.points(px.ravel(), py.ravel())
    corr = np.zeros_like(Wv)
    for r in roads:
        corr.ravel()[shapely.contains(LineString(r["line"]).buffer(r["width"] / 2.0 + 1.5), pts)] = True
    cand = Wv & ~corr
    for P in exclude:
        cand.ravel()[shapely.contains(P, pts)] = False
    cand = ndimage.binary_erosion(cand, iterations=1)
    lab, n = ndimage.label(cand)
    sizes = ndimage.sum(cand, lab, range(1, n + 1))
    lots = []
    for i in range(n):
        if sizes[i] < min_area_m2:
            continue
        cells = np.argwhere(lab == i + 1)
        cx, cy = float(cells[:, 1].mean() + nav.x0 + 0.5), float(cells[:, 0].mean() + nav.y0 + 0.5)
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        near = min(((LineString(r["line"]).distance(Point(cx, cy)), r) for r in roads), key=lambda t: t[0])
        if near[0] > 60.0:
            continue
        road = near[1]
        line = LineString(road["line"])
        s = line.project(Point(cx, cy))
        p0, p1 = line.interpolate(max(0.0, s - 1)), line.interpolate(min(line.length, s + 1))
        ux, uy = p1.x - p0.x, p1.y - p0.y
        ln = math.hypot(ux, uy) or 1.0
        ux, uy = ux / ln, uy / ln
        vx, vy = -uy, ux
        mask = lab == i + 1
        slots = []
        # lattice: rows along the road direction (slot every 2.7 m), rows every 6 m across; cars park perpendicular to rows
        for r_ in range(-12, 13):
            for k in range(-30, 31):
                qx, qy = cx + ux * k * 2.7 + vx * r_ * 6.0, cy + uy * k * 2.7 + vy * r_ * 6.0
                c = nav.cell(qx, qy)
                if c is None or not mask[c[1], c[0]]:
                    continue
                # require 1 m clearance all around inside the component
                ok = all(
                    nav.cell(qx + ox, qy + oy) is not None
                    and mask[nav.cell(qx + ox, qy + oy)[1], nav.cell(qx + ox, qy + oy)[0]]
                    for ox, oy in ((1.3, 0), (-1.3, 0), (0, 2.6), (0, -2.6))
                )
                if ok:
                    slots.append(
                        {
                            "id": "",
                            "xy": [round(qx, 2), round(qy, 2)],
                            "yaw_deg": round(math.degrees(math.atan2(vy, vx)), 1),
                        }
                    )
        if len(slots) < 3:
            continue
        ent = line.interpolate(s)
        # polygon: convex hull of the component cells
        hull = shapely.MultiPoint(
            [(nav.x0 + c[1] + 0.5, nav.y0 + c[0] + 0.5) for c in cells[:: max(1, len(cells) // 400)]]
        ).convex_hull
        poly = [[round(a, 2), round(b, 2)] for a, b in hull.exterior.coords[:-1]] if hull.geom_type == "Polygon" else []
        lid = f"lot-{len(lots):02d}"
        for k, sl in enumerate(slots):
            sl["id"] = f"{lid}-{k:02d}"
        lots.append(
            {
                "id": lid,
                "name": None,
                "polygon": poly,
                "slots": slots,
                "entrance": [round(ent.x, 2), round(ent.y, 2)],
                "road": road["id"],
                "sumo": sumo_pos(road["id"], "f", (ent.x, ent.y)),
                "block": block_of(cx, cy),
                "area_m2": float(sizes[i]),
                "synthetic": True,
            }
        )
    return lots
