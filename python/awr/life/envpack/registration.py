"""Horizontal registration of mapped features to the reconstruction, from independent evidence in the DSM.

Mapped building footprints should stand above their surroundings and mapped road centre lines should lie on open,
even ground. For translations of the mapped features over a grid, `scores` rates both against the world package's
DSM (2 m, highest mesh hit): buildings by the mean height inside the footprint above the median of a 2-6 m ring
around it (a partial overlap scores lower than a full one), roads by the
share of centre-line samples whose DSM lies within 1 m of the local ground (the 5th percentile of the DSM in a
12 m window). The support layers are not used, so the estimate is independent of the support build. `estimate`
searches the whole map and tiles of it; `fit_similarity` fits rotation, scale and translation to the tile
translations. Used by awr.life.envpack.validate (frame.feature_registration) and scripts/envpack/estimate_registration.py.
"""

import numpy as np
from scipy.ndimage import percentile_filter
from shapely.geometry import Polygon

from .validate import Raster, line_samples, polygon_lattice


def local_ground(dsm, window_m=12.0):
    """A copy of the DSM raster holding the 5th percentile in a window: the open ground level around each cell."""
    ground = Raster.__new__(Raster)
    ground.__dict__.update(dsm.__dict__)
    window = int(round(window_m / dsm.cell)) | 1
    ground.data = percentile_filter(np.where(np.isfinite(dsm.data), dsm.data, np.nanmax(dsm.data)), 5, size=window)
    return ground


def features(sem):
    """Building interiors and rings (with centroids) and road centre-line samples of a semantics document."""
    buildings = []
    for b in sem.get("buildings", []):
        poly = Polygon(b["polygon"])
        if not poly.is_valid or poly.area < 30:
            continue
        inner = polygon_lattice(poly.buffer(-1.0), 1.0) if poly.buffer(-1.0).area > 4 else polygon_lattice(poly, 1.0)
        ring = polygon_lattice(poly.buffer(6.0).difference(poly.buffer(2.0)), 1.0)
        buildings.append((inner, ring, np.asarray(poly.centroid.coords[0])))
    roads = [line_samples(r["line"], 2.0)[0] for r in sem.get("roads", [])]
    return buildings, (np.vstack(roads) if roads else np.zeros((0, 2)))


def scores(dsm, ground, buildings, roads, shifts):
    """Building lift (m) and open-road share for each translation."""
    out = {}
    for dx in shifts:
        for dy in shifts:
            d = np.array([dx, dy])
            lift = []
            for inner, ring in buildings:
                lift.append(np.nanmean(dsm.at(inner + d, fill=np.nan)) - np.nanmedian(dsm.at(ring + d, fill=np.nan)))
            road = roads + d
            open_ = np.abs(dsm.at(road, fill=np.nan).astype(float) - ground.at(road, fill=np.nan)) <= 1.0
            out[(float(dx), float(dy))] = (
                float(np.nanmean(lift)) if lift else np.nan,
                float(open_.mean()) if len(road) else np.nan,
            )
    return out


def transform(points, params, centre):
    """Apply a similarity (a, b, tx, ty): p' = c + [[a, -b], [b, a]] (p - c) + t."""
    a, b, tx, ty = params
    q = np.asarray(points, float) - centre
    return np.column_stack([a * q[:, 0] - b * q[:, 1], b * q[:, 0] + a * q[:, 1]]) + centre + [tx, ty]


def fit_similarity(tiles, centre):
    """Weighted least squares of the tile translations t_i = (A - I)(c_i - c) + T for A = [[a, -b], [b, a]]."""
    rows, rhs, w = [], [], []
    for t in tiles:
        q = np.asarray(t["centre"]) - centre
        sx, sy = t["shift_m"]
        weight = np.sqrt(t["road_samples"])
        rows += [[q[0], -q[1], 1.0, 0.0], [q[1], q[0], 0.0, 1.0]]
        rhs += [sx + q[0], sy + q[1]]
        w += [weight, weight]
    A, y, w = np.asarray(rows), np.asarray(rhs), np.asarray(w)
    params, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    predicted = A @ params
    residual = (predicted - y).reshape(-1, 2)
    return params, residual


def score_transformed(dsm, ground, buildings, roads, params, centre):
    lift = [
        np.nanmean(dsm.at(transform(i, params, centre), fill=np.nan))
        - np.nanmedian(dsm.at(transform(r, params, centre), fill=np.nan))
        for i, r in buildings
    ]
    road = transform(roads, params, centre)
    open_ = np.abs(dsm.at(road, fill=np.nan).astype(float) - ground.at(road, fill=np.nan)) <= 1.0
    return float(np.nanmean(lift)), float(open_.mean())


def best(table):
    """Translation maximising the product of the normalised building lift and open-road share."""
    lifts = np.array([v[0] for v in table.values()])
    opens = np.array([v[1] for v in table.values()])
    if np.isfinite(lifts).any():
        lo, hi = np.nanmin(lifts), np.nanmax(lifts)
        combined = ((lifts - lo) / max(hi - lo, 1e-9)) * opens
    else:  # no building in this part of the map: roads alone
        combined = opens
    # Equal scores (a footprint smaller than its building, a uniform road) resolve to the smallest shift.
    size = np.array([np.hypot(*key) for key in table])
    k = int(np.nanargmax(np.where(np.isfinite(combined), combined, -np.inf) - 1e-9 * size))
    key = list(table)[k]

    def r(v, n):
        return round(v, n) if np.isfinite(v) else None

    return dict(
        shift_m=list(key),
        building_lift_m=r(table[key][0], 3),
        open_road_share=r(table[key][1], 4),
        at_zero=dict(building_lift_m=r(table[(0.0, 0.0)][0], 3), open_road_share=r(table[(0.0, 0.0)][1], 4)),
        best_by_buildings=list(list(table)[int(np.nanargmax(lifts))]) if np.isfinite(lifts).any() else None,
        best_by_roads=list(list(table)[int(np.nanargmax(opens))]) if np.isfinite(opens).any() else None,
    )


def estimate(world, sem, search_m=8.0, step_m=0.5, tiles=0, tile_step_m=1.0):
    """Best translation over the whole map (and per tile, with a similarity fit, when `tiles` > 0)."""
    dsm = Raster(world, "dsm_2m")
    ground = local_ground(dsm)
    buildings, roads = features(sem)
    shifts = np.arange(-search_m, search_m + 1e-9, step_m)
    overall = best(scores(dsm, ground, [(i, r) for i, r, _ in buildings], roads, shifts))
    out = dict(
        buildings=len(buildings),
        road_samples=int(len(roads)),
        search=dict(range_m=search_m, step_m=step_m),
        overall=overall,
        similarity=None,
        tiles=[],
    )
    if not tiles:
        return out
    x0, y0, x1, y1 = sem["bounds"]
    for tx in range(tiles):
        for ty in range(tiles):
            lo = np.array([x0 + (x1 - x0) * tx / tiles, y0 + (y1 - y0) * ty / tiles])
            hi = np.array([x0 + (x1 - x0) * (tx + 1) / tiles, y0 + (y1 - y0) * (ty + 1) / tiles])
            bs = [(i, r) for i, r, c in buildings if (c >= lo).all() and (c < hi).all()]
            rs = roads[((roads >= lo) & (roads < hi)).all(1)]
            if len(rs) < 200:
                continue
            t = best(scores(dsm, ground, bs, rs, np.arange(-search_m, search_m + 1e-9, tile_step_m)))
            out["tiles"].append(
                dict(tile=[tx, ty], centre=((lo + hi) / 2).tolist(), buildings=len(bs), road_samples=int(len(rs)), **t)
            )
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    if len(out["tiles"]) >= 3:
        params, residual = fit_similarity(out["tiles"], centre)
        lift, open_share = score_transformed(dsm, ground, [(i, r) for i, r, _ in buildings], roads, params, centre)
        out["similarity"] = dict(
            centre=centre.tolist(),
            a=float(params[0]),
            b=float(params[1]),
            translation_m=[float(params[2]), float(params[3])],
            rotation_deg=round(float(np.degrees(np.arctan2(params[1], params[0]))), 4),
            scale=round(float(np.hypot(params[0], params[1])), 6),
            tile_residual_m=[round(float(np.hypot(*r)), 2) for r in residual],
            building_lift_m=round(lift, 3),
            open_road_share=round(open_share, 4),
        )
    return out
