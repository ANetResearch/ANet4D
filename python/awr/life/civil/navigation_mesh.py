"""Compile measured, permitted surface triangles for optional navigation backends.

This is a static candidate mesh, never vehicle/pedestrian motion admission.
Backends such as Recast may build a reusable route guide from it. Actual body,
turning, collision, directional-road and parking checks remain authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import shapely


def support_mesh(surface, area, profile, *, bounds=None, chunk_rows=32):
    """Return compact Recast (east, up, -north) vertices and triangle indices.

    Only complete quads between four finite, profile-supported cell centres are
    admitted. The *entire quad* must be covered by the explicit semantic area,
    including its holes, not merely its four corners. Unknown cells are not
    filled. Work is chunked by rows; no whole-world XYZ/polygon grid is built.
    Output ordering is stable across chunk sizes.
    """
    if profile not in ("person", "vehicle"):
        raise ValueError("Explicit person or vehicle profile required")
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        raise ValueError("Positive integer chunk_rows required")
    z = np.asarray(surface.z)
    valid = np.asarray(surface.walk[profile], bool)
    nx, ny, cell = surface.nx, surface.ny, float(surface.cell_m)
    if z.shape != (ny, nx) or valid.shape != z.shape or nx < 2 or ny < 2:
        raise ValueError("Support arrays must match a grid of at least 2 x 2")
    if not np.isfinite([surface.x0, surface.y0, cell]).all() or cell <= 0:
        raise ValueError("Finite support origin and positive cell size required")
    if area is None or area.is_empty or not area.is_valid or area.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("Explicit nonempty valid polygonal permission required")
    if bounds is not None:
        b = np.asarray(bounds, float)
        if b.shape != (4,) or not np.isfinite(b).all() or b[0] >= b[2] or b[1] >= b[3]:
            raise ValueError("Bounds must be finite xmin,ymin,xmax,ymax")
        area = area.intersection(shapely.box(*b))
    if area.is_empty:
        return np.empty((0, 3), "<f4"), np.empty((0, 3), "<u4")
    xmin, ymin, xmax, ymax = area.bounds
    # Vertex indices; the end is inclusive. Quads always stay inside the grid.
    c0 = max(0, math.ceil((xmin - surface.x0) / cell - 0.5))
    c1 = min(nx - 1, math.floor((xmax - surface.x0) / cell - 0.5))
    r0 = max(0, math.ceil((ymin - surface.y0) / cell - 0.5))
    r1 = min(ny - 1, math.floor((ymax - surface.y0) / cell - 0.5))
    quads = []
    shapely.prepare(area)
    for start in range(r0, r1, chunk_rows):
        end = min(start + chunk_rows, r1)
        ok = valid[start : end + 1, c0 : c1 + 1] & np.isfinite(z[start : end + 1, c0 : c1 + 1])
        full = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
        rr, cc = np.nonzero(full)
        rr += start
        cc += c0
        x = surface.x0 + (cc + 0.5) * cell
        y = surface.y0 + (rr + 0.5) * cell
        permitted = shapely.covers(area, shapely.box(x, y, x + cell, y + cell))
        a = rr[permitted].astype(np.int64) * nx + cc[permitted]
        if len(a):
            quads.append(np.stack([a, a + 1, a + nx, a + nx + 1], 1))
    if not quads:
        return np.empty((0, 3), "<f4"), np.empty((0, 3), "<u4")
    q = np.concatenate(quads)
    used, inverse = np.unique(q, return_inverse=True)
    if len(used) > np.iinfo(np.uint32).max:
        raise ValueError("Navigation vertex index exceeds uint32 capacity")
    ids = inverse.reshape(-1, 4)
    # Positive up-normal under the ENU -> Recast handedness-preserving transform.
    triangles = np.stack([ids[:, [0, 1, 2]], ids[:, [1, 3, 2]]], 1).reshape(-1, 3).astype("<u4")
    rr, cc = used // nx, used % nx
    vertices = np.stack([surface.x0 + (cc + 0.5) * cell, z[rr, cc], -(surface.y0 + (rr + 0.5) * cell)], 1).astype("<f4")
    return vertices, triangles


def export(support, permission, profile, out, *, bounds=None, chunk_rows=32):
    """Write immutable navigation input plus provenance, rejecting empty input."""
    from .surface import SurfaceMap

    support, permission, out = Path(support), Path(permission), Path(out)
    area = shapely.from_geojson(permission.read_text())
    vertices, triangles = support_mesh(SurfaceMap(support), area, profile, bounds=bounds, chunk_rows=chunk_rows)
    if not len(triangles):
        raise ValueError("No fully supported and permitted triangles")
    out.mkdir(parents=True, exist_ok=False)
    vertices.tofile(out / "positions.f32")
    triangles.tofile(out / "indices.u32")
    inputs = [support / "meta.json", support / "ground.npy", support / f"walk_{profile}.npy", permission]

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    report = dict(
        schema="awr.life.navigation_candidate_mesh.v1",
        release_evidence=False,
        profile=profile,
        vertices=len(vertices),
        triangles=len(triangles),
        bounds_enu=bounds,
        transform="Recast(x,y,z) = ENU(x,z,-y)",
        chunk_rows=chunk_rows,
        inputs={str(p): digest(p) for p in inputs},
        outputs={name: digest(out / name) for name in ("positions.f32", "indices.u32")},
        limitations=[
            "Support-derived surface; not the original collision triangle mesh.",
            "Route guidance only; body, turning, road direction and parking admission still required.",
        ],
    )
    (out / "input.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--support", type=Path, required=True)
    p.add_argument("--permission", type=Path, required=True, help="Polygon/MultiPolygon GeoJSON geometry")
    p.add_argument("--profile", choices=("person", "vehicle"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bounds", nargs=4, type=float)
    p.add_argument("--chunk-rows", type=int, default=32)
    a = p.parse_args()
    print(json.dumps(export(a.support, a.permission, a.profile, a.out, bounds=a.bounds, chunk_rows=a.chunk_rows)))
