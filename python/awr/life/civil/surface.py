"""Validated support surfaces, separate from the photogrammetric top surface.

The terrain model is only a height *hint*. A cell is accepted only after a
raycast-mesh intersection close to that hint. Missing intersections stay blocked.
This first implementation deliberately rejects unsupported multi-level regions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from awr.life.common.schema import schema_id


class SurfaceError(ValueError):
    pass


def flat_vehicle_ground(z, person, cell_m, max_slope=0.25):
    """Cells a vehicle can stand on by the measured ground alone: finite ground of slope at most `max_slope`, one cell
    in from the edge of such ground, and clear for a person (`person`: the support's person cells).

    The slope is the central difference over two cells, so a step of more than `max_slope` times two cells (0.25 m at
    0.5 m cells: a raised kerb, a wall foot, a bank) bounds these cells. A town support restricts them further to the
    mapped roads (scripts/sandbox/build_town_support.py); a measurement of the mapped roads against the scene uses them
    without that restriction."""
    from scipy.ndimage import binary_erosion

    dy, dx = np.gradient(z, cell_m)
    slope = np.hypot(dx, dy)
    return binary_erosion(np.isfinite(z) & np.isfinite(slope) & (slope <= max_slope), iterations=1) & person


def graded_vehicle_ground(z, person, cell_m, *, window_m=2.5, max_grade=0.25, max_step=0.1, min_measured=0.8):
    """Cells a vehicle can stand on with the ground taken at the scale of the car: the ground smoothed by a median over
    `window_m` has a grade of at most `max_grade`, the cell lies within `max_step` of that smoothed ground, at least
    `min_measured` of the window is measured ground, one cell in from the edge of such ground, and clear for a person.

    flat_vehicle_ground takes the slope over two cells of the measured ground. Measured road surfaces carry
    centimetre to decimetre noise (photogrammetric reconstruction), so cells of a level road with a step of ten
    centimetres beside them exceed that slope and, eroded by one cell, leave holes in the carriageway that no car
    could pass. A median keeps a sharp edge sharp: across a step of height h the smoothed grade is still about h per
    two cells, so the step a car may cross is bounded as in flat_vehicle_ground; only isolated deviations of up to
    `max_step` (well below a kerb and a car's ground clearance) are taken as the road they lie on."""
    from scipy.ndimage import binary_erosion, distance_transform_edt, median_filter, uniform_filter

    finite = np.isfinite(z)
    if not finite.any():
        return np.zeros(z.shape, bool)
    size = max(int(round(window_m / cell_m)) | 1, 3)
    filled = z[tuple(distance_transform_edt(~finite, return_distances=False, return_indices=True))]
    smooth = median_filter(filled, size=size, mode="nearest")
    measured = uniform_filter(finite.astype(float), size, mode="constant")
    dy, dx = np.gradient(smooth, cell_m)
    with np.errstate(invalid="ignore"):
        ok = (
            finite
            & (measured >= min_measured - 1e-9)
            & (np.hypot(dx, dy) <= max_grade)
            & (np.abs(z - smooth) <= max_step)
        )
    return binary_erosion(ok, iterations=1) & person


class SurfaceMap:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        if schema_id(self.meta.get("schema")) != "awr.life.support.v1":
            raise SurfaceError("Unvalidated legacy DSM grid: rebuild with awr.life.civil.surface")
        self.x0, self.y0 = self.meta["x0"], self.meta["y0"]
        self.nx, self.ny = self.meta["nx"], self.meta["ny"]
        self.cell_m = float(self.meta.get("cell_m", 1.0))
        if not np.isfinite(self.cell_m) or self.cell_m <= 0:
            raise SurfaceError("Invalid support cell size")
        self.width_m, self.height_m = self.nx * self.cell_m, self.ny * self.cell_m
        self.z = np.load(self.root / "ground.npy")
        self.walk = {k: np.load(self.root / f"walk_{k}.npy") for k in ("person", "vehicle")}

    def supported(self, xy, profile="person"):
        """Per-point version of sample's validity gate for batched swept footprints."""
        p = np.asarray(xy, float).reshape(-1, 2)
        finite = np.isfinite(p).all(1)
        p = np.where(np.isfinite(p), p, 0)
        gx, gy = (p[:, 0] - self.x0) / self.cell_m - 0.5, (p[:, 1] - self.y0) / self.cell_m - 0.5
        ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
        inside = finite & (ix >= 0) & (iy >= 0) & (ix + 1 < self.nx) & (iy + 1 < self.ny)
        ix = np.clip(ix, 0, self.nx - 2)
        iy = np.clip(iy, 0, self.ny - 2)
        valid = np.isfinite(
            np.stack([self.z[iy, ix], self.z[iy, ix + 1], self.z[iy + 1, ix], self.z[iy + 1, ix + 1]], 1)
        ).all(1)
        cx = np.clip(np.floor((p[:, 0] - self.x0) / self.cell_m).astype(int), 0, self.nx - 1)
        cy = np.clip(np.floor((p[:, 1] - self.y0) / self.cell_m).astype(int), 0, self.ny - 1)
        return inside & valid & self.walk[profile][cy, cx]

    def sample(self, xy, profile="person", *, require_walkable=True):
        p = np.asarray(xy, float).reshape(-1, 2)
        if not np.isfinite(p).all():
            raise SurfaceError("Non-finite surface coordinates")
        # Heights are stored at cell centres. Bilinear interpolation prevents
        # the 1 m staircase that arose from nearest-cell height sampling.
        gx, gy = (p[:, 0] - self.x0) / self.cell_m - 0.5, (p[:, 1] - self.y0) / self.cell_m - 0.5
        ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix + 1 < self.nx) & (iy + 1 < self.ny)
        if not inside.all():
            raise SurfaceError(f"{int((~inside).sum())} positions outside supported surface")
        values = np.stack([self.z[iy, ix], self.z[iy, ix + 1], self.z[iy + 1, ix], self.z[iy + 1, ix + 1]], 1)
        valid = np.isfinite(values).all(1)
        if require_walkable:
            cx, cy = (
                np.floor((p[:, 0] - self.x0) / self.cell_m).astype(int),
                np.floor((p[:, 1] - self.y0) / self.cell_m).astype(int),
            )
            valid &= self.walk[profile][cy, cx]
        if not valid.all():
            bad = p[~valid][:5].tolist()
            raise SurfaceError(f"{int((~valid).sum())} unsupported {profile} positions, examples {bad}")
        u, v = gx - ix, gy - iy
        return (
            values[:, 0] * (1 - u) * (1 - v)
            + values[:, 1] * u * (1 - v)
            + values[:, 2] * (1 - u) * v
            + values[:, 3] * u * v
        )


def build(world: Path, nav: Path, out: Path, semantics: Path | None = None, *, local_hint=False):
    import shapely
    from scipy.ndimage import binary_erosion
    from shapely.geometry import LineString, Polygon

    from awr.sim.events.ground import build_ground_index
    from awr.world.mesh.query import open_mesh_world_query

    meta = json.loads((nav / "meta.json").read_text())
    x0, y0, nx, ny = (meta[k] for k in ("x0", "y0", "nx", "ny"))
    yy, xx = np.mgrid[:ny, :nx]
    points = np.column_stack([(xx + x0 + 0.5).ravel(), (yy + y0 + 0.5).ravel()])
    td = world / "geometry/terrain"
    tm = json.loads((td / "dtm_10m.json").read_text())
    dtm = np.fromfile(td / tm["href"], "<f4").reshape(tm["height"], tm["width"])
    coords = [
        (points[:, 1] - tm["originXY"][1]) / tm["cellM"] - 0.5,
        (points[:, 0] - tm["originXY"][0]) / tm["cellM"] - 0.5,
    ]
    hint = map_coordinates(dtm, coords, order=1, mode="constant", cval=np.nan)
    if local_hint:
        from awr.life.civil.terrain_hint import terrain_hint

        fine = terrain_hint(np.load(nav / "z_person.npy")).ravel()
        # The top-surface observation has finer spatial resolution than the
        # 10 m DTM. Neither it nor its interpolation is accepted without a hit.
        hint = np.where(np.isfinite(fine), fine, hint)
    q = open_mesh_world_query(world, world.parent.parent / ".cache")
    gi = build_ground_index(q, (x0, y0, x0 + nx, y0 + ny), margin_m=2.0)
    good_hint = np.isfinite(hint)
    z = np.full(len(points), np.nan)
    found = np.zeros(len(points), bool)
    z[good_hint], found[good_hint] = gi.ground_z_batch(points[good_hint], hint[good_hint])
    z[~found] = np.nan
    Z = z.reshape(ny, nx)
    dy, dx = np.gradient(Z)
    slope = np.hypot(dx, dy)
    valid = np.isfinite(Z) & np.isfinite(slope)
    # Conservative one-cell clearance from unknown or steep ground. Do not
    # connect a roof to the ground merely because their XY projections touch.
    person = binary_erosion(valid & (slope <= 0.7), iterations=1, border_value=0)
    vehicle = binary_erosion(valid & (slope <= 0.25), iterations=1, border_value=0)
    if semantics:
        sem = json.loads(semantics.read_text())
        pts = shapely.points(points)
        roads = np.zeros(len(points), bool)
        for r in sem["roads"]:
            roads |= shapely.contains(LineString(r["line"]).buffer(r["width_m"] / 2 + 0.5), pts)
        vehicle &= roads.reshape(ny, nx)
        for b in sem.get("buildings", []):
            poly = b.get("polygon") or b.get("footprint")
            if poly and isinstance(poly, list) and len(poly) >= 3:
                blocked = shapely.contains(Polygon(poly).buffer(0.3), pts).reshape(ny, nx)
                person &= ~blocked
                vehicle &= ~blocked
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ground.npy", Z.astype("<f8"))
    for profile, mask in (("person", person), ("vehicle", vehicle)):
        np.save(out / f"walk_{profile}.npy", mask)
        np.save(out / f"z_{profile}.npy", Z.astype("<f8"))
    meta.update(
        schema="awr.life.support.v1",
        method="dtm-guided-mesh-intersection",
        unknown_is_blocked=True,
        ground_sha256=hashlib.sha256(Z.tobytes()).hexdigest(),
        cells_found=int(found.sum()),
        person_cells=int(person.sum()),
        vehicle_cells=int(vehicle.sum()),
    )
    if local_hint:
        meta["method"] = "progressive-local-hint-then-mesh-intersection"
        meta["interpolated_heights_accepted_without_mesh"] = False
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    for k in ("world", "nav", "out"):
        ap.add_argument("--" + k, required=True, type=Path)
    ap.add_argument("--semantics", type=Path)
    ap.add_argument("--local-hint", action="store_true")
    a = ap.parse_args()
    build(a.world, a.nav, a.out, a.semantics, local_hint=a.local_hint)


if __name__ == "__main__":
    main()
