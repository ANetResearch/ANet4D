"""Measured support compiler with explicit world and semantic inputs.

Road widths and mapped parking polygons only select candidate locations. Neither
they nor a terrain hint create ground. Every accepted height needs a mesh hit;
vehicle routes, bays and actor-sized clearance still require their own admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import shapely
from scipy.ndimage import binary_erosion, map_coordinates
from shapely.geometry import LineString, Polygon, box

from awr.life.common.integrity import sha256_file, validate_content_path


def domains(semantics, bounds):
    stage = box(*bounds)
    roads = [
        LineString(r["line"]).buffer(float(r["width_m"]) / 2)
        for r in semantics.get("roads", [])
        if float(r["width_m"]) > 0
    ]
    streets = [
        LineString(r["line"]).buffer(float(r["width_m"]) / 2 + 12)
        for r in semantics.get("roads", [])
        if float(r["width_m"]) > 0
    ]
    paths = [LineString(r["line"]).buffer(float(r.get("width_m", 2)) / 2) for r in semantics.get("paths", [])]
    areas = [Polygon(p["polygon"]) for p in semantics.get("open_areas", []) if "polygon" in p]
    parking = [
        Polygon(p["polygon"])
        for p in semantics.get("verified_parking_regions", [])
        if p.get("verified") is True and p.get("evidence")
    ]
    blocked = shapely.union_all([Polygon(b["polygon"]).buffer(0.15) for b in semantics.get("buildings", [])])
    walk = shapely.union_all(streets + paths + areas + parking).intersection(stage).difference(blocked)
    drive = shapely.union_all(roads + parking).intersection(stage).difference(blocked)
    return walk, drive


def compile_grid(semantics, bounds, cell, hint_at, ground_at, clear_at, *, batch=80000):
    x0, y0, x1, y1 = map(float, bounds)
    if not np.isfinite([x0, y0, x1, y1, cell]).all() or cell <= 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid support extent or resolution")
    nx, ny = int((x1 - x0) // cell), int((y1 - y0) // cell)
    if nx < 3 or ny < 3 or batch < 1:
        raise ValueError("Support grid or batch too small")
    walk, drive = domains(semantics, bounds)
    shapely.prepare(walk)
    shapely.prepare(drive)
    z = np.full(nx * ny, np.nan, dtype=np.float32)
    road = np.zeros(nx * ny, bool)
    candidates = 0
    hits = 0

    def xy(ids):
        y, x = np.divmod(ids, nx)
        return np.column_stack([x0 + (x + 0.5) * cell, y0 + (y + 0.5) * cell])

    for lo in range(0, z.size, batch):
        ids = np.arange(lo, min(lo + batch, z.size))
        p = xy(ids)
        road[ids] = shapely.intersects_xy(drive, p[:, 0], p[:, 1])
        use = shapely.intersects_xy(walk, p[:, 0], p[:, 1])
        ids = ids[use]
        p = p[use]
        if not len(ids):
            continue
        candidates += len(ids)
        h = np.asarray(hint_at(p))
        finite = np.isfinite(h)
        if finite.any():
            measured, found = ground_at(p[finite], h[finite])
            measured = np.asarray(measured)
            found = np.asarray(found, bool) & np.isfinite(measured)
            z[ids[finite][found]] = measured[found]
            hits += int(found.sum())
        if lo % (batch * 12) == 0:
            print("mesh support cells", lo, "/", z.size, "hits", hits, flush=True)
    z = z.reshape(ny, nx)
    dy, dx = np.gradient(z, cell)
    slope = np.hypot(dx, dy)
    finite = np.isfinite(z) & np.isfinite(slope)
    person = binary_erosion(finite & (slope <= 0.7), iterations=1)
    ids = np.flatnonzero(person)
    clear = np.ones(len(ids), bool)
    for elevation in (0.5, 1.0, 1.5, 2.0):
        for lo in range(0, len(ids), batch):
            jj = ids[lo : lo + batch]
            clear[lo : lo + len(jj)] &= np.asarray(clear_at(xy(jj), z.ravel()[jj] + elevation), bool)
        print("body clearance", elevation, "remaining", int(clear.sum()), flush=True)
    person.ravel()[ids[~clear]] = False
    vehicle = binary_erosion(finite & (slope <= 0.25), iterations=1) & person & road.reshape(ny, nx)
    return (
        z,
        person,
        vehicle,
        dict(
            x0=x0,
            y0=y0,
            nx=nx,
            ny=ny,
            cell_m=cell,
            candidate_cells=candidates,
            mesh_hit_cells=hits,
            person_cells=int(person.sum()),
            vehicle_cells=int(vehicle.sum()),
            rejected_clearance_cells=int((~clear).sum()),
        ),
    )


def build(world, semantics, out, *, cell=0.5, bounds=None):
    if out.exists():
        raise ValueError("Use a fresh immutable support directory")
    meta = json.loads((world / "world.json").read_bytes())
    sem = json.loads(semantics.read_bytes())
    world_bounds = [*meta["bounds"]["min"][:2], *meta["bounds"]["max"][:2]]
    if bounds is None:
        s = sem["bounds"]
        w = world_bounds
        bounds = [max(s[0], w[0]), max(s[1], w[1]), min(s[2], w[2]), min(s[3], w[3])]
    if not box(*world_bounds).covers(box(*bounds)):
        raise ValueError("Support bounds outside world")
    required = ["geometry/mesh/raycast/" + f for f in ("index.json", "xyz.f32", "tri.u32", "q.u8", "src.u32")]
    required += ["geometry/terrain/dsm_2m.json", "geometry/terrain/dsm_2m.f32"]
    records = {r["path"]: r for r in meta["files"]}
    if not set(required) <= records.keys():
        raise ValueError("Required geometry missing from manifest")
    # DSM query also consumes coordinate, DTM, observation-count and zone layers.
    # Validate the whole declared package once offline, not just the mesh arrays.
    if len(records) != len(meta["files"]):
        raise ValueError("Duplicate world file declaration")
    for rel in records:
        validate_content_path(rel)
        row = records[rel]
        p = world / rel
        if p.stat().st_size != row["bytes"] or sha256_file(p) != row["sha256"]:
            raise ValueError(f"World input integrity failed: {rel}")
    from awr.life.civil.terrain_hint import terrain_hint
    from awr.sim.events.ground import build_ground_index
    from awr.world.mesh.query import open_mesh_world_query

    start = time.perf_counter()
    tm = json.loads((world / "geometry/terrain/dsm_2m.json").read_bytes())
    top = np.fromfile(world / "geometry/terrain" / tm["href"], "<f4").reshape(tm["height"], tm["width"])
    hint = terrain_hint(top)
    query = open_mesh_world_query(world, Path(".cache"))
    ground = build_ground_index(query, tuple(bounds), margin_m=2.0)
    preparation_s = time.perf_counter() - start
    print("real mesh prepared", preparation_s, "s", flush=True)

    def hint_at(p):
        return map_coordinates(
            hint,
            [(p[:, 1] - tm["originXY"][1]) / tm["cellM"] - 0.5, (p[:, 0] - tm["originXY"][0]) / tm["cellM"] - 0.5],
            order=1,
            mode="constant",
            cval=np.nan,
        )

    def clear_at(p, z):
        return query.closest_batch(np.column_stack([p, z]), 0.28) >= 0.27

    z, person, vehicle, info = compile_grid(sem, bounds, cell, hint_at, ground.ground_z_batch, clear_at)
    info.update(
        schema="awr.life.support.v1",
        unknown_is_blocked=True,
        method="explicit semantic candidates; DSM hint; actual raycast-mesh intersection; body-height clearance",
        interpolated_heights_accepted_without_mesh=False,
        clearance_heights_m=[0.5, 1.0, 1.5, 2.0],
        clearance_sphere_r_m=0.27,
        world=str(world),
        world_content_version=meta["contentVersion"],
        ground_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
        source_hashes={str(p): sha256_file(p) for p in [world / "world.json", semantics, Path(__file__)]},
        preparation_s=preparation_s,
        total_s=time.perf_counter() - start,
        admission=False,
        limitations=[
            "Semantic widths select candidates; not a lane-width or parking-capacity survey.",
            "Actor footprints, connectivity, driving maneuvers and behavior require subsequent validation.",
            "Source world may be a labelled observed overlay; preserve its provenance limitations.",
        ],
    )
    out.mkdir(parents=True)
    for name, array in [
        ("ground", z),
        ("walk_person", person),
        ("walk_vehicle", vehicle),
        ("z_person", z),
        ("z_vehicle", z),
    ]:
        np.save(out / (name + ".npy"), array)
    (out / "semantics.json").write_bytes(semantics.read_bytes())
    (out / "meta.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("world", "semantics", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--cell", type=float, default=0.5)
    ap.add_argument("--bounds", type=float, nargs=4)
    a = ap.parse_args()
    build(a.world, a.semantics, a.out, cell=a.cell, bounds=a.bounds)
