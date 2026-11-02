"""Build an ambient scene from a scenario configuration (configs/ambient/*.json) up to the inputs of publishing.

    python -m awr.life.ambient.build CONFIG --support S --semantics SEM --stage STAGE_DIR --catalog C --out OUT [--export-only]

OUT/scene holds the joint SUMO / JuPedSim recording (awr.life.ambient.scene), OUT/work the event inputs
(awr.life.ambient.export) for awr.life.ambient.publish. The configuration names the stage box, the public regions (a grid of
districts and semantic polygons), the parked cars of the queries (stalls of lot rows, kerb places of a street; types
and colours), the persons of the queries (where they start and which places they may go to) and the scenario overlay.
Street furniture (awr.life.ambient.furniture) is laid out before the people walk (OUT/furniture.json). --export-only
writes OUT/work again from the recording and furniture of an earlier build.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from awr.life.ambient import export, furniture, layout, scene
from awr.life.ambient.crowd import compile_places
from awr.life.ambient.support import Support
from awr.life.ambient.walkway import Walkway


def public_regions(spec, semantics, bbox):
    """Districts (a grid of boxes) and semantic polygons clipped to the stage, as anchors."""
    out = []
    g = spec["districts"]
    for r in range(len(g["y"]) - 1):
        for c in range(len(g["x"]) - 1):
            x0, x1, y0, y1 = g["x"][c], g["x"][c + 1], g["y"][r], g["y"][r + 1]
            out.append(
                dict(
                    id=f"ev_r{r}c{c}",
                    label_zh=g["names"][r][c],
                    kind="zone",
                    category="manual",
                    profile="person",
                    polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    source=dict(
                        kind="authored_region",
                        rationale="District of the town-centre stage: a fixed "
                        "grid over the stage box, not a surveyed administrative district.",
                    ),
                )
            )
    stage = box(*bbox)
    for s in spec["semantic"]:
        item = next(a for a in semantics[s["collection"]] if a["id"] == s["id"])
        poly = Polygon(item["polygon"]).intersection(stage)
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        out.append(
            dict(
                id=s["id"],
                label_zh=s["label_zh"],
                kind="zone",
                category="manual",
                profile="person",
                polygon=[list(map(float, p)) for p in poly.exterior.coords[:-1]],
                source=dict(
                    kind="semantic_polygon",
                    collection=s["collection"],
                    id=s["id"],
                    clipped_to_stage=not Polygon(item["polygon"]).within(stage),
                ),
            )
        )
    return out


def parked_cars(spec, semantics, support, sizes, lanes):
    """The parked cars of the configuration and the public zones around the sets that queries read."""
    cars, zones = [], []
    for entry in spec:
        if entry["kind"] == "row":
            lot = next(r for r in semantics["verified_parking_regions"] if r["id"] == entry["lot"])
            types = {vt for vt, _ in entry["cars"]}
            stalls = {
                s["stall"]: s
                for s in layout.row_stalls(lot["rows"][entry["row"]], support, {t: sizes[t] for t in types}, lanes)
            }
            missing = [k for k in entry["stalls"] if k not in stalls]
            if missing:
                raise ValueError(f"Stalls {missing} of {entry['lot']} row {entry['row']} do not hold the cars")
            poses = [(stalls[k]["xy"], stalls[k]["yaw"]) for k in entry["stalls"]]
            role = "lot_parked"
        else:
            road = next(r for r in semantics["roads"] if r["id"] == entry["road"])
            line = LineString(road["line"])
            big = max({vt for vt, _ in entry["cars"]}, key=lambda t: sizes[t][0] * sizes[t][1])
            places = [
                p
                for p in layout.kerb_places(line, support, sizes[big], big, lanes, side=entry["side"])
                if all(layout.supported(p[1], p[2], sizes[vt], vt, support) for vt, _ in entry["cars"])
            ]
            chosen = None
            for k in range(len(places)):
                run = [places[k]]
                for p in places[k + 1 :]:
                    if p[0] - run[-1][0] >= layout.KERB_SPACING_M:
                        run.append(p)
                    if len(run) == entry["count"]:
                        break
                if (
                    len(run) == entry["count"]
                    and run[-1][0] - run[0][0] <= layout.KERB_SPACING_M * (entry["count"] - 1) + 3
                ):
                    chosen = run
                    break
            if chosen is None:
                raise ValueError(f"No {entry['count']} kerb places on {entry['road']}")
            poses = [(list(map(float, p[1])), p[2]) for p in chosen]
            role = "kerb_parked"
        bodies = []
        for (xy, yaw), (vt, colour) in zip(poses, entry["cars"], strict=False):
            b = layout.body(xy, yaw, sizes[vt])
            if any(b.distance(layout.body(c["xy"], c["yaw"], sizes[c["vtype"]])) < layout.PARKED_GAP_M for c in cars):
                raise ValueError("Parked cars too close")
            cars.append(
                dict(
                    id=f"parked-{len(cars):02d}",
                    vtype=vt,
                    xy=[float(v) for v in xy],
                    yaw=float(yaw),
                    colour=colour,
                    role=role,
                    purpose=entry["purpose"],
                    sets=list(entry["sets"]),
                )
            )
            bodies.append(b)
        if entry.get("zone"):
            zone = unary_union([b.buffer(1.5) for b in bodies]).convex_hull
            zones.append(
                dict(
                    id=entry["zone"],
                    label_zh=entry["label_zh"],
                    kind="zone",
                    category="manual",
                    profile="vehicle",
                    polygon=[list(map(float, p)) for p in zone.exterior.coords[:-1]],
                    source=dict(
                        kind="authored_region",
                        rationale="Public inspection boundary around the "
                        "parked cars of the set, 1.5 m beyond their bodies; not a parking capacity.",
                    ),
                )
            )
    return cars, zones


def build(config, *, support, semantics, stage_dir, catalog, out, export_only=False, log=print):
    cfg = json.loads(Path(config).read_text())
    bbox = tuple(cfg["bbox"])
    sup = Support.load(support, bbox)
    sem = json.loads(Path(semantics).read_text())
    cat = json.loads(Path(catalog).read_text())
    sizes = {a["id"]: a["dimensions_m"] for a in cat["assets"]}
    lanes = layout.lane_lines(str(Path(stage_dir) / "stage-geometry.json"))
    parked, zones = parked_cars(cfg["parked"], sem, sup, sizes, lanes)
    regions = public_regions(cfg["regions"], sem, bbox) + zones
    by_id = {r["id"]: Polygon(r["polygon"]) for r in regions}
    roles = {
        k: dict(start=by_id[r["start"]], places=by_id[r["places"]] if r.get("places") else None, kinds=r.get("kinds"))
        for k, r in cfg["roles"].items()
    }
    obstacles = [layout.body(p["xy"], p["yaw"], sizes[p["vtype"]]) for p in parked]
    out = Path(out)
    if export_only:
        furn = json.loads((out / "furniture.json").read_text())
        report = json.loads((out / "scene" / "scene-report.json").read_text())
        shutil.rmtree(out / "work", ignore_errors=True)
    else:
        # street furniture on the walkway around the parked cars, by the places of a fixed layout seed (the same
        # furniture whatever the seed of the people and cars); then an obstacle of the walkway the people walk
        walkway = Walkway.build(sup, bbox, stage_dir, obstacles)
        furn = furniture.place(
            walkway,
            compile_places(walkway, sem, bbox, seed=furniture.LAYOUT_SEED),
            sem,
            lanes,
            parked,
            sizes,
            bbox,
            support=sup,
        )
        obstacles += furniture.footprints(furn, sizes)
        out.mkdir(parents=True, exist_ok=True)
        (out / "furniture.json").write_text(json.dumps(furn, ensure_ascii=False, indent=1) + "\n")
        report = scene.run(
            stage_dir,
            sup,
            sem,
            bbox,
            out / "scene",
            catalog,
            people=cfg["people"],
            vehicles=cfg["vehicles"],
            warmup_s=cfg["warmup_s"],
            duration_s=cfg["duration_s"],
            seed=cfg["seed"],
            obstacles=obstacles,
            roles=roles,
            log=log,
        )
    (out / "parked.json").write_text(json.dumps(parked, ensure_ascii=False, indent=1) + "\n")
    exported = export.write(
        out / "scene",
        out / "work",
        catalog=cat,
        parked=parked,
        roles=report["roles"],
        overlay=cfg["overlay"],
        regions=regions,
        seed=cfg["seed"],
        furniture=furn,
    )
    life = dict(life_gates=cfg.get("life_gates", {}))
    (out / "work" / "life-config.json").write_text(json.dumps(life, indent=1) + "\n")
    for name in ("scene-report.json",):
        (out / "work" / name).write_text((out / "scene" / name).read_text())
    log(json.dumps(dict(scene=report, export=exported), ensure_ascii=False))
    return report, exported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    for k in ("support", "semantics", "stage", "catalog", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    ap.add_argument("--export-only", action="store_true")
    a = ap.parse_args()
    build(
        a.config,
        support=a.support,
        semantics=a.semantics,
        stage_dir=a.stage,
        catalog=a.catalog,
        out=a.out,
        export_only=a.export_only,
    )


if __name__ == "__main__":
    main()
