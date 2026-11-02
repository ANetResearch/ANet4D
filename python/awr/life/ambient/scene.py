"""The ambient scene of a stage: SUMO cars, JuPedSim people and their coupling, stepped together.

`run` writes `out_dir`/scene.npz: per recorded tick (step_s) and actor the position (x, y; a car's body middle), the
heading (rad) and the speed, with each actor's id, class ('vehicle' or 'person') and vehicle type or persona, and per
person whether it waits for a car (`held`, awr.life.ambient.coupling); the recording starts after `warmup_s`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from awr.life.ambient.coupling import Coupling
from awr.life.ambient.crowd import Crowd, compile_places
from awr.life.ambient.traffic import Traffic
from awr.life.ambient.walkway import Walkway


def run(
    stage_dir,
    support,
    semantics,
    bbox,
    out_dir,
    catalog,
    *,
    people=150,
    vehicles=16,
    warmup_s=120.0,
    duration_s=900.0,
    seed=7,
    step_s=0.1,
    obstacles=(),
    roles=None,
    log=print,
):
    """`obstacles`: standing bodies (parked cars) cut out of the walkway; `roles`: key -> dict(start, places, kinds)
    (shapely polygons; places None for all places) for the persons of the queries."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    walkway = Walkway.build(support, bbox, stage_dir, obstacles)
    places = compile_places(walkway, semantics, bbox, seed=seed)
    from shapely import contains_xy

    resolved = {}
    for key, r in (roles or {}).items():
        ok = [
            p
            for p in places
            if (r["places"] is None or contains_xy(r["places"], *p.xy)) and (not r.get("kinds") or p.kind in r["kinds"])
        ]
        start = [p.id for p in ok if contains_xy(r["start"], *p.xy)]
        if not ok or not start:
            raise ValueError(f"No places for role {key}")
        # the persons of the queries stand at their start through the warm-up: the recording begins with them in
        # their start region (a query's search region)
        resolved[key] = dict(places=[p.id for p in ok], start=start, stay_s=warmup_s)
    crowd = Crowd(walkway, places, people, seed=seed, step_s=step_s, roles=resolved)
    traffic = Traffic(stage_dir, out, catalog, vehicles=vehicles, seed=seed, step_s=step_s)
    coupling = Coupling(traffic, crowd)
    steps, first = int(round((warmup_s + duration_s) / step_s)), int(round(warmup_s / step_s))
    nc, npp = len(traffic.ids), len(crowd.people)
    n = nc + npp
    xy = np.zeros((steps - first + 1, n, 2))
    yaw = np.zeros((steps - first + 1, n))
    speed = np.zeros((steps - first + 1, n))
    held = np.zeros((steps - first + 1, npp), bool)
    for k in range(steps + 1):
        if k:
            coupling.update()
            crowd.step()
            traffic.step()
        if k >= first:
            r = k - first
            xy[r, :nc], yaw[r, :nc], speed[r, :nc] = traffic.poses()
            xy[r, nc:], yaw[r, nc:], speed[r, nc:] = crowd.poses()
            held[r] = [p.hold for p in crowd.people]
        if k % 1000 == 0:
            log(
                f"t {k * step_s:.0f} s wall {time.perf_counter() - began:.0f} s walking "
                f"{sum(p.state == 'walk' for p in crowd.people)} holds {coupling.holds} ({coupling.standing_holds} by standing cars) "
                f"yields {coupling.yields}"
            )
    traffic.close()
    ids = list(traffic.ids) + [f"person-{p.i:03d}" for p in crowd.people]
    cls = ["vehicle"] * nc + ["person"] * npp
    kind = list(traffic.types) + [p.persona for p in crowd.people]
    np.savez_compressed(
        out / "scene.npz",
        xy=xy,
        yaw=yaw,
        speed=speed,
        ids=np.array(ids),
        cls=np.array(cls),
        kind=np.array(kind),
        step_s=step_s,
        warmup_s=warmup_s,
        held=held,
    )
    (out / "crowd-events.json").write_text(json.dumps(crowd.events) + "\n")
    (out / "places.json").write_text(
        json.dumps(
            [dict(id=p.id, kind=p.kind, area=p.area, xy=p.xy.tolist(), slots=p.slots.tolist()) for p in places],
            indent=1,
        )
        + "\n"
    )
    groups = {p.i: p.followers for p in crowd.people if p.followers}
    report = dict(
        people=npp,
        vehicles=nc,
        holds=coupling.holds,
        standing_holds=coupling.standing_holds,
        yields=coupling.yields,
        roles={k: f"person-{i:03d}" for k, i in crowd.roles.items()},
        groups={f"person-{k:03d}": [f"person-{f:03d}" for f in v] for k, v in groups.items()},
        section_holds=traffic.sections.holds,
        collisions_reported=int(traffic.collisions),
        wall_s=round(time.perf_counter() - began, 1),
    )
    (out / "scene-report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report
