"""The ambient scene as the inputs of a D3 civil event (awr.life.civil.assemble) and of task binding.

`write` turns a scene (awr.life.ambient.scene: scene.npz, crowd-events.json, places.json) and the parked cars
(awr.life.ambient.layout) into
  sim/tracks.npz, sim/actors.json, sim/sim_events.jsonl   the recording as awr.life.civil.assemble reads it
  anchors.overlay.json                                     public regions of the stage
  scenario-overlay.json                                    persons and parked sets of the queries, with bindings
  activity-decisions.jsonl, activity-places.json           activities of the people (selected / completed)
  activity-events.jsonl                                    parking events of the parked cars
Times of events are seconds from the start of the recording (the warm-up is negative).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from awr.life.appearance.identity import PALETTE, TROUSERS, clothing, plate_text
from awr.life.civil.population import CAR_COLOURS, VEHICLE_TYPES

DT = 0.1
MOVING_MPS = 0.025  # tape state 1 above this speed, 2 at or below it (as awr.life.civil.grounded)
STAND_DRIFT_M = 0.5  # a standing run restarts once it drifts this far from where it began


def outfit_ordinal(top, bottom):
    """The clothing ordinal (awr.life.appearance.identity.clothing) of a top and trousers colour name."""
    t = [n for n, _ in PALETTE].index(top)
    b = [n for n, _ in TROUSERS].index(bottom)
    return t + len(PALETTE) * b


def agendas(scene_dir, rename, kinds, warm):
    """Per person the activity phases the life audit reads (awr.life.civil.life_audit): a stay from the start of the
    warm-up at the place the person starts at, then journey_started at each departure, visit_started at each arrival
    and visit_completed when the stay ends. Followers of a group share their leader's phases."""
    report = json.loads((scene_dir / "scene-report.json").read_text())
    events = json.loads((scene_dir / "crowd-events.json").read_text())
    rows = {}
    for e in events:
        rows.setdefault(f"person-{e['person']:03d}", []).append(e)
    people = [k for k in rename if k.startswith("person-")]
    leader = {f: k for k, fs in report["groups"].items() for f in fs}
    out = []
    for pid in people:
        own = rows.get(leader.get(pid, pid), [])
        visit = 0
        start = next(e["place"] for e in own if e["kind"] == "start")
        ev = [dict(kind="visit_started", t_s=-warm, place=start, purpose="visit_" + kinds[start], visit_index=visit)]
        for e in own:
            t = round(e["t"] - warm, 1)
            if e["kind"] == "depart":
                ev += [
                    dict(kind="visit_completed", t_s=t, visit_index=visit),
                    dict(kind="journey_started", t_s=t, place=e["place"], purpose="visit_" + kinds[e["place"]]),
                ]
                visit += 1
            elif e["kind"] == "arrive":
                ev += [
                    dict(kind="arrived", t_s=t, place=e["place"]),
                    dict(
                        kind="visit_started",
                        t_s=t,
                        place=e["place"],
                        purpose="visit_" + kinds[e["place"]],
                        visit_index=visit,
                    ),
                ]
        out.append(dict(actor=rename[pid], departure_s=-warm, events=[dict(e, actor=rename[pid]) for e in ev]))
    return out


def write(scene_dir, out_dir, *, catalog, parked, roles, overlay, regions, seed=7, furniture=()):
    """`parked`: list of dict(id, vtype, xy, yaw, colour (name), role, purpose); `roles`: key -> actor id of the
    persons of the overlay; `overlay`: the scenario overlay (persons, parked_sets); `regions`: public regions
    (anchors with polygons); `furniture`: street objects (awr.life.ambient.furniture)."""
    scene_dir, out = Path(scene_dir), Path(out_dir)
    d = np.load(scene_dir / "scene.npz")
    xy, yaw, cls, kind, ids = d["xy"], d["yaw"], d["cls"], d["kind"], [str(v) for v in d["ids"]]
    warm = float(d["warmup_s"])
    assets = {a["id"]: a for a in catalog["assets"]}
    rng = np.random.default_rng(seed)
    T = len(xy)
    moving = [i for i in range(len(ids)) if cls[i] == "vehicle"]
    people = [i for i in range(len(ids)) if cls[i] == "person"]
    n = len(moving) + len(parked) + len(people) + len(furniture)
    XY = np.zeros((T, n, 2))
    YAW = np.zeros((T, n))
    actors, rename = [], {}
    colour_weights = np.array([c[1] for c in CAR_COLOURS], float)

    def vehicle(vt, rgb, name, role, desc_role):
        j = len(actors)
        spec = assets[vt]
        return dict(
            idx=j,
            id=f"vehicle-{j:03d}",
            **{"class": "vehicle"},
            scripted=True,
            profile="vehicle",
            size_mm=[int(round(v * 1000)) for v in spec["dimensions_m"]],
            appearance=dict(
                asset=vt + "@1",
                asset_sha256=spec["npz_sha256"],
                body_rgb=list(rgb),
                colour_name=name,
                plate_text=plate_text(j),
                plate_synthetic=True,
            ),
            role=role,
            vtype=vt,
            semantic_type=vt,
            description=f"{name}{VEHICLE_TYPES[vt][2]}{desc_role}",
        )

    for i in moving:
        c = CAR_COLOURS[rng.choice(len(CAR_COLOURS), p=colour_weights / colour_weights.sum())]
        a = vehicle(str(kind[i]), c[0], c[2], "through_traffic", "，行驶中")
        rename[ids[i]] = a["id"]
        XY[:, a["idx"]], YAW[:, a["idx"]] = xy[:, i], yaw[:, i]
        actors.append(a)
    parked_cols = []
    for p in parked:
        c = next(c for c in CAR_COLOURS if c[2] == p["colour"])
        a = vehicle(p["vtype"], c[0], c[2], p["role"], "，停放")
        rename[p["id"]] = a["id"]
        XY[:, a["idx"]], YAW[:, a["idx"]] = p["xy"], p["yaw"]
        parked_cols.append(a["idx"])
        actors.append(a)
    reserved = {outfit_ordinal(*o) for o in (e["appearance"].values() for e in overlay["persons"])}
    outfits = [o for o in range(len(PALETTE) * len(TROUSERS)) if o not in reserved]
    humans = sorted((k for k, a in assets.items() if a.get("kind") == "human"), key=lambda k: int(k.split("-")[1]))
    keyed = {aid: key for key, aid in roles.items()}
    entry = {e["key"]: e for e in overlay["persons"]}
    for m, i in enumerate(people):
        j = len(actors)
        h = humans[m % len(humans)]
        spec = assets[h]
        key = keyed.get(ids[i])
        o = outfit_ordinal(*entry[key]["appearance"].values()) if key else outfits[(m * 7) % len(outfits)]
        look = clothing(o)
        a = dict(
            idx=j,
            id=f"person-{m:03d}",
            **{"class": "person"},
            scripted=True,
            profile="person",
            size_mm=[int(round(v * 1000)) for v in spec["dimensions_m"]],
            appearance=dict(asset=h + "@1", asset_sha256=spec["npz_sha256"], **look),
            role="resident",
            semantic_type="pedestrian",
            persona=str(kind[i]),
            description=f"成年人，{look['top_name']}上衣、{look['bottom_name']}裤子",
        )
        rename[ids[i]] = a["id"]
        XY[:, j], YAW[:, j] = xy[:, i], yaw[:, i]
        actors.append(a)
    for f in furniture:
        j = len(actors)
        spec = assets[f["asset"]]
        actors.append(
            dict(
                idx=j,
                id=f["id"],
                **{"class": "object"},
                scripted=True,
                profile=None,
                size_mm=[int(round(v * 1000)) for v in spec["dimensions_m"]],
                appearance=dict(asset=f["asset"] + "@1", asset_sha256=spec["npz_sha256"]),
                role="street_furniture",
                semantic_type=f["asset"],
                description=f["label"],
            )
        )
        XY[:, j], YAW[:, j] = f["xy"], f["yaw"]
    V = np.zeros_like(XY)
    V[1:] = np.diff(XY, axis=0) / DT
    V[0] = V[1]
    sim = out / "sim"
    sim.mkdir(parents=True, exist_ok=False)
    # tape states: 1 moving, 2 standing. The stop checks of the queries read a person's stops from state 2 and the
    # parking membership reads a parked car's from it; parked cars and street furniture stand throughout
    state = np.where(np.hypot(V[..., 0], V[..., 1]) > MOVING_MPS, 1, 2).astype(np.uint8)
    state[:, parked_cols] = 2
    state[:, n - len(furniture) :] = 2
    for j in range(n):
        k = 0
        while k < T:
            if state[k, j] != 2:
                k += 1
                continue
            e = k + int(np.argmax(np.r_[state[k:, j] != 2, True]))
            far = np.flatnonzero(np.hypot(*(XY[k:e, j] - XY[k, j]).T) > STAND_DRIFT_M)
            if len(far):
                e = k + int(far[0])
                state[e, j] = 1
            k = e + 1
    np.savez_compressed(
        sim / "tracks.npz", xy=XY, v=V, yaw=np.degrees(YAW), state=state, container=np.full((T, n), -1, np.int64)
    )
    (sim / "actors.json").write_text(json.dumps(actors, ensure_ascii=False, indent=1) + "\n")
    # moving cars are on a trip from the start of the warm-up to the end; parked cars stand from then on
    park_events = [dict(kind="depart", actors=[rename[ids[i]]], t_s=-warm, purpose="through_traffic") for i in moving]
    park_events += [
        dict(
            kind="park",
            actors=[rename[p["id"]]],
            t_s=-warm,
            xy=[float(v) for v in p["xy"]],
            purpose=p["purpose"],
            legal=True,
        )
        for p in parked
    ]
    (sim / "sim_events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in park_events))
    (out / "activity-events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in park_events))
    places = json.loads((scene_dir / "places.json").read_text())
    kinds = {p["id"]: p["kind"] for p in places}
    (out / "activity-places.json").write_text(
        json.dumps(
            [
                dict(id=p["id"], kind=p["kind"], xy=p["xy"], group=p["area"], parking=[], access_xy=p["xy"])
                for p in places
            ],
            ensure_ascii=False,
        )
        + "\n"
    )
    decisions, current = [], {}
    for e in json.loads((scene_dir / "crowd-events.json").read_text()):
        aid = rename[f"person-{e['person']:03d}"]
        t = round(e["t"] - warm, 1)
        if e["kind"] == "depart":
            if aid in current:
                decisions.append(
                    dict(
                        kind="activity_completed",
                        actor=aid,
                        t_s=t,
                        place=current[aid],
                        purpose="visit_" + kinds[current[aid]],
                        mode="walk",
                    )
                )
            decisions.append(
                dict(
                    kind="activity_selected",
                    actor=aid,
                    t_s=t,
                    place=e["place"],
                    purpose="visit_" + kinds[e["place"]],
                    mode="walk",
                )
            )
            current[aid] = e["place"]
    (out / "activity-decisions.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in decisions))
    (out / "agendas.json").write_text(json.dumps(agendas(scene_dir, rename, kinds, warm), ensure_ascii=False) + "\n")
    (out / "anchors.overlay.json").write_text(json.dumps(dict(anchors=regions), ensure_ascii=False, indent=1) + "\n")
    index = {a["id"]: a["idx"] for a in actors}
    bound = dict(overlay)
    bound["persons"] = [dict(e, outfit=outfit_ordinal(*e["appearance"].values())) for e in overlay["persons"]]
    bound["bindings"] = [
        dict(key=k, actor=rename[aid], home_xy=[float(v) for v in XY[0, index[rename[aid]]]])
        for k, aid in roles.items()
    ]
    bound["parked_sets_initial"] = [
        dict(key=s["key"], vehicles=[dict(id=rename[p["id"]]) for p in parked if s["key"] in p.get("sets", ())])
        for s in overlay.get("parked_sets", [])
    ]
    (out / "scenario-overlay.json").write_text(json.dumps(bound, ensure_ascii=False, indent=1) + "\n")
    return dict(
        actors=len(actors),
        ticks=T,
        moving=len(moving),
        parked=len(parked),
        people=len(people),
        decisions=len(decisions),
    )
