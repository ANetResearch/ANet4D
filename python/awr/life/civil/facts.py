"""Facts table: what every entity is and what happened, in tape time (seconds from tape start).

Record kinds (one JSON object per line): ``entity``, ``parking``, ``trip``, ``stay``, ``event`` (programmed incidents with
their realised time where available), ``sim`` (realised simulation events: board, alight, enter, exit, bus_stop,
breakdown, crossing_wait, dog_off_leash, drop_object, leave_stage).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _block_of(sem: dict, xy) -> str | None:
    for b in sem["blocks"]:
        P = b["polygon"]
        if P[0][0] <= xy[0] < P[2][0] and P[0][1] <= xy[1] < P[2][1]:
            return b["id"]
    return None


def write_facts(
    sim_dir: Path, sem: dict, plans: dict, pop: dict, actors: list, keep: np.ndarray, out: Path, duration_s: float
) -> int:
    ev = [json.loads(l) for l in (sim_dir / "sim_events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    d = np.load(sim_dir / "tracks.npz")
    state, xy = d["state"], d["xy"]
    vis = (state == 1) | (state == 2)
    persons = {p["id"]: p for p in pop["persons"]}
    vehicles = {v["id"]: v for v in pop["vehicles"]}
    slots = {s["id"]: s for s in sem["curb_slots"]} | {
        s["id"]: {**s, "lot": l["id"]} for l in sem["parking_lots"] for s in l["slots"]
    }
    rows = []
    keep_set = set(int(k) for k in keep)
    idx_of = {a["id"]: a["idx"] for a in actors}
    # entities
    for a in actors:
        if a["idx"] not in keep_set:
            continue
        vmask = vis[:, a["idx"]]
        t_first = float(np.argmax(vmask)) / 10.0
        t_last = float(len(vmask) - 1 - np.argmax(vmask[::-1])) / 10.0
        p0 = xy[int(t_first * 10), a["idx"]]
        rec = {
            "kind": "entity",
            "id": a["id"],
            "class": a["class"],
            "role": a.get("role"),
            "description": a.get("description"),
            "appearance": a["appearance"],
            "size_mm": a["size_mm"],
            "visible_s": [round(t_first, 1), round(t_last, 1)],
            "visible_frac": round(float(vmask.mean()), 3),
            "first_xy": [round(float(p0[0]), 1), round(float(p0[1]), 1)],
            "first_block": _block_of(sem, p0),
        }
        if a["class"] == "person":
            pp = persons.get(a["id"], {})
            rec.update(
                {
                    "age": pp.get("age"),
                    "sex": pp.get("sex"),
                    "home": pp.get("home"),
                    "top": pp.get("top_name"),
                    "bottom": pp.get("bottom_name"),
                    "height_m": pp.get("height_m"),
                    "dog": pp.get("dog"),
                    "car": pp.get("car"),
                }
            )
        elif a["class"] == "vehicle":
            vv = vehicles.get(a["id"], {})
            rec.update(
                {
                    "vtype": vv.get("vtype"),
                    "colour": vv.get("colour_name"),
                    "owner": vv.get("owner"),
                    "home_slot": vv.get("home_slot"),
                }
            )
        rows.append(rec)
    # parking: realised park events paired with the next depart of the same vehicle; plus vehicles parked from the start
    departs = {}
    for e in ev:
        if e["kind"] == "depart":
            departs.setdefault(e["actors"][0], []).append(e["t_s"])
    for vid, slot in plans.get("vehicle_home_slots", {}).items():
        if slot and vid in idx_of and idx_of[vid] in keep_set:
            s = slots[slot]
            t1 = min([t for t in departs.get(vid, []) if t >= 0] or [duration_s])
            rows.append(
                {
                    "kind": "parking",
                    "vehicle": vid,
                    "slot": slot,
                    "xy": s["xy"],
                    "legal": bool(s.get("legal", True)),
                    "reasons": s.get("reasons", []),
                    "where": s.get("where", "lot"),
                    "block": _block_of(sem, s["xy"]),
                    "t0_s": 0.0,
                    "t1_s": round(t1, 1),
                    "from_start": True,
                }
            )
    for e in ev:
        if e["kind"] == "park":
            vid = e["actors"][0]
            if idx_of.get(vid) not in keep_set:
                continue
            later = [t for t in departs.get(vid, []) if t > e["t_s"]]
            rows.append(
                {
                    "kind": "parking",
                    "vehicle": vid,
                    "slot": e.get("slot"),
                    "xy": e["xy"],
                    "legal": e.get("legal", True),
                    "reasons": e.get("reasons", []),
                    "where": e.get("where"),
                    "block": _block_of(sem, e["xy"]),
                    "t0_s": e["t_s"],
                    "t1_s": round(min(later) if later else duration_s, 1),
                    "purpose": e.get("purpose"),
                }
            )
    # stays (enter → exit)
    opened = {}
    for e in ev:
        if e["kind"] == "enter":
            opened[e["actors"][0]] = e
        elif e["kind"] == "exit" and e["actors"][0] in opened:
            o = opened.pop(e["actors"][0])
            rows.append(
                {
                    "kind": "stay",
                    "person": e["actors"][0],
                    "place": o.get("place"),
                    "xy": o["xy"],
                    "t0_s": o["t_s"],
                    "t1_s": e["t_s"],
                    "block": _block_of(sem, o["xy"]),
                }
            )
    for pid, o in opened.items():
        rows.append(
            {
                "kind": "stay",
                "person": pid,
                "place": o.get("place"),
                "xy": o["xy"],
                "t0_s": o["t_s"],
                "t1_s": duration_s,
                "block": _block_of(sem, o["xy"]),
            }
        )
    # trips: depart → park / leave of the same vehicle
    pend = {}
    vehicle_ids = {a["id"] for a in actors if a["class"] == "vehicle"}
    for e in ev:
        if e["kind"] == "depart":
            pend[e["actors"][0]] = e
        elif (
            e["kind"] in ("park", "leave_stage", "bus_stop")
            and e["actors"][0] in pend
            and e["actors"][0] in vehicle_ids
        ):
            s = pend.pop(e["actors"][0])
            rows.append(
                {
                    "kind": "trip",
                    "vehicle": e["actors"][0],
                    "purpose": s.get("purpose"),
                    "t0_s": s["t_s"],
                    "t1_s": e["t_s"],
                    "from_xy": s["xy"],
                    "to_xy": e["xy"],
                    "end": e["kind"],
                    "to_block": _block_of(sem, e["xy"]),
                }
            )
    # programmed events (with realised time if a matching sim event exists)
    for e in plans.get("events", []):
        rows.append({"kind": "event", **e})
    for e in ev:
        if e["kind"] in (
            "board",
            "alight",
            "bus_stop",
            "breakdown",
            "breakdown_end",
            "crossing_wait",
            "dog_off_leash",
            "drop_object",
            "leave_stage",
            "double_park",
            "delivered",
            "bike_park",
        ):
            rows.append({"kind": "sim", **e, "block": _block_of(sem, e["xy"])})
    rows = [
        r
        for r in rows
        if not (("t1_s" in r and r["t1_s"] < 0) or ("t_s" in r and r["t_s"] < -1 and r["kind"] == "sim"))
    ]
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)
