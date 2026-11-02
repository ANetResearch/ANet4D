"""Activity scheduling: turns the population into per-actor plans (legs for persons, trips for vehicles) over the
simulation window, following activity-based modelling (home/work/education/shopping/leisure/escort/maintenance/
social/walk/wait) with role templates, plus the programmed incidents of the event catalogue.

Times are simulation seconds from the start of the warm-up; recording starts at ``warmup_s`` so that the town is
already in motion at tape time 0. Output ``plans.json`` is deterministic for a seed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

DEFAULTS = {
    "warmup_s": 600,
    "duration_s": 1500,
    "incidents": {
        "breakdown": 1,
        "double_parking": 2,
        "jaywalk_frac": 0.08,
        "speeding": 2,
        "drop_object": 1,
        "dog_off_leash": 2,
        "gathering": 1,
        "meetings": 4,
        "deliveries": 5,
        "taxi_rides": 3,
        "ambulance": 1,
        "roadworks": 1,
        "vendor_customers": 5,
    },
}


class Scheduler:
    def __init__(self, sem: dict, pop: dict, cfg: dict, seed: int, sumo_dir: Path) -> None:
        import sumolib

        self.sem, self.pop, self.seed = sem, pop, seed
        self.cfg = {**DEFAULTS, **cfg}
        self.cfg["incidents"] = {**DEFAULTS["incidents"], **cfg.get("incidents", {})}
        self.rng = np.random.default_rng(seed + 1000)
        self.W = float(self.cfg["warmup_s"])
        self.D = float(self.cfg["duration_s"])
        self.T = self.W + self.D
        self.bld = {b["id"]: b for b in sem["buildings"]}
        self.poi = {p["id"]: p for p in sem["pois"]}
        self.slots = {s["id"]: s for s in sem["curb_slots"]} | {
            s["id"]: {**s, "lot": l["id"]} for l in sem["parking_lots"] for s in l["slots"]
        }
        self.persons = {p["id"]: p for p in pop["persons"]}
        self.vehicles = {v["id"]: v for v in pop["vehicles"]}
        self.dogs = {d["id"]: d for d in pop["dogs"]}
        self.plans_p: dict[str, list] = {}
        self.plans_v: dict[str, list] = {}
        self.objects: list[dict] = []
        self.machines: list[dict] = []
        self.events: list[dict] = []
        self.slot_busy: dict[str, list] = {}  # slot id → [(t0, t1)]
        for v in pop["vehicles"]:
            if v.get("home_slot"):
                self.slot_busy.setdefault(v["home_slot"], []).append((0.0, self.T + 1))
        self.net = sumolib.net.readNet(str(sumo_dir / "net.net.xml"))
        x0, y0, x1, y1 = sem["stage"]
        self.stage = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        self.entries, self.exits = self._boundary_edges()

    # ------------------------------------------------------------------ helpers
    def _boundary_edges(self):
        ent, ex = [], []
        for e in self.net.getEdges():
            if e.getFunction() or not e.allows("passenger"):
                continue
            sh = e.getShape()
            a, b = Point(sh[0][:2]), Point(sh[-1][:2])
            if not self.stage.contains(a) and self.stage.contains(b):
                ent.append(e.getID())
            elif self.stage.contains(a) and not self.stage.contains(b):
                ex.append(e.getID())
        if not ent:  # fall back to the outermost edges
            far = sorted(self.net.getEdges(), key=lambda e: -math.hypot(*e.getShape()[0][:2]))
            ent = [e.getID() for e in far[:6] if e.allows("passenger")]
            ex = ent
        return ent, ex

    def U(self, a, b):
        return float(self.rng.uniform(a, b))

    def dwell(self, med_s: float, sigma: float = 0.5) -> float:
        return float(np.clip(self.rng.lognormal(math.log(med_s), sigma), 20.0, 1800.0))

    def pick(self, items, weights=None):
        if not items:
            return None
        if weights is None:
            return items[int(self.rng.integers(len(items)))]
        w = np.asarray(weights, float)
        return items[int(self.rng.choice(len(items), p=w / w.sum()))]

    def poi_of_kind(self, kinds, near=None):
        c = [p for p in self.sem["pois"] if p["kind"] in kinds]
        if not c:
            return None
        if near is not None:
            w = [p["weight"] / (1.0 + math.dist(p["xy"], near) / 150.0) for p in c]
        else:
            w = [p["weight"] for p in c]
        return self.pick(c, w)

    def random_building(self, kinds=("house", "apartments")):
        return self.pick([b for b in self.sem["buildings"] if b["kind"] in kinds])

    def free_slot_near(self, xy, t0, t1, r=60.0, illegal_p=0.06, exclude_sidewalk=False):
        cands = []
        for s in self.slots.values():
            if math.dist(s["xy"], xy) > r:
                continue
            if any(a < t1 and b > t0 for a, b in self.slot_busy.get(s["id"], [])):
                continue
            if exclude_sidewalk and s.get("where") == "sidewalk":
                continue
            cands.append(s)
        legal = [s for s in cands if s.get("legal", True)]
        illegal = [s for s in cands if not s.get("legal", True)]
        if not legal:
            # no legal slot nearby: most drivers accept a longer walk before parking illegally
            far = [
                s
                for s in self.slots.values()
                if r < math.dist(s["xy"], xy) <= 2.5 * r
                and s.get("legal", True)
                and not any(a < t1 and b > t0 for a, b in self.slot_busy.get(s["id"], []))
            ]
            if far:
                legal = far
        if not legal and not illegal:
            return None
        pool = illegal if (illegal and self.rng.random() < illegal_p) else (legal or illegal)
        pool.sort(key=lambda s: math.dist(s["xy"], xy))
        s = pool[min(int(self.rng.exponential(1.5)), len(pool) - 1)]
        self.slot_busy.setdefault(s["id"], []).append((t0, t1))
        return s

    def event(self, kind: str, t0: float, actors: list, xy, **extra):
        self.events.append(
            {
                "kind": kind,
                "t0_s": round(t0 - self.W, 1),
                "actors": actors,
                "xy": [round(xy[0], 2), round(xy[1], 2)],
                **extra,
            }
        )

    # ------------------------------------------------------------------ leg builders
    def P(self, pid: str) -> list:
        return self.plans_p.setdefault(pid, [])

    def V(self, vid: str) -> list:
        return self.plans_v.setdefault(vid, [])

    def home_door(self, p):
        return self.bld[p["home"]]["door"] if p.get("home") else None

    def walk_to(self, pid, xy, place=None, **kw):
        self.P(pid).append({"kind": "walk", "to": [round(xy[0], 2), round(xy[1], 2)], "place": place, **kw})

    def enter(self, pid, place, dur):
        self.P(pid).append({"kind": "enter", "place": place, "dur": round(dur, 1)})

    def stand(self, pid, dur, label=None):
        self.P(pid).append({"kind": "stand", "dur": round(dur, 1), "label": label})

    def car_trip(
        self,
        vid,
        driver,
        t_depart,
        from_slot,
        to_xy,
        *,
        park_r=60.0,
        illegal_p=0.06,
        stop_dur=None,
        passengers=(),
        purpose="",
    ):
        """Driver walks to the car (if parked), boards, drives to a slot near to_xy, parks; returns the slot."""
        t_arr_est = t_depart + 60.0 + math.dist(self.slots[from_slot]["xy"] if from_slot else to_xy, to_xy) / 6.0 * 1.6
        slot = self.free_slot_near(
            to_xy,
            t_arr_est - 30,
            (t_arr_est + stop_dur + 60) if stop_dur else self.T + 1,
            r=park_r,
            illegal_p=illegal_p,
        )
        if slot is None:
            return None
        for pid in (driver, *passengers):
            if from_slot:
                self.walk_to(pid, self.slots[from_slot]["xy"], place=from_slot, until_t=t_depart)
                self.P(pid).append({"kind": "board", "vehicle": vid})
            if not (self.P(pid) and self.P(pid)[-1].get("kind") == "ride" and self.P(pid)[-1].get("vehicle") == vid):
                self.P(pid).append({"kind": "ride", "vehicle": vid})
        self.V(vid).append(
            {
                "kind": "trip",
                "depart": {"after_board": [driver, *passengers]} if from_slot else round(t_depart, 1),
                "from": {"slot": from_slot} if from_slot else {"entry": self.pick(self.entries)},
                "to": {"slot": slot["id"]},
                "stop_dur": stop_dur,
                "driver": driver,
                "passengers": list(passengers),
                "purpose": purpose,
            }
        )
        return slot

    def car_exit(self, vid, driver, t_depart, from_slot, passengers=()):
        for pid in (driver, *passengers):
            if from_slot:
                self.walk_to(pid, self.slots[from_slot]["xy"], place=from_slot, until_t=t_depart)
                self.P(pid).append({"kind": "board", "vehicle": vid})
            if not (self.P(pid) and self.P(pid)[-1].get("kind") == "ride" and self.P(pid)[-1].get("vehicle") == vid):
                self.P(pid).append({"kind": "ride", "vehicle": vid})
            self.P(pid).append({"kind": "gone"})
        self.V(vid).append(
            {
                "kind": "trip",
                "depart": {"after_board": [driver, *passengers]} if from_slot else round(t_depart, 1),
                "from": {"slot": from_slot} if from_slot else {"entry": self.pick(self.entries)},
                "to": {"exit": self.pick(self.exits)},
                "stop_dur": None,
                "driver": driver,
                "passengers": list(passengers),
                "purpose": "leave",
            }
        )

    # ------------------------------------------------------------------ role templates
    def schedule_resident(self, p: dict) -> None:
        pid, role = p["id"], p["role"]
        home = self.home_door(p)
        hb = p["home"]
        rng = self.rng
        car = p.get("car")
        car_v = self.vehicles.get(car) if car else None
        start = self.U(0, 0.55 * self.T)
        self.P(pid).append({"kind": "inside", "place": hb, "until": round(start, 1)})
        if role == "worker":
            wp = self.poi.get(p.get("workplace")) if p.get("workplace") else None
            if wp is None:  # works outside the stage
                if car_v and car_v.get("home_slot"):
                    self.car_exit(car, pid, start, car_v["home_slot"])
                else:
                    self.P(pid).append({"kind": "leave"})
                return
            d = math.dist(home, wp["xy"])
            if car_v and car_v.get("home_slot") and d > 220 and rng.random() < 0.7:
                slot = self.car_trip(car, pid, start, car_v["home_slot"], wp["xy"], purpose="work")
                if slot is None:
                    self.walk_to(pid, wp["xy"], place=wp["id"])
                else:
                    self.walk_to(pid, wp["xy"], place=wp["id"])
            elif p.get("bike") and d > 120 and rng.random() < 0.6:
                self.P(pid).append({"kind": "bike", "vehicle": p["bike"], "to": wp["xy"], "place": wp["id"]})
            else:
                self.walk_to(pid, wp["xy"], place=wp["id"])
            if rng.random() < 0.3:
                self.enter(pid, wp["id"], self.dwell(420))
                cafe = self.poi_of_kind(("cafe", "kiosk", "supermarket", "bar"), wp["xy"])
                if cafe:
                    self.walk_to(pid, cafe["xy"], place=cafe["id"])
                    self.enter(pid, cafe["id"], self.dwell(300))
                    self.walk_to(pid, wp["xy"], place=wp["id"])
            self.enter(pid, wp["id"], 9999)
        elif role == "student":
            sch = self.poi_of_kind(("school",) if p["age"] == "teen" else ("school", "kindergarten"), home)
            if sch is None:
                return
            escort = None
            if p["age"] == "child":
                hh = next((h for h in self.pop["households"] if pid in h["members"]), None)
                adults = [
                    m
                    for m in (hh["members"] if hh else [])
                    if self.persons[m]["age"] in ("adult", "elderly")
                    and self.persons[m]["role"] in ("homemaker", "retiree", "worker")
                ]
                if adults and rng.random() < 0.55:
                    escort = adults[0]
            if (
                escort
                and car_v is None
                and self.persons[escort].get("car")
                and rng.random() < 0.35
                and math.dist(home, sch["xy"]) > 250
            ):
                ev = self.vehicles[self.persons[escort]["car"]]
                if ev.get("home_slot") and not self.plans_p.get(escort):
                    self.P(escort).append({"kind": "inside", "place": hb, "until": round(start, 1)})
                    slot = self.car_trip(
                        ev["id"],
                        escort,
                        start,
                        ev["home_slot"],
                        sch["xy"],
                        park_r=45.0,
                        illegal_p=0.3,
                        stop_dur=45.0,
                        passengers=(pid,),
                        purpose="school_dropoff",
                    )
                    if slot:
                        self.walk_to(pid, sch["xy"], place=sch["id"])
                        self.enter(pid, sch["id"], 9999)
                        self.event("school_dropoff", start + 120, [escort, pid, ev["id"]], sch["xy"], place=sch["id"])
                        self.P(escort).append({"kind": "board", "vehicle": ev["id"]})
                        self.P(escort).append({"kind": "ride", "vehicle": ev["id"]})
                        self.walk_to(escort, home, place=hb)
                        self.enter(escort, hb, 9999)
                        self.V(ev["id"]).append(
                            {
                                "kind": "trip",
                                "depart": {"after_board": [escort]},
                                "from": {"slot": slot["id"]},
                                "to": {"slot": ev["home_slot"]},
                                "stop_dur": None,
                                "driver": escort,
                                "passengers": [],
                                "purpose": "home",
                            }
                        )
                        return
            if escort and not self.plans_p.get(escort):
                self.P(escort).append({"kind": "inside", "place": hb, "until": round(start, 1)})
                self.walk_to(escort, sch["xy"], place=sch["id"], group=f"esc-{pid}")
                self.walk_to(pid, sch["xy"], place=sch["id"], group=f"esc-{pid}", follow=escort)
                self.enter(pid, sch["id"], 9999)
                self.stand(escort, self.U(20, 60), "school_gate")
                shop = self.poi_of_kind(("supermarket", "kiosk", "bank", "post_office"), sch["xy"])
                if shop and rng.random() < 0.5:
                    self.walk_to(escort, shop["xy"], place=shop["id"])
                    self.enter(escort, shop["id"], self.dwell(480))
                self.walk_to(escort, home, place=hb)
                self.enter(escort, hb, 9999)
                self.event("school_escort", start, [escort, pid], sch["xy"], place=sch["id"])
                return
            if p["age"] == "teen" and p.get("bike") and rng.random() < 0.5:
                self.P(pid).append({"kind": "bike", "vehicle": p["bike"], "to": sch["xy"], "place": sch["id"]})
            else:
                self.walk_to(
                    pid, sch["xy"], place=sch["id"], jaywalk=rng.random() < self.cfg["incidents"]["jaywalk_frac"]
                )
            self.enter(pid, sch["id"], 9999)
        elif role in ("retiree", "homemaker", "visitor"):
            if rng.random() > {"retiree": 0.6, "homemaker": 0.55, "visitor": 0.8}[role]:
                self.plans_p.pop(pid, None)  # stays home: not emitted
                return
            kinds = {
                "retiree": (
                    "supermarket",
                    "kiosk",
                    "post_office",
                    "bank",
                    "place_of_worship",
                    "park",
                    "square",
                    "cafe",
                ),
                "homemaker": ("supermarket", "kiosk", "post_office", "bank", "kindergarten", "park"),
                "visitor": ("cafe", "square", "park", "bar", "townhall", "courthouse", "bank"),
            }[role]
            first = self.poi_of_kind(kinds, home)
            if first is None:
                return
            if car_v and car_v.get("home_slot") and math.dist(home, first["xy"]) > 300 and rng.random() < 0.5:
                slot = self.car_trip(car, pid, start, car_v["home_slot"], first["xy"], purpose=first["kind"])
                self.walk_to(pid, first["xy"], place=first["id"])
                self.enter(pid, first["id"], self.dwell(480))
                if slot:
                    self.V(car).append(
                        {
                            "kind": "trip",
                            "depart": {"after_board": [pid]},
                            "from": {"slot": slot["id"]},
                            "to": {"slot": car_v["home_slot"]},
                            "stop_dur": None,
                            "driver": pid,
                            "passengers": [],
                            "purpose": "home",
                        }
                    )
                    self.walk_to(pid, slot["xy"], place=slot["id"])
                    self.P(pid).append({"kind": "board", "vehicle": car})
                    self.P(pid).append({"kind": "ride", "vehicle": car})
                self.walk_to(pid, home, place=hb)
                self.enter(pid, hb, 9999)
                return
            spread = {"spread": 22.0} if first["kind"] in ("park", "square") else {}
            biked = bool(p.get("bike")) and math.dist(home, first["xy"]) > 120 and rng.random() < 0.5
            if biked:
                self.P(pid).append(
                    {"kind": "bike", "vehicle": p["bike"], "to": first["xy"], "place": first["id"], **spread}
                )
            else:
                self.walk_to(
                    pid,
                    first["xy"],
                    place=first["id"],
                    jaywalk=rng.random() < self.cfg["incidents"]["jaywalk_frac"],
                    **spread,
                )
            if first["kind"] in ("park", "square"):
                self.stand(pid, self.dwell(400), first["kind"])
            else:
                self.enter(pid, first["id"], self.dwell(420))
            if rng.random() < 0.5:
                second = self.poi_of_kind(kinds, first["xy"])
                if second and second["id"] != first["id"]:
                    spread2 = {"spread": 22.0} if second["kind"] in ("park", "square") else {}
                    if biked:
                        self.P(pid).append(
                            {"kind": "bike", "vehicle": p["bike"], "to": second["xy"], "place": second["id"], **spread2}
                        )
                    else:
                        self.walk_to(pid, second["xy"], place=second["id"], **spread2)
                    if second["kind"] in ("park", "square"):
                        self.stand(pid, self.dwell(300), second["kind"])
                    else:
                        self.enter(pid, second["id"], self.dwell(360))
            if biked:
                self.P(pid).append({"kind": "bike", "vehicle": p["bike"], "to": home, "place": hb})
            else:
                self.walk_to(pid, home, place=hb)
            self.enter(pid, hb, 9999)
        elif role == "jogger":
            pts = self.loop_points(home, self.U(900, 2200))
            self.P(pid).append({"kind": "loop", "points": pts, "speed": round(self.U(2.5, 3.3), 2)})
            self.walk_to(pid, home, place=hb, speed=3.0)
            self.enter(pid, hb, 9999)
        elif role == "dogwalker":
            dog = p.get("dog")
            area = self.poi_of_kind(("park", "square"), home)
            pts = self.loop_points(home, self.U(500, 1100), via=area["xy"] if area else None)
            self.P(pid).append(
                {"kind": "loop", "points": pts, "speed": round(self.U(0.9, 1.3), 2), "dog": dog, "pauses": 3}
            )
            self.walk_to(pid, home, place=hb, dog=dog)
            self.enter(pid, hb, 9999)
            self.event("dog_walk", start, [pid, dog], home, place=hb)
        else:
            self.plans_p.pop(pid, None)

    def loop_points(self, start_xy, length_m: float, via=None) -> list:
        """Waypoints for a roughly closed loop of the given length through walkable points (paths, parks, roads)."""
        cands = [c for e in self.sem["paths"] + self.sem["roads"] for c in e["line"] if self.stage.contains(Point(c))]
        pts, cur, total = [], start_xy, 0.0
        if via is not None:
            pts.append(via)
            total += math.dist(cur, via)
            cur = via
        tries = 0
        while total < length_m and tries < 40:
            tries += 1
            far = [c for c in cands if 120 < math.dist(c, cur) < 320]
            if not far:
                break
            c = self.pick(far)
            pts.append([round(c[0], 2), round(c[1], 2)])
            total += math.dist(cur, c)
            cur = c
        return pts

    # ------------------------------------------------------------------ services and non-residents
    def schedule_services(self) -> None:
        rng = self.rng
        inc = self.cfg["incidents"]
        sq = self.poi_of_kind(("square", "townhall", "courthouse", "bank")) or self.sem["pois"][0]
        for v in self.pop["vehicles"]:
            vid, role, drv = v["id"], v["role"], v["owner"]
            if role == "through":
                t = self.U(0, self.T - 120)
                sf = (
                    1.4
                    if inc["speeding"] > 0
                    and v["vtype"] in ("sedan", "suv", "hatchback", "motorcycle")
                    and rng.random() < 0.12
                    else 1.0
                )
                if sf > 1.0:
                    inc["speeding"] -= 1
                    self.event("speeding", t, [vid], [0, 0], speed_factor=sf)
                self.V(vid).append(
                    {
                        "kind": "trip",
                        "depart": round(t, 1),
                        "from": {"entry": self.pick(self.entries)},
                        "to": {"exit": self.pick(self.exits)},
                        "stop_dur": None,
                        "driver": drv,
                        "passengers": [],
                        "purpose": "through",
                        "speed_factor": sf,
                    }
                )
                self.P(drv).append({"kind": "ride", "vehicle": vid})
                self.P(drv).append({"kind": "gone"})
                if (
                    inc["breakdown"] > 0
                    and v["vtype"] in ("sedan", "hatchback", "van", "truck")
                    and self.U(0, 1) < 0.08
                ):
                    inc["breakdown"] -= 1
                    self.V(vid)[-1]["breakdown"] = {"dur": round(self.U(240, 480), 0)}
                    self.event("breakdown", t + 90, [vid, drv], [0, 0])
            elif role == "commuter_in":
                t = self.U(0, 0.6 * self.T)
                wp = self.poi.get(self.persons[drv].get("workplace")) or sq
                self.P(drv).append({"kind": "ride", "vehicle": vid})
                slot = self.car_trip(vid, drv, t, None, wp["xy"], purpose="work", illegal_p=0.08)
                if slot:
                    self.walk_to(drv, wp["xy"], place=wp["id"])
                    self.enter(drv, wp["id"], 9999)
            elif role == "delivery":
                if inc["deliveries"] <= 0:
                    continue
                t = self.U(0, 0.35 * self.T)
                self.P(drv).append({"kind": "ride", "vehicle": vid})
                n = min(inc["deliveries"], int(rng.integers(3, 6)))
                inc["deliveries"] -= n
                prev_slot = None
                for k in range(n):
                    b = self.random_building(("house", "apartments", "bank", "post_office", "supermarket", "townhall"))
                    dur = self.U(60, 180)
                    if prev_slot is None:
                        slot = self.car_trip(
                            vid,
                            drv,
                            t,
                            None,
                            b["door"],
                            park_r=50.0,
                            illegal_p=0.3,
                            stop_dur=dur + 40,
                            purpose="delivery",
                        )
                    else:
                        slot = self.free_slot_near(b["door"], t, t + dur + 300, r=50.0, illegal_p=0.35)
                        if slot is None:
                            continue
                        self.V(vid).append(
                            {
                                "kind": "trip",
                                "depart": {"after_board": [drv]},
                                "from": {"slot": prev_slot["id"]},
                                "to": {"slot": slot["id"]},
                                "stop_dur": dur + 40,
                                "driver": drv,
                                "passengers": [],
                                "purpose": "delivery",
                            }
                        )
                        self.P(drv).append({"kind": "board", "vehicle": vid})
                        self.P(drv).append({"kind": "ride", "vehicle": vid})
                    if slot is None:
                        continue
                    oid = f"o-{len(self.objects):03d}"
                    self.objects.append(
                        {
                            "id": oid,
                            "kind": "parcel",
                            "size_m": [0.5, 0.4, 0.4],
                            "body_rgb": [170, 130, 90],
                            "carried_by": drv,
                            "from_vehicle": vid,
                        }
                    )
                    self.walk_to(drv, b["door"], place=b["id"], carry=oid)
                    self.enter(drv, b["id"], dur)
                    self.walk_to(drv, slot["xy"], place=slot["id"])
                    self.event(
                        "delivery",
                        t + 200 * (k + 1),
                        [drv, vid, b["id"]],
                        b["door"],
                        place=b["id"],
                        slot=slot["id"],
                        legal=slot.get("legal", True),
                    )
                    prev_slot = slot
                    t += dur + 240
                if prev_slot:
                    self.car_exit(vid, drv, t, prev_slot["id"])
            elif role == "taxi":
                rank = self.free_slot_near(sq["xy"], 0, self.T + 1, r=80.0, illegal_p=0.0)
                if rank is None:
                    continue
                v["home_slot"] = rank["id"]
                self.P(drv).append({"kind": "ride", "vehicle": vid})
                t = self.U(60, 0.4 * self.T)
                cur = rank
                for k in range(int(rng.integers(1, 3))):
                    if inc["taxi_rides"] <= 0:
                        break
                    riders = [
                        q
                        for q in self.pop["persons"]
                        if q["role"] in ("visitor", "retiree") and q["id"] not in self.plans_p and q.get("home")
                    ]
                    if not riders:
                        break
                    inc["taxi_rides"] -= 1
                    r = self.pick(riders)
                    hd = self.home_door(r)
                    pick = self.free_slot_near(hd, t, t + 400, r=50.0, illegal_p=0.15)
                    dest = self.poi_of_kind(
                        ("bank", "post_office", "supermarket", "townhall", "courthouse", "cafe"), hd
                    )
                    if pick is None or dest is None:
                        continue
                    drop = self.free_slot_near(dest["xy"], t + 200, t + 900, r=50.0, illegal_p=0.15)
                    if drop is None:
                        continue
                    self.P(r["id"]).append({"kind": "inside", "place": r["home"], "until": round(t, 1)})
                    self.walk_to(r["id"], pick["xy"], place=pick["id"])
                    self.P(r["id"]).append({"kind": "wait_vehicle", "vehicle": vid})
                    self.P(r["id"]).append({"kind": "board", "vehicle": vid})
                    self.P(r["id"]).append({"kind": "ride", "vehicle": vid})
                    self.walk_to(r["id"], dest["xy"], place=dest["id"])
                    self.enter(r["id"], dest["id"], 9999)
                    self.V(vid).append(
                        {
                            "kind": "trip",
                            "depart": round(t, 1),
                            "from": {"slot": cur["id"]},
                            "to": {"slot": pick["id"]},
                            "stop_dur": {"until_board": [r["id"]]},
                            "driver": drv,
                            "passengers": [],
                            "purpose": "taxi_pickup",
                        }
                    )
                    self.V(vid).append(
                        {
                            "kind": "trip",
                            "depart": {"after_board": [r["id"]]},
                            "from": {"slot": pick["id"]},
                            "to": {"slot": drop["id"]},
                            "stop_dur": 25.0,
                            "driver": drv,
                            "passengers": [r["id"]],
                            "purpose": "taxi_ride",
                        }
                    )
                    self.event("taxi_pickup", t + 150, [vid, r["id"]], pick["xy"], slot=pick["id"])
                    cur = drop
                    t += 700
                if cur is not rank:
                    self.V(vid).append(
                        {
                            "kind": "trip",
                            "depart": round(t, 1),
                            "from": {"slot": cur["id"]},
                            "to": {"slot": rank["id"]},
                            "stop_dur": None,
                            "driver": drv,
                            "passengers": [],
                            "purpose": "return_rank",
                        }
                    )
            elif role == "bus":
                stops = self.sem["bus_stops"]
                if not stops:
                    continue
                t = self.U(30, 300)
                self.P(drv).append({"kind": "ride", "vehicle": vid})
                for loop in range(2):
                    prev = {"entry": self.pick(self.entries)} if loop == 0 else {"stop": stops[-1]["id"]}
                    for st in stops:
                        self.V(vid).append(
                            {
                                "kind": "trip",
                                "depart": round(t, 1) if loop == 0 and st is stops[0] else {"after_stop": True},
                                "from": prev,
                                "to": {"stop": st["id"]},
                                "stop_dur": 30.0,
                                "driver": drv,
                                "passengers": [],
                                "purpose": "bus_line",
                            }
                        )
                        prev = {"stop": st["id"]}
                        t += 240
                self.V(vid).append(
                    {
                        "kind": "trip",
                        "depart": {"after_stop": True},
                        "from": prev,
                        "to": {"exit": self.pick(self.exits)},
                        "stop_dur": None,
                        "driver": drv,
                        "passengers": [],
                        "purpose": "bus_line",
                    }
                )
                # waiting passengers
                for st in stops:
                    for k in range(int(rng.integers(2, 5))):
                        riders = [
                            q
                            for q in self.pop["persons"]
                            if q["role"] in ("worker", "retiree", "visitor", "student")
                            and q["id"] not in self.plans_p
                            and q.get("home")
                            and math.dist(self.home_door(q), st["xy"]) < 350
                        ]
                        if not riders:
                            break
                        r = self.pick(riders)
                        t0 = self.U(0, 0.5 * self.T)
                        self.P(r["id"]).append({"kind": "inside", "place": r["home"], "until": round(t0, 1)})
                        self.walk_to(r["id"], st["xy"], place=st["id"])
                        self.P(r["id"]).append({"kind": "wait_vehicle", "vehicle": vid, "label": "bus_stop"})
                        self.P(r["id"]).append({"kind": "board", "vehicle": vid})
                        self.P(r["id"]).append(
                            {
                                "kind": "ride",
                                "vehicle": vid,
                                "alight_at": self.pick([s["id"] for s in stops if s is not st] or [None]),
                            }
                        )
                        self.P(r["id"]).append({"kind": "gone"})
            elif role == "police":
                self.P(drv).append({"kind": "ride", "vehicle": vid})
                pax = [q["id"] for q in self.pop["persons"] if q.get("extras", {}).get("passenger_of") == vid]
                for q in pax:
                    self.P(q).append({"kind": "ride", "vehicle": vid})
                st = self.poi_of_kind(("police", "townhall")) or sq
                t = self.U(0, 200)
                via = self.pick(self.exits)
                self.V(vid).append(
                    {
                        "kind": "trip",
                        "depart": round(t, 1),
                        "from": {"entry": self.pick(self.entries)},
                        "to": {"exit": via},
                        "stop_dur": None,
                        "driver": drv,
                        "passengers": pax,
                        "purpose": "patrol",
                        "speed_factor": 0.7,
                    }
                )
                slot = self.free_slot_near(st["xy"], t + 500, self.T + 1, r=60.0, illegal_p=0.0)
                if slot:
                    self.V(vid).append(
                        {
                            "kind": "trip",
                            "depart": round(t + 420, 1),
                            "from": {"entry": self.pick(self.entries)},
                            "to": {"slot": slot["id"]},
                            "stop_dur": None,
                            "driver": drv,
                            "passengers": pax,
                            "purpose": "patrol_park",
                            "speed_factor": 0.7,
                        }
                    )
                    for q in (drv, *pax):
                        self.walk_to(q, st["xy"], place=st["id"])
                        self.enter(q, st["id"], 9999)
                else:
                    for q in (drv, *pax):
                        self.P(q).append({"kind": "gone"})
            elif role == "ambulance":
                if inc["ambulance"] <= 0:
                    continue
                t = self.U(0.3 * self.T, 0.7 * self.T)
                b = self.random_building(("house", "apartments"))
                self.P(drv).append({"kind": "ride", "vehicle": vid})
                slot = self.car_trip(
                    vid,
                    drv,
                    t,
                    None,
                    b["door"],
                    park_r=40.0,
                    illegal_p=0.9,
                    stop_dur=self.U(360, 600),
                    purpose="emergency",
                )
                if slot:
                    self.walk_to(drv, b["door"], place=b["id"], speed=1.9)
                    self.enter(drv, b["id"], self.U(240, 420))
                    self.walk_to(drv, slot["xy"], place=slot["id"], speed=1.6)
                    self.car_exit(vid, drv, t + 900, slot["id"])
                    self.event("emergency_run", t, [vid, drv, b["id"]], b["door"], place=b["id"])
        # ---- persons without vehicles
        for p in self.pop["persons"]:
            pid, role = p["id"], p["role"]
            if role == "construction_worker" and inc["roadworks"] > 0:
                site = self._roadworks_site()
                self.P(pid).append(
                    {"kind": "wander", "center": site["xy"], "radius": 10.0, "until": self.T + 1, "speed": 0.8}
                )
            elif role == "vendor":
                stall = [sq["xy"][0] + self.U(-8, 8), sq["xy"][1] + self.U(-8, 8)]
                oid = f"o-{len(self.objects):03d}"
                self.objects.append(
                    {
                        "id": oid,
                        "kind": "stall",
                        "size_m": [2.0, 1.0, 1.2],
                        "body_rgb": [200, 60, 50],
                        "xy": [round(stall[0], 2), round(stall[1], 2)],
                        "static": True,
                    }
                )
                self.P(pid).append(
                    {
                        "kind": "wander",
                        "center": stall,
                        "radius": 2.0,
                        "until": self.T + 1,
                        "speed": 0.5,
                        "stand_frac": 0.8,
                    }
                )
                self.event("vendor", 0, [pid, oid], stall, place=sq["id"])
                for k in range(int(inc["vendor_customers"])):
                    cust = [
                        q
                        for q in self.pop["persons"]
                        if q["role"] in ("retiree", "visitor", "homemaker")
                        and q["id"] not in self.plans_p
                        and q.get("home")
                    ]
                    if not cust:
                        break
                    c = self.pick(cust)
                    t0 = self.U(0, 0.6 * self.T)
                    self.P(c["id"]).append({"kind": "inside", "place": c["home"], "until": round(t0, 1)})
                    self.walk_to(c["id"], [stall[0] + self.U(-2.5, 2.5), stall[1] + self.U(-2.5, 2.5)], place=oid)
                    self.stand(c["id"], self.U(60, 180), "vendor_customer")
                    self.walk_to(c["id"], self.home_door(c), place=c["home"])
                    self.enter(c["id"], c["home"], 9999)
            elif role == "tourist":
                pass
            elif role == "police_foot":
                pass
        if inc["roadworks"] > 0:
            inc["roadworks"] = 0
        # tourists: group walk between landmarks
        tour = [p for p in self.pop["persons"] if p["role"] == "tourist"]
        if tour:
            lm = [
                q
                for q in self.sem["pois"]
                if q["kind"]
                in ("place_of_worship", "townhall", "courthouse", "square", "park", "community_centre", "post_office")
            ]
            lm = lm[:5] if lm else self.sem["pois"][:3]
            ent = self.pick(self.sem["bus_stops"])["xy"] if self.sem["bus_stops"] else lm[0]["xy"]
            lead = tour[0]["id"]
            t0 = self.U(0, 0.3 * self.T)
            for q in tour:
                self.P(q["id"]).append(
                    {"kind": "appear", "xy": [ent[0] + self.U(-3, 3), ent[1] + self.U(-3, 3)], "at": round(t0, 1)}
                )
                for L in lm:
                    self.walk_to(
                        q["id"],
                        L["xy"],
                        place=L["id"],
                        group="g-tourists",
                        follow=None if q["id"] == lead else lead,
                        speed=1.0,
                    )
                    self.stand(q["id"], 200.0, "sightseeing")
                self.walk_to(q["id"], ent, group="g-tourists", follow=None if q["id"] == lead else lead, speed=1.0)
                self.P(q["id"]).append({"kind": "leave"})
            self.event("tourist_group", t0, [q["id"] for q in tour], ent)
        pf = [p for p in self.pop["persons"] if p["role"] == "police_foot"]
        if pf:
            lead = pf[0]["id"]
            route = [sq["xy"]] + [q["xy"] for q in self.sem["pois"][:6]]
            for q in pf:
                self.P(q["id"]).append(
                    {"kind": "appear", "xy": [sq["xy"][0] + self.U(-3, 3), sq["xy"][1] + self.U(-3, 3)], "at": 0.0}
                )
                for k in range(2):
                    for xy in route:
                        self.walk_to(
                            q["id"], xy, group="g-police-foot", follow=None if q["id"] == lead else lead, speed=1.1
                        )
                        self.stand(q["id"], 120.0, "patrol")
                self.P(q["id"]).append({"kind": "leave"})
        # meetings between pairs of residents
        for k in range(int(inc["meetings"])):
            free = [
                q
                for q in self.pop["persons"]
                if q["role"] in ("visitor", "retiree", "worker", "homemaker")
                and q["id"] not in self.plans_p
                and q.get("home")
            ]
            if len(free) < 2:
                break
            a, b = self.pick(free), self.pick(free)
            if a["id"] == b["id"]:
                continue
            place = self.poi_of_kind(("square", "cafe", "park", "bank", "post_office"))
            if place is None:
                break
            t_meet = self.U(0.15 * self.T, 0.7 * self.T)
            mxy = [place["xy"][0] + self.U(-4, 4), place["xy"][1] + self.U(-4, 4)]
            dur = self.U(180, 540)
            for q in (a, b):
                dist = math.dist(self.home_door(q), mxy)
                self.P(q["id"]).append(
                    {"kind": "inside", "place": q["home"], "until": round(t_meet - dist / 1.2 - self.U(0, 60), 1)}
                )
                self.walk_to(q["id"], [mxy[0] + self.U(-1, 1), mxy[1] + self.U(-1, 1)], place=place["id"])
                self.P(q["id"]).append(
                    {
                        "kind": "meet",
                        "with": b["id"] if q is a else a["id"],
                        "dur": round(dur, 1),
                        "until": round(t_meet + dur, 1),
                    }
                )
                if place["kind"] in ("cafe", "bank", "post_office") and q is a:
                    self.enter(q["id"], place["id"], self.dwell(400))
                self.walk_to(q["id"], self.home_door(q), place=q["home"])
                self.enter(q["id"], q["home"], 9999)
            self.event("meeting", t_meet, [a["id"], b["id"]], mxy, place=place["id"], dur_s=round(dur, 0))
        # gathering at the square
        if inc["gathering"] > 0:
            free = [
                q
                for q in self.pop["persons"]
                if q["role"] in ("visitor", "retiree", "worker", "homemaker", "jogger")
                and q["id"] not in self.plans_p
                and q.get("home")
            ]
            n = min(len(free), int(self.rng.integers(12, 22)))
            t_g = self.U(0.35 * self.T, 0.6 * self.T)
            dur = self.U(240, 420)
            chosen = list(self.rng.choice(len(free), size=n, replace=False)) if n else []
            for i in chosen:
                q = free[int(i)]
                gxy = [sq["xy"][0] + self.U(-9, 9), sq["xy"][1] + self.U(-9, 9)]
                dist = math.dist(self.home_door(q), gxy)
                self.P(q["id"]).append(
                    {"kind": "inside", "place": q["home"], "until": round(t_g - dist / 1.2 - self.U(0, 150), 1)}
                )
                self.walk_to(q["id"], gxy, place=sq["id"])
                self.P(q["id"]).append(
                    {"kind": "stand", "dur": 0.0, "until": round(t_g + dur + self.U(-60, 90), 1), "label": "gathering"}
                )
                self.walk_to(q["id"], self.home_door(q), place=q["home"])
                self.enter(q["id"], q["home"], 9999)
            self.event(
                "gathering",
                t_g,
                [free[int(i)]["id"] for i in chosen],
                sq["xy"],
                place=sq["id"],
                dur_s=round(dur, 0),
                count=n,
            )
        # dropped object
        for k in range(int(inc["drop_object"])):
            free = [
                q
                for q in self.pop["persons"]
                if q["role"] in ("visitor", "retiree", "homemaker") and q["id"] not in self.plans_p and q.get("home")
            ]
            if not free:
                break
            q = self.pick(free)
            place = self.poi_of_kind(("square", "park", "bank", "post_office"))
            if place is None:
                break
            t0 = self.U(0.1 * self.T, 0.6 * self.T)
            oid = f"o-{len(self.objects):03d}"
            self.objects.append(
                {"id": oid, "kind": "bag", "size_m": [0.45, 0.3, 0.35], "body_rgb": [40, 40, 45], "carried_by": q["id"]}
            )
            self.P(q["id"]).append({"kind": "inside", "place": q["home"], "until": round(t0, 1)})
            self.walk_to(
                q["id"], [place["xy"][0] + self.U(-5, 5), place["xy"][1] + self.U(-5, 5)], place=place["id"], carry=oid
            )
            self.stand(q["id"], self.U(90, 240), "rest")
            self.P(q["id"]).append({"kind": "drop", "object": oid})
            self.walk_to(q["id"], self.home_door(q), place=q["home"])
            self.enter(q["id"], q["home"], 9999)
            self.event("drop_object", t0 + 300, [q["id"], oid], place["xy"], place=place["id"])

    def _roadworks_site(self) -> dict:
        if getattr(self, "_site", None):
            return self._site
        sw = [s for s in self.sem["curb_slots"] if s["where"] == "sidewalk"]
        s = self.pick(sw) if sw else {"xy": self.sem["pois"][0]["xy"]}
        self._site = {"xy": s["xy"]}
        self.machines.append(
            {
                "id": "m-000",
                "kind": "excavator",
                "size_m": [4.5, 2.2, 2.8],
                "body_rgb": [230, 170, 30],
                "xy": s["xy"],
                "yaw_deg": s.get("yaw_deg", 0),
            }
        )
        for k in range(4):
            a = 2 * math.pi * k / 4
            self.objects.append(
                {
                    "id": f"o-{len(self.objects):03d}",
                    "kind": "cone",
                    "size_m": [0.4, 0.4, 0.7],
                    "body_rgb": [240, 90, 20],
                    "xy": [round(s["xy"][0] + 6 * math.cos(a), 2), round(s["xy"][1] + 6 * math.sin(a), 2)],
                    "static": True,
                }
            )
        self.event("roadworks", 0, ["m-000"], s["xy"])
        return self._site

    # ------------------------------------------------------------------ driver
    def run(self) -> dict:
        self.schedule_services()
        for p in self.pop["persons"]:
            if p["id"] in self.plans_p or not p.get("home"):
                continue
            self.schedule_resident(p)
        # double parking: private cars with a trip get a short lane stop beside parked cars (handled by traffic layer)
        dp = int(self.cfg["incidents"]["double_parking"])
        for vid, trips in self.plans_v.items():
            if dp <= 0:
                break
            v = self.vehicles[vid]
            if (
                v["role"] == "private"
                and trips
                and trips[0].get("purpose") in ("work", "supermarket", "bank", "post_office", "cafe")
            ):
                trips[0]["double_park"] = {"dur": round(self.U(45, 120), 0)}
                self.event("double_parking", 0, [vid], [0, 0])
                dp -= 1
        plans_p = {k: v for k, v in self.plans_p.items() if v}
        return {
            "schema": "awr.life.civil.plans.v1",
            "seed": self.seed,
            "warmup_s": self.W,
            "duration_s": self.D,
            "persons": plans_p,
            "vehicles": self.plans_v,
            "objects": self.objects,
            "machines": self.machines,
            "events": self.events,
            "vehicle_home_slots": {v["id"]: v.get("home_slot") for v in self.pop["vehicles"]},
        }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--semantics", required=True, type=Path)
    ap.add_argument("--population", required=True, type=Path)
    ap.add_argument("--sumo", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cfg = {}
    if a.config:
        import yaml

        cfg = (yaml.safe_load(a.config.read_text(encoding="utf-8")) or {}).get("schedule", {})
    sem = json.loads(a.semantics.read_text(encoding="utf-8"))
    pop = json.loads(a.population.read_text(encoding="utf-8"))
    plans = Scheduler(sem, pop, cfg, a.seed, a.sumo).run()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plans, ensure_ascii=False), encoding="utf-8")
    from collections import Counter

    legs = Counter(l["kind"] for L in plans["persons"].values() for l in L)
    print(f"persons with plans {len(plans['persons'])}, legs {dict(legs)}")
    print(
        f"vehicles with trips {len(plans['vehicles'])}, trips {sum(len(t) for t in plans['vehicles'].values())}, objects {len(plans['objects'])}, machines {len(plans['machines'])}"
    )
    print("events", dict(Counter(e["kind"] for e in plans["events"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
