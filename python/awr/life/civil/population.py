"""Population synthesis: households, residents, vehicles, dogs and service roles with appearance, from the semantics layer.

Role and entity categories follow the surveyed systems (SUMO vClasses for road users; MATSim/ActivitySim activity
roles for residents). Everything is sampled from a seeded generator; the output ``population.json`` is deterministic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ appearance palettes (weights are rough real-world frequencies)
TOPS = [
    ((25, 25, 28), 14, "黑色"),
    ((235, 235, 230), 12, "白色"),
    ((128, 128, 130), 12, "灰色"),
    ((30, 40, 90), 10, "藏青色"),
    ((40, 70, 140), 6, "深蓝色"),
    ((190, 30, 35), 6, "红色"),
    ((130, 170, 220), 6, "浅蓝色"),
    ((40, 110, 60), 4, "绿色"),
    ((100, 110, 60), 4, "橄榄色"),
    ((205, 190, 160), 5, "米色"),
    ((110, 75, 45), 4, "棕色"),
    ((230, 200, 40), 2, "黄色"),
    ((235, 120, 30), 2, "橙色"),
    ((230, 140, 170), 3, "粉色"),
    ((110, 50, 130), 2, "紫色"),
    ((120, 30, 50), 3, "酒红色"),
    ((30, 120, 120), 2, "蓝绿色"),
    ((60, 60, 65), 3, "深灰色"),
]
BOTTOMS = [
    ((40, 50, 90), 30, "深蓝色"),
    ((25, 25, 28), 25, "黑色"),
    ((120, 120, 125), 12, "灰色"),
    ((110, 140, 190), 10, "浅蓝色"),
    ((195, 180, 140), 10, "卡其色"),
    ((100, 70, 45), 5, "棕色"),
    ((230, 230, 225), 3, "白色"),
    ((35, 70, 45), 3, "深绿色"),
    ((170, 35, 40), 2, "红色"),
]
CAR_COLOURS = [
    ((235, 235, 232), 35, "白色"),
    ((22, 22, 25), 18, "黑色"),
    ((105, 108, 112), 15, "灰色"),
    ((190, 192, 196), 12, "银色"),
    ((35, 60, 140), 8, "蓝色"),
    ((175, 25, 30), 6, "红色"),
    ((40, 95, 55), 2, "绿色"),
    ((150, 120, 90), 2, "棕色"),
    ((225, 195, 35), 1, "黄色"),
    ((225, 110, 30), 1, "橙色"),
]
VEHICLE_TYPES = {
    # vClass, size l w h (m), label
    "sedan": ("passenger", (4.5, 1.8, 1.45), "轿车"),
    "hatchback": ("passenger", (4.1, 1.75, 1.5), "两厢车"),
    "suv": ("passenger", (4.7, 1.9, 1.7), "SUV"),
    "pickup": ("passenger", (5.3, 1.9, 1.8), "皮卡"),
    "van": ("delivery", (5.2, 2.0, 2.2), "厢式货车"),
    "truck": ("truck", (8.0, 2.5, 3.2), "卡车"),
    "bus": ("bus", (12.0, 2.5, 3.2), "公交车"),
    "minibus": ("bus", (6.0, 2.0, 2.5), "小巴"),
    "taxi": ("taxi", (4.6, 1.8, 1.5), "出租车"),
    "police": ("emergency", (4.8, 1.85, 1.55), "警车"),
    "ambulance": ("emergency", (5.5, 2.2, 2.5), "救护车"),
    "fire": ("emergency", (8.0, 2.5, 3.2), "消防车"),
    "motorcycle": ("motorcycle", (2.1, 0.7, 1.3), "摩托车"),
    "bicycle": ("bicycle", (1.8, 0.6, 1.7), "自行车"),
}
DOG_COLOURS = [
    ((150, 110, 70), 30, "棕色"),
    ((30, 30, 32), 25, "黑色"),
    ((230, 225, 215), 20, "白色"),
    ((200, 170, 120), 15, "浅棕色"),
    ((110, 110, 110), 10, "灰色"),
]


def _pick(rng: np.random.Generator, table: list) -> tuple:
    w = np.array([t[1] for t in table], float)
    i = int(rng.choice(len(table), p=w / w.sum()))
    return table[i]


@dataclass
class Person:
    id: str
    role: str
    age: str  # child, teen, adult, elderly
    sex: str
    height_m: float
    top_rgb: tuple
    bottom_rgb: tuple
    top_name: str
    bottom_name: str
    home: str | None  # building id
    household: str | None
    car: str | None = None
    bike: str | None = None
    dog: str | None = None
    group: str | None = None  # walks together with the group's members
    workplace: str | None = None  # poi id
    speed_mps: float = 1.3
    extras: dict = field(default_factory=dict)

    def description(self) -> str:
        a = {"child": "小孩", "teen": "少年", "adult": "成年人", "elderly": "老人"}[self.age]
        return f"{a}，{self.top_name}上衣、{self.bottom_name}裤子，身高约 {self.height_m:.2f} m"


@dataclass
class Vehicle:
    id: str
    vtype: str
    vclass: str
    size_m: tuple
    body_rgb: tuple
    colour_name: str
    owner: str | None  # person id (driver)
    home_slot: str | None  # parking slot id where it starts
    role: str  # private, delivery, taxi, bus, police, ambulance, fire, through, commuter_in
    extras: dict = field(default_factory=dict)

    def description(self) -> str:
        return f"{self.colour_name}{VEHICLE_TYPES[self.vtype][2]}，长约 {self.size_m[0]:.1f} m"


@dataclass
class Dog:
    id: str
    owner: str
    body_rgb: tuple
    colour_name: str
    size_m: tuple


def _height(rng, age: str, sex: str) -> float:
    if age == "child":
        return float(rng.uniform(1.10, 1.45))
    if age == "teen":
        return float(rng.uniform(1.50, 1.78))
    mu = 1.76 if sex == "m" else 1.63
    if age == "elderly":
        mu -= 0.04
    return float(np.clip(rng.normal(mu, 0.065), 1.45, 1.98))


def _speed(rng, age: str) -> float:
    base = {"child": 1.15, "teen": 1.40, "adult": 1.38, "elderly": 1.05}[age]
    return float(np.clip(rng.normal(base, 0.12), 0.7, 1.9))


def synthesize(sem: dict, cfg: dict, seed: int) -> dict:
    """Returns {'persons': [...], 'vehicles': [...], 'dogs': [...], 'households': [...]} as plain dicts."""
    rng = np.random.default_rng(seed)
    persons: list[Person] = []
    vehicles: list[Vehicle] = []
    dogs: list[Dog] = []
    households = []
    slots_free = {s["id"]: s for s in sem["curb_slots"]} | {
        s["id"]: {**s, "lot": l["id"]} for l in sem["parking_lots"] for s in l["slots"]
    }
    used_slots: set[str] = set()
    pois = sem["pois"]
    work_pois = [
        p
        for p in pois
        if p["kind"]
        in (
            "bank",
            "post_office",
            "supermarket",
            "townhall",
            "courthouse",
            "school",
            "kindergarten",
            "police",
            "community_centre",
            "cafe",
            "kiosk",
            "hospital",
        )
    ]

    def slot_near(xy, r=70.0, allow_illegal_p=0.05):
        cands = [s for s in slots_free.values() if s["id"] not in used_slots and math.dist(s["xy"], xy) < r]

        def is_legal(s):
            return ("lot" in s) or (s.get("legal", True) and s.get("where", "curb") == "curb")

        legal = [s for s in cands if is_legal(s)]
        illegal = [s for s in cands if s not in legal]
        for far_r in (150.0, 260.0):
            if legal:
                break
            legal = [
                s
                for s in slots_free.values()
                if s["id"] not in used_slots and math.dist(s["xy"], xy) < far_r and is_legal(s)
            ]
        pool = illegal if (illegal and rng.random() < allow_illegal_p) else (legal or illegal)
        if not pool:
            return None
        pool.sort(key=lambda s: math.dist(s["xy"], xy))
        s = pool[min(int(rng.exponential(2.0)), len(pool) - 1)]
        used_slots.add(s["id"])
        return s["id"]

    def new_person(role, age, home, hh, sex=None, top=None, bottom=None) -> Person:
        sex = sex or ("m" if rng.random() < 0.49 else "f")
        t = top or _pick(rng, TOPS)
        b = bottom or _pick(rng, BOTTOMS)
        p = Person(
            f"p-{len(persons):04d}",
            role,
            age,
            sex,
            round(_height(rng, age, sex), 2),
            t[0],
            b[0],
            t[2],
            b[2],
            home,
            hh,
            speed_mps=round(_speed(rng, age), 2),
        )
        persons.append(p)
        return p

    def new_vehicle(vtype, role, owner=None, slot=None, colour=None) -> Vehicle:
        vc, size, _ = VEHICLE_TYPES[vtype]
        if colour is None:
            if vtype == "van":
                colour = ((235, 235, 232), 60, "白色") if rng.random() < 0.6 else _pick(rng, CAR_COLOURS)
            elif vtype == "taxi":
                colour = ((240, 205, 40), 1, "黄色")
            elif vtype == "police":
                colour = ((240, 240, 240), 1, "白色")
            elif vtype == "ambulance":
                colour = ((245, 245, 240), 1, "白色")
            elif vtype == "fire":
                colour = ((200, 25, 25), 1, "红色")
            elif vtype == "bus":
                colour = ((225, 180, 40), 1, "黄色") if rng.random() < 0.5 else ((40, 90, 170), 1, "蓝色")
            elif vtype == "bicycle":
                colour = _pick(
                    rng,
                    [
                        ((30, 30, 32), 30, "黑色"),
                        ((175, 25, 30), 20, "红色"),
                        ((35, 60, 140), 20, "蓝色"),
                        ((230, 230, 225), 15, "白色"),
                        ((40, 140, 60), 15, "绿色"),
                    ],
                )
            else:
                colour = _pick(rng, CAR_COLOURS)
        v = Vehicle(f"v-{len(vehicles):04d}", vtype, vc, size, colour[0], colour[2], owner, slot, role)
        vehicles.append(v)
        return v

    # ---------------------------------------------------------------- residents
    res_blds = [b for b in sem["buildings"] if b["households"] > 0]
    active_frac = float(cfg.get("active_household_frac", 0.55))
    for b in res_blds:
        for k in range(b["households"]):
            if rng.random() > active_frac:
                continue
            hid = f"hh-{len(households):04d}"
            size = int(_pick(rng, [(1, 28, ""), (2, 34, ""), (3, 20, ""), (4, 14, ""), (5, 4, "")])[0])
            members = []
            ages = []
            n_adults = 1 if size == 1 else 2
            for i in range(size):
                if i < n_adults:
                    ages.append("elderly" if rng.random() < 0.28 else "adult")
                else:
                    ages.append("child" if rng.random() < 0.6 else "teen")
            for age in ages:
                role = {"child": "student", "teen": "student"}.get(age)
                if role is None:
                    r = rng.random()
                    if age == "elderly":
                        role = "retiree"
                    else:
                        role = (
                            "worker"
                            if r < 0.62
                            else ("homemaker" if r < 0.78 else ("visitor" if r < 0.9 else "jogger"))
                        )
                p = new_person(role, age, b["id"], hid)
                if role == "worker" and work_pois:
                    p.workplace = (
                        work_pois[int(rng.integers(len(work_pois)))]["id"] if rng.random() < 0.45 else None
                    )  # None = works outside the stage
                members.append(p)
            car = None
            if rng.random() < float(cfg.get("car_ownership", 0.55)):
                slot = slot_near(b["door"])
                if slot:
                    drv = next((m for m in members if m.age in ("adult", "elderly")), members[0])
                    vt = _pick(
                        rng,
                        [("sedan", 40, ""), ("hatchback", 30, ""), ("suv", 20, ""), ("pickup", 6, ""), ("van", 4, "")],
                    )[0]
                    car = new_vehicle(vt, "private", drv.id, slot)
                    drv.car = car.id
            if rng.random() < float(cfg.get("bike_ownership", 0.18)):
                rider = next((m for m in members if m.age in ("adult", "teen")), None)
                if rider:
                    bk = new_vehicle("bicycle", "private", rider.id, None)
                    rider.bike = bk.id
            if rng.random() < float(cfg.get("dog_ownership", 0.12)):
                owner = next((m for m in members if m.age in ("adult", "elderly", "teen")), members[0])
                c = _pick(rng, DOG_COLOURS)
                sz = float(rng.uniform(0.5, 1.0))
                d = Dog(
                    f"d-{len(dogs):03d}",
                    owner.id,
                    c[0],
                    c[2],
                    (round(0.9 * sz, 2), round(0.3 * sz, 2), round(0.6 * sz, 2)),
                )
                dogs.append(d)
                owner.dog = d.id
                if owner.role in ("worker", "homemaker", "retiree", "visitor"):
                    owner.role = "dogwalker"
            households.append(
                {"id": hid, "building": b["id"], "members": [m.id for m in members], "car": car.id if car else None}
            )
    # ---------------------------------------------------------------- service fleet and non-residents
    n = cfg.get("fleet", {})
    for k in range(int(n.get("delivery_vans", 3))):
        d = new_person(
            "courier",
            "adult",
            None,
            None,
            top=((60, 60, 160), 1, "蓝色工装") if k % 2 == 0 else ((235, 235, 232), 1, "白色工装"),
        )
        v = new_vehicle("van", "delivery", d.id)
        d.car = v.id
    for k in range(int(n.get("taxis", 2))):
        d = new_person("taxi_driver", "adult", None, None)
        v = new_vehicle("taxi", "taxi", d.id)
        d.car = v.id
    for k in range(int(n.get("buses", 2))):
        d = new_person("bus_driver", "adult", None, None)
        v = new_vehicle("bus" if k == 0 else "minibus", "bus", d.id)
        d.car = v.id
    for k in range(int(n.get("police_cars", 1))):
        d = new_person(
            "police", "adult", None, None, top=((25, 40, 90), 1, "深蓝色制服"), bottom=((25, 30, 60), 1, "深蓝色")
        )
        v = new_vehicle("police", "police", d.id)
        d.car = v.id
        p2 = new_person(
            "police", "adult", None, None, top=((25, 40, 90), 1, "深蓝色制服"), bottom=((25, 30, 60), 1, "深蓝色")
        )
        p2.extras["passenger_of"] = v.id
    if n.get("ambulance", True):
        d = new_person(
            "paramedic", "adult", None, None, top=((230, 230, 60), 1, "黄色反光服"), bottom=((40, 40, 45), 1, "黑色")
        )
        v = new_vehicle("ambulance", "ambulance", d.id)
        d.car = v.id
    for k in range(int(n.get("through_cars", 140))):
        d = new_person("through_driver", "adult", None, None)
        vt = _pick(
            rng,
            [
                ("sedan", 40, ""),
                ("hatchback", 25, ""),
                ("suv", 20, ""),
                ("pickup", 5, ""),
                ("van", 6, ""),
                ("truck", 4, ""),
            ],
        )[0]
        v = new_vehicle(vt, "through", d.id)
        d.car = v.id
    for k in range(int(n.get("commuters_in", 30))):
        d = new_person("commuter_in", "adult", None, None)
        vt = _pick(rng, [("sedan", 45, ""), ("hatchback", 30, ""), ("suv", 25, "")])[0]
        v = new_vehicle(vt, "commuter_in", d.id)
        d.car = v.id
        d.workplace = work_pois[int(rng.integers(len(work_pois)))]["id"] if work_pois else None
    for k in range(int(n.get("motorcycles", 5))):
        d = new_person("rider", "adult", None, None, top=((25, 25, 28), 1, "黑色骑行服"))
        v = new_vehicle("motorcycle", "through", d.id)
        d.car = v.id
    for k in range(int(n.get("construction_workers", 3))):
        new_person(
            "construction_worker",
            "adult",
            None,
            None,
            top=((255, 190, 0), 1, "荧光黄反光背心"),
            bottom=((40, 45, 55), 1, "深灰色"),
        )
    for k in range(int(n.get("vendors", 1))):
        new_person("vendor", "adult" if rng.random() < 0.6 else "elderly", None, None)
    tg = int(n.get("tourist_group", 5))
    if tg:
        gid = "g-tourists"
        for k in range(tg):
            p = new_person("tourist", "adult" if rng.random() < 0.8 else "elderly", None, None)
            p.group = gid
    for k in range(int(n.get("foot_police", 2))):
        p = new_person(
            "police_foot", "adult", None, None, top=((25, 40, 90), 1, "深蓝色制服"), bottom=((25, 30, 60), 1, "深蓝色")
        )
        p.group = "g-police-foot"
    return {
        "schema": "awr.life.civil.population.v1",
        "seed": seed,
        "households": households,
        "persons": [{**p.__dict__, "description": p.description()} for p in persons],
        "vehicles": [{**v.__dict__, "description": v.description()} for v in vehicles],
        "dogs": [d.__dict__ for d in dogs],
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--semantics", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cfg = {}
    if a.config:
        import yaml

        cfg = (yaml.safe_load(a.config.read_text(encoding="utf-8")) or {}).get("population", {})
    sem = json.loads(a.semantics.read_text(encoding="utf-8"))
    pop = synthesize(sem, cfg, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(pop, ensure_ascii=False), encoding="utf-8")
    from collections import Counter

    print(
        f"households {len(pop['households'])}, persons {len(pop['persons'])} {dict(Counter(p['role'] for p in pop['persons']))}"
    )
    print(
        f"vehicles {len(pop['vehicles'])} {dict(Counter(v['vtype'] for v in pop['vehicles']))}, dogs {len(pop['dogs'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
