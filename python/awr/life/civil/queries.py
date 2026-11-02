"""Natural query generation from a civil event's facts table.

Reads ``facts.jsonl``, ``plan.json``, ``anchors.json`` and the simulation tracks, selects situations that are
unambiguous and checkable, and writes task candidates (``awr.life.tasks.v1``) with deterministic checkpoints.
Query families: locate person (area), list illegal parking (area), count parked vehicles at a time (point/area),
track delivery van (multisite), gathering report (point), locate dog walker (area), vehicle breakdown (area).

    python -m awr.life.civil.queries --event <event dir> --sim <sim dir> --semantics … --out tasks/<scenario>.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

AGE_ZH = {"child": "小孩", "teen": "少年", "adult": "成年人", "elderly": "老人"}
AGE_EN = {"child": "child", "teen": "teenager", "adult": "adult", "elderly": "elderly person"}
COLOUR_EN = {
    "黑色": "black",
    "白色": "white",
    "灰色": "grey",
    "藏青色": "navy",
    "深蓝色": "dark blue",
    "红色": "red",
    "浅蓝色": "light blue",
    "绿色": "green",
    "橄榄色": "olive",
    "米色": "beige",
    "棕色": "brown",
    "黄色": "yellow",
    "橙色": "orange",
    "粉色": "pink",
    "紫色": "purple",
    "酒红色": "burgundy",
    "蓝绿色": "teal",
    "深灰色": "dark grey",
    "卡其色": "khaki",
    "深绿色": "dark green",
    "银色": "silver",
    "蓝色": "blue",
    "蓝色工装": "blue work jacket",
    "白色工装": "white work jacket",
    "深蓝色制服": "dark blue uniform",
    "黄色反光服": "yellow hi-vis jacket",
    "荧光黄反光背心": "hi-vis yellow vest",
    "黑色骑行服": "black riding jacket",
}
BLOCK_EN = {
    "block-NW": "north-west block",
    "block-N": "north block",
    "block-NE": "north-east block",
    "block-W": "west block",
    "block-C": "centre block",
    "block-E": "east block",
    "block-SW": "south-west block",
    "block-S": "south block",
    "block-SE": "south-east block",
}
FLEET = [
    {
        "uav_id": f"uav-0{i}",
        "vehicle_ref": "x500",
        "start": {"pos_enu_m": [76.0, -78.0 - 4 * (i - 1), 0.0], "yaw_deg": 90.0},
    }
    for i in range(1, 5)
]


def clock(origin: str, t_s: float) -> str:
    base = dt.datetime.fromisoformat(origin.replace("Z", "+00:00")) + dt.timedelta(hours=3)  # Bulgaria, summer time
    return (base + dt.timedelta(seconds=float(t_s))).strftime("%H:%M")


class QueryGen:
    def __init__(self, event_dir: Path, sim_dir: Path, sem: dict) -> None:
        self.ev = event_dir
        self.sem = sem
        self.plan = json.loads((event_dir / "plan.json").read_text(encoding="utf-8"))
        self.anchors = {
            a["id"]: a for a in json.loads((event_dir / "anchors.json").read_text(encoding="utf-8"))["anchors"]
        }
        self.facts = [
            json.loads(l) for l in (event_dir / "facts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        self.ent = {f["id"]: f for f in self.facts if f["kind"] == "entity"}
        self.origin = self.plan.get("time_origin_utc", "2026-06-15T08:30:00Z")
        self.duration = self.plan["duration_ms"] / 1000.0
        d = np.load(sim_dir / "tracks.npz")
        A = json.loads((sim_dir / "actors.json").read_text(encoding="utf-8"))
        self.aidx = {a["id"]: a["idx"] for a in A}
        self.xy, self.st = d["xy"], d["state"]
        self.blocks = {b["id"]: b for b in sem["blocks"]}
        self.block_label = {b["id"]: b["label"] for b in sem["blocks"]}
        self.tasks: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def block_at(self, xy) -> str | None:
        for bid, b in self.blocks.items():
            P = b["polygon"]
            if P[0][0] <= xy[0] < P[2][0] and P[0][1] <= xy[1] < P[2][1]:
                return bid
        return None

    def block_series(self, actor: str) -> tuple[np.ndarray, list]:
        i = self.aidx[actor]
        vis = (self.st[:, i] == 1) | (self.st[:, i] == 2)
        xs = self.xy[:, i]
        bl = []
        for b in self.blocks.values():
            P = b["polygon"]
            bl.append((xs[:, 0] >= P[0][0]) & (xs[:, 0] < P[2][0]) & (xs[:, 1] >= P[0][1]) & (xs[:, 1] < P[2][1]))
        return vis, bl

    def visible_in_block(self, actor: str) -> list[tuple[str, float, float, float]]:
        """[(block, t0, t1, seconds visible in block within [t0,t1])] for the longest stay per block."""
        vis, bl = self.block_series(actor)
        out = []
        for (bid, _), m in zip(self.blocks.items(), bl, strict=False):
            mm = vis & m
            if mm.sum() < 10:
                continue
            idx = np.flatnonzero(mm)
            out.append((bid, idx[0] / 10.0, idx[-1] / 10.0, mm.sum() / 10.0))
        return out

    def add(self, task: dict) -> None:
        task.setdefault("t_start_s", 0)
        self.tasks.append(task)

    # ------------------------------------------------------------------ families
    def locate_persons(self, n: int = 4) -> None:
        persons = [
            e
            for e in self.ent.values()
            if e["class"] == "person"
            and e.get("role")
            not in (
                "courier",
                "taxi_driver",
                "bus_driver",
                "police",
                "police_foot",
                "paramedic",
                "vendor",
                "construction_worker",
            )
        ]
        cands = []
        for e in persons:
            for bid, t0, t1, secs in self.visible_in_block(e["id"]):
                if secs >= 240 and (t1 - t0) <= 900:
                    cands.append((e, bid, t0, t1, secs))
        # uniqueness of (age, top, bottom) among persons visible in the same block during the window
        rows = []
        for e, bid, t0, t1, secs in cands:
            key = (e["age"], e["top"], e["bottom"])
            clash = 0
            for f in persons:
                if f["id"] == e["id"] or (f["age"], f["top"], f["bottom"]) != key:
                    continue
                for b2, s0, s1, _ in self.visible_in_block(f["id"]):
                    if b2 == bid and s0 <= t1 and s1 >= t0:
                        clash += 1
            if clash == 0:
                rows.append((secs, e, bid, t0, t1))
        rows.sort(key=lambda r: -r[0])
        used_blocks = set()
        k = 0
        for secs, e, bid, t0, t1 in rows:
            if bid in used_blocks:
                continue
            used_blocks.add(bid)
            k += 1
            dog = e.get("dog")
            extra = "，牵着一条狗" if dog else ""
            brief = (
                f"有人报告约 {clock(self.origin, t0)}–{clock(self.origin, t1)} 在{self.block_label[bid]}见到一名{AGE_ZH[e['age']]}，"
                f"{e['top']}上衣、{e['bottom']}裤子{extra}。请找到此人并报告位置，找到后持续报告其位置直到其离开视野。"
            )
            cps = [
                {"id": "P1", "check": "enter_area", "params": {"zone": bid, "agl_m": [30, 120]}},
                {"id": "P2", "check": "declared_in_time"},
                {"id": "L1", "check": "found_within", "params": {"actor": e["id"], "radius_m": 5.0, "t_tol_s": 15}},
                {
                    "id": "L2",
                    "check": "ident_evidence",
                    "params": {"actor": e["id"], "min_unocc_px": 16, "min_visible_ratio": 0.5},
                },
                {"id": "L3", "check": "no_misdeclaration", "params": {"actor": e["id"]}},
                {"id": "L4", "check": "observed_duration", "params": {"actor": e["id"], "min_s": 3.0}},
                {
                    "id": "T1",
                    "check": "track_coverage",
                    "params": {
                        "actor": e["id"],
                        "radius_m": 6.0,
                        "min_fraction": 0.5,
                        "from_s": round(t0 + 90, 0),
                        "to_s": round(min(t1, t0 + 420), 0),
                    },
                },
            ]
            brief_en = (
                f"A witness reports seeing a{'n' if AGE_EN[e['age']][0] in 'aeiou' else ''} {AGE_EN[e['age']]} in a {COLOUR_EN.get(e['top'], e['top'])} top and "
                f"{COLOUR_EN.get(e['bottom'], e['bottom'])} trousers{' walking a dog' if dog else ''} in the {BLOCK_EN[bid]} around {clock(self.origin, t0)}–{clock(self.origin, t1)}. "
                f"Find this person, report the position, and keep reporting it until they leave view."
            )
            self.add(
                {
                    "task_id": f"C{len(self.tasks) + 1:02d}-locate-person",
                    "type": "locate",
                    "scope": "area",
                    "t_max_s": int(min(self.duration, t1 + 120)),
                    "brief_zh": brief,
                    "brief_en": brief_en,
                    "public_anchors": [bid],
                    "targets": [e["id"]],
                    "report": {"type": "found", "fields": ["position_enu_m", "t_ns", "uav_id"]},
                    "features": {"window_s": [round(t0), round(t1)], "visible_s": round(secs)},
                    "checkpoints": cps,
                }
            )
            if k >= n:
                break

    def illegal_parking(self, n: int = 2) -> None:
        parks = [f for f in self.facts if f["kind"] == "parking" and not f["legal"] and f["t1_s"] - f["t0_s"] >= 120]
        by_block = defaultdict(list)
        for f in parks:
            if f.get("block"):
                by_block[f["block"]].append(f)
        rows = sorted([kv for kv in by_block.items() if 2 <= len(kv[1]) <= 6], key=lambda kv: -len(kv[1]))
        for bid, lst in rows[:n]:
            if len(lst) < 2 or len(lst) > 6:
                continue
            t0 = max(f["t0_s"] for f in lst)
            t1 = min(f["t1_s"] for f in lst)
            if t1 - t0 < 120:
                t0, t1 = float(np.median([f["t0_s"] for f in lst])), float(np.median([f["t1_s"] for f in lst]))
            t0 = max(0.0, t0)
            brief = (
                f"请检查{self.block_label[bid]}在 {clock(self.origin, t0)}–{clock(self.origin, t1)} 期间的违章停车：违停指停在人行道上、路口 10 m 内、人行横道上、"
                f"公交站旁或学校、消防站等设施门前。对每辆违停车辆报告位置、颜色与车型。"
            )
            cps = [
                {"id": "P1", "check": "enter_area", "params": {"zone": bid, "agl_m": [30, 120]}},
                {"id": "P2", "check": "declared_in_time"},
                {
                    "id": "S1",
                    "check": "set_match",
                    "params": {
                        "actors": [f["vehicle"] for f in lst],
                        "radius_m": 6.0,
                        "min_recall": 0.75,
                        "max_false": 1,
                        "report_type": "found",
                    },
                },
            ]
            brief_en = (
                f"Check the {BLOCK_EN[bid]} for illegal parking between {clock(self.origin, t0)} and {clock(self.origin, t1)}: vehicles parked on the pavement, within 10 m of a junction, "
                f"on a crossing, next to a bus stop or in front of a school, fire station or similar facility. Report the position, colour and type of every illegally parked vehicle."
            )
            self.add(
                {
                    "task_id": f"C{len(self.tasks) + 1:02d}-illegal-parking",
                    "type": "identify",
                    "scope": "area",
                    "t_max_s": int(min(self.duration, t1 + 60)),
                    "brief_zh": brief,
                    "brief_en": brief_en,
                    "public_anchors": [bid],
                    "targets": [f["vehicle"] for f in lst],
                    "report": {"type": "found", "fields": ["position_enu_m", "t_ns", "uav_id", "label"]},
                    "features": {"n_truth": len(lst), "window_s": [round(t0), round(t1)]},
                    "checkpoints": cps,
                }
            )

    def parked_count(self, n: int = 2) -> None:
        lots = [l for l in self.sem["parking_lots"] if l["id"] in self.anchors]
        rows = []
        for l in lots:
            P = np.asarray(l["polygon"])
            inside = []
            vids = [e["id"] for e in self.ent.values() if e["class"] == "vehicle"]
            idx = [self.aidx[v] for v in vids]
            X = self.xy[:, idx]
            S = self.st[:, idx]
            from awr.life.common.truth import point_in_polygon

            cnt = np.zeros(len(X), int)
            for j in range(len(idx)):
                m = point_in_polygon(P, X[:, j, 0], X[:, j, 1]) & (S[:, j] == 2)
                cnt += m
            k = int(np.argmax(cnt))
            rows.append((int(cnt[k]), l["id"], k / 10.0, cnt))
        rows.sort(key=lambda r: -r[0])
        for c, lid, t, cnt in rows[:n]:
            if c < 3:
                continue
            # choose a time where the count is stable for ≥ 60 s
            stable = None
            for s in range(1200, len(cnt) - 600, 100):
                if (cnt[s : s + 600] == cnt[s]).all() and cnt[s] >= 3:
                    stable = (s / 10.0, int(cnt[s]))
                    break
            if stable is None:
                continue
            ts, want = stable
            brief = f"请于 {clock(self.origin, ts)} 前后统计停车场 {lid}（公开锚点）内停放的车辆数并报告。"
            cps = [
                {
                    "id": "P1",
                    "check": "zone_observed",
                    "params": {"zone": lid, "during_s": [round(ts), round(ts + 60)], "min_coverage": 0.5},
                },
                {"id": "P2", "check": "declared_in_time"},
                {
                    "id": "N1",
                    "check": "count_at_time",
                    "params": {
                        "zone": lid,
                        "t_s": round(ts + 30),
                        "classes": ["vehicle"],
                        "states": [2],
                        "tolerance": 0,
                    },
                },
            ]
            brief_en = f"Around {clock(self.origin, ts)}, count the vehicles parked in car park {lid} (public anchor) and report the number."
            self.add(
                {
                    "task_id": f"C{len(self.tasks) + 1:02d}-parked-count",
                    "type": "count",
                    "scope": "area",
                    "t_max_s": int(min(self.duration, ts + 300)),
                    "brief_zh": brief,
                    "brief_en": brief_en,
                    "public_anchors": [lid],
                    "targets": [],
                    "report": {"type": "count", "fields": ["zone", "count", "t_ns"]},
                    "features": {"truth_count": want, "t_s": ts},
                    "checkpoints": cps,
                }
            )

    def track_delivery(self) -> None:
        vans = [e for e in self.ent.values() if e["class"] == "vehicle" and e.get("role") == "delivery"]
        for e in vans[:1]:
            parks = [
                f for f in self.facts if f["kind"] == "parking" and f["vehicle"] == e["id"] and not f.get("from_start")
            ]
            if len(parks) < 2:
                continue
            t0, t1 = e["visible_s"]
            brief = (
                f"一辆{e['colour']}厢式货车约在 {clock(self.origin, t0)} 进入镇区送货。请找到该车并持续跟踪，每次它停靠时报告停靠位置与时刻，"
                f"并在其驶离镇区时报告。"
            )
            cps = [
                {"id": "P2", "check": "declared_in_time"},
                {
                    "id": "T1",
                    "check": "acquired_within",
                    "params": {"actor": e["id"], "radius_m": 10.0, "before_s": round(t0 + 180)},
                },
                {
                    "id": "T2",
                    "check": "track_coverage",
                    "params": {
                        "actor": e["id"],
                        "radius_m": 10.0,
                        "min_fraction": 0.6,
                        "from_s": round(t0 + 120),
                        "to_s": round(t1),
                    },
                },
                {
                    "id": "T3",
                    "check": "stops_reported",
                    "params": {"actor": e["id"], "radius_m": 15.0, "t_tol_s": 60, "min_fraction": 0.6},
                },
            ]
            brief_en = (
                f"A {COLOUR_EN.get(e['colour'], e['colour'])} van enters the town around {clock(self.origin, t0)} to make deliveries. Find it and keep tracking it; "
                f"each time it stops, report the stop position and time, and report when it leaves the town."
            )
            self.add(
                {
                    "task_id": f"C{len(self.tasks) + 1:02d}-track-van",
                    "type": "track",
                    "scope": "multisite",
                    "t_max_s": int(min(self.duration, t1 + 60)),
                    "brief_zh": brief,
                    "brief_en": brief_en,
                    "public_anchors": [b for b in self.blocks],
                    "targets": [e["id"]],
                    "report": {"type": "track", "fields": ["position_enu_m", "t_ns", "state"]},
                    "features": {"n_stops": len(parks), "window_s": [round(t0), round(t1)]},
                    "checkpoints": cps,
                }
            )

    def gathering(self) -> None:
        for f in self.facts:
            if f["kind"] == "event" and f.get("kind_") is None and f.get("count") and f.get("dur_s"):
                pass
        evs = [f for f in self.facts if f["kind"] == "event" and f.get("count") and f.get("dur_s")]
        for g in evs[:1]:
            place = g.get("place")
            t0 = float(g["t0_s"])
            brief = f"请监视{self.anchors.get(place, {}).get('label', '广场')}一带；若出现 10 人以上的人群聚集，报告聚集开始的时刻、位置与大致人数，并在人群散去时报告。"
            cps = [
                {
                    "id": "P1",
                    "check": "zone_observed",
                    "params": {"zone": place, "during_s": [round(t0), round(t0 + g["dur_s"])], "min_coverage": 0.4},
                },
                {"id": "P2", "check": "declared_in_time"},
                {
                    "id": "E1",
                    "check": "gathering_reported",
                    "params": {
                        "zone": place,
                        "t0_s": round(t0),
                        "t1_s": round(t0 + g["dur_s"]),
                        "count": g["count"],
                        "t_tol_s": 120,
                        "pos_tol_m": 40,
                        "count_tol": 0.4,
                    },
                },
            ]
            brief_en = "Watch the square area; if a crowd of ten or more people gathers, report the start time, position and approximate number, and report when the crowd disperses."
            self.add(
                {
                    "task_id": f"C{len(self.tasks) + 1:02d}-gathering",
                    "type": "event",
                    "scope": "point",
                    "t_max_s": int(min(self.duration, t0 + g["dur_s"] + 240)),
                    "brief_zh": brief,
                    "brief_en": brief_en,
                    "public_anchors": [place] if place else [],
                    "targets": [],
                    "report": {"type": "event", "fields": ["event_type", "position_enu_m", "t_ns", "count"]},
                    "features": {"t0_s": t0, "dur_s": g["dur_s"], "count": g["count"]},
                    "checkpoints": cps,
                }
            )

    def breakdown(self) -> None:
        bd = [
            f
            for f in self.facts
            if f["kind"] == "sim" and f.get("kind") == "sim" and f.get("actors") and "breakdown" in str(f)
        ]
        bd = [
            f
            for f in self.facts
            if f["kind"] == "sim" and f.get("dur_s") and len(f["actors"]) == 2 and f["actors"][0].startswith("v-")
        ]
        for f in bd[:1]:
            vid, drv = f["actors"]
            if vid not in self.ent:
                continue
            e = self.ent[vid]
            t0 = float(f["t_s"])
            brief = (
                f"约 {clock(self.origin, t0)} 有人报告一辆车在镇区道路上抛锚停在车道中。请找到这辆车并报告位置、颜色与车型，"
                f"并报告驾驶员是否已下车站在车旁。"
            )
            cps = [
                {"id": "P2", "check": "declared_in_time"},
                {"id": "L1", "check": "found_within", "params": {"actor": vid, "radius_m": 8.0, "t_tol_s": 30}},
                {
                    "id": "L2",
                    "check": "ident_evidence",
                    "params": {"actor": vid, "min_unocc_px": 16, "min_visible_ratio": 0.5},
                },
                {"id": "L3", "check": "found_within", "params": {"actor": drv, "radius_m": 6.0, "t_tol_s": 30}},
            ]
            brief_en = (
                f"Around {clock(self.origin, t0)} a vehicle was reported broken down in a traffic lane in town. Find it, report its position, colour and type, "
                f"and report whether the driver has got out and is standing next to it."
            )
            self.add(
                {
                    "task_id": f"C{len(self.tasks) + 1:02d}-breakdown",
                    "type": "locate",
                    "scope": "area",
                    "t_max_s": int(min(self.duration, t0 + f["dur_s"] + 60)),
                    "brief_zh": brief,
                    "brief_en": brief_en,
                    "public_anchors": [b for b in self.blocks],
                    "targets": [vid, drv],
                    "report": {"type": "found", "fields": ["position_enu_m", "t_ns", "uav_id", "label"]},
                    "features": {"t0_s": t0, "dur_s": f["dur_s"]},
                    "checkpoints": cps,
                }
            )

    def run(self) -> dict:
        self.locate_persons()
        self.illegal_parking()
        self.parked_count()
        self.track_delivery()
        self.gathering()
        self.breakdown()
        return {
            "schema": "awr.life.tasks.v1",
            "scenario": self.plan["event_id"],
            "world": self.plan["world"]["id"],
            "generated_by": "awr.life.civil.queries",
            "fleet": FLEET,
            "altitude_envelope_agl_m": [30, 120],
            "safety": {"min_sep_person_m": 5.0, "min_sep_vehicle_m": 3.0, "no_fly": []},
            "tasks": self.tasks,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, type=Path)
    ap.add_argument("--sim", required=True, type=Path)
    ap.add_argument("--semantics", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    sem = json.loads(a.semantics.read_text(encoding="utf-8"))
    doc = json.loads(
        json.dumps(QueryGen(a.event, a.sim, sem).run(), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        "# 由 awr.life.civil.queries 从事实表生成的候选任务；人工审阅后冻结。\n"
        + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )
    for t in doc["tasks"]:
        print(t["task_id"], "|", t["brief_zh"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
