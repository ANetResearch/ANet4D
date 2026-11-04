"""Device descriptions and the fidelity ladder of the embodied device engine (``awr.device_ladder.v1``).

A device is described by its vehicle profile (``vehicles/<dir>/params.yaml``, ``awr.vehicle.v1``: mass, inertia,
motors, aero, battery, limits, sensors) and the sensor models it mounts (``vehicles/sensor_models/*.yaml``, gimbal
limits and rates). This module does not add a second description format; it reads those files and states, per
device and per tier, what the engine can actually do with them today:

================  ==============================================================================================
tier              meaning
================  ==============================================================================================
``l0``            replay / kinematic: poses come from a recording or a provider, no dynamics (``Fidelity.L0``)
``l1``            PX4-lite cascade at 125 Hz: position → attitude → motor → aero → integrate (``Fidelity.L1``)
``l2``            realistic: gz-sim lockstep with the PX4 compose (``awr.phys.gz.GzSimBackend``), rate loop,
                  allocation, rotor lag and aerodynamics inside gz; today only the ``x500_d4d`` model
``l3``            software in the loop: the real PX4 / ArduPilot flight stack (``px4_sih`` is the only hook)
``fw``            fixed-wing / VTOL fast tier: kinematic flight with energy (``Fidelity.FW``, stage ``fw_kin``)
================  ==============================================================================================

and for gimbals the servo tiers ``l1`` (rate-limited) and ``l2`` (second-order servo), see ``awr.sim.sensors.servo``.

``ladder()`` is what the device page shows; a tier is *available* only when the profile declares it and the code
implements it. Land and sea devices are listed as planned rows until step 5 adds their profiles.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

SCHEMA = "awr.device_ladder.v1"

TIERS = {
    "l0": {"label_zh": "回放 / 运动学", "module": "awr.sim.fleet.stages.kinematic", "flag": "L0"},
    "l1": {"label_zh": "PX4-lite 级联（125 Hz）", "module": "awr.sim.fleet.stages.l1", "flag": "L1"},
    "l2": {
        "label_zh": "Realistic：gz-sim 锁步（速率环、控制分配、转子滞后、气动在 gz 内）",
        "module": "awr.phys.gz.backend",
        "flag": "L2",
    },
    "l3": {"label_zh": "SITL（PX4 / ArduPilot）", "module": None, "flag": None},
    "fw": {"label_zh": "固定翼 / VTOL 快速档", "module": "awr.sim.fixedwing.runtime", "flag": "FW"},
}
# names the vehicle profile schema uses for its `fidelity` field, mapped to the ladder tiers
DECLARED = {"l0": "l0", "l1": "l1", "sih": "l2", "fw_kin": "fw"}
GIMBAL_TIERS = {"l1": "限速逼近", "l2": "二阶伺服（加速度上限、速度环时间常数）"}
# vehicle models the gz Realistic tier has today (awr.phys.tier_compare: "gz 只有 x500_d4d")
GZ_MODELS = {"x500": "x500_d4d", "x500_sih": "x500_d4d"}

PLANNED = [
    {
        "id": "ugv_rover",
        "domain": "land",
        "display_name_zh": "轮式地面车（JPL 开源漫游车级）",
        "category": "ugv",
        "size_class": "small",
        "note": "第 5 步：L1 运动学 + gz L2；参考 JPL Open Source Rover",
    },
    {
        "id": "ugv_f1tenth",
        "domain": "land",
        "display_name_zh": "1/10 自动驾驶车（F1TENTH 级）",
        "category": "ugv",
        "size_class": "small",
        "note": "第 5 步：L1 自行车模型 + gz L2；参考 F1TENTH",
    },
    {
        "id": "ugv_podcar",
        "domain": "land",
        "display_name_zh": "小型乘用车（OpenPodcar 级）",
        "category": "ugv",
        "size_class": "medium",
        "note": "第 5 步：L1 自行车模型；参考 OpenPodcar 2",
    },
    {
        "id": "usv_small",
        "domain": "sea",
        "display_name_zh": "小型无人艇（BlueBoat 级）",
        "category": "usv",
        "size_class": "small",
        "note": "第 5 步：L1 三自由度艇模型 + gz L2；参考 BlueBoat",
    },
    {
        "id": "usv_medium",
        "domain": "sea",
        "display_name_zh": "中型无人艇（NAC-USV 级）",
        "category": "usv",
        "size_class": "medium",
        "note": "第 5 步：L1 三自由度艇模型；参考 NAC-USV",
    },
]


def _val(d):
    return d.get("value") if isinstance(d, dict) and "value" in d else d


def implemented() -> dict[str, dict]:
    """Which tiers the code implements, by looking for the modules that carry them (not by trusting a table)."""
    out = {}
    for tier, t in TIERS.items():
        mod = t["module"]
        present = bool(mod) and importlib.util.find_spec(mod) is not None
        if tier == "l2":
            sih = importlib.util.find_spec("awr.sim.backends.px4_sih.adapter") is not None
            out[tier] = {
                "implemented": present,
                "status": "可用（仅 x500，d3-gz 容器）" if present else "无",
                "detail": "awr.phys.gz.GzSimBackend（gz-sim 锁步，PX4 compose）；gz 模型只有 x500_d4d；"
                + ("px4_sih 后端是骨架（open 返回 213）" if sih else "无 px4_sih")
                + "；其他机型第 5 步",
            }
        elif tier == "l3":
            out[tier] = {"implemented": False, "status": "无", "detail": "无 SITL 代码；第 5 步经 PX4 / ArduPilot 接入"}
        else:
            out[tier] = {"implemented": present, "status": "可用" if present else "无", "detail": mod or ""}
    return out


def _gimbal_of(sensor_doc: dict) -> dict | None:
    g = sensor_doc.get("gimbal")
    if not isinstance(g, dict):
        return None
    servo = g.get("servo") or {}
    return {
        "pitch_deg": [g.get("pitch_min_deg"), g.get("pitch_max_deg")],
        "yaw_deg": [g.get("yaw_min_deg"), g.get("yaw_max_deg")],
        "rate_max_deg_s": g.get("rate_max_deg_s", 90.0),
        "servo_tier": servo.get("tier", "l1"),
        "acc_max_deg_s2": servo.get("acc_max_deg_s2"),
        "tau_s": servo.get("tau_s"),
        "integrated": bool(g.get("integrated", False)),
    }


def _sensor_doc(vdir: Path, vehicle_dir: Path, rel: str, yaml_load) -> dict:
    p = vehicle_dir / rel
    doc = yaml_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    ref = doc.get("spec_ref") if isinstance(doc, dict) else None
    if ref:  # a mount file referencing a model in sensor_models/
        q = vdir / "sensor_models" / (str(ref).split("@")[0] + ".yaml")
        if q.exists():
            base = yaml_load(q.read_text(encoding="utf-8"))
            base.update({k: v for k, v in doc.items() if k != "spec_ref"})
            doc = base
    return doc if isinstance(doc, dict) else {}


def ladder(vdir: Path | None = None) -> dict:
    from awr.sim.fleet.profiles import load_vehicle_docs, read_catalog, vehicles_dir, yaml_load

    vdir = Path(vdir) if vdir else vehicles_dir()
    cat = read_catalog(vdir) or {"models": []}
    cat_by_id = {m["id"]: m for m in cat["models"] if isinstance(m, dict)}
    impl = implemented()
    docs = load_vehicle_docs(vdir)
    devices = []
    for pid, (doc, path) in docs.items():
        fid = doc.get("fidelity") or {}
        declared = {DECLARED.get(x, x) for x in fid.get("supported", [])}
        default = DECLARED.get(fid.get("default"), fid.get("default"))
        tiers = {}
        for tier in TIERS:
            d, i = tier in declared, impl[tier]["implemented"]
            if tier == "l2":  # the gz tier is chosen by run_spec, not by the profile's fidelity field
                d = pid in GZ_MODELS
                tiers[tier] = {
                    "declared": d,
                    "implemented": i,
                    "gz_model": GZ_MODELS.get(pid),
                    "status": "可用（gz）" if d and i else "无（gz 无该机型）",
                }
                continue
            tiers[tier] = {
                "declared": d,
                "implemented": i,
                "status": "可用" if d and i else ("已声明未实现" if d else ("代码有、机型未声明" if i else "无")),
            }
        sensors = {}
        for name, s in (doc.get("sensors") or {}).items():
            sd = _sensor_doc(vdir, path.parent, str(s.get("file", "")), yaml_load) if isinstance(s, dict) else {}
            sensors[name] = {
                "file": s.get("file") if isinstance(s, dict) else None,
                "default": bool(s.get("default", False)) if isinstance(s, dict) else False,
                "schema": sd.get("schema"),
                "gimbal": _gimbal_of(sd),
            }
        cm = cat_by_id.get(pid, {})
        frame = doc.get("frame") or {}
        batt = doc.get("battery") or {}
        devices.append(
            {
                "id": pid,
                "domain": "air",
                "display_name": doc.get("display_name"),
                "display_name_zh": cm.get("display_name_zh", doc.get("display_name")),
                "category": cm.get("category", frame.get("type")),
                "size_class": cm.get("size_class"),
                "frame": frame.get("type"),
                "mass_kg": _val(doc.get("mass_kg")),
                "mtow_kg": _val(doc.get("mtow_kg")),
                "hover_endurance_s": _val(batt.get("hover_endurance_s")),
                "capacity_wh": _val(batt.get("capacity_wh")),
                "wind_rating_mps": _val(doc.get("wind_rating_mps")),
                "status": doc.get("status"),
                "fidelity_declared": sorted(declared),
                "fidelity_default": default,
                "tiers": tiers,
                "sensors": sensors,
                "summary_zh": cm.get("summary_zh", ""),
                "source": str(path.relative_to(vdir)),
            }
        )
    devices.extend(
        {
            **p,
            "status": "planned",
            "fidelity_declared": [],
            "fidelity_default": None,
            "tiers": {t: {"declared": False, "implemented": impl[t]["implemented"], "status": "无"} for t in TIERS},
            "sensors": {},
            "source": None,
        }
        for p in PLANNED
    )
    return {
        "schema": SCHEMA,
        "tiers": {t: {**TIERS[t], **impl[t]} for t in TIERS},
        "gimbal_tiers": GIMBAL_TIERS,
        "devices": devices,
    }


def servo_responses(dt: float = 0.02, n: int = 150) -> dict:
    """Step responses of a 90°/s gimbal axis to a 60° command under both servo tiers (the device page draws them)."""
    from awr.sim.sensors import servo as SV

    r1 = SV.response(60.0, 90.0, dt, n, "l1")
    r2 = SV.response(60.0, 90.0, dt, n, "l2", acc_max=SV.L2_DEFAULTS["acc_max_deg_s2"], tau=SV.L2_DEFAULTS["tau_s"])
    t1 = next((i * dt for i, a in enumerate(r1) if abs(60.0 - a) <= 0.05), math.inf)
    t2 = SV.settle_time_s(60.0, 90.0, SV.L2_DEFAULTS["acc_max_deg_s2"], SV.L2_DEFAULTS["tau_s"], dt, 0.05)
    return {
        "dt_s": dt,
        "delta_deg": 60.0,
        "rate_max_deg_s": 90.0,
        "l2": dict(SV.L2_DEFAULTS),
        "l1": r1,
        "l2_response": r2,
        "settle_s": {"l1": t1, "l2": t2},
    }


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="awr.sim.device")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ladder", help="device fidelity ladder")
    s.add_argument("--vehicles", type=Path)
    s.add_argument("--json", action="store_true")
    s.add_argument("--responses", action="store_true", help="include the gimbal servo step responses")
    a = ap.parse_args(argv)
    doc = ladder(a.vehicles)
    if a.responses:
        doc["servo_responses"] = servo_responses()
    if a.json:
        print(json.dumps(doc, ensure_ascii=False, indent=1))
        return 0
    print("tiers:", {t: v["status"] for t, v in doc["tiers"].items()})
    for d in doc["devices"]:
        print(
            f"{d['id']:14s} {d['domain']:4s} {d.get('category')!s:11s} declared={d['fidelity_declared']} "
            + " ".join(f"{t}:{v['status']}" for t, v in d["tiers"].items())
        )
    return 0


__all__ = [
    "DECLARED",
    "GIMBAL_TIERS",
    "GZ_MODELS",
    "PLANNED",
    "SCHEMA",
    "TIERS",
    "implemented",
    "ladder",
    "main",
    "servo_responses",
]
