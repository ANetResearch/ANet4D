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
``l2``            realistic: the unified L2 engine (``awr.phys.l2``: rotor- or actuator-level actuation, rate loop,
                  control allocation, rigid body with inertia, aerodynamics) with one domain model per device family;
                  gz-sim lockstep (``awr.phys.gz``) stays as a cross-check where it has a model (``x500_d4d``)
``l3``            software in the loop: the real PX4 / ArduPilot flight stack (``px4_sih`` is the only hook)
``fw``            fixed-wing / VTOL fast tier: kinematic flight with energy (``Fidelity.FW``, stage ``fw_kin``)
================  ==============================================================================================

and for gimbals the servo tiers ``l1`` (rate-limited) and ``l2`` (second-order servo), see ``awr.sim.sensors.servo``.

``ladder()`` is what the device page shows; a tier is *available* only when the profile declares it and the code
implements it (for ``l2``: when the unified engine has a domain model for the device's family). Land and sea devices are
listed as planned rows until their profiles and domain models land.
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
        "label_zh": "Realistic：统一 L2 引擎（转子 / 执行器级驱动、速率环、控制分配、刚体惯量、气动）；gz-sim 锁步为交叉验证",
        "module": "awr.phys.l2",
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
# domain models of the unified L2 engine, by device family (catalog category); a family without a module has no l2 yet
L2_DOMAINS = {"multirotor": "awr.phys.l2.multirotor", "fixed_wing": "awr.phys.l2.fixedwing", "vtol": "awr.phys.l2.fixedwing",
              "ugv": "awr.phys.l2.ugv", "usv": "awr.phys.l2.usv"}

PLANNED = [
    {
        "id": "ugv_rover",
        "domain": "land",
        "display_name_zh": "轮式地面车（JPL 开源漫游车级）",
        "category": "ugv",
        "size_class": "small",
        "note": "第 5 步：机型包 + 统一引擎 ugv 域模型（自行车 + 轮胎力）；参考 JPL Open Source Rover",
    },
    {
        "id": "ugv_f1tenth",
        "domain": "land",
        "display_name_zh": "1/10 自动驾驶车（F1TENTH 级）",
        "category": "ugv",
        "size_class": "small",
        "note": "第 5 步：机型包 + 统一引擎 ugv 域模型；参考 F1TENTH",
    },
    {
        "id": "ugv_podcar",
        "domain": "land",
        "display_name_zh": "小型乘用车（OpenPodcar 级）",
        "category": "ugv",
        "size_class": "medium",
        "note": "第 5 步：机型包 + 统一引擎 ugv 域模型；参考 OpenPodcar 2",
    },
    {
        "id": "usv_small",
        "domain": "sea",
        "display_name_zh": "小型无人艇（BlueBoat 级）",
        "category": "usv",
        "size_class": "small",
        "note": "第 5 步：机型包 + 统一引擎 usv 域模型（Fossen 三自由度）；参考 BlueBoat",
    },
    {
        "id": "usv_medium",
        "domain": "sea",
        "display_name_zh": "中型无人艇（NAC-USV 级）",
        "category": "usv",
        "size_class": "medium",
        "note": "第 5 步：机型包 + 统一引擎 usv 域模型；参考 NAC-USV",
    },
]


def _val(d):
    return d.get("value") if isinstance(d, dict) and "value" in d else d


def _domain(category, frame_type) -> str:
    """Device family of a catalog row for the L2 domain models: multirotor, fixed_wing, vtol, ugv, usv."""
    if category in L2_DOMAINS:
        return str(category)
    if frame_type in ("quad_x", "hex_x"):
        return "multirotor"
    if frame_type == "fixed_wing":
        return "fixed_wing"
    if frame_type and str(frame_type).startswith("vtol"):
        return "vtol"
    return str(category or "unknown")


def _planned_tier(tier: str, impl: dict, category: str) -> dict:
    row = {"declared": False, "implemented": impl[tier]["implemented"], "status": "无"}
    if tier == "l2":
        row.update(domain=category, engine=None, gz_model=None, status=f"无（统一引擎尚无 {category} 模型）")
    return row


def implemented() -> dict[str, dict]:
    """Which tiers the code implements, by looking for the modules that carry them (not by trusting a table)."""
    out = {}
    for tier, t in TIERS.items():
        mod = t["module"]
        present = bool(mod) and importlib.util.find_spec(mod) is not None
        if tier == "l2":
            doms = {d: importlib.util.find_spec(m) is not None for d, m in L2_DOMAINS.items()}
            have = sorted({d for d, ok in doms.items() if ok})
            gz = importlib.util.find_spec("awr.phys.gz.backend") is not None
            out[tier] = {
                "implemented": present,
                "status": f"可用（统一引擎：{'、'.join(have)}）" if present and have else "无",
                "domains": doms,
                "gz": gz,
                "detail": "awr.phys.l2（统一 L2 引擎，CPU，与 Fast 同一指令映射与参考跟踪器）；域模型："
                + "，".join(f"{d} {'有' if ok else '无'}" for d, ok in doms.items())
                + ("；gz-sim 锁步交叉验证（d3-gz 容器，模型 x500_d4d）" if gz else "；无 gz"),
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
        cm = cat_by_id.get(pid, {})
        frame = doc.get("frame") or {}
        fid = doc.get("fidelity") or {}
        declared = {DECLARED.get(x, x) for x in fid.get("supported", [])}
        default = DECLARED.get(fid.get("default"), fid.get("default"))
        tiers = {}
        for tier in TIERS:
            d, i = tier in declared, impl[tier]["implemented"]
            if tier == "l2":  # chosen by run_spec, not by the profile's fidelity field: available when the engine has the domain model
                dom = _domain(cm.get("category", frame.get("type")), frame.get("type"))
                have = bool(impl["l2"]["domains"].get(dom)) and i
                gz = pid in GZ_MODELS and bool(impl["l2"].get("gz"))
                tiers[tier] = {
                    "declared": have,
                    "implemented": i,
                    "domain": dom,
                    "engine": "unified" if have else None,
                    "gz_model": GZ_MODELS.get(pid) if gz else None,
                    "status": ("可用（统一引擎 + gz）" if gz else "可用（统一引擎）") if have else f"无（统一引擎尚无 {dom} 模型）",
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
            "tiers": {t: _planned_tier(t, impl, p["category"]) for t in TIERS},
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
    "L2_DOMAINS",
    "PLANNED",
    "SCHEMA",
    "TIERS",
    "implemented",
    "ladder",
    "main",
    "servo_responses",
]
