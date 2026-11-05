"""`bench/act.v1` 动作构造（Env 侧副本）：技能执行器在进程内直接组装动作 dict，不导入研究者 SDK 包 `anet4d`（SDK 不依赖 `awr`，
`awr` 也不依赖 SDK；两处构造函数的输出相同，`tests/d3/skills/test_skills_runner.py` 对拍）。"""

from __future__ import annotations

from typing import Any

__all__ = ["act", "gimbal_look_at", "nav_goto", "nav_hover", "nav_land", "nav_takeoff"]

SCHEMA = "drone4d.bench.act.v1"


def act(k: int, *, nav: dict | None = None, gimbal: dict | None = None) -> dict:
    a: dict[str, Any] = {"schema": SCHEMA, "k": int(k)}
    if nav is not None:
        a["nav"] = nav
    if gimbal is not None:
        a["gimbal"] = gimbal
    return a


def nav_goto(pos_enu_m, speed_mps: float | None = None) -> dict:
    n = {"mode": "goto", "pos_enu_m": [float(x) for x in pos_enu_m]}
    if speed_mps is not None:
        n["speed_mps"] = float(speed_mps)
    return n


def nav_hover() -> dict:
    return {"mode": "hover"}


def nav_takeoff(alt_agl_m: float) -> dict:
    return {"mode": "takeoff", "alt_agl_m": float(alt_agl_m)}


def nav_land() -> dict:
    return {"mode": "land"}


def gimbal_look_at(sensor_id: str, target_enu_m) -> dict:
    return {"sensor_id": sensor_id, "mode": "look_at", "target_enu_m": [float(x) for x in target_enu_m]}
