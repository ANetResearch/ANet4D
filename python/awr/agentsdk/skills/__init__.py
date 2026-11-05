"""技能词表（接口 ⑤ 观测与动作的动作侧；引擎计划第 6 步，ADR-227）：外部指挥者（视觉语言模型、脚本计划、HTTP 智能体）用一个封闭
词表指挥整个机队，Env 侧把每条技能展开成逐 tick 的 `bench/act.v1` 导航与云台动作（`runner`、`fleet`）。

词表（`SKILLS`；契约 `bench/decision.v1` 的 `SkillCommand`）：

| 技能 | 参数 | 含义 |
|---|---|---|
| goto | x, y, [agl, speed] | 飞到一点（地面以上 agl 米），到达后相机对着它 |
| sweep | polygon 或 x, y, [size]; [agl, lane, speed] | 割草机式覆盖，相机垂直向下 |
| orbit | x, y, [radius, period, agl, speed] | 绕点盘旋，相机对着圆心 |
| follow | x, y, [agl, standoff, speed] | 飞到一个（移动的）地面点上方并保持相机对着它 |
| look_at | x, y | 只改相机瞄点，不改飞行 |
| hold | — | 悬停 |
| rtl | — | 返回起飞点降落（结束该机） |
| zoom | factor | 光学变焦（只在运行开启光学镜头时可用） |

一条决策（`validate_decision`）是 `{commands: [SkillCommand], reports: [object], done: bool, summary?}`：`commands` 按词表校验，
`reports` 的类型与字段由任务层定义（CHIMERA 的任务类型），本包只校验其为对象数组，任务层经 `reports=` 注入自己的校验；全部数字
须有限（不接受 NaN 与无穷）。

2026-11-05 自 CHIMERA 基准移入（`chimera.commander.skills`、`chimera.sandbox.bridge` 的指令校验）。校验的拒绝文案与 CHIMERA 的 HTTP
网关相同，已接入的外部智能体不需要改动；唯一的收紧是 `speed` 与其他数值参数一起在网关校验（此前由执行器在下一 tick 拒绝）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["NUMERIC_PARAMS", "SKILLS", "SkillError", "SkillSpec", "validate_command", "validate_decision"]


class SkillError(ValueError):
    """一条不合词表的指令或决策；文案面向外部智能体（HTTP 网关以 400 返回）。"""


@dataclass(frozen=True)
class SkillSpec:
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    motion: bool = True          # 替换该机当前的运动技能（look_at 与 zoom 不替换）
    help: str = ""


SKILLS: dict[str, SkillSpec] = {
    "goto": SkillSpec(("x", "y"), ("agl", "speed"), help="fly to a point at the given height above ground and keep the camera on it after arrival"),
    "sweep": SkillSpec((), ("polygon", "x", "y", "size", "agl", "lane", "speed"), help="lawnmower coverage of an area, camera straight down"),
    "orbit": SkillSpec(("x", "y"), ("radius", "period", "agl", "speed"), help="circle a point with the camera on it (defaults: radius 50 m, period 90 s)"),
    "follow": SkillSpec(("x", "y"), ("agl", "standoff", "speed"), help="fly over a ground point and keep the camera on it; resend each step to follow"),
    "look_at": SkillSpec(("x", "y"), motion=False, help="aim the camera at a ground point without changing the flight"),
    "hold": SkillSpec((), help="hover in place"),
    "rtl": SkillSpec((), help="return to the home pad and land"),
    "zoom": SkillSpec(("factor",), motion=False, help="optical zoom of that drone's camera (1 to 12)"),
}

NUMERIC_PARAMS = ("x", "y", "agl", "standoff", "radius", "period", "size", "lane", "factor", "speed")
ZOOM_RANGE = (1.0, 12.0)
ORBIT_MIN_RADIUS_M = 20.0


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_command(cmd, *, uavs=None) -> dict:
    """校验一条 `SkillCommand`；通过时返回原对象，否则抛 `SkillError`。`uavs` 给出时要求 `uav` 在其中。"""
    if not isinstance(cmd, dict) or not isinstance(cmd.get("uav"), str) or not isinstance(cmd.get("skill"), str) \
            or not isinstance(cmd.get("params", {}), dict):
        raise SkillError("invalid command")
    if uavs is not None and cmd["uav"] not in uavs:
        raise SkillError("unknown UAV")
    kind, params = cmd["skill"], cmd.get("params", {})
    spec = SKILLS.get(kind)
    if spec is None:
        raise SkillError("unknown skill")
    if any(k not in params for k in spec.required):
        raise SkillError("missing skill parameter")
    for key in NUMERIC_PARAMS:
        if key in params and not _num(params[key]):
            raise SkillError("skill coordinates/limits must be numeric")
    if kind == "orbit" and (params.get("radius", 50) < ORBIT_MIN_RADIUS_M or params.get("period", 90) <= 0):
        raise SkillError("invalid orbit")
    if kind == "zoom" and not ZOOM_RANGE[0] <= params["factor"] <= ZOOM_RANGE[1]:
        raise SkillError("zoom must be 1..12")
    if kind == "sweep":
        if "polygon" in params:
            polygon = params["polygon"]
            if not isinstance(polygon, list) or len(polygon) < 3 or any(not isinstance(v, list) or len(v) != 2 or any(not _num(n) for n in v) for v in polygon):
                raise SkillError("invalid sweep polygon")
        elif not all(k in params for k in ("x", "y")):
            raise SkillError("sweep needs polygon or x,y")
    return cmd


def validate_decision(action, *, uavs=None, reports: Callable[[dict], None] | None = None) -> dict:
    """校验一条决策 `{commands, reports, done, ...}`（`bench/decision.v1`）；通过时返回原对象。

    `reports(report)` 为任务层的报告校验（类型、证据、计数……），对每条报告对象调用，不合规时由它抛 `ValueError`。"""
    if not isinstance(action, dict):
        raise SkillError("action must be an object")
    if not isinstance(action.get("commands", []), list) or not isinstance(action.get("reports", []), list):
        raise SkillError("commands/reports must be arrays")
    if not isinstance(action.get("done", False), bool):
        raise SkillError("done must be boolean")
    for c in action.get("commands", []):
        validate_command(c, uavs=uavs)
    for r in action.get("reports", []):
        if not isinstance(r, dict):
            raise SkillError("invalid report")
        if reports is not None:
            reports(r)
    try:                                                      # 拒绝 NaN 与无穷（含嵌套参数）：线上 JSON 没有这些扩展
        json.dumps(action, allow_nan=False)
    except ValueError as e:
        raise SkillError("numbers must be finite") from e
    return action
