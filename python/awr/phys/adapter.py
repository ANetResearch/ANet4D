"""档位适配层的公共类型（M24 §6.3.4、§6.3.5、§6.2、§7.1）：TierCommand、ContactEvent、TickResult、VehicleStatus 统一派生、
时钟档校验与 PhysicsTier 协议。三档共用，坐标一律 ENU（m、m/s），机体系 FLU。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol

import numpy as np

from awr.phys import PhysError

__all__ = ["CLOCK_PROFILES", "IMPACT_VZ_MPS", "LAND_STILL_S", "STILL_S", "STILL_V_MPS", "TAKEOFF_GRACE_S", "ClockProfile", "ContactEvent", "DispatchResult",
           "OrbitSpec", "PhysicsTier", "StatusSignals", "TickResult", "TickTiming", "TierCommand", "VehicleStatus", "derive_status",
           "steps_per_tick", "validate_clock"]

# §6.5.9 接触阈值（与 D2 ContactCfg 一致）：落地 = 触地且 ‖v‖ < 0.25 m/s 持续 0.5 s（land）或 1.0 s（其他）；硬着陆竖直速度 > 3 m/s
STILL_V_MPS = 0.25
LAND_STILL_S = 0.5
STILL_S = 1.0
IMPACT_VZ_MPS = 3.0
# 起飞宽限：飞行指令生效后、首次离地前的 3 s 内不做落地判定（静止起飞的电机加速与推力爬升可超过 1 s，不能判为"已落地"）
TAKEOFF_GRACE_S = 3.0

CmdKind = Literal["velocity", "goto", "follow_path", "orbit", "hover", "takeoff", "land", "kill"]
ContactKind = Literal["TOUCHDOWN", "LIFTOFF", "COLLISION_WORLD", "IMPACT", "COLLISION_UAV"]


class VehicleStatus(StrEnum):
    LOST = "LOST"
    CRASHED = "CRASHED"
    GROUNDED = "GROUNDED"
    LANDING = "LANDING"
    AIRBORNE = "AIRBORNE"
    MOVING = "MOVING"                              # ground vehicles under way (GROUNDED = at rest); air vehicles never report it


@dataclass(frozen=True)
class OrbitSpec:
    center_enu_m: tuple[float, float, float]
    radius_m: float
    cw: bool = False
    turns: float | None = None


@dataclass(frozen=True)
class TierCommand:
    """M26 动作映射与守卫之后的机体指令；`apply_tick` 为生效的主时钟 tick。"""
    uav: str
    kind: CmdKind
    apply_tick: int
    vel_enu_mps: tuple[float, float, float] | None = None
    yaw_rate_rps: float | None = None
    target_enu_m: tuple[float, float, float] | None = None
    path_enu_m: np.ndarray | None = None
    speed_mps: float | None = None
    orbit: OrbitSpec | None = None
    alt_m: float | None = None


@dataclass(frozen=True)
class DispatchResult:
    entity_id: str
    ok: bool
    code: int = 0
    native_ack: bool = True
    detail: str | None = None


@dataclass(frozen=True)
class ContactEvent:
    uav: str
    t_ns: int
    kind: ContactKind
    detector: Literal["dsm", "mesh", "gz", "px4"]
    pos_enu_m: tuple[float, float, float]
    normal_enu: tuple[float, float, float] | None
    speed_mps: float
    prim_id: int | None = None
    semantic: int | None = None


@dataclass
class TickTiming:
    """墙钟 ns。"""
    encode: int = 0
    physics: int = 0
    wait: int = 0
    decode: int = 0
    mirror: int = 0
    append: int = 0


@dataclass(frozen=True)
class TickResult:
    k: int
    t_start_ns: int
    t_end_ns: int
    n_steps: int
    contacts: tuple[ContactEvent, ...]
    status: Mapping[str, VehicleStatus]
    timing: TickTiming = field(default_factory=TickTiming)


@dataclass(frozen=True)
class StatusSignals:
    link_ok: bool = True
    crash_cause: Literal["none", "world", "impact", "uav", "mesh"] = "none"
    landed: bool = False
    in_air: bool = False
    cmd_mode: str = "HOVER"


def derive_status(sig: StatusSignals) -> VehicleStatus:
    """§6.3.4 优先级：LOST > CRASHED > GROUNDED > LANDING > AIRBORNE（三档同一函数）。"""
    if not sig.link_ok:
        return VehicleStatus.LOST
    if sig.crash_cause != "none":
        return VehicleStatus.CRASHED
    if sig.landed:
        return VehicleStatus.GROUNDED
    if sig.in_air and sig.cmd_mode in ("LAND", "ELAND", "DESCENT"):
        return VehicleStatus.LANDING
    if sig.in_air:
        return VehicleStatus.AIRBORNE
    return VehicleStatus.GROUNDED


@dataclass(frozen=True)
class ClockProfile:
    name: str
    main_hz: int
    physics_hz: int
    l1_hz: int


CLOCK_PROFILES = {"d1_250": ClockProfile("d1_250", 250, 250, 250), "d1_250_l1_125": ClockProfile("d1_250", 250, 125, 125),
                  "sandbox100": ClockProfile("sandbox100", 100, 100, 100)}
_NO_L1_HALF = frozenset({"imu", "lidar", "bag"})


def steps_per_tick(tick_ms: int, physics_hz: int) -> int:
    return tick_ms * physics_hz // 1000


def validate_clock(tier: Literal["fast", "realistic", "sitl"], cp: ClockProfile, tick_ms: int, outputs: Sequence[str] = ()) -> None:
    """§6.2 规则；违反任一条抛 PhysError(637)。"""
    bad: list[str] = []
    if tier in ("realistic", "sitl") and (cp.main_hz, cp.physics_hz, cp.l1_hz) != (250, 250, 250):
        bad.append("realistic and sitl run only {d1_250, 250, 250, 250}")
    if cp.main_hz % cp.l1_hz:
        bad.append("main_hz / l1_hz must be an integer")
    step_ms = 1000 / cp.physics_hz
    if (tick_ms * cp.physics_hz) % 1000:
        bad.append(f"tick {tick_ms} ms is not a multiple of the physics step {step_ms:g} ms")
    if cp.l1_hz != cp.main_hz and _NO_L1_HALF & set(outputs):
        bad.append("l1_hz below main_hz excludes imu, lidar and bag outputs")
    if cp.name == "sandbox100" and _NO_L1_HALF & set(outputs):
        bad.append("sandbox100 excludes imu, lidar and bag outputs")
    if bad:
        raise PhysError(637, "; ".join(bad), {"violations": bad})


class PhysicsTier(Protocol):
    """§7.1：Env 驱动的物理档。open 在 PREPARING 阶段调用；advance 推进一个 Env tick。"""
    name: str

    def submit(self, cmds: Sequence[TierCommand]) -> list[DispatchResult]: ...

    def advance(self, k: int) -> TickResult: ...

    def status(self) -> Mapping[str, VehicleStatus]: ...

    def close(self) -> dict: ...
