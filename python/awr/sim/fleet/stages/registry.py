"""sim-core 扩展点注册表（M08 §7.1.1，签名在 D1-MS1 冻结；AWR-03 §4.3；M08-FR-011、FR-012、FR-055）。

M07、M09、M10、M13 在各自插件模块被组合根导入时调用下列函数登记，不修改 `awr/sim/runtime`：

- `@register_stage(name, every, phase, order)`：pipeline stage；`owner` 缺省按函数所在包推断（`budgets.PACKAGE_OWNER`），
  推断不出时按 order 所在区段推断；`budget_core` 缺省取 `budgets.BUDGET_CORE`，stage 名不在表中且未显式给出时拒绝；
- `register_state_block(name, owner, fields)`：他模块的 SoA 由 M08 统一分配（FleetState.blocks[name]）；
- `register_admission_check(step, name, fn)`：准入第 ④ 或 ⑧ 步的检查（也可作装饰器）；
- `register_slow_task(name, fn, ...)`：慢任务轮转（主循环剩余预算内执行，FR-004）；
- `register_query(name, fn)`：`ctl/sim-core/query` 的路由（M07 `env/query`）；
- `register_energy_model(model)`、`register_safety_hooks(hooks)`：M09 各调用一次，重复登记即失败。

度量注册表在 `awr/sim/core/metrics.py`，运动提供者注册在 `awr/sim/core/command.py`（M08 §7.1.1）。

D2 追加（M08 §15.3.1，只追加关键字与新函数，D1-MS1 冻结的签名不变，M08-FR-065）：
- `register_stage(..., hz=, phase_slot=)`：按频率声明（M08-FR-202）。给出 `hz` 时 `every = tick_hz / hz`（必须整除，否则
  构建失败），`phase = phase_slot mod every`；只给 `every`、`phase` 的 D1 写法按 d1_250 的 tick 解释，其他档位由按档调度表
  `stages/rates.py` 给出（表中缺项构建失败）。D2 新模块（M18–M21）一律用 `hz` 写法；
- `register_entity_block(kind, name, owner, fields)`：识别物实体表的列（M08-FR-216；容量 32，按 `target_no` 升序）；
- `register_power_model(frame_kind, model)`：按构型的功率模型（M08-FR-214）；
- `register_profile_check(frame_types, check_id, fn)`：按构型登记的机型自洽检查（M08-FR-207，M21 的 FW、VT 检查）；
- `register_admission_check(6, ...)`：准入第 ⑥ 步参数检查（M08-FR-219），可按构型过滤；
- `register_command_handler(..., vehicle=True)`：机体范围的插件命令（`gimbal`、`zoom`、`sensor/mode`，M08 先做准入 ④⑤⑦）；
- `register_session_loader(section, fn)`、`register_snapshot_exporter(section, fn)`：会话配置（`session.json`、
  `sim/reset{sandbox_config}`、`ctl/sim-core/snapshot`）中各领域段的装配与导出；
- `register_batch_output(name, fn)`：批处理结束时写出的结果文件（M20 `relay_stats`，M08-FR-221）。

校验（M08-FR-012；M08-AC-003）：`tick_hz % every == 0`、`0 ≤ phase < every`、name 与 order 唯一、order 落在所有者区段、
Σbudget ≤ 0.40（按档位，M08-FR-203）、同一字段只有一个写者。登记时即可判定的项在登记时抛 ValueError/KeyError，其余在
`Pipeline.build` 时判定。本模块只保存登记，不执行任何 stage；不使用墙钟（ADR-049）。
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from enum import IntFlag
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from . import budgets as B

if TYPE_CHECKING:
    from awr.sim.core.interfaces import EnergyModel, SafetyHooks
    from awr.sim.fleet.pipeline import StageCtx
    from awr.sim.fleet.state import FleetState

__all__ = [
    "AdmissionCheck",
    "CheckpointExtSpec",
    "CommandHandlerSpec",
    "EntityBlockSpec",
    "Fidelity",
    "ProfileCheckSpec",
    "QuerySpec",
    "Registry",
    "SlowTaskSpec",
    "StageFn",
    "StageSpec",
    "StateBlockSpec",
    "energy_model",
    "infer_owner",
    "isolated_registry",
    "power_models",
    "register_admission_check",
    "register_batch_output",
    "register_checkpoint_extension",
    "register_command_handler",
    "register_energy_model",
    "register_entity_block",
    "register_fleet_update_hook",
    "register_power_model",
    "register_profile_check",
    "register_query",
    "register_safety_hooks",
    "register_session_loader",
    "register_setpoint_filter",
    "register_slow_task",
    "register_snapshot_exporter",
    "register_stage",
    "register_state_block",
    "register_state_ext_hook",
    "registry",
    "reset_registry",
    "safety_hooks",
]


class Fidelity(IntFlag):
    L1 = 1
    L2 = 2
    EXT = 4
    L0 = 8
    FW = 16  # 固定翼与 FW 模式的 VTOL（M08-FR-210）：slot 置 L1 | FW，M08 的 L1 物理 stage 跳过带 FW 位的 slot
    ALL = 31


ENTITY_KINDS = ("target",)
FRAME_KINDS = {"multirotor": 0, "fixed_wing": 1, "vtol": 2}


StageFn = Callable[["FleetState", "StageCtx"], None]


@dataclass(frozen=True)
class StageSpec:
    name: str
    every: int
    phase: int
    order: int
    fn: Callable[..., None]
    owner: str
    fidelity: Fidelity = Fidelity.ALL
    budget_core: float = 0.0
    writes: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    shards: int = 1
    shard: tuple[int, int] = (0, 1)  # (k, n)：分片展开后的第 k 片
    builtin: bool = False
    hz: float | None = None  # 按频率声明（M08-FR-202）；None 为 D1 写法（every、phase 按 d1_250 的 tick）
    phase_slot: int = 0
    budget_explicit: bool = False  # 登记时显式给出 budget_core（按档预算表优先，其次显式值，M08-FR-203）


@dataclass(frozen=True)
class EntityBlockSpec:
    """实体表的列（M08-FR-216）：kind 目前只有 `target`；字段格式同 `register_state_block`。"""

    kind: str
    name: str
    owner: str
    fields: dict[str, tuple[np.dtype, tuple[int, ...]]]


@dataclass(frozen=True)
class ProfileCheckSpec:
    """按构型登记的机型检查（M08-FR-207）：`fn(doc) -> list[{check, value, range, pass, msg}] | dict | bool`。"""

    frame_types: frozenset[str]
    check_id: str
    fn: Callable[[dict], Any]
    owner: str | None


@dataclass(frozen=True)
class StateBlockSpec:
    name: str
    owner: str
    fields: dict[str, tuple[np.dtype, tuple[int, ...]]]
    checkpoint: bool = True


@dataclass(frozen=True)
class AdmissionCheck:
    step: int
    name: str
    fn: Callable[..., Any]
    owner: str | None
    frames: frozenset[int] | None = None  # 只对这些构型（frame_kind）的机体执行；None 为全部（M08-FR-219）


@dataclass(frozen=True)
class SlowTaskSpec:
    name: str
    fn: Callable[..., None]
    period_wall_s: float | None
    period_sim_s: float | None
    budget_us: int


@dataclass(frozen=True)
class QuerySpec:
    name: str
    fn: Callable[..., bytes]


@dataclass(frozen=True)
class CommandHandlerSpec:
    op: str
    fn: Callable[..., dict]
    owner: str | None
    need_seat: bool = True
    vehicle: bool = False  # 机体范围的命令（`uav/{id}/cmd/<op>`）：M08 先执行准入 ④（生命周期、状态行）⑤（租约）⑦


@dataclass(frozen=True)
class CheckpointExtSpec:
    """checkpoint 扩展段（M08-FR-083 的插件登记点；M20-FR-075）：`save() -> bytes` 在 checkpoint 捕获时（持 sim-core 状态锁，
    tick 边界）调用，`restore(bytes)` 在 checkpoint 恢复的最后（机群、识别物实体表、roster、租约与调用表之后）调用。"""

    name: str
    save: Callable[[], bytes]
    restore: Callable[[bytes], Any]
    owner: str | None


@dataclass(frozen=True)
class SessionLoaderSpec:
    section: str
    fn: Callable[..., Any]
    owner: str | None
    order: int


@dataclass
class Registry:
    stages: list[StageSpec] = field(default_factory=list)
    blocks: dict[str, StateBlockSpec] = field(default_factory=dict)
    admission: list[AdmissionCheck] = field(default_factory=list)
    slow: list[SlowTaskSpec] = field(default_factory=list)
    queries: dict[str, QuerySpec] = field(default_factory=dict)
    commands: dict[str, CommandHandlerSpec] = field(default_factory=dict)
    state_ext_hooks: list[tuple[str | None, Callable[..., None]]] = field(default_factory=list)
    energy: Any = None
    hooks: Any = None
    # D2（M08 §15.3）
    entity_blocks: dict[str, dict[str, EntityBlockSpec]] = field(default_factory=dict)
    power: dict[int, Any] = field(default_factory=dict)
    profile_checks: list[ProfileCheckSpec] = field(default_factory=list)
    session_loaders: dict[str, SessionLoaderSpec] = field(default_factory=dict)
    snapshot_exporters: dict[str, Callable[..., Any]] = field(default_factory=dict)
    batch_outputs: dict[str, Callable[..., Any]] = field(default_factory=dict)
    ckpt_exts: dict[str, CheckpointExtSpec] = field(default_factory=dict)
    fleet_update_hooks: list[Callable[..., Any]] = field(default_factory=list)
    setpoint_filters: list[tuple[str | None, Callable[..., Any]]] = field(default_factory=list)
    # plugin runtimes by plugin name: a runtime lives and dies with the registry it was installed into (never keyed by id(),
    # which CPython reuses once a registry is collected)
    plugins: dict[str, Any] = field(default_factory=dict)

    def stage_names(self) -> set[str]:
        return {s.name for s in self.stages}

    def admission_checks(self, step: int) -> list[AdmissionCheck]:
        return [c for c in self.admission if c.step == step]


_REG = Registry()


def registry() -> Registry:
    """当前进程的登记表（组合根导入插件之后读取）。"""
    return _REG


def reset_registry() -> None:
    """清空全部登记（只用于测试；sim-core 进程内不调用）。"""
    global _REG
    _REG = Registry()


@contextlib.contextmanager
def isolated_registry() -> Iterator[Registry]:
    """测试用：在一段代码内使用空登记表，结束后恢复原表。"""
    global _REG
    saved = _REG
    _REG = Registry()
    try:
        yield _REG
    finally:
        _REG = saved


# ---------------------------------------------------------------- owner 推断
def infer_owner(module: str | None, order: int | None = None) -> str | None:
    """按包前缀推断所有者；推断不出时按 order 所在区段推断（测试桩插件与第三方位置）。"""
    if module:
        for prefix, owner in B.PACKAGE_OWNER:
            if module == prefix or module.startswith(prefix + "."):
                return owner
    if order is not None:
        for owner, ranges in B.ORDER_RANGES.items():
            if any(lo <= order <= hi for lo, hi in ranges):
                return owner
    return None


def _check_timing(name: str, every: int, phase: int, order: int, shards: int) -> None:
    if not isinstance(every, int) or isinstance(every, bool) or every < 1 or B.TICK_HZ % every != 0:
        raise ValueError(f"stage {name!r}: every={every} 必须整除 tick_hz={B.TICK_HZ}（M08-FR-012）")
    if not isinstance(phase, int) or not 0 <= phase < every:
        raise ValueError(f"stage {name!r}: phase={phase} 必须满足 0 ≤ phase < every={every}")
    if not isinstance(order, int) or not 0 <= order <= 199:
        raise ValueError(f"stage {name!r}: order={order} 超出 0..199")
    if not isinstance(shards, int) or shards < 1:
        raise ValueError(f"stage {name!r}: shards={shards} 必须 ≥ 1")


def _check_owner_range(name: str, owner: str, order: int, shards: int) -> None:
    ranges = B.ORDER_RANGES.get(owner)
    if ranges is None:
        raise ValueError(f"stage {name!r}: 未知所有者 {owner!r}")
    for k in range(shards):
        o = order + k
        if not any(lo <= o <= hi for lo, hi in ranges):
            raise ValueError(f"stage {name!r}: order {o} 不在 {owner} 的区段 {ranges} 内（M08 §6.4.2）")


def _resolve_budget(name: str, budget_core: float | None) -> float:
    if budget_core is not None:
        if budget_core < 0:
            raise ValueError(f"stage {name!r}: budget_core 不能为负")
        return float(budget_core)
    key = B.base_name(name)
    if key not in B.BUDGET_CORE:
        raise KeyError(f"stage {name!r} 不在预算表（budgets.BUDGET_CORE）中且未显式给出 budget_core（M08-FR-012）")
    return B.BUDGET_CORE[key]


def shard_step(every: int, shards: int) -> int:
    """分片相位间隔：every 为 50 Hz 周期（5 tick）的整数倍时取 5 的倍数，各片与首片落在同一 tick % 5 相位类（不与 50 Hz stage
    同 tick；every 25 且 ≤ 4 片时为 5，与此前 fleet_guard 的 4、9、14、19 相同，ADR-070）；否则 every // (shards + 1)。"""
    if every % 5 == 0 and every // 5 > shards:
        return 5 * max(1, (every // 5) // (shards + 1))
    return max(1, every // (shards + 1))


def _d1_every(name: str, hz: float, phase_slot: int) -> tuple[int, int]:
    """hz 写法在 d1_250 下的 (every, phase)；250 不能被整除时为 (0, 0)（只能在其他档位运行，构建时判定）。"""
    if not isinstance(hz, (int, float)) or isinstance(hz, bool) or hz <= 0:
        raise ValueError(f"stage {name!r}: hz={hz} 必须 > 0（M08-FR-202）")
    if not isinstance(phase_slot, int) or isinstance(phase_slot, bool) or phase_slot < 0:
        raise ValueError(f"stage {name!r}: phase_slot={phase_slot} 必须为非负整数")
    q = B.TICK_HZ / float(hz)
    e = round(q)
    if e < 1 or abs(q - e) > 1e-9:
        return 0, 0
    return int(e), int(phase_slot % e)


def make_stage(name: str, every: int | None, phase: int | None, order: int, fn: Callable[..., None], *,
               owner: str | None = None, fidelity: Fidelity = Fidelity.ALL, budget_core: float | None = None,
               writes: tuple[str, ...] = (), reads: tuple[str, ...] = (), shards: int = 1, builtin: bool = False,
               hz: float | None = None, phase_slot: int = 0) -> list[StageSpec]:
    """构造（并在 shards > 1 时展开）StageSpec；不登记。M08 自身的 stage 经此函数构造（builtin = True）。
    `hz` 与 `every`/`phase` 二选一（M08-FR-202）；hz 写法的 (every, phase) 在 `Pipeline.build` 时按档位解析。"""
    if hz is not None:
        if every is not None or phase is not None:
            raise ValueError(f"stage {name!r}: hz 与 every/phase 不能同时给出（M08-FR-202）")
        every, phase = _d1_every(name, hz, phase_slot)
        if not isinstance(order, int) or not 0 <= order <= 199:
            raise ValueError(f"stage {name!r}: order={order} 超出 0..199")
        if not isinstance(shards, int) or shards < 1:
            raise ValueError(f"stage {name!r}: shards={shards} 必须 ≥ 1")
    else:
        if every is None or phase is None:
            raise ValueError(f"stage {name!r}: 须给出 every 与 phase，或 hz（M08-FR-202）")
        _check_timing(name, every, phase, order, shards)
    own = owner or infer_owner(getattr(fn, "__module__", None), order)
    if own is None:
        raise ValueError(f"stage {name!r}: 无法推断所有者（请显式给出 owner）")
    _check_owner_range(name, own, order, shards)
    budget = _resolve_budget(name, budget_core)
    base = StageSpec(name, every, phase, order, fn, own, Fidelity(fidelity), budget, tuple(writes), tuple(reads), shards,
                     (0, 1), builtin, None if hz is None else float(hz), int(phase_slot), budget_core is not None)
    if shards == 1:
        return [base]
    step = shard_step(every, shards) if every else 1
    per = budget / shards
    return [replace(base, name=f"{name}.{k}", order=order + k, phase=(phase + k * step) % every if every else 0,
                    budget_core=per, shard=(k, shards)) for k in range(shards)]


# ---------------------------------------------------------------- 登记函数（签名冻结，M08 §7.1.1）
def register_stage(name: str, every: int | None = None, phase: int | None = None, order: int | None = None, *,
                   owner: str | None = None, fidelity: Fidelity = Fidelity.ALL, budget_core: float | None = None,
                   writes: tuple[str, ...] = (), reads: tuple[str, ...] = (), shards: int = 1, hz: float | None = None,
                   phase_slot: int = 0) -> Callable[[StageFn], StageFn]:
    """装饰器：登记 pipeline stage。`(tick - phase) % every == 0` 时执行；shards > 1 时展开为 `<name>.<k>`。
    D2：`hz`、`phase_slot` 按频率声明（M08-FR-202）。"""
    if order is None:
        raise ValueError(f"stage {name!r}: 须给出 order")

    def deco(fn: StageFn) -> StageFn:
        specs = make_stage(name, every, phase, order, fn, owner=owner, fidelity=fidelity, budget_core=budget_core,
                           writes=writes, reads=reads, shards=shards, hz=hz, phase_slot=phase_slot)
        names = _REG.stage_names()
        orders = {s.order for s in _REG.stages}
        for s in specs:
            if s.name in names:
                raise ValueError(f"stage 重名：{s.name!r}")
            if s.order in orders:
                raise ValueError(f"stage {s.name!r}: order {s.order} 已被占用")
        _REG.stages.extend(specs)
        return fn

    return deco


def register_state_block(name: str, owner: str, fields: dict[str, tuple[np.dtype, tuple[int, ...]]], *,
                         checkpoint: bool = True) -> None:
    """登记状态块；FleetState 构造时按容量分配 `blocks[name][field]`，形状 (capacity, *shape)。"""
    if name in _REG.blocks:
        raise ValueError(f"状态块重复登记：{name!r}")
    norm: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
    for f, spec in fields.items():
        dt, shape = spec
        norm[f] = (np.dtype(dt), tuple(int(x) for x in shape))
    _REG.blocks[name] = StateBlockSpec(name, owner, norm, checkpoint)


def register_admission_check(step: Literal[4, 6, 8], name: str, fn: Callable[..., Any] | None = None, *,
                             owner: str | None = None, frames: frozenset[int] | set[int] | None = None) -> Callable:
    """准入第 ④（状态）、⑥（参数，D2：M08-FR-219）或 ⑧（围栏粗校验）步的检查；`fn(req, ctx) -> AdmitResult`。
    `frames` 为构型集合（`frame_kind`：0 多旋翼、1 固定翼、2 VTOL）时只对这些构型的机体执行。fn 缺省时作装饰器使用。"""
    if step not in (4, 6, 8):
        raise ValueError(f"准入检查只能登记在第 4、6 或 8 步：{step}")
    fr = None if frames is None else frozenset(int(x) for x in frames)

    def add(f: Callable[..., Any]) -> Callable[..., Any]:
        if any(c.name == name for c in _REG.admission):
            raise ValueError(f"准入检查重名：{name!r}")
        _REG.admission.append(AdmissionCheck(int(step), name, f, owner or infer_owner(getattr(f, "__module__", None)),
                                             fr))
        return f

    if fn is None:
        return add
    return add(fn)


def register_slow_task(name: str, fn: Callable[[StageCtx], None], *, period_wall_s: float | None = None,
                       period_sim_s: float | None = None, budget_us: int = 200) -> None:
    """慢任务：主循环剩余预算内轮转执行（FR-004）；可按墙钟或仿真时钟设置周期，缺省每轮都有机会执行。"""
    if any(t.name == name for t in _REG.slow):
        raise ValueError(f"慢任务重名：{name!r}")
    if budget_us <= 0:
        raise ValueError(f"慢任务 {name!r}: budget_us 必须 > 0")
    _REG.slow.append(SlowTaskSpec(name, fn, period_wall_s, period_sim_s, int(budget_us)))


def register_query(name: str, fn: Callable[..., bytes]) -> None:
    """`ctl/sim-core/query` 的 op 路由（例如 `env/query`）；处理在慢任务中执行。"""
    if name in _REG.queries:
        raise ValueError(f"查询重复登记：{name!r}")
    _REG.queries[name] = QuerySpec(name, fn)


VEHICLE_PLUGIN_OPS = ("gimbal", "zoom", "sensor/mode")  # 17 §17.6.3 的机体载荷命令（M13 处理，M08 做准入 ④⑤⑦）


def register_command_handler(op: str, fn: Callable[..., dict], *, owner: str | None = None, need_seat: bool = True,
                             vehicle: bool = False) -> None:
    """非机体命令的 `ctl/sim-core/cmd` 路由（追加的扩展点，M08 §7.1.1 之外；例如 M07 `env/set`、`env/preset`、`env/gust`，
    M10 `mission/start|pause|resume|abort`，M09 `fault/inject`）。

    CommandEngine 在验签与角色检查（外部写操作须 operator/admin 且持席位，`need_seat = False` 时只验签）之后，于步顶同步调用
    `fn(msg, apply_tick, ctx) -> {status: accepted|rejected, code, detail?, warnings?, result?}`，把结果作为 Admission 回复，
    并按 cid 进入幂等表。处理函数不得阻塞（重活放慢任务）；同一 op 只允许一个处理者。

    D2（M08-FR-218）：`vehicle = True` 的命令以 `uav` 指定机体（`uav/{id}/cmd/gimbal`、`zoom`、`sensor/mode`，M13 处理），
    M08 先执行准入 ④（机体存在 107、生命周期 108、状态行：UNKNOWN 与 CRASHED 105）⑤（租约，按 operator 隐式取得）⑦，
    处理函数的 msg 另带 `_slot`（FleetState 下标）。`VEHICLE_PLUGIN_OPS` 中的 op 恒按机体命令路由（不论 `vehicle` 实参）。"""
    if op in _REG.commands:
        raise ValueError(f"命令处理者重复登记：{op!r}")
    if not op or op in ("fleet/add", "fleet/remove", "fleet/update", "scenario/metric") or op.startswith("fleet/cmd"):
        raise ValueError(f"保留的 op：{op!r}")
    _REG.commands[op] = CommandHandlerSpec(op, fn, owner or infer_owner(getattr(fn, "__module__", None)), bool(need_seat),
                                           bool(vehicle) or op in VEHICLE_PLUGIN_OPS)


def register_state_ext_hook(fn: Callable[..., None], *, owner: str | None = None) -> None:
    """`state/sim-core/ext` 打包钩子（追加的扩展点；M13-to-M08 第 2 条）：`fn(slots, t_sim_ns, out)` 在每片（≤ 16 架）打包时于
    主循环线程调用，`out[i]` 为第 i 个 slot 正在装配的 state_ext 对象，钩子原地合并本模块负责的字段（M13：`loc.gnss_fix/sats/
    eph_m/epv_m/hdop`、`loc.err_enu_m`、`sens`）；字段须已在 `uav_state_ext.schema.json` 登记。钩子不得使用 RNG 或阻塞。
    同一函数重复登记被忽略（插件对登记表幂等）。"""
    if not callable(fn):
        raise TypeError("state_ext 钩子不可调用")
    if any(f is fn for _o, f in _REG.state_ext_hooks):
        return
    _REG.state_ext_hooks.append((owner or infer_owner(getattr(fn, "__module__", None)), fn))


def register_energy_model(model: EnergyModel) -> None:
    """M09 调用一次；重复登记即失败（M08 §7.1.1）。"""
    if _REG.energy is not None:
        raise ValueError("EnergyModel 已登记（只允许一个实现）")
    _check_protocol(model, ("estimate", "rtl_plan", "path_wh"), "EnergyModel")
    _REG.energy = model


def register_safety_hooks(hooks: SafetyHooks) -> None:
    """M09 调用一次；重复登记即失败。"""
    if _REG.hooks is not None:
        raise ValueError("SafetyHooks 已登记（只允许一个实现）")
    _check_protocol(hooks, ("on_stream_watchdog", "on_spawn", "on_remove", "apply_operator", "on_lease_event",
                            "on_gcs_beacon", "on_agent_liveliness", "matrix_verdict"), "SafetyHooks")
    _REG.hooks = hooks


def register_entity_block(kind: Literal["target"], name: str, owner: str,
                          fields: dict[str, tuple[np.dtype, tuple[int, ...]]]) -> None:
    """识别物实体表的列（M08-FR-216；M18 运动、判据、行为列，M19 感知列，M20 接力列）；字段格式同 `register_state_block`。
    "一字段一写者"：同一字段名只能由一个所有者登记（M08 保留 `active`、`target_no`、`roster_flags`）。"""
    if kind not in ENTITY_KINDS:
        raise ValueError(f"未知实体类型：{kind!r}（可选 {ENTITY_KINDS}）")
    blocks = _REG.entity_blocks.setdefault(kind, {})
    if name in blocks:
        raise ValueError(f"实体块重复登记：{kind}/{name}")
    reserved = {"active", "target_no", "roster_flags"}
    taken = {f: b.owner for b in blocks.values() for f in b.fields} | {f: "M08" for f in reserved}
    norm: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
    for f, spec in fields.items():
        if f in taken and not (f in reserved and owner == "M08"):
            raise ValueError(f"实体字段 {f!r} 已由 {taken[f]} 登记（一字段一写者，M08-FR-216）")
        dt, shape = spec
        norm[f] = (np.dtype(dt), tuple(int(x) for x in shape))
    blocks[name] = EntityBlockSpec(kind, name, owner, norm)


def register_power_model(frame_kind: int, model: Any) -> None:
    """按构型的功率模型（M08-FR-214）：每个 `frame_kind`（0 多旋翼、1 固定翼、2 VTOL）一次，重复登记失败。"""
    fk = int(frame_kind)
    if fk not in FRAME_KINDS.values():
        raise ValueError(f"未知构型 frame_kind={frame_kind}")
    if fk in _REG.power:
        raise ValueError(f"功率模型已登记：frame_kind={fk}")
    _check_protocol(model, ("power_w", "leg_power_w"), "PowerModel")
    _REG.power[fk] = model


def power_models() -> dict[int, Any]:
    return dict(_REG.power)


def register_profile_check(frame_types: frozenset[str] | set[str] | tuple[str, ...], check_id: str,
                           fn: Callable[[dict], Any], *, owner: str | None = None) -> None:
    """按构型登记机型自洽检查（M08-FR-207；M21 的 FW-1…FW-8、VT-1…VT-4）。`fn(doc)` 返回检查行
    `{id|check, value, range, pass, msg}`、其列表或布尔；ProfileTable 加载与会话克隆时执行。"""
    if any(c.check_id == check_id for c in _REG.profile_checks):
        raise ValueError(f"机型检查重名：{check_id!r}")
    _REG.profile_checks.append(ProfileCheckSpec(frozenset(frame_types), str(check_id), fn,
                                                owner or infer_owner(getattr(fn, "__module__", None))))


def register_session_loader(section: str, fn: Callable[..., Any], *, owner: str | None = None, order: int = 50) -> None:
    """会话配置（`awr.sandbox.snapshot.v1`）某段的装配函数 `fn(items, api) -> list[problem]`；按 order 升序执行
    （缺省次序：models 10、calendar/env 20、nests 30、vehicles 40（M08）、targets 50、tasks 60）。"""
    if section in _REG.session_loaders:
        raise ValueError(f"会话配置段重复登记：{section!r}")
    _REG.session_loaders[section] = SessionLoaderSpec(section, fn, owner or infer_owner(getattr(fn, "__module__", None)),
                                                      int(order))


def register_snapshot_exporter(section: str, fn: Callable[..., Any]) -> None:
    """`ctl/sim-core/snapshot{op: export}` 时导出某段配置的函数 `fn(api) -> list | dict | None`。"""
    if section in _REG.snapshot_exporters:
        raise ValueError(f"快照导出段重复登记：{section!r}")
    _REG.snapshot_exporters[section] = fn


def register_fleet_update_hook(fn: Callable[..., Any]) -> None:
    """`fleet/update` 的领域钩子 `fn(vehicle_id, slot, patch, ctx) -> None | {code, detail}`（M13 挂载与载荷、M21 挂载质量与
    MT 规则）；在 M08 的通用检查之后、生效之前按登记顺序调用，返回非零 code 即拒绝整条 update。"""
    if fn not in _REG.fleet_update_hooks:
        _REG.fleet_update_hooks.append(fn)


def register_setpoint_filter(fn: Callable[..., Any], *, owner: str | None = None) -> None:
    """velocity setpoint 入口的软围栏钩子（`keepout_guard`，AWR-04 §10.6、§14.3；M18 登记）：ingest 在步边界写入新鲜 setpoint
    之后调用 `fn(S, slots, modes, ctx)`，`slots` 为本步收到新 setpoint 的 slot（int64），`modes` 为对应会话的
    `keepout_guard`（`clamp`、`warn`、`off`；沙盒档缺省 clamp，d1_250 缺省 off）；钩子可原地修改 `S.vel_cmd`（NED，world 帧；
    body 帧的会话由钩子自行换算）并发事件（`uav.keepout_guard`、警告 KEEPOUT_CLAMPED）。钩子不得阻塞。"""
    if any(f is fn for _o, f in _REG.setpoint_filters):
        return
    _REG.setpoint_filters.append((owner or infer_owner(getattr(fn, "__module__", None)), fn))


def register_checkpoint_extension(name: str, save: Callable[[], bytes], restore: Callable[[bytes], Any], *,
                                  owner: str | None = None) -> None:
    """插件的 checkpoint 扩展段（d2-WP-07-to-M08 第 1 条、d2-INT-to-M08-M20 第 1 条；M20-FR-075 的 `tasking` 段）：捕获时以
    `x.<name>` 写入 checkpoint 元数据的 `ext`，恢复时在 `restore_ext` 的最后调用 `restore(blob)`。段名不得与 M08 的固有段
    （safety、env、metrics）重名；同名重复登记抛 ValueError。"""
    if name in _REG.ckpt_exts or name in ("safety", "env", "metrics"):
        raise ValueError(f"checkpoint 扩展段重名或保留：{name!r}")
    _REG.ckpt_exts[name] = CheckpointExtSpec(name, save, restore, owner or infer_owner(getattr(save, "__module__", None)))


def register_batch_output(name: str, fn: Callable[..., Any]) -> None:
    """批处理结束时写出 `<name>.json` 的函数 `fn(ctx) -> dict`（M20 `relay_stats`，M08-FR-221）。"""
    if name in _REG.batch_outputs or name in ("events", "summary"):
        raise ValueError(f"批处理输出重名或保留：{name!r}")
    _REG.batch_outputs[name] = fn


def energy_model() -> EnergyModel | None:
    return _REG.energy


def safety_hooks() -> SafetyHooks | None:
    return _REG.hooks


def _check_protocol(obj: Any, methods: tuple[str, ...], what: str) -> None:
    missing = [m for m in methods if not callable(getattr(obj, m, None))]
    if missing:
        raise TypeError(f"{what} 缺少方法：{missing}")


def describe(spec: StageSpec) -> dict[str, Any]:
    """构建日志与 meta.json 用的 stage 描述。"""
    return {"name": spec.name, "every": spec.every, "phase": spec.phase, "order": spec.order, "owner": spec.owner,
            "budget_core": spec.budget_core, "fidelity": int(spec.fidelity), "shard": list(spec.shard), "hz": spec.hz,
            "fn": f"{getattr(spec.fn, '__module__', '?')}.{getattr(spec.fn, '__qualname__', type(spec.fn).__name__)}",
            "coroutine": inspect.iscoroutinefunction(spec.fn)}
