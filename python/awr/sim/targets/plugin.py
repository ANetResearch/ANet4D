"""M18 组合根入口（M18-FR-001、FR-018 至 FR-022、FR-027；M18-M19 PRD §9.1；M08 §15.3.1 扩展点；d2-WP-02-to-M18）。

导入本模块即调用 `install()`（sim-core 的 `plugins:` 写 `awr.sim.targets.plugin`；`AWR_PLUGINS` 列出包名 `awr.sim.targets` 时包
初始化也会转而导入本模块）。`install()` 对当前登记表幂等，登记：

- 实体块 `targets`（kind `target`，block.FIELDS，参与 checkpoint；M08 据此装配 `tap_targets` 发布 TargetLite32）；
- stage `targets`（10 Hz，order 170，两档时钟由 M08 按档调度表换算；预算 `targets` 0.004 核）；
- 命令 `target/add`、`target/update`、`target/remove`（需席位，ADR-016 准入；写入于下一 tick 生效）；
- 查询（`ctl/sim-core/target` 与 `ctl/sim-core/query`）：`target/list`、`target/get`、`target/state`、`target/suggested_radius`、
  `target/keepout`、`target/classes`；
- 会话配置段 `targets`（装配 order 50 与导出）；软围栏 `register_setpoint_filter`；慢任务 `m18.detail`（2 Hz【仿真】发布
  `state/sim-core/target_detail`）。
`runtime()` 返回当前登记表对应的 TargetRuntime（M19、M20 经它取 `view()`、`keepout()`、`los`）。
"""

from __future__ import annotations

import logging
from typing import Any

from .block import BLOCK, FIELDS, OWNER
from .keepout_guard import KeepoutGuard
from .model import TargetError, class_defaults
from .runtime import TargetRuntime

__all__ = ["install", "runtime", "uninstall"]

log = logging.getLogger("awr.sim.targets")
_KEY = "targets"                                     # the runtime lives in registry.plugins of the table it was installed into
ORDER = 170
HZ = 10.0
DETAIL_PERIOD_S = 0.5


def _registry():
    from awr.sim.fleet.stages import registry as R

    return R


def runtime(create: bool = True) -> TargetRuntime | None:
    R = _registry()
    reg = R.registry()
    rt = reg.plugins.get(_KEY)
    if rt is not None and BLOCK not in (reg.entity_blocks.get("target") or {}):
        rt = None
    if rt is None and create:
        return install()
    return rt


QUERIES = ("target/list", "target/get", "target/state", "target/suggested_radius", "target/keepout", "target/classes")
COMMANDS = ("target/add", "target/update", "target/remove")


def uninstall() -> None:
    """从当前登记表撤销本插件的全部登记（测试用；sim-core 进程内不调用）。"""
    R = _registry()
    reg = R.registry()
    rt = reg.plugins.pop(_KEY, None)
    if rt is None:
        return
    reg.stages[:] = [s for s in reg.stages if s.name != "targets"]
    (reg.entity_blocks.get("target") or {}).pop(BLOCK, None)
    for op in COMMANDS:
        h = reg.commands.get(op)
        if h is not None and h.owner == OWNER:
            reg.commands.pop(op, None)
    for q in QUERIES:
        reg.queries.pop(q, None)
    reg.session_loaders.pop("targets", None)
    reg.snapshot_exporters.pop("targets", None)
    reg.setpoint_filters[:] = [f for f in reg.setpoint_filters if f[0] != OWNER]
    reg.slow[:] = [t for t in reg.slow if not t.name.startswith("m18.")]


def _rej(e: TargetError) -> dict:
    return {"status": "rejected", "code": e.code, "detail": e.detail}


def _target_arg(args: dict) -> dict:
    t = args.get("target")
    if isinstance(t, dict):
        return t
    return {k: v for k, v in args.items() if k not in ("id_key",)} if "class" in args else {}


_WARMED = [False]


def _warm_kernels() -> None:
    """识别物运动与判据核（`kernels_motion`）及 M13 计数器 RNG 外积核的 numba 预热：装配即按运行期签名编译或读缓存
    （M08-FR-005），sim-core 在 ready 之前完成，主循环内不编译。同一进程只做一次；失败只记日志。"""
    if _WARMED[0]:
        return
    _WARMED[0] = True
    try:
        from awr.sim.fleet.stages import tap_targets
        from awr.sim.sensors import cbrng

        from . import kernels_los, kernels_motion

        kernels_motion.warmup()
        kernels_los.warmup()
        cbrng.warmup()
        tap_targets.warmup()
    except Exception:
        log.exception("m18 numba warmup failed")


def install() -> TargetRuntime:
    R = _registry()
    reg = R.registry()
    rt = reg.plugins.get(_KEY)
    if rt is not None and BLOCK in (reg.entity_blocks.get("target") or {}) and "target/add" in reg.commands:
        return rt   # 同一登记表，且登记内容还在（测试可能撤销过）
    _warm_kernels()
    rt = TargetRuntime()
    R.register_entity_block("target", BLOCK, OWNER, dict(FIELDS))

    def stage(S: Any, ctx: Any) -> None:
        rt.stage(S, ctx)

    stage.__module__ = __name__
    R.register_stage("targets", hz=HZ, order=ORDER, owner=OWNER, writes=tuple(f"ent.target.{f}" for f in FIELDS))(stage)

    def add(msg: dict, apply_tick: int, ctx: Any = None) -> dict:
        try:
            tid, cfg, warns = rt.add(ctx, _target_arg(msg.get("args") or {}))
        except TargetError as e:
            return _rej(e)
        out: dict[str, Any] = {"status": "accepted", "result": {"id": tid, "cfg": cfg}}
        if warns:
            out["warnings"] = warns
        return out

    def update(msg: dict, apply_tick: int, ctx: Any = None) -> dict:
        a = msg.get("args") or {}
        try:
            cfg, warns = rt.update(ctx, str(a.get("id", "")), a.get("patch") if isinstance(a.get("patch"), dict) else {})
        except TargetError as e:
            return _rej(e)
        out: dict[str, Any] = {"status": "accepted", "result": {"id": cfg["id"], "cfg": cfg}}
        if warns:
            out["warnings"] = warns
        return out

    def remove(msg: dict, apply_tick: int, ctx: Any = None) -> dict:
        a = msg.get("args") or {}
        try:
            rt.remove(ctx, str(a.get("id", "")))
        except TargetError as e:
            return _rej(e)
        return {"status": "accepted", "result": {"id": str(a.get("id"))}}

    for op, fn in (("target/add", add), ("target/update", update), ("target/remove", remove)):
        fn.__module__ = __name__
        R.register_command_handler(op, fn, owner=OWNER)

    def q_list(m: dict, ctx: Any = None) -> dict:
        a = m.get("args") or {}
        rt.bind(ctx)
        items = []
        for T in rt.ordered():
            if a.get("class") and T.cfg.cls != a["class"]:
                continue
            if a.get("state") and a["state"] not in (T.aware, str(T.aware), rt_state_name(T.aware), rt_state_name(T.aware).lower()):
                continue
            items.append(rt.cfg_dict(T))
        return {"v": 1, "code": 0, "items": items, "rev": int(rt.rev)}

    def q_get(m: dict, ctx: Any = None) -> dict:
        rt.bind(ctx)
        tid = str((m.get("args") or {}).get("id", ""))
        T = rt._by_id(tid)
        if T is None:
            return {"v": 1, "code": 543, "detail": {"id": tid, "why": "TARGET_NOT_FOUND"}}
        return {"v": 1, "code": 0, "item": rt.cfg_dict(T)}

    def q_state(m: dict, ctx: Any = None) -> dict:
        rt.bind(ctx)
        tid = str((m.get("args") or {}).get("id", ""))
        T = rt._by_id(tid)
        if T is None:
            return {"v": 1, "code": 543, "detail": {"id": tid, "why": "TARGET_NOT_FOUND"}}
        rt._roster = getattr(ctx, "roster", None)
        return {"v": 1, "code": 0, "item": rt.detail(T, int(getattr(ctx, "t_ns", 0)), rt.env(ctx))}

    def q_suggest(m: dict, ctx: Any = None) -> dict:
        rt.bind(ctx)
        a = m.get("args") or {}
        try:
            return {"v": 1, "code": 0, "item": rt.suggested(ctx, str(a.get("id", "")), a.get("model_id"), a.get("light"), a.get("bg"))}
        except TargetError as e:
            return {"v": 1, "code": e.code, "detail": e.detail}

    def q_keepout(m: dict, ctx: Any = None) -> dict:
        rt.bind(ctx)
        a = m.get("args") or {}
        margin = float(a.get("margin_m", 20.06))
        ids = [str(a["id"])] if a.get("id") else rt.ids()
        out = []
        for tid in ids:
            out.extend(v.to_wire(f"{tid}/{v.vol_idx}") for v in rt.keepout(tid, margin_m=margin, mode=a.get("mode")))
        return {"v": 1, "code": 0, "items": out}

    def q_classes(m: dict, ctx: Any = None) -> dict:
        return {"v": 1, "code": 0, "item": class_defaults()}

    for name, fn in (("target/list", q_list), ("target/get", q_get), ("target/state", q_state), ("target/suggested_radius", q_suggest),
                     ("target/keepout", q_keepout), ("target/classes", q_classes)):
        if name not in reg.queries:
            R.register_query(name, fn)

    def loader(items: list, api: Any) -> list:
        return rt.import_targets(api.ctx, list(items or []))

    def exporter(api: Any) -> list:
        rt.bind(api.ctx)
        return rt.export_targets()

    loader.__module__ = exporter.__module__ = __name__
    R.register_session_loader("targets", loader, owner=OWNER, order=50)
    R.register_snapshot_exporter("targets", exporter)
    guard = KeepoutGuard(rt)
    rt.guard = guard  # type: ignore[attr-defined]
    R.register_setpoint_filter(guard, owner=OWNER)
    R.register_slow_task("m18.detail", rt.publish_detail, period_sim_s=DETAIL_PERIOD_S, budget_us=800)
    reg.plugins[_KEY] = rt
    return rt


def rt_state_name(code: int) -> str:
    return ("UNAWARE", "SUSPICIOUS", "REACTING", "ALERTED", "CALMING")[int(code)] if 0 <= int(code) <= 4 else "UNAWARE"


install()
