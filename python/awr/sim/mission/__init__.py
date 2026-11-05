"""任务引擎、生成器、原语、剧本加载器与剧本导演（M10；AWR-03 §4.3）。

插件入口（M10 §7.4.1）：sim-core 组合根按 `AWR_PLUGINS` 导入本包时调用 `install()`，登记

- 状态块 `mission`（`tracker.STATE_FIELDS`，含 M08 tap 读取的 `mission_item`、`track_state`）；
- stage：`mission`（027/2/1，125 Hz，奇数 tick，写 `tr_x/tr_v/tr_a/yaw_sp`；ADR-065）、`mission_engine`（150/25/8）、`coverage`（155/50/1，ADR-073）、
  `director`（160/25/8）；
- 运动提供者 `m10.follow_path`、`m10.orbit`、`m10.goto_route`（M08-FR-086）与细校验执行者（M08-FR-089）；
- 度量 `missions_done`、`mission_progress`、`facade_coverage`、`area_coverage`、`formation_err_rms_m`、`agl_min_m`、
  `agl_rms_err_m`、`link_quality_min`（M08-FR-088）；
- 查询 `mission/list`、`mission/detail`、`mission/create`、`mission/preview`、`mission/preview/get`、`mission/control`、
  `scenario/result`（`ctl/sim-core/query`）；
- 命令 `mission/start`、`mission/pause`、`mission/resume`、`mission/abort`（`ctl/sim-core/cmd`，`args.mid`；M08
  `register_command_handler`，验签、角色与席位由 CommandEngine 完成）。

`install()` 对当前登记表幂等；进程内测试在 `isolated_registry()` 中显式调用。环境变量 `AWR_M10_AUTOINSTALL=0` 时 import
不自动登记（M10 自己的单测用）。
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["install", "installed_runtime", "standby_warm"]

_KEY = "mission"                                     # the runtime lives in registry.plugins of the table it was installed into


def install() -> Any:
    """登记 M10 的全部扩展点；返回该登记表对应的 M10Runtime。"""
    from awr.sim.core import command as CMD
    from awr.sim.core import metrics as MET
    from awr.sim.fleet.stages import registry as R

    from .metrics import provided
    from .queries import Queries
    from .runtime import M10Runtime
    from .tracker import STATE_FIELDS, FollowPathProvider, GotoRouteProvider, OrbitProvider

    reg = R.registry()
    rt = reg.plugins.get(_KEY)
    if rt is not None and "mission" in reg.blocks:
        return rt
    rt = M10Runtime()
    R.register_state_block("mission", owner="M10", fields=STATE_FIELDS)
    # phase 1（奇数 tick）：l1、contact、tap 在偶数 tick（phase 0）。跟踪器在 l1 之前一个 tick 写出同一参考（τ 推进同为
    # 8 ms，机体位置在奇数 tick 不变），l1 看到的参考序列不变；125 Hz 的两组重核错开到奇偶 tick，单步峰值下降（ADR-065）
    R.register_stage("mission", every=2, phase=1, order=27, owner="M10", budget_core=0.012,
                     writes=("tr_x", "tr_v", "tr_a", "yaw_sp"))(rt.st_tracker)
    R.register_stage("mission_engine", every=25, phase=8, order=150, owner="M10", budget_core=0.003)(rt.st_engine)
    # coverage 相位 13 → 23（ADR-070，与 mission_guard 错开）→ 1（ADR-073：成对推进下按 tick 对错峰，(23, 24) 与 fleet_guard.3、
    # cmd_watch 叠加是最重的几类 tick 对之一，(1, 2) 是最轻的几类之一）
    R.register_stage("coverage", every=50, phase=1, order=155, owner="M10", budget_core=0.005)(rt.st_coverage)
    R.register_stage("director", every=25, phase=8, order=160, owner="M10", budget_core=0.002)(rt.st_director)
    q = Queries(rt)
    rt.queries = q
    for name, fn in q.ops().items():
        R.register_query(name, fn)
    provs = CMD.motion_providers()
    stale = {op for op, p in provs.items() if str(getattr(p, "name", "")).startswith("m10.")}
    if stale:
        keep = {op: p for op, p in provs.items() if op not in stale}
        CMD.reset_motion_providers()
        for p in {id(p): p for p in keep.values()}.values():
            CMD.register_motion_provider(p)
    for p in (FollowPathProvider(rt), OrbitProvider(rt), GotoRouteProvider(rt)):
        CMD.register_motion_provider(p)
    CMD.register_fine_checker(rt.fine_check)
    # `mission/start|pause|resume|abort`：网关作为 Command 发到 ctl/sim-core/cmd（M08 register_command_handler）
    for op in ("mission/start", "mission/pause", "mission/resume", "mission/abort"):
        if op not in reg.commands:
            R.register_command_handler(op, q.command_handler(op.split("/", 1)[1]), owner="M10")
    for name, fn in provided(rt).items():
        spec = {m.name: m for m in MET.list_metrics()}.get(name)
        if spec is not None and spec.owner == "M10":
            MET.unregister_metric(name)
        if spec is None or spec.owner == "M10":
            MET.register_metric(name, fn, owner="M10")
    reg.plugins[_KEY] = rt
    _warm_kernels()
    return rt


def _warm_kernels() -> None:
    """跟踪核的 numba 预热（含运行期的只读视图签名）：装配即预热，使 sim-core 在 ready 之前完成编译或读缓存，
    运行期不在主循环内 JIT（D1 验收第 1 轮 4.1；M10-NFR-015）。同一进程只做一次。"""
    global _WARMED
    if _WARMED:
        return
    _WARMED = True
    try:
        from awr.sim.planning import kernels_track as KT

        if KT.HAVE_NUMBA:
            KT.warmup()
    except Exception:  # 预热失败不阻止装配：运行期按需编译（只损失时延）
        import logging

        logging.getLogger("awr.sim.mission").exception("m10 numba warmup failed")


_WARMED = False


def standby_warm() -> None:
    """sim-core 热备用进程的预热钩子（M08-FR-005，AWR-03 ADR-070）：导入生成器模块、构造剧本校验器并做一次空校验，使接替后
    的剧本加载不再承担这部分冷启动（约 0.15–0.25 s）。由 M08 在组合根装配后按插件名调用，不改变登记状态。"""
    from . import generators, scenario_loader

    del generators
    scenario_loader.warm_validators()


def installed_runtime() -> Any:
    from awr.sim.fleet.stages import registry as R

    return R.registry().plugins.get(_KEY)


if os.environ.get("AWR_M10_AUTOINSTALL", "1") != "0":
    install()
