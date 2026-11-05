"""M13 组合根入口（M13-FR-004；M13 §6.2、§7.1；M08 §7.1.1；AWR-10 §3.3 规则 1）。

导入本模块即调用 `install()`（`configs/runtime.yaml` 的 sim-core `plugins:` 写 `awr.sim.sensors.plugin`；在该条目改名之前，
包 `awr.sim.sensors` 只在 `AWR_PLUGINS` 列出它时转而导入本模块，见包文档）。`install()` 对当前登记表幂等，登记：

- 状态块 `sensors`（`block.FIELDS`）与 `sensor_targets`（Mock 目标表，前 64 行），均参与 checkpoint；
- stage `sensors`（every 5、phase 2、order 100，budget 0.010 核）；
- 慢任务 `m13.sensor_pose`（SensorPose48 10 Hz 分片打包，自计时）与 `m13.noisy_obs`（2 Hz【仿真】白噪声观测）；
- `GET /api/fleet/profiles/{id}` 的 `sensors{}` 描述函数（M08 `profiles.register_sensor_describer`，每进程一次）；
- M08 的 roster 传感器组钩子与 state_ext 钩子（`register_state_ext_hook`，GNSS、云台、IMU 字段）；M08 若提供
  `register_estimate_hook`，同时登记 `conf_expected`。

D2（M13 §15.7）：状态块 `payload`（`payload.PAYLOAD_FIELDS`，参与 checkpoint）；慢任务 `m13.payload_pub`（2 Hz）、`m13.nav_pub`
（10 Hz，订阅时）、`m13.lidar_scan`（≤ 2 Hz，订阅时，每轮 ≤ 300 µs）；机体命令 `gimbal`、`zoom`、`sensor/mode` 的处理函数经
`register_command_handler` 登记（M08-FR-218；op 已被他者登记时跳过并记日志），也可由 M08 直接调用 `vehicle_command()`。

`runtime()` 返回当前登记表对应的 SensorRuntime（M10、M14 经它调用云台、目标、传感器开关与几何 API；M19、M20 经它读取载荷与引导）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .block import BLOCK, FIELDS, TARGET_BLOCK
from .detector import TARGET_FIELDS
from .payload import PAYLOAD_BLOCK, PAYLOAD_FIELDS
from .runtime import SensorRuntime
from .stage import EVERY, ORDER, PHASE, make_stage

__all__ = ["install", "installed", "runtime", "uninstall", "vehicle_command"]

log = logging.getLogger("awr.sim.sensors")
_KEY = "sensors"                                     # the runtime lives in registry.plugins of the table it was installed into
_DESCRIBER = [False]

WRITES = tuple(f"{BLOCK}.{f}" for f in FIELDS) + tuple(f"{PAYLOAD_BLOCK}.{f}" for f in PAYLOAD_FIELDS)


def _registry():
    from awr.sim.fleet.stages import registry as R

    return R


def install() -> SensorRuntime:
    R = _registry()
    reg = R.registry()
    rt = reg.plugins.get(_KEY)
    if rt is not None:
        return rt
    rt = SensorRuntime()
    R.register_state_block(BLOCK, "M13", dict(FIELDS))
    R.register_state_block(TARGET_BLOCK, "M13", dict(TARGET_FIELDS))
    R.register_state_block(PAYLOAD_BLOCK, "M13", dict(PAYLOAD_FIELDS))
    R.register_stage("sensors", every=EVERY, phase=PHASE, order=ORDER, owner="M13", fidelity=R.Fidelity.ALL,
                     budget_core=0.010, writes=WRITES)(make_stage(rt))
    R.register_slow_task("m13.sensor_pose", rt.packer.run_slow, budget_us=200)
    R.register_slow_task("m13.noisy_obs", rt.obs.run_slow, period_sim_s=0.5, budget_us=300)
    R.register_slow_task("m13.payload_pub", rt.payload_pub.run_slow, budget_us=300)
    R.register_slow_task("m13.nav_pub", rt.nav.run_slow, budget_us=200)
    R.register_slow_task("m13.lidar_scan", rt.scanner.run_slow, budget_us=300)
    _register_commands(rt)
    _register_optional_hooks(rt)
    reg.plugins[_KEY] = rt
    _warm_kernels()
    return rt


_WARMED = [False]


def _warm_kernels() -> None:
    """云台融合核的 numba 预热（运行期签名：状态块 C 连续数组、ENU 只读视图；M08-FR-005，ADR-070）。装配即预热，
    sim-core 在 ready 之前完成编译或读缓存；同一进程只做一次。失败只记日志（运行期按需编译，只损失时延）。"""
    if _WARMED[0]:
        return
    _WARMED[0] = True
    try:
        from . import cbrng, kernels_gimbal, kernels_gnss
        from .lidar import scan_rt

        kernels_gimbal.warmup()
        kernels_gnss.warmup()
        scan_rt.warmup()
        cbrng.warmup()
    except Exception:
        log.exception("m13 numba warmup failed")


def _register_commands(rt: SensorRuntime) -> None:
    """`uav/{id}/cmd/{gimbal,zoom,sensor/mode}`（M08-FR-218：经 `register_command_handler` 交 M13）。"""
    from .commands import OPS, handle_message

    R = _registry()
    reg = R.registry()
    for op in OPS:
        if op in reg.commands:
            log.warning("command handler %s already registered by %s", op, reg.commands[op].owner)
            continue

        def fn(msg: dict, apply_tick: int, ctx: Any = None, _op: str = op) -> dict:
            return handle_message(rt, _op, msg, apply_tick, ctx)

        fn.__module__ = __name__
        R.register_command_handler(op, fn, owner="M13", need_seat=False)
    if "payload/state" not in reg.queries:  # `ctl/sim-core/query`：一架机的 awr.uav.payload.v1（M08 `vehicle_sensors` 的回退）
        def payload_state(m: dict, ctx: Any = None) -> dict:
            vid = str((m.get("args") or {}).get("id", ""))
            s = rt.slot_of(vid)
            item = rt.payload.message(s, rt.tick_ns()) if s >= 0 else None
            return {"v": 1, "code": 0 if item is not None else 107, "item": item}

        R.register_query("payload/state", payload_state)


def vehicle_command(op: str, vehicle_id: str, args: dict, apply_tick: int, ctx: Any = None) -> dict:
    """M08 机体命令分派的直接入口（op ∈ gimbal、zoom、sensor/mode）：返回 `{status, code, detail?, result?}`。"""
    from .commands import vehicle_command as _vc

    rt = runtime()
    if rt is None:
        return {"status": "rejected", "code": 211, "detail": {"detail": "M13_NOT_INSTALLED"}}
    return _vc(rt, op, vehicle_id, args, apply_tick, ctx)


def _register_optional_hooks(rt: SensorRuntime) -> None:
    if not _DESCRIBER[0]:
        try:
            from awr.sim.fleet import profiles

            from .describe import describe

            profiles.register_sensor_describer(describe)
            _DESCRIBER[0] = True
        except Exception:
            log.warning("sensor describer not registered", exc_info=True)
    R = _registry()
    ext = getattr(R, "register_state_ext_hook", None)  # M13-to-M08 第 2 条：GNSS、云台、IMU 字段并入 state_ext
    if callable(ext):
        try:
            ext(rt.state_ext_fields, owner="M13")
        except Exception:
            log.warning("state_ext hook not registered", exc_info=True)
    hook = getattr(R, "register_estimate_hook", None)
    if callable(hook):
        try:
            hook("conf_expected", _conf_expected)
        except Exception:
            log.warning("estimate hook not registered", exc_info=True)
    for modname in ("awr.sim.core.roster", "awr.sim.fleet.profiles"):
        try:
            import importlib

            mod = importlib.import_module(modname)
        except Exception:
            continue
        fn = getattr(mod, "register_sensor_rig", None)
        if callable(fn):
            try:
                fn(_rig_entries_for_profile)
            except Exception:
                log.warning("roster sensor rig hook not registered", exc_info=True)
            break


def _rig_entries_for_profile(profile_id: str, names: list[str] | None = None) -> list[dict]:
    """roster `sensors[]`：names 缺省时，D2 机型取沙盒实例的缺省挂载（D1 机型为全部有视场的传感器）；机队实例应传入
    `SensorRuntime.mounted_names(vehicle_id)`（M08）。"""
    from .describe import dir_for_profile
    from .spec import rig_entries, rig_for_model

    d = dir_for_profile(profile_id)
    if d is None:
        return []
    rig = rig_for_model(d.name, d.parent)
    if names is None and rig.d2:
        rt = runtime()
        sb = rt._is_sandbox() if rt is not None else True
        names = [s.name for s in rig.subset(rig.default_mask(sandbox=sb)).specs]
    return rig_entries(rig, names)


def _conf_expected(profile_id: str, p_uav: Any, p_tgt: Any, capability: str, env: Any = None, t_sim_ns: int = 0) -> float | None:
    """M08 estimate 的 `conf_expected`（M13-FR-045；M14 §6.10.5）：机型上声明该能力的检测器的期望 P_d（FOV 视为 1）。"""
    from awr.sim.core.awareness import get_awareness

    from .describe import dir_for_profile
    from .detector import expected_pd
    from .spec import rig_for_model

    d = dir_for_profile(profile_id)
    if d is None:
        return None
    aw = get_awareness()
    if not aw.visibility:
        env = None
    los: Any = 1.0
    if aw.los:
        w = getattr(getattr(runtime(), "ctx", None), "world", None)
        if w is not None and callable(getattr(w, "los_batch", None)):
            try:
                los = np.asarray(w.los_batch(np.asarray(p_uav, np.float64).reshape(1, 3),
                                             np.asarray(p_tgt, np.float64).reshape(1, 3)), np.float64)
            except Exception:
                los = 1.0
    for s in rig_for_model(d.name, d.parent).specs:
        if s.detector is not None and s.detector.capability == capability:
            return float(expected_pd(s.detector, p_uav, p_tgt, env, t_sim_ns, los=los)[0])
    return None


def runtime() -> SensorRuntime | None:
    R = _registry()
    return R.registry().plugins.get(_KEY)


def installed() -> dict[int, SensorRuntime]:
    """The runtime installed into the current registry, keyed by that registry's id (empty when none)."""
    reg = _registry().registry()
    rt = reg.plugins.get(_KEY)
    return {} if rt is None else {id(reg): rt}


def uninstall() -> None:
    """测试用：从当前登记表移除本模块登记的 stage、状态块与慢任务。"""
    R = _registry()
    reg = R.registry()
    rt = reg.plugins.pop(_KEY, None)
    if rt is None:
        return
    reg.stages[:] = [s for s in reg.stages if s.name != "sensors"]
    reg.blocks.pop(BLOCK, None)
    reg.blocks.pop(TARGET_BLOCK, None)
    reg.blocks.pop(PAYLOAD_BLOCK, None)
    for op in ("gimbal", "zoom", "sensor/mode"):
        h = reg.commands.get(op)
        if h is not None and h.owner == "M13":
            reg.commands.pop(op, None)
    reg.queries.pop("payload/state", None)
    reg.slow[:] = [t for t in reg.slow if not t.name.startswith("m13.")]
    hooks = getattr(reg, "state_ext_hooks", None)
    if hooks is not None:
        hooks[:] = [h for h in hooks if h[0] != "M13"]


install()
