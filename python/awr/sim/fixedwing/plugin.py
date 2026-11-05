"""M21 固定翼与 VTOL 的 sim-core 组合根入口（AWR-04 §5.3；ADR-096；M08-FR-210 至 FR-215、FR-219；d2-WP-02-to-M21）。

导入本模块即调用 `install()`（`configs/runtime.yaml` 的 sim-core `plugins:` 写 `awr.sim.fixedwing.plugin`）；对当前登记表幂等，登记：

- 状态块 `fw`（参与 checkpoint：空速、姿态、模式、盘旋参数与 VTOL 悬停计时，sim-core 重启接续时恢复）；
- stage `fw_kin`（order 086，50 Hz：d1_250 every 5 / phase 1，sandbox100 every 2 / phase 1，按档调度表 `stages/rates.py`）；
- 构型运动提供者（`register_motion_provider`，`frames = {1, 2}`，进程级：重复装配时替换上一次装配的 M21 提供者，其他提供者保留）；
- 功率模型 `register_power_model(1, …)`、`(2, …)`（固定翼与 VTOL，M08-FR-214）；
- 准入第 ⑥ 步 `m21.fw`（`frames = {1, 2}`：空速越界与盘旋半径下限 110、FW 速度控制帧 110、VTOL 悬停上限 524）；
- 查询 `fw/state`（并入 `GET .../vehicles/{vid}/state` 的 `fw`：mode、airspeed_mps、bank_rad、gamma_rad、hover_used_s）。
"""

from __future__ import annotations

from typing import Any

from .energy import FwPower
from .runtime import FW_BLOCK, FW_FIELDS, FwRuntime

__all__ = ["install", "runtime"]

_KEY = "fixedwing"                                     # the runtime lives in registry.plugins of the table it was installed into


def _replace_provider(rt: FwRuntime) -> None:
    from awr.sim.core import command as CMD

    olds = {k: CMD.fw_provider(k) for k in (1, 2)}
    if all(v is None for v in olds.values()):
        CMD.register_motion_provider(rt)
        return
    keep_ops = CMD.motion_providers()
    keep_fw = {id(v): v for v in olds.values() if v is not None and getattr(v, "name", "") != rt.name}
    CMD.reset_motion_providers()
    for p in {id(p): p for p in keep_ops.values()}.values():
        CMD.register_motion_provider(p)
    for p in keep_fw.values():
        CMD.register_motion_provider(p)
    CMD.register_motion_provider(rt)


def install() -> FwRuntime:
    from awr.sim.fleet.stages import registry as R

    reg = R.registry()
    rt = reg.plugins.get(_KEY)
    if rt is not None:
        return rt
    rt = FwRuntime()
    if FW_BLOCK not in reg.blocks:
        R.register_state_block(FW_BLOCK, "M21", dict(FW_FIELDS))
    R.register_stage("fw_kin", order=86, owner="M21", hz=50.0, phase_slot=1, fidelity=R.Fidelity.ALL)(rt.stage)

    def rt_fn() -> Any:
        return rt

    for fk in (1, 2):
        if fk not in reg.power:
            R.register_power_model(fk, FwPower(fk, rt_fn))
    R.register_admission_check(6, "m21.fw", rt.admission, owner="M21", frames={1, 2})
    if "fw/state" not in reg.queries:
        R.register_query("fw/state", rt.q_state)
    _replace_provider(rt)
    reg.plugins[_KEY] = rt
    return rt


def runtime() -> FwRuntime | None:
    from awr.sim.fleet.stages import registry as R

    return R.registry().plugins.get(_KEY)


install()
