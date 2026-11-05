"""M19 组合根入口（M19-FR-011 至 FR-020、FR-025；M18-M19 PRD §9.1；M08 §15.3.1 扩展点；d2-WP-02-to-M19、d2-WP-05-to-M19）。

导入本模块即调用 `install()`（sim-core 的 `plugins:` 写 `awr.sim.perception.plugin`；`AWR_PLUGINS` 列出包名 `awr.sim.perception`
时包初始化也会转而导入本模块；须在 `awr.sim.targets` 之后装配）。`install()` 对当前登记表幂等，登记：

- 实体块列 `perception`（kind `target`：`tgt_best_level`、`tgt_captured`、`tgt_best_sensor`、`tgt_capture_uav_no`，一字段一写者）；
- stage `perception`（order 175；按档调度表：d1_250 每 2 tick、sandbox100 每 tick；片内轮转使每机 5 Hz；预算 `perception` 0.025 核）；
- 查询（`ctl/sim-core/query`）：`perception/vehicle`（SB-38）、`perception/detections`（SB-39）、`perception/tracks`（SB-66，P1）、
  `perception/feasibility`（SB-67，P1）、`perception/capture`；
- 度量 `ray_rate_hz`、`rays_dropped`（`state/sim-core/perf` 自报，d2-WP-02-to-M19 第 2 条）；
- 装配时预热成像感知的 numba 核（`kernels.warmup`，D2-FX-R2-perception）；
- D2.1（WP-19b，M19-FR-034、FR-035）：查询 `perception/quant`（SB-73）、`perception/field`（SB-74）、`perception/volumes`（SB-75）、
  `perception/samples`（SB-78，P1）；慢任务 `m19.field`（10 Hz【墙钟】检查，按订阅生成剖面、感知体表与逐对读数，`field.py`）。
`runtime()` 返回当前登记表对应的 PerceptionRuntime（M20 经它调用 `set_requirement`、`capture_state`、`track`、`set_cue`）。
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from .runtime import BLOCK, FIELDS, OWNER, PerceptionRuntime

__all__ = ["QUERIES", "install", "runtime", "uninstall"]

log = logging.getLogger("awr.sim.perception")
_KEY = "perception"                                     # the runtime lives in registry.plugins of the table it was installed into
ORDER = 175
HZ = 125.0          # d1_250 每 2 tick；sandbox100 由按档调度表给出每 tick
QUERIES = ("perception/vehicle", "perception/detections", "perception/tracks", "perception/feasibility", "perception/capture",
           "perception/quant", "perception/field", "perception/volumes", "perception/samples")
FIELD_TASK = "m19.field"
FIELD_TASK_PERIOD_S = 0.1
METRICS = ("ray_rate_hz", "rays_dropped")


def _registry():
    from awr.sim.fleet.stages import registry as R

    return R


def runtime(create: bool = True) -> PerceptionRuntime | None:
    R = _registry()
    rt = R.registry().plugins.get(_KEY)
    if rt is None and create:
        return install()
    return rt


service = runtime


def uninstall() -> None:
    """从当前登记表撤销本插件的全部登记（测试用；sim-core 进程内不调用）。"""
    R = _registry()
    reg = R.registry()
    rt = reg.plugins.pop(_KEY, None)
    if rt is None:
        return
    reg.stages[:] = [s for s in reg.stages if s.name != "perception"]
    (reg.entity_blocks.get("target") or {}).pop(BLOCK, None)
    for q in QUERIES:
        reg.queries.pop(q, None)
    reg.slow[:] = [t for t in reg.slow if t.name != FIELD_TASK]
    trt = rt._bound_trt
    if trt is not None and getattr(trt, "capture_fn", None) == rt.capture_detail:
        trt.capture_fn = None


def _metrics() -> None:
    from awr.sim.core import metrics as M

    have = {m.name for m in M.list_metrics()}

    def ray_rate(**_: Any) -> float:
        rt = runtime(create=False)
        return float(rt.rays_s) if rt is not None else 0.0

    def dropped(**_: Any) -> float:
        rt = runtime(create=False)
        return float(rt.stats["deferred"]) if rt is not None else 0.0

    for name, fn in (("ray_rate_hz", ray_rate), ("rays_dropped", dropped)):
        if name not in have:
            with contextlib.suppress(ValueError):
                M.register_metric(name, fn, owner=OWNER)


def _q(rt: PerceptionRuntime, ctx: Any) -> None:
    rt.bind(ctx)


_WARMED = [False]


def _warm_kernels() -> None:
    """成像感知核（`kernels.py`）的 numba 预热：装配即按运行期签名编译或读缓存（M08-FR-005；与 M13 `_warm_kernels` 同一做法），
    sim-core 在 ready 之前完成，主循环内不编译。同一进程只做一次；失败只记日志（运行期按需编译，只损失首次时延）。"""
    if _WARMED[0]:
        return
    _WARMED[0] = True
    try:
        from . import kernels, tracks

        kernels.warmup()
        tracks.warmup()
    except Exception:
        log.exception("m19 numba warmup failed")


def install() -> PerceptionRuntime:
    R = _registry()
    reg = R.registry()
    rt = reg.plugins.get(_KEY)
    if rt is not None and BLOCK in (reg.entity_blocks.get("target") or {}) and "perception/vehicle" in reg.queries:
        return rt
    _warm_kernels()
    rt = PerceptionRuntime()
    if BLOCK not in (reg.entity_blocks.get("target") or {}):
        R.register_entity_block("target", BLOCK, OWNER, dict(FIELDS))

    def stage(S: Any, ctx: Any) -> None:
        rt.stage(S, ctx)

    stage.__module__ = __name__
    if not any(s.name == "perception" for s in reg.stages):
        R.register_stage("perception", hz=HZ, order=ORDER, owner=OWNER, writes=tuple(f"ent.target.{f}" for f in FIELDS))(stage)

    def q_vehicle(m: dict, ctx: Any = None) -> dict:
        _q(rt, ctx)
        uid = str((m.get("args") or {}).get("id", ""))
        item = rt.vehicle(uid)
        if item is None:
            return {"v": 1, "code": 107, "detail": {"id": uid, "why": "NO_PERCEPTION_FRAME"}}
        return {"v": 1, "code": 0, "item": item}

    def q_detections(m: dict, ctx: Any = None) -> dict:
        a = m.get("args") or {}
        from .params import LEVEL_INDEX

        lvl = a.get("level_min")
        code, rep = rt.ring.page(int(a.get("since_seq") or 0), uav=a.get("uav"), target=a.get("target"), sensor=a.get("sensor"),
                                 level_min=LEVEL_INDEX.get(str(lvl), 0) if lvl is not None else 0, limit=int(a.get("limit") or 500))
        if code:
            return {"v": 1, "code": code, "detail": rep}
        return {"v": 1, "code": 0, "item": rep}

    def q_tracks(m: dict, ctx: Any = None) -> dict:
        _q(rt, ctx)
        a = m.get("args") or {}
        t = int(getattr(ctx, "t_ns", 0)) * 1e-9 if ctx is not None else 0.0
        return {"v": 1, "code": 0, "item": {"items": rt.tracks_wire(t, a.get("target"))}}

    def q_feasibility(m: dict, ctx: Any = None) -> dict:
        _q(rt, ctx)
        from .service import feasibility_query
        from .specs import PerceptionError

        try:
            return {"v": 1, "code": 0, "item": feasibility_query(rt, ctx, m.get("args") or {})}
        except PerceptionError as e:
            return {"v": 1, "code": e.code, "detail": e.detail}

    def q_capture(m: dict, ctx: Any = None) -> dict:
        _q(rt, ctx)
        a = m.get("args") or {}
        cs = rt.capture_state(str(a.get("target", "")), str(a.get("level_req") or "R"))
        return {"v": 1, "code": 0, "item": {"captured": cs.captured, "by": list(cs.by), "since_ns": cs.since_ns,
                                             "t_last_ok_ns": cs.t_last_ok_ns, "level": cs.level}}

    from .field import FieldError

    def _field(fn: Any) -> Any:
        def run(m: dict, ctx: Any = None) -> dict:
            if not rt.bind(ctx):
                return {"v": 1, "code": 211, "detail": {"why": "TARGETS_NOT_READY"}}
            try:
                return {"v": 1, "code": 0, "item": fn(m.get("args") or {}, ctx)}
            except FieldError as e:
                return {"v": 1, "code": e.code, "detail": e.detail}
        return run

    def q_quant(a: dict, ctx: Any) -> dict:
        uav = a.get("uav")
        return rt.field.quant(str(a.get("target", "")), ctx, uav=str(uav) if uav else None)

    def q_field(a: dict, ctx: Any) -> dict:
        return rt.field.profile_full(str(a.get("kind", "")), ctx)

    def q_volumes(a: dict, ctx: Any) -> dict:
        return rt.field.volumes_full(ctx)

    def q_samples(a: dict, ctx: Any) -> dict:
        n = a.get("n")
        j = a.get("j")
        return rt.field.samples(str(a.get("kind", "")), str(a.get("anchor", "")), n if isinstance(n, int) else -1, ctx,
                               model_id=a.get("model_id"), j=int(j) if isinstance(j, int) and not isinstance(j, bool) else None)

    for name, fn in (("perception/vehicle", q_vehicle), ("perception/detections", q_detections), ("perception/tracks", q_tracks),
                     ("perception/feasibility", q_feasibility), ("perception/capture", q_capture), ("perception/quant", _field(q_quant)),
                     ("perception/field", _field(q_field)), ("perception/volumes", _field(q_volumes)),
                     ("perception/samples", _field(q_samples))):
        fn.__module__ = __name__
        if name not in reg.queries:
            R.register_query(name, fn)

    def field_task(ctx: Any) -> Any:
        return rt.field.tick(ctx)

    field_task.__module__ = __name__
    if not any(t.name == FIELD_TASK for t in reg.slow):
        R.register_slow_task(FIELD_TASK, field_task, period_wall_s=FIELD_TASK_PERIOD_S, budget_us=3000)
    _metrics()
    reg.plugins[_KEY] = rt
    return rt


install()
