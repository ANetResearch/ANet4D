"""M21 目录的 sim-core 组合根入口（AWR-04 §5.4；AWR-20 §7.6、§8.3；M08 §7.1.1、§15.3）。

导入本模块即调用 `install()`（`configs/runtime.yaml` 的 sim-core `plugins:` 写 `awr.sim.catalog.plugin`）；对当前登记表幂等，登记：

- 机型检查（M08 `register_profile_check`）：FW-1 至 FW-8、MT-5（固定翼与 VTOL）、VT-1 至 VT-4（VTOL）、SG-1 至 SG-3、MT-2、SN-7、
  SN-9（全部构型）——ProfileTable 加载（353）与会话克隆（CL-3，522）时执行；
- 命令（M08 `register_command_handler`，验签与角色由 CommandEngine 完成，不要求席位）：`catalog/clone`（SB-25）、
  `catalog/clone_update`（SB-26）、`catalog/clone_delete`（SB-27）；在步顶同步执行（ProfileTable 写时复制重建矩阵），结果
  `{model_id, base, version}`，失败以 522（`detail.violations[]`）、505、520、560、561、105 拒绝；
- 查询（`ctl/sim-core/query`）：`catalog/models`（SB-24：本会话克隆的摘要）、`catalog/model`（克隆或预置机型的详情）、
  `catalog/nest_compat`（MT-8：`{model_id, nest{kind, chargers[]}}` -> `{ok, code, why?}`，供 M20 机巢与 M08 实例创建调用）、
  `catalog/capabilities`（`{model_id, mounted?[]}` -> 有效能力集，供 M20 能力匹配）；
- `fleet/update` 钩子（M08 `register_fleet_update_hook`）：改挂载后的 MT-3（质量 ≤ MTOW）与 MT-5（固定翼 v_min ≥ 1.1·V_s(m')），
  不通过以 522（detail `MOUNT_MTOW`、`FW_VMIN`）拒绝。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from ..fleet import profiles as PF
from . import checks as C
from . import clone as CL
from . import derive as D
from .loader import Catalog, CatalogError, capabilities_of, nest_compatible

__all__ = ["COMMANDS", "QUERIES", "CatalogRuntime", "install", "runtime"]

log = logging.getLogger("awr.sim.catalog")
COMMANDS = ("catalog/clone", "catalog/clone_update", "catalog/clone_delete")
QUERIES = ("catalog/models", "catalog/model", "catalog/nest_compat", "catalog/capabilities")
_KEY = "catalog"                                     # the runtime lives in registry.plugins of the table it was installed into


def _ok(result: Any = None) -> dict:
    out: dict[str, Any] = {"status": "accepted", "code": 0}
    if result is not None:
        out["result"] = result
    return out


def _rej(code: int, detail: Any = None) -> dict:
    return {"status": "rejected", "code": int(code), "detail": detail}


class CatalogRuntime:
    """会话内的目录服务：克隆与查询。ProfileTable 取自 `ctx.profiles`，实例表取自 `ctx.roster`。"""

    def __init__(self) -> None:
        self._cat: dict[str, Catalog | None] = {}
        self.stats = {"clone": 0, "clone_rejected": 0}

    # ------------------------------------------------------------ 目录索引（按机型目录缓存）
    def catalog(self, T: Any) -> Catalog | None:
        vdir = str(getattr(T, "vdir", None) or PF.vehicles_dir())
        if vdir not in self._cat:
            try:
                self._cat[vdir] = Catalog.load(vdir) if (Path(vdir) / "catalog.json").exists() else None
            except CatalogError:
                log.warning("catalog.json invalid; session catalog queries use profile documents only")
                self._cat[vdir] = None
        return self._cat[vdir]

    def _entry(self, T: Any, pid: str) -> dict:
        cat = self.catalog(T)
        base = pid
        p = T.profiles.get(pid)
        if p is not None and p.base_ref is not None:
            base = str(p.base_ref).split("@", 1)[0]
        e = (cat.entry(base) if cat is not None else None) or {}
        return dict(e)

    # ------------------------------------------------------------ 命令
    def cmd_clone(self, msg: dict, apply_tick: int, ctx: Any) -> dict:
        T = ctx.profiles
        args = msg.get("args") if isinstance(msg.get("args"), dict) else {}
        quota = None
        lim = getattr(getattr(ctx, "calls", None), "fleet_ops", None)
        if lim is not None and isinstance(getattr(lim, "limits", None), dict):
            quota = int(lim.limits.get("clones", PF.CLONE_SESSION_QUOTA))
        try:
            cid = CL.create_clone(T, args, quota=quota)
        except CL.CloneFailure as e:
            self.stats["clone_rejected"] += 1
            return _rej(e.code, e.detail)
        self.stats["clone"] += 1
        p = T.get(cid)
        return _ok({"model_id": cid, "base": T.sessions[cid]["base"], "version": p.version})

    def cmd_update(self, msg: dict, apply_tick: int, ctx: Any) -> dict:
        args = msg.get("args") if isinstance(msg.get("args"), dict) else {}
        cid = str(args.get("model_id") or "")
        try:
            CL.update_clone(ctx.profiles, cid, args, getattr(ctx, "roster", None))
        except CL.CloneFailure as e:
            return _rej(e.code, e.detail)
        return _ok({"model_id": cid})

    def cmd_delete(self, msg: dict, apply_tick: int, ctx: Any) -> dict:
        args = msg.get("args") if isinstance(msg.get("args"), dict) else {}
        cid = str(args.get("model_id") or "")
        try:
            CL.delete_clone(ctx.profiles, cid, getattr(ctx, "roster", None))
        except CL.CloneFailure as e:
            return _rej(e.code, e.detail)
        return _ok({"model_id": cid})

    # ------------------------------------------------------------ 查询
    def _summary(self, T: Any, cid: str) -> dict:
        from .view import summary_of

        p = T.get(cid)
        mdir = Path(p.model_dir) if p.model_dir else C.model_dir_for(p.doc)
        return summary_of(p.doc, self._entry(T, cid), mdir, model_id=cid, session_clone=p.base_ref is not None)

    def _detail(self, T: Any, cid: str) -> dict:
        from .view import detail_of

        p = T.get(cid)
        mdir = Path(p.model_dir) if p.model_dir else C.model_dir_for(p.doc)
        cat = self.catalog(T)
        rec = getattr(T, "sessions", {}).get(cid) or {}
        meta = getattr(T, "meta", {}).get(p.ref) or {}
        out = detail_of(p.doc, self._entry(T, cid), mdir, cat.doc if cat is not None else {}, model_id=cid,
                        session_clone=p.base_ref is not None, with_sensors=False, checks=meta.get("checks"),
                        extra_w=float(rec.get("extra_power_w") or 0.0))
        if p.base_ref is not None:
            out["base"] = rec.get("base")
            out["set"] = dict(rec.get("set") or {})
            out["default_mounts"] = dict(rec.get("default_mounts") or {})
        return out

    def q_models(self, req: Any, ctx: Any) -> dict:
        T = ctx.profiles
        return {"v": 1, "code": 0, "items": [self._summary(T, cid) for cid in getattr(T, "sessions", {})]}

    def q_model(self, req: Any, ctx: Any) -> dict:
        T = ctx.profiles
        mid = str(((req or {}).get("args") or {}).get("model_id") or "")
        if not T.has(mid) or mid in T.hidden_ids():
            return {"v": 1, "code": CL.CODE_MODEL_UNKNOWN, "detail": {"model_id": mid}}
        return {"v": 1, "code": 0, "item": self._detail(T, mid)}

    def q_nest_compat(self, req: Any, ctx: Any) -> dict:
        T = ctx.profiles
        a = (req or {}).get("args") or {}
        mid = str(a.get("model_id") or "")
        if not T.has(mid):
            return {"v": 1, "code": CL.CODE_MODEL_UNKNOWN, "detail": {"model_id": mid}}
        r = nest_compatible(T.get(mid).doc, dict(a.get("nest") or {}))
        return {"v": 1, "code": 0, "item": r}

    def q_capabilities(self, req: Any, ctx: Any) -> dict:
        T = ctx.profiles
        a = (req or {}).get("args") or {}
        mid = str(a.get("model_id") or "")
        if not T.has(mid):
            return {"v": 1, "code": CL.CODE_MODEL_UNKNOWN, "detail": {"model_id": mid}}
        p = T.get(mid)
        rig = None
        try:
            rig = C.default_rig(p.doc, Path(p.model_dir) if p.model_dir else None)
        except Exception:
            rig = None
        return {"v": 1, "code": 0, "items": capabilities_of(p.doc, list(rig.specs) if rig is not None else [])}

    # ------------------------------------------------------------ fleet/update 钩子（MT-3、MT-5）
    def fleet_update_hook(self, vehicle_id: str, slot: int, patch: dict, ctx: Any) -> dict | None:
        if "sensors" not in patch:
            return None
        import sys

        fn = getattr(sys.modules.get("awr.sim.sensors.plugin"), "runtime", None)
        rt = fn() if callable(fn) else None
        if rt is None:
            return None
        T = ctx.profiles
        try:
            S = rt.S
            pid = T.ids[int(S.profile_id[int(slot)])]
            p = T.get(pid)
            ms = rt.mount_summary(int(slot)) or {}
            mdir = Path(p.model_dir) if p.model_dir else None
            base_rig = C.default_rig(p.doc, mdir)
        except Exception:
            return None
        if not ms or base_rig is None:
            return None
        base_m = sum(float(s.mass_kg) for s in base_rig.specs if math.isfinite(float(s.mass_kg)))
        m1 = float(p.mass_kg) + float(ms.get("mass_kg") or 0.0) - base_m
        if p.mtow_kg is not None and m1 > float(p.mtow_kg) + 1e-9:
            return {"code": CL.CODE_CLONE_INVALID, "detail": {"why": "MOUNT_MTOW", "violations": [
                {"field": "sensors", "rule": "MT-3", "value": round(m1, 4), "limit": [0.0, p.mtow_kg],
                 "message_zh": "挂载后的起飞重量超过最大起飞重量"}]}}
        fw = D.fw_aero(p.doc, mass_kg=m1)
        if fw is not None and fw.v_min_mps < 1.1 * fw.v_stall() - 1e-9:
            return {"code": CL.CODE_CLONE_INVALID, "detail": {"why": "FW_VMIN", "violations": [
                {"field": "sensors", "rule": "MT-5", "value": round(fw.v_min_mps, 3), "limit": [round(1.1 * fw.v_stall(), 3), None],
                 "message_zh": "挂载后 v_min 低于 1.1 倍失速速度"}]}}
        return None


def install() -> CatalogRuntime:
    from ..fleet.stages import registry as R

    reg = R.registry()
    rt = reg.plugins.get(_KEY)
    if rt is not None:
        return rt
    rt = CatalogRuntime()
    C.register(R)
    for op, fn in zip(COMMANDS, (rt.cmd_clone, rt.cmd_update, rt.cmd_delete), strict=True):
        if op not in reg.commands:
            R.register_command_handler(op, fn, owner="M21", need_seat=False)
    for name, fn in zip(QUERIES, (rt.q_models, rt.q_model, rt.q_nest_compat, rt.q_capabilities), strict=True):
        if name not in reg.queries:
            R.register_query(name, fn)
    R.register_fleet_update_hook(rt.fleet_update_hook)
    reg.plugins[_KEY] = rt
    return rt


def runtime() -> CatalogRuntime | None:
    from ..fleet.stages import registry as R

    return R.registry().plugins.get(_KEY)


install()
