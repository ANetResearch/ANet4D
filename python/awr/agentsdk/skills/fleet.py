"""机队技能执行：指挥者的技能指令 → 每架机的阶段机 → 逐 tick 的 `bench/act.v1` 动作。

每架机一个阶段：

    landed        停在起飞点，未起飞；第一条运动技能触发起飞（hold、rtl 在地面上被忽略）
    takeoff       起飞中（`nav takeoff` 到待执行技能的 agl，缺省 `takeoff_agl`）；到达前 3 m 内转 airborne 并安装待执行技能
    airborne      执行技能；rtl 完成且物理档报告 landed 后转 landed_final
    landed_final  已在起飞点降落，任务结束，不再接受指令
    exhausted     电量耗尽（`battery_soc <= 0`）：原地紧急降落一次，不再接受指令

`observe(uav, ob, t_s)` 记录最新观测并处理电量耗尽；`command(uav, skill, params, t_s)` 安装指令，返回面向指挥者的拒绝文案
（None 为接受或按规则忽略）；`actions(k, obs, agents, t_s)` 给出本 tick 全部在飞机体的动作。`zoom(uav, factor, t_s)` 由运行注入
（光学镜头属于 L2 传感器，不在本包）；未注入时 zoom 指令被拒绝。

2026-11-05 自 CHIMERA 基准移入（`chimera.commander.harness.Commander` 的逐机阶段与逐 tick 动作部分），规则与文案不变。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from awr.agentsdk.skills.actions import act, nav_takeoff
from awr.agentsdk.skills.runner import SkillRunner

__all__ = ["PHASES", "SkillFleet", "UavState"]

PHASES = ("landed", "takeoff", "airborne", "landed_final", "exhausted")


@dataclass
class UavState:
    phase: str = "landed"
    exhausted_at: float | None = None       # episode time of the emergency landing (battery at 0)
    exhausted_xy: tuple | None = None
    pending: tuple[str, dict] | None = None
    last_obs: dict | None = None
    launched: bool = False


class SkillFleet:
    def __init__(self, uavs, cams: dict[str, str], homes: dict[str, tuple[float, float]], alt_range: tuple[float, float], *,
                 terrain=None, takeoff_agl: float = 60.0, zoom: Callable[[str, float, float], None] | None = None) -> None:
        self.runners = {u: SkillRunner(u, cams[u], (float(alt_range[0]), float(alt_range[1])), homes[u], terrain=terrain) for u in uavs}
        self.state = {u: UavState() for u in uavs}
        self.takeoff_agl = float(takeoff_agl)
        self.zoom = zoom

    @property
    def uavs(self) -> list[str]:
        return list(self.runners)

    # ------------------------------------------------------------------ observations
    def observe(self, uav: str, ob: dict, t_s: float) -> str | None:
        """Record the latest observation of `uav`. Returns the log line when the battery ran out (emergency landing)."""
        s = self.state[uav]
        s.last_obs = ob
        soc = ob["nav"].get("battery_soc")
        if s.phase in ("takeoff", "airborne") and soc is not None and float(soc) <= 0.0:
            # The battery is exhausted: the simulator would let the drone fly on, the physical drone cannot.
            # Emergency landing where it is; the drone takes no further commands.
            p = ob["nav"]["pos_enu_m"]
            s.phase, s.exhausted_at, s.exhausted_xy = "exhausted", t_s, (float(p[0]), float(p[1]))
            self.runners[uav].set("land", {}, t_s, (float(p[0]), float(p[1])))
            return f"{uav} battery exhausted at t={t_s:.0f}s: emergency landing at ({p[0]:.0f}, {p[1]:.0f})"
        return None

    # ------------------------------------------------------------------ commands
    def command(self, uav: str, skill: str, params: dict | None, t_s: float) -> str | None:
        """Install a skill for `uav`; returns the rejection text, None when accepted (or ignored by the rules below)."""
        if uav not in self.runners:
            return "unknown UAV"
        params = params or {}
        s = self.state[uav]
        if skill == "land":          # the emergency landing of the executor, not a skill of the interface
            return "unknown skill 'land'"
        if skill == "zoom":
            try:
                if self.zoom is None:
                    raise ValueError("Optical lens unavailable in this run")
                self.zoom(uav, params["factor"], t_s)
            except (ValueError, TypeError, KeyError) as e:
                return str(e)
            return None
        if s.phase == "exhausted":
            return (f"battery exhausted at t={s.exhausted_at:.0f} s, the drone made an emergency landing at "
                    f"({s.exhausted_xy[0]:.0f}, {s.exhausted_xy[1]:.0f}) and is no longer available")
        if s.phase == "landed_final":
            return "the drone has landed at the pad and ended its mission"
        if s.phase in ("landed", "takeoff"):
            if skill in ("hold", "rtl"):
                return None
            s.pending = (skill, params)
            if s.phase == "landed":
                s.phase = "takeoff"
            return None
        ob = s.last_obs
        pos = ob["nav"]["pos_enu_m"] if ob else (0.0, 0.0, 0.0)
        return self.runners[uav].set(skill, params, t_s, (float(pos[0]), float(pos[1])))

    # ------------------------------------------------------------------ per-tick actions
    def actions(self, k: int, obs: dict, agents, t_s: float) -> dict[str, dict]:
        """`obs`: the Env observation batch `{uav: {"obs": ...}}`; `agents`: the UAVs still in the episode."""
        actions: dict[str, dict] = {}
        for u, o in obs.items():
            if u not in agents or u not in self.runners:
                continue
            ob = o["obs"]
            gz = ob["ground_hint"]["dtm_m"]
            mode = ob["status"]["mode"]
            s = self.state[u]
            runner = self.runners[u]
            if s.phase == "landed":
                actions[u] = act(k)
            elif s.phase == "takeoff":
                alt = s.pending[1].get("agl", self.takeoff_agl) if s.pending else self.takeoff_agl
                alt = max(float(alt), 2.0)      # the envelope is scored, not enforced (2 m physical floor only)
                if not s.launched:
                    actions[u] = act(k, nav=nav_takeoff(alt))
                    s.launched = True
                elif mode == "airborne" and float(ob["nav"]["pos_enu_m"][2]) - gz >= alt - 3.0:
                    s.phase = "airborne"
                    if s.pending:
                        p = ob["nav"]["pos_enu_m"]
                        runner.set(s.pending[0], s.pending[1], t_s, (float(p[0]), float(p[1])))
                        s.pending = None
                    nav, gim = runner.tick(ob, gz, t_s)
                    actions[u] = act(k, nav=nav, gimbal=gim)
                else:
                    actions[u] = act(k)
            elif s.phase == "landed_final":
                actions[u] = act(k)
            elif s.phase == "exhausted":
                nav, _gim = runner.tick(ob, gz, t_s)        # the land command once, then nothing
                actions[u] = act(k, nav=nav)
            else:
                nav, gim = runner.tick(ob, gz, t_s)
                actions[u] = act(k, nav=nav, gimbal=gim)
                if runner.state.kind == "rtl" and runner.state.done and mode == "landed":
                    s.phase = "landed_final"
        return actions

    # ------------------------------------------------------------------ status
    def phase(self, uav: str) -> str:
        return self.state[uav].phase

    def status(self, uav: str) -> dict:
        """`{skill, done, progress, look_at}` of the active skill (the fleet-status table of the commander)."""
        return self.runners[uav].status()
