"""BenchEnv（M26 §6.6.1；P0a）：lockstep 主循环、观测投影（L1 检测级；L2、L2-depth 渲染级见 `env.l2`）、决策时延与动作注入，
PettingZoo Parallel 风格接口。

    env = BenchEnv(spec)
    obs, infos = env.reset()                      # {uav: {"obs": Observation, "inbox": [Message]}}，只含本 tick 应调用的 Agent
    while env.agents:
        acts = {u: agent[u].step(o["obs"], o["inbox"]) for u, o in obs.items()}
        obs, rewards, terms, truncs, infos = env.step(acts)
    result = env.close()

时序：控制步 100 ms；tick k 的观测取 t_k 的真值与导航解，L1 检测取 (t_last, t_k] 内的相机帧（10 Hz，帧序号 = tick）；动作在
`apply_tick = k_obs + max(1, ⌈τ / 100 ms⌉)` 生效（固定延迟生效、未到即阻塞），在途期间不再调用该 Agent；物理档 `advance(k)` 推进
(t_k, t_{k+1}]。L2 的传感器帧在 tick 末按物理档样本渲染，Agent 未被调用的 tick 的帧累积到下一次观测（按 seq 升序）。
M25 会话（`sensorsvc`，L2 与 L2-depth）：到期帧以 `(uav, sensor, seq)` 请求提交到 Benchmark 会话，`wait_all` 阻塞至全部就绪才放行
观测（641 时按原键重提交，仿真时钟不推进、不降频、不转离线，ADR-162）；L2Sensors 为该会话的进程内 worker 后端，每个 tick 生成一次
并按键交付，帧字节与不经会话时相同。观测墙预览（`wall_preview`）由写入器旁路从交付的相机帧降采样（2 Hz、480 × 270、JPEG 75），
写入 `preview/<uav>/<seq>.jpg` 或交给回调，不进入 M25 预览队列（ADR-173）。
物理档（`spec.physics`）：`realistic`（gz 锁步，正式成绩）、`fast`（FleetSim，`awr.phys.fast`）、`kinematic`（开发夹具）；同一 Agent
代码与观测接口在各档不变，`clock_profile` 先按 M24 时钟档规则与输出集合校验（637）。
goto 目标解释在本机导航系，每个控制步按当前导航误差换算为真值系目标（变化 ≥ 5 cm 时更新跟踪器）。
预滚（`env.preroll`，M26-FR-004）：reset 时先推进事件零点之前的 `[−t_pre, 0)`（缺省 2 s；地面起飞为静止、空中出生为悬停保持），
不调用 Agent、不交付观测；此后物理档以事件时间视图出现，k = 0 观测时刻为 0，窗口判定写入 `tier_info.preroll`。
未返回动作的 Agent 计超时（回执 `timeout`），沿用上一导航指令。回合在事件时长到达或全部机体终止时结束。
磁带校验（`env.tapecheck`）：构造时先核对磁带与所加载世界包的版本（不符 615 `TAPE_WORLD_MISMATCH`，不创建运行目录）；正式运行（排名
且 `realistic` 档）另拒绝栅格后端编译（`generator_version` 以 `-grid` 结尾）或 NavMesh 哈希与世界图层不符的磁带（ADR-196 决策 (5)），
结果写入运行清单 `tape_check`。
世界查询（`open_world_query`；K07）：缺省打开网格查询 `MeshWorldQuery`（M22），失败时正式运行（`is_formal`）拒绝并抛
`WorldQueryFallbackRefused`，不退回 DSM；非正式运行记录 warning 后退回 DSM 查询。实际使用的查询类型与退回原因写入运行清单
`world_query`（`{kind: mesh | dsm | other, class, fallback?}`）。
运行种子（`write_run_seed`；K18）：不排名运行写入 `truth/run_seed.json`；排名运行不在运行目录留种子明文（写入隐藏区
`hidden/<eid>/truth/runs/<run_id>/` 或 `spec.seed_sink`，没有隐藏区时不写），落点记入运行清单 `seed_store`。
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import numpy as np

from awr.agentsdk.env.camera import CameraModel, Gimbal, cam_axes
from awr.agentsdk.env.guard import ActionGuard, Limits
from awr.agentsdk.env.l1 import L1Model
from awr.agentsdk.env.l2 import CAM_SENSOR, GNSS_SENSOR, IMU_SENSOR, LIDAR_SENSOR, L2Config
from awr.agentsdk.env.link import LinkModel
from awr.agentsdk.env.nav import NavModel
from awr.agentsdk.env.preroll import PREROLL_DEFAULT_S, plan_preroll, run_preroll
from awr.agentsdk.env.recorder import EnvRecorder
from awr.agentsdk.env.spec import EnvSpec
from awr.agentsdk.env.tapecheck import check_tape_world, is_formal
from awr.agentsdk.task import brief_for_agent, task_from_event
from awr.phys.adapter import TierCommand, VehicleStatus
from awr.sensorsvc.gpuhealth import gpu_errors

__all__ = ["SEED_FILE", "TICK_MS", "TICK_NS", "BenchEnv", "WorldQueryFallbackRefused", "open_world_query", "seed_sink_for",
           "world_query_info", "write_run_seed"]

log = logging.getLogger(__name__)

TICK_MS = 100
TICK_NS = TICK_MS * 1_000_000
_MODE = {VehicleStatus.GROUNDED: "landed", VehicleStatus.LANDING: "landing", VehicleStatus.AIRBORNE: "airborne",
         VehicleStatus.CRASHED: "failsafe", VehicleStatus.LOST: "failsafe", VehicleStatus.MOVING: "driving", VehicleStatus.AFLOAT: "afloat"}
_NAV_CMD = {"velocity": "velocity", "goto": "goto", "hover": "hover", "takeoff": "takeoff", "land": "land", "kill": "hover"}


def _yaw(q) -> float:
    x, y, z, w = (float(a) for a in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class _Agent:
    def __init__(self, uav: str, no: int) -> None:
        self.uav, self.no = uav, no
        self.next_k = 0
        self.k_obs: int | None = None
        self.acks: list[dict] = []
        self.inbox: list[dict] = []
        self.dets: list[dict] = []
        self.frames: dict[str, list[dict]] = {}
        self.cmd_kind = "hover"
        self.goto_nav: np.ndarray | None = None
        self.goto_true: np.ndarray | None = None
        self.goto_speed = 8.0
        self.done = False
        self.term = {"terminated": False, "truncated": False, "reason": "none"}
        self.timeouts = 0
        self.flight_s = 0.0
        self.wall_s = 0.0


class BenchEnv:
    def __init__(self, spec: EnvSpec, *, world=None, tape=None, tier=None, cam: CameraModel | None = None, limits: Limits | None = None,
                 l2=None, sensorsvc=None, inflight_cap: int | None = None, wall_preview=False, preroll_s: float = PREROLL_DEFAULT_S,
                 l2_cache_dir: Path | None = None) -> None:
        """`l2`：L2 传感器工厂 `(env) -> L2Sensors`（测试注入 CPU 渲染替身）；缺省按 `spec.l2` 构造 EGL 光栅相机。
        `sensorsvc`：M25 调度器（`awr.sensorsvc.scheduler.Scheduler`；进程内 GL 渲染须 `threaded=False`），L2 帧经其 Benchmark 会话交付；
        `inflight_cap` 为会话在途上限（缺省 2 个 tick 的到期帧数）。`wall_preview`：True 写入运行目录，或回调
        `(uav, seq, t_ns, jpeg_bytes)`。`preroll_s`：事件零点之前的预滚窗口（`env.preroll`；缺省 2 s，0 关闭）。`l2_cache_dir`：L2 网格
        世界查询的派生缓存根（缺省按 `env.l2.worldquery_cache_dir` 取 `AWR_WORLDQUERY_CACHE_DIR`，再按世界目录推导；不进入运行哈希）。"""
        from awr.sim.events.tape import open_tape

        if spec.obs_level not in ("L1", "L2", "L2-depth"):
            raise ValueError(f"obs_level {spec.obs_level!r}: ranked Env runs are L1, L2 or L2-depth")
        self.spec = spec
        self.uavs = [f.uav for f in spec.fleet]
        self.tape = tape or open_tape(Path(spec.event_dir))
        self.tape_check = check_tape_world(self.tape, Path(spec.world_dir), formal=is_formal(spec))   # 615：磁带须为所加载世界版本编译
        if world is None:
            self.world, self.world_query = open_world_query(Path(spec.world_dir), formal=is_formal(spec))
        else:
            self.world, self.world_query = world, world_query_info(world)
            if is_formal(spec) and self.world_query["kind"] == "dsm":
                raise WorldQueryFallbackRefused(Path(spec.world_dir), "a DSM world query was injected into a formal run")
        self.cam = cam or spec.camera_model()          # L1 相机：标称相机或 run_spec 相机覆盖（ADR-217）
        self.lim = limits or Limits()
        self.task = task_from_event(Path(spec.event_dir))
        self.duration_s = float(spec.duration_s or self.task["brief"]["duration_s"])
        self.n_ticks = round(self.duration_s * 1000 / TICK_MS)
        self.nav = NavModel(spec.seed, spec.noise)
        self.l1 = L1Model(spec.seed, self.world, self.tape, self.cam)
        self.link = LinkModel(spec.seed, tuple(self.uavs), self.world, self.tape, spec.comm)
        self.rec = EnvRecorder(spec.out_dir, spec=spec, tape=self.tape)
        self._tier = tier
        self._l2_factory = l2
        self._l2_cache = Path(l2_cache_dir) if l2_cache_dir else None
        self.l2 = None
        self._svc, self._svc_cap, self._sess = sensorsvc, inflight_cap, None
        self._svc_wait = {"ticks": 0, "rejected_641": 0, "resubmits": 0, "wait_s": 0.0, "backpressure_s": 0.0, "blocked_ticks": 0}
        self._svc_cache: tuple[int, dict] | None = None
        self._truth_q: list = []                          # 推迟计算的 L2 真值行（Future，按 tick、机序）
        # 观测记录（摘要与 raw.mcap 观测通道）在后台线程按交付顺序写出，下一步结算动作之前取齐：只在观测交付后不再被修改时开启
        # （`run_remote_episode`：观测经线协议编码发送）；写入顺序与同步写入相同
        self.async_obs_record = False
        self._rec_pool = None
        self._rec_futs: list = []
        self._wall = _WallPreview(spec.out_dir, wall_preview) if wall_preview else None
        self.preroll = plan_preroll(((f.uav, f.airborne) for f in spec.fleet), preroll_s, TICK_MS)
        self.possible_agents = list(self.uavs)
        self.agents: list[str] = []
        self.k = 0

    # ------------------------------------------------------------------ 组装
    def _make_l2(self):
        import json as _json

        from awr.agentsdk.env.l2 import L2Sensors

        wd = Path(self.spec.world_dir)
        meta = _json.loads((wd / "world.json").read_text(encoding="utf-8")) if (wd / "world.json").is_file() else {}
        cfg = L2Config(**{k: v for k, v in (self.spec.l2 or {}).items() if k in ("camera_profile", "jpeg_quality", "lidar", "imu", "gnss")},
                       cache_dir=self._l2_cache)
        return L2Sensors(world_dir=wd, tape=self.tape, uavs=self.uavs, seed=self.spec.seed, noise=self.spec.noise, obs_level=self.spec.obs_level,
                         nav=self.nav, mount_t_bs_m=self.cam.mount_t_bs_m, hfov_deg=self.cam.hfov_deg, cfg=cfg, world_meta=meta)

    def _feed_l2(self, k_prev: int) -> None:
        """物理档 (t_{k_prev}, t_k] 的样本与 t_k 的云台角交给 L2 传感器。"""
        if self.l2 is None:
            return
        t = self.k * TICK_NS
        for u in self.uavs:
            sl = self.tier.history.slice(u, k_prev * TICK_NS, t)
            self.l2.ingest(u, sl.t_ns, sl.state)
            g = self.gim[u]
            self.l2.gimbal(u, t, g.yaw_deg, g.pitch_deg)

    def _make_tier(self):
        if self._tier is not None:
            return self._tier
        spawn = {f.uav: (f.pos_enu_m, f.yaw_rad) for f in self.spec.fleet}
        grounded = [f.uav for f in self.spec.fleet if not f.airborne]
        if self.spec.physics == "kinematic":
            from awr.phys.kinematic import KinematicTier

            return KinematicTier(self.uavs, spawn, lambda xy: self.world.height_dsm(np.asarray(xy, np.float64)), grounded=grounded)
        if self.spec.physics == "fast":
            from awr.phys.fast import FastTier

            return FastTier(self.uavs, spawn, self.world, grounded=grounded, vehicles={f.uav: f.vehicle_ref for f in self.spec.fleet},
                            clock_profile=self.spec.clock_profile, tick_ms=TICK_MS, outputs=self._tier_outputs())
        if self.spec.physics == "l2":
            from awr.phys.l2 import L2Tier

            return L2Tier(self.uavs, spawn, self.world, grounded=grounded, vehicles={f.uav: f.vehicle_ref for f in self.spec.fleet},
                          clock_profile=self.spec.clock_profile, tick_ms=TICK_MS, outputs=self._tier_outputs())
        if self.spec.physics == "realistic":
            from awr.phys.adapter import validate_clock
            from awr.phys.fast import check_physics_hz, clock_of

            cp = clock_of(self.spec.clock_profile)
            validate_clock("realistic", cp, TICK_MS, self._tier_outputs())
            check_physics_hz(cp, self._tier_outputs())
            return _GzTier.create(self.spec, spawn, grounded, self.world)
        raise ValueError(f"physics {self.spec.physics!r} is not available (kinematic, fast, l2, realistic)")

    def _tier_outputs(self) -> tuple[str, ...]:
        """时钟档相容性校验用的输出集合（M24-FR-005）：L2 与 L2-depth 按 `spec.l2` 开关给出 camera、lidar、imu。"""
        if self.spec.obs_level == "L1":
            return ("detections",)
        cfg = L2Config(**{k: v for k, v in (self.spec.l2 or {}).items() if k in ("lidar", "imu", "gnss")})
        return ("camera", *(["lidar"] if cfg.lidar else []), *(["imu"] if cfg.imu else []))

    def agent_config(self, uav: str) -> dict:
        nominal = {"gnss_pos_std_m": self.nav.pos_std_m, "l1_ground_sigma_m": [self.l1.p.sig0_m, self.l1.p.sig_per_m]}
        brief = brief_for_agent(self.task, Path(self.spec.event_dir), nominal_noise=nominal)
        for a in brief["anchors"]:
            a["pos_enu_m"][2] = float(self.world.ground_dtm(np.array([a["pos_enu_m"][:2]]))[0])
        f = next(x for x in self.spec.fleet if x.uav == uav)
        e = self.nav.error(self.uavs.index(uav), 0)
        return {"schema": "drone4d.bench.agent_config.v1", "uav_id": uav, "kind": "uav", "peers": [u for u in self.uavs if u != uav],
                "vehicle": {"ref": f.vehicle_ref, "limits": self.lim.config(), "endurance_s": self.nav.endurance_s},
                "sensors": self.l2.configs(self.cam.config()) if self.l2 is not None else [self.cam.config()],
                "start": {"kind": "pose", "state": "airborne" if f.airborne else "landed",
                          "nav_pos_enu_m": [float(x) for x in np.asarray(f.pos_enu_m) + e]},
                "obs_level": self.spec.obs_level, "control_hz": 10, "decision": self.spec.decision.config(),
                "comm": {"enabled": self.spec.comm.enabled, "bucket_Bps": self.spec.comm.bucket_Bps, "burst_B": self.spec.comm.burst_B,
                         "msg_max_B": self.spec.comm.msg_max_B, "model": self.spec.comm.model},
                "episode": {"duration_s": self.duration_s},
                "world": {"id": Path(self.spec.world_dir).name, "content_version": self.tape.header["world"]["content_version"],
                          "origin": {"lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0}},
                "brief": brief, "protocol": "drone4d.bench.wire.v1"}

    # ------------------------------------------------------------------ PettingZoo 风格接口
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None and seed != self.spec.seed:
            raise ValueError("the run seed is fixed by the run specification")
        self.tier = self._make_tier()
        self.tier.open()
        self.tier = run_preroll(self.tier, self.preroll)    # [−t_pre, 0)：不调用 Agent、不交付观测；窗口内接触只记入 tier_info.preroll
        if self.spec.obs_level != "L1":
            with gpu_errors("l2 open"):                     # GPU 丢失（设备节点、EGL、Warp）→ 647 平台失败，不再崩溃（K18）
                self.l2 = self._l2_factory(self) if self._l2_factory is not None else self._make_l2()
            if self._svc is None and hasattr(self.l2, "defer_truth"):
                self.l2.defer_truth = True                 # 真值可见度归约在等待 Agent 动作期间由后台线程完成（`_flush_truth`）
            if self._svc is not None:
                cap = self._svc_cap or 2 * len(self.uavs) * len(self._l2_sensors(1))
                self._sess = self._svc.open(f"{self.spec.run_id}/0", "benchmark", produce=self._produce, inflight_cap=cap)
        self.st = {u: _Agent(u, i) for i, u in enumerate(self.uavs)}
        self.gim = {u: Gimbal(cam=self.cam) for u in self.uavs}
        self.guard = {u: ActionGuard(u, [p for p in self.uavs if p != u], self.lim, self.task["report_types"], self.spec.comm.enabled,
                                     self.spec.comm.msg_max_B, self.cam) for u in self.uavs}
        self.pending_gim: list[tuple[int, str, tuple[float, float]]] = []
        self.pending_tx: list[tuple[int, str, list[dict]]] = []
        self._fleet_t, self._fleet = [], {"pos": [], "vel": [], "yaw": [], "nav": [], "status": []}
        self.k = 0
        self.agents = list(self.uavs)
        self.status = dict(self.tier.status())
        self.rec.write_json("task.json", self.task)
        self.seed_store = write_run_seed(self.spec, self.rec)
        self.configs = {u: self.agent_config(u) for u in self.uavs}
        for u, c in self.configs.items():
            self.rec.write_json(f"agents/{u}/config.json", c)
        self._t0 = time.perf_counter()
        if self.l2 is not None:                             # 预滚样本只进入 L2 的 IMU 预热，不产生帧
            for u in self.uavs:
                self.l2.ingest(u, *self.tier.preroll_samples(u))
        self._feed_l2(0)
        self._sense(0)
        self._log_fleet()
        return self._observe(), {u: {"config": self.configs[u]} for u in self.uavs}

    def _flush_truth(self) -> None:
        """按序写出推迟计算的 L2 真值行（上一次感知的各机）：在每次推进之前（本 tick 的接触事件之前）与关闭时写出，真值层的写入顺序
        与不推迟时相同。"""
        q, self._truth_q = self._truth_q, []
        with gpu_errors("l2 truth"):
            for fut in q:
                for tr in fut.result():
                    self.rec.on_vis(tr)

    def _flush_obs(self) -> None:
        futs, self._rec_futs = self._rec_futs, []
        for f in futs:
            f.result()

    def step(self, actions: dict[str, dict]):
        self._flush_obs()
        for u in list(actions):
            if u not in self.st or self.st[u].k_obs is None:
                raise KeyError(f"{u} has no observation in flight")
        for u, a in self.st.items():
            if a.k_obs is None or a.done:
                continue
            if u in actions:
                self._settle(a, actions[u])
            else:                                           # 超时：沿用上一导航与云台指令（不产生新的物理档指令），计数并回执 661
                a.timeouts += 1
                r = self.guard[u].timeout(a.k_obs)
                a.acks.extend(r.acks)
                self.rec.on_act(u, a.k_obs, None, r.acks, None)
                a.next_k = a.k_obs + 1
            a.k_obs = None
        while True:
            self._advance()
            if self.k >= self.n_ticks or all(a.done for a in self.st.values()):
                for a in self.st.values():
                    if not a.done:
                        a.term = {"terminated": False, "truncated": True, "reason": "time_limit"}
                break
            if any(not a.done and a.next_k <= self.k for a in self.st.values()):
                break
        obs = self._observe(final=self.k >= self.n_ticks)
        terms = {u: o["obs"]["term"]["terminated"] for u, o in obs.items()}
        truncs = {u: o["obs"]["term"]["truncated"] for u, o in obs.items()}
        for u in obs:
            if terms[u] or truncs[u]:
                self.st[u].done = True
        self.agents = [u for u in self.uavs if not self.st[u].done]
        return obs, dict.fromkeys(obs, 0.0), terms, truncs, {u: {} for u in obs}

    # ------------------------------------------------------------------ 内部
    def _row(self, u: str) -> np.ndarray:
        s = self.tier.history.last_sample(u)
        r = np.zeros(16)
        r[0:3], r[3:7], r[7:10], r[10:13] = s.pos_enu_m, s.q_ws_xyzw, s.vel_enu_mps, s.omega_b_rps
        return r

    def _settle(self, a: _Agent, act: dict) -> None:
        t_wall = time.perf_counter()
        apply_tick = a.k_obs + self.spec.decision.lat_ticks(TICK_MS)
        t_apply = apply_tick * TICK_NS
        row = self._row(a.uav)
        e = self.nav.error(a.no, t_apply)
        nav_pos = row[0:3] + self.nav.error(a.no, a.k_obs * TICK_NS)
        gz = float(self.world.ground_dtm(np.array([nav_pos[:2]]))[0])
        r = self.guard[a.uav].check(act, a.k_obs, nav_pos, _yaw(row[3:7]), e, gz)
        if r.nav is not None:
            n = r.nav
            kind = n["kind"]
            a.cmd_kind = kind
            a.goto_nav = a.goto_true = None
            if kind == "velocity":
                cmd = TierCommand(a.uav, "velocity", apply_tick, vel_enu_mps=n["vel_enu"], yaw_rate_rps=n["yaw_rate"])
            elif kind == "goto":
                a.goto_nav, a.goto_true, a.goto_speed = np.array(n["target_enu_nav"]), np.array(n["target_enu_true"]), n["speed"]
                cmd = TierCommand(a.uav, "goto", apply_tick, target_enu_m=n["target_enu_true"], speed_mps=n["speed"])
            elif kind == "takeoff":
                cmd = TierCommand(a.uav, "takeoff", apply_tick, alt_m=n["alt_m"])
            else:
                cmd = TierCommand(a.uav, kind, apply_tick)
            res = self.tier.submit([cmd])[0]
            if not res.ok:
                self.guard[a.uav].tier_rejected()
                r.acks = [x for x in r.acks if x["item"] != "nav"] + [{"k": a.k_obs, "item": "nav", "index": 0, "result": "rejected",
                                                                     "code": res.code, "why": "tier"}]
        if r.gimbal is not None:
            self.pending_gim.append((apply_tick, a.uav, r.gimbal))
        if r.msgs:
            self.pending_tx.append((apply_tick, a.uav, r.msgs))
            self.link.stats[a.uav]["rejected"] += sum(1 for x in r.acks if x["item"] == "msg" and x["result"] == "rejected")
        for rep in r.reports:
            self.rec.on_report(a.uav, a.k_obs, apply_tick, t_apply, rep)
        self.rec.on_act(a.uav, a.k_obs, act, r.acks, apply_tick)
        a.acks.extend(r.acks)
        a.next_k = apply_tick
        a.wall_s += time.perf_counter() - t_wall

    def _advance(self) -> None:
        self._flush_truth()
        k = self.k
        for a in self.st.values():                          # 估计系目标随导航误差更新（≥ 5 cm）
            if a.cmd_kind == "goto" and a.goto_nav is not None and not a.done:
                tgt = a.goto_nav - self.nav.error(a.no, k * TICK_NS)
                if float(np.linalg.norm(tgt - a.goto_true)) >= 0.05:
                    a.goto_true = tgt
                    self.tier.submit([TierCommand(a.uav, "goto", k, target_enu_m=tuple(float(x) for x in tgt), speed_mps=a.goto_speed)])
        r = self.tier.advance(k)
        for c in r.contacts:
            self.rec.on_contact(c)
        self.status = dict(r.status)
        self.k = k + 1
        t = self.k * TICK_NS
        for g in self.gim.values():
            g.slew(TICK_MS / 1000.0)
        for item in [p for p in self.pending_gim if p[0] <= self.k]:
            self.gim[item[1]].command(*item[2])
        self.pending_gim = [p for p in self.pending_gim if p[0] > self.k]
        self._feed_l2(k)
        self.link.refill(TICK_MS / 1000.0)
        pos = {u: self._row(u)[0:3] for u in self.uavs}
        for apply_tick, u, msgs in [p for p in self.pending_tx if p[0] <= self.k]:
            self.link.transmit(u, msgs, apply_tick * TICK_NS, pos)
        self.pending_tx = [p for p in self.pending_tx if p[0] > self.k]
        for u, msgs in self.link.deliver_due(t).items():
            self.st[u].inbox.extend(msgs)
        for u, a in self.st.items():
            if self.status[u] in (VehicleStatus.AIRBORNE, VehicleStatus.LANDING, VehicleStatus.MOVING):
                a.flight_s += TICK_MS / 1000.0
            if self.status[u] == VehicleStatus.CRASHED and not a.term["terminated"]:
                a.term = {"terminated": True, "truncated": False, "reason": "contact"}
                a.next_k = self.k
        self._sense(self.k)
        self._log_fleet()

    # ------------------------------------------------------------------ M25 会话（Benchmark 锁步）
    def _l2_sensors(self, k: int) -> list[tuple[str, int]]:
        """tick k 每架机的到期帧 (sensor, seq)，顺序与 `L2Sensors.sense_all` 的 `sensors` 映射一致；集合只由规格与 k 决定。"""
        cfg = getattr(self.l2, "cfg", None) or L2Config()
        out = [(CAM_SENSOR, k)]
        if k > 0 and cfg.lidar:
            out.append((LIDAR_SENSOR, k - 1))
        if k > 0 and cfg.imu:
            out.append((IMU_SENSOR, k - 1))
        if cfg.gnss:
            out.append((GNSS_SENSOR, k))
        return out

    def _produce(self, batch):
        """会话的 worker 后端：tick 的首个请求触发一次 `sense_all`（同一实现、同一顺序），其余请求按键取出。
        帧为 `(frame | None, 真值行)`；IMU 窗口不足时为 None（与 `sense_all` 不输出该键一致）。"""
        out = []
        for r in batch:
            k, live = r.payload
            if self._svc_cache is None or self._svc_cache[0] != k:
                self._svc_cache = (k, self.l2.sense_all(k, list(live)))
            frames, truth = self._svc_cache[1][r.key.uav]
            fs = frames.get(r.key.sensor) or []
            fr = next((f for f in fs if int(f["seq"]) == r.key.seq), None)
            out.append((r, (fr, truth if r.key.sensor == CAM_SENSOR else [])))
        return out

    def _sense_m25(self, k: int, live: list[tuple[str, int]]) -> dict:
        from awr.sensorsvc.scheduler import FrameKey, Request, wait_all

        payload = (k, tuple(live))
        reqs = [Request(FrameKey(u, sid, seq), "online", payload) for u, _no in live for sid, seq in self._l2_sensors(k)]
        got, st = wait_all(self._svc, self._sess, reqs)
        self._svc_cache = None
        w = self._svc_wait
        w["ticks"] += 1
        w["rejected_641"] += st.rejected_641
        w["resubmits"] += st.resubmits
        w["wait_s"] += st.wait_s
        w["backpressure_s"] += st.backpressure_s
        w["blocked_ticks"] += int(st.rejected_641 > 0)
        res = {}
        for u, _no in live:
            sensors, truth = {}, []
            for sid, seq in self._l2_sensors(k):
                fr, tr = got[FrameKey(u, sid, seq)]
                if fr is not None:
                    sensors[sid] = [fr]
                truth.extend(tr)
            res[u] = (sensors, truth)
        return res

    def _sense(self, k: int) -> None:
        t = k * TICK_NS
        if self.l2 is not None:
            live = [(u, a.no) for u, a in self.st.items() if not a.done and self.status[u] != VehicleStatus.CRASHED]
            with gpu_errors("l2 sense"):                    # 运行中 CUDA / EGL 设备丢失类异常 → 647（K18）
                res = self._sense_m25(k, live) if self._sess is not None else self.l2.sense_all(k, live)
            for u, (frames, truth) in res.items():
                a = self.st[u]
                for sid, fs in frames.items():
                    a.frames.setdefault(sid, []).extend(fs)
                if isinstance(truth, list):
                    for tr in truth:
                        self.rec.on_vis(tr)
                else:
                    self._truth_q.append(truth)
                if self._wall is not None:
                    self._wall.on_frames(u, k, frames.get(CAM_SENSOR) or [])
            return
        env = self.tape.env_at(t)
        for u, a in self.st.items():
            if a.done or self.status[u] == VehicleStatus.CRASHED:
                continue
            row = self._row(u)
            yaw = _yaw(row[3:7])
            g = self.gim[u]
            c, s = math.cos(yaw), math.sin(yaw)
            mt = self.cam.mount_t_bs_m
            o = row[0:3] + np.array([c * mt[0] - s * mt[1], s * mt[0] + c * mt[1], mt[2]])
            dets, truth = self.l1.frame(a.no, k, t, o, cam_axes(yaw, g.pitch_deg, g.yaw_deg), self.nav.error(a.no, t), env)
            for d, tr in zip(dets, truth, strict=True):
                self.rec.on_l1(u, k, t, d, tr)
            a.dets.extend(dets)

    def _observe(self, final: bool = False) -> dict:
        out = {}
        t = self.k * TICK_NS
        for u, a in self.st.items():
            if a.done or (a.next_k > self.k and not final and not a.term["terminated"]):
                continue
            row = self._row(u)
            nav, _e = self.nav.solution(a.no, self.k, t, row, a.flight_s)
            st = self.status[u]
            xy = np.array([nav["pos_enu_m"][:2]])
            reached = a.goto_nav is not None and float(np.linalg.norm(np.asarray(nav["pos_enu_m"]) - a.goto_nav)) < 1.0
            obs = {"schema": "drone4d.bench.obs.v1", "k": self.k, "t_ns": t, "nav": nav,
                   "status": {"mode": _MODE[st], "nav_cmd": _NAV_CMD.get(self.tier.tracker.kind(u), "hover"), "goto_reached": bool(reached),
                              "contact": st in (VehicleStatus.GROUNDED, VehicleStatus.CRASHED, VehicleStatus.MOVING, VehicleStatus.AFLOAT),
                              "safety": "low_battery" if nav["battery_soc"] < 0.2 else "normal", "locked_sensors": []},
                   "ground_hint": {"dsm_m": float(self.world.height_dsm(xy)[0]), "dtm_m": float(self.world.ground_dtm(xy)[0])},
                   "sensors": a.frames if self.l2 is not None else {"detections": a.dets}, "acks": a.acks, "comm": self.link.status(u), "dropped": {},
                   "term": a.term if (a.term["terminated"] or a.term["truncated"]) else {"terminated": False, "truncated": False, "reason": "none"}}
            msg = {"obs": obs, "inbox": a.inbox}
            self.guard[u].saw(a.dets)
            if self.l2 is not None:                         # L2 的 report 证据引用交付的相机帧（框证据，无 det_id）
                self.guard[u].saw_frames(a.frames)
            if self.async_obs_record:
                if self._rec_pool is None:
                    from concurrent.futures import ThreadPoolExecutor

                    self._rec_pool = ThreadPoolExecutor(1, thread_name_prefix="env-obs-rec")
                self._rec_futs.append(self._rec_pool.submit(self.rec.on_obs, u, self.k, msg))
            else:
                self.rec.on_obs(u, self.k, msg)
            out[u] = msg
            a.dets, a.acks, a.inbox, a.frames = [], [], [], {}
            a.k_obs = self.k
        return out

    def _log_fleet(self) -> None:
        self._fleet_t.append(self.k * TICK_NS)
        rows = [self._row(u) for u in self.uavs]
        self._fleet["pos"].append([r[0:3] for r in rows])
        self._fleet["vel"].append([r[7:10] for r in rows])
        self._fleet["yaw"].append([_yaw(r[3:7]) for r in rows])
        self._fleet["nav"].append([r[0:3] + self.nav.error(i, self.k * TICK_NS) for i, r in enumerate(rows)])
        self._fleet["status"].append([str(self.status[u]) for u in self.uavs])

    def close(self) -> dict:
        self._flush_obs()
        if self._rec_pool is not None:
            self._rec_pool.shutdown(wait=True)
            self._rec_pool = None
        self._flush_truth()
        info = self.tier.close() if hasattr(self, "tier") else {}
        svc = None
        if self._sess is not None:
            svc = {**self._svc.close_session(self._sess), **{k: round(v, 3) if isinstance(v, float) else v for k, v in self._svc_wait.items()},
                   "delay_s": self._svc.delay_s}
            self._sess = None
        if self.l2 is not None:
            self.l2.close()
        wall = time.perf_counter() - self._t0 if hasattr(self, "_t0") else 0.0
        self.rec.net(self.link.records)
        self.rec.fleet(self.uavs, self._fleet_t, self._fleet["pos"], self._fleet["vel"], self._fleet["yaw"], self._fleet["nav"],
                       self._fleet["status"])
        stats = {u: {"timeouts": a.timeouts, "flight_s": round(a.flight_s, 1), "agent_wall_s": round(a.wall_s, 3),
                     "comm": self.link.stats[u], "term": a.term} for u, a in self.st.items()}
        manifest = {"schema": "drone4d.bench.env_run.v1", "run_id": self.spec.run_id, "world": Path(self.spec.world_dir).name,
                    "event": {"id": self.tape.header["event_id"], "version": self.tape.header["event_version"],
                              "content_sha256": self.tape.hashes()["content_sha256"]},
                    "task_id": self.task["task_id"], "fleet": [{"uav": f.uav, "vehicle_ref": f.vehicle_ref, "pos_enu_m": list(f.pos_enu_m),
                                                                "yaw_rad": f.yaw_rad} for f in self.spec.fleet],
                    "obs_level": self.spec.obs_level, "noise": self.spec.noise, "physics": self.spec.physics, "tier": getattr(self.tier, "name", None),
                    "ranked": self.spec.ranked, "policy": self.spec.policy, "decision": self.spec.decision.config(),
                    "comm": {"enabled": self.spec.comm.enabled, "bucket_Bps": self.spec.comm.bucket_Bps},
                    "duration_s": self.duration_s, "n_ticks": self.k, "pose_sha256": info.get("pose_sha256"), "tier_info": info,
                    "wall_s": round(wall, 2), "l1_rays": self.l1.n_rays, "agents": stats, "tape_check": self.tape_check,
                    "world_query": self.world_query, "seed_store": getattr(self, "seed_store", None)}
        if not self.spec.ranked or self.spec.seed_commit:     # 运行身份（M27-FR-007）；排名运行须有 M27 的种子承诺值
            manifest.update(self.spec.identity())
        if self.cam != CameraModel():                       # L1 相机不是标称相机（相机覆盖，ADR-217）：评测器的 AO oracle 按此相机计算
            manifest["camera"] = self.cam.manifest()
        if self.l2 is not None:                             # 网格 LiDAR 后端（cpu：Open3D 参考；warp：M25 GPU 产品路径）
            manifest["l2"] = {"lidar_backend": getattr(self.l2, "lidar_backend", None)}
            probe = getattr(self.l2, "gpu_probe", None)
            if isinstance(probe, dict):                     # 建立 GPU 上下文之前的体检（`awr.sensorsvc.gpuhealth.require_gpu`）
                manifest["l2"]["gpu_probe"] = {k: probe[k] for k in ("mode", "expected", "warp_cuda_devices") if k in probe}
        if svc is not None:
            manifest["sensorsvc"] = svc
        if self._wall is not None:
            manifest["wall_preview"] = self._wall.summary()
        self.rec.write_json("manifest.json", manifest)
        self.rec.close()
        return manifest


class _WallPreview:
    """观测墙预览（ADR-173；M27 FR-171）：从交付给 Agent 的相机帧按仿真时间 2 Hz 降采样为 480 × 270 JPEG（质量 75），
    写入 `preview/<uav>/<seq:06d>.jpg` 或交给回调；不经 M25 队列，不影响观测。"""

    HZ, SIZE, QUALITY = 2.0, (480, 270), 75

    def __init__(self, out_dir, target) -> None:
        self.sink = target if callable(target) else None
        self.dir = Path(out_dir) / "preview" if (self.sink is None and out_dir) else None
        self.every = max(1, round(1000 / (self.HZ * TICK_MS)))
        self.n: dict[str, int] = {}

    def on_frames(self, uav: str, k: int, frames: list[dict]) -> None:
        import io

        from PIL import Image

        for f in frames:
            if int(f["seq"]) % self.every:
                continue
            im = Image.open(io.BytesIO(f["rgb"]["data"])).convert("RGB").resize(self.SIZE, Image.BILINEAR)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=self.QUALITY)
            b = buf.getvalue()
            if self.sink is not None:
                self.sink(uav, int(f["seq"]), int(f["t_mid_ns"]), b)
            elif self.dir is not None:
                p = self.dir / uav / f"{int(f['seq']):06d}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b)
            self.n[uav] = self.n.get(uav, 0) + 1

    def summary(self) -> dict:
        return {"hz": self.HZ, "size": list(self.SIZE), "jpeg_quality": self.QUALITY, "frames": dict(self.n), "source": "delivered"}


SEED_FILE = "run_seed.json"


def seed_sink_for(spec: EnvSpec) -> tuple[Path | None, bool]:
    """排名运行种子明文的隐藏区目录 → (目录, 是否显式给出)：`spec.seed_sink`；未给出而运行目录为 M27 数据集布局
    `<data>/dataset/<eid>/runs/<run_id>`（subworker、board_v1_formal 同）时取同一数据根的 `<data>/hidden/<eid>/truth/runs/<run_id>`
    （`HiddenStore.truth_dir` 与 `seal-truth` 的布局）；其余为 None。"""
    if getattr(spec, "seed_sink", None) is not None:
        return Path(spec.seed_sink), True
    if spec.out_dir is None:
        return None, False
    p = Path(spec.out_dir)
    if p.parent.name == "runs" and p.parent.parent.parent.name == "dataset":
        return p.parent.parent.parent.parent / "hidden" / p.parent.parent.name / "truth" / "runs" / p.name, False
    return None, False


def _mkdir_hidden(d: Path) -> None:
    """建目录链，新建的各级设为 0700（隐藏区 `HiddenStore._mkdir` 的同一口径；已有目录不改属主与权限）。"""
    missing = [x for x in (d, *d.parents) if not x.exists()]
    d.mkdir(parents=True, exist_ok=True)
    for x in missing:
        os.chmod(x, 0o700)


def write_run_seed(spec: EnvSpec, rec: EnvRecorder | None = None) -> str | None:
    """运行种子明文的落点（ADR-161 决策 (3)、M27-FR-005；K18，B21 遗留 8）→ 运行清单 `seed_store`：

    - 不排名运行：写入运行目录真值层 `truth/run_seed.json`（不排名运行的种子本就公开，Rerun 导出与复现取用），记 `run_dir`；
    - 排名运行：运行目录不留种子明文，只写入隐藏区目录（`seed_sink_for`：显式 `spec.seed_sink`，或数据集布局下同一数据根的
      `hidden/<eid>/truth/runs/<run_id>/`）的 `run_seed.json`（新建目录 0700、文件 0600，先写临时文件再改名），记 `hidden`；评测
      catalog 的导出（`rerun_export`）从该处取种子。没有隐藏区目录时不写，种子以 M27 评测库（`EvalStore.seeds`，与 `seed_commit`
      的 HMAC 承诺对应）为准，记 `evalstore`。由布局推导的隐藏区不可写（例如属主为 Evaluator 服务用户）时同样不写并记 warning
      （不含种子）；显式给出的目录不可写时抛出。
    运行目录未给出（`out_dir = None`）且不排名时不写盘，记 None。"""
    if not spec.ranked:
        if rec is None or not getattr(rec, "dir", None):
            return None
        rec.write_json(f"truth/{SEED_FILE}", {"run_seed": spec.seed})
        return "run_dir"
    sink, explicit = seed_sink_for(spec)
    if sink is None:
        return "evalstore"
    import json as _json

    tmp = sink / f".{SEED_FILE}.{os.getpid()}.tmp"
    try:
        _mkdir_hidden(sink)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_json.dumps({"run_seed": int(spec.seed), "run_id": spec.run_id}, sort_keys=True) + "\n")
        os.replace(tmp, sink / SEED_FILE)
    except OSError as e:
        if explicit:
            raise
        log.warning("run %s: hidden seed store %s is not writable (%s); the seed stays only in the EvalStore", spec.run_id, sink,
                    type(e).__name__)
        return "evalstore"
    return "hidden"


class WorldQueryFallbackRefused(RuntimeError):
    """正式运行的网格查询打不开时拒绝退回 DSM（K07；A10 遗留 F-78 (c)）：`detail` 含世界目录与网格查询的错误。"""

    def __init__(self, world_dir: Path, why: str, error: BaseException | None = None) -> None:
        msg = f"formal runs need the mesh world query for {Path(world_dir).name}: {why}"
        super().__init__(msg + (f" ({type(error).__name__}: {error})" if error is not None else ""))
        self.detail = {"why": "mesh_query_unavailable", "world": Path(world_dir).name, "reason": why,
                       **({"error": f"{type(error).__name__}: {str(error)[:300]}"} if error is not None else {})}


def world_query_info(world) -> dict:
    """运行清单的 `world_query`：按查询对象的类型判定 `mesh`（`MeshWorldQuery`）、`dsm`（`DsmWorldQuery`）或 `other`（测试替身）。"""
    names = [c.__name__ for c in type(world).__mro__]
    kind = "mesh" if "MeshWorldQuery" in names else "dsm" if "DsmWorldQuery" in names else "other"
    return {"kind": kind, "class": type(world).__name__}


def open_world_query(world_dir: Path, *, formal: bool = False):
    """→ (世界查询, `world_query` 记录)。网格查询打不开时：`formal` 为真抛 `WorldQueryFallbackRefused`；否则记录 warning，
    退回 DSM 查询，并在记录的 `fallback` 中给出原因（`no_mesh_layer`：世界包没有 `mesh.raycast` 图层，如 DSM 合成夹具；
    `mesh_open_failed`：图层存在但打开失败，如派生缓存不可写）。"""
    world_dir = Path(world_dir)
    try:
        from awr.world.mesh.query import open_mesh_world_query

        w = open_mesh_world_query(world_dir)
        return w, world_query_info(w)
    except Exception as e:
        why = "mesh_open_failed" if _has_mesh_layer(world_dir) else "no_mesh_layer"
        if formal:
            raise WorldQueryFallbackRefused(world_dir, why, e) from e
        log.warning("world %s: mesh query unavailable (%s: %s: %s); falling back to the DSM query (non-formal run)",
                    world_dir.name, why, type(e).__name__, str(e)[:300])
        from awr.world.geometry.query import open_dsm_world_query

        w = open_dsm_world_query(world_dir)
        return w, {**world_query_info(w), "fallback": {"reason": why, "error": f"{type(e).__name__}: {str(e)[:300]}"}}


def _has_mesh_layer(world_dir: Path) -> bool:
    try:
        import json as _json

        doc = _json.loads((Path(world_dir) / "world.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    layers = doc.get("layers") or []
    ids = [x.get("id") for x in layers] if isinstance(layers, list) else list(layers)
    return "mesh.raycast" in ids


def _open_world(world_dir: Path, *, formal: bool = False):
    """世界查询（只要对象时的简写；规则见 `open_world_query`）。"""
    return open_world_query(Path(world_dir), formal=formal)[0]


SPAWN_FOOT = np.array([[dx, dy] for dx in (-0.7, -0.35, 0.0, 0.35, 0.7) for dy in (-0.7, -0.35, 0.0, 0.35, 0.7)])
SPAWN_LEVEL_M = 0.15


class _GzTier:
    """Realistic 档：每 Run 一个 gz 实例（GzInstance + compose_run + SupervisedGzBackend：看门狗 632 / 638 时重启实例并从回合起点
    重放，M24-FR-078、FR-079），关闭时停止实例。机型按 `fleet[].vehicle_ref` 映射（`awr.phys.gz.vehicles`，M24-FR-083）：`p600_bench`
    在 x500_d4d 几何上换用其质量、惯量与最大转速，派生模型写入 `<run_dir>/models/` 并只读挂载进 gz 实例；其余机型用 x500 原型。
    映射参数写入运行清单 `tier_info.vehicle`。"""

    @staticmethod
    def create(spec: EnvSpec, spawn: dict, grounded: list[str], world):
        from awr.agentsdk.env.geom import surface_z
        from awr.phys.gz.backend import SupervisedGzBackend
        from awr.phys.gz.compose import UavSpawn, compose_run
        from awr.phys.gz.instance import GzInstance
        from awr.phys.gz.vehicles import apply_vehicle, base_model_sdf, quad_for_fleet, write_model
        from awr.phys.kinematic import GEAR_Z_M
        from awr.world.export.gz import export_world

        o = spec.gz
        exp = export_world(Path(spec.world_dir), out_root=o.get("export_root"), check_sdf=o.get("check_sdf"))
        run_dir = Path(o.get("run_dir") or (Path(spec.out_dir) / "phys" if spec.out_dir else Path("/tmp") / f"d4d-{spec.run_id}"))
        launcher = o.get("launcher", "docker")
        q = quad_for_fleet(f.vehicle_ref for f in spec.fleet)
        mounts = {}
        md = write_model(run_dir / "models", q, base_model_sdf()) if q.mapped else None
        if md is not None:
            if launcher == "docker":
                mounts[str(md)] = f"/opt/d4d/models/{q.model}"
            else:
                rp = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
                if str(md.parent) not in rp.split(":"):
                    os.environ["GZ_SIM_RESOURCE_PATH"] = ":".join(p for p in (str(md.parent), rp) if p)
        inst = GzInstance(run_dir, o.get("name", f"d3-gz-{spec.run_id}"), image=o.get("image", "d3-gz:dev"), launcher=launcher,
                          world_export=exp.root, extra_mounts=mounts)
        spawns = []
        for u, (p, yaw) in spawn.items():
            z = float(p[2])
            if u in grounded:
                foot = np.asarray(p[:2], np.float64) + SPAWN_FOOT
                zs = surface_z(world, foot)
                if float(np.ptp(zs)) > SPAWN_LEVEL_M:            # 机体跨在台阶或坑上：起飞失败或侧向碰撞，运行无效
                    raise ValueError(f"{u}: spawn surface not level ({np.ptp(zs):.2f} m over the airframe footprint)")
                z = float(zs.max()) + GEAR_Z_M + 0.01
            spawns.append(UavSpawn(u, (float(p[0]), float(p[1]), z), yaw))
        inst.start(apply_vehicle(compose_run(spawns, world_static_uri=exp.uri, endpoint=inst.gz_endpoint, spherical=exp.spherical), q, len(spawns)))
        try:
            inst.wait_socket(float(o.get("ready_timeout_s", 120)))
            be = SupervisedGzBackend(inst, list(spawn), grounded=grounded, run_id=spec.run_id, step_timeout_s=float(o.get("step_timeout_s", 5)),
                                     reuse_policy=o.get("reuse_policy", "per_run"))
        except Exception:
            inst.stop()
            raise
        return _GzHandle(be, inst, q.as_dict())


class _GzHandle:
    def __init__(self, be, inst, vehicle: dict | None = None) -> None:
        self.be, self.inst, self.vehicle = be, inst, vehicle
        self.name = be.name

    def __getattr__(self, k):
        return getattr(self.be, k)

    def close(self) -> dict:
        try:
            info = self.be.close()
            return {**info, "vehicle": self.vehicle} if self.vehicle is not None and isinstance(info, dict) else info
        finally:
            self.inst.stop()
