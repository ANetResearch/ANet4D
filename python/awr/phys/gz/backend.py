"""GzSimBackend（M24 §6.5.5、§6.5.9；Realistic 档宿主侧）：经 `drone4d.gz.v1` 锁步推进 gz-sim，样本写入 PoseHistoryBuffer。

- `open()`：Hello → HelloAck（就绪屏障）；`t0_gz` 取 HelloAck.t_ns，HelloAck.initial_state 作为回合 t = 0 的样本写入。
- `submit()`：TierCommand 按 `apply_tick` 排队（orbit 与 follow_path 的参数校验与 Fast 档相同，P09）；`advance(k)` 在 tick 起点把到期指令交给跟踪器，以 tick 起点状态求一次设定，
  与上次下发相同的设定省略（接收端零阶保持），Control 授予 `steps_per_tick` 步。
- StepDone 逐项校验 seq、tick、n_uav 与网格 `t_ns[j] − t0_gz = t_k + (j + 1) · step_ns`（635），整块写入 PoseHistory。
- 接触（§6.5.9）：gz 侧分类触地开始、离地、撞击；宿主判定落地（触地且 ‖v‖ < 0.25 m/s 持续 0.5 s（land）或 1.0 s（其他），
  随后电机失能）与坠毁（IMPACT、COLLISION_WORLD、COLLISION_UAV，电机零转速，保持）。TickResult 的 TOUCHDOWN 在落地判定成立时给出，
  地面出生的机体处于落地状态、电机失能，直到首条起飞类指令。
- 看门狗（M24-FR-079、M24-AC-013）：等待 StepDone 超过 `step_timeout_s`（缺省 5 s 墙钟，run_spec `physics_opts.step_timeout_s`）
  记 632；等待期间每 `poll_s` 经 `alive()`（通常为 `GzInstance.alive`）检查实例，进程已退出记 638（≤ 1 s 识别）。Hello 与 Shutdown
  沿用 `timeout_s`。632 / 638 之后套接字作废（REQ 状态机已失步），由 `SupervisedGzBackend` 重启实例并重放。
- 每 tick 记录控制消息与样本块的 sha256（`last_control_sha256`、`last_block_sha256`），供重放逐 tick 比对。
- 发散守卫（K18，B21 遗留 5）：机间碰撞后 gz 中已坠毁机体的状态可能发散（位置约 1e124、速度溢出，录制溢出告警）。样本块写入
  PoseHistory 之前逐机逐步检查：状态非有限、速度 > `GUARD_V_MPS`、角速度 > `GUARD_OMEGA_RPS`、位置分量绝对值 > `GUARD_POS_M`，
  或已坠毁机体离坠毁点 > `GUARD_DRIFT_M` 即判发散。发散机体自该步起冻结为最后一个合理样本（速度与角速度置零），此后各 tick
  的样本均为冻结值；未坠毁机体发散按坠毁处理（`crash = impact`，给出 IMPACT 接触，Env 终止该机），不再把发散值交给 Env 与录制。
  守卫只取决于样本值，重放逐位一致；触发记录在 `close()` 的 `crash_guard`。
- field 风模式（M24-FR-046，§6.5.8）：`wind` 为 `awr.phys.gz.wind.FieldWind` 时，每 tick 以 tick 起点状态求本 tick 的逐机风矢量
  放入 `Control.wind`（`Drone4DClock` 写入 D4dWind，`Drone4DWindDrag` 施力）；重放时 FieldWind 返回缓存，控制消息逐字节相同。

`SupervisedGzBackend`（M24-FR-078、§6.8.2 RUNNING → PREPARING；D3-AC-17、35）：包装 GzSimBackend 与 GzInstance，以同一 PhysicsTier
接口推进；逐 tick 记录 submit 批次（控制日志的宿主侧来源）及控制消息、样本块摘要。632 / 638 时按进程组回收并重启实例，新建后端从回合
起点按日志重放到故障前 tick，逐 tick 比对控制消息与样本块摘要，逐位一致后沿用原 PoseHistoryBuffer 继续；任何不一致记 638 并标记
`detail.platform_failure = True`（M27 判平台失败、整体重跑，不计入提交方）。同一故障码连续 `max_consecutive`（缺省 3）次即失败。

实例生命周期（M24-FR-045、FR-060；D3-AC-35）：`REUSE_POLICY = "per_run"`。同种子 reset 复用验证（`verify_reset_reuse`，
tests/d3/phys/determinism/test_fault.py）判定插件不支持 Reset、世界复位后无法回到回合起点，因此一律每 Run 重启；`same_seed` 被拒绝（630）。
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from awr.phys import PhysError
from awr.phys.adapter import (
    LAND_STILL_S,
    STILL_S,
    STILL_V_MPS,
    TAKEOFF_GRACE_S,
    ContactEvent,
    DispatchResult,
    StatusSignals,
    TickResult,
    TickTiming,
    TierCommand,
    VehicleStatus,
    derive_status,
)
from awr.phys.gz import protocol as P
from awr.phys.gz.compose import STEP_NS
from awr.phys.pose_history import PoseHistoryBuffer
from awr.phys.tracker import Setpoint, Tracker

__all__ = ["RECOVERABLE", "REUSE_POLICY", "GzSimBackend", "SupervisedGzBackend", "verify_reset_reuse"]

_KIND = {1: "TOUCHDOWN", 2: "LIFTOFF", 3: "COLLISION_WORLD", 4: "IMPACT", 5: "COLLISION_UAV"}
_CRASH = {"COLLISION_WORLD": "world", "IMPACT": "impact", "COLLISION_UAV": "uav"}
_FLY = frozenset({"takeoff", "goto", "velocity", "orbit", "follow_path"})

RECOVERABLE = frozenset({632, 638})                  # 步进超时、实例退出：重启实例 + 从回合起点重放
REUSE_POLICY = "per_run"                             # M24-FR-060 验证结论（verify_reset_reuse）；same_seed 未通过验证
MAX_PATH_POINTS = 256                                # follow_path 航点上限（与 Fast 档相同）
STEP_TIMEOUT_S = 5.0                                 # M24-FR-079 gz_step_timeout_s；run_spec physics_opts.step_timeout_s 缺省
GUARD_V_MPS = 100.0                                  # 发散守卫：速度上限（x500 平飞上限约 20 m/s，150 m 自由落体约 54 m/s）
GUARD_OMEGA_RPS = 1000.0                             # 角速度上限（约 160 转/s，坠毁翻滚远低于此值）
GUARD_POS_M = 1.0e5                                  # 位置分量绝对值上限（世界包范围为数公里）
GUARD_DRIFT_M = 500.0                                # 已坠毁机体离坠毁点的位移上限


@dataclass
class _Veh:
    uav: str
    idx: int
    landed: bool
    in_contact: bool
    still_ns: int = 0
    grace_ns: int = 0                    # 起飞宽限剩余（飞行指令生效时置为 TAKEOFF_GRACE_S，首次离地即清零）
    crash: str = "none"
    sent: Setpoint | None = None
    crash_pos: np.ndarray | None = None  # 坠毁步的机体位置，发散守卫的位移基准
    frozen: np.ndarray | None = None     # 发散守卫冻结的状态行（16）；None 未冻结


@dataclass
class GzSimBackend:
    uavs: Sequence[str]
    endpoint: str
    tick_ms: int = 100
    step_ns: int = STEP_NS
    grounded: Sequence[str] | None = None            # 地面出生的机体；缺省全部
    tracker: Tracker = field(default_factory=Tracker)
    history_s: float = 2.0
    retain_s: float | None = 1.0                     # 每 tick 后释放早于 t_end − retain_s 的样本；None 由调用方释放
    run_id: str = ""
    attempt: int = 0
    timeout_s: float = 30.0                          # Hello / Shutdown
    step_timeout_s: float = STEP_TIMEOUT_S           # 看门狗：单 tick 等待 StepDone 的墙钟上限（632）
    alive: Callable[[], bool] | None = field(default=None, repr=False)   # 实例存活检查（638）；None 只靠超时
    poll_s: float = 0.2                              # 等待回复期间检查 alive() 的间隔
    name: str = "gz"                                 # the physics tier name (`physics = gz`; "realistic" before 5e, ADR-226)
    wind: Any = None                                 # field 风模式采样器（FieldWind）；None 为 uniform / 无风

    def __post_init__(self) -> None:
        self.uavs = tuple(self.uavs)
        if (self.tick_ms * 1_000_000) % self.step_ns:
            raise PhysError(637, f"tick {self.tick_ms} ms is not a multiple of the {self.step_ns} ns step")
        self.n_steps = self.tick_ms * 1_000_000 // self.step_ns
        self.tracker.tick_s = self.tick_ms / 1000.0                    # orbit 参考点按 tick 推进
        g = set(self.uavs if self.grounded is None else self.grounded)
        self._v = {u: _Veh(u, i, landed=u in g, in_contact=u in g) for i, u in enumerate(self.uavs)}
        self._pending: list[TierCommand] = []
        self.history = PoseHistoryBuffer(self.uavs, hz=1_000_000_000 // self.step_ns, capacity_s=self.history_s)
        self._sock = None
        self._ctx = None
        self.seq = 0
        self.t_k_ns = 0
        self.t0_gz_ns: int | None = None
        self.hello_ack: dict = {}
        self._rtf: list[float] = []
        self._ticks = 0
        self.digest = hashlib.sha256()                   # 样本流摘要（t_ns i64 ‖ state f64，逐 tick 顺序；确定性对照）
        self.open_sha256 = b""                           # HelloAck 初始样本块的 sha256
        self.last_control_sha256 = b""                   # 最近一个 tick 的 Control 信封字节 sha256
        self.last_block_sha256 = b""                     # 最近一个 tick 的样本块（t_rel ‖ state）sha256
        self.guard_events: list[dict] = []               # 发散守卫触发记录（uav、t_ns、reason、crashed_before）

    # ------------------------------------------------------------ 连接
    def open(self) -> dict:
        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))
        self._sock.connect(self.endpoint)
        hello = P.msg("Hello", run_id=self.run_id, attempt=self.attempt, uav_models=list(self.uavs), step_ns=self.step_ns,
                      tick_steps=self.n_steps)
        body, ack = self._rpc("hello", hello, self.timeout_s)
        if body != "hello_ack" or ack.n_uav != len(self.uavs):
            raise PhysError(635, f"expected HelloAck for {len(self.uavs)} vehicles, got {body}")
        self.t0_gz_ns = int(ack.t_ns)
        init = np.frombuffer(ack.initial_state, "<f8").reshape(len(self.uavs), P.STATE_N)
        self.history.append_block(np.array([0], np.int64), init[None])
        blk = np.array([0], "<i8").tobytes() + np.ascontiguousarray(init, "<f8").tobytes()
        self.digest.update(blk)
        self.open_sha256 = hashlib.sha256(blk).digest()
        self.hello_ack = {"gz_sim_version": ack.gz_sim_version, "plugin_build": ack.plugin_build, "t0_gz_ns": self.t0_gz_ns}
        if hasattr(self.tracker, "bind"):                                # RefTracker（P09）：影子参考状态按初始样本出生
            self.tracker.bind(self.uavs, {u: init[i] for i, u in enumerate(self.uavs)}, [u for u, v in self._v.items() if v.landed])
        return self.hello_ack

    def _rpc(self, body: str, m, timeout_s: float, raw_out: list | None = None):
        import zmq

        raw_req = P.encode(body, m)
        if raw_out is not None:
            raw_out.append(raw_req)
        self._sock.send(raw_req)
        deadline = time.monotonic() + timeout_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self.abandon()
                raise PhysError(632, f"no reply from gz within {timeout_s:g} s ({body})", detail={"reason": "watchdog", "timeout_s": timeout_s})
            if self._sock.poll(int(min(self.poll_s, left) * 1000) + 1, zmq.POLLIN):
                raw = self._sock.recv()
                break
            if self.alive is not None and not self.alive():
                self.abandon()
                raise PhysError(638, f"gz instance exited while waiting for {body}", detail={"reason": "instance_exited"})
        rb, rm = P.decode(raw)
        if rb == "error":
            raise PhysError(int(rm.code) or 635, rm.detail)
        return rb, rm

    # ------------------------------------------------------------ 指令
    def submit(self, cmds: Sequence[TierCommand]) -> list[DispatchResult]:
        out = []
        for c in cmds:
            if c.uav not in self._v:
                out.append(DispatchResult(c.uav, False, 630, detail="unknown vehicle"))
            elif c.kind == "follow_path" and (c.path_enu_m is None or not 1 <= len(c.path_enu_m) <= MAX_PATH_POINTS):
                out.append(DispatchResult(c.uav, False, 630, detail=f"follow_path needs 1..{MAX_PATH_POINTS} points"))
            elif c.kind == "orbit" and (c.orbit is None or not 1.0 <= c.orbit.radius_m <= 1000.0):
                out.append(DispatchResult(c.uav, False, 630, detail="orbit radius must be in [1, 1000] m"))
            else:
                self._pending.append(c)
                out.append(DispatchResult(c.uav, True))
        return out

    def _due(self, k: int) -> None:
        keep = []
        for c in sorted(self._pending, key=lambda c: c.apply_tick):
            if c.apply_tick > k:
                keep.append(c)
                continue
            v = self._v[c.uav]
            if v.crash != "none":
                continue
            self.tracker.set(c, self.history.last_sample(c.uav).pos_enu_m)
            if c.kind in _FLY:
                if v.landed:
                    v.grace_ns = int(TAKEOFF_GRACE_S * 1e9)
                v.landed = False
            v.still_ns = 0
        self._pending = keep

    def _observation(self) -> dict[str, tuple]:
        out = {}
        for u, v in self._v.items():
            s = self.history.last_sample(u)
            row = np.concatenate([s.pos_enu_m, s.q_ws_xyzw, s.vel_enu_mps, s.omega_b_rps, s.f_b_mps2])
            out[u] = (row, v.in_contact, v.landed, v.crash != "none")
        return out

    def _setpoint(self, v: _Veh) -> Setpoint:
        if v.crash != "none" or v.landed:
            return Setpoint((0.0, 0.0, 0.0), 0.0, enabled=False)
        s = self.history.last_sample(v.uav)
        return self.tracker.setpoint(v.uav, s.pos_enu_m, s.vel_enu_mps, s.q_ws_xyzw)

    # ------------------------------------------------------------ 推进
    def advance(self, k: int) -> TickResult:
        if self._sock is None:
            raise PhysError(631, "backend is not open")
        tm = TickTiming()
        w0 = time.perf_counter_ns()
        planned = None
        if hasattr(self.tracker, "plan"):                                # RefTracker：tick 起点状态 → 逐物理步设定（apply_step 0..n−1）
            self.tracker.observe(self.t_k_ns, self._observation())
            self._due(k)
            self.tracker.observe(self.t_k_ns, self._observation())
            planned = self.tracker.plan(k, self.t_k_ns, self.n_steps)
        else:
            self._due(k)
        sps = []
        for v in self._v.values():
            for j, sp in enumerate(planned[v.uav] if planned is not None else (self._setpoint(v),)):
                if sp != v.sent:
                    sps.append(P.msg("VelSp", uav_index=v.idx, apply_step=j, vx=sp.v_enu[0], vy=sp.v_enu[1], vz=sp.v_enu[2],
                                     yaw_rate=sp.yaw_rate, enabled=sp.enabled))
                    v.sent = sp
        wind = self.wind.for_tick(k, self.t_k_ns, self.n_steps, self.history) if self.wind is not None else []
        ctl = P.msg("Control", seq=self.seq + 1, tick=k, grant_steps=self.n_steps, setpoints=sps, wind=wind)
        w1 = time.perf_counter_ns()
        tm.encode = w1 - w0
        raw_ctl: list[bytes] = []
        body, sd = self._rpc("control", ctl, self.step_timeout_s, raw_ctl)
        w2 = time.perf_counter_ns()
        tm.physics = w2 - w1
        if body != "step_done":
            raise PhysError(635, f"expected StepDone, got {body}")
        t_rel = np.asarray(sd.t_ns, np.int64) - self.t0_gz_ns
        expect = self.t_k_ns + self.step_ns * np.arange(1, self.n_steps + 1, dtype=np.int64)
        if sd.seq != ctl.seq or sd.tick != k or sd.n_uav != len(self.uavs) or sd.n_steps != self.n_steps:
            raise PhysError(635, f"StepDone seq {sd.seq} tick {sd.tick} n_uav {sd.n_uav} n_steps {sd.n_steps} does not match")
        if not np.array_equal(t_rel, expect):
            raise PhysError(635, f"StepDone times {t_rel[:2].tolist()}... are off the grid {expect[:2].tolist()}...")
        state, guard_contacts = self._guard(P.samples_array(sd, len(self.uavs)), t_rel)
        w3 = time.perf_counter_ns()
        tm.decode = w3 - w2
        self.history.append_block(t_rel, state)
        blk = t_rel.astype("<i8").tobytes() + np.ascontiguousarray(state, "<f8").tobytes()
        self.digest.update(blk)
        self.last_block_sha256 = hashlib.sha256(blk).digest()
        self.last_control_sha256 = hashlib.sha256(raw_ctl[0]).digest()
        contacts = guard_contacts + self._contacts(sd, t_rel, state)
        t_start = self.t_k_ns
        self.t_k_ns = int(expect[-1])
        self.seq = int(ctl.seq)
        if self.retain_s is not None:
            self.history.release_before(self.t_k_ns - int(self.retain_s * 1e9))
        tm.append = time.perf_counter_ns() - w3
        self._rtf.append(float(sd.gz_rtf))
        self._ticks += 1
        return TickResult(k, t_start, self.t_k_ns, self.n_steps, tuple(contacts), self.status(), tm)

    def _guard(self, state: np.ndarray, t_rel: np.ndarray) -> tuple[np.ndarray, list[ContactEvent]]:
        """发散守卫（见模块说明）：→ (守卫后的样本块, 未坠毁机体判发散时的 IMPACT 接触)。无触发且无冻结机体时原样返回。"""
        out: np.ndarray | None = None
        contacts: list[ContactEvent] = []
        for v in self._v.values():
            if v.frozen is None:
                col = state[:, v.idx, :]
                with np.errstate(invalid="ignore", over="ignore"):
                    bad = ~np.isfinite(col).all(axis=1)
                    fin = np.where(np.isfinite(col), col, 0.0)
                    speed = np.linalg.norm(fin[:, 7:10], axis=1)
                    omega = np.linalg.norm(fin[:, 10:13], axis=1)
                    far = np.abs(fin[:, 0:3]).max(axis=1) > GUARD_POS_M
                    drift = (np.linalg.norm(fin[:, 0:3] - v.crash_pos, axis=1) > GUARD_DRIFT_M if v.crash_pos is not None
                             else np.zeros(len(col), bool))
                why = np.where(bad, 1, np.where(speed > GUARD_V_MPS, 2, np.where(omega > GUARD_OMEGA_RPS, 3, np.where(far, 4,
                                                                                                              np.where(drift, 5, 0)))))
                if not why.any():
                    if v.crash != "none" and v.crash_pos is None:
                        v.crash_pos = np.array(col[0, 0:3], np.float64)
                    continue
                j = int(np.argmax(why > 0))
                last = np.array(col[j - 1] if j > 0 else self._last_row(v.uav), np.float64)
                last[7:13] = 0.0                              # 冻结：速度与角速度置零，位置、姿态、比力取最后一个合理样本
                v.frozen = last
                reason = ("nonfinite", "speed", "omega", "position", "drift")[int(why[j]) - 1]
                self.guard_events.append({"uav": v.uav, "t_ns": int(t_rel[j]), "reason": reason, "crashed_before": v.crash != "none"})
                if v.crash == "none":                         # 未坠毁机体发散：按坠毁处理，Env 以 CRASHED 终止该机
                    v.crash = "impact"
                    p = last[0:3]
                    contacts.append(ContactEvent(v.uav, int(t_rel[j]), "IMPACT", "gz", (float(p[0]), float(p[1]), float(p[2])), None,
                                                 float(np.linalg.norm(fin[j - 1, 7:10])) if j > 0 else 0.0))
                if out is None:
                    out = np.array(state, np.float64)
                out[j:, v.idx, :] = v.frozen
            else:
                if out is None:
                    out = np.array(state, np.float64)
                out[:, v.idx, :] = v.frozen
        return (state if out is None else out), contacts

    def _last_row(self, uav: str) -> np.ndarray:
        s = self.history.last_sample(uav)
        return np.concatenate([s.pos_enu_m, s.q_ws_xyzw, s.vel_enu_mps, s.omega_b_rps, s.f_b_mps2])

    def _contacts(self, sd, t_rel: np.ndarray, state: np.ndarray) -> list[ContactEvent]:
        by_step: dict[int, list] = {}
        for c in sd.contacts:
            by_step.setdefault(int(c.step), []).append(c)
        out: list[ContactEvent] = []
        for j in range(self.n_steps):
            for c in by_step.get(j, ()):
                v = self._v[self.uavs[c.uav_index]]
                kind = _KIND.get(int(c.kind))
                if kind is None or v.crash != "none":
                    continue
                ev = ContactEvent(v.uav, int(t_rel[j]), kind, "gz", (c.px, c.py, c.pz), (c.nx, c.ny, c.nz), float(c.speed))
                if kind in _CRASH:
                    v.crash = _CRASH[kind]
                    if v.crash_pos is None and v.frozen is None:  # 发散守卫的位移基准：坠毁步的机体位置（有限时）
                        ps = state[j, v.idx, 0:3]
                        v.crash_pos = np.array(ps, np.float64) if np.isfinite(ps).all() else None
                    out.append(ev)
                elif kind == "TOUCHDOWN":
                    v.in_contact = True
                elif kind == "LIFTOFF":
                    v.grace_ns = 0
                    ps = state[j, v.idx, 0:3]                # 离地时已无接触点，取机体位置
                    ev = ContactEvent(v.uav, int(t_rel[j]), kind, "gz", (float(ps[0]), float(ps[1]), float(ps[2])), (0.0, 0.0, 1.0), float(c.speed))
                    v.in_contact = False
                    v.still_ns = 0
                    if not v.landed:                     # 落地待机中的弹跳不作为离地事件
                        out.append(ev)
            for v in self._v.values():
                if v.crash != "none" or v.landed:
                    continue
                if v.grace_ns > 0:
                    v.grace_ns -= self.step_ns
                    continue
                speed = float(np.linalg.norm(state[j, v.idx, 7:10]))
                v.still_ns = v.still_ns + self.step_ns if v.in_contact and speed < STILL_V_MPS else 0
                need = LAND_STILL_S if self.tracker.kind(v.uav) == "land" else STILL_S
                if v.still_ns >= int(need * 1e9):
                    v.landed = True
                    p = state[j, v.idx, 0:3]
                    out.append(ContactEvent(v.uav, int(t_rel[j]), "TOUCHDOWN", "gz", (float(p[0]), float(p[1]), float(p[2])), (0.0, 0.0, 1.0),
                                            speed))
        return out

    # ------------------------------------------------------------ 状态
    def status(self) -> Mapping[str, VehicleStatus]:
        out = {}
        for u, v in self._v.items():
            mode = "LAND" if self.tracker.kind(u) == "land" else self.tracker.kind(u).upper()
            out[u] = derive_status(StatusSignals(crash_cause=v.crash, landed=v.landed, in_air=not v.landed and v.crash == "none",
                                                 cmd_mode=mode))
        return out

    def close(self) -> dict:
        info = {"ticks": self._ticks, "t_end_ns": self.t_k_ns,
                "gz_rtf_mean": float(np.mean(self._rtf)) if self._rtf else 0.0, "pose_sha256": self.digest.hexdigest()}
        if self.wind is not None:
            info["wind"] = self.wind.describe()
        if self.guard_events:
            info["crash_guard"] = list(self.guard_events)
        if self._sock is not None:
            try:
                body, bye = self._rpc("shutdown", P.msg("Shutdown"), self.timeout_s)
                if body == "bye":
                    info["iterations"] = int(bye.iterations)
            except PhysError as e:
                info["close_error"] = str(e)
            self.abandon()
        return info

    def abandon(self) -> None:
        """关闭控制套接字而不发送 Shutdown（632 / 638 之后 REQ 状态机已失步；实例由调用方回收）。"""
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None


# ---------------------------------------------------------------------------------------------------------------- 故障恢复
@dataclass(frozen=True)
class _TickLog:
    k: int
    batches: tuple[tuple[TierCommand, ...], ...]     # advance(k) 之前的 submit 批次（按调用顺序）
    control_sha256: bytes
    block_sha256: bytes


class SupervisedGzBackend:
    """GzSimBackend + GzInstance 的看门狗与重启重放（M24-FR-078、FR-079；用法与 GzSimBackend 相同）。

    `instance` 需提供 `endpoint`、`alive()`、`restart(ready_timeout_s)` 与 `restarts`（GzInstance）；其余关键字参数原样传给每个
    GzSimBackend（`grounded`、`tick_ms`、`retain_s`、`run_id`、`step_timeout_s` 等；`attempt` 取实例重启次数）。实例的启动与最终
    停止仍由调用方负责。`on_recovered(event)` 在每次恢复完成、重试故障 tick 之前调用（M27 事件上报与故障注入测试）。
    """

    REPLAY_RETAIN_S = 0.5

    def __init__(self, instance: Any, uavs: Sequence[str], *, max_consecutive: int = 3, ready_timeout_s: float = 60.0,
                 reuse_policy: str = REUSE_POLICY, on_recovered: Callable[[dict], None] | None = None, **backend_kw: Any) -> None:
        if reuse_policy != "per_run":
            raise PhysError(630, f"reuse_policy {reuse_policy!r} has not passed the same-seed reset verification (M24-FR-060); use per_run")
        if "endpoint" in backend_kw or "alive" in backend_kw or "attempt" in backend_kw:
            raise TypeError("endpoint, alive and attempt are taken from the instance")
        self.instance = instance
        self.uavs = tuple(uavs)
        self.max_consecutive = int(max_consecutive)
        self.ready_timeout_s = float(ready_timeout_s)
        self.reuse_policy = reuse_policy
        self.on_recovered = on_recovered
        self._kw = dict(backend_kw)
        self._tracker0 = copy.deepcopy(self._kw.pop("tracker", None))
        self._log: list[_TickLog] = []
        self._pending: list[tuple[TierCommand, ...]] = []
        self._consec: dict[int, int] = {}
        self._stale = False
        self.recoveries: list[dict] = []
        self.be = self._make()

    def __getattr__(self, k: str):
        if k == "be":
            raise AttributeError(k)
        return getattr(self.be, k)

    def _make(self, *, retain_s: Any = "same") -> GzSimBackend:
        kw = dict(self._kw)
        if self._tracker0 is not None:
            kw["tracker"] = copy.deepcopy(self._tracker0)
        if retain_s != "same":
            kw["retain_s"] = retain_s
        return GzSimBackend(self.uavs, self.instance.endpoint, alive=self.instance.alive, attempt=int(self.instance.restarts), **kw)

    # ------------------------------------------------------------ PhysicsTier
    def open(self) -> dict:
        ack = self._call(lambda be: be.open(), tick=None)
        self._open_sha256 = self.be.open_sha256
        self._history = self.be.history                  # 原运行的 PoseHistoryBuffer：每次恢复后交接回去
        return ack

    def submit(self, cmds: Sequence[TierCommand]) -> list[DispatchResult]:
        batch = tuple(cmds)
        self._pending.append(batch)
        return self.be.submit(batch)

    def advance(self, k: int) -> TickResult:
        r = self._call(lambda be: be.advance(k), tick=k)
        self._log.append(_TickLog(k, tuple(self._pending), self.be.last_control_sha256, self.be.last_block_sha256))
        self._pending = []
        return r

    def status(self) -> Mapping[str, VehicleStatus]:
        return self.be.status()

    def close(self) -> dict:
        info = self.be.close()
        info.update(restarts=len(self.recoveries), recoveries=list(self.recoveries), reuse_policy=self.reuse_policy,
                    replayed_ticks=sum(e["replayed_ticks"] for e in self.recoveries))
        return info

    # ------------------------------------------------------------ 恢复
    def _call(self, op: Callable[[GzSimBackend], Any], *, tick: int | None):
        while True:
            w0 = time.monotonic()
            try:
                if self._stale:
                    self._rebuild(opened=tick is not None)
                    w0 = time.monotonic()
                out = op(self.be)
            except PhysError as e:
                if e.code not in RECOVERABLE or (isinstance(e.detail, dict) and e.detail.get("platform_failure")):
                    raise
                self._fault(e, tick, time.monotonic() - w0)
                continue
            self._consec.clear()
            return out

    def _fault(self, e: PhysError, tick: int | None, waited_s: float) -> None:
        n = self._consec[e.code] = self._consec.get(e.code, 0) + 1
        self.be.abandon()
        self._stale = True
        ev = {"tick": tick, "code": e.code, "reason": (e.detail or {}).get("reason", "") if isinstance(e.detail, dict) else "",
              "consecutive": n, "detect_s": round(waited_s, 3), "message": str(e), "completed": False, "attempt": None,
              "restart_s": None, "replay_s": None, "replayed_ticks": 0}
        if n >= self.max_consecutive:
            raise PhysError(e.code, f"{n} consecutive gz failures at tick {tick}: {e}",
                            detail={**ev, "platform_failure": True, "reason": "consecutive_failures"}) from e
        self.recoveries.append(ev)

    def _rebuild(self, *, opened: bool) -> None:
        """FAILED → STOPPED → 重启实例 → 新后端从回合起点按日志重放到故障前 tick，逐 tick 比对后交接。"""
        ev = self.recoveries[-1]
        w0 = time.monotonic()
        self.instance.restart(self.ready_timeout_s)
        w1 = time.monotonic()
        be = self._make(retain_s=self.REPLAY_RETAIN_S) if opened else self._make()
        self.be = be                                     # 重放中的故障同样计入（_call 捕获后再次重建）
        if opened:
            be.open()
            if be.open_sha256 != self._open_sha256:
                self._diverged(ev, -1, "initial state")
            for rec in self._log:
                for b in rec.batches:
                    be.submit(b)
                be.advance(rec.k)
                if be.last_control_sha256 != rec.control_sha256:
                    self._diverged(ev, rec.k, "control message")
                if be.last_block_sha256 != rec.block_sha256:
                    self._diverged(ev, rec.k, "pose samples")
            # 交接：重放与原运行逐位相同，沿用原 PoseHistoryBuffer（调用方的释放状态不变）；本 tick 已提交的批次重新提交
            be.history = self._history
            be.retain_s = self._kw.get("retain_s", GzSimBackend.retain_s)
            for b in self._pending:
                be.submit(b)
        self._stale = False
        ev.update(completed=True, attempt=int(self.instance.restarts), restart_s=round(w1 - w0, 3), replay_s=round(time.monotonic() - w1, 3),
                  replayed_ticks=len(self._log) if opened else 0)
        if self.on_recovered is not None:
            self.on_recovered(dict(ev))

    def _diverged(self, ev: dict, k: int, what: str) -> None:
        self.be.abandon()
        raise PhysError(638, f"replay after gz restart diverged at tick {k} ({what}); platform failure, rerun the whole Run",
                        detail={**ev, "platform_failure": True, "reason": "replay_diverged", "diverged_tick": k, "what": what})


# ---------------------------------------------------------------------------------------------------------------- reset 复用验证
def verify_reset_reuse(make_instance: Callable[[str], Any], sdf_for: Callable[[str], str], uavs: Sequence[str],
                       plan: Mapping[int, Sequence[TierCommand]], n_ticks: int, *, world: str = "d4d", **backend_kw: Any) -> dict:
    """同种子 reset 复用验证（M24-FR-060、§6.10.5；D3-AC-35）。

    实例 A 运行 `n_ticks`（摘要 a）→ 先以协议 Reset、再以 `/world/<w>/control reset` 请求复位 → 同一实例重新 Hello 并运行 `n_ticks`
    （摘要 a2）；新实例 B 运行 `n_ticks`（摘要 b）。a = a2 = b 才允许 `same_seed`，否则 `per_run`，原因逐条写入 `reasons`。
    `b == a` 是每 Run 重启方案本身的前提（`restart_bitwise`）。`make_instance(tag)` 返回未启动的 GzInstance，`sdf_for(gz_endpoint)`
    返回该实例的 run.sdf（控制端点随实例而变）。
    """
    import zmq

    def run(inst) -> tuple[str, dict]:
        be = GzSimBackend(uavs, inst.endpoint, alive=inst.alive, **backend_kw)
        ack = be.open()
        for k in range(n_ticks):
            be.submit(plan.get(k, ()))
            be.advance(k)
        be.abandon()                                     # 不发 Shutdown：Shutdown 会结束 gz 的服务循环
        return be.digest.hexdigest(), {**ack, "open_sha256": be.open_sha256.hex()}

    out: dict = {"n_ticks": n_ticks, "reasons": [], "sha256": {}}
    a = make_instance("a")
    a.start(sdf_for(a.gz_endpoint))
    try:
        a.wait_socket(120)
        out["sha256"]["a"], ack_a = run(a)
        # 1. 协议 Reset（drone4d.gz.v1 Envelope.reset）
        sock = zmq.Context.instance().socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.connect(a.endpoint)
        sock.send(P.encode("reset", P.msg("Reset")))
        try:
            rb, rm = P.decode(sock.recv())
            out["protocol_reset"] = f"{rb} {getattr(rm, 'code', '')} {getattr(rm, 'detail', '')}".strip()
            if rb == "error":
                out["reasons"].append(f"plugin rejects Reset ({rm.code} {rm.detail})")
        except zmq.Again:
            out["protocol_reset"] = "no reply"
            out["reasons"].append("plugin did not answer Reset")
        sock.close(0)
        # 2. 世界复位服务
        ok, txt = a.request_world_reset(world)
        out["world_reset"] = {"ok": ok, "output": txt}
        if not ok:
            out["reasons"].append("world control reset was not acknowledged")
        # 3. 同一实例重新握手并运行
        try:
            out["sha256"]["a2"], ack_a2 = run(a)
            for key in ("t0_gz_ns", "open_sha256"):
                if ack_a2[key] != ack_a[key]:
                    out["reasons"].append(f"HelloAck after reset differs in {key} ({ack_a[key]} → {ack_a2[key]})")
        except PhysError as e:
            out["sha256"]["a2"] = None
            out["reasons"].append(f"rerun on the reset instance failed: {e}")
    finally:
        a.stop(force=True)
    b = make_instance("b")
    b.start(sdf_for(b.gz_endpoint))
    try:
        b.wait_socket(120)
        out["sha256"]["b"], _ = run(b)
    finally:
        b.stop(force=True)
    d = out["sha256"]
    out["restart_bitwise"] = d["b"] == d["a"]
    if d.get("a2") is not None and d["a2"] != d["a"]:
        out["reasons"].append("trajectory after reset differs from the first episode")
    out["reuse_policy"] = "same_seed" if d["a"] == d.get("a2") == d["b"] and not out["reasons"] else "per_run"
    return out
