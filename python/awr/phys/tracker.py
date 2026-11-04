"""路径跟踪器（M24 §6.5.7；P0a 版）：把 TierCommand 换算为逐步速度设定（ENU m/s、偏航角速度），Realistic 与 SITL 共用。

P0a 以 tick 为周期做位置反馈（每 tick 用 tick 起点状态求一次设定，批内零阶保持）：
- velocity、hover：设定即指令（hover 为零速度）；
- goto：v = sat(K·e, speed)，K = (0.95, 0.95, 1.0) s⁻¹，水平与竖直分别限幅；到达半径 0.3 m 内转悬停；
- takeoff：以不超过 `takeoff_vz_mps` 的速度爬升至起飞点上方 alt_m（v_z = min(K_z·e_z, takeoff_vz)），水平保持起飞点；
- land：以 `land_vz_mps` 下降，触地后由宿主失能电机；
- kill：电机零转速（enabled = false）；
- orbit（P0b，P09）：距圆 > 1 m 时先按 goto 入圈（目标为圆上最近点）；随后参考点按角速度斜坡（加速度 `acc_mps2`、
  ω ≤ min(v, √(acc·R)) / R，与 FleetSim M_ORBIT 同式）沿圆推进，设定 = 切向前馈 + K·(参考点 − 位置)；偏航按航向误差朝向圆心
  （K_YAW·Δψ，限幅）；`turns` > 0 时绕满后在终点悬停；
- follow_path（P0b，P09）：逐航点 goto（v = sat(K·e, speed)），进入 `path_switch_m` 后切换到下一航点，末航点按 goto 到达。
orbit 与 follow_path 的参考状态按 tick 推进（`tick_s`，GzSimBackend 按自身 tick 设定）；未实现的种类与缺参的指令返回 630。

`RefTracker`（P0b，P09；M24-FR-081）：Realistic 档的 PX4-lite 参考跟踪器，与 Fast 档共用指令映射与 M08 参考生成器，逐物理步
给出设定（GzSimBackend 按 apply_step 下发），见类说明；`Tracker` 保留给 KinematicTier 与未切换的 Realistic 运行。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from awr.phys import PhysError
from awr.phys.adapter import TierCommand

__all__ = ["RefTracker", "Setpoint", "Tracker", "TrackerLimits"]

K_POS = np.array([0.95, 0.95, 1.0])
K_YAW = 1.0                                        # 偏航误差增益（1/s），orbit 朝向圆心
TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TrackerLimits:
    v_xy_mps: float = 12.0
    v_z_up_mps: float = 3.0
    v_z_dn_mps: float = 2.0
    yaw_rate_rps: float = 1.0
    takeoff_vz_mps: float = 1.5
    land_vz_mps: float = 0.8
    arrive_m: float = 0.3
    acc_mps2: float = 3.0                          # orbit 角速度斜坡与向心加速度上限（compose VelocityLimits.acc_mps2）
    orbit_entry_m: float = 1.0                     # 距圆 ≤ 1 m（水平与竖直）视为已入圈（FleetSim begin_orbit 同阈值）
    path_switch_m: float = 1.0                     # follow_path 中间航点的切换半径


@dataclass(frozen=True)
class Setpoint:
    v_enu: tuple[float, float, float]
    yaw_rate: float = 0.0
    enabled: bool = True
    a_ff: tuple[float, float, float] = (0.0, 0.0, 0.0)   # 参考加速度前馈（`RefTracker(inner_pid=True)`，L2 引擎自己做速度 PID）
    yaw: float = math.nan                                 # 航向设定（ENU，rad；NaN 为只给角速度）


@dataclass
class _Ref:
    cmd: TierCommand
    origin: np.ndarray | None = None
    phase: int = 0                                 # orbit：0 入圈、1 绕圈、2 绕满悬停；follow_path：当前航点序号
    th: float = 0.0                                # orbit 参考点相位（rad，ENU 俯视逆时针为正）
    w: float = 0.0                                 # orbit 角速度幅值（rad/s）
    turn: float = 0.0                              # orbit 已绕圈数


@dataclass
class Tracker:
    limits: TrackerLimits = field(default_factory=TrackerLimits)
    tick_s: float = 0.1                            # 设定的求值周期（每 tick 一次）；orbit 参考点按此推进
    _ref: dict[str, _Ref] = field(default_factory=dict)

    def set(self, cmd: TierCommand, pos: np.ndarray) -> None:
        if cmd.kind == "orbit" and (cmd.orbit is None or not 1.0 <= float(cmd.orbit.radius_m) <= 1000.0):
            raise PhysError(630, "orbit needs an OrbitSpec with radius in [1, 1000] m")
        if cmd.kind == "follow_path" and (cmd.path_enu_m is None or len(cmd.path_enu_m) == 0):
            raise PhysError(630, "follow_path needs at least one point")
        self._ref[cmd.uav] = _Ref(cmd, np.asarray(pos, np.float64).copy())

    def kind(self, uav: str) -> str:
        r = self._ref.get(uav)
        return r.cmd.kind if r else "hover"

    def setpoint(self, uav: str, pos: np.ndarray, vel: np.ndarray, q: np.ndarray | None = None) -> Setpoint:
        """tick 起点状态 → 本 tick 的设定；`q`（WORLD←FLU xyzw）只用于 orbit 的偏航（缺省不转偏航）。"""
        r = self._ref.get(uav)
        L = self.limits
        if r is None or r.cmd.kind == "hover":
            return Setpoint((0.0, 0.0, 0.0), float((r.cmd.yaw_rate_rps or 0.0) if r else 0.0))
        c = r.cmd
        if c.kind == "kill":
            return Setpoint((0.0, 0.0, 0.0), 0.0, enabled=False)
        if c.kind == "velocity":
            v = np.asarray(c.vel_enu_mps, np.float64)
            return Setpoint(self._sat(v), float(np.clip(c.yaw_rate_rps or 0.0, -L.yaw_rate_rps, L.yaw_rate_rps)))
        if c.kind == "takeoff":
            z_goal = float(r.origin[2]) + float(c.alt_m or 0.0)
            e = np.array([r.origin[0] - pos[0], r.origin[1] - pos[1], z_goal - pos[2]])
            v = K_POS * e
            v[2] = min(v[2], L.takeoff_vz_mps)                  # 匀速爬升，接近目标高度时按位置误差收敛
            return Setpoint(self._sat(v))
        if c.kind == "land":
            return Setpoint((0.0, 0.0, -L.land_vz_mps))
        if c.kind == "goto":
            return self._goto(np.asarray(c.target_enu_m, np.float64), pos, c.speed_mps)
        if c.kind == "follow_path":
            path = np.asarray(c.path_enu_m, np.float64).reshape(-1, 3)
            while r.phase < len(path) - 1 and float(np.linalg.norm(path[r.phase] - pos)) < L.path_switch_m:
                r.phase += 1
            return self._goto(path[r.phase], pos, c.speed_mps)
        if c.kind == "orbit":
            return self._orbit(r, pos, q)
        raise PhysError(630, f"command kind {c.kind!r} is not available")

    def _goto(self, target: np.ndarray, pos: np.ndarray, speed: float | None) -> Setpoint:
        e = target - pos
        if float(np.linalg.norm(e)) < self.limits.arrive_m:
            return Setpoint(tuple(float(x) for x in K_POS * e))
        v = K_POS * e
        if speed is not None and np.linalg.norm(v[:2]) > speed:
            v[:2] *= speed / np.linalg.norm(v[:2])
        return Setpoint(self._sat(v))

    def _orbit(self, r: _Ref, pos: np.ndarray, q: np.ndarray | None) -> Setpoint:
        L, o = self.limits, r.cmd.orbit
        c = np.asarray(o.center_enu_m, np.float64)
        R = float(o.radius_m)
        sp = float(r.cmd.speed_mps) if r.cmd.speed_mps is not None else 5.0
        d = pos[:2] - c[:2]
        de = float(np.hypot(d[0], d[1]))
        if r.phase == 0:                                           # 入圈：goto 到圆上最近点
            th0 = math.atan2(d[1], d[0]) if de > 1e-6 else 0.0
            if abs(de - R) <= L.orbit_entry_m and abs(float(pos[2] - c[2])) <= L.orbit_entry_m:
                r.phase, r.th, r.w = 1, th0, 0.0
            else:
                return self._goto(np.array([c[0] + R * math.cos(th0), c[1] + R * math.sin(th0), c[2]]), pos, sp)
        dr = -1.0 if o.cw else 1.0
        if r.phase == 1:
            w_max = min(sp, math.sqrt(L.acc_mps2 * R)) / R
            r.w = min(r.w + L.acc_mps2 / R * self.tick_s, w_max)
            goal = float(o.turns or 0.0)
            dth = r.w * self.tick_s
            if goal > 0.0 and r.turn + dth / TWO_PI >= goal:           # 绕满：参考点停在终点
                dth = (goal - r.turn) * TWO_PI
                r.phase = 2
            r.th += dr * dth
            r.turn += dth / TWO_PI
        w = r.w if r.phase == 1 else 0.0
        cs, sn = math.cos(r.th), math.sin(r.th)
        carrot = np.array([c[0] + R * cs, c[1] + R * sn, c[2]])
        v = np.array([dr * -sn * w * R, dr * cs * w * R, 0.0]) + K_POS * (carrot - pos)
        yr = 0.0
        if q is not None and de > 1e-6:                            # 朝向圆心
            err = math.atan2(c[1] - pos[1], c[0] - pos[0]) - yaw_of(q)
            err = (err + math.pi) % TWO_PI - math.pi
            yr = float(np.clip(K_YAW * err + dr * w, -L.yaw_rate_rps, L.yaw_rate_rps))
        return Setpoint(self._sat(v), yr)

    def _sat(self, v: np.ndarray) -> tuple[float, float, float]:
        L = self.limits
        v = np.asarray(v, np.float64).copy()
        h = float(np.hypot(v[0], v[1]))
        if h > L.v_xy_mps:
            v[:2] *= L.v_xy_mps / h
        v[2] = min(max(v[2], -L.v_z_dn_mps), L.v_z_up_mps)
        return (float(v[0]), float(v[1]), float(v[2]))


G_ENU = 9.80665


def _qrot_xyzw(q, v) -> np.ndarray:
    """R(q)·v（q 为 WORLD←FLU，xyzw）。"""
    q = np.asarray(q, np.float64)
    u, w = q[0:3], q[3]
    t = 2.0 * np.cross(u, v)
    return np.asarray(v, np.float64) + w * t + np.cross(u, t)


def yaw_of(q_xyzw) -> float:
    x, y, z, w = (float(a) for a in q_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------------------------------------------------------- 参考跟踪器
VEL_P_ACC = np.array([1.8, 1.8, 4.0])              # PX4 MPC_XY_VEL_P_ACC、MPC_Z_VEL_P_ACC（gz Lee velocityGain / 质量，compose "px4" 档）
VEL_I_ACC = np.array([0.4, 0.4, 2.0])              # PX4 MPC_XY_VEL_I_ACC、MPC_Z_VEL_I_ACC
VEL_D_ACC = np.array([0.2, 0.2, 0.0])              # PX4 MPC_XY_VEL_D_ACC、MPC_Z_VEL_D_ACC（实测加速度阻尼，tick 起点值）
VEL_I_MAX = np.array([3.0, 3.0, 2.5])              # 积分项（速度当量，m/s）上限；z 取 g / MPC_Z_VEL_P_ACC
CMD_HEADROOM_MPS = 3.0                             # Lee 速度限幅高于参考限幅的余量（compose "px4" 档）；水平设定由本类按参考限幅截断


@dataclass(frozen=True)
class _Obs:
    pos: np.ndarray                                # 机体原点 ENU
    vel: np.ndarray
    yaw: float                                     # ENU 偏航
    wz: float                                      # 机体 z 角速度（≈ 偏航角速度）
    acc: np.ndarray                                # 世界系加速度 R·f_b + g（ENU）
    in_contact: bool
    landed: bool
    crashed: bool


class RefTracker:
    """PX4-lite 参考跟踪器（P09；M24-FR-081"两档同一接口语义"；D3-AC-35）：Realistic 档用 Fast 档的同一指令映射
    （`FastTier._apply`，M08 `actions.begin_*`）与同一参考生成器（M08 `setpoint.refgen`：PositionSmoothing-lite、HOLD 刹停锁定、
    TAKEOFF 1 s 怠速 + 3 s 斜坡、LAND 剖面与触地爬行、ORBIT 角速度斜坡、PATH TOPP-lite、VELOCITY 零速轴锁定），参考状态
    （tr_x、tr_v、tr_a、yaw_sp）在影子 FleetState 中逐物理步（4 ms，与 Fast 档 L1 同频）推进；位置环与 FleetSim `pos_ctrl` 同式
    （v_sp = P·(tr_x − p) + tr_v，限幅相同）。gz 侧速度环为 Lee（compose "px4" 档：velocityGain = PX4 速度 P 增益 × 质量），
    PX4 的速度积分、加速度前馈与实测加速度阻尼在本类中以速度当量叠加到设定上：v_cmd = v_sp + I + (tr_a − K_d·a) / K_p
    （a 为 tick 起点的世界系加速度），I 按 tick 起点误差积分
    （I += K_i / K_p · (v_sp − v) · tick，触地、怠速与失能时清零）。

    tick 内只有 tick 起点状态：参考推进与位置环用线性外推的位置 p_k + v_k·τ 与偏航 ψ_k + ω_z·τ，逐步设定由宿主按 apply_step
    下发（与上一步相同则省略）。水平设定按参考限幅截断，截断时积分反算（不越过限幅）。偏航：ψ_sp 由 refgen 给出，偏航角速度
    = MC_YAW_P · Δψ（与 FleetSim 姿态环同式，无前馈），按机型偏航限幅。
    地面高 `ground_h(xy)`（ENU，缺省 0）给出 AGL 与起飞基准；接触、落地与坠毁取宿主判定（`observe`）。"""

    def __init__(self, *, vehicle: str = "x500", ground_h: Callable[[np.ndarray], float] | None = None, step_ns: int = 4_000_000,
                 vehicles: Mapping[str, str] | None = None, inner_pid: bool = False) -> None:
        self.vehicle = vehicle
        self.vehicles = dict(vehicles or {})                    # per-vehicle profile refs (L2 engine); others take `vehicle`
        # inner_pid（统一 L2 引擎）：只给参考（限幅后的速度设定、加速度前馈、航向设定），速度积分、加速度阻尼与偏航 P 由引擎内
        # 的飞控级联按物理步执行（与 FleetSim pos_ctrl / att_ctrl 同式）；False（gz）时折算为速度当量叠加到设定上。
        self.inner_pid = bool(inner_pid)
        self.ground_h = ground_h
        self.step_ns = int(step_ns)
        self.tick_s = 0.1
        self._ft = None
        self._obs: dict[str, _Obs] = {}
        self._i: dict[str, np.ndarray] = {}
        self._t_s = 0.0
        self.stats = {"plans": 0, "refgen_calls": 0}

    # -------------------------------------------------------------- 装配
    def bind(self, uavs: Sequence[str], init: Mapping[str, np.ndarray], grounded: Sequence[str]) -> None:
        """回合起点：影子 FastTier（无世界、不推进物理）按初始样本出生；`init[u]` 为 16 列 PoseHistory 行。"""
        from awr.phys.fast import BODY_Z_M, FastTier

        spawn = {}
        for u in uavs:
            r = np.asarray(init[u], np.float64)
            z = float(r[2]) if u not in grounded else BODY_Z_M
            spawn[u] = ((float(r[0]), float(r[1]), z), yaw_of(r[3:7]))
        self._ft = FastTier(list(uavs), spawn, None, grounded=list(grounded), vehicles={u: self.vehicles.get(u, self.vehicle) for u in uavs},
                            history_s=0.5)
        self._slot = {u: i for i, u in enumerate(uavs)}
        self._i = {u: np.zeros(3) for u in uavs}

    @property
    def bound(self) -> bool:
        return self._ft is not None

    def _gh(self, xy) -> float:
        return 0.0 if self.ground_h is None else float(self.ground_h(np.asarray(xy, np.float64)))

    def observe(self, t_k_ns: int, obs: Mapping[str, tuple]) -> None:
        """tick 起点：`obs[u] = (PoseHistory 行, in_contact, landed, crashed)`；落地与坠毁同步到影子状态（IDLE / KILLED）。"""
        from awr.sim.fleet import kernels_l1 as K

        S = self._ft.f.S
        self._t_s = t_k_ns * 1e-9
        for u, (row, ic, ld, cr) in obs.items():
            r = np.asarray(row, np.float64)
            acc = _qrot_xyzw(r[3:7], r[13:16]) + np.array([0.0, 0.0, -G_ENU])
            o = _Obs(r[0:3].copy(), r[7:10].copy(), yaw_of(r[3:7]), float(r[12]), acc, bool(ic), bool(ld), bool(cr))
            self._obs[u] = o
            s = self._slot[u]
            self._write(s, o, 0.0)
            if cr and int(S.ctrl_mode[s]) != K.M_KILLED:
                S.ctrl_mode[s] = K.M_KILLED
            elif ld and int(S.ctrl_mode[s]) not in (K.M_IDLE, K.M_TAKEOFF, K.M_SPOOLUP):
                S.ctrl_mode[s] = K.M_IDLE
                S.ctrl_phase[s] = 0
            S.landed[s] = ld
            S.in_air[s] = not ld and not cr

    def _write(self, s: int, o: _Obs, tau: float) -> None:
        from awr.phys.fast import BODY_Z_M
        from awr.sim.fleet.px4lite import quat_from_yaw

        S = self._ft.f.S
        p = o.pos + o.vel * tau
        gh = self._gh(p[:2])
        S.p[s] = (p[1], p[0], -(p[2] - BODY_Z_M))
        S.v[s] = (o.vel[1], o.vel[0], -o.vel[2])
        S.q[s] = quat_from_yaw(np.array([math.pi / 2 - (o.yaw + o.wz * tau)]))[0]
        S.agl[s] = p[2] - BODY_Z_M - gh
        S.ground_z[s] = -gh
        S.in_contact[s] = o.in_contact

    # -------------------------------------------------------------- 指令
    def set(self, cmd: TierCommand, pos: np.ndarray) -> None:
        from awr.sim.fleet import kernels_l1 as K

        if cmd.kind == "orbit" and (cmd.orbit is None or not 1.0 <= float(cmd.orbit.radius_m) <= 1000.0):
            raise PhysError(630, "orbit needs an OrbitSpec with radius in [1, 1000] m")
        if cmd.kind == "follow_path" and (cmd.path_enu_m is None or len(cmd.path_enu_m) == 0):
            raise PhysError(630, "follow_path needs at least one point")
        ft, s = self._ft, self._slot[cmd.uav]
        S = ft.f.S
        if cmd.kind in ("takeoff",) and int(S.ctrl_mode[s]) == K.M_IDLE:
            self._i[cmd.uav] = np.zeros(3)
        ft._apply(cmd, s, self._t_s)
        mb = ft.f.mailbox
        if cmd.kind == "velocity" and mb.fresh[s]:                    # 与 IngestStage._velocity 同式：ENU → NED，ψ_ned = π/2 − ψ_enu
            v = mb.vel[s]
            S.vel_cmd[s] = (v[1], v[0], -v[2])
            S.vel_frame[s] = 0
            S.vel_yawrate[s] = -mb.yaw_rate[s]
            mb.fresh[s] = False

    def kind(self, uav: str) -> str:
        return self._ft.tracker.kind(uav) if self._ft is not None else "hover"

    def setpoint(self, uav: str, pos: np.ndarray, vel: np.ndarray, q: np.ndarray | None = None) -> Setpoint:
        raise PhysError(630, "RefTracker plans per physics step; use plan()")

    # -------------------------------------------------------------- 规划
    def plan(self, k: int, t_k_ns: int, n_steps: int) -> dict[str, list[Setpoint]]:
        """本 tick 的逐步设定：参考在影子状态中逐物理步推进（外推位置），每步按 FleetSim pos_ctrl 换算速度设定。"""
        from awr.sim.fleet import kernels_l1 as K
        from awr.sim.fleet import setpoint as SP

        ft = self._ft
        S, T, PB = ft.f.S, ft.f.T, ft.f.PB
        dt = self.step_ns * 1e-9
        uavs = list(self._slot)
        idx = np.array([self._slot[u] for u in uavs], np.int64)
        out: dict[str, list[Setpoint]] = {u: [] for u in uavs}
        self.stats["plans"] += 1
        for j in range(n_steps):
            tau = j * dt
            for u in uavs:
                self._write(self._slot[u], self._obs[u], tau)
            cls = SP.classify(S, idx)
            act = idx[cls == SP.CLS_ACTIVE]
            if act.size:
                SP.refgen(S, T.LT, PB, act, dt, (t_k_ns + j * self.step_ns) * 1e-9, time_stretch=True)
                self.stats["refgen_calls"] += 1
            for u, s, c in zip(uavs, idx, cls, strict=True):
                out[u].append(self._sp(u, int(s), int(c), j, tau, dt, K))
        return out

    def _sp(self, u: str, s: int, cls: int, j: int, tau: float, dt: float, K) -> Setpoint:
        from awr.sim.fleet import setpoint as SP

        S, LT = self._ft.f.S, self._ft.f.T.LT
        o = self._obs[u]
        mode, phase = int(S.ctrl_mode[s]), int(S.ctrl_phase[s])
        spool = mode == K.M_SPOOLUP or (mode == K.M_TAKEOFF and phase == 0)
        if cls != SP.CLS_ACTIVE or spool or o.landed or o.crashed:
            self._i[u] = np.zeros(3)
            return Setpoint((0.0, 0.0, 0.0), 0.0, enabled=False)
        lim = int(S.limits_id[s])
        is_vel = mode == K.M_VELOCITY
        ff = mode != K.M_OFFBOARD and not is_vel
        trx, trv, tra, p = S.tr_x[s], S.tr_v[s], S.tr_a[s], S.p[s]
        vs = trv.copy() if is_vel else np.array([(trx[0] - p[0]) * K.XY_P, (trx[1] - p[1]) * K.XY_P, (trx[2] - p[2]) * K.Z_P]) + (trv if ff else 0.0)
        nxy = math.sqrt(vs[0] * vs[0] + vs[1] * vs[1])
        sc = min(1.0, float(LT[lim, K.L_VXY]) / max(nxy, 1e-6))
        vs[0] *= sc
        vs[1] *= sc
        vs[2] = min(max(vs[2], -(K.TKO_SPEED if mode == K.M_TAKEOFF else float(LT[lim, K.L_ZUP]))), float(LT[lim, K.L_ZDN]))
        v_enu = np.array([vs[1], vs[0], -vs[2]])
        a_enu = np.array([tra[1], tra[0], -tra[2]]) if ff else np.zeros(3)
        if self.inner_pid:
            yaw_sp = (math.pi / 2 - float(S.yaw_sp[s]) + math.pi) % TWO_PI - math.pi
            return Setpoint((float(v_enu[0]), float(v_enu[1]), float(v_enu[2])), 0.0, a_ff=tuple(float(x) for x in a_enu), yaw=yaw_sp)
        touch = mode == K.M_LAND and phase >= 2
        if j == 0:                                                    # 速度积分（tick 起点误差，PX4 K_i 的速度当量）
            if o.in_contact or touch:
                self._i[u] = np.zeros(3)
            else:
                self._i[u] = np.clip(self._i[u] + VEL_I_ACC / VEL_P_ACC * (v_enu - o.vel) * self.tick_s, -VEL_I_MAX, VEL_I_MAX)
        cmd = v_enu + self._i[u] + (a_enu - VEL_D_ACC * o.acc) / VEL_P_ACC
        h = math.hypot(cmd[0], cmd[1])
        cap = float(LT[lim, K.L_VXY])
        if h > cap:                                                   # 水平限幅 + 反算抗积分饱和：积分不推高到限幅之外
            cmd[:2] *= cap / h
            if j == 0:
                self._i[u][:2] = cmd[:2] - v_enu[:2] - (a_enu[:2] - VEL_D_ACC[:2] * o.acc[:2]) / VEL_P_ACC[:2]
        err = (math.pi / 2 - float(S.yaw_sp[s]) - (o.yaw + o.wz * tau) + math.pi) % TWO_PI - math.pi
        ymax = float(LT[lim, K.L_YAWRATE])
        yr = float(np.clip(K.K_YAW * err, -ymax, ymax))              # 与 FleetSim 姿态环同式：偏航只有 P（MC_YAW_P），无前馈
        return Setpoint((float(cmd[0]), float(cmd[1]), float(cmd[2])), yr)
