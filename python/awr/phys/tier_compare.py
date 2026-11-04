"""两档一致性用例集与档间差异报告（M24-FR-081、FR-063、FR-062 子集；AWR-05 §5.3.5 一致性用例集行；D3-AC-35）。

同一组 DroneAdapter 一致性用例（起降、悬停、速度阶跃、限幅、偏航阶跃、航点、环绕、接触与坠毁、恒风扰动）以同一段驱动代码在
Fast 档（`awr.phys.fast.FastTier`）与 Realistic 档（`awr.phys.gz.backend.GzSimBackend`）上运行，经 PhysicsTier 接口
（`open`、`submit`、`advance`、`status`、`close`、`history`）取得逐 tick 状态、接触事件、指令回执与 PoseHistory：

- 接口层（必须一致，`interface` 判定）：指令回执（`DispatchResult` 的 ok / code）逐条相同；`VehicleStatus` 序列（逐 tick，相邻
  去重）与用例期望相同；用例登记的前若干个接触事件种类相同；PoseHistory 网格相同（t = 0, 4 ms, …，两档时刻逐位相等、四元数单位化）；
  空中机体的指令生效步（速度相对生效时刻变化超过 `ONSET_DV_MPS` 的首个样本）晚于生效时刻且在 `ONSET_MAX_S`（3 tick）内
  （两档同为 tick 起点生效；响应起点含电机、姿态滞后与参考斜坡）；限幅用例两档速度均不超过限幅 + 0.3 m/s。
- 动力学（只以容差报告，`metrics` 判定）：两档逐样本对齐的位置 RMS、到达时刻差、上升时间与超调差、倾角差、触地速度差、坠毁时刻
  与位置差等；初始容差取 M24 §6.10.6（速度阶跃、偏航阶跃、恒风悬停、方形航线、爬升与下降），其余为本文件设定（P09 首轮实测，
  冻结前以 ADR 登记）。

对照运行口径（D3-AC-35）：Realistic 侧用 `awr.phys.tracker.RefTracker`（与 Fast 档同一指令映射与 M08 参考生成器，逐物理步推进
参考，位置环与 FleetSim pos_ctrl 同式，PX4 速度积分、加速度前馈与加速度阻尼以速度当量叠加）与 compose "px4" 速度控制档（Lee
velocityGain = PX4 速度 P 增益 × 质量），档间差异只来自内环与动力学；两档都关闭改变轨迹的 M09 自主动作（Fast 不装配 M09，gz 不受
M09 管理）；风为 `field` 模式与 sigma_turb = 0 的恒定风（`ConstWind`：Fast 经 `FleetState.set_wind_from_enu`，即 M07 env stage 写入的同一字段；gz 经
`FieldWind` → `Drone4DWindDrag`，composite 力律，与 FleetSim 同式）；静态碰撞为同一组盒体（`BoxWorld`：Fast 档 `mesh_contact`
用其解析求交，gz 侧为同尺寸静态盒体与 z = 0 地面）；机型 x500（gz 只有 x500_d4d）。

两档采样一致性（M24-FR-063、§6.10.7）：参考轨迹为一次 Fast 运行（`REF_CASE`，250 Hz，含起飞、机动、偏航与触地）的 PoseHistory；
Realistic 侧以测试构建的运动学注入（`Drone4DClock` `<inject>`，`inject_sdf`）在无重力、无静态碰撞的世界中逐步复现参考位姿，
由 `Drone4DClock` 照常采集。`pose_history_diff` 逐样本比较：时刻逐位相同、位置差 ≤ 1e-6 m、姿态差 ≤ 1e-9 rad、速度、角速度与
比力逐位相同。两份 PoseHistory 送入同一 M25（`awr.agentsdk.env.l2.L2Sensors` 的 LiDAR 节点、`MeshLidar` 与 IMU 合成）：LiDAR
逐点位姿差（槽位位姿原点与按测得距离换算的世界点位移）≤ 1 mm，IMU 输出差在重采样容差内（`IMU_TOL`）。同一 PoseHistory 字节两次
送入 M25（相机、LiDAR、IMU 全路径）时输出逐字节相同（`m25_digest`）。

报告 `tier_compare.json`（`drone4d.phys.tier_compare.v1`，`build_report`）：每个用例的接口判定、各档指标、档间差与容差、
是否在容差内；两档采样一致性与 M25 逐字节结果；机型、时钟档、风模式、碰撞源、力律、内核与 gz 插件版本。

用法（dmax，见 tests/d3/gz/test_tier_compare.py）：
  python -m awr.phys.tier_compare run --tier fast --out DIR                        # Fast 档全部用例与参考轨迹
  python -m awr.phys.tier_compare run --tier realistic --out DIR                   # d3-gz 容器内（launcher = local），含注入
  python -m awr.phys.tier_compare m25 --out DIR [--world WORLD_DIR]                # M25 两档采样对拍与逐字节（--world 时 GPU 相机）
  python -m awr.phys.tier_compare report --out DIR                                 # 汇总为 DIR/tier_compare.json
  python -m awr.phys.tier_compare run --set ground --tier l2|st --out DIR          # 地面域：统一引擎 ugv 对 CommonRoad ST 基准
  python -m awr.phys.tier_compare report --set ground --tiers l2,st --out DIR      # DIR/tier_compare.ground.l2-st.json
  python -m awr.phys.tier_compare run --set water --tier l2|fossen --out DIR       # 海面域：统一引擎 usv 对 Fossen 三自由度基准
  python -m awr.phys.tier_compare report --set water --tiers l2,fossen --out DIR   # DIR/tier_compare.water.l2-fossen.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from awr.phys.adapter import OrbitSpec, TierCommand
from awr.phys.pose_history import STATE_N

__all__ = [
    "CASES",
    "GROUND_CASES",
    "IMU_TOL",
    "REF_CASE",
    "SCHEMA",
    "WATER_CASES",
    "BoxWorld",
    "Case",
    "CaseRun",
    "ConstWind",
    "build_report",
    "case_metrics",
    "compare_case",
    "inject_sdf",
    "lidar_point_poses",
    "m25_compare",
    "m25_digest",
    "pose_history_diff",
    "run_case",
    "run_fast",
    "run_l2",
    "run_realistic",
    "run_st",
    "write_inject",
]

SCHEMA = "drone4d.phys.tier_compare.v1"
TICK_MS = 100
TICK_NS = TICK_MS * 1_000_000
STEP_NS = 4_000_000
NS = 1_000_000_000
VEHICLE = "x500"
ONSET_DV_MPS = 0.05                     # 指令生效判据：速度相对生效时刻的变化
ONSET_MAX_S = 0.3                       # 响应起点须晚于生效时刻、且在 3 tick 内（含电机与姿态滞后、参考斜坡）
ARRIVE_M = 1.0                          # 航点到达半径（与 tracker.path_switch_m 相同）
GZ_GROUND_Z = 0.25                      # gz 地面出生的 base_link 初始高度（落到起落架静止高度约 0.224 m）

# 两档采样一致性（M24 §6.10.7 第 3 条）与 M25 对拍（D3-AC-35）
POS_TOL_M = 1e-6
ATT_TOL_RAD = 1e-9
LIDAR_TOL_M = 1e-3
# IMU 重采样容差：两侧输入为同一 250 Hz 网格上的物理步样本（omega_b、f_b 来自注入参考值，位姿差 ≤ 1e-6 m），IMU 合成对输入为
# 线性（抗混叠一阶 IIR、线性插值到 200 Hz）加杆臂项，输出差以输入差为界；容差取 f32 交付量化（2^-23 相对）以上、远低于噪声密度
IMU_TOL = {"acc_mps2": 1e-5, "gyro_rps": 1e-7}


# ---------------------------------------------------------------------------------------------------------------- 世界
@dataclass(frozen=True)
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]


@dataclass
class _Tile:
    id: int
    origin: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    t0: int
    nt: int
    V: np.ndarray
    F: np.ndarray


# 盒体 12 个三角形：每面 2 个，面序 −x、+x、−y、+y、−z、+z（顶点序号按 (x, y, z) 位 0/1 编码，外法向绕序）
_BOX_F = np.array([[0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5], [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3], [0, 2, 3], [0, 3, 1],
                   [4, 5, 7], [4, 7, 6]], np.uint32)
_FACE_N = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], np.float64)
_GROUND_HALF_M = 1000.0
_MISS = np.uint32(0xFFFFFFFF)


class BoxWorld:
    """解析世界：z = 0 地面（2 个三角形）加轴对齐静态盒体，提供 FastTier（`dsm_grid`、`dtm_grid`、`tiles`、`raycast_batch`、
    `closest_points_batch`、`semantic_of`）与 M25 网格 LiDAR（`_cast`）所需的 MeshWorldQuery 子集，以及 gz 侧同尺寸的 SDF。
    语义码：地面 0（未分类），盒体 `box_semantic`。"""

    def __init__(self, boxes: Sequence[Box] = (), *, box_semantic: int = 0, dsm_cell_m: float = 2.0) -> None:
        self.boxes = tuple(boxes)
        self.box_semantic = int(box_semantic)
        h = _GROUND_HALF_M
        tiles = [_Tile(0, np.zeros(3), np.array([-h, -h, 0.0]), np.array([h, h, 0.0]), 0, 2,
                       np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], np.float32), np.array([[0, 1, 2], [0, 2, 3]], np.uint32))]
        for i, b in enumerate(self.boxes):
            lo, hi = np.asarray(b.lo, np.float64), np.asarray(b.hi, np.float64)
            V = np.array([[(hi if (k >> a) & 1 else lo)[a] for a in range(3)] for k in range(8)], np.float32)
            tiles.append(_Tile(i + 1, np.zeros(3), lo, hi, 2 + 12 * i, 12, V, _BOX_F.copy()))
        self.tiles = tiles
        self.n_prim = 2 + 12 * len(self.boxes)
        n = round(2 * h / dsm_cell_m)
        a = np.zeros((n, n), np.float32)
        xs = -h + (np.arange(n) + 0.5) * dsm_cell_m
        for b in self.boxes:                                       # 柱体语义：格心落在盒体足迹内的格取盒顶
            cx = (xs >= b.lo[0]) & (xs < b.hi[0])
            cy = (xs >= b.lo[1]) & (xs < b.hi[1])
            a[np.ix_(cy, cx)] = np.maximum(a[np.ix_(cy, cx)], float(b.hi[2]))
        a.setflags(write=False)
        self._dsm = SimpleNamespace(a=a, x0_m=-h, y0_m=-h, cell_m=float(dsm_cell_m))
        dtm = np.zeros((int(2 * h / 10.0), int(2 * h / 10.0)), np.float32)
        dtm.setflags(write=False)
        self._dtm = SimpleNamespace(a=dtm, x0_m=-h, y0_m=-h, cell_m=10.0)

    # -------------------------------------------------------------- 栅格
    def dsm_grid(self):
        return self._dsm

    def dtm_grid(self):
        return self._dtm

    # -------------------------------------------------------------- 求交
    def _cast(self, o: np.ndarray, d: np.ndarray, tmax: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """o (N, 3)、单位方向 d、tmax (N,) → (t 未命中 inf、prim_id 未命中 MISS、法线 (N, 3) f32)。"""
        o = np.asarray(o, np.float64).reshape(-1, 3)
        d = np.asarray(d, np.float64).reshape(-1, 3)
        tmax = np.broadcast_to(np.asarray(tmax, np.float64), (len(o),))
        n = len(o)
        best = np.full(n, np.inf)
        prim = np.full(n, _MISS, np.uint32)
        nrm = np.zeros((n, 3), np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            tg = np.where(d[:, 2] != 0.0, -o[:, 2] / d[:, 2], np.inf)        # 地面 z = 0
            gx, gy = o[:, 0] + tg * d[:, 0], o[:, 1] + tg * d[:, 1]
            okg = (tg >= 0) & (tg <= tmax) & (np.abs(gx) <= _GROUND_HALF_M) & (np.abs(gy) <= _GROUND_HALF_M)
            best[okg] = tg[okg]
            prim[okg] = (gy[okg] > gx[okg]).astype(np.uint32)              # 地面三角形 0（x ≥ y 一侧）与 1
            nrm[okg] = (0.0, 0.0, 1.0)
            for i, b in enumerate(self.boxes):
                lo, hi = np.asarray(b.lo), np.asarray(b.hi)
                inv = 1.0 / d
                t1, t2 = (lo - o) * inv, (hi - o) * inv
                tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
                tn = np.where(np.isnan(tn), -np.inf, tn)
                tf = np.where(np.isnan(tf), np.inf, tf)
                t_in = tn.max(axis=1)
                t_out = tf.min(axis=1)
                ax = tn.argmax(axis=1)
                hit = (t_in <= t_out) & (t_in >= 0) & (t_in <= tmax) & (t_in < best)
                if hit.any():
                    j = np.flatnonzero(hit)
                    face = 2 * ax[j] + (d[j, ax[j]] < 0)                      # 沿 +轴射入 → −面；沿 −轴射入 → +面
                    best[j] = t_in[j]
                    prim[j] = (2 + 12 * i + 2 * face).astype(np.uint32)
                    nrm[j] = _FACE_N[face]
        return best, prim, nrm

    def raycast_batch(self, o, d, tmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        org = np.asarray(o, np.float64).reshape(-1, 3)
        dirs = np.asarray(d, np.float64).reshape(-1, 3)
        L = np.linalg.norm(dirs, axis=1)
        unit = dirs / np.maximum(L, 1e-300)[:, None]
        t, prim, _ = self._cast(org, unit, np.full(len(org), float(tmax)))
        return t.astype(np.float32), prim, np.zeros(len(org), np.uint32)

    def closest_points_batch(self, p, max_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        P = np.asarray(p, np.float64).reshape(-1, 3)
        best = np.minimum(np.abs(P[:, 2]), float(max_m))                       # 地面
        pts = P.copy()
        pts[:, 2] = np.where(np.abs(P[:, 2]) < max_m, 0.0, P[:, 2])
        prim = np.where(np.abs(P[:, 2]) < max_m, np.uint32(0), _MISS).astype(np.uint32)
        for i, b in enumerate(self.boxes):
            lo, hi = np.asarray(b.lo), np.asarray(b.hi)
            c = np.clip(P, lo, hi)
            inside = np.all((lo < P) & (hi > P), axis=1)
            if inside.any():                                                   # 盒内：投影到最近面
                dl, dh = P[inside] - lo, hi - P[inside]
                dd = np.hstack([dl, dh])
                k = dd.argmin(axis=1)
                cc = P[inside].copy()
                ax = k % 3
                cc[np.arange(len(k)), ax] = np.where(k < 3, lo[ax], hi[ax])
                c[inside] = cc
            d = np.linalg.norm(c - P, axis=1)
            face_ax = np.argmax(np.abs(c - (lo + hi) / 2) / np.maximum((hi - lo) / 2, 1e-9), axis=1)
            side = c[np.arange(len(c)), face_ax] >= ((lo + hi) / 2)[face_ax]
            face = 2 * face_ax + side
            better = d < best
            best[better], pts[better] = d[better], c[better]
            prim[better] = (2 + 12 * i + 2 * face[better]).astype(np.uint32)
        return best, pts, prim

    def semantic_of(self, prim_id) -> np.ndarray:
        pid = np.asarray(prim_id, np.uint32)
        return np.where((pid >= 2) & (pid < self.n_prim), self.box_semantic, 0).astype(np.uint8)

    # -------------------------------------------------------------- gz
    def sdf_models(self) -> str:
        """gz 侧同尺寸的静态盒体（碰撞即可；地面由 `compose_run(ground_plane=True)` 给出）。"""
        out = []
        for i, b in enumerate(self.boxes):
            c = [(b.lo[a] + b.hi[a]) / 2 for a in range(3)]
            s = [b.hi[a] - b.lo[a] for a in range(3)]
            out.append(f"""
    <model name="box{i}"><static>true</static><pose>{c[0]!r} {c[1]!r} {c[2]!r} 0 0 0</pose>
      <link name="l"><collision name="c"><geometry><box><size>{s[0]!r} {s[1]!r} {s[2]!r}</size></box></geometry></collision></link>
    </model>""")
        return "".join(out)


class ConstWind:
    """sigma_turb = 0 的恒定风（ENU 去向，m/s），`t_on_ns` 起生效：M07 服务的最小形状（FieldWind 用到的 `on_env_tick_all`、
    `step_dryden`、`query`），Fast 档按 tick 起点的同一函数值写入 `FleetState.set_wind_from_enu`。`t_on_ns` 须为 tick 整数倍，
    使两档的风在同一时刻（tick 起点 = 环境网格点）生效。"""

    def __init__(self, w_enu_mps: tuple[float, float, float], t_on_ns: int = 0) -> None:
        if t_on_ns % TICK_NS:
            raise ValueError("wind onset must be on a tick boundary")
        self.w = np.asarray(w_enu_mps, np.float64)
        self.t_on = int(t_on_ns)

    def at(self, t_ns: int) -> np.ndarray:
        return self.w if t_ns >= self.t_on else np.zeros(3)

    def on_env_tick_all(self, tick: int) -> list:
        return []

    def step_dryden(self, *a) -> None:
        return None

    def query(self, pos, t, *, fields, vel=None, agent_idx=None, out=None):
        t_ns = round(float(t) * NS) if isinstance(t, float) else int(t)
        return SimpleNamespace(wind_mps=np.tile(self.at(t_ns), (len(pos), 1)))


# ---------------------------------------------------------------------------------------------------------------- 用例
@dataclass(frozen=True)
class Case:
    """一个一致性用例：机队出生（ENU，机体原点高度）、地面出生机体、tick 数、指令脚本（apply_tick → 指令）、世界、风、
    期望的状态序列（相邻去重）与前若干个接触事件种类、指标与容差。"""
    id: str
    title: str
    spawn: Mapping[str, tuple[tuple[float, float, float], float]]
    n_ticks: int
    script: Mapping[int, tuple[TierCommand, ...]]
    grounded: tuple[str, ...] = ()
    boxes: tuple[Box, ...] = ()
    wind: tuple[tuple[float, float, float], int] | None = None
    expect_status: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    expect_contacts: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    tol: Mapping[str, float] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    vehicle: str = VEHICLE                                 # profile ref for every vehicle of the case (air x500, ground f1tenth …)

    @property
    def uavs(self) -> tuple[str, ...]:
        return tuple(self.spawn)

    @property
    def metric(self) -> str:
        """Which metric set applies (``params["metric"]``, default the case id): ground cases reuse the air metrics by name."""
        return str(self.params.get("metric", self.id))

    def world(self) -> BoxWorld | None:
        return BoxWorld(self.boxes) if self.boxes else None

    def ground(self) -> Callable[[np.ndarray], float] | None:
        """Terrain for the ground cases: ``params["plane"] = (gx, gy)`` is the plane z = gx·x + gy·y; None is the world's DSM."""
        pl = self.params.get("plane")
        if pl is None:
            return None
        gx, gy = float(pl[0]), float(pl[1])
        return lambda xy: gx * float(xy[0]) + gy * float(xy[1])


def _c(uav: str, kind: str, k: int, **kw) -> TierCommand:
    return TierCommand(uav, kind, k, **kw)


_SQUARE = np.array([[50.0, 0.0, 30.0], [50.0, 50.0, 30.0], [0.0, 50.0, 30.0], [0.0, 0.0, 30.0]])
_WALL = Box((30.0, -50.0, 0.0), (50.0, -30.0, 30.0))
A, G_, L_, C_ = "AIRBORNE", "GROUNDED", "LANDING", "CRASHED"

CASES: tuple[Case, ...] = (
    Case("takeoff_land", "起飞至 20 m 后降落（爬升与下降）", {"u1": ((0.0, 0.0, 0.0), 0.0)}, 450,
         {5: (_c("u1", "takeoff", 5, alt_m=20.0),), 200: (_c("u1", "land", 200),)}, grounded=("u1",),
         expect_status={"u1": (G_, A, L_, G_)}, expect_contacts={"u1": ("LIFTOFF", "TOUCHDOWN")},
         tol={"t_reach_s": 1.0, "t_touchdown_s": 1.0, "vz_touchdown_mps": 0.3, "pos_rms_m": 1.5},
         params={"alt_m": 20.0, "takeoff_tick": 5, "land_tick": 200}),
    Case("hover", "空中悬停 15 s", {"u1": ((0.0, 0.0, 10.0), 0.0)}, 150, {0: (_c("u1", "hover", 0),)},
         expect_status={"u1": (A,)}, tol={"pos_rms_m": 0.5, "drift_max_m": 0.5}),
    Case("velocity_step", "悬停 → 5 m/s 水平速度阶跃 → 停止", {"u1": ((0.0, 0.0, 10.0), 0.0)}, 110,
         {0: (_c("u1", "hover", 0),), 10: (_c("u1", "velocity", 10, vel_enu_mps=(5.0, 0.0, 0.0)),),
          60: (_c("u1", "velocity", 60, vel_enu_mps=(0.0, 0.0, 0.0)),)},
         expect_status={"u1": (A,)}, tol={"rise_s": 0.3, "overshoot_pct": 10.0, "pos_rms_m": 1.5},
         params={"step_tick": 10, "v_mps": 5.0, "until_tick": 60}),
    Case("velocity_limit", "20 m/s 速度指令按 12 m/s 限幅", {"u1": ((0.0, 0.0, 10.0), 0.0)}, 110,
         {0: (_c("u1", "hover", 0),), 10: (_c("u1", "velocity", 10, vel_enu_mps=(20.0, 0.0, 0.0)),)},
         expect_status={"u1": (A,)}, tol={"v_steady_mps": 1.0},
         params={"v_cap_mps": 12.0, "cap_margin_mps": 0.3}),
    Case("yaw_step", "偏航角速度阶跃 0 → 0.5 rad/s", {"u1": ((0.0, 0.0, 10.0), 0.0)}, 60,
         {0: (_c("u1", "hover", 0),), 10: (_c("u1", "velocity", 10, vel_enu_mps=(0.0, 0.0, 0.0), yaw_rate_rps=0.5),)},
         expect_status={"u1": (A,)}, tol={"t90_s": 0.2}, params={"step_tick": 10, "yaw_rate_rps": 0.5}),
    Case("waypoints", "50 m × 50 m 方形航线，6 m/s，30 m", {"u1": ((0.0, 0.0, 30.0), 0.0)}, 600,
         {5: (_c("u1", "follow_path", 5, path_enu_m=_SQUARE, speed_mps=6.0),)},
         expect_status={"u1": (A,)}, tol={"pos_rms_m": 1.5, "arrival_s_per_100m": 1.0},
         params={"start_tick": 5, "path": _SQUARE.tolist()}),
    Case("orbit", "半径 20 m、5 m/s 环绕一圈", {"u1": ((20.0, 0.0, 15.0), 0.0)}, 400,
         {5: (_c("u1", "orbit", 5, orbit=OrbitSpec((0.0, 0.0, 15.0), 20.0, cw=False, turns=1.0), speed_mps=5.0),)},
         expect_status={"u1": (A,)}, tol={"pos_rms_m": 1.5, "arrival_s_per_100m": 1.0, "radius_rms_m": 0.5},
         params={"start_tick": 5, "center": (0.0, 0.0, 15.0), "radius_m": 20.0}),
    Case("contact_crash", "侧向撞墙与空中失能坠地", {"wall": ((20.0, -40.0, 10.0), 0.0), "drop": ((-40.0, 40.0, 20.0), 0.0)}, 80,
         {0: (_c("wall", "hover", 0), _c("drop", "hover", 0)), 5: (_c("wall", "velocity", 5, vel_enu_mps=(4.0, 0.0, 0.0)),),
          10: (_c("drop", "kill", 10),)}, boxes=(_WALL,),
         expect_status={"wall": (A, C_), "drop": (A, C_)}, expect_contacts={"wall": ("COLLISION_WORLD",), "drop": ("IMPACT",)},
         tol={"t_crash_s": 0.5, "crash_pos_m": 0.5}),
    Case("wind_hover", "8 m/s 恒定风（sigma_turb = 0）中悬停", {"u1": ((0.0, 0.0, 10.0), 0.0)}, 250, {0: (_c("u1", "hover", 0),)},
         wind=((8.0, 0.0, 0.0), 5 * NS), expect_status={"u1": (A,)}, tol={"tilt_deg": 0.5, "pos_rms_m": 0.5},
         params={"steady_from_s": 17.0}),
)

# 两档采样一致性的参考轨迹（FR-063：250 Hz，含机动与触地）
REF_CASE = Case("inject_ref", "参考轨迹：起飞、斜向速度带偏航、环绕段、降落触地", {"u1": ((0.0, 0.0, 0.0), 0.3)}, 300,
                {5: (_c("u1", "takeoff", 5, alt_m=8.0),),
                 70: (_c("u1", "velocity", 70, vel_enu_mps=(4.0, 2.0, 0.5), yaw_rate_rps=0.4),),
                 130: (_c("u1", "goto", 130, target_enu_m=(10.0, 0.0, 8.0), speed_mps=5.0),),
                 190: (_c("u1", "land", 190),)}, grounded=("u1",))

# 地面域用例（统一引擎 ugv 域模型对 CommonRoad ST 基准，`--set ground`）：机型缺省 F1TENTH 级（f1tenth_gym 参数，A 级），
# 另两台（OpenPodcar 级、JPL 漫游车级）各一个用例看参数是否起作用；撞墙用例只有引擎（ST 基准无世界）
M_ = "MOVING"
_GSQ = np.array([[20.0, 0.0, 0.0], [20.0, 20.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 0.0]])
_GWALL = Box((16.0, -5.0, 0.0), (21.0, 5.0, 3.0))                    # face on a DSM cell edge: the mesh query sees it first
GROUND_CASES: tuple[Case, ...] = (
    Case("g_speed_step", "静止 → 3 m/s 速度阶跃 → 停车", {"c1": ((0.0, 0.0, 0.0), 0.0)}, 100,
         {5: (_c("c1", "velocity", 5, vel_enu_mps=(3.0, 0.0, 0.0)),), 60: (_c("c1", "velocity", 60, vel_enu_mps=(0.0, 0.0, 0.0)),)},
         expect_status={"c1": (G_, M_, G_)}, tol={"rise_s": 0.3, "overshoot_pct": 10.0, "pos_rms_m": 1.0},
         params={"metric": "velocity_step", "step_tick": 5, "v_mps": 3.0, "until_tick": 60}, vehicle="f1tenth"),
    Case("g_speed_limit", "30 m/s 速度指令按 20 m/s 限幅（功率段 a_max·v_switch/v）", {"c1": ((0.0, 0.0, 0.0), 0.0)}, 150,
         {5: (_c("c1", "velocity", 5, vel_enu_mps=(30.0, 0.0, 0.0)),)},
         expect_status={"c1": (G_, M_)}, tol={"v_steady_mps": 1.0},
         params={"metric": "velocity_limit", "v_cap_mps": 20.0, "cap_margin_mps": 0.3}, vehicle="f1tenth"),
    Case("g_goto_stop", "goto 到 (20, 10) 并停车，3 m/s", {"c1": ((0.0, 0.0, 0.0), 0.0)}, 200,
         {5: (_c("c1", "goto", 5, target_enu_m=(20.0, 10.0, 0.0), speed_mps=3.0),)},
         expect_status={"c1": (G_, M_, G_)}, tol={"t_arrive_s": 1.0, "rest_err_m": 1.0, "pos_rms_m": 1.0},
         params={"metric": "goto_stop", "start_tick": 5, "target": (20.0, 10.0)}, vehicle="f1tenth"),
    Case("g_waypoints", "20 m × 20 m 方形路线，3 m/s", {"c1": ((0.0, 0.0, 0.0), 0.0)}, 400,
         {5: (_c("c1", "follow_path", 5, path_enu_m=_GSQ, speed_mps=3.0),)},
         expect_status={"c1": (G_, M_, G_)}, tol={"pos_rms_m": 1.5, "arrival_s_per_100m": 2.0},
         params={"metric": "waypoints", "start_tick": 5, "path": _GSQ.tolist()}, vehicle="f1tenth"),
    Case("g_orbit", "半径 6 m、2 m/s 绕一圈", {"c1": ((6.0, 0.0, 0.0), math.pi / 2)}, 300,
         {5: (_c("c1", "orbit", 5, orbit=OrbitSpec((0.0, 0.0, 0.0), 6.0, cw=False, turns=1.0), speed_mps=2.0),)},
         expect_status={"c1": (G_, M_, G_)}, tol={"pos_rms_m": 1.5, "arrival_s_per_100m": 2.0, "radius_rms_m": 0.5},
         params={"metric": "orbit", "start_tick": 5, "center": (0.0, 0.0, 0.0), "radius_m": 6.0}, vehicle="f1tenth"),
    Case("g_slope", "15 % 坡上 2 m/s 定速爬坡（z 跟随地形、姿态沿坡面法向）", {"c1": ((0.0, 0.0, 0.0), 0.0)}, 150,
         {5: (_c("c1", "velocity", 5, vel_enu_mps=(2.0, 0.0, 0.0)),)},
         expect_status={"c1": (G_, M_)}, tol={"v_hold_mps": 0.3, "z_err_m": 0.05, "pos_rms_m": 1.0},
         params={"metric": "slope_hold", "plane": (0.15, 0.0), "v_mps": 2.0, "steady_from_s": 8.0}, vehicle="f1tenth"),
    Case("g_wall", "3 m/s 直行撞墙（只有引擎：ST 基准无世界）", {"c1": ((0.0, 0.0, 0.0), 0.0)}, 100,
         {5: (_c("c1", "velocity", 5, vel_enu_mps=(3.0, 0.0, 0.0)),)}, boxes=(_GWALL,),
         expect_status={"c1": (G_, M_, C_)}, expect_contacts={"c1": ("COLLISION_WORLD",)}, tol={"t_crash_s": 0.5, "crash_pos_m": 0.5},
         params={"metric": "contact_crash", "oracle": False}, vehicle="f1tenth"),
    Case("g_speed_step_podcar", "OpenPodcar 级：0 → 3 m/s → 停车（a_max 1.2 m/s²，刹车 2.5）", {"c1": ((0.0, 0.0, 0.0), 0.0)}, 200,
         {5: (_c("c1", "velocity", 5, vel_enu_mps=(3.0, 0.0, 0.0)),), 120: (_c("c1", "velocity", 120, vel_enu_mps=(0.0, 0.0, 0.0)),)},
         expect_status={"c1": (G_, M_, G_)}, tol={"rise_s": 0.5, "overshoot_pct": 10.0, "pos_rms_m": 1.0},
         params={"metric": "velocity_step", "step_tick": 5, "v_mps": 3.0, "until_tick": 120}, vehicle="podcar"),
    Case("g_waypoints_osr", "JPL 漫游车级：20 m 方形路线，1.2 m/s", {"c1": ((0.0, 0.0, 0.0), 0.0)}, 900,
         {5: (_c("c1", "follow_path", 5, path_enu_m=_GSQ, speed_mps=1.2),)},
         expect_status={"c1": (G_, M_, G_)}, tol={"pos_rms_m": 1.5, "arrival_s_per_100m": 2.0},
         params={"metric": "waypoints", "start_tick": 5, "path": _GSQ.tolist()}, vehicle="osr"),
)


# 海面域用例（统一引擎 usv 域模型对 Fossen 三自由度基准，`--set water`）：机型缺省 Otter 级（MSS otter.py 参数，A 级），
# 另两艘（WAM-V 16 级、BlueBoat 级）各一个用例看参数是否起作用；流中定点是两档各自的上界；搁浅用例只有引擎（基准无世界）
AF_ = "AFLOAT"
_WSQ = np.array([[40.0, 0.0, 0.0], [40.0, 40.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 0.0]])
WATER_CASES: tuple[Case, ...] = (
    Case("w_speed_step", "静止 → 2 m/s 速度阶跃 → 停船", {"b1": ((0.0, 0.0, 0.0), 0.0)}, 300,
         {5: (_c("b1", "velocity", 5, vel_enu_mps=(2.0, 0.0, 0.0)),), 200: (_c("b1", "velocity", 200, vel_enu_mps=(0.0, 0.0, 0.0)),)},
         expect_status={"b1": (AF_, M_, AF_)}, tol={"rise_s": 0.5, "overshoot_pct": 10.0, "pos_rms_m": 1.0},
         params={"metric": "velocity_step", "step_tick": 5, "v_mps": 2.0, "until_tick": 200}, vehicle="otter"),
    Case("w_speed_limit", "10 m/s 速度指令按 3.09 m/s（6 kn）限幅", {"b1": ((0.0, 0.0, 0.0), 0.0)}, 300,
         {5: (_c("b1", "velocity", 5, vel_enu_mps=(10.0, 0.0, 0.0)),)},
         expect_status={"b1": (AF_, M_)}, tol={"v_steady_mps": 0.3},
         params={"metric": "velocity_limit", "v_cap_mps": 3.086, "cap_margin_mps": 0.2}, vehicle="otter"),
    Case("w_goto_stop", "goto 到 (30, 15) 并停船，2 m/s", {"b1": ((0.0, 0.0, 0.0), 0.0)}, 400,
         {5: (_c("b1", "goto", 5, target_enu_m=(30.0, 15.0, 0.0), speed_mps=2.0),)},
         expect_status={"b1": (AF_, M_, AF_)}, tol={"t_arrive_s": 2.0, "rest_err_m": 1.5, "pos_rms_m": 2.0},
         params={"metric": "goto_stop", "start_tick": 5, "target": (30.0, 15.0)}, vehicle="otter"),
    Case("w_waypoints", "40 m × 40 m 方形航线，2 m/s", {"b1": ((0.0, 0.0, 0.0), 0.0)}, 1200,
         {5: (_c("b1", "follow_path", 5, path_enu_m=_WSQ, speed_mps=2.0),)},
         expect_status={"b1": (AF_, M_, AF_)}, tol={"pos_rms_m": 3.0, "arrival_s_per_100m": 3.0},
         params={"metric": "waypoints", "start_tick": 5, "path": _WSQ.tolist()}, vehicle="otter"),
    Case("w_orbit", "半径 10 m、1.5 m/s 绕一圈", {"b1": ((10.0, 0.0, 0.0), math.pi / 2)}, 800,
         {5: (_c("b1", "orbit", 5, orbit=OrbitSpec((0.0, 0.0, 0.0), 10.0, cw=False, turns=1.0), speed_mps=1.5),)},
         expect_status={"b1": (AF_, M_, AF_)}, tol={"pos_rms_m": 3.0, "arrival_s_per_100m": 3.0, "radius_rms_m": 1.0},
         params={"metric": "orbit", "start_tick": 5, "center": (0.0, 0.0, 0.0), "radius_m": 10.0}, vehicle="otter"),
    Case("w_current_hold", "0.5 m/s 水流中定点（hover 为保位）", {"b1": ((0.0, 0.0, 0.0), 0.0)}, 600, {0: (_c("b1", "hover", 0),)},
         expect_status={"b1": (M_,)}, tol={"drift_max_m": 8.0, "offset_steady_m": 1.0, "pos_rms_m": 2.0},
         params={"metric": "station_keep", "current": (0.5, 0.0), "steady_from_s": 30.0}, vehicle="otter"),
    Case("w_grounding", "2 m/s 驶向 5 % 坡的岸边直至搁浅（只有引擎：基准无世界）", {"b1": ((-40.0, 0.0, 0.0), 0.0)}, 300,
         {5: (_c("b1", "velocity", 5, vel_enu_mps=(2.0, 0.0, 0.0)),)},
         expect_status={"b1": (AF_, M_, C_)}, expect_contacts={"b1": ("COLLISION_WORLD",)}, tol={"t_crash_s": 0.5, "crash_pos_m": 0.5},
         params={"metric": "contact_crash", "oracle": False, "plane": (0.05, 0.0)}, vehicle="otter"),
    Case("w_speed_step_wamv", "WAM-V 16 级：0 → 1.2 m/s → 停船（VRX 阻尼系数）", {"b1": ((0.0, 0.0, 0.0), 0.0)}, 400,
         {5: (_c("b1", "velocity", 5, vel_enu_mps=(1.2, 0.0, 0.0)),), 250: (_c("b1", "velocity", 250, vel_enu_mps=(0.0, 0.0, 0.0)),)},
         expect_status={"b1": (AF_, M_, AF_)}, tol={"rise_s": 1.0, "overshoot_pct": 10.0, "pos_rms_m": 1.0},
         params={"metric": "velocity_step", "step_tick": 5, "v_mps": 1.2, "until_tick": 250}, vehicle="wamv"),
    Case("w_waypoints_blueboat", "BlueBoat 级：40 m 方形航线，2 m/s", {"b1": ((0.0, 0.0, 0.0), 0.0)}, 1200,
         {5: (_c("b1", "follow_path", 5, path_enu_m=_WSQ, speed_mps=2.0),)},
         expect_status={"b1": (AF_, M_, AF_)}, tol={"pos_rms_m": 3.0, "arrival_s_per_100m": 3.0},
         params={"metric": "waypoints", "start_tick": 5, "path": _WSQ.tolist()}, vehicle="blueboat"),
)


# ---------------------------------------------------------------------------------------------------------------- 运行
@dataclass
class CaseRun:
    case: str
    tier: str
    t_ns: np.ndarray                                       # (n,) PoseHistory 时刻（0 起，4 ms 网格）
    state: dict[str, np.ndarray]                           # uav → (n, 16)
    status: dict[str, list[str]]                           # uav → 逐 tick（advance 之后）
    contacts: list[dict]
    dispatch: list[dict]
    info: dict
    wall_s: float = 0.0

    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{self.case}.{self.tier}.npz", t_ns=self.t_ns, **{f"s_{u}": s for u, s in self.state.items()})
        doc = {"case": self.case, "tier": self.tier, "uavs": list(self.state), "status": self.status, "contacts": self.contacts,
               "dispatch": self.dispatch, "info": self.info, "wall_s": self.wall_s}
        (d / f"{self.case}.{self.tier}.json").write_text(json.dumps(doc, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, d: Path, case: str, tier: str) -> CaseRun:
        doc = json.loads((Path(d) / f"{case}.{tier}.json").read_text(encoding="utf-8"))
        z = np.load(Path(d) / f"{case}.{tier}.npz")
        return cls(case, tier, z["t_ns"], {u: z[f"s_{u}"] for u in doc["uavs"]}, doc["status"], doc["contacts"], doc["dispatch"], doc["info"],
                   float(doc.get("wall_s", 0.0)))

    def pose_bytes(self, uav: str) -> bytes:
        """该机 PoseHistory 的 `drone4d.pose_history.v1` 消息字节（M25 输入与 sha256 对照）。"""
        from awr.phys.pose_history import pack_pose_history

        return pack_pose_history(uav, NS // STEP_NS, self.t_ns, self.state[uav])


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.hex()
    return str(o)


def run_case(case: Case, tier, *, pre_tick: Callable[[int], None] | None = None) -> CaseRun:
    """在一个已构造的 PhysicsTier 上执行用例脚本（两档同一驱动代码）；`pre_tick(k)` 在 advance(k) 之前调用（Fast 档的恒定风）。"""
    w0 = time.perf_counter()
    tier.open()
    status: dict[str, list[str]] = {u: [] for u in case.uavs}
    contacts: list[dict] = []
    dispatch: list[dict] = []
    for k in range(case.n_ticks):
        cmds = case.script.get(k, ())
        if cmds:
            for c, r in zip(cmds, tier.submit(list(cmds)), strict=True):
                dispatch.append({"tick": k, "uav": c.uav, "kind": c.kind, "ok": bool(r.ok), "code": int(r.code)})
        if pre_tick is not None:
            pre_tick(k)
        res = tier.advance(k)
        contacts.extend({"uav": e.uav, "t_ns": int(e.t_ns), "kind": e.kind, "detector": e.detector,
                         "pos_enu_m": [float(x) for x in e.pos_enu_m], "speed_mps": float(e.speed_mps)} for e in res.contacts)
        for u, s in res.status.items():
            status[u].append(str(s))
    t_end = case.n_ticks * TICK_NS
    state = {}
    t_ns = None
    for u in case.uavs:
        sl = tier.history.slice(u, 0, t_end)
        t_ns = sl.t_ns
        state[u] = sl.state
    info = tier.close()
    return CaseRun(case.id, tier.name, np.asarray(t_ns, np.int64), state, status, contacts, dispatch, _jsonable(info),
                   round(time.perf_counter() - w0, 3))


def _jsonable(d):
    return json.loads(json.dumps(d, default=_json_default))


def run_fast(case: Case) -> CaseRun:
    from awr.phys.fast import FastTier

    tier = FastTier(case.uavs, dict(case.spawn), case.world(), grounded=list(case.grounded), vehicles=dict.fromkeys(case.uavs, VEHICLE),
                    tick_ms=TICK_MS, history_s=case.n_ticks * TICK_MS / 1000 + 2.0, retain_s=None)
    pre = None
    if case.wind is not None:
        wind = ConstWind(*case.wind)
        slots = np.arange(len(case.uavs), dtype=np.int64)

        def pre(k: int) -> None:
            tier.f.S.set_wind_from_enu(np.tile(wind.at(k * TICK_NS), (len(slots), 1)), slots)
    return run_case(case, tier, pre_tick=pre)


def run_l2(case: Case) -> CaseRun:
    """The unified L2 engine (`awr.phys.l2.L2Tier`) on the same cases: same commands, reference tracker, BoxWorld and wind."""
    from awr.phys.l2 import L2Tier

    tier = L2Tier(case.uavs, dict(case.spawn), case.world(), grounded=list(case.grounded), vehicles=dict.fromkeys(case.uavs, case.vehicle),
                  tick_ms=TICK_MS, history_s=case.n_ticks * TICK_MS / 1000 + 2.0, retain_s=None, ground=case.ground())
    pre = None
    if case.wind is not None:
        wind = ConstWind(*case.wind)

        def pre(k: int) -> None:
            tier.set_wind(np.tile(wind.at(k * TICK_NS), (len(case.uavs), 1)))
    _set_current(case, tier)
    return run_case(case, tier, pre_tick=pre)


def _set_current(case: Case, tier) -> None:
    """Water cases: ``params["current"] = (cx, cy)`` is a uniform current (ENU, m/s) felt by every vessel for the whole case."""
    cur = case.params.get("current")
    if cur is not None:
        tier.set_current(np.tile(np.asarray(cur, np.float64)[:2], (len(case.uavs), 1)))


def run_st(case: Case) -> CaseRun:
    """The CommonRoad single-track oracle (`awr.phys.l2.st_oracle`) on the ground cases: same commands and tracker, the
    published ST equations with RK4 instead of the engine's rigid body (no world: cases with boxes are engine-only)."""
    from awr.phys.l2.st_oracle import STOracleTier

    tier = STOracleTier(case.uavs, dict(case.spawn), None, vehicles=dict.fromkeys(case.uavs, case.vehicle), tick_ms=TICK_MS,
                        history_s=case.n_ticks * TICK_MS / 1000 + 2.0, retain_s=None, ground=case.ground())
    return run_case(case, tier)


def run_fossen(case: Case) -> CaseRun:
    """The Fossen 3-DoF oracle (`awr.phys.l2.fossen_oracle`) on the water cases: same commands and tracker, the published
    manoeuvring equations in matrix form with RK4 instead of the engine's rigid body (no world: grounding cases are engine-only)."""
    from awr.phys.l2.fossen_oracle import FossenOracleTier

    tier = FossenOracleTier(case.uavs, dict(case.spawn), None, vehicles=dict.fromkeys(case.uavs, case.vehicle), tick_ms=TICK_MS,
                            history_s=case.n_ticks * TICK_MS / 1000 + 2.0, retain_s=None)
    _set_current(case, tier)
    return run_case(case, tier)


RUNNERS = {"fast": run_fast, "l2": run_l2, "st": run_st, "fossen": run_fossen}
ORACLES = ("st", "fossen")


def compose_case_sdf(case: Case, endpoint: str, *, uavs_z: Mapping[str, float] | None = None, field_wind: bool = False) -> str:
    """gz 侧 run.sdf：机队（地面出生取 `GZ_GROUND_Z`；速度控制 "px4" 档）、z = 0 地面、用例盒体；有风时加 field 风插件
    （composite，x500）。"""
    from awr.phys.gz.compose import UavSpawn, VelocityLimits, compose_run

    sp = []
    for u, ((x, y, z), yaw) in case.spawn.items():
        zz = (uavs_z or {}).get(u, GZ_GROUND_Z if u in case.grounded else z)
        sp.append(UavSpawn(u, (x, y, zz), yaw, limits=VelocityLimits(profile="px4")))
    sdf = compose_run(sp, ground_plane=True, endpoint=endpoint)
    if case.boxes:
        sdf = sdf.replace("\n  </world>", BoxWorld(case.boxes).sdf_models() + "\n  </world>")
    if field_wind:
        from awr.phys.gz.wind import field_plugin_xml, load_drag_spec, with_field_wind

        sdf = with_field_wind(sdf, field_plugin_xml(list(case.uavs), load_drag_spec(VEHICLE), rho=1.225))
    return sdf


def run_realistic(case: Case, run_dir: Path, *, launcher: str = "local", image: str = "d3-gz:dev", inject: Path | None = None,
                  ready_s: float = 120.0) -> CaseRun:
    """Realistic 档（GzSimBackend）执行用例；`inject` 为参考轨迹文件时以运动学注入世界运行（无重力、无静态碰撞、无指令）。"""
    from awr.phys.gz.backend import GzSimBackend
    from awr.phys.gz.compose import UavSpawn, compose_run
    from awr.phys.gz.instance import GzInstance
    from awr.phys.gz.wind import FieldWind
    from awr.phys.tracker import RefTracker

    run_dir = Path(run_dir)
    inst = GzInstance(run_dir, f"d3-tc-{case.id}".replace("_", "-"), image=image, launcher=launcher)
    if inject is not None:
        ref = np.frombuffer(Path(inject).read_bytes(), "<f8").reshape(-1, len(case.uavs), STATE_N)
        sp = [UavSpawn(u, tuple(float(x) for x in ref[0, i, 0:3]), _yaw(ref[0, i, 3:7])) for i, u in enumerate(case.uavs)]
        sdf = inject_sdf(compose_run(sp, ground_plane=False, endpoint=inst.gz_endpoint), str(Path(inject).resolve()))
    else:
        sdf = compose_case_sdf(case, inst.gz_endpoint, field_wind=case.wind is not None)
    inst.start(sdf)
    try:
        inst.wait_socket(ready_s)
        fw = FieldWind(ConstWind(*case.wind), list(case.uavs)) if case.wind is not None and inject is None else None
        kw = {} if inject is not None else {"tracker": RefTracker(vehicle=VEHICLE, ground_h=lambda xy: 0.0)}
        be = GzSimBackend(case.uavs, inst.endpoint, tick_ms=TICK_MS, grounded=list(case.grounded) if inject is None else list(case.uavs),
                          history_s=case.n_ticks * TICK_MS / 1000 + 2.0, retain_s=None, alive=inst.alive, wind=fw,
                          run_id=f"tier-compare-{case.id}", **kw)
        sub = case if inject is None else Case(case.id, case.title, case.spawn, case.n_ticks, {}, grounded=case.uavs)
        run = run_case(sub, be)
        run.info.update(be.hello_ack)
        if inject is not None:
            run.case = f"{case.id}_inject"
    finally:
        inst.stop()
    return run


def _yaw(q) -> float:
    x, y, z, w = (float(a) for a in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------------------------------------------------------- 注入
def write_inject(run: CaseRun, path: Path) -> Path:
    """参考 PoseHistory → `Drone4DClock` 注入文件 f64[n][n_uav][16]（行 n 在 HelloAck 之后 n·step）。"""
    s = np.stack([run.state[u] for u in run.state], axis=1)
    Path(path).write_bytes(np.ascontiguousarray(s, "<f8").tobytes())
    return Path(path)


def inject_sdf(run_sdf: str, inject_path: str) -> str:
    """运动学注入世界（M24 §6.10.7 第 2 条）：重力置零、Drone4DClock 加 `<inject>`；调用方不加地面与静态碰撞体。"""
    import re

    out, n = re.subn(r"<gravity>[^<]*</gravity>", "<gravity>0 0 0</gravity>", run_sdf, count=1)
    if n != 1 or '<model name="ground">' in out or "world_static" in out:
        raise ValueError("inject world must come from compose_run without ground plane or static world")
    tag = "<lockstep>"
    i = out.index(tag, out.index('name="drone4d::Drone4DClock"'))
    return out[:i] + f"<inject>{inject_path}</inject>\n      " + out[i:]


def _quat_angle(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    d = np.abs(np.sum(qa * qb, axis=1)) / (np.linalg.norm(qa, axis=1) * np.linalg.norm(qb, axis=1))
    # 小角度下 2·acos(d) 的数值误差约 1e-8 rad：改用 2·asin(‖相对四元数虚部‖)
    x1, y1, z1, w1 = qa.T
    x2, y2, z2, w2 = qb.T
    vx = w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2
    vy = w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2
    vz = w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2
    s = np.sqrt(vx * vx + vy * vy + vz * vz)
    return np.where(d > 0.5, 2.0 * np.arcsin(np.minimum(s, 1.0)), 2.0 * np.arccos(np.minimum(d, 1.0)))


def pose_history_diff(ref: CaseRun, got: CaseRun) -> dict:
    """两档 PoseHistory 逐样本比较（M24 §6.10.7 第 3 条）。"""
    out: dict = {"t_bitwise": bool(np.array_equal(ref.t_ns, got.t_ns)), "n": len(ref.t_ns), "uavs": {}}
    ok = out["t_bitwise"]
    for u in ref.state:
        a, b = ref.state[u], got.state[u]
        if a.shape != b.shape:
            out["uavs"][u] = {"shape": [list(a.shape), list(b.shape)], "pass": False}
            ok = False
            continue
        dp = float(np.max(np.linalg.norm(a[:, 0:3] - b[:, 0:3], axis=1)))
        da = float(np.max(_quat_angle(a[:, 3:7], b[:, 3:7])))
        rest = {k: bool(np.array_equal(a[:, s], b[:, s])) for k, s in (("vel", slice(7, 10)), ("omega", slice(10, 13)), ("f_b", slice(13, 16)))}
        p = dp <= POS_TOL_M and da <= ATT_TOL_RAD and all(rest.values())
        out["uavs"][u] = {"pos_max_m": dp, "att_max_rad": da, **{f"{k}_bitwise": v for k, v in rest.items()}, "pass": p}
        ok = ok and p
    out["tol"] = {"pos_m": POS_TOL_M, "att_rad": ATT_TOL_RAD}
    out["pass"] = bool(ok)
    return out


# ---------------------------------------------------------------------------------------------------------------- M25
class _NoTape:
    """无演员的磁带替身（M25 的 `tape.actors` 与 `actor_states`）。"""

    actors: tuple = ()
    path = Path(".")

    @staticmethod
    def actor_states(t_ns: int):
        return SimpleNamespace(ids=np.zeros(0, np.int64), pos_enu_m=np.zeros((0, 3)), yaw_rad=np.zeros(0), state=np.zeros(0, np.uint8))


class _FlatRaster:
    """光栅相机的 CPU 替身（无 GPU 时的相机路径：按位姿生成确定性渐变图）。"""

    def render(self, cm, R_ws, p_w, *, fx=None, rs_nodes=None, actors=None, linear=False):
        from awr.sensorsvc.camera.raster import RenderOut

        h, w = cm.height, cm.width
        v = int(abs(float(p_w[0]) * 7 + float(p_w[1]) * 3 + float(p_w[2]) * 11)) % 200
        rgb = np.broadcast_to((np.arange(w, dtype=np.uint16)[None, :, None] * 255 // w + v).astype(np.uint8) % 255, (h, w, 3)).copy()
        z16, z8 = np.zeros((h, w), np.uint16), np.zeros((h, w), np.uint8)
        return RenderOut(rgb, np.zeros((h, w), np.float32), z16, z8, 0.0, 0.0, 0, instance_unocc=z16)


def _l2(uavs: Sequence[str], *, world=None, world_dir: Path | None = None, renderer=None, seed: int = 7, noise: str = "realistic",
        camera_profile: str = "rgb720-gs"):
    """M25 L2 传感器（Env 同一实现）：LiDAR 为 `world`（BoxWorld 或 MeshWorldQuery）上的 MeshLidar，IMU 为 MID-360 内置 IMU。"""
    from awr.agentsdk.env.l2 import L2Config, L2Sensors, worldquery_cache_dir
    from awr.sensorsvc.lidar.mesh import MeshLidar
    from awr.sensorsvc.lidar.model import load_lidar_model

    if world is None:
        from awr.world.mesh.query import open_mesh_world_query

        world = open_mesh_world_query(Path(world_dir), worldquery_cache_dir(world_dir))      # AWR_WORLDQUERY_CACHE_DIR 或按世界目录推导
    lidar = MeshLidar(world, load_lidar_model(), [])
    return L2Sensors(world_dir=Path(world_dir or "."), tape=_NoTape(), uavs=list(uavs), seed=seed, noise=noise, obs_level="L2", nav=None,
                     mount_t_bs_m=(0.1, 0.0, -0.05), hfov_deg=80.0,
                     cfg=L2Config(camera_profile=camera_profile, gnss=False, lidar_proc=False),
                     renderer=renderer if renderer is not None else (_FlatRaster() if world_dir is None else None), lidar=lidar)


def lidar_point_poses(nodes: Sequence[tuple[np.ndarray, np.ndarray]], model) -> tuple[np.ndarray, np.ndarray]:
    """M25 网格 LiDAR 的逐槽位位姿（与 `MeshLidar.scan` 同式：节点间位置线性、四元数同半球 nlerp）→ (o (N, 3), q (N, 4) xyzw)。"""
    from awr.sensorsvc.poses import R_to_quat_xyzw

    N, K = model.slots_per_frame, len(nodes)
    t_rel = np.arange(N) * model.point_dt_ns
    seg_ns = model.frame_ns // (K - 1)
    seg = np.minimum(t_rel // seg_ns, K - 2)
    u = (t_rel - seg * seg_ns) / seg_ns
    P = np.array([p for _, p in nodes], np.float64)
    Q = np.array([np.asarray(R_to_quat_xyzw(np.asarray(R, np.float64)), np.float64) for R, _ in nodes])
    for k in range(1, K):
        if np.dot(Q[k], Q[k - 1]) < 0:
            Q[k] = -Q[k]
    o = P[seg] * (1 - u)[:, None] + P[seg + 1] * u[:, None]
    q = Q[seg] * (1 - u)[:, None] + Q[seg + 1] * u[:, None]
    return o, q / np.linalg.norm(q, axis=1, keepdims=True)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:, 0:3], q[:, 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _feed(l2, run: CaseRun, k: int) -> None:
    """tick k 的样本（k = 0：初始样本；否则 (t_{k−1}, t_k]）交给 L2 传感器，与 Env `_feed_l2` 同一切分。"""
    t_hi = k * TICK_NS
    m = run.t_ns == 0 if k == 0 else (run.t_ns > t_hi - TICK_NS) & (run.t_ns <= t_hi)
    for u in run.state:
        l2.ingest(u, run.t_ns[m], run.state[u][m])


def m25_compare(ref: CaseRun, got: CaseRun, *, world: BoxWorld | None = None, seed: int = 7, noise: str = "realistic",
                max_ticks: int | None = None) -> dict:
    """两份 PoseHistory（Fast 参考、Realistic 注入采集）送入同一 M25：逐帧 LiDAR 槽位位姿与世界点位移、点记录、IMU 输出的差。"""
    from awr.agentsdk.env.l2 import _scan

    world = world or BoxWorld()
    la, lb = _l2(ref.state, world=world, seed=seed, noise=noise), _l2(got.state, world=world, seed=seed, noise=noise)
    n_ticks = int(ref.t_ns[-1] // TICK_NS) if max_ticks is None else int(max_ticks)
    acc = {"origin_max_m": 0.0, "att_max_rad": 0.0, "point_world_max_m": 0.0, "frames": 0, "points": 0, "points_equal_frames": 0,
           "range_max_diff_m": 0.0, "slot_set_equal_frames": 0}
    imu = {"acc_max_mps2": 0.0, "gyro_max_rps": 0.0, "samples": 0, "bitwise_blocks": 0, "blocks": 0}
    m = la.lmodel
    for k in range(n_ticks + 1):
        _feed(la, ref, k)
        _feed(lb, got, k)
        if k == 0:
            continue
        t_s = (k - 1) * TICK_NS
        for no, u in enumerate(ref.state):
            na, nb = la._lidar_nodes(u, t_s), lb._lidar_nodes(u, t_s)
            oa, qa = lidar_point_poses(na, m)
            ob, qb = lidar_point_poses(nb, m)
            sa = _scan(la.lidar, u, no, k, t_s, na, None, seed, la.prof.name)
            sb = _scan(lb.lidar, u, no, k, t_s, nb, None, seed, lb.prof.name)
            xa = np.frombuffer(sa["xyz_m"], "<f4").reshape(-1, 3).astype(np.float64)
            xb = np.frombuffer(sb["xyz_m"], "<f4").reshape(-1, 3).astype(np.float64)
            ta = np.frombuffer(sa["offset_time_ns"], "<u4")
            tb = np.frombuffer(sb["offset_time_ns"], "<u4")
            slot = (ta // m.point_dt_ns).astype(np.int64)
            pw_a = oa[slot] + _qrot(qa[slot], xa)                       # 测得点的世界坐标（同一传感器系点，两侧位姿）
            pw_b = ob[slot] + _qrot(qb[slot], xa)
            acc["origin_max_m"] = max(acc["origin_max_m"], float(np.max(np.linalg.norm(oa - ob, axis=1))))
            acc["att_max_rad"] = max(acc["att_max_rad"], float(np.max(_quat_angle(qa, qb))))
            if len(slot):
                acc["point_world_max_m"] = max(acc["point_world_max_m"], float(np.max(np.linalg.norm(pw_a - pw_b, axis=1))))
            acc["frames"] += 1
            acc["points"] += len(slot)
            if np.array_equal(ta, tb):
                acc["slot_set_equal_frames"] += 1
                if len(ta):
                    acc["range_max_diff_m"] = max(acc["range_max_diff_m"],
                                                  float(np.max(np.abs(np.linalg.norm(xa, axis=1) - np.linalg.norm(xb, axis=1)))))
            if sa == sb:
                acc["points_equal_frames"] += 1
            ia, ib = la._imu(u, no, k, t_s, k * TICK_NS), lb._imu(u, no, k, t_s, k * TICK_NS)
            if ia is None or ib is None:
                continue
            aa, ab = np.frombuffer(ia["acc_mps2"], "<f4"), np.frombuffer(ib["acc_mps2"], "<f4")
            ga, gb = np.frombuffer(ia["gyro_rps"], "<f4"), np.frombuffer(ib["gyro_rps"], "<f4")
            imu["blocks"] += 1
            if ia["t0_ns"] != ib["t0_ns"] or len(aa) != len(ab):
                imu["acc_max_mps2"] = math.inf
                continue
            imu["samples"] += ia["n"]
            imu["acc_max_mps2"] = max(imu["acc_max_mps2"], float(np.max(np.abs(aa.astype(np.float64) - ab))))
            imu["gyro_max_rps"] = max(imu["gyro_max_rps"], float(np.max(np.abs(ga.astype(np.float64) - gb))))
            imu["bitwise_blocks"] += int(ia == ib)
    lidar_ok = acc["origin_max_m"] <= LIDAR_TOL_M and acc["point_world_max_m"] <= LIDAR_TOL_M and acc["frames"] > 0
    imu_ok = imu["acc_max_mps2"] <= IMU_TOL["acc_mps2"] and imu["gyro_max_rps"] <= IMU_TOL["gyro_rps"] and imu["blocks"] > 0
    return {"lidar": {**acc, "tol_m": LIDAR_TOL_M, "pass": bool(lidar_ok)}, "imu": {**imu, "tol": dict(IMU_TOL), "pass": bool(imu_ok)},
            "seed": seed, "noise": noise, "pass": bool(lidar_ok and imu_ok)}


def m25_digest(run: CaseRun, *, world=None, world_dir: Path | None = None, renderer=None, seed: int = 7, noise: str = "realistic",
               camera_profile: str = "rgb720-gs", dz_m: float = 0.0, max_ticks: int | None = None) -> dict:
    """一份 PoseHistory 送入 M25 全路径（`L2Sensors.sense_all`：相机 JPEG、LiDAR 点记录、IMU 块）→ 逐传感器输出 sha256。
    `dz_m` 把轨迹整体抬高（在真实世界上运行时置于地表之上）；相同输入两次调用的摘要逐字节相同即"给定相同 PoseHistory 时 M25
    输出逐字节相同"。"""
    if dz_m:
        run = CaseRun(run.case, run.tier, run.t_ns, {u: s + np.r_[0.0, 0.0, dz_m, np.zeros(13)] for u, s in run.state.items()},
                      run.status, run.contacts, run.dispatch, run.info)
    l2 = _l2(run.state, world=world if world_dir is None else None, world_dir=world_dir, renderer=renderer, seed=seed, noise=noise,
             camera_profile=camera_profile)
    h = {k: hashlib.sha256() for k in ("eo", "lidar0", "lidar0_imu", "pose_history")}
    for u in run.state:
        h["pose_history"].update(run.pose_bytes(u))
    n_ticks = int(run.t_ns[-1] // TICK_NS) if max_ticks is None else int(max_ticks)
    counts = dict.fromkeys(h, 0)
    ctx = getattr(l2.cam, "ctx", None)
    items = [(u, i) for i, u in enumerate(run.state)]
    try:
        for k in range(n_ticks + 1):
            _feed(l2, run, k)
            if ctx is not None:
                with ctx:
                    out = l2.sense_all(k, items)
            else:
                out = l2.sense_all(k, items)
            for u, _no in items:
                sensors, _rows = out[u]
                for sid, frames in sensors.items():
                    for f in frames:
                        h[sid].update(json.dumps(f, sort_keys=True, default=_json_default).encode())
                        counts[sid] += 1
    finally:
        l2.close()
    return {"sha256": {k: v.hexdigest() for k, v in h.items()}, "frames": counts, "encoder": l2.encoder, "ticks": n_ticks,
            "renderer": type(l2.cam).__name__, "camera_profile": camera_profile, "noise": noise, "seed": seed}


# ---------------------------------------------------------------------------------------------------------------- 指标
def _tilt_deg(s: np.ndarray) -> np.ndarray:
    qx, qy = s[:, 3], s[:, 4]
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)))


def _collapse(seq: Sequence[str]) -> list[str]:
    out: list[str] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _first_t(t: np.ndarray, mask: np.ndarray) -> float | None:
    i = np.flatnonzero(mask)
    return float(t[i[0]]) if i.size else None


def _rise(t: np.ndarray, v: np.ndarray, t0: float, v1: float) -> tuple[float | None, float]:
    """10–90% 上升时间与超调（%）：t ≥ t0 的速度分量相对目标 v1。"""
    m = t >= t0
    tt, vv = t[m], v[m]
    a, b = _first_t(tt, vv >= 0.1 * v1), _first_t(tt, vv >= 0.9 * v1)
    return (None if a is None or b is None else b - a), float(max(0.0, (vv.max() - v1) / v1 * 100.0))


def case_metrics(case: Case, run: CaseRun) -> dict:
    """单档指标（各用例按其设定）。"""
    t = run.t_ns / NS
    out: dict[str, Any] = {}
    p = case.params
    u0 = case.uavs[0]
    s = run.state[u0]
    kind = case.metric
    if kind == "takeoff_land":
        z_goal = s[0, 2] + p["alt_m"]
        t_to = p["takeoff_tick"] * TICK_MS / 1000
        out["t_reach_s"] = (lambda x: None if x is None else x - t_to)(_first_t(t, (t >= t_to) & (s[:, 2] >= z_goal - 0.3)))
        td = [c for c in run.contacts if c["uav"] == u0 and c["kind"] == "TOUCHDOWN"]
        t_land = p["land_tick"] * TICK_MS / 1000
        out["t_touchdown_s"] = td[0]["t_ns"] / NS - t_land if td else None
        # 触地竖直速度：触地前 0.3 s 内的最大下降速度（宿主落地判定成立前，gz 与 Fast 的事件时刻定义不同）
        z0 = s[0, 2]
        near = np.flatnonzero((t >= t_land) & (s[:, 2] <= z0 + 0.05))
        if near.size:
            j = near[0]
            w = slice(max(j - 75, 0), j + 1)
            out["vz_touchdown_mps"] = float(-np.min(s[w, 9]))
        else:
            out["vz_touchdown_mps"] = None
        out["z_rest_m"] = float(s[-1, 2])
    elif kind == "hover":
        out["drift_max_m"] = float(np.max(np.linalg.norm(s[:, 0:3] - s[0, 0:3], axis=1)))
        out["tilt_mean_deg"] = float(np.mean(_tilt_deg(s)))
    elif kind == "velocity_step":
        t0 = p["step_tick"] * TICK_MS / 1000
        win = t < p["until_tick"] * TICK_MS / 1000
        out["rise_s"], out["overshoot_pct"] = _rise(t[win], s[win, 7], t0, p["v_mps"])
        out["onset_s"] = _onset(t, s, t0)
    elif kind == "velocity_limit":
        tail = t >= t[-1] - 2.0
        out["v_steady_mps"] = float(np.mean(np.linalg.norm(s[tail, 7:9], axis=1)))
        out["v_max_mps"] = float(np.max(np.linalg.norm(s[:, 7:9], axis=1)))
    elif kind == "yaw_step":
        t0 = p["step_tick"] * TICK_MS / 1000
        wz = s[:, 12]
        t90 = _first_t(t, (t >= t0) & (wz >= 0.9 * p["yaw_rate_rps"]))
        out["t90_s"] = None if t90 is None else t90 - t0
        out["yaw_rate_end_rps"] = float(np.mean(wz[t >= t[-1] - 1.0]))
    elif kind == "waypoints":
        path = np.asarray(p["path"], np.float64)
        t0 = p["start_tick"] * TICK_MS / 1000
        arr, ta, start = [], t0, s[0, 0:3]
        for wp in path:
            x = _first_t(t, (t >= ta) & (np.linalg.norm(s[:, 0:3] - wp, axis=1) < ARRIVE_M))
            arr.append(None if x is None else x - t0)
            if x is None:
                break
            ta = x
        out["arrival_s"] = arr
        legs = np.linalg.norm(np.diff(np.vstack([start, path]), axis=0), axis=1)
        out["path_m"] = np.cumsum(legs).tolist()
        out["onset_s"] = _onset(t, s, t0)
    elif kind == "orbit":
        c = np.asarray(p["center"], np.float64)
        t0 = p["start_tick"] * TICK_MS / 1000
        m = t >= t0
        th = np.unwrap(np.arctan2(s[m, 1] - c[1], s[m, 0] - c[0]))
        out["t_turn_s"] = (lambda x: None if x is None else x - t0)(_first_t(t[m], np.abs(th - th[0]) >= 2 * math.pi - 1e-3))
        r = np.linalg.norm(s[m, 0:2] - c[:2], axis=1)
        out["radius_rms_m"] = float(np.sqrt(np.mean((r - p["radius_m"]) ** 2)))
        out["path_m"] = 2 * math.pi * p["radius_m"]
        out["onset_s"] = _onset(t, s, t0)
    elif kind == "contact_crash":
        for u in case.uavs:
            ev = [x for x in run.contacts if x["uav"] == u]
            out[f"{u}.t_crash_s"] = ev[0]["t_ns"] / NS if ev else None
            out[f"{u}.crash_pos_m"] = ev[0]["pos_enu_m"] if ev else None
            out[f"{u}.crash_kind"] = ev[0]["kind"] if ev else None
    elif kind == "wind_hover":
        tail = t >= p["steady_from_s"]
        out["tilt_deg"] = float(np.mean(_tilt_deg(s[tail])))
        out["offset_steady_m"] = float(np.mean(np.linalg.norm(s[tail, 0:3] - s[0, 0:3], axis=1)))
        out["dev_max_m"] = float(np.max(np.linalg.norm(s[:, 0:3] - s[0, 0:3], axis=1)))
    elif kind == "goto_stop":
        tgt = np.asarray(p["target"], np.float64)
        t0 = p["start_tick"] * TICK_MS / 1000
        d = np.linalg.norm(s[:, 0:2] - tgt, axis=1)
        out["t_arrive_s"] = (lambda x: None if x is None else x - t0)(_first_t(t, (t >= t0) & (d < ARRIVE_M)))
        out["rest_err_m"] = float(d[-1])
        out["v_end_mps"] = float(np.linalg.norm(s[-1, 7:9]))
        out["onset_s"] = _onset(t, s, t0)
    elif kind == "station_keep":
        tail = t >= p["steady_from_s"]
        out["drift_max_m"] = float(np.max(np.linalg.norm(s[:, 0:2] - s[0, 0:2], axis=1)))
        out["offset_steady_m"] = float(np.mean(np.linalg.norm(s[tail, 0:2] - s[0, 0:2], axis=1)))
        out["v_mean_mps"] = float(np.mean(np.linalg.norm(s[tail, 7:9], axis=1)))
    elif kind == "slope_hold":
        tail = t >= p["steady_from_s"]
        out["v_hold_mps"] = float(np.mean(np.linalg.norm(s[tail, 7:9], axis=1)))
        gx, gy = p["plane"]
        off = s[:, 2] - (gx * s[:, 0] + gy * s[:, 1])
        out["z_err_m"] = float(np.max(np.abs(off - off[0])))
        out["tilt_deg"] = float(np.mean(_tilt_deg(s[tail])))
    return out


def _onset(t: np.ndarray, s: np.ndarray, t_apply: float) -> float | None:
    """指令生效时刻：生效时刻之后速度相对生效时刻速度的变化首次超过 ONSET_DV_MPS 的样本时刻（相对 t_apply）。"""
    i0 = int(np.searchsorted(t, t_apply - 1e-9))
    dv = np.linalg.norm(s[i0:, 7:10] - s[i0, 7:10], axis=1)
    j = np.flatnonzero(dv > ONSET_DV_MPS)
    return float(t[i0 + j[0]] - t_apply) if j.size else None


def _diff(a, b):
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def compare_case(case: Case, fast: CaseRun, real: CaseRun, labels: tuple[str, str] = ("fast", "realistic")) -> dict:
    """一个用例的接口判定与动力学差异（两档；`labels` 为报告中两档的名字，缺省 Fast 对 Realistic，也用于 Fast 对统一 L2 引擎）。"""
    iface: dict[str, Any] = {}
    la, lb = labels
    fails: list[str] = []
    da = [(d["tick"], d["uav"], d["kind"], d["ok"], d["code"]) for d in fast.dispatch]
    db = [(d["tick"], d["uav"], d["kind"], d["ok"], d["code"]) for d in real.dispatch]
    iface["dispatch_equal"] = da == db and all(x[3] for x in da)
    if not iface["dispatch_equal"]:
        fails.append("dispatch receipts differ or were rejected")
    st = {}
    for u in case.uavs:
        sa, sb = _collapse(fast.status[u]), _collapse(real.status[u])
        exp = list(case.expect_status.get(u, ()))
        st[u] = {la: sa, lb: sb, "expected": exp, "pass": sa == sb and (not exp or sa == exp)}
        if not st[u]["pass"]:
            fails.append(f"{u}: status sequence {la} {sa} {lb} {sb} expected {exp}")
    iface["status"] = st
    ct = {}
    for u, exp in case.expect_contacts.items():
        ka = [c["kind"] for c in fast.contacts if c["uav"] == u][:len(exp)]
        kb = [c["kind"] for c in real.contacts if c["uav"] == u][:len(exp)]
        ct[u] = {la: ka, lb: kb, "expected": list(exp), "pass": ka == kb == list(exp)}
        if not ct[u]["pass"]:
            fails.append(f"{u}: contacts {la} {ka} {lb} {kb} expected {list(exp)}")
    iface["contacts"] = ct
    n_exp = case.n_ticks * TICK_NS // STEP_NS + 1
    grid = bool(np.array_equal(fast.t_ns, real.t_ns)) and len(fast.t_ns) == n_exp and bool(np.all(np.diff(fast.t_ns) == STEP_NS))
    unit = all(float(np.max(np.abs(np.linalg.norm(r.state[u][:, 3:7], axis=1) - 1.0))) < 1e-9 for r in (fast, real) for u in case.uavs)
    iface["pose_history_grid"] = {"samples": n_exp, "t_bitwise": grid, "quat_unit": unit, "pass": grid and unit}
    if not (grid and unit):
        fails.append("PoseHistory grid differs")
    ma, mb = case_metrics(case, fast), case_metrics(case, real)
    if "onset_s" in ma:
        ok = all(x is not None and 0.0 < x <= ONSET_MAX_S + 1e-9 for x in (ma["onset_s"], mb["onset_s"]))
        iface["command_onset"] = {f"{la}_s": ma["onset_s"], f"{lb}_s": mb["onset_s"], "max_s": ONSET_MAX_S, "pass": ok}
        if not ok:
            fails.append(f"command onset is not within (0, {ONSET_MAX_S}] s after the apply tick")
    iface["pass"] = not fails
    # 动力学差异
    diffs: dict[str, Any] = {}
    tol = dict(case.tol)
    pos_rms = {u: float(np.sqrt(np.mean(np.sum((fast.state[u][:, 0:3] - real.state[u][:, 0:3]) ** 2, axis=1)))) for u in case.uavs}
    if "pos_rms_m" in tol:
        diffs["pos_rms_m"] = max(pos_rms.values())
    for k in ("t_reach_s", "t_touchdown_s", "vz_touchdown_mps", "rise_s", "t90_s", "tilt_deg", "radius_rms_m", "v_steady_mps", "t_arrive_s",
              "v_hold_mps"):
        if k in tol:
            diffs[k] = _diff(ma.get(k), mb.get(k))
    if "overshoot_pct" in tol:
        diffs["overshoot_pct"] = _diff(ma.get("overshoot_pct"), mb.get("overshoot_pct"))
    for k in ("drift_max_m", "offset_steady_m", "rest_err_m", "z_err_m"):
        if k in tol:
            diffs[k] = max(ma[k], mb[k])                                            # 两档各自的上界（悬停漂移、定点偏差、停车误差、地形跟随）
    kind = case.metric
    if "arrival_s_per_100m" in tol:
        if kind == "waypoints":
            aa, ab, pm = ma["arrival_s"], mb["arrival_s"], ma["path_m"]
            if len(aa) == len(ab) == len(pm) and None not in aa and None not in ab:
                diffs["arrival_s_per_100m"] = max(abs(x - y) / (d / 100.0) for x, y, d in zip(aa, ab, pm, strict=True))
            else:
                diffs["arrival_s_per_100m"] = None
        else:
            d = _diff(ma.get("t_turn_s"), mb.get("t_turn_s"))
            diffs["arrival_s_per_100m"] = None if d is None else d / (ma["path_m"] / 100.0)
    if kind == "contact_crash":
        tc, pc = [], []
        for u in case.uavs:
            tc.append(_diff(ma[f"{u}.t_crash_s"], mb[f"{u}.t_crash_s"]))
            pa, pb = ma[f"{u}.crash_pos_m"], mb[f"{u}.crash_pos_m"]
            # 坠地事件的位置为接触点（gz）或机体参考点（Fast），只比水平位置；撞墙比墙法向坐标
            pc.append(None if pa is None or pb is None else float(np.linalg.norm(np.subtract(pa, pb)[:2])))
        diffs["t_crash_s"] = None if None in tc else max(tc)
        diffs["crash_pos_m"] = None if None in pc else max(pc)
    within = {k: (v is not None and v <= tol[k]) for k, v in diffs.items()}
    if kind == "velocity_limit":
        cap = case.params["v_cap_mps"] + case.params["cap_margin_mps"]
        ok_cap = ma["v_max_mps"] <= cap and mb["v_max_mps"] <= cap
        iface["limit_cap"] = {"cap_mps": cap, f"{la}_max_mps": ma["v_max_mps"], f"{lb}_max_mps": mb["v_max_mps"], "pass": ok_cap}
        if not ok_cap:
            iface["pass"] = False
            fails.append("speed above the 12 m/s limit")
    return {"id": case.id, "title": case.title, "interface": iface, "failures": fails,
            "metrics": {la: ma, lb: mb}, "tiers": [la, lb], "pos_rms_m": pos_rms,
            "diff": diffs, "tol": tol, "within": within, "tol_pass": all(within.values()),
            "pass": bool(iface["pass"] and all(within.values())),
            "wall_s": {la: fast.wall_s, lb: real.wall_s}}


# ---------------------------------------------------------------------------------------------------------------- 报告
def build_report(cases: Sequence[dict], *, sampling: dict | None = None, m25: dict | None = None, meta: Mapping[str, Any] | None = None) -> dict:
    """`tier_compare.json`（drone4d.phys.tier_compare.v1）。"""
    from awr.phys.fast import clock_of

    cp = clock_of("d1_250")
    doc = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vehicle": VEHICLE,
        "clock_profile": {"name": cp.name, "main_hz": cp.main_hz, "physics_hz": cp.physics_hz, "l1_hz": cp.l1_hz},
        "tick_ms": TICK_MS,
        "pose_history_hz": NS // STEP_NS,
        "wind_mode": "field",
        "turbulence_sigma": 0.0,
        "drag_law": "composite",
        "collision_source": "box primitives (analytic, identical on both tiers)",
        "m09_autonomy": "off (Bench composition)",
        "tolerance_status": "initial (M24 §6.10.6 where defined; others set by P09, frozen by ADR after first measurement)",
        **dict(meta or {}),
        "cases": list(cases),
        "summary": {"cases": len(cases), "interface_pass": sum(bool(c["interface"]["pass"]) for c in cases),
                    "tol_pass": sum(bool(c["tol_pass"]) for c in cases), "pass": sum(bool(c["pass"]) for c in cases)},
    }
    if sampling is not None:
        doc["sampling_consistency"] = sampling
    if m25 is not None:
        doc["m25_bytewise"] = m25
    doc["pass"] = bool(all(c["pass"] for c in cases) and (sampling is None or sampling.get("pass")) and (m25 is None or m25.get("pass")))
    return doc


# ---------------------------------------------------------------------------------------------------------------- CLI
SETS = {"air": CASES, "ground": GROUND_CASES, "water": WATER_CASES}


def _cases(sel: str | None, set_: str = "air") -> list[Case]:
    pool = SETS[set_]
    if not sel:
        return list(pool)
    want = set(sel.split(","))
    return [c for c in pool if c.id in want]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m awr.phys.tier_compare")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tier", choices=("fast", "l2", "st", "fossen", "realistic"), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cases")
    r.add_argument("--set", default="air", choices=tuple(SETS), help="case set: air (x500, Fast / L2 / gz), ground (L2 / ST oracle) or water (L2 / Fossen oracle)")
    r.add_argument("--launcher", default="local")
    r.add_argument("--image", default="d3-gz:dev")
    m = sub.add_parser("m25")
    m.add_argument("--out", required=True)
    m.add_argument("--world")
    m.add_argument("--camera-profile", default="rgb1080-rs")
    m.add_argument("--max-ticks", type=int)
    m.add_argument("--dz", type=float, default=0.0)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--tiers", default="fast,realistic", help="the two tiers to compare, e.g. fast,l2, l2,st or l2,fossen")
    p.add_argument("--set", default="air", choices=tuple(SETS))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        for c in _cases(a.cases, a.set):
            if a.tier in ORACLES and c.params.get("oracle") is False:
                print(f"[tier_compare] {c.id} {a.tier}: engine-only case, skipped", flush=True)
                continue
            run = RUNNERS[a.tier](c) if a.tier in RUNNERS else run_realistic(c, out / "gz" / c.id, launcher=a.launcher, image=a.image)
            run.save(out)
            print(f"[tier_compare] {c.id} {a.tier}: {run.wall_s} s", flush=True)
        if a.set == "air" and a.tier not in ("l2", *ORACLES) and (not a.cases or REF_CASE.id in a.cases.split(",")):
            if a.tier == "fast":
                ref = run_fast(REF_CASE)
                ref.save(out)
                write_inject(ref, out / "inject_ref.f64")
            else:
                got = run_realistic(REF_CASE, out / "gz" / "inject", launcher=a.launcher, image=a.image, inject=out / "inject_ref.f64")
                got.save(out)
            print(f"[tier_compare] {REF_CASE.id} {a.tier} done", flush=True)
        return 0
    if a.cmd == "m25":
        ref = CaseRun.load(out, REF_CASE.id, "fast")
        doc: dict = {}
        if (out / f"{REF_CASE.id}_inject.realistic.json").is_file():
            got = CaseRun.load(out, f"{REF_CASE.id}_inject", "realistic")
            doc["sampling"] = {"pose_history": pose_history_diff(ref, got), "m25": m25_compare(ref, got)}
            doc["sampling"]["pass"] = bool(doc["sampling"]["pose_history"]["pass"] and doc["sampling"]["m25"]["pass"])
        wd = Path(a.world) if a.world else None
        runs = [m25_digest(ref, world=BoxWorld() if wd is None else None, world_dir=wd, camera_profile=a.camera_profile if wd else "rgb720-gs",
                           dz_m=a.dz, max_ticks=a.max_ticks) for _ in range(2)]
        doc["m25_bytewise"] = {"runs": runs, "world": str(wd) if wd else "BoxWorld", "pass": runs[0]["sha256"] == runs[1]["sha256"]}
        (out / "m25.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
        print(json.dumps({k: v.get("pass") for k, v in doc.items()}), flush=True)
        return 0
    cases = []
    ta, tb = (x.strip() for x in a.tiers.split(","))
    meta: dict = {"tiers": [ta, tb], "case_set": a.set}
    if a.set != "air":
        meta["vehicle"] = sorted({c.vehicle for c in SETS[a.set]})
    skipped = []
    for c in SETS[a.set]:
        if not all((out / f"{c.id}.{t}.json").is_file() for t in (ta, tb)):
            skipped.append(c.id)                                                   # engine-only case (no oracle run)
            continue
        fa, re_ = CaseRun.load(out, c.id, ta), CaseRun.load(out, c.id, tb)
        cases.append(compare_case(c, fa, re_, (ta, tb)))
        meta.setdefault("kernel", fa.info.get("kernel"))
        meta.setdefault(tb, {k: re_.info.get(k) for k in ("gz_sim_version", "plugin_build", "engine", "kernel")})
    if skipped:
        meta["skipped"] = skipped
    m25 = json.loads((out / "m25.json").read_text(encoding="utf-8")) if (out / "m25.json").is_file() and a.set == "air" else {}
    doc = build_report(cases, sampling=m25.get("sampling"), m25=m25.get("m25_bytewise"), meta=meta)
    name = "tier_compare.json" if (ta, tb, a.set) == ("fast", "realistic", "air") else f"tier_compare.{a.set + '.' if a.set != 'air' else ''}{ta}-{tb}.json"
    (out / name).write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(doc["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
