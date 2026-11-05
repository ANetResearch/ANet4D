"""bench-scheduler 的成本估算与准入（AWR-05 §4.7、§5.6.2、§5.6.4、ADR-162、ADR-163、ADR-178；M27 §6.7.2、§6.7.3、
FR-019、FR-021、FR-022、FR-061；D3-AC-22）。

本模块是纯函数式的调度核心：不启动进程、不读写数据库，由调度器进程（M27 §8.1 `python -m awr.bench.sched`）以队列与运行表、
cgroup 与 `io.stat` 实测驱动，测试直接驱动。时刻一律为 UTC 墙钟秒。

**优先级与评测优先**：优先级类 `preview` > `eval`（正式评测与平台复现审计）> `research` > `batch`（M27-FR-019 v1.1），类内按租户
轮转（DRR，量子 1 个 Run），租户内先到先服务。评测队列非空（有 `eval` 作业排队或运行）时：

- 单卡同时运行的正式评测 Run ≤ 3 个（`eval_max_runs_per_gpu`），每个按 1/3 卡预留 GPU 时间份额（保证份额 ≥ 1/3，空闲时可多用）；
- 排队中的正式评测按"只与评测自身资源池比较"可准入的部分预留余量（GPU 份额、显存、内存、写带宽、存储），其他类作业只能使用
  余量之外的资源；生产、预览与离线类的份额上限按 `1 − 评测已占 − 评测预留` 等比例回收（§4.7 混合负载容量模型）；不中断、不降频已
  准入的运行（ADR-162），卡上负载自然回落后由评测接续；
- `d3.slice` 内正式评测核数 ≤ 30（`d3-eval.slice`），其他批作业合计 ≤ 8 核，FAST-LIVO2（`slam_eval`）并行度 ≤ 4（§4.7、ADR-178）。

**GPU 日历**（M27-FR-022）：独占卡上的验收窗口只准入 `acceptance = true` 的作业，且须在卡上无其他作业时以整卡（份额 1.0，排他锁，
ADR-033）准入，窗口内卡上不再准入其他作业；训练窗口内生产类份额上限降至 50%、交互预览暂停，评测不受影响。正式评测只在独占卡上
执行（P0b 起；`allow_shared_eval` 只用于 P0a 原型榜，决定标注 `shared_card`）；开发卡不准入验收作业。

**写带宽**（ADR-163、M27-FR-061）：预算 `B_write = min(300 MB/s, 0.5 × P-1 实测 /data 持续写带宽)`；按运行中作业估计写速率之和
准入，超出时排队并记录 686；`io.stat` 实测 1 min 均值超预算 10% 时暂停全部新的写入作业准入（686）。`/data/d3` 占用加新作业持久与派生
体积超过配额 95% 时排队并记录 685（保留与清理见 `awr.bench.retention`）。实测 CPU、内存与显存高于估计时以实测为准（§6.7.3）。

成本估算（§6.7.2）按 Run 规格逐通道折算：体积口径取 AWR-05 §5.6.4（720p JPEG 0.33 MB/帧按像素数线性折算），GPU 时间按 720p 等效帧
与最坏档容量 C（暂定 80 帧/s）折算，CPU 与内存按 ADR-178 每实例预算（标准轨正式 Run 10 核、约 30 GB）；系数标"暂定"者在 P-1 实测
后以 ADR 冻结（`CostModel`）。
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from awr.contracts.reasons import Reason

__all__ = ["CLASS_ORDER", "EVAL_CLASS", "Channel", "CostModel", "CostVector", "Decision", "GpuSlot", "Job", "Running", "SchedConfig",
           "Scheduler", "Telemetry", "Window", "cgroup_parent", "channels_of", "estimate_cost", "make_job", "resource_args",
           "slice_quota", "write_budget_mbps"]

PriorityClass = Literal["preview", "eval", "research", "batch"]
GpuClass = Literal["none", "preview", "online", "offline"]
CLASS_ORDER: tuple[str, ...] = ("preview", "eval", "research", "batch")
EVAL_CLASS = "eval"
EPS = 1e-9
MB = 1_000_000
PX_720P = 1280 * 720

# 作业类型 → cgroup 子组（M27 §8.2）；正式评测一律在 d3-eval.slice
_JOBS_SLICE_KINDS = frozenset({"evaluate", "export", "export_rrd", "publish_static", "release", "retention_gc", "world_build",
                               "event_compile", "thermal_bake"})
_LIVO_KINDS = frozenset({"slam_eval", "slam_regress"})


# ── 配置 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def write_budget_mbps(measured_disk_mbps: float | None, *, cap_mbps: float = 300.0, fraction: float = 0.5) -> float:
    """写带宽预算 `min(300 MB/s, 0.5 × 实测 /data 持续写带宽)`（ADR-163 (4)、M27-FR-061）；未给出实测时只取上限 300 MB/s。"""
    if measured_disk_mbps is None:
        return float(cap_mbps)
    if measured_disk_mbps <= 0:
        raise ValueError("measured disk write bandwidth must be positive")
    return min(float(cap_mbps), fraction * float(measured_disk_mbps))


@dataclass(frozen=True)
class SchedConfig:
    """调度与准入参数（M27 §6.8 `configs/bench/bench.yaml`；v1.3 按 ADR-178 修订 `d3.slice` 为 40 核）。"""

    eval_max_runs_per_gpu: int = 3                 # 单卡并发正式 Run 上限（§4.7 评测窗口）
    eval_share: float = 1.0 / 3.0                  # 每个正式 Run 预留并保证的 GPU 时间份额
    share_cap: Mapping[str, float] = field(default_factory=lambda: {"preview": 0.10, "online": 0.60, "offline": 0.30})
    training_production_factor: float = 0.5        # 训练窗口内生产吞吐上限降至 50%，预览暂停
    vram_cap_gb: float = 24.0                      # M25 Worker 显存上限
    gz_max_per_gpu: int = 4                        # gz `xval` 实例每卡上限
    slice_cpu: float = 40.0                        # d3.slice CPU（ADR-178）
    slice_mem_gb: float = 384.0                    # d3.slice 内存（Q-D3-11）
    eval_cpu: float = 30.0                         # d3-eval.slice：正式评测核数上限
    other_cpu_when_eval: float = 8.0               # 评测队列非空时其他批作业合计核数上限
    kind_limit_when_eval: Mapping[str, int] = field(default_factory=lambda: {"slam_eval": 4})   # FAST-LIVO2 并行度
    write_cap_mbps: float = 300.0
    write_fraction: float = 0.5
    measured_disk_mbps: float | None = None        # P-1 fio 实测 /data 持续写带宽（MB/s）
    write_pause_ratio: float = 1.10                # io.stat 1 min 均值超预算 10% 暂停新准入
    storage_quota_bytes: int = 10 * 10**12         # /data/d3 10 TB
    storage_block_ratio: float = 0.95              # ≥ 95% 停止准入（685）
    allow_shared_eval: bool = False                # P0a 原型榜：允许在共享开发卡上执行正式评测并标注
    exclusive_classes: tuple[str, ...] = ("preview", "eval", "batch")   # 可用独占卡的类；开发运行（research）不得绑定独占卡

    @property
    def write_budget_mbps(self) -> float:
        return write_budget_mbps(self.measured_disk_mbps, cap_mbps=self.write_cap_mbps, fraction=self.write_fraction)

    @property
    def write_budget_source(self) -> str:
        return "cap_only" if self.measured_disk_mbps is None else "measured"


@dataclass(frozen=True)
class GpuSlot:
    """一张可调度的卡：`exclusive` 为独占卡（正式评测、生产与验收），否则为共享开发卡（开发期 GPU 2、3 之一）。"""

    index: int
    exclusive: bool = False


@dataclass(frozen=True)
class Window:
    """GPU 日历窗口（UTC 秒，左闭右开）：`acceptance`（验收）或 `training`（训练）；`gpus` 为空时适用于全部独占卡。"""

    kind: Literal["acceptance", "training"]
    start: float
    end: float
    gpus: tuple[int, ...] = ()

    def covers(self, now: float, gpu: int) -> bool:
        return self.start <= now < self.end and (not self.gpus or gpu in self.gpus)


# ── 成本估算 ─────────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostModel:
    """成本系数（M27 §6.7.2、§6.8；AWR-05 §4.7、§5.6.4）；标"暂定"者由 P-1 实测后以 ADR 冻结。"""

    jpeg_mb_720p: float = 0.33                     # 720p nvJPEG q90 每帧（r10 实测 301–360 kB），按像素数线性折算
    raw_rgb_mb_720p: float = 2.0                   # rgb_codec = raw（zstd 后估）
    thermal_mb_per_mpx: float = 0.30 / 0.32768     # therm512 u16 + zstd 估 0.23–0.30 MB/帧（640 × 512）
    depth_mb_720p: float = 1.8                     # L2-depth f32 + 字节重排 + zstd 估 1.4–2.2 MB/帧
    lidar_mb_frame: float = 0.2                    # lidar_frame.v2 + zstd，约 2 MB/s/机
    aux_mb_per_uav_min: float = 5.0                # IMU、GNSS、导航解、动作、消息
    truth_inst_mb_720p: float = 0.03               # 实例、objects[]、invalid_mask，按像素数折算（暂定）
    truth_lidar_mb_frame: float = 0.06             # LiDAR 逐点真值
    truth_pose_mb_per_uav_min: float = 3.0         # 真值位姿 50 Hz
    bag_mb_per_uav_min: float = 430.0              # 派生全点 bag
    rrd_mb_per_uav_min: float = 560.0              # 派生 .rrd（dataset 档）
    l1_bytes_per_uav_s: float = 406.0              # L1 3 机 360 s raw.mcap 439 KB（r6）
    l1_truth_bytes_per_uav_s: float = 400.0
    cam_capacity_worst_fps: float = 80.0           # 最坏档容量 C（720p 等效帧/s，暂定，D3-AC-11）
    thermal_eq_weight: float = 0.0                 # 热成像与可见光同趟渲染，§6.7.2 公式不单列（暂定）
    lidar_tick_ms_per_unit: float = 0.5            # MeshLidar 每 tick 每机 GPU 时间（暂定，r4 §4.4）
    lidar_hz: float = 10.0
    l2_rtf_target: float = 1.0                     # 锁步目标
    l1_rtf: tuple[tuple[int, float], ...] = ((1, 84.0), (3, 40.0), (10, 12.0), (30, 4.6), (60, 1.85))
    prepare_l1_s: float = 5.0
    prepare_l2_s: float = 30.0
    prepare_gz_s: float = 45.0
    env_cpu: float = 2.0                           # Env 与写入器每 Run（ADR-178）
    env_mem_gb: float = 4.0
    gz_physics_cpu: float = 2.0
    gz_physics_mem_gb: float = 2.0
    gz_xval_cpu: float = 4.0
    gz_xval_mem_gb: float = 6.0
    gz_xval_vram_gb: float = 2.5
    gz_xval_share: float = 0.10
    agent_cpu: float = 2.0                         # Agent 容器每机（std）
    agent_mem_gb: float = 8.0
    vram_session_gb_per_uav: float = 0.4           # M25 会话（标准轨 3 机 ≤ 1.2 GB）

    def l1_rtf_at(self, n: int) -> float:
        """L1 实时因子按机数对数插值（r6 §10.4、§12.3）。"""
        xs = [a for a, _ in self.l1_rtf]
        if n <= xs[0]:
            return self.l1_rtf[0][1]
        if n >= xs[-1]:
            return self.l1_rtf[-1][1]
        i = bisect.bisect_right(xs, n)
        (x0, y0), (x1, y1) = self.l1_rtf[i - 1], self.l1_rtf[i]
        t = (math.log(n) - math.log(x0)) / (math.log(x1) - math.log(x0))
        return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))


@dataclass(frozen=True)
class Channel:
    """一个录制通道的成本描述：`kind ∈ {camera, thermal, depth, lidar, aux}`。"""

    uav: str
    sensor: str
    kind: Literal["camera", "thermal", "depth", "lidar", "aux"]
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = "jpeg"

    @property
    def px(self) -> int:
        return self.width * self.height


_DEFAULT_RIG = ("eo", "lidar0", "lidar0_imu", "gnss0")       # P0a L2 缺省传感器（awr.agentsdk.env.l2）
_CAM_DEFAULT = (1920, 1080, 10.0)                             # rgb1080-rs（标准轨，L2Config 缺省）
_THERM_DEFAULT = (640, 512, 10.0)                             # therm512


def _kind_of(sid: str) -> str:
    s = sid.lower()
    if s.endswith("_imu") or s.startswith(("imu", "gnss")):
        return "aux"
    if s.startswith("therm"):
        return "thermal"
    if s.startswith("lidar"):
        return "lidar"
    if s.startswith("depth"):
        return "depth"
    return "camera"                                           # eo、cam*、rgb*、slam_cam


def channels_of(rs: Mapping) -> list[Channel]:
    """Run 规格 → 录制通道。L1 无渲染通道；L2 各机从缺省传感器（`eo`、`lidar0`、`lidar0_imu`、`gnss0`）起，按 `fleet[].sensors[]`
    的 `enabled` 开关与 `overrides`（`width_px`、`height_px`、有理数 `fps`）修改或增加；`L2-depth` 为每个相机增加同分辨率深度通道。
    不交付给 Agent 的传感器（`role = slam`）同样写入 `raw.mcap`，照常计入。"""
    level = rs.get("obs_level", "L2")
    if level in ("L1", "T0"):
        return []
    out: list[Channel] = []
    for f in rs["fleet"]:
        uav = f["uav_id"]
        sensors: dict[str, dict] = {sid: {"enabled": True} for sid in _DEFAULT_RIG}
        for s in f.get("sensors") or []:
            sensors[s["sensor_id"]] = {"enabled": bool(s.get("enabled", True)), **(s.get("overrides") or {})}
        for sid, o in sensors.items():
            if not o.get("enabled", True):
                continue
            kind = _kind_of(sid)
            if kind in ("camera", "thermal", "depth"):
                w0, h0, f0 = _THERM_DEFAULT if kind == "thermal" else _CAM_DEFAULT
                ch = Channel(uav, sid, kind, int(o.get("width_px", w0)), int(o.get("height_px", h0)), float(o.get("fps", f0)),
                             "u16_zstd" if kind == "thermal" else ("f32_zstd" if kind == "depth" else "jpeg"))
                out.append(ch)
                if kind == "camera" and level == "L2-depth":
                    out.append(replace(ch, sensor=f"{sid}.depth", kind="depth", codec="f32_zstd"))
            elif kind == "lidar":
                out.append(Channel(uav, sid, "lidar", fps=float(o.get("fps", 10.0))))
            else:
                out.append(Channel(uav, sid, "aux"))
    return out


@dataclass(frozen=True)
class CostVector:
    """一个作业的资源估计（M27 §6.2.1）。`gpu_share` 为准入时预留的卡 GPU 时间份额（正式评测为 1/3），`write_mbps` 为平均写速率。"""

    gpu_class: GpuClass = "none"
    gpu_share: float = 0.0
    vram_gb: float = 0.0
    gz_instances: int = 0                          # 占用 GPU 的 gz 实例（xval 导出档）；physics 档在 CPU 池
    cpu_cores: float = 1.0
    mem_gb: float = 2.0
    bytes_persist: int = 0
    bytes_derived: int = 0
    write_mbps: float = 0.0
    wall_s_est: float = 60.0
    gpu_s_per_sim_s: float = 0.0                   # 传感器 GPU 时间需求（GPU·s/仿真 s），供报告与核对

    @property
    def needs_gpu(self) -> bool:
        return self.gpu_class != "none"


def _channel_rates(ch: Channel, m: CostModel) -> tuple[float, float]:
    """→ (raw.mcap 字节/仿真 s, 真值字节/仿真 s)。"""
    r = ch.px / PX_720P
    if ch.kind == "camera":
        per = (m.raw_rgb_mb_720p if ch.codec == "raw" else m.jpeg_mb_720p) * r
        return per * ch.fps * MB, m.truth_inst_mb_720p * r * ch.fps * MB
    if ch.kind == "thermal":
        return m.thermal_mb_per_mpx * ch.px / 1e6 * ch.fps * MB, m.truth_inst_mb_720p * r * ch.fps * MB
    if ch.kind == "depth":
        return m.depth_mb_720p * r * ch.fps * MB, 0.0
    if ch.kind == "lidar":
        return m.lidar_mb_frame * ch.fps * MB, m.truth_lidar_mb_frame * ch.fps * MB
    return 0.0, 0.0


def estimate_cost(rs: Mapping, *, priority_class: str = "research", duration_s: float | None = None, model: CostModel | None = None,
                  config: SchedConfig | None = None, channels: Sequence[Channel] | None = None) -> CostVector:
    """Run 规格 → 成本向量（M27 §6.7.2；CPU 与内存按 ADR-178）。

    GPU 份额：正式评测（`eval`）按 1/3 卡预留；其他在线闭环按需求份额与 `online` 上限取小。墙钟 = 准备时长 + max(T / RTF 目标,
    传感器 GPU 时间 / 预留份额)，平均写速率 = (raw.mcap + 真值) 字节 / 墙钟；正式评测运行先写 `/data/d3/scratch` 再精简（ADR-163
    v1.3），写速率与存储准入按完整录制计。"""
    m = model or CostModel()
    cfg = config or SchedConfig()
    T = float(duration_s if duration_s is not None else rs.get("duration_s") or 360.0)
    fleet = rs["fleet"]
    n = len(fleet)
    physics = rs.get("physics", "realistic")
    gz_profile = (rs.get("physics_opts") or {}).get("gz_profile", "physics")
    gz = 1 if physics in ("gz", "sitl") else 0                   # realistic is the unified L2 engine inside the Env process (ADR-226)
    xval = gz if gz_profile == "xval" else 0
    in_containers = (rs.get("policy") or {}).get("kind", "sdk") in ("reference", "image")
    cpu = m.env_cpu + (m.gz_xval_cpu if xval else m.gz_physics_cpu) * gz + (m.agent_cpu * n if in_containers else 0.0)
    mem = m.env_mem_gb + (m.gz_xval_mem_gb if xval else m.gz_physics_mem_gb) * gz + (m.agent_mem_gb * n if in_containers else 0.0)
    chans = list(channels) if channels is not None else channels_of(rs)
    outputs = set(rs.get("outputs") or [])

    if not chans:                                             # L1：无渲染通道，不占 GPU（xval 除外）
        rtf = m.l1_rtf_at(n) if physics == "fast" else m.l2_rtf_target
        persist = (m.l1_bytes_per_uav_s + m.l1_truth_bytes_per_uav_s) * n * T
        wall = (m.prepare_gz_s if gz else m.prepare_l1_s) + T / rtf
        share = m.gz_xval_share * xval
        return CostVector("online" if xval else "none", share, m.gz_xval_vram_gb * xval, xval, cpu, mem, int(persist),
                          _derived_bytes(outputs, n, T, persist, m), max(0.1, persist / wall / MB), wall, 0.0)

    eq_fps = sum(c.fps * c.px / PX_720P for c in chans if c.kind in ("camera", "depth")) \
        + m.thermal_eq_weight * sum(c.fps * c.px / PX_720P for c in chans if c.kind == "thermal")
    n_lidar = sum(1 for c in chans if c.kind == "lidar")
    gpu_need = eq_fps / m.cam_capacity_worst_fps + m.lidar_hz * m.lidar_tick_ms_per_unit * n_lidar / 1000.0   # GPU·s/仿真 s
    need_share = min(1.0, gpu_need * m.l2_rtf_target) + m.gz_xval_share * xval
    share = cfg.eval_share if priority_class == EVAL_CLASS else min(need_share, cfg.share_cap["online"])
    gpu_class: GpuClass = "preview" if priority_class == "preview" else "online"
    if priority_class == "preview":
        share = min(need_share, cfg.share_cap["preview"])
    sensor_share = max(share - m.gz_xval_share * xval, EPS)
    wall_run = max(T / m.l2_rtf_target, gpu_need * T / sensor_share)
    wall = (m.prepare_gz_s if gz else m.prepare_l2_s) + wall_run
    raw_bps = truth_bps = 0.0
    for c in chans:
        a, b = _channel_rates(c, m)
        raw_bps, truth_bps = raw_bps + a, truth_bps + b
    raw_bps += m.aux_mb_per_uav_min * MB / 60.0 * n
    truth_bps += m.truth_pose_mb_per_uav_min * MB / 60.0 * n
    persist = (raw_bps + truth_bps) * T
    vram = m.vram_session_gb_per_uav * n + m.gz_xval_vram_gb * xval
    return CostVector(gpu_class, share, vram, xval, cpu, mem, int(persist), _derived_bytes(outputs, n, T, raw_bps * T, m),
                      persist / wall / MB, wall, gpu_need)


def _derived_bytes(outputs: set, n: int, T: float, raw_bytes: float, m: CostModel) -> int:
    d = 0.0
    if "bag" in outputs:
        d += m.bag_mb_per_uav_min * MB * n * T / 60.0
    if "rrd" in outputs:
        d += m.rrd_mb_per_uav_min * MB * n * T / 60.0
    if "files" in outputs:
        d += raw_bytes
    return int(d)


# ── 作业、运行表与实测 ───────────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Job:
    """队列中的一个可调度单元（M27 §3 作业）。`kind` 为作业类型（`realistic_run`、`l2_run`、`slam_eval`、`export` 等）。"""

    job_id: str
    tenant: str
    priority_class: str
    cost: CostVector
    submitted: float
    kind: str = "realistic_run"
    acceptance: bool = False

    def __post_init__(self) -> None:
        if self.priority_class not in CLASS_ORDER:
            raise ValueError(f"unknown priority class {self.priority_class!r}")

    @property
    def is_eval(self) -> bool:
        return self.priority_class == EVAL_CLASS


def make_job(job_id: str, rs: Mapping, *, tenant: str, submitted: float, priority_class: str | None = None, kind: str = "realistic_run",
             acceptance: bool = False, model: CostModel | None = None, config: SchedConfig | None = None) -> Job:
    """Run 规格 → 作业：排名运行（`ranked = true`）缺省进入 `eval` 类，其余进入 `research` 类。"""
    cls = priority_class or (EVAL_CLASS if rs.get("ranked") else "research")
    return Job(job_id, tenant, cls, estimate_cost(rs, priority_class=cls, model=model, config=config), float(submitted), kind, acceptance)


@dataclass(frozen=True)
class Running:
    job: Job
    gpu: int | None
    share: float
    started: float

    @property
    def ends_est(self) -> float:
        return self.started + self.job.cost.wall_s_est


@dataclass(frozen=True)
class Telemetry:
    """调度器每 10 s 采集的实测（M27 §6.7.3）：cgroup `cpu.stat`、`memory.current`，`io.stat` 1 min 均值写速率，
    `nvidia-smi` 显存，`/data/d3` 占用；实测高于估计时以实测为准。"""

    cpu_cores: float | None = None
    mem_gb: float | None = None
    write_mbps_1min: float | None = None
    vram_gb: Mapping[int, float] = field(default_factory=dict)
    store_used_bytes: int = 0


@dataclass(frozen=True)
class Decision:
    """一次调度循环中一个作业的结论。`blocked_by` 为不满足的维度（`calendar`、`gpu`、`cpu`、`mem`、`write`、`write_paused`、
    `storage`、`kind_limit`、`eval_priority`），`reasons` 为应记录的原因码（685、686）。"""

    job_id: str
    admitted: bool
    gpu: int | None = None
    share: float = 0.0
    blocked_by: tuple[str, ...] = ()
    reasons: tuple[int, ...] = ()
    condition: str | None = None                   # `shared_card`：正式评测在共享卡上执行（原型榜）


# ── 容量账 ───────────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class _Card:
    slot: GpuSlot
    share_by: dict[str, float] = field(default_factory=dict)  # 账类（preview、online、offline、eval、acceptance）→ 份额
    vram: float = 0.0
    gz: int = 0
    n_eval: int = 0
    n_jobs: int = 0
    locked: bool = False                           # 验收作业的排他锁

    @property
    def used(self) -> float:
        return sum(self.share_by.values())

    @property
    def eval_used(self) -> float:
        return self.share_by.get(EVAL_CLASS, 0.0)


@dataclass
class _Ledger:
    cards: dict[int, _Card]
    cpu_eval: float = 0.0
    cpu_other: float = 0.0
    cpu_extra: float = 0.0                         # cgroup 实测高于估计的部分（以实测为准）
    mem: float = 0.0
    write: float = 0.0
    store: float = 0.0
    kinds: dict[str, int] = field(default_factory=dict)

    @property
    def cpu_total(self) -> float:
        return self.cpu_eval + self.cpu_other + self.cpu_extra

    def book(self, r: Running, remaining: float = 1.0) -> None:
        """把一个运行记入账本；`remaining` 为尚未写出的存储比例（运行中作业按估计墙钟线性折算）。"""
        c = r.job.cost
        if r.gpu is not None and r.gpu in self.cards:
            card = self.cards[r.gpu]
            acct = _acct(r.job)
            card.share_by[acct] = card.share_by.get(acct, 0.0) + r.share
            card.vram += c.vram_gb
            card.gz += c.gz_instances
            card.n_eval += int(r.job.is_eval)
            card.n_jobs += 1
            card.locked |= r.job.acceptance
        if r.job.is_eval:
            self.cpu_eval += c.cpu_cores
        else:
            self.cpu_other += c.cpu_cores
        self.mem += c.mem_gb
        self.write += c.write_mbps
        self.store += (c.bytes_persist + c.bytes_derived) * remaining
        self.kinds[r.job.kind] = self.kinds.get(r.job.kind, 0) + 1


def _acct(job: Job) -> str:
    return "acceptance" if job.acceptance else (EVAL_CLASS if job.is_eval else job.cost.gpu_class)


@dataclass(frozen=True)
class _Reserve:
    """为排队中的正式评测预留的资源：每卡预留的 Run 数与显存、CPU、内存、写速率与存储。"""

    pending: Mapping[int, int] = field(default_factory=dict)
    vram: Mapping[int, float] = field(default_factory=dict)
    cpu: float = 0.0
    mem: float = 0.0
    write: float = 0.0
    store: float = 0.0


_NO_RESERVE = _Reserve()


class Scheduler:
    """调度与准入核心（M27 §6.7.3）。`submit` 入队，`tick(now, telemetry)` 运行一次调度循环并返回每个排队作业的结论（被准入的作业
    进入运行表），`finish` 与 `cancel` 释放资源。`events` 记录 685、686 原因码（每个作业在阻塞维度变化时记一次）。"""

    def __init__(self, gpus: Iterable[GpuSlot], *, config: SchedConfig | None = None, windows: Iterable[Window] = ()) -> None:
        self.cfg = config or SchedConfig()
        self.gpus = tuple(gpus)
        if len({g.index for g in self.gpus}) != len(self.gpus):
            raise ValueError("duplicate GPU index")
        self.windows = tuple(windows)
        self.queue: list[Job] = []
        self.running: dict[str, Running] = {}
        self.events: list[tuple[float, str, int, tuple[str, ...]]] = []
        self._rr: dict[str, list[str]] = {}          # 类 → 租户轮转次序（被准入的租户移到末尾）
        self._blocked: dict[str, tuple[str, ...]] = {}
        self._queued_ids: set[str] = set()

    # 队列操作
    def submit(self, job: Job) -> None:
        if job.job_id in self.running or job.job_id in self._queued_ids:
            raise ValueError(f"duplicate job {job.job_id}")
        self.queue.append(job)
        self._queued_ids.add(job.job_id)

    def cancel(self, job_id: str) -> None:
        self.queue = [j for j in self.queue if j.job_id != job_id]
        self._queued_ids.discard(job_id)
        self.running.pop(job_id, None)
        self._blocked.pop(job_id, None)

    def finish(self, job_id: str) -> Running:
        self._blocked.pop(job_id, None)
        return self.running.pop(job_id)

    @property
    def eval_active(self) -> bool:
        """评测队列非空：有正式评测排队或运行。"""
        return any(j.is_eval for j in self.queue) or any(r.job.is_eval for r in self.running.values())

    def in_window(self, now: float, gpu: int, kind: str) -> bool:
        return any(w.kind == kind and w.covers(now, gpu) for w in self.windows)

    def ordered_queue(self) -> list[Job]:
        """类优先级 → 类内租户轮转（量子 1 个 Run；租户按首次排队先后进入轮转次序，每次被准入后移到末尾）→ 租户内先到先服务。"""
        out: list[Job] = []
        for cls in CLASS_ORDER:
            by_tenant: dict[str, list[Job]] = {}
            for j in sorted((j for j in self.queue if j.priority_class == cls), key=lambda j: (j.submitted, j.job_id)):
                by_tenant.setdefault(j.tenant, []).append(j)
            if not by_tenant:
                continue
            rot = self._rr.setdefault(cls, [])
            rot.extend(t for t in sorted(by_tenant, key=lambda t: (by_tenant[t][0].submitted, t)) if t not in rot)
            tenants = [t for t in rot if t in by_tenant]
            depth = max(len(v) for v in by_tenant.values())
            out.extend(by_tenant[t][i] for i in range(depth) for t in tenants if i < len(by_tenant[t]))
        return out

    # 容量账
    def _ledger(self, now: float, tel: Telemetry) -> _Ledger:
        led = _Ledger({g.index: _Card(g) for g in self.gpus}, store=float(tel.store_used_bytes))
        for r in self.running.values():
            wall = max(r.job.cost.wall_s_est, EPS)
            led.book(r, remaining=min(1.0, max(0.0, 1.0 - (now - r.started) / wall)))
        if tel.cpu_cores is not None:
            led.cpu_extra = max(0.0, float(tel.cpu_cores) - led.cpu_eval - led.cpu_other)
        if tel.mem_gb is not None:
            led.mem = max(led.mem, float(tel.mem_gb))
        for g, v in tel.vram_gb.items():
            if g in led.cards:
                led.cards[g].vram = max(led.cards[g].vram, float(v))
        return led

    def _eval_cards(self, led: _Ledger, job: Job, now: float, pending: Mapping[int, int], *, own_pool_only: bool) -> list[_Card]:
        """可承载一个正式评测 Run 的卡（独占卡优先、已用份额少者优先）；`own_pool_only` 时只与评测自身占用比较，用于计算预留。"""
        cfg, c = self.cfg, job.cost
        out = []
        for card in led.cards.values():
            g = card.slot
            if not (g.exclusive or cfg.allow_shared_eval) or card.locked or self.in_window(now, g.index, "acceptance"):
                continue
            p = pending.get(g.index, 0)
            if card.n_eval + p >= cfg.eval_max_runs_per_gpu:
                continue
            used = (card.eval_used if own_pool_only else card.used) + p * cfg.eval_share
            if used + cfg.eval_share > 1.0 + EPS:
                continue
            if not own_pool_only and (card.vram + c.vram_gb > cfg.vram_cap_gb + EPS or card.gz + c.gz_instances > cfg.gz_max_per_gpu):
                continue
            out.append(card)
        out.sort(key=lambda k: (not k.slot.exclusive, k.used + pending.get(k.slot.index, 0) * cfg.eval_share, k.slot.index))
        return out

    def _reserve_for_eval(self, led: _Ledger, pending_jobs: Sequence[Job], now: float) -> _Reserve:
        """评测队列非空时为排队中的正式评测预留资源：按队列顺序，只与评测自身资源池（每卡 ≤ 3 个且份额合计 ≤ 1、d3-eval.slice ≤ 30 核）
        比较可准入的作业逐个预留；其他类作业只能使用余量之外的资源（评测优先准入）。"""
        pending: dict[int, int] = {}
        vram: dict[int, float] = {}
        cpu = mem = write = store = 0.0
        cfg = self.cfg
        for j in pending_jobs:
            c = j.cost
            if c.cpu_cores > cfg.eval_cpu or c.mem_gb > cfg.slice_mem_gb or c.write_mbps > cfg.write_budget_mbps:
                continue                                      # 单个作业已超出绝对上限，永不可准入，不为其预留
            if led.cpu_eval + cpu + c.cpu_cores > cfg.eval_cpu + EPS:
                continue
            if c.needs_gpu:
                cands = self._eval_cards(led, j, now, pending, own_pool_only=True)
                if not cands:
                    continue
                g = cands[0].slot.index
                pending[g] = pending.get(g, 0) + 1
                vram[g] = vram.get(g, 0.0) + c.vram_gb
            cpu, mem, write = cpu + c.cpu_cores, mem + c.mem_gb, write + c.write_mbps
            store += c.bytes_persist + c.bytes_derived
        return _Reserve(pending, vram, cpu, mem, write, store)

    def _gpu_other(self, led: _Ledger, job: Job, now: float, rsv: _Reserve) -> _Card | None:
        """非评测作业选卡：开发卡随时可用；独占卡只给 `exclusive_classes`（预览与批量生产；开发运行不得绑定独占卡，§4.7），
        验收窗口内只给验收作业（卡上无其他作业时整卡排他）。独占卡上各类份额上限按
        `1 − 评测已占 − 评测预留` 等比例回收，训练窗口内生产类上限再乘 0.5、预览为 0；卡上份额合计含评测预留 ≤ 1。"""
        cfg, c = self.cfg, job.cost
        best: _Card | None = None
        for card in led.cards.values():
            g = card.slot
            accept_win = g.exclusive and self.in_window(now, g.index, "acceptance")
            if job.acceptance:
                if accept_win and card.n_jobs == 0:
                    return card
                continue
            if card.locked or accept_win or (g.exclusive and job.priority_class not in cfg.exclusive_classes):
                continue
            held = rsv.pending.get(g.index, 0) * cfg.eval_share
            factor = max(0.0, 1.0 - card.eval_used - held) if g.exclusive else 1.0
            if g.exclusive and self.in_window(now, g.index, "training"):
                factor *= 0.0 if c.gpu_class == "preview" else cfg.training_production_factor
            if card.share_by.get(c.gpu_class, 0.0) + c.gpu_share > cfg.share_cap.get(c.gpu_class, 1.0) * factor + EPS:
                continue
            if card.used + held + c.gpu_share > 1.0 + EPS:
                continue
            if card.vram + rsv.vram.get(g.index, 0.0) + c.vram_gb > cfg.vram_cap_gb + EPS or card.gz + c.gz_instances > cfg.gz_max_per_gpu:
                continue
            if best is None or (card.used, g.index) < (best.used, best.slot.index):
                best = card
        return best

    def _fits(self, led: _Ledger, job: Job, now: float, rsv: _Reserve, eval_active: bool,
              measured_write: float | None) -> tuple[tuple[str, ...], _Card | None]:
        cfg, c = self.cfg, job.cost
        blocking: list[str] = []
        card: _Card | None = None
        if job.acceptance and not any(g.exclusive and self.in_window(now, g.index, "acceptance") for g in self.gpus):
            return ("calendar",), None
        if job.is_eval and not job.acceptance:
            if c.needs_gpu:
                cands = self._eval_cards(led, job, now, {}, own_pool_only=False)
                card = cands[0] if cands else None
                if card is None:
                    blocking.append("gpu")
            if led.cpu_eval + c.cpu_cores > cfg.eval_cpu + EPS or led.cpu_total + c.cpu_cores > cfg.slice_cpu + EPS:
                blocking.append("cpu")
        else:
            if c.needs_gpu or job.acceptance:
                card = self._gpu_other(led, job, now, rsv)
                if card is None:
                    blocking.append("gpu")
            if (led.cpu_total + c.cpu_cores + rsv.cpu > cfg.slice_cpu + EPS
                    or (eval_active and led.cpu_other + c.cpu_cores > cfg.other_cpu_when_eval + EPS)):
                blocking.append("cpu")
            lim = cfg.kind_limit_when_eval.get(job.kind) if eval_active else None
            if lim is not None and led.kinds.get(job.kind, 0) >= lim:
                blocking.append("kind_limit")
        if led.mem + rsv.mem + c.mem_gb > cfg.slice_mem_gb + EPS:
            blocking.append("mem")
        budget = cfg.write_budget_mbps
        if c.write_mbps > 0 and measured_write is not None and measured_write > cfg.write_pause_ratio * budget:
            blocking.append("write_paused")
        elif led.write + rsv.write + c.write_mbps > budget + EPS:
            blocking.append("write")
        if led.store + rsv.store + c.bytes_persist + c.bytes_derived > cfg.storage_block_ratio * cfg.storage_quota_bytes:
            blocking.append("storage")
        return tuple(blocking), card

    def tick(self, now: float, telemetry: Telemetry | None = None) -> list[Decision]:
        """一次调度循环（每 1 s【墙钟】，或在作业结束、提交、取消时触发）：按序逐个判定，被准入的作业立即记账并移入运行表。
        非评测作业只因评测预留而不能准入时 `blocked_by = ("eval_priority",)`。"""
        tel = telemetry or Telemetry()
        led = self._ledger(now, tel)
        eval_active = self.eval_active
        ordered = self.ordered_queue()
        pending = [j for j in ordered if j.is_eval]
        rsv: _Reserve | None = None                   # 只在账本或待评测集合变化后重算
        decisions: list[Decision] = []
        admitted: set[str] = set()
        for job in ordered:
            if job.is_eval or not pending:
                blocking, card = self._fits(led, job, now, _NO_RESERVE, eval_active, tel.write_mbps_1min)
            else:
                if rsv is None:
                    rsv = self._reserve_for_eval(led, pending, now)
                blocking, card = self._fits(led, job, now, rsv, eval_active, tel.write_mbps_1min)
                if blocking and not self._fits(led, job, now, _NO_RESERVE, eval_active, tel.write_mbps_1min)[0]:
                    blocking = ("eval_priority",)
            if blocking:
                reasons = sorted({int(Reason.STORAGE_QUOTA_EXCEEDED) for b in blocking if b == "storage"}
                                 | {int(Reason.WRITE_BUDGET_EXCEEDED) for b in blocking if b in ("write", "write_paused")})
                decisions.append(Decision(job.job_id, False, blocked_by=blocking, reasons=tuple(reasons)))
                if self._blocked.get(job.job_id) != blocking:
                    self._blocked[job.job_id] = blocking
                    self.events.extend((now, job.job_id, code, blocking) for code in reasons)
                continue
            gpu = card.slot.index if card is not None else None
            share = 0.0 if gpu is None else (1.0 if job.acceptance else (self.cfg.eval_share if job.is_eval else job.cost.gpu_share))
            run = Running(job, gpu, share, now)
            led.book(run)
            self.running[job.job_id] = run
            admitted.add(job.job_id)
            self._blocked.pop(job.job_id, None)
            rot = self._rr.setdefault(job.priority_class, [])
            if job.tenant in rot:
                rot.remove(job.tenant)
            rot.append(job.tenant)
            if job.is_eval:
                pending = [j for j in pending if j.job_id != job.job_id]
            rsv = None
            cond = "shared_card" if job.is_eval and card is not None and not card.slot.exclusive else None
            decisions.append(Decision(job.job_id, True, gpu, share, condition=cond))
        self.queue = [j for j in self.queue if j.job_id not in admitted]
        self._queued_ids -= admitted
        return decisions

    def status(self, now: float = 0.0) -> dict:
        """健康与指标摘要（M27-FR-158）：每卡份额与正式评测数、`d3.slice` 核数与配额、写带宽估计、预算与来源。"""
        led = self._ledger(now, Telemetry())
        return {"eval_active": self.eval_active, "queued": len(self.queue), "running": len(self.running),
                "gpus": {i: {"exclusive": c.slot.exclusive, "share_used": round(c.used, 6), "eval_runs": c.n_eval,
                             "share_by": {k: round(v, 6) for k, v in sorted(c.share_by.items())}} for i, c in led.cards.items()},
                "cpu": {"eval": led.cpu_eval, "other": led.cpu_other, "quota": slice_quota(self.eval_active, self.cfg)},
                "write": {"est_mbps": round(led.write, 3), "budget_mbps": self.cfg.write_budget_mbps,
                          "budget_source": self.cfg.write_budget_source}}


# ── cgroup 与容器参数 ────────────────────────────────────────────────────────────────────────────────────────────────

def slice_quota(eval_active: bool, config: SchedConfig | None = None) -> dict[str, float]:
    """`d3.slice` 的核数配额（M27 §8.2）：`d3-eval.slice` ≤ 30 核；评测队列非空时其他批作业合计 ≤ 8 核，否则受 `d3.slice` 总量约束。"""
    cfg = config or SchedConfig()
    return {"d3.slice": cfg.slice_cpu, "d3-eval.slice": cfg.eval_cpu,
            "other_jobs": cfg.other_cpu_when_eval if eval_active else cfg.slice_cpu}


def cgroup_parent(job: Job) -> str:
    """作业 → `--cgroup-parent`（M27 §8.2、FR-025）。"""
    if job.is_eval:
        return "d3.slice/d3-eval.slice"
    if job.kind in _LIVO_KINDS:
        return "d3.slice/d3-livo.slice"
    if job.kind in _JOBS_SLICE_KINDS:
        return "d3.slice/d3-jobs.slice"
    return "d3.slice/d3-run.slice"


def resource_args(job: Job, decision: Decision) -> list[str]:
    """准入结论 → docker 资源参数：`--cpus`、`--memory`、`--cgroup-parent`；GPU 作业只绑一张卡（`--gpus device=N`）。"""
    if not decision.admitted:
        raise ValueError(f"{job.job_id} is not admitted")
    c = job.cost
    args = ["--cpus", f"{c.cpu_cores:g}", "--memory", f"{max(1, math.ceil(c.mem_gb))}g", "--cgroup-parent", cgroup_parent(job)]
    if decision.gpu is not None:
        args += ["--gpus", f"device={decision.gpu}", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"]
    return args
