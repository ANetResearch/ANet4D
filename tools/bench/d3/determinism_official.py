#!/usr/bin/env python3
"""tools/bench/d3/determinism_official.py：正式成绩配置两次运行逐字节对拍（D3-AC-17；P05；ADR-162、ADR-177）。

正式成绩配置 = Realistic 档（ADR-226 起为统一 L2 引擎 `awr.phys.l2`；`--physics gz` 时为 gz-sim 锁步 `SupervisedGzBackend`）+ M25 GPU 帧（EGL 光栅相机、rgb1080-rs、
倒装 MID-360、IMU、GNSS）+ L2 + 参考基线策略（含参考检测器），Run 规格取 `configs/bench.d3/crop-handoff-001.l2-3uav.run_spec.json`。
同一 Run 规格、同一 `run_id`、同一种子、同一镜像与同一张卡上按下列变体各跑一次（每个变体一个独立子进程，各自一个 gz 实例与 GL 上下文）：

    a     基线：不经 M25 会话（正式运行器当前的交付路径）
    b     与 a 完全相同的第二次运行
    m25   经 M25 Benchmark 会话交付（`Scheduler(threaded=False)`，缺省在途上限 = 2 个 tick 的到期帧数），无注入
    bp1   M25 背压：在途上限 1、每帧人为延迟 50 ms（641 → 按原键重提交，仿真时钟阻塞）
    bp2   bp1 + 写入器背压（`AWR_REC_INJECT_DELAY_MS=10`、`AWR_REC_QUEUE_MSGS=8`：写线程每条消息额外等待，队列满阻塞 Env 主循环；
          正式配置每 tick 约 18 条观测层与 54 条真值层消息，每条 10 ms 时每 tick 的突发均超过队列上限而阻塞，360 s 回合墙钟实测
          5361 s；每条 50 ms 时真值写线程单独即需 2.7 s/tick，回合约 2.7 h 以上）

对拍项（`compare`）：

- `repeat`（a 对 b）：gz 位姿摘要 `pose_sha256` 相同；每架机 Agent 收到的观测流（`agents/<uav>/obs.sha256`，L2 帧字节以 base64 进入
  规范 JSON）与动作流（`acts.jsonl`，动作与回执）逐字节一致；逐帧摘要（`awr.bench.platform_audit.frames_digest`：obs、act、逐 tick 位姿、
  report、net、truth/*）逐项相同；`run_hash` 相同；
- `m25_vs_free`（m25 对 a）、`bp_repeat`（bp1 对 bp2）、`bp_vs_free`（bp1 对 a）：`(uav, sensor, seq)` 集合与逐帧字节（交付帧的规范
  JSON sha256 与 raw.mcap 存档消息 sha256 两种口径）逐一相同，且上述各流逐帧一致；
- 每个运行内部：raw.mcap 存档帧集合与交付帧集合相同、写入器 `dropped = 0`、gz 无重启（`restarts = 0`）；
- 背压确实发生：bp 变体 `max_inflight = 1`、每个 tick 都收到 641 并阻塞（`blocked_ticks = ticks`）、等待墙钟 ≥ 每帧 50 ms × 完成帧数
  的 95 %；bp2 写入器发生阻塞（`raw.blocked > 0`）。

产物（`--out`）：`runs/<变体>/`（运行目录，`frames.digest`、`frames.keys`、`archive.keys`、`p05_run.json`）、`digests/<变体>.digest`、
`logs/<变体>.log`，摘要 `determinism_official.json`（`drone4d.bench.determinism_official.v1`：平台与镜像摘要、GPU 型号与驱动、CPU ISA、
各运行要点与逐项对拍结论、`ok`）。墙钟量（`wall_s`、背压等待）只进入摘要的运行要点，不参与对拍。

用法（dmax 宿主容器 d3-sensorsvc、绑定一张卡、挂载 docker 套接字，gz 为兄弟容器）：
    python tools/bench/d3/determinism_official.py run --spec configs/bench.d3/crop-handoff-001.l2-3uav.run_spec.json \\
        --worlds-root worlds --events-root d3data/events --out var/p05/formal --policy track_share --seed 1 \\
        [--duration-s 360] [--runs a,b,m25,bp1,bp2] [--jobs 5] [--gz-launcher docker] [--gz-image d3-gz:dev] \\
        [--image gz=sha256:… --image sensorsvc=sha256:…]
    python tools/bench/d3/determinism_official.py compare --out var/p05/formal      # 由已有运行目录重算摘要
退出码 0 全部一致、1 有不一致或运行失败、2 参数错误。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "python", ROOT / "tools/sdk/anet4d/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = "drone4d.bench.determinism_official.v1"
RUN_SCHEMA = "drone4d.bench.determinism_run.v1"
SUMMARY_NAME = "determinism_official.json"
RUN_INFO = "p05_run.json"
DELIVERED_KEYS = "frames.keys"
ARCHIVE_KEYS = "archive.keys"
WAIT_FRAC = 0.95                                     # 背压等待墙钟下限：每帧注入延迟 × 完成帧数的 95 %
SENSOR_TOPIC = re.compile(r"^/uav/([^/]+)/(?:camera/([A-Za-z0-9_]+)/image|lidar/([A-Za-z0-9_]+)|imu/([A-Za-z0-9_]+)|gnss/([A-Za-z0-9_]+))$")


@dataclass(frozen=True)
class Variant:
    """一次运行的交付方式与背压注入（只改变墙钟，不改变 Run 规格）。"""

    name: str
    m25: bool = False                  # 经 M25 Benchmark 会话交付 L2 帧
    delay_ms: float = 0.0              # M25 每帧人为延迟
    inflight_cap: int | None = None    # M25 会话在途上限（None：2 个 tick 的到期帧数）
    rec_delay_ms: float = 0.0          # 写入器每条消息注入延迟（AWR_REC_INJECT_DELAY_MS）
    rec_queue_msgs: int | None = None  # 写入器在途消息上限（AWR_REC_QUEUE_MSGS）

    @property
    def backpressure(self) -> bool:
        return self.m25 and self.delay_ms > 0


VARIANTS = {v.name: v for v in (
    Variant("a"),
    Variant("b"),
    Variant("m25", m25=True),
    Variant("bp1", m25=True, delay_ms=50.0, inflight_cap=1),
    Variant("bp2", m25=True, delay_ms=50.0, inflight_cap=1, rec_delay_ms=10.0, rec_queue_msgs=8),
)}
DEFAULT_RUNS = ("a", "b", "m25", "bp1", "bp2")
# 对拍对：(名称, 参照, 对象, 条款)；只比较两侧都存在的对
PAIRS = (
    ("repeat", "a", "b", "two runs of the formal configuration: gz poses, per-agent observation and action streams"),
    ("m25_vs_free", "a", "m25", "M25 session delivery without injection matches direct delivery"),
    ("bp_repeat", "bp1", "bp2", "two runs under M25 backpressure (in-flight cap 1, 50 ms): (uav, sensor, seq) set and frame bytes"),
    ("bp_vs_free", "a", "bp1", "backpressure run matches the run without backpressure"),
)


# ------------------------------------------------------------------ 单次运行
@contextlib.contextmanager
def _rec_env(v: Variant):
    """写入器背压注入经环境变量交给 `RecorderOptions.from_env`（EnvRecorder 构造时读取）。"""
    keys = {"AWR_REC_INJECT_DELAY_MS": f"{v.rec_delay_ms:g}" if v.rec_delay_ms else None,
            "AWR_REC_QUEUE_MSGS": str(v.rec_queue_msgs) if v.rec_queue_msgs else None}
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k, x in keys.items():
            if x is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = x
        yield
    finally:
        for k, x in old.items():
            if x is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = x


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _collect(delivered: dict, obs: dict) -> None:
    """交付给 Agent 的每帧：`(uav, sensor, seq)` → 帧对象规范 JSON 的 sha256（字节以 base64 进入，与 obs.sha256 同一编码）。"""
    from awr.agentsdk.env.recorder import canonical_json

    for u, m in obs.items():
        for sid, fs in (m["obs"].get("sensors") or {}).items():
            if sid == "detections":
                continue
            for f in fs:
                key = (u, sid, int(f["seq"]))
                if key in delivered:
                    raise RuntimeError(f"frame {key} delivered twice")
                delivered[key] = _sha(canonical_json(f))


def archive_frames(run_dir: Path) -> dict[tuple[str, str, int], str]:
    """raw.mcap 中每帧存档消息：`(uav, sensor, seq)` → 消息字节 sha256（相机 CompressedImage、LiDAR lidar_frame.v2、IMU 与 GNSS msgpack）。"""
    from mcap.reader import make_reader

    out: dict[tuple[str, str, int], str] = {}
    p = Path(run_dir) / "raw.mcap"
    if not p.is_file():
        return out
    with p.open("rb") as f:
        for _s, c, m in make_reader(f).iter_messages(log_time_order=False):
            g = SENSOR_TOPIC.match(c.topic)
            if g is None:
                continue
            key = (g.group(1), next(x for x in g.groups()[1:] if x), int(m.sequence))
            if key in out:
                raise RuntimeError(f"{p}: frame {key} archived twice")
            out[key] = _sha(bytes(m.data))
    return out


def write_keys(path: Path, frames: dict[tuple[str, str, int], str]) -> str:
    """`<uav> <sensor> <seq> <sha256>` 逐行（按键排序）；→ 文件 sha256。"""
    body = "".join(f"{u} {s} {q} {h}\n" for (u, s, q), h in sorted(frames.items())).encode()
    Path(path).write_bytes(body)
    return _sha(body)


def read_keys(path: Path) -> dict[tuple[str, str, int], str]:
    out = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln:
            u, s, q, h = ln.split(" ")
            out[(u, s, int(q))] = h
    return out


def _file_sha(p: Path) -> str | None:
    return _sha(p.read_bytes()) if p.is_file() else None


def run_one(spec, policy: str, v: Variant, *, l2=None, agent_factory=None) -> dict:
    """按变体运行一次 Env（参考策略在进程内，与 `anet4d.runner.run_episode` 同一循环），写运行目录内的对拍产物；→ 运行要点。"""
    from awr.agentsdk.env.core import BenchEnv
    from awr.bench.platform_audit import frames_digest, write_digest

    if agent_factory is None:
        from anet4d.baselines import BASELINES

        def agent_factory(_u, _p=policy):
            return BASELINES[_p]()

    run_dir = Path(spec.out_dir)
    sched = None
    if v.m25:
        from awr.sensorsvc.scheduler import Scheduler

        sched = Scheduler(delay_s=v.delay_ms / 1000.0, threaded=False)      # 进程内 GL 渲染：在调用方线程执行（上下文线程归属）
    delivered: dict = {}
    t0 = time.time()
    try:
        with _rec_env(v):
            env = BenchEnv(spec, l2=l2, sensorsvc=sched, inflight_cap=v.inflight_cap)
            obs, infos = env.reset()
            _collect(delivered, obs)
            cam = getattr(env.l2, "cam", None)
            sensors = {"renderer": type(cam).__name__ if cam is not None else None, "gl": getattr(cam, "info", None)}
            pool = {u: agent_factory(u) for u in env.possible_agents}
            for u, ag in pool.items():
                ag.reset(infos[u]["config"])
            try:
                while env.agents:
                    obs, *_ = env.step({u: pool[u].step(o["obs"], o["inbox"]) for u, o in obs.items() if u in env.agents})
                    _collect(delivered, obs)
            finally:
                sensors["jpeg_encoder"] = getattr(env.l2, "encoder", None)
                man = env.close()
    finally:
        if sched is not None:
            sched.close()
    wall = time.time() - t0
    fd = frames_digest(run_dir)
    write_digest(fd, run_dir / "frames.digest")
    d_sha = write_keys(run_dir / DELIVERED_KEYS, delivered)
    arch = archive_frames(run_dir)
    a_sha = write_keys(run_dir / ARCHIVE_KEYS, arch)
    per_sensor: dict[str, int] = {}
    for _u, s, _q in delivered:
        per_sensor[s] = per_sensor.get(s, 0) + 1
    rec = man.get("recorder") or {}
    info = {"schema": RUN_SCHEMA, "variant": asdict(v), "run_id": man.get("run_id"), "policy": policy, "physics": man.get("physics"),
            "tier": man.get("tier"), "obs_level": man.get("obs_level"), "duration_s": man.get("duration_s"), "n_ticks": man.get("n_ticks"),
            "pose_sha256": man.get("pose_sha256"), "run_hash": man.get("run_hash"), "versions": man.get("versions"),
            "digest_sha256": fd.sha256(), "delivered_keys_sha256": d_sha, "archive_keys_sha256": a_sha,
            "frames": {"delivered": len(delivered), "archived": len(arch), "per_sensor": dict(sorted(per_sensor.items()))},
            "streams": {u: {"obs_sha256": _file_sha(run_dir / "agents" / u / "obs.sha256"),
                            "acts_sha256": _file_sha(run_dir / "agents" / u / "acts.jsonl"),
                            "n_obs": len(fd.streams.get(f"obs/{u}", [])), "n_acts": len(fd.streams.get(f"act/{u}", []))}
                        for u in sorted(man.get("agents") or {})},
            "timeouts": {u: a.get("timeouts", 0) for u, a in sorted((man.get("agents") or {}).items())},
            "gz": {k: (man.get("tier_info") or {}).get(k) for k in ("ticks", "restarts", "replayed_ticks", "reuse_policy", "gz_rtf_mean")},
            "sensors": sensors, "sensorsvc": man.get("sensorsvc"),
            "recorder": {"dropped": rec.get("dropped"), "writer_block_s": rec.get("writer_block_s"), "options": rec.get("options"),
                         "raw_blocked": (rec.get("raw") or {}).get("blocked"), "truth_blocked": (rec.get("truth") or {}).get("blocked"),
                         "raw_max_depth": (rec.get("raw") or {}).get("max_depth")} if rec else None,
            "wall_s": round(wall, 2), "env_wall_s": man.get("wall_s")}
    (run_dir / RUN_INFO).write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return info


# ------------------------------------------------------------------ 对拍
def _frame_diff(a: dict, b: dict, limit: int = 5) -> dict:
    ka, kb = set(a), set(b)
    bad = sorted(k for k in ka & kb if a[k] != b[k])
    return {"equal": a == b, "n_ref": len(a), "n_got": len(b), "missing": [list(k) for k in sorted(ka - kb)[:limit]],
            "extra": [list(k) for k in sorted(kb - ka)[:limit]], "n_missing": len(ka - kb), "n_extra": len(kb - ka),
            "n_bytes_differ": len(bad), "first_bytes_differ": [list(k) for k in bad[:limit]]}


def compare_runs(ref_dir: Path, got_dir: Path) -> dict:
    """两运行目录逐项对拍：逐帧摘要（obs、act、pose、report、net、truth/*）、`pose_sha256`、`run_hash`、每 Agent 观测与动作流文件、
    `(uav, sensor, seq)` 集合与帧字节（交付与存档两种口径）。"""
    from awr.bench.platform_audit import compare_digests, load_digest

    ra = json.loads((Path(ref_dir) / RUN_INFO).read_text(encoding="utf-8"))
    rb = json.loads((Path(got_dir) / RUN_INFO).read_text(encoding="utf-8"))
    diffs = compare_digests(load_digest(Path(ref_dir) / "frames.digest"), load_digest(Path(got_dir) / "frames.digest"))
    deliv = _frame_diff(read_keys(Path(ref_dir) / DELIVERED_KEYS), read_keys(Path(got_dir) / DELIVERED_KEYS))
    arch = _frame_diff(read_keys(Path(ref_dir) / ARCHIVE_KEYS), read_keys(Path(got_dir) / ARCHIVE_KEYS))
    streams = {u: {"obs": ra["streams"][u]["obs_sha256"] == (rb["streams"].get(u) or {}).get("obs_sha256"),
                   "acts": ra["streams"][u]["acts_sha256"] == (rb["streams"].get(u) or {}).get("acts_sha256")} for u in ra["streams"]}
    pose_ok = bool(ra.get("pose_sha256")) and ra.get("pose_sha256") == rb.get("pose_sha256")
    ok = (not diffs and deliv["equal"] and arch["equal"] and pose_ok and set(ra["streams"]) == set(rb["streams"])
          and all(x["obs"] and x["acts"] for x in streams.values()) and ra.get("run_hash") == rb.get("run_hash")
          and deliv["n_ref"] > 0)
    return {"ok": ok, "pose_sha256": {"equal": pose_ok, "ref": ra.get("pose_sha256"), "got": rb.get("pose_sha256")},
            "run_hash_equal": ra.get("run_hash") == rb.get("run_hash"), "digest_sha256_equal": ra["digest_sha256"] == rb["digest_sha256"],
            "digest_diffs": diffs[:20], "n_digest_diffs": len(diffs), "streams": streams, "delivered_frames": deliv, "archived_frames": arch}


def run_checks(info: dict) -> dict:
    """单个运行的内部核对：存档帧 = 交付帧、写入器无丢失、gz 无重启、Agent 无超时；背压变体另核背压确实发生。"""
    v = Variant(**info["variant"])
    c = {"archive_equals_delivered": info["archive_keys_ok"], "recorder_dropped_0": (info.get("recorder") or {}).get("dropped") == 0,
         "gz_restarts_0": (info.get("gz") or {}).get("restarts") in (0, None), "agent_timeouts_0": not any(info.get("timeouts", {}).values()),
         "frames_nonempty": info["frames"]["delivered"] > 0}
    sv = info.get("sensorsvc")
    if v.m25:
        # 会话交付：到期键全部完成（IMU 窗口不足的键完成但不交付），不丢帧、不降频
        c["m25_session"] = bool(sv) and sv.get("mode") == "benchmark" and sv.get("accepted") == sv.get("completed") \
            and int(sv.get("completed", 0)) >= info["frames"]["delivered"]
    if v.backpressure:
        c["m25_inflight_cap"] = bool(sv) and sv.get("inflight_cap") == v.inflight_cap and sv.get("max_inflight") == v.inflight_cap
        c["m25_blocked_every_tick"] = bool(sv) and sv.get("rejected_641", 0) > 0 and sv.get("blocked_ticks") == sv.get("ticks")
        c["m25_delay_applied"] = bool(sv) and abs(float(sv.get("delay_s", 0)) - v.delay_ms / 1000.0) < 1e-9 \
            and float(sv.get("wait_s", 0)) >= WAIT_FRAC * v.delay_ms / 1000.0 * int(sv.get("completed", 0))
    if v.rec_delay_ms:
        r = info.get("recorder") or {}
        c["recorder_blocked"] = (r.get("raw_blocked") or 0) > 0 and float(r.get("writer_block_s") or 0) > 0 \
            and float((r.get("options") or {}).get("inject_delay_ms", 0)) == v.rec_delay_ms
    return c


def _platform() -> dict:
    """GPU 型号与驱动、CPU 型号与 ISA 等级、主机与内核（运行清单的复现等级条件，ADR-177）。"""
    out = {"host": platform.node(), "machine": platform.machine(), "kernel": platform.release(), "python": platform.python_version()}
    try:
        txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^model name\s*:\s*(.+)$", txt, re.M)
        flags = set((re.search(r"^flags\s*:\s*(.+)$", txt, re.M) or [None, ""])[1].split())
        lvl = "x86-64"
        if {"sse4_2", "popcnt", "ssse3"} <= flags:
            lvl = "x86-64-v2"
            if {"avx2", "fma", "bmi2", "movbe"} <= flags:
                lvl = "x86-64-v3"
                if {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"} <= flags:
                    lvl = "x86-64-v4"
        out["cpu"] = {"model": m.group(1).strip() if m else None, "isa_level": lvl, "n_cpu": os.cpu_count()}
    except OSError:
        pass
    if shutil.which("nvidia-smi"):
        with contextlib.suppress(Exception):
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,uuid,memory.used,utilization.gpu",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=True)
            out["gpu"] = [dict(zip(("index", "name", "driver", "uuid", "memory_used", "utilization"), (x.strip() for x in ln.split(",")), strict=False))
                          for ln in r.stdout.splitlines() if ln.strip()]
    out["threads"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS") if os.environ.get(k)}
    return out


def compare(out: Path, names: tuple[str, ...] | None = None, *, pairs=PAIRS, job: dict | None = None) -> dict:
    """由 `<out>/runs/<变体>/` 重算摘要并写 `<out>/determinism_official.json`；→ 摘要。"""
    out = Path(out)
    runs_dir = out / "runs"
    names = tuple(names or sorted(p.name for p in runs_dir.iterdir() if (p / RUN_INFO).is_file()))
    infos, failed = {}, {}
    for n in names:
        p = runs_dir / n / RUN_INFO
        if not p.is_file():
            failed[n] = (runs_dir / n / "error.txt").read_text(encoding="utf-8")[-2000:] if (runs_dir / n / "error.txt").is_file() else "missing"
            continue
        info = json.loads(p.read_text(encoding="utf-8"))
        # 存档与交付的 (uav, sensor, seq) 集合相同（两侧摘要口径不同：交付为帧对象规范 JSON，存档为 MCAP 消息字节）
        info["archive_keys_ok"] = set(read_keys(runs_dir / n / DELIVERED_KEYS)) == set(read_keys(runs_dir / n / ARCHIVE_KEYS))
        info["checks"] = run_checks(info)
        infos[n] = info
        (out / "digests").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runs_dir / n / "frames.digest", out / "digests" / f"{n}.digest")
    cmp = {}
    for name, ref, got, clause in pairs:
        if ref in infos and got in infos:
            cmp[name] = {"ref": ref, "got": got, "clause": clause, **compare_runs(runs_dir / ref, runs_dir / got)}
    first = next(iter(infos.values()), {})
    ok = (not failed and bool(cmp) and all(c["ok"] for c in cmp.values()) and all(all(i["checks"].values()) for i in infos.values()))
    doc = {"schema": SCHEMA, "acceptance": "D3-AC-17", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ok": ok,
           "config": {**(job or {}), "run_id": first.get("run_id"), "physics": first.get("physics"), "tier": first.get("tier"),
                      "obs_level": first.get("obs_level"), "duration_s": first.get("duration_s"), "n_ticks": first.get("n_ticks"),
                      "versions": first.get("versions"), "sensors": first.get("sensors")},
           "platform": _platform(), "variants": {n: asdict(VARIANTS[n]) if n in VARIANTS else None for n in names},
           "runs": {n: {k: i.get(k) for k in ("pose_sha256", "run_hash", "digest_sha256", "delivered_keys_sha256", "archive_keys_sha256", "frames",
                                              "streams", "gz", "sensors", "sensorsvc", "recorder", "timeouts", "wall_s", "env_wall_s", "checks")}
                    for n, i in infos.items()},
           "failed": failed, "comparisons": cmp}
    if (out / "job.json").is_file() and job is None:                 # 运行参数在前，运行实测的身份字段优先
        doc["config"] = {**json.loads((out / "job.json").read_text(encoding="utf-8")), **{k: v for k, v in doc["config"].items() if v is not None}}
    tmp = out / (SUMMARY_NAME + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, out / SUMMARY_NAME)
    return doc


# ------------------------------------------------------------------ 命令行
def _spec_for(job: dict, name: str):
    from awr.agentsdk.env.spec import EnvSpec
    from awr.bench.eval.evaluator import dev_seed_commit

    rs = json.loads(Path(job["spec"]).read_text(encoding="utf-8"))
    rs = {**rs, "policy": {"kind": "reference", "name": job["policy"]}}
    if job.get("duration_s"):
        rs["duration_s"] = float(job["duration_s"])
    out = Path(job["out"]) / "runs" / name
    gz = {k: v for k, v in (job.get("gz") or {}).items() if v is not None}
    if (job.get("physics") or rs.get("physics", "realistic")) == "gz":                     # gz lockstep only (ADR-226)
        gz = {**gz, "name": f"{job.get('gz_prefix', 'd3-gz-p05')}-{name}", "run_dir": str(out / "phys"),
              "export_root": gz.get("export_root") or str(Path(job["out"]) / "export")}
    run_id = job.get("run_id") or f"p05-{job['policy']}-s{job['seed']}"
    seed = int(job["seed"])
    return EnvSpec.from_run_spec(rs, run_id=run_id, seed=seed, worlds_root=Path(job["worlds_root"]), events_root=Path(job["events_root"]),
                                 out_dir=out, physics=job.get("physics"), gz=gz, images=job.get("images") or None,
                                 seed_commit=dev_seed_commit(run_id, seed) if rs.get("ranked") else None)


def _cmd_one(a) -> int:
    job = json.loads(Path(a.job).read_text(encoding="utf-8"))
    name = a.name
    out = Path(job["out"]) / "runs" / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    try:
        info = run_one(_spec_for(job, name), job["policy"], VARIANTS[name])
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return 1
    print(json.dumps({k: info[k] for k in ("pose_sha256", "digest_sha256", "delivered_keys_sha256", "wall_s")}))
    return 0


def _cmd_run(a) -> int:
    names = tuple(x for x in a.runs.split(",") if x)
    bad = [n for n in names if n not in VARIANTS]
    if bad:
        print(f"unknown run variants {bad}; choose from {sorted(VARIANTS)}", file=sys.stderr)
        return 2
    out = Path(a.out).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    images = dict(x.split("=", 1) for x in a.image)
    job = {"spec": str(Path(a.spec).resolve()), "policy": a.policy, "seed": a.seed, "duration_s": a.duration_s, "out": str(out),
           "worlds_root": str(Path(a.worlds_root).resolve()), "events_root": str(Path(a.events_root).resolve()), "physics": a.physics,
           "gz": {"launcher": a.gz_launcher, "image": a.gz_image, "export_root": a.gz_export_root}, "images": images,
           "gz_prefix": a.gz_prefix, "run_id": a.run_id, "runs": list(names)}
    (out / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # 世界级导出与 MeshWorldQuery 缓存先行一次，避免并行的运行同时写同一目录
    world_dir = Path(job["worlds_root"]) / json.loads(Path(a.spec).read_text(encoding="utf-8"))["world"]["id"]
    if (job["physics"] or "realistic") == "gz":                                                # the gz world export is only needed by the gz tier
        from awr.world.export.gz import export_world

        export_world(world_dir, out_root=Path(job["gz"]["export_root"] or out / "export"))
    with contextlib.suppress(Exception):
        from awr.world.mesh.query import open_mesh_world_query

        open_mesh_world_query(world_dir, world_dir.resolve().parent.parent / "cache" / "worldquery")

    def one(n: str) -> tuple[str, int, float]:
        t0 = time.time()
        with (out / "logs" / f"{n}.log").open("w", encoding="utf-8") as log:
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "one", "--job", str(out / "job.json"), "--name", n],
                                stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        print(f"[p05] {n}: rc {rc}  {time.time() - t0:.0f} s", flush=True)
        return n, rc, time.time() - t0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        res = list(ex.map(one, names))
    doc = compare(out, names)
    doc["wall_total_s"] = round(time.time() - t0, 1)
    doc["exit_codes"] = {n: rc for n, rc, _ in res}
    (out / SUMMARY_NAME).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _print(doc)
    return 0 if doc["ok"] else 1


def _print(doc: dict) -> None:
    for n, c in doc["comparisons"].items():
        print(f"[p05] {n} ({c['ref']} vs {c['got']}): {'ok' if c['ok'] else 'MISMATCH'}  pose {c['pose_sha256']['equal']}  "
              f"frames {c['delivered_frames']['n_ref']}/{c['delivered_frames']['n_got']} equal {c['delivered_frames']['equal']}  "
              f"digest diffs {c['n_digest_diffs']}")
    for n, r in doc["runs"].items():
        bad = [k for k, x in r["checks"].items() if not x]
        print(f"[p05] run {n}: wall {r['wall_s']} s  checks {bad if bad else 'ok'}")
    if doc["failed"]:
        print(f"[p05] failed runs: {sorted(doc['failed'])}")
    print(f"[p05] verdict: {'ok' if doc['ok'] else 'FAILED'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the formal-configuration variants and compare them")
    r.add_argument("--spec", default=str(ROOT / "configs/bench.d3/crop-handoff-001.l2-3uav.run_spec.json"))
    r.add_argument("--policy", default="track_share")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--duration-s", type=float, default=None, help="override the run spec duration (default: the spec, 360 s)")
    r.add_argument("--worlds-root", default=str(ROOT / "worlds"))
    r.add_argument("--events-root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    r.add_argument("--jobs", type=int, default=len(DEFAULT_RUNS))
    r.add_argument("--physics", choices=("realistic", "fast", "kinematic"), default=None)
    r.add_argument("--gz-launcher", choices=("docker", "local"), default="docker")
    r.add_argument("--gz-image", default="d3-gz:dev")
    r.add_argument("--gz-export-root", default=None)
    r.add_argument("--gz-prefix", default="d3-gz-p05", help="gz container name prefix (one container per variant)")
    r.add_argument("--run-id", default=None, help="shared run_id of every variant (default p05-<policy>-s<seed>)")
    r.add_argument("--image", action="append", default=[], help="platform image digest KEY=DIGEST (gz, sensorsvc, env, agent)")
    o = sub.add_parser("one", help="internal: run one variant")
    o.add_argument("--job", required=True)
    o.add_argument("--name", required=True)
    c = sub.add_parser("compare", help="recompute the summary from existing run directories")
    c.add_argument("--out", required=True)
    c.add_argument("--runs", default="")
    a = ap.parse_args(argv)
    if a.cmd == "one":
        return _cmd_one(a)
    if a.cmd == "run":
        return _cmd_run(a)
    doc = compare(Path(a.out), tuple(x for x in a.runs.split(",") if x) or None)
    _print(doc)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
