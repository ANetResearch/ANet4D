#!/usr/bin/env python3
"""tools/bench/d3/write_load.py：1 h 生产负载下的 D3 持续写速率与 MCAP 写入器阻塞测量（D3-AC-22；B23；AWR-05 §4.7、§5.6.4、ADR-162、
ADR-163、ADR-194 (7)、ADR-195；M27-NFR-003、M27-AC-006、M27-AC-011）。

验收原文（D3-AC-22 末两项）："1 h 生产负载下 D3 持续写速率 p95 ≤ 写带宽预算，MCAP 写入器阻塞时间占墙钟 ≤ 1%"；写带宽预算
`B_write = min(300 MB/s, P-1 实测 /data 持续写带宽 × 50%)`（§5.6.4；P-1 实测见 D00 `disk.json`）。

**生产负载**：生产窗口内一张独占卡上由 bench-scheduler（`awr.bench.scheduler`）准入的正式评测队列。`load` 子命令持续向调度器提交
排名运行（`ranked = true`，`eval` 类，缺省 Run 规格 `configs/bench.d3/crop-handoff-001.l2-3uav.run_spec.json`：Realistic 档 gz
`physics` 导出档锁步 + M25 GPU 帧（rgb1080-rs、倒装 MID-360、IMU、GNSS）+ L2 + 参考基线策略，360 s），每 1 s 以实测写速率
（`sample` 写出的 cgroup `io.stat` 1 min 均值）驱动一次调度循环：单卡并发正式 Run ≤ 3、各 1/3 卡份额、写带宽按估计写速率准入
（超出记 686）。被准入的 Run 各在独立子进程（`worker`）中按正式运行器的同一循环执行（`determinism_official.run_one` 的变体 a：
直接交付、无注入），`EnvRecorder` 经 `RunWriter` 写 `raw.mcap` 与真值 MCAP（有界队列 256 条 / 256 MiB，满则阻塞 Env 主循环）；
Run 结束后按 `ranked_eval` 保留规则精简（`retention.compact_ranked`：只留逐帧摘要与 `truth_min`，ADR-163），与生产一致地控制
`/data` 占用。测量窗口自首个 Run 准入起 `--duration-s`（缺省 3600 s）；窗口结束后不再准入，运行中的 Run 跑完（尾段不进入写速率
窗口，其写入器统计照常计入阻塞判定）。

**写速率**（`sample`，宿主机，只用标准库）：每 `--period-s`（缺省 1 s）采样一次，记录累计量：
- `load`：本负载容器（名称前缀 `--prefix`，宿主容器与 gz 兄弟容器）cgroup v2 `io.stat` 中 `/data` 所在磁盘的 `wbytes`
  （块层实际写入，回写按脏页归属记入容器 cgroup）——判定口径；
- `proc`：同一批容器内各线程 `/proc/<pid>/task/<tid>/io` 的 `write_bytes` 增量累计（进程层，脏页产生时记账）——交叉核对。按线程
  而不按进程读取：进程级计数含已回收子进程的累计（`wait` 时并入父进程），会把每个结束的 Run 重复计一次；采样器只累计两次采样
  都存在的线程的增量；
- `d3`：宿主机上全部 `d3-*` 容器的 `io.stat`（含其他 D3 任务，只作背景）；
- `dev`：`/proc/diskstats` 中 `/data` 分区的写扇区（含共享盘上其他租户，只作背景）。
相邻两次采样的增量 / 间隔得到 1 s 速率；`analyze` 按 1 s、10 s 不重叠窗口与 60 s 滑动窗口（步长 10 s）聚合，**判定取 60 s
滑动窗口的 p95**：与调度器的实测口径（`io.stat` 1 min 均值，M27 §6.7.3、M27-AC-011 的"实测写速率 p95 ≤ B_write"）和预算来源
（fio 60 s 持续写）同一口径。1 s 与 10 s 只作记录：缓冲写经页缓存由内核回写线程每 5 s 成批落盘（`vm.dirty_writeback_centisecs
= 500`），短窗口反映回写批次而非持续写速率（dmax 实测：合成负载均值 42 MB/s 时 1 s p50 为 0、p95 为 229 MB/s；预算量级合成负载
均值 260 MB/s 时 10 s p95 为 419 MB/s、60 s p95 为 288 MB/s）。

**MCAP 阻塞**：每个 Run 的清单 `recorder.writer_block_frac`（`raw` 与真值两个写入器的阻塞时长之和 / 写入器墙钟）≤ 1%，同时
给出全部 Run 的合计占比（Σ 阻塞 / Σ 墙钟）与最大单次阻塞；`dropped = 0`（无损，ADR-162）。

**独占口径**（ADR-194 (7)、ADR-195）：写带宽属性能类条款，只计独占卡数字。`load` 应由 `p1_measure.py guard --gpu N --name
<宿主容器名>` 包装（排他锁、loadavg、窗口前与窗口内 `nvidia-smi pmon` 1 s 采样）；`analyze --guard` 读其记录给出 A 级（计入）、
B 级（外部进程只驻留显存，"独占（驻留豁免）"，`pending_user_confirmation`）或不计入。

产物（`--out`）：`load_job.json`（Run 规格与运行参数）、`load.json`（`drone4d.bench.write_load_run.v1`：调度配置、预算与来源、逐 Run
准入与结束时刻、估计成本、调度事件与每分钟状态）、`runs/<名>/load_run.json`（`drone4d.bench.write_load_one.v1`：墙钟、实时因子、
写入器统计、精简前字节与平均写速率）、`io_samples.jsonl`（`drone4d.bench.write_load_sample.v1`，逐行累计量）、`logs/`；
`analyze` 写 `write_load.json`（`drone4d.bench.p1_measure_part.v1`，part = `write_load`，与 `p1_measure.py` 各部分同一格式，供 D01
汇总；判定项 `counted = "by_guard"` 按 guard 级别改写）。

用法（dmax；宿主容器与挂载约定同 P05 `run_p05.sh`，`$D=/data/anet-drone4d-d3/d3dev`，源码快照在 `$D/b23/src`）：
    python3 tools/bench/d3/write_load.py sample --out var/b23/load --prefix d3-B23- --data-path /data/anet-drone4d-d3 &   # 宿主机
    python3 tools/bench/d3/p1_measure.py guard --gpu 1 --name d3-B23-load --parts write_load --out var/b23/load/guard-gpu1.json \\
        --lock $D/.perf.lock --pre-s 600 -- docker run --name d3-B23-load --gpus device=1 ... d3-sensorsvc:dev \\
        python3 tools/bench/d3/write_load.py load --gpu 1 --out var/b23/load --events-root d3data/events --gz-prefix d3-B23-gz \\
            --measured-disk-mbps 892.6 --duration-s 3600
    python3 tools/bench/d3/write_load.py analyze --out var/b23/load --guard var/b23/load/guard-gpu1.json
本机冒烟（无 GPU）：`load --worker synthetic --synthetic-mbps 20 --run-duration-s 5 --duration-s 12`，写入器为同一 `RunWriter`。
退出码：0 判定全部通过、1 有判定项不通过或 Run 失败、2 参数错误、14 前置条件不满足（数据不足、路径不允许）。
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "python", ROOT / "tools/sdk/anet4d/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _sibling(name: str):
    """同目录工具模块（`p1_measure` 的阈值、分位数与判定项格式，`determinism_official` 的正式配置 Run 规格）；模块级只用标准库。"""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pm = _sibling("p1_measure")

__all__ = ["BLOCK_FRAC_MAX", "Sampler", "analyze", "block_stats", "bucket_rates", "concurrency", "coverage", "disk_of", "guard_exclusivity",
           "intervals", "main", "measured_1min", "parse_io_stat", "parse_proc_io", "rate_stats", "recorder_brief", "synthetic_run"]

PART = "write_load"
RUN_SCHEMA = "drone4d.bench.write_load_run.v1"
ONE_SCHEMA = "drone4d.bench.write_load_one.v1"
SAMPLE_SCHEMA = "drone4d.bench.write_load_sample.v1"
LOAD_JOB, LOAD_DOC, RUN_INFO, SAMPLES, STOP = "load_job.json", "load.json", "load_run.json", "io_samples.jsonl", "STOP"
EXIT_OK, EXIT_FAIL, EXIT_ARGS, EXIT_ENV = 0, 1, 2, 14
MB = 1_000_000

# ---- 阈值（逐条对照原文）
WINDOW_S = 3600.0                          # D3-AC-22：1 h 生产负载
WRITE_P95_Q = 0.95                         # D3-AC-22：持续写速率 p95 ≤ 写带宽预算
BLOCK_FRAC_MAX = 0.01                      # D3-AC-22、M27-NFR-003、M27-AC-006：MCAP 写入器阻塞时间 ≤ 墙钟 1%
RATE_WINDOWS_S = ((1, 1), (10, 10), (60, 10))   # (窗口, 步长) 秒；1 s 只记录（回写批次粒度）
JUDGED_WINDOWS = ("60s",)                 # 判定：60 s 滑动窗口 p95 ≤ 预算（与调度器 io.stat 1 min 均值同口径）
WINDOW_COVERAGE_MIN = 0.98                 # 采样覆盖测量窗口的比例下限（低于此值判数据不足）
DEFAULT_SPEC = "configs/bench.d3/crop-handoff-001.l2-3uav.run_spec.json"


def utc(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _write_json(p: Path, doc: Mapping) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# ================================================================================================================ 采样（宿主机，标准库）
def parse_io_stat(text: str) -> dict[str, dict[str, int]]:
    """cgroup v2 `io.stat` → {"maj:min": {rbytes, wbytes, rios, wios, dbytes, dios}}。"""
    out: dict[str, dict[str, int]] = {}
    for ln in text.splitlines():
        f = ln.split()
        if not f or ":" not in f[0]:
            continue
        kv = {}
        for x in f[1:]:
            k, _, v = x.partition("=")
            with contextlib.suppress(ValueError):
                kv[k] = int(v)
        out[f[0]] = kv
    return out


def parse_proc_io(text: str) -> dict[str, int]:
    """`/proc/<pid>/io` → {rchar, wchar, syscr, syscw, read_bytes, write_bytes, cancelled_write_bytes}。"""
    out = {}
    for ln in text.splitlines():
        k, _, v = ln.partition(":")
        with contextlib.suppress(ValueError):
            out[k.strip()] = int(v)
    return out


def disk_of(path: Path) -> tuple[str | None, str | None]:
    """路径所在分区 → (diskstats 中的分区名, 所属整盘的 "maj:min")；cgroup `io.stat` 按整盘记账。"""
    try:
        st = os.stat(path)
    except OSError:
        return None, None
    mm = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
    link = Path(f"/sys/dev/block/{mm}")
    try:
        part = link.resolve()
    except OSError:
        return None, mm
    disk = part.parent if (part / "partition").exists() else part
    try:
        dev = (disk / "dev").read_text().strip()
    except OSError:
        dev = mm
    return part.name, dev


def cgroup_dir_of(pid: int) -> Path | None:
    """进程所在 cgroup v2 目录（`/proc/<pid>/cgroup` 的 `0::` 行）。"""
    try:
        for ln in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if ln.startswith("0::"):
                return Path("/sys/fs/cgroup") / ln[3:].lstrip("/")
    except OSError:
        return None
    return None


def list_containers(prefix: str) -> dict[str, tuple[str, int]]:
    """`docker ps` 中名称以 `prefix` 开头的运行中容器 → {短 ID: (名称, 主进程 PID)}。"""
    try:
        r = subprocess.run(["docker", "ps", "--no-trunc", "--filter", f"name=^/{prefix}", "--format", "{{.ID}} {{.Names}}"],
                           capture_output=True, text=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return {}
    ids = [ln.split() for ln in r.stdout.splitlines() if ln.strip()]
    if not ids:
        return {}
    try:
        r2 = subprocess.run(["docker", "inspect", "-f", "{{.Id}} {{.State.Pid}}", *[i[0] for i in ids]], capture_output=True, text=True,
                            timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return {}
    pids = {a: int(b) for a, b in (ln.split() for ln in r2.stdout.splitlines() if ln.strip()) if b.isdigit()}
    return {cid[:12]: (name, pids[cid]) for cid, name in ids if pids.get(cid)}


class Sampler:
    """累计量采样器：容器集合每 `refresh_s` 刷新一次，计数每次采样读取。容器退出后其 cgroup 消失，最后一个采样间隔的增量随之丢失
    （gz 兄弟容器写入量很小；宿主容器贯穿全程）。"""

    def __init__(self, data_path: Path, prefix: str, *, d3_prefix: str = "d3-", refresh_s: float = 5.0) -> None:
        self.part, self.disk = disk_of(data_path)
        self.prefix, self.d3_prefix, self.refresh_s = prefix, d3_prefix, refresh_s
        self._cg: dict[str, dict[str, Path]] = {"load": {}, "d3": {}}
        self._names: dict[str, str] = {}
        self._t_refresh = 0.0
        self._thr: dict[str, int] = {}
        self.proc_cum = 0

    def refresh(self, now: float) -> None:
        if now - self._t_refresh < self.refresh_s:
            return
        self._t_refresh = now
        for scope, pre in (("load", self.prefix), ("d3", self.d3_prefix)):
            cur = {}
            for cid, (name, pid) in list_containers(pre).items():
                d = self._cg[scope].get(cid) or cgroup_dir_of(pid)
                if d is not None:
                    cur[cid] = d
                    self._names[cid] = name
            self._cg[scope] = cur

    def _wbytes(self, d: Path) -> int | None:
        try:
            st = parse_io_stat((d / "io.stat").read_text())
        except OSError:
            return None
        return (st.get(self.disk) or {}).get("wbytes", 0) if self.disk else sum(x.get("wbytes", 0) for x in st.values())

    def _threads(self, cid: str, d: Path) -> dict[str, int]:
        """容器内各线程的 `write_bytes`（`/proc/<pid>/task/<tid>/io`，只含本线程，不含已回收子进程）。"""
        out = {}
        try:
            pids = [int(x) for x in (d / "cgroup.procs").read_text().split()]
        except (OSError, ValueError):
            return out
        for pid in pids:
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                continue
            for tid in tids:
                with contextlib.suppress(OSError):
                    out[f"{cid}:{tid}"] = parse_proc_io(Path(f"/proc/{pid}/task/{tid}/io").read_text()).get("write_bytes", 0)
        return out

    def sample(self) -> dict:
        now = time.time()
        self.refresh(now)
        rec: dict[str, Any] = {"t": round(now, 3)}
        for scope in ("load", "d3"):
            rec[scope] = {cid: w for cid, d in self._cg[scope].items() if (w := self._wbytes(d)) is not None}
        thr: dict[str, int] = {}
        for cid, d in self._cg["load"].items():
            thr.update(self._threads(cid, d))
        self.proc_cum += sum(v - self._thr[k] for k, v in thr.items() if k in self._thr and v >= self._thr[k])
        self._thr = thr
        rec["proc"] = self.proc_cum
        rec["n_threads"] = len(thr)
        rec["dev_sectors_w"] = pm.diskstats_sectors_written(self.part) if self.part else None
        return rec


def run_sample(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = Sampler(Path(a.data_path), a.prefix, refresh_s=a.refresh_s)
    if s.disk is None:
        sys.stderr.write(f"[sample] cannot resolve the block device of {a.data_path}\n")
        return EXIT_ENV
    head = {"schema": SAMPLE_SCHEMA, "kind": "header", "utc": utc(), "data_path": str(a.data_path), "partition": s.part, "disk": s.disk,
            "prefix": a.prefix, "period_s": a.period_s, "host": pm.host_info()}
    stop = out / STOP
    t_end = time.time() + a.max_s
    n = 0
    with (out / SAMPLES).open("a", encoding="utf-8") as f:
        f.write(json.dumps(head) + "\n")
        t_next = time.time()
        while time.time() < t_end:
            rec = s.sample()
            if n % 60 == 0:                                   # 容器名每分钟随行写一次（判读日志用）
                rec["names"] = {cid: s._names.get(cid) for cid in rec["load"]}
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush()
            n += 1
            if stop.exists():
                break
            t_next += a.period_s
            time.sleep(max(0.0, t_next - time.time()))
    sys.stderr.write(f"[sample] {n} samples, stop file {'seen' if stop.exists() else 'absent'}\n")
    return EXIT_OK


def load_samples(path: Path) -> tuple[dict, list[dict]]:
    head, recs = {}, []
    if not Path(path).is_file():
        return head, recs
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        with contextlib.suppress(json.JSONDecodeError):
            r = json.loads(ln)
            if r.get("kind") == "header":
                head = r
            else:
                recs.append(r)
    return head, recs


# ================================================================================================================ 速率与统计
def intervals(recs: Sequence[Mapping], key: str, *, scale: float = 1.0) -> list[tuple[float, float, float]]:
    """相邻采样的增量 → [(t0, t1, 字节)]。`recs[i][key]` 为 {id: 累计量}（只计两次采样都存在的 id；累计量回退视为计数重置，
    该 id 本间隔不计）或标量累计量（`dev_sectors_w`，`scale` 为每单位字节数）。"""
    out = []
    for a, b in itertools.pairwise(recs):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None or b["t"] <= a["t"]:
            continue
        if isinstance(va, Mapping):
            d = sum(vb[k] - va[k] for k in vb.keys() & va.keys() if vb[k] >= va[k])
        else:
            d = max(0, vb - va)
        out.append((float(a["t"]), float(b["t"]), d * scale))
    return out


def bucket_rates(iv: Sequence[tuple[float, float, float]], t0: float, t1: float, w_s: float, step_s: float | None = None) -> list[float]:
    """[t0, t1) 内长 `w_s`、步长 `step_s`（缺省 = `w_s`，即不重叠；须整除 `w_s`）的窗口写速率（MB/s）：采样间隔按与步长格的重叠时长
    比例分摊字节，窗口为连续 `w_s / step_s` 格之和；只计完整落在 [t0, t1) 内的窗口，且窗口内采样覆盖 ≥ 90%（采样缺口不以 0 速率填充）。"""
    step = float(step_s or w_s)
    k = round(w_s / step)
    if k < 1 or abs(k * step - w_s) > 1e-9:
        raise ValueError("window length must be a multiple of the step")
    n = math.floor((t1 - t0) / step + 1e-9)
    if n < k:
        return []
    byts = [0.0] * n
    cov = [0.0] * n
    for a, b, x in iv:
        if b <= t0 or a >= t1:
            continue
        span = b - a
        i0 = max(0, int((max(a, t0) - t0) // step))
        i1 = min(n - 1, int((min(b, t1) - t0 - 1e-9) // step))
        for i in range(i0, i1 + 1):
            lo, hi = t0 + i * step, t0 + (i + 1) * step
            ov = min(b, hi) - max(a, lo)
            if ov > 0:
                byts[i] += x * ov / span
                cov[i] += ov
    return [sum(byts[i:i + k]) / w_s / MB for i in range(n - k + 1) if sum(cov[i:i + k]) >= 0.9 * w_s]


def rate_stats(rates: Sequence[float]) -> dict:
    q = pm.quantiles(rates, (0.5, WRITE_P95_Q, 0.99))
    return {"n": len(rates), "mean": round(sum(rates) / len(rates), 3) if rates else None, **{k: (round(v, 3) if v is not None else None)
                                                                                              for k, v in q.items()},
            "max": round(max(rates), 3) if rates else None}


def coverage(recs: Sequence[Mapping], t0: float, t1: float, max_gap_s: float = 5.0) -> float:
    """测量窗口内被采样覆盖的比例（相邻采样间隔 > `max_gap_s` 视为缺口）。"""
    if t1 <= t0:
        return 0.0
    got = 0.0
    for a, b in itertools.pairwise(recs):
        lo, hi = max(a["t"], t0), min(b["t"], t1)
        if hi > lo and b["t"] - a["t"] <= max_gap_s:
            got += hi - lo
    return round(got / (t1 - t0), 6)


def measured_1min(path: Path, now: float | None = None, *, key: str = "load") -> float | None:
    """调度器遥测（M27 §6.7.3）：`io_samples.jsonl` 末尾 60 s 的 `load` 口径平均写速率（MB/s）；数据不足 30 s 时 None。"""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 400_000))
            tail = f.read().decode("utf-8", errors="replace").splitlines()[1:]
    except OSError:
        return None
    recs = []
    for ln in tail:
        with contextlib.suppress(json.JSONDecodeError):
            r = json.loads(ln)
            if "t" in r and key in r:
                recs.append(r)
    if not recs:
        return None
    now = now if now is not None else recs[-1]["t"]
    recs = [r for r in recs if now - 60.0 <= r["t"] <= now + 1.0]
    if len(recs) < 2 or recs[-1]["t"] - recs[0]["t"] < 30.0:
        return None
    iv = intervals(recs, key)
    span = recs[-1]["t"] - recs[0]["t"]
    return round(sum(x for _, _, x in iv) / span / MB, 3)


def block_stats(runs: Sequence[Mapping]) -> dict:
    """各 Run 的写入器阻塞：`writer_block_frac`（Σ 两个写入器阻塞 / 写入器墙钟）的最大值与合计占比、最大单次阻塞、丢弃数。"""
    rows = [r for r in runs if r.get("recorder")]
    if not rows:
        return {"n_runs": 0, "max_frac": None, "aggregate_frac": None}
    fr = [float(r["recorder"]["writer_block_frac"]) for r in rows]
    blk = sum(float(r["recorder"]["writer_block_s"]) for r in rows)
    wall = sum(float(r["recorder"]["wall_s"]) for r in rows)
    return {"n_runs": len(rows), "max_frac": round(max(fr), 6), "median_frac": pm.median(fr), "aggregate_frac": round(blk / wall, 6) if wall else None,
            "block_s_total": round(blk, 3), "writer_wall_s_total": round(wall, 1),
            "blocked_total": sum(int(r["recorder"].get("blocked", 0)) for r in rows),
            "block_max_s": round(max(float(r["recorder"].get("block_max_s", 0.0)) for r in rows), 6),
            "max_depth": max(int(r["recorder"].get("max_depth", 0)) for r in rows),
            "max_depth_bytes": max(int(r["recorder"].get("max_depth_bytes", 0)) for r in rows),
            "dropped_total": sum(int(r["recorder"].get("dropped", 0)) for r in rows)}


def recorder_brief(rec: Mapping) -> dict:
    """清单 `recorder` 块（`drone4d.bench.recorder_stats.v1`）→ 本工具的写入器统计（raw 与真值合并，分项保留）。"""
    raw, tr = rec.get("raw") or {}, rec.get("truth") or {}

    def g(d: Mapping, k: str, x=0):
        return d.get(k, x) or x
    return {"writer_block_s": float(rec.get("writer_block_s", 0.0)), "writer_block_frac": float(rec.get("writer_block_frac", 0.0)),
            "wall_s": float(g(raw, "wall_s", 0.0)), "blocked": int(g(raw, "blocked")) + int(g(tr, "blocked")),
            "block_max_s": max(float(g(raw, "block_max_s", 0.0)), float(g(tr, "block_max_s", 0.0))),
            "max_depth": max(int(g(raw, "max_depth")), int(g(tr, "max_depth"))),
            "max_depth_bytes": max(int(g(raw, "max_depth_bytes")), int(g(tr, "max_depth_bytes"))),
            "dropped": int(rec.get("dropped", 0)), "options": rec.get("options"),
            "raw": {k: raw.get(k) for k in ("bytes", "messages", "blocked", "block_s", "block_max_s", "max_depth", "max_depth_bytes", "write_s",
                                            "wall_s")},
            "truth": {k: tr.get(k) for k in ("bytes", "messages", "blocked", "block_s", "block_max_s", "max_depth", "max_depth_bytes", "write_s",
                                             "wall_s")}}


def concurrency(runs: Sequence[Mapping], t0: float, t1: float) -> dict:
    """测量窗口内运行中 Run 数的时间加权分布（由各 Run 的准入与结束时刻）。"""
    ev = []
    for r in runs:
        a, b = r.get("t_admit"), r.get("t_end")
        if a is None:
            continue
        b = t1 if b is None else b
        lo, hi = max(a, t0), min(b, t1)
        if hi > lo:
            ev += [(lo, 1), (hi, -1)]
    ev.sort()
    hist: dict[int, float] = {}
    cur, last = 0, t0
    for t, d in ev:
        hist[cur] = hist.get(cur, 0.0) + (t - last)
        cur, last = cur + d, t
    hist[cur] = hist.get(cur, 0.0) + max(0.0, t1 - last)
    span = max(t1 - t0, 1e-9)
    return {"time_frac_by_n": {str(k): round(v / span, 4) for k, v in sorted(hist.items()) if v > 0},
            "mean": round(sum(k * v for k, v in hist.items()) / span, 3), "busy_frac": round(sum(v for k, v in hist.items() if k > 0) / span, 4)}


# ================================================================================================================ worker（单个 Run）
def _tree_bytes(p: Path) -> int:
    n = 0
    for q in Path(p).rglob("*"):
        with contextlib.suppress(OSError):
            if q.is_file() and not q.is_symlink():
                n += q.stat().st_size
    return n


def _retention(run_dir: Path, root: Path, mode: str) -> dict:
    """生产保留：`compact` 为 `ranked_eval` 精简（逐帧摘要 + truth_min，ADR-163）；`delete` 只删 MCAP 与帧索引；`keep` 不动。"""
    t = time.perf_counter()
    if mode == "compact":
        from awr.bench.retention import compact_ranked

        try:
            res = compact_ranked(run_dir, root=root)
        except Exception as e:                                # 精简失败不影响本 Run 的写入测量：记录原因，退回只删 MCAP 控制占用
            fb = _retention(run_dir, root, "delete")
            return {"mode": mode, "compacted": False, "error": f"{type(e).__name__}: {e}", "fallback": fb, "s": round(time.perf_counter() - t, 2)}
        return {"mode": mode, "compacted": res.get("compacted"), "deleted_bytes": sum(int(d.get("bytes", 0)) for d in res.get("deleted", [])),
                "bytes_after": res.get("bytes_after"), "s": round(time.perf_counter() - t, 2)}
    if mode == "delete":
        n = 0
        for q in [run_dir / "raw.mcap", run_dir / "frames.idx", *sorted((run_dir / "truth").glob("*.mcap"))]:
            if q.is_file():
                n += q.stat().st_size
                q.unlink()
        return {"mode": mode, "deleted_bytes": n, "s": round(time.perf_counter() - t, 2)}
    return {"mode": "keep"}


def _env_run(job: Mapping, name: str, seed: int) -> dict:
    """正式配置的一个 Run（与 `determinism_official.run_one` 变体 a 同一循环：直接交付、参考策略在进程内、无背压注入）。"""
    do = _sibling("determinism_official")
    from anet4d.baselines import BASELINES

    from awr.agentsdk.env.core import BenchEnv

    spec = do._spec_for({**job, "seed": seed, "run_id": f"{job.get('run_id_prefix', 'b23')}-{job['policy']}-s{seed}"}, name)
    for k in ("AWR_REC_INJECT_DELAY_MS", "AWR_REC_RATE_MBPS", "AWR_REC_QUEUE_MSGS", "AWR_REC_QUEUE_MB"):
        os.environ.pop(k, None)                               # 生产负载：写入器缺省参数，不注入
    env = BenchEnv(spec)
    obs, infos = env.reset()
    pool = {u: BASELINES[job["policy"]]() for u in env.possible_agents}
    for u, ag in pool.items():
        ag.reset(infos[u]["config"])
    try:
        while env.agents:
            obs, *_ = env.step({u: pool[u].step(o["obs"], o["inbox"]) for u, o in obs.items() if u in env.agents})
    finally:
        man = env.close()
    ti = man.get("tier_info") or {}
    return {"run_id": man.get("run_id"), "sim_s": man.get("duration_s"), "n_ticks": man.get("n_ticks"), "env_wall_s": man.get("wall_s"),
            "recorder": man.get("recorder") or {}, "gz": {k: ti.get(k) for k in ("ticks", "restarts", "replayed_ticks", "gz_rtf_mean")},
            "timeouts": {u: x.get("timeouts", 0) for u, x in sorted((man.get("agents") or {}).items())}}


def synthetic_run(run_dir: Path, *, sim_s: float, mbps: float, msg_kb: float = 512.0, truth_frac: float = 0.8, seed: int = 0,
                  queue_msgs: int = 256, queue_mb: float = 256.0) -> dict:
    """合成负载（本机冒烟与写入器压测）：以实时步调（100 ms tick）经同一 `RunWriter`（异步有界队列）写 `raw.mcap` 与真值 MCAP，
    平均 `mbps` MB/s，其中 `truth_frac` 进入真值写入器；载荷为不可压缩随机字节（磁盘写入量 ≈ 载荷量）。→ 与 `EnvRecorder.finish`
    同口径的 `recorder` 块。"""
    import numpy as np

    from awr.bench.recorder.writer import RunWriter

    rng = np.random.default_rng(seed)
    kw = {"queue_msgs": queue_msgs, "queue_bytes": int(queue_mb * (1 << 20))}
    (run_dir / "truth").mkdir(parents=True, exist_ok=True)
    raw = RunWriter(run_dir / "raw.mcap", **kw)
    tru = RunWriter(run_dir / "truth" / "run_synthetic.mcap", **kw)
    tick = 0.1
    per_tick = mbps * MB * tick
    msg = max(1024, int(msg_kb * 1024))
    t0 = time.perf_counter()
    n_ticks = round(sim_s / tick)
    for k in range(n_ticks):
        left = per_tick
        i = 0
        while left > 0:
            n = int(min(msg, left))
            w, topic = (tru, "/eval/hidden/synthetic") if rng.random() < truth_frac else (raw, "/uav/uav-01/obs/synthetic")
            w.write_json(topic, "awr.bench.Synthetic", k * 1000 + i, k * 100_000_000, rng.integers(0, 256, n, dtype=np.uint8).tobytes())
            left -= n
            i += 1
        lag = t0 + (k + 1) * tick - time.perf_counter()
        if lag > 0:
            time.sleep(lag)
    files = {}
    for layer, w in (("raw", raw), ("truth", tru)):
        w.close()
        st = w.summary()
        files[layer] = {**{k: st[k] for k in ("mode", "enqueued", "written", "dropped", "blocked", "block_s", "block_max_s", "block_frac",
                                              "max_depth", "max_depth_bytes", "write_s", "wall_s")}, "bytes": w.path.stat().st_size,
                        "messages": st["messages"]}
    blk = files["raw"]["block_s"] + files["truth"]["block_s"]
    wall = max(files["raw"]["wall_s"], 1e-9)
    return {"run_id": run_dir.name, "sim_s": sim_s, "n_ticks": n_ticks, "env_wall_s": round(time.perf_counter() - t0, 3),
            "recorder": {"schema": "drone4d.bench.recorder_stats.v1", "raw": files["raw"], "truth": files["truth"],
                         "options": {"queue_msgs": queue_msgs, "queue_mb": queue_mb, "inject_delay_ms": 0.0, "rate_MBps": None},
                         "writer_block_s": round(blk, 6), "writer_block_frac": round(blk / wall, 6),
                         "dropped": files["raw"]["dropped"] + files["truth"]["dropped"]},
            "gz": None, "timeouts": {}}


def run_worker(a: argparse.Namespace) -> int:
    job = json.loads(Path(a.job).read_text(encoding="utf-8"))
    out = Path(job["out"])
    run_dir = out / "runs" / a.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    t0, u0 = time.time(), utc()
    info: dict[str, Any] = {"schema": ONE_SCHEMA, "name": a.name, "seed": a.seed, "worker": job["worker"], "utc_start": u0, "t_start": round(t0, 3)}
    try:
        if job["worker"] == "synthetic":
            r = synthetic_run(run_dir, sim_s=float(job["run_duration_s"] or 5.0), mbps=float(job["synthetic_mbps"]), seed=a.seed,
                              msg_kb=float(job.get("synthetic_msg_kb", 512.0)))
        else:
            r = _env_run(job, a.name, a.seed)
    except Exception as e:                                # 失败记录到运行目录，由 load 计入失败
        import traceback

        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        info.update(ok=False, error=f"{type(e).__name__}: {e}", wall_s=round(time.time() - t0, 2), utc_end=utc())
        _write_json(run_dir / RUN_INFO, info)
        return EXIT_FAIL
    wall = time.time() - t0
    rec = recorder_brief(r["recorder"]) if r.get("recorder") else None
    mcap_bytes = sum(int((r["recorder"].get(k) or {}).get("bytes") or 0) for k in ("raw", "truth")) if rec else 0
    total = _tree_bytes(run_dir)
    info.update(ok=bool(rec) and rec["dropped"] == 0, run_id=r["run_id"], sim_s=r["sim_s"], n_ticks=r["n_ticks"], env_wall_s=r["env_wall_s"],
                wall_s=round(wall, 2), rtf=round(float(r["sim_s"] or 0) / wall, 4) if wall > 0 else None, recorder=rec, gz=r.get("gz"),
                timeouts=r.get("timeouts"), bytes={"mcap": mcap_bytes, "run_dir_before_retention": total},
                mean_write_MBps=round(total / wall / MB, 3) if wall > 0 else None,
                mcap_MB_per_sim_s=round(mcap_bytes / MB / float(r["sim_s"]), 3) if r.get("sim_s") else None)
    info["t_env_end"] = round(time.time(), 3)
    info["retention"] = _retention(run_dir, out / "runs", job.get("retention", "compact"))
    info["utc_end"], info["t_end"] = utc(), round(time.time(), 3)
    _write_json(run_dir / RUN_INFO, info)
    sys.stderr.write(f"[worker] {a.name}: wall {wall:.0f} s, rtf {info['rtf']}, block_frac {rec and rec['writer_block_frac']}, "
                     f"{total / 1e9:.2f} GB\n")
    return EXIT_OK if info["ok"] else EXIT_FAIL


# ================================================================================================================ load（调度器驱动的生产负载）
def make_load_job(a: argparse.Namespace, out: Path) -> dict:
    spec_path = Path(a.spec)
    rs = json.loads(spec_path.read_text(encoding="utf-8"))
    images = dict(x.split("=", 1) for x in a.image)
    return {"spec": str(spec_path.resolve()), "policy": a.policy, "out": str(out), "worker": a.worker, "run_duration_s": a.run_duration_s,
            "duration_s": a.run_duration_s, "worlds_root": str(Path(a.worlds_root).resolve()),
            "events_root": str(Path(a.events_root).resolve()) if a.events_root else None, "physics": a.physics,
            "gz": {"launcher": a.gz_launcher, "image": a.gz_image, "export_root": a.gz_export_root}, "images": images,
            "gz_prefix": a.gz_prefix, "run_id_prefix": a.run_id_prefix, "retention": a.retention, "synthetic_mbps": a.synthetic_mbps,
            "synthetic_msg_kb": a.synthetic_msg_kb, "world": (rs.get("world") or {}).get("id"), "ranked": bool(rs.get("ranked"))}


def _prepare_world(job: Mapping) -> None:
    """世界级导出、几何查询派生缓存（`AWR_GEO_CACHE_DIR`，世界目录只读挂载时须指向可写目录）与 MeshWorldQuery 缓存先行一次，
    避免并行 Run 同时派生、写同一目录（同 determinism_official）；Env 打开世界失败在此直接抛出，不进入准入循环。"""
    if job["worker"] != "env":
        return
    world_dir = Path(job["worlds_root"]) / job["world"]
    if (job.get("physics") or "realistic") == "gz":                                        # gz world export only for the gz tier (ADR-226)
        from awr.world.export.gz import export_world

        export_world(world_dir, out_root=Path(job["gz"]["export_root"] or Path(job["out"]) / "export"))
    from awr.agentsdk.env.core import _open_world

    _open_world(world_dir)                                    # 与 BenchEnv 同一路径（缺省缓存根）
    with contextlib.suppress(Exception):
        from awr.world.mesh.query import open_mesh_world_query

        open_mesh_world_query(world_dir, world_dir.resolve().parent.parent / "cache" / "worldquery")


def run_load(a: argparse.Namespace) -> int:
    from awr.bench.scheduler import GpuSlot, SchedConfig, Scheduler, Telemetry, make_job, resource_args

    out = Path(a.out).resolve()
    if a.worker == "env" and not a.events_root:
        sys.stderr.write("[load] --events-root is required for the env worker\n")
        return EXIT_ARGS
    from awr.phys.gz.instance import host_path

    if not pm.d3_dir_ok(Path(host_path(out))) and not a.allow_any_dir:          # 容器内按 AWR_HOST_PATH_MAP 换算为宿主路径再核对
        sys.stderr.write(f"[load] {out} (host {host_path(out)}): output must be under a d3 / anet-drone4d-d3 directory\n")
        return EXIT_ENV
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        (out / STOP).unlink()
    job = make_load_job(a, out)
    _write_json(out / LOAD_JOB, job)
    _prepare_world(job)
    rs = json.loads(Path(job["spec"]).read_text(encoding="utf-8"))
    if a.run_duration_s:
        rs = {**rs, "duration_s": float(a.run_duration_s)}
    cfg = SchedConfig(measured_disk_mbps=a.measured_disk_mbps)
    sched = Scheduler([GpuSlot(a.gpu, exclusive=True)], config=cfg)
    cls = a.priority_class or None
    t_start = time.time()
    t0: float | None = None                                   # 测量窗口起点：首个 Run 准入
    seed = a.seed0
    runs: dict[str, dict] = {}
    procs: dict[str, subprocess.Popen] = {}
    status_log: list[dict] = []
    t_status = 0.0
    stopping = [False]
    fail_streak = 0

    def _sig(*_):
        stopping[0] = True
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    samples = Path(a.samples) if a.samples else out / SAMPLES
    try:
        while True:
            now = time.time()
            window_open = not stopping[0] and (t0 is None or now < t0 + a.duration_s) and now - t_start < a.duration_s + a.start_grace_s
            if window_open:
                while len(sched.queue) < a.queue_depth and (a.max_runs is None or seed - a.seed0 < a.max_runs):
                    name = f"r{seed:03d}"
                    j = make_job(name, rs, tenant=a.tenant, submitted=now, priority_class=cls, kind="realistic_run")
                    sched.submit(j)
                    runs[name] = {"name": name, "seed": seed, "class": j.priority_class, "t_submit": round(now, 3),
                                  "est": {"write_mbps": round(j.cost.write_mbps, 3), "wall_s": round(j.cost.wall_s_est, 1), "gpu_share": j.cost.gpu_share,
                                          "cpu": j.cost.cpu_cores, "mem_gb": j.cost.mem_gb, "bytes_persist": j.cost.bytes_persist}}
                    seed += 1
            elif sched.queue:
                for j in list(sched.queue):
                    sched.cancel(j.job_id)
                    runs[j.job_id]["cancelled"] = True
            tel = Telemetry(write_mbps_1min=measured_1min(samples, now))
            for d in sched.tick(now, tel):
                r = runs[d.job_id]
                if not d.admitted:
                    r["blocked_by"] = list(d.blocked_by)
                    continue
                if t0 is None:
                    t0 = now
                r.update(t_admit=round(now, 3), utc_admit=utc(now), gpu=d.gpu, share=d.share, condition=d.condition,
                         docker_args=resource_args(sched.running[d.job_id].job, d))
                log = (out / "logs" / f"{d.job_id}.log").open("w", encoding="utf-8")
                procs[d.job_id] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--job", str(out / LOAD_JOB),
                                                    "--name", d.job_id, "--seed", str(r["seed"])], stdout=log, stderr=subprocess.STDOUT)
                log.close()
                sys.stderr.write(f"[load] {utc(now)} admit {d.job_id} (running {len(sched.running)}, est write "
                                 f"{sched.status(now)['write']['est_mbps']} MB/s, measured 1 min {tel.write_mbps_1min})\n")
            for name, p in list(procs.items()):
                rc = p.poll()
                if rc is None:
                    continue
                del procs[name]
                sched.finish(name)
                runs[name].update(rc=rc, t_end=round(time.time(), 3), utc_end=utc())
                fail_streak = fail_streak + 1 if rc != 0 else 0
                if fail_streak >= a.max_fail_streak and not stopping[0]:      # 连续失败：停止准入，不以失败 Run 冒充负载
                    stopping[0] = True
                    sys.stderr.write(f"[load] {fail_streak} consecutive failed runs; stop admitting\n")
                info = out / "runs" / name / RUN_INFO
                if info.is_file():
                    runs[name]["info"] = json.loads(info.read_text(encoding="utf-8"))
                sys.stderr.write(f"[load] {utc()} done {name} rc {rc}\n")
            if now - t_status >= 60.0:
                t_status = now
                st = sched.status(now)
                status_log.append({"t": round(now, 3), "running": st["running"], "queued": st["queued"], "write_est_mbps": st["write"]["est_mbps"],
                                   "write_measured_1min": tel.write_mbps_1min, "gpu": st["gpus"]})
                _dump_load(out, a, job, cfg, sched, runs, status_log, t_start, t0)
            if not window_open and not procs and not sched.queue:
                break
            time.sleep(a.tick_s)
    finally:
        for p in procs.values():                              # 异常退出：终止未完成的 Run（正常路径此处为空）
            with contextlib.suppress(OSError):
                p.terminate()
        doc = _dump_load(out, a, job, cfg, sched, runs, status_log, t_start, t0)
        if not a.no_stop_file:
            (out / STOP).write_text(utc() + "\n", encoding="utf-8")
    failed = [n for n, r in runs.items() if r.get("t_admit") and r.get("rc") != 0]
    sys.stderr.write(f"[load] window {doc['window']}, admitted {doc['n_admitted']}, failed {failed}\n")
    return EXIT_FAIL if failed or t0 is None else EXIT_OK


def _dump_load(out: Path, a: argparse.Namespace, job: Mapping, cfg, sched, runs: Mapping, status_log: list, t_start: float, t0: float | None) -> dict:
    doc = {"schema": RUN_SCHEMA, "acceptance": "D3-AC-22", "utc_start": utc(t_start), "utc_written": utc(), "job": job, "argv": sys.argv[1:],
           "scheduler": {"gpu": a.gpu, "exclusive": True, "write_budget_mbps": cfg.write_budget_mbps, "budget_source": cfg.write_budget_source,
                         "measured_disk_mbps": cfg.measured_disk_mbps, "write_cap_mbps": cfg.write_cap_mbps, "write_fraction": cfg.write_fraction,
                         "eval_max_runs_per_gpu": cfg.eval_max_runs_per_gpu, "eval_share": cfg.eval_share,
                         "events": [{"t": round(t, 3), "job": j, "reason": c, "blocked_by": list(b)} for t, j, c, b in sched.events]},
           "window": {"t0": t0, "t1": (t0 + a.duration_s) if t0 is not None else None, "duration_s": a.duration_s,
                      "utc0": utc(t0) if t0 is not None else None},
           "n_admitted": sum(1 for r in runs.values() if r.get("t_admit")), "runs": list(runs.values()), "status": status_log}
    _write_json(out / LOAD_DOC, doc)
    return doc


# ================================================================================================================ analyze
def guard_exclusivity(g: Mapping | None) -> dict:
    """guard 记录 → 独占级别与计入状态（ADR-195 (2)：B 级须窗口前空闲显存 ≥ 本负载显存需求 + 4 GB）。"""
    if not g:
        return {"level": None, "counted": False, "label": "未经 guard 测量（共享卡，开发参考）"}
    pre_free, run_free = g.get("pre_free_min_mib"), g.get("run_free_min_mib")
    need = max(0, pre_free - run_free) if pre_free is not None and run_free is not None else None
    bok = pm.b_level_ok(pre_free, need)
    level = g.get("level")
    return {"gpu": g.get("gpu"), "level": level, "need_mib": need, "b_free_ok": bok, "external_pids": (g.get("run") or {}).get("external_pids"),
            "counted": pm.counted_for(level, bok), "utc": [g.get("utc_start"), g.get("utc_end")], "load": g.get("load"),
            "label": {"A": "独占（A 级）", "B": "独占（驻留豁免，待用户确认）" if bok else "共享"}.get(level, "无效或共享")}


def analyze(out: Path, *, guard: Mapping | None = None, primary: str = "load") -> tuple[dict, list[dict]]:
    """→ (result, checks)。判定：1 s 写速率 p95 ≤ 预算（`primary` 口径）、写入器阻塞最大 Run ≤ 1% 墙钟、窗口 ≥ 1 h 且采样覆盖充分、
    全部 Run 成功且无丢弃。"""
    doc = json.loads((out / LOAD_DOC).read_text(encoding="utf-8"))
    head, recs = load_samples(out / SAMPLES)
    w = doc["window"]
    if w.get("t0") is None:
        raise ValueError("no run was admitted; nothing to analyze")
    t0, t1 = float(w["t0"]), float(w["t1"])
    t_last = recs[-1]["t"] if recs else t0
    t1_eff = min(t1, t_last)
    budget = float(doc["scheduler"]["write_budget_mbps"])
    scopes = {"load": intervals(recs, "load"), "proc": intervals(recs, "proc"), "d3": intervals(recs, "d3"),
              "dev": intervals(recs, "dev_sectors_w", scale=512.0)}
    rates = {s: {f"{ws}s": rate_stats(bucket_rates(iv, t0, t1_eff, ws, st)) for ws, st in RATE_WINDOWS_S} for s, iv in scopes.items()}
    tot = {s: round(sum(x for a, b, x in iv if a >= t0 and b <= t1_eff) / 1e9, 3) for s, iv in scopes.items()}
    runs = [r for r in doc["runs"] if r.get("t_admit")]
    infos = [r.get("info") or {} for r in runs]
    blk = block_stats(infos)
    failed = [r["name"] for r in runs if r.get("rc") != 0 or not (r.get("info") or {}).get("ok")]
    per_run = [{"name": r["name"], "seed": r["seed"], "utc_admit": r.get("utc_admit"), "utc_end": r.get("utc_end"), "rc": r.get("rc"),
                "wall_s": i.get("wall_s"), "rtf": i.get("rtf"), "est_write_mbps": r["est"]["write_mbps"], "mean_write_MBps": i.get("mean_write_MBps"),
                "mcap_MB_per_sim_s": i.get("mcap_MB_per_sim_s"), "bytes_before_retention": (i.get("bytes") or {}).get("run_dir_before_retention"),
                "writer_block_frac": (i.get("recorder") or {}).get("writer_block_frac"), "writer_block_s": (i.get("recorder") or {}).get("writer_block_s"),
                "blocked": (i.get("recorder") or {}).get("blocked"), "dropped": (i.get("recorder") or {}).get("dropped"),
                "gz": i.get("gz"), "retention": i.get("retention")} for r, i in zip(runs, infos, strict=True)]
    in_window = [r for r in runs if float(r["t_admit"]) < t1]
    cov = coverage(recs, t0, t1)
    p95s = [rates[primary][w].get("p95") for w in JUDGED_WINDOWS]
    p95 = None if any(x is None for x in p95s) else max(p95s)
    n_load_containers = max((len(r.get("load") or {}) for r in recs), default=0)
    excl = guard_exclusivity(guard)
    est = [x["est_write_mbps"] for x in per_run if x["mean_write_MBps"]]
    meas = [x["mean_write_MBps"] for x in per_run if x["mean_write_MBps"]]
    result = {"window": {"utc0": w.get("utc0"), "utc1": utc(t1), "duration_s": round(t1 - t0, 1), "sampled_until_utc": utc(t_last),
                         "coverage": cov, "samples": len(recs), "period_s": head.get("period_s"), "disk": head.get("disk"),
                         "partition": head.get("partition"), "data_path": head.get("data_path")},
              "budget": {"budget_MBps": budget, "source": doc["scheduler"]["budget_source"], "measured_disk_MBps": doc["scheduler"]["measured_disk_mbps"],
                         "formula": "min(300 MB/s, 0.5 x measured /data sustained write)"},
              "primary_scope": primary, "write_rate_MBps": rates, "bytes_in_window_GB": tot,
              "proc_vs_load_bytes_ratio": round(tot["proc"] / tot["load"], 4) if tot["load"] else None,
              "max_load_containers": n_load_containers,
              "load": {"runs_admitted_in_window": len(in_window), "runs_total": len(runs), "failed": failed,
                       "concurrency": concurrency([{"t_admit": r["t_admit"], "t_end": r.get("t_end")} for r in runs], t0, t1),
                       "rtf": {"median": pm.median(x["rtf"] for x in per_run), "min": min((x["rtf"] for x in per_run if x["rtf"]), default=None)},
                       "est_write_mbps_per_run": pm.median(est), "measured_write_MBps_per_run": pm.median(meas),
                       "est_over_measured": round(pm.median(est) / pm.median(meas), 2) if est and meas and pm.median(meas) else None,
                       "events_686": sum(1 for e in doc["scheduler"]["events"] if e["reason"] == 686),
                       "events_685": sum(1 for e in doc["scheduler"]["events"] if e["reason"] == 685),
                       "worker": doc["job"]["worker"], "spec": doc["job"]["spec"], "retention": doc["job"]["retention"]},
              "mcap_blocking": blk, "runs": per_run, "exclusivity": excl}
    data_ok = cov >= WINDOW_COVERAGE_MIN and (t1 - t0) >= WINDOW_S - 1e-6 and n_load_containers > 0
    checks = [
        pm.check("AC22-write-p95", "D3 sustained write rate p95 over the 1 h production load (sliding 60 s windows, step 10 s, "
                 "cgroup io.stat of the load)",
                 "D3-AC-22: 1 h 生产负载下 D3 持续写速率 p95 ≤ 写带宽预算 min(300 MB/s, 实测 × 50%)", p95, threshold=budget,
                 passed=(p95 is not None and p95 <= budget) if data_ok else False, unit="MB/s", counted="by_guard",
                 note=f"10 s p95 {rates[primary]['10s'].get('p95')}, 60 s p95 {rates[primary]['60s'].get('p95')}, 1 s p95 (writeback batches, "
                      f"record only) {rates[primary]['1s'].get('p95')}, mean {rates[primary]['10s'].get('mean')}"),
        pm.check("AC22-mcap-block", "MCAP writer blocking time / wall clock, worst run of the load",
                 "D3-AC-22、M27-NFR-003：MCAP 写入器阻塞时间占墙钟 ≤ 1%", blk.get("max_frac"), threshold=BLOCK_FRAC_MAX,
                 passed=blk.get("max_frac") is not None and blk["max_frac"] <= BLOCK_FRAC_MAX and blk["n_runs"] == len(runs), unit="fraction",
                 counted="by_guard", note=f"aggregate {blk.get('aggregate_frac')} over {blk.get('n_runs')} runs, longest block {blk.get('block_max_s')} s"),
        pm.check("AC22-window", "measurement window length and sample coverage", "D3-AC-22：1 h 生产负载",
                 {"duration_s": round(t1 - t0, 1), "coverage": cov, "runs_admitted_in_window": len(in_window)},
                 threshold={"duration_s": WINDOW_S, "coverage": WINDOW_COVERAGE_MIN}, passed=data_ok, unit=""),
        pm.check("AC22-runs-lossless", "all runs completed, writer dropped 0", "ADR-162、M27-FR-043：写入器无损（不丢弃）",
                 {"failed": failed, "dropped_total": blk.get("dropped_total")}, threshold={"failed": [], "dropped_total": 0},
                 passed=not failed and blk.get("dropped_total") == 0, unit=""),
    ]
    for c in checks:
        if c.get("counted") == "by_guard":
            c["counted"] = excl["counted"]
            c["exclusivity"] = excl["label"]
    return result, checks


def run_analyze(a: argparse.Namespace) -> int:
    out = Path(a.out)
    if not (out / LOAD_DOC).is_file() or not (out / SAMPLES).is_file():
        sys.stderr.write(f"[analyze] {out}: {LOAD_DOC} and {SAMPLES} are required\n")
        return EXIT_ENV
    guard = json.loads(Path(a.guard).read_text(encoding="utf-8")) if a.guard else None
    t, u0 = time.perf_counter(), utc()
    try:
        result, checks = analyze(out, guard=guard, primary=a.primary)
    except ValueError as e:
        sys.stderr.write(f"[analyze] {e}\n")
        return EXIT_ENV
    proto = {"window_s": WINDOW_S, "rate_windows_s": [list(x) for x in RATE_WINDOWS_S], "judged_windows": list(JUDGED_WINDOWS),
             "judged_stat": "p95 of the sliding 60 s window rate (max over judged windows)", "primary_scope": a.primary,
             "block_frac_max": BLOCK_FRAC_MAX, "guard": a.guard}
    p = pm.write_part(Path(a.part_out or out), PART, result, checks, protocol=proto, t0=t, utc0=u0)
    for c in checks:
        v = c["value"] if not isinstance(c["value"], dict) else json.dumps(c["value"], ensure_ascii=False)
        sys.stderr.write(f"[analyze] {c['id']}: {v} {c['unit']} (threshold {c['threshold']}) pass={c['pass']} counted={c['counted']}\n")
    sys.stderr.write(f"[analyze] wrote {p}\n")
    return EXIT_OK if all(c["pass"] for c in checks) else EXIT_FAIL


# ================================================================================================================ CLI
def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="write_load.py", description="1 h production write load and MCAP writer blocking (D3-AC-22)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="host-side io.stat / proc io / diskstats sampler (stdlib only)")
    s.add_argument("--out", required=True)
    s.add_argument("--prefix", required=True, help="container name prefix of the load (e.g. d3-B23-)")
    s.add_argument("--data-path", default="/data/anet-drone4d-d3")
    s.add_argument("--period-s", type=float, default=1.0)
    s.add_argument("--refresh-s", type=float, default=5.0)
    s.add_argument("--max-s", type=float, default=4 * 3600.0, help="hard stop of the sampler")
    lo = sub.add_parser("load", help="scheduler-driven production load (inside the host container)")
    lo.add_argument("--out", required=True)
    lo.add_argument("--spec", default=str(ROOT / DEFAULT_SPEC))
    lo.add_argument("--policy", default="track_share")
    lo.add_argument("--gpu", type=int, default=1, help="host index of the exclusive card (scheduler slot)")
    lo.add_argument("--duration-s", type=float, default=WINDOW_S, help="measurement window from the first admission")
    lo.add_argument("--start-grace-s", type=float, default=600.0, help="give up when nothing is admitted within this time")
    lo.add_argument("--run-duration-s", type=float, default=None, help="override the run spec duration (default: the spec, 360 s)")
    lo.add_argument("--measured-disk-mbps", type=float, default=None, help="P-1 fio O_DIRECT median of /data (D00 disk.json)")
    lo.add_argument("--priority-class", choices=("eval", "research", "batch"), default=None, help="default: eval for ranked specs")
    lo.add_argument("--tenant", default="b23")
    lo.add_argument("--seed0", type=int, default=1)
    lo.add_argument("--max-runs", type=int, default=None)
    lo.add_argument("--queue-depth", type=int, default=3, help="jobs kept queued ahead of admission")
    lo.add_argument("--tick-s", type=float, default=1.0)
    lo.add_argument("--max-fail-streak", type=int, default=3, help="stop admitting after this many consecutive failed runs")
    lo.add_argument("--samples", default=None, help="io_samples.jsonl for scheduler telemetry (default <out>/io_samples.jsonl)")
    lo.add_argument("--worker", choices=("env", "synthetic"), default="env")
    lo.add_argument("--synthetic-mbps", type=float, default=20.0)
    lo.add_argument("--synthetic-msg-kb", type=float, default=512.0)
    lo.add_argument("--retention", choices=("compact", "delete", "keep"), default="compact")
    lo.add_argument("--worlds-root", default=str(ROOT / "worlds"))
    lo.add_argument("--events-root", default=None)
    lo.add_argument("--physics", choices=("realistic", "gz", "fast", "kinematic"), default=None)
    lo.add_argument("--gz-launcher", choices=("docker", "local"), default="docker")
    lo.add_argument("--gz-image", default="d3-gz:dev")
    lo.add_argument("--gz-export-root", default=None)
    lo.add_argument("--gz-prefix", default="d3-B23-gz")
    lo.add_argument("--run-id-prefix", default="b23")
    lo.add_argument("--image", action="append", default=[], help="platform image digest KEY=DIGEST (gz, sensorsvc, env)")
    lo.add_argument("--allow-any-dir", action="store_true", help=argparse.SUPPRESS)
    lo.add_argument("--no-stop-file", action="store_true")
    w = sub.add_parser("worker", help="internal: one run")
    w.add_argument("--job", required=True)
    w.add_argument("--name", required=True)
    w.add_argument("--seed", type=int, required=True)
    an = sub.add_parser("analyze", help="write rate p95 and MCAP blocking verdict (part write_load for D01)")
    an.add_argument("--out", required=True)
    an.add_argument("--guard", default=None, help="p1_measure guard record of the load window")
    an.add_argument("--primary", choices=("load", "proc"), default="load")
    an.add_argument("--part-out", default=None, help="directory of write_load.json (default --out)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = _args(argv)
    return {"sample": run_sample, "load": run_load, "worker": run_worker, "analyze": run_analyze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

