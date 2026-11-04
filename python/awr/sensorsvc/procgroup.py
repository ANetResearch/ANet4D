"""M25 Worker 渲染进程组（docs/05 ADR-214 决策 (4)；M25 §6.14.1、§6.21；ADR-139"每 GPU 一个 Worker"）。

一个 M25 Worker 服务实例（统一的调度、会话管理与对外接口）由若干渲染进程组成：每个渲染进程持有自己的 EGL 上下文、CUDA 主上下文、
编码器与常驻世界副本（按进程复制），在单个渲染线程中提交 GPU 工作，帧组装与编码在本进程的线程池中流水并行；Worker 进程（本模块的
调用方）只做调度与会话管理，不持有 GPU 上下文。显存按 ADR-213 决策 (2) 核算：每个渲染进程的会话前基线含上下文与常驻世界，会话增量
为会话期间各渲染进程峰值减去各自基线之和；Worker 合计（各渲染进程之和）≤ 24 GB。

调度有两种方式（`dispatch_mode`），由会话中传感器是否有逐帧状态决定：

- `sensor`（缺省，`assign_units`）：调度单元为"一个会话中一台无人机的全部传感器"，单元在会话期间固定属于一个渲染进程（粘滞），
  同一传感器的帧按序在同一进程内产生，曝光反馈（可见光自动曝光以上一帧测光为输入）等逐帧状态与单进程完全相同，帧序与帧内容
  逐字节一致（测试覆盖）。会话数不少于进程数时按会话分派（一个会话的全部单元在同一进程，会话 k → 进程 k mod R）；会话数少于
  进程数时，进程按 `p mod 会话数` 分给各会话，会话内的单元在分得的进程间轮转。
- `tick`（`tick_owner`）：会话的全部传感器都没有逐帧状态（帧内容只由传感器、帧序号与磁带决定，例如热成像：噪声按帧序号取键，
  FPN 按机取键，演员热状态按磁带时刻求值）时，第 seq 帧（全部无人机）由进程 seq mod R 产生；同一 tick 各机的演员状态与热状态
  在一个进程内只求值一次（与单进程相同的缓存命中），各传感器每个帧序号的内容与单进程逐字节一致，同一传感器相邻帧由不同进程
  并行产生，录制按帧时刻排序。

持续模式的在途上限按每个渲染进程计（`INFLIGHT_PER_PROC`，S09 缺省 4 帧）。

内存预算（docs/05 §4.7"M25 Worker 内存 16 GB"；K13）：每个渲染进程复制一份 EGL/CUDA 上下文、编码器、Warp 上下文与常驻世界，
进程内存与承担的会话数基本无关。`render_procs_for` 在调用方请求的进程数之上按并发会话数取上限：并发会话数不少于
`SESSIONS_CAP_FROM`（3）时渲染进程数不超过 `SESSIONS_CAP_PROCS`（3），即每个会话一个进程（K03 三并发同构），3 个并发标准轨
会话的渲染进程组合计内存落在 16 GB 之内（K08：R = 6 时 6 个进程 RSS 合计约 21 GB）。`tick` 调度（单会话、无逐帧状态，
如热成像持续测量）不受此上限约束。

进程模型（`RenderGroup`）：`multiprocessing` 的 spawn 方式启动（EGL 与 CUDA 上下文不能跨 fork 继承），每个进程按 `target`
（`模块:函数` 或 `文件路径:函数`）构造处理器 `handler = factory(arg, index)`，就绪后在一条或多条消息通道上串行处理请求
`handler.handle(msg) -> reply`；每条通道只由 Worker 的一个控制线程使用（多会话时一个会话一条通道），进程内按到达顺序处理。
`("close",)` 调用 `handler.close()` 并退出。处理器异常以 `("error", 回溯)` 返回，调用方抛出 648 `SENSOR_WORKER_LOST`；进程退出或
超时同样为 648。GPU 丢失例外（K18，B21 遗留 3）：回溯带设备丢失特征（CUDA error 100 / 999、NVML、EGL 初始化失败、647），或进程
退出、失联、超时时本应有 GPU 而 NVIDIA 设备节点已打不开（宿主 systemd reload 后渲染进程在 EGL 中段错误即属此类），改抛 647
`SENSOR_GPU_UNAVAILABLE`（`cause = gpu_lost`，平台失败、可重跑，`awr.sensorsvc.gpuhealth`），与单进程 L2 路径的归类相同。
消息只含小对象（命令、计数与时刻）；帧数据在渲染进程内写入录制，不经进程间通道。

时刻：各进程的 `time.perf_counter()` 在 Linux 上为 CLOCK_MONOTONIC（系统范围），Worker 的 tick 请求时刻与渲染进程的交付时刻可
直接相减（`shared_clock()` 校验）。
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

from awr.sensorsvc.errors import SensorSvcError
from awr.sensorsvc.gpuhealth import check_device_nodes, gpu_expected, gpu_lost, is_gpu_lost_message

__all__ = ["DISPATCH_MODES", "INFLIGHT_PER_PROC", "RENDER_PROCS_MAX", "SESSIONS_CAP_FROM", "SESSIONS_CAP_PROCS", "STATEFUL_KINDS",
           "WORKER_MEM_BUDGET_MIB", "RenderGroup", "Unit", "assign_units", "dispatch_mode", "load_target", "proc_units", "render_procs_for",
           "shared_clock", "tick_owner"]

INFLIGHT_PER_PROC = 4                    # 持续模式每个渲染进程的在途帧上限（S09 缺省，ADR-214 决策 (4)）
RENDER_PROCS_MAX = 16                    # 每个 Worker 的渲染进程数上限（Worker ≤ 16 核，docs/05 §4.7）
READY_TIMEOUT_S = 900.0
DISPATCH_MODES = ("sensor", "tick")
WORKER_MEM_BUDGET_MIB = 16 * 1024        # M25 Worker 内存预算（docs/05 §4.7，ADR-139）
SESSIONS_CAP_FROM = 3                    # 并发会话数达到该值时按会话限制渲染进程数（K13，§4.7 内存预算）
SESSIONS_CAP_PROCS = 3                   # 此时的渲染进程数上限：每会话一个进程（K03 三并发同构）
STATEFUL_KINDS = frozenset({"camera"})   # 有逐帧状态的传感器类别：可见光（自动曝光以同机上一帧测光为输入，M25 §6.5.3）
WORKER_LOST = 648


@dataclass(frozen=True, order=True)
class Unit:
    """调度单元：会话 `session`（序号）中第 `uav` 台无人机的全部传感器。"""
    session: int
    uav: int


def assign_units(sessions: Sequence[int], procs: int) -> dict[Unit, int]:
    """会话（按序号给出各会话的无人机数）→ 单元到渲染进程的固定分派（确定性，与调用次序无关）。

    会话数 S ≥ 进程数 R：会话 k 的全部单元 → 进程 k mod R（按会话分派）。S < R：进程 p 属于会话 p mod S，会话内第 i 个单元 → 该会话
    第 (i mod 分得进程数) 个进程。"""
    R = int(procs)
    if R < 1 or R > RENDER_PROCS_MAX:
        raise SensorSvcError(640, f"render procs must be in 1..{RENDER_PROCS_MAX}, got {procs}")
    S = len(sessions)
    out: dict[Unit, int] = {}
    if S == 0:
        return out
    for s, n in enumerate(sessions):
        if int(n) < 1:
            raise SensorSvcError(640, f"session {s} has no units")
        own = [s % R] if S >= R else [p for p in range(R) if p % S == s]
        for i in range(int(n)):
            out[Unit(s, i)] = own[i % len(own)]
    return out


def render_procs_for(requested: int, sessions: int, dispatch: str = "sensor") -> int:
    """请求的渲染进程数 → §4.7 内存预算下实际启动的进程数上限（模块文档"内存预算"）。`sensor` 调度且并发会话数 ≥ `SESSIONS_CAP_FROM`
    时取 min(请求, `SESSIONS_CAP_PROCS`)；`tick` 调度（单会话）与会话数更少时不变。请求超出 1..`RENDER_PROCS_MAX` 时 640。"""
    R = int(requested)
    if R < 1 or R > RENDER_PROCS_MAX:
        raise SensorSvcError(640, f"render procs must be in 1..{RENDER_PROCS_MAX}, got {requested}")
    if dispatch not in DISPATCH_MODES:
        raise SensorSvcError(640, f"unknown dispatch mode {dispatch!r}")
    if dispatch == "sensor" and int(sessions) >= SESSIONS_CAP_FROM:
        R = min(R, SESSIONS_CAP_PROCS)
    return R


def proc_units(assign: dict[Unit, int], procs: int) -> list[dict[int, list[int]]]:
    """分派表 → 每个渲染进程承担的 {会话: [无人机序号]}（升序）；无单元的进程为空表。"""
    out: list[dict[int, list[int]]] = [{} for _ in range(int(procs))]
    for u in sorted(assign):
        out[assign[u]].setdefault(u.session, []).append(u.uav)
    return out


def dispatch_mode(kinds: Sequence[str]) -> str:
    """会话的传感器类别（camera、thermal、lidar）→ 调度方式：含有逐帧状态的类别时为 `sensor`，否则为 `tick`。"""
    return "sensor" if set(kinds) & STATEFUL_KINDS or not kinds else "tick"


def tick_owner(seq: int, procs: int) -> int:
    """`tick` 调度：第 seq 帧（全部无人机）所属的渲染进程。"""
    if int(procs) < 1:
        raise SensorSvcError(640, f"render procs must be >= 1, got {procs}")
    return int(seq) % int(procs)


def shared_clock() -> bool:
    """`time.perf_counter` 是否为系统范围的单调时钟（跨进程可比）。"""
    impl = time.get_clock_info("perf_counter").implementation
    return "CLOCK_MONOTONIC" in impl and "RAW" not in impl


def load_target(target: str):
    """`包.模块:函数` 或 `文件路径.py:函数` → 可调用对象。按文件路径加载时模块名取 `awr_rp_<文件名>`，不改变 `sys.path`。"""
    mod_name, _, fn = target.rpartition(":")
    if not mod_name or not fn:
        raise ValueError(f"render target must be 'module:function' or 'path.py:function', got {target!r}")
    if mod_name.endswith(".py"):
        path = Path(mod_name)
        name = f"awr_rp_{path.stem}"
        mod = sys.modules.get(name)
        if mod is None:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_name)
    return getattr(mod, fn)


def _child_main(target: str, arg: Any, index: int, conns: list[Connection], stdout_to_stderr: bool = True) -> None:
    """渲染进程主循环：构造处理器，就绪后在全部通道上等待请求并串行处理。`stdout_to_stderr` 时把文件描述符 1 并入 2（Worker 的
    标准输出常为控制协议，库的初始化输出不得混入）。"""
    if stdout_to_stderr:
        with contextlib.suppress(OSError):
            sys.stdout.flush()
            os.dup2(2, 1)
    try:
        handler = load_target(target)(arg, index)
        info = getattr(handler, "info", None)
        conns[0].send(("ready", {"pid": os.getpid(), "index": index, **(info() if callable(info) else (info or {}))}))
    except BaseException:  # 构造失败原样报告给 Worker
        with contextlib.suppress(OSError, BrokenPipeError):
            conns[0].send(("error", traceback.format_exc()))
        return
    live = list(conns)
    while live:
        for c in wait(live):
            try:
                msg = c.recv()
            except (EOFError, OSError):
                live.remove(c)
                continue
            if msg and msg[0] == "close":
                try:
                    c.send(("ok", handler.close() if hasattr(handler, "close") else None))
                except BaseException:
                    with contextlib.suppress(OSError, BrokenPipeError):
                        c.send(("error", traceback.format_exc()))
                return
            try:
                rep = ("ok", handler.handle(msg))
            except BaseException:  # 处理器异常回报，进程继续服务其他通道
                rep = ("error", traceback.format_exc())
            try:
                c.send(rep)
            except (OSError, BrokenPipeError):
                live.remove(c)


class RenderGroup:
    """一个 Worker 的渲染进程组。`args[k]` 为第 k 个渲染进程的构造参数（可序列化）；`channels` 为每个进程的消息通道数。"""

    def __init__(self, target: str, args: Sequence[Any], *, channels: int = 1, ready_timeout_s: float = READY_TIMEOUT_S,
                 stdout_to_stderr: bool = True) -> None:
        if not args:
            raise SensorSvcError(640, "render group needs at least one process")
        if len(args) > RENDER_PROCS_MAX:
            raise SensorSvcError(640, f"render procs must be <= {RENDER_PROCS_MAX}, got {len(args)}")
        ctx = mp.get_context("spawn")
        self.target = target
        self.procs: list[Any] = []
        self.conns: list[list[Connection]] = []
        self.infos: list[dict] = []
        self._closed = False
        for k, a in enumerate(args):
            pairs = [ctx.Pipe(duplex=True) for _ in range(max(1, int(channels)))]
            p = ctx.Process(target=_child_main, args=(target, a, k, [c for _, c in pairs], stdout_to_stderr), name=f"m25-render-{k}", daemon=False)
            p.start()
            for _, c in pairs:
                c.close()                                                 # 子进程一端在本进程关闭，子进程退出时 recv 得到 EOF
            self.procs.append(p)
            self.conns.append([c for c, _ in pairs])
        try:
            deadline = time.monotonic() + ready_timeout_s
            for k in range(len(self.procs)):
                _st, info = self._recv(k, 0, max(0.0, deadline - time.monotonic()))
                self.infos.append(info)
        except BaseException:
            self.terminate()
            raise

    @property
    def size(self) -> int:
        return len(self.procs)

    @property
    def pids(self) -> list[int]:
        return [int(p.pid) for p in self.procs]

    def _lost(self, k: int, detail: str, traceback_text: str = "") -> SensorSvcError:
        """渲染进程 k 失败时的错误：GPU 丢失（回溯带设备丢失特征，或本应有 GPU 而设备节点已打不开）为 647，其余为 648。
        647 的 `exitcode` 为负时即被信号终止（EGL 段错误为 -11）。"""
        proc = self.procs[k]
        if proc.exitcode is None:                                      # the pipe closes before the child is reaped: wait for it
            proc.join(2.0)
        code = proc.exitcode
        if is_gpu_lost_message(traceback_text):
            return gpu_lost(f"render proc {k} {detail}", proc=k, exitcode=code)
        if gpu_expected():
            nodes = check_device_nodes()
            if not nodes["ok"] or not nodes["nodes"]:
                return gpu_lost(f"render proc {k} {detail} (exitcode {code}); NVIDIA device nodes lost: {nodes['lost'] or 'none present'}",
                                proc=k, exitcode=code)
        return SensorSvcError(WORKER_LOST, f"render proc {k} {detail}" + ("" if traceback_text else f" (exitcode {code})"), proc=k)

    def _recv(self, k: int, ch: int, timeout_s: float | None) -> tuple[str, Any]:
        c = self.conns[k][ch]
        if timeout_s is not None and not c.poll(timeout_s):
            raise self._lost(k, "timed out" if self.procs[k].is_alive() else "exited")
        try:
            st, val = c.recv()
        except (EOFError, OSError) as e:
            raise self._lost(k, f"lost: {e!r}") from e
        if st == "error":
            raise self._lost(k, f"failed:\n{val}", str(val))
        return st, val

    def send(self, k: int, msg: tuple, ch: int = 0) -> None:
        try:
            self.conns[k][ch].send(msg)
        except (OSError, BrokenPipeError) as e:
            raise self._lost(k, f"lost: {e!r}") from e

    def recv(self, k: int, ch: int = 0, timeout_s: float | None = None) -> Any:
        return self._recv(k, ch, timeout_s)[1]

    def request(self, k: int, msg: tuple, ch: int = 0, timeout_s: float | None = None) -> Any:
        self.send(k, msg, ch)
        return self.recv(k, ch, timeout_s)

    def scatter(self, msgs: dict[int, tuple], ch: int = 0, timeout_s: float | None = None) -> dict[int, Any]:
        """向若干进程同时发出请求（先全部发出、再按进程序号收取），返回 {进程: 回复}。"""
        for k, m in msgs.items():
            self.send(k, m, ch)
        return {k: self.recv(k, ch, timeout_s) for k in msgs}

    def close(self, timeout_s: float = 120.0) -> list[Any]:
        """依次关闭各渲染进程（`handler.close()` 的返回值按进程序号给出）；超时或失联的进程强制结束并返回 None。"""
        if self._closed:
            return []
        self._closed = True
        out: list[Any] = []
        for k in range(len(self.procs)):
            try:
                out.append(self.request(k, ("close",), 0, timeout_s))
            except SensorSvcError:
                out.append(None)
        for p in self.procs:
            p.join(timeout_s)
        self.terminate()
        return out

    def terminate(self) -> None:
        self._closed = True
        for p in self.procs:
            if p.is_alive():
                p.terminate()
                p.join(10)
            if p.is_alive():
                p.kill()
        for cs in self.conns:
            for c in cs:
                with contextlib.suppress(OSError):
                    c.close()

    def __enter__(self) -> RenderGroup:
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.close()
        else:
            self.terminate()
