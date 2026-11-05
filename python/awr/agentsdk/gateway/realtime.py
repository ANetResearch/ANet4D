"""墙钟配速与单个异步决策，与物理档无关。

`RealtimeClock.pace(episode_s)`：仿真不比墙钟快（睡到该 tick 的墙钟时刻），落后超过 `max_lag_s` 抛 `RealtimeOverrun`：基础设施
失败，不能把世界悄悄放慢再记成智能体失败。`DecisionWorker` 只在工作线程里调用策略；仿真状态的全部修改留在仿真线程。
"""

from __future__ import annotations

import threading
import time

__all__ = ["DecisionWorker", "RealtimeClock", "RealtimeOverrun"]


class RealtimeOverrun(RuntimeError):
    pass


class RealtimeClock:
    def __init__(self, max_lag_s=2.0, *, now=time.monotonic, sleep=time.sleep):
        self.now, self.sleep = now, sleep
        self.start = now()
        self.max_lag_s = float(max_lag_s)
        self.peak_lag_s = 0.0

    def pace(self, episode_s):
        delay = self.start + episode_s - self.now()
        if delay > 0:
            self.sleep(delay)
        elapsed = self.now() - self.start
        lag = max(0.0, elapsed - episode_s)
        self.peak_lag_s = max(self.peak_lag_s, lag)
        if lag > self.max_lag_s:
            raise RealtimeOverrun(f"Render/physics clock lag {lag:.3f}s exceeds {self.max_lag_s:.3f}s; invalid infrastructure run")
        return elapsed, lag


class DecisionWorker:
    """The worker only calls the policy; all simulator mutations remain on its thread."""

    def __init__(self, name: str = "anet4d-policy"):
        self.name = name
        self.thread = None
        self.result = None
        self.error = None
        self.context = None

    @property
    def busy(self):
        return self.thread is not None

    @property
    def ready(self):
        return self.thread is not None and not self.thread.is_alive()

    def start(self, fn, context):
        if self.busy:
            raise RuntimeError("Only one policy call may be in flight")
        self.context, self.result, self.error = context, None, None

        def work():
            try:
                self.result = fn()
            except BaseException as e:
                self.error = e

        self.thread = threading.Thread(target=work, name=self.name, daemon=True)
        self.thread.start()

    def take(self):
        if not self.ready:
            return None
        context, result, error = self.context, self.result, self.error
        self.thread = None
        self.context = None
        if error is not None:
            raise error
        return context, result

    def join(self, timeout):
        if self.thread is not None:
            self.thread.join(timeout)
