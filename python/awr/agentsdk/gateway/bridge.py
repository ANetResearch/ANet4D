"""一次只有一个决策在途的观测 / 动作交换；仿真在另一个线程独立推进。

`complete(system, history, turn)` 在仿真侧（指挥者线程）调用：登记一个待决观测并阻塞到外部提交动作、会话取消或回合结束；
`observation(after, timeout)` 与 `submit(sequence, action)` 在 HTTP 侧调用。提交先经 `pre_validate(sequence, action)`（任务层钩子，
如调用账本：被拒绝的动作也是计费调用产生的，须在动作校验之前入账），再经 `validate(action)`（缺省 `validate_decision`，词表 +
任务层报告校验），最后核对序号（过期、重复或没有待决观测 → `RuntimeError`，HTTP 409）。
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace

from awr.agentsdk.skills import validate_decision

__all__ = ["DecisionBridge", "SessionCancelled"]


class SessionCancelled(BaseException):
    """不是 `Exception`：供应者的重试逻辑不能把取消吞掉。"""


class DecisionBridge:
    name = "external"
    model = "http-agent"

    def __init__(self, uavs=None, *, validate: Callable[[dict], dict] | None = None, reports: Callable[[dict], None] | None = None) -> None:
        self.uavs = None if uavs is None else set(uavs)
        self._validate = validate or (lambda a: validate_decision(a, uavs=self.uavs, reports=reports))
        self.cv = threading.Condition()
        self.seq = 0
        self.pending = None
        self.reply = None
        self.cancelled = False
        self.finished = False
        self.result = None
        self.world_time_s = 0.0
        self.world_ended = False
        self.inputs_closed = False
        self.usage = SimpleNamespace(input_tokens=0, output_tokens=0, calls=0, wall_s=0.0)

    # ------------------------------------------------------------------ simulator side
    def complete(self, system, history, turn):
        started = time.monotonic()
        with self.cv:
            if self.cancelled or self.inputs_closed:
                raise SessionCancelled()
            self.seq += 1
            self.pending = dict(sequence=self.seq, system=system, history=history,
                                sensor_metadata=getattr(turn, "sensor_metadata", {}),
                                observation=turn.text,
                                images=[dict(label=label, mime_type="image/jpeg", data_base64=base64.b64encode(data).decode())
                                        for label, data in turn.images],
                                clock_policy="realtime", observation_time_s=getattr(turn, "episode_s", self.world_time_s))
            self.reply = None
            self.cv.notify_all()
            self.cv.wait_for(lambda: self.reply is not None or self.cancelled or self.inputs_closed or self.finished)
            if self.cancelled or self.inputs_closed or (self.finished and self.reply is None):
                raise SessionCancelled()
            reply = self.reply
            self.pending = None
            self.reply = None
            self.usage.calls += 1
            self.usage.wall_s += time.monotonic() - started
            return json.dumps(reply, ensure_ascii=False, allow_nan=False)

    def advance_clock(self, t_s, world_ended=False):
        with self.cv:
            self.world_time_s = float(t_s)
            self.world_ended = bool(world_ended)

    def end_episode(self):
        with self.cv:
            self.inputs_closed = True
            self.cv.notify_all()

    def finish(self, result):
        with self.cv:
            self.finished = True
            self.result = result
            self.cv.notify_all()

    # ------------------------------------------------------------------ HTTP side
    def observation(self, after=0, timeout=15.0):
        with self.cv:
            self.cv.wait_for(lambda: self.cancelled or self.finished
                             or (self.pending is not None and self.pending["sequence"] > after and self.reply is None), timeout)
            if self.cancelled:
                return dict(status="cancelled")
            if self.finished:
                return dict(status="finished", result=self.result)
            if self.pending is not None and self.pending["sequence"] > after and self.reply is None:
                return dict(status="awaiting_action", world_time_s=self.world_time_s, world_ended=self.world_ended, **self.pending)
            return dict(status="running", world_time_s=self.world_time_s, world_ended=self.world_ended)

    def submit(self, sequence, action):
        with self.cv:
            return self._submit_locked(sequence, action)

    def _submit_locked(self, sequence, action):
        # Validate before consumption so a malformed response can be corrected.
        if not isinstance(action, dict):
            raise ValueError("action must be an object")
        action = self.pre_validate(sequence, dict(action))
        action = self._validate(action)
        self.check_fresh(sequence)
        self.reply = action
        self.cv.notify_all()

    def pre_validate(self, sequence, action: dict) -> dict:
        """任务层钩子（持锁调用）：在动作校验之前处理并返回动作，如取出 `_usage` 账本记录并入账。缺省原样返回。"""
        return action

    def check_fresh(self, sequence) -> None:
        if self.cancelled or self.finished or self.inputs_closed or self.pending is None or sequence != self.seq or self.reply is not None:
            raise RuntimeError("stale, duplicate or unavailable decision sequence")

    def cancel(self):
        with self.cv:
            self.cancelled = True
            self.cv.notify_all()
