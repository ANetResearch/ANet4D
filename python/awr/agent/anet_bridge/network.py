"""AnetDaemonNetwork（`AgentNetwork` 的真 ANet 实现；M14 §6.15；D1 为桩，V1.0 实现；M14-FR-047、FR-067、FR-070）。

映射（d05 §4.2；v0.1 wire 1）：register → `POST /hub-register`；find → `POST /find`；delegate → `POST /delegate`；
result → 轮询 `POST /results`（按 `interaction_id` 过滤，读 `receipt_verified`）；cancel → `POST /end`；证据 → `POST /evidence`。
v0.2 起改用 `/tasks/send`、`/tasks/get`、`/tasks/wait`、`/tasks/cancel`，接口签名不变。启动时比对 `X-ANet-Wire` 与
`tools/anet/versions.lock`，不一致即拒绝（484 ANET_WIRE_MISMATCH）；daemon 不可达时该 AID 不健康（485）。

两段式（v0.1 `service` 模块无 LongRunning，daemon 60 s 截断）：第一段立即返回 `accepted = 1`、`eta_s`、`task_ref`；第二段由
请求方委派 `task.result` 取回（每 5 s【墙钟】）。`service` 模块效果封顶 V1（FR-071：completed 需请求方读回证据）。

控制面传输抽象为 `ControlPlane.post(path, body) -> dict`：真网为回环 HTTP（`HttpControlPlane`，只允许 127.0.0.1），
契约测试用 FakeDaemon。
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol
from urllib.parse import urlsplit

from ..anet_mock.identity import aid_short
from ..runtime.network import AgentView, TaskResult
from ..runtime.types import Effect, EffectStatus, Update

__all__ = ["WIRE_VERSION", "AnetDaemonNetwork", "AnetError", "ControlPlane", "HttpControlPlane"]

WIRE_VERSION = "1"
RESULT_POLL_S = 5.0  # V1.0 task.result 轮询【墙钟】
SERVICE_TRUST_CAP = 1  # `service` 模块效果封顶 V1（d05 §0 第 6 条）


class AnetError(Exception):
    def __init__(self, code: int, msg: str = "") -> None:
        super().__init__(f"{code} {msg}")
        self.code = code


class ControlPlane(Protocol):
    async def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]: ...

    async def wire(self) -> str: ...


def _loopback_base(base: str) -> bool:
    """控制面地址只能是 `http://127.0.0.1:端口` 或 `http://[::1]:端口`。按解析后的主机判断，不按前缀：
    `http://127.0.0.1:1@evil.example/` 前缀相符，真正的主机却是 evil.example，Bearer 令牌会发到那里。"""
    try:
        u = urlsplit(base)
        port = u.port
    except ValueError:
        return False
    return (u.scheme == "http" and u.hostname in ("127.0.0.1", "::1") and port is not None
            and u.username is None and u.password is None and u.path in ("", "/") and not u.query and not u.fragment)


class HttpControlPlane:
    """daemon 控制面（回环 HTTP，`control_token.txt` 作为 Bearer）。V1.0 使用；D1 不启动 daemon。"""

    def __init__(self, base: str, token: str = "", timeout_s: float = 2.0) -> None:
        if not _loopback_base(base):
            raise AnetError(485, "control plane must be loopback")
        self.base = base.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def _req(self, path: str, body: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, str]]:
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST",
                                   headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(r, timeout=self.timeout_s) as resp:
            return json.loads(resp.read() or b"{}"), dict(resp.headers)

    async def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        return (await asyncio.to_thread(self._req, path, body))[0]

    async def wire(self) -> str:
        _b, h = await asyncio.to_thread(self._req, "/version", None)
        return str(h.get("X-ANet-Wire", ""))


class AnetDaemonNetwork:
    """每 AID 一个 daemon；协调者经自己的 daemon 发起 find 与 delegate。"""

    network = "anet"

    def __init__(self, planes: Mapping[str, ControlPlane], *, coordinator_aid: str, wire_expected: str = WIRE_VERSION) -> None:
        self.planes = dict(planes)
        self.coordinator_aid = coordinator_aid
        self.wire_expected = wire_expected
        self.views: dict[str, AgentView] = {}
        self._pending: dict[str, dict[str, Any]] = {}

    def _plane(self, aid: str) -> ControlPlane:
        p = self.planes.get(aid) or self.planes.get(self.coordinator_aid)
        if p is None:
            raise AnetError(485, f"no daemon for {aid}")
        return p

    async def check_wire(self) -> None:
        for aid, p in self.planes.items():
            w = await p.wire()
            if w != self.wire_expected:
                raise AnetError(484, f"{aid} wire {w!r} != {self.wire_expected!r}")

    async def register(self, agent: Any) -> str:
        caps = list(agent.capabilities())
        await self._plane(agent.aid).post("/hub-register", {"aid": agent.aid, "name": getattr(agent, "name", ""),
                                                            "capabilities": caps})
        self.views[agent.aid] = AgentView(agent.aid, aid_short(agent.aid), int(getattr(agent, "agent_no", 0)),
                                          str(getattr(agent, "vehicle_id", "")), str(getattr(agent, "name", "")), "uav",
                                          str(getattr(agent, "profile_id", "")),
                                          tuple(c for c in caps if not c.startswith(("agent.", "task.", "blackboard."))), "anet")
        return agent.aid

    async def unregister(self, aid: str) -> None:
        self.views.pop(aid, None)

    async def find(self, requester: str, pattern: str, *, t0_ns: int | None = None) -> list[AgentView]:
        rep = await self._plane(requester).post("/find", {"capability": pattern})
        out = []
        for it in rep.get("agents") or []:
            aid = str(it.get("aid"))
            v = self.views.get(aid) or AgentView(aid, aid_short(aid), int(it.get("agent_no", 0)), "", str(it.get("name", "")),
                                                 caps=tuple(it.get("capabilities") or ()), network="anet")
            out.append(v)
        return sorted(out, key=lambda v: v.agent_no)

    def view(self, aid: str) -> AgentView | None:
        return self.views.get(aid)

    async def delegate(self, requester: str, provider: str, capability: str, args: Mapping[str, Any], *, task_id: str,
                       attempt: int | None = None) -> str:
        rep = await self._plane(requester).post("/delegate", {
            "provider": provider, "task_doc": {"capability": capability, "args": dict(args),
                                               "requires": [{"id": capability, "type": "capability", "necessity": "must"}]},
            "task_id": task_id})
        ix = str(rep.get("interaction_id") or "")
        if not ix:
            raise AnetError(476, "delegate returned no interaction id")
        self._pending[ix] = {"requester": requester, "provider": provider}
        return ix

    async def _poll(self, requester: str, ix: str) -> dict[str, Any] | None:
        rep = await self._plane(requester).post("/results", {"interaction_id": ix})
        items = [r for r in rep.get("results") or [] if r.get("interaction_id") == ix]
        return items[-1] if items else None

    @staticmethod
    def _effect(r: Mapping[str, Any]) -> Effect:
        d = dict(r.get("deliverable") or {})
        e = Effect.from_dict({"status": d.get("status", "FAILED"), **{k: v for k, v in d.items() if k != "status"}})
        if e.verify_trust > SERVICE_TRUST_CAP or e.simulated:
            e = Effect(e.status, min(e.verify_trust, SERVICE_TRUST_CAP), e.auth_trust, False, e.native_ack, e.protocol, e.requested,
                       e.observed_state, e.latency_ms, e.quirk, e.message, e.metrics, e.artifacts)
        return e

    async def updates(self, requester: str, ix: str) -> AsyncIterator[Update]:
        res = await self.result(requester, ix, 60.0)
        yield Update(ix, "final", res.effect, receipt=res.receipt, receipt_ok=res.receipt_ok)

    async def result(self, requester: str, ix: str, timeout_s: float) -> TaskResult:
        end = time.monotonic() + timeout_s
        while True:
            r = await self._poll(requester, ix)
            if r is not None and r.get("status") in ("completed", "failed", "rejected", "canceled"):
                return TaskResult(ix, self._effect(r), r.get("receipt"), bool(r.get("receipt_verified")), None)
            if time.monotonic() >= end:
                return TaskResult(ix, Effect(EffectStatus.UNVERIFIED, message="result timeout"), None, False, None)
            await asyncio.sleep(min(RESULT_POLL_S, max(0.0, end - time.monotonic())))

    async def cancel(self, requester: str, ix: str) -> None:
        await self._plane(requester).post("/end", {"interaction_id": ix})

    async def evidence(self, aid: str, record: Mapping[str, Any]) -> None:
        await self._plane(aid).post("/evidence", dict(record))
