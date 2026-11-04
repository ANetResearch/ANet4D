"""tests/rt 参考客户端（awr.rt.v1）：握手、自动 ack、BATCH 与 TIME 解码、控制消息 schema 校验、msgpack 载荷校验。

只用生成的 `awr.contracts.frame` 与 `layouts` 解码（与前端 rt.worker 同一契约，不手写格式串）。
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

from awr.contracts import CONTRACTS_VERSION
from awr.contracts import frame as F
from awr.contracts.layouts import DRONE_STATE64, SWARM_LITE32

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "packages" / "contracts"
WORLD = "shenzhen"
OPS = "rt/ops.schema.json"
S2C = "#/$defs/serverToClient"


def world_ready() -> bool:
    return (ROOT / "worlds" / WORLD / "world.json").exists()


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@cache
def _registry():
    from referencing import Registry, Resource

    reg = Registry()
    for p in sorted(CONTRACTS.rglob("*.schema.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "$id" in d:
            reg = reg.with_resource(d["$id"], Resource.from_contents(d))
    return reg


@cache
def validator(rel: str, ref: str | None = None):
    from jsonschema import Draft202012Validator

    d = json.loads((CONTRACTS / rel).read_text(encoding="utf-8"))
    schema = {"$ref": f"{d['$id']}{ref}"} if ref else d
    return Draft202012Validator(schema, registry=_registry())


def schema_errors(rel: str, inst: Any, ref: str | None = None) -> list[str]:
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}" for e in validator(rel, ref).iter_errors(inst)]


def ops_errors(msg: dict) -> list[str]:
    return schema_errors(OPS, msg, S2C)


@dataclass
class Batch:
    header: F.BatchHeader
    records: list[F.Record]
    buf: bytes

    def payload(self, r: F.Record) -> bytes:
        return self.buf[r.payload_off:r.payload_off + r.length]

    def by_channel(self, cid: int) -> F.Record | None:
        return next((r for r in self.records if r.channel_id == cid), None)


@dataclass
class Client:
    ws: Any
    texts: list[dict] = field(default_factory=list)
    batches: list[Batch] = field(default_factory=list)
    times: list[Any] = field(default_factory=list)
    channels: dict[int, dict] = field(default_factory=dict)
    auto_ack: bool = True
    pending: list[tuple[str, Any]] = field(default_factory=list)      # messages settle() took in ahead of time, replayed by recv()

    @property
    def topic_ids(self) -> dict[str, int]:
        return {c["topic"]: cid for cid, c in self.channels.items()}

    async def recv(self, timeout: float = 5.0) -> tuple[str, Any]:
        if self.pending:
            return self.pending.pop(0)
        m = await asyncio.wait_for(self.ws.recv(), timeout)
        if isinstance(m, bytes):
            if m[0] == F.OP_TIME:
                t = F.decode_time(m)
                self.times.append(t)
                return "time", t
            h = F.decode_batch_header(m)
            b = Batch(h, list(F.iter_records(m)), bytes(m))
            self.batches.append(b)
            if self.auto_ack:
                await self.send({"op": "ack", "frame": h.frame_seq})
            return "batch", b
        j = json.loads(m)
        self.texts.append(j)
        if j["op"] == "advertise":
            for c in j["channels"]:
                self.channels[c["id"]] = c
        elif j["op"] == "unadvertise":
            for i in j["ids"]:
                self.channels.pop(i, None)
        return "json", j

    async def until(self, pred, timeout: float = 10.0) -> tuple[str, Any]:
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                raise TimeoutError("等待超时")
            kind, x = await self.recv(left)
            if pred(kind, x):
                return kind, x

    async def settle(self, quiet_s: float = 0.3, max_s: float = 3.0) -> None:
        """接收到 advertise 停止（quiet_s 内没有新的 advertise）为止，最多 max_s：网关按车辆到达逐批 advertise，慢机器上
        刚握手完 `topic_ids` 可能还不全（托管 CI 的 4 核 runner）。其间收到的其他消息排进 `pending`，之后的 recv() 先按原序
        返回它们，所以等某条消息的测试不受影响。"""
        end = time.monotonic() + max_s
        last = time.monotonic()
        taken: list[tuple[str, Any]] = []
        while time.monotonic() < end:
            try:
                kind, x = await self.recv(max(0.0, min(quiet_s - (time.monotonic() - last), end - time.monotonic())))
            except TimeoutError:
                break
            if kind == "json" and x["op"] == "advertise":
                last = time.monotonic()
            else:
                taken.append((kind, x))
        self.pending.extend(taken)

    async def drain(self, seconds: float) -> None:
        """持续接收（自动 ack）seconds 秒。"""
        end = time.monotonic() + seconds
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            try:
                await self.recv(left)
            except TimeoutError:
                return

    async def send(self, obj: dict) -> None:
        await self.ws.send(json.dumps(obj, separators=(",", ":")))

    async def handshake(self) -> tuple[dict, dict, Any]:
        _, info = await self.recv()
        _, adv = await self.recv()
        kind, t = await self.recv()
        assert info["op"] == "serverInfo" and adv["op"] == "advertise" and kind == "time", (info, adv, kind)
        return info, adv, t

    async def hello(self, *, resume: dict | None = None, contracts: str = CONTRACTS_VERSION) -> None:
        m: dict[str, Any] = {"op": "hello", "client": "tests-rt/0.1", "contracts": contracts}
        if resume is not None:
            m["resume"] = resume
        await self.send(m)

    async def result(self, cid: str, *, final: bool = True, timeout: float = 30.0) -> dict:
        _, r = await self.until(lambda k, x: k == "json" and x["op"] == "result" and x["id"] == cid
                                and (bool(x.get("final")) or not final), timeout)
        return r

    def results(self, cid: str) -> list[dict]:
        return [m for m in self.texts if m["op"] in ("result", "progress") and m.get("id") == cid]

    def events(self) -> list[dict]:
        out: list[dict] = []
        for m in self.texts:
            if m["op"] == "event":
                out.append(m)
            elif m["op"] == "events":
                out.extend(m["items"])
        return out

    def latest_full(self, cid: int) -> np.ndarray | None:
        for b in reversed(self.batches):
            r = b.by_channel(cid)
            if r is not None:
                return np.frombuffer(b.payload(r), DRONE_STATE64)
        return None

    def latest_lite(self, cid: int) -> np.ndarray | None:
        for b in reversed(self.batches):
            r = b.by_channel(cid)
            if r is not None:
                return np.frombuffer(b.payload(r), SWARM_LITE32)
        return None

    def latest_msgpack(self, cid: int) -> Any:
        for b in reversed(self.batches):
            r = b.by_channel(cid)
            if r is not None:
                return msgpack.unpackb(b.payload(r), raw=False, strict_map_key=False)
        return None


# ---------------------------------------------------------------- 连接辅助（GwStack 与进程内栈共用）
PROTO = "awr.rt.v1"


def token(base: str, role: str = "operator", hint: str | None = None, **kw: Any) -> dict:
    import httpx

    body: dict[str, Any] = {"role": role, **kw}
    if hint is not None:
        body["principal_hint"] = hint
    r = httpx.post(f"{base}/api/auth/token", json=body, timeout=10)
    assert r.status_code == 200, r.text
    return r.json()


async def open_client(stack: Any, tok: str | None, *, hello: bool = True, auto_ack: bool = True,
                      origin: str | None = None, resume: dict | None = None) -> Client:
    from websockets.asyncio.client import connect

    protos = [PROTO] + ([f"bearer.{tok}"] if tok else [])
    ws = await connect(stack.ws_url, subprotocols=protos, origin=origin or stack.origin, max_size=None,
                       compression=None, open_timeout=10)
    c = Client(ws, auto_ack=auto_ack)
    await c.handshake()
    if hello:
        await c.hello(resume=resume)
        await c.settle()
    return c


def hint_of(name: str) -> str:
    """测试用固定 principal_hint（16–64 位 base32）。"""
    s = "".join(ch for ch in name.upper() if ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
    return (s + "TESTPRINCIPALHINTAAAAAAAA")[:24]
