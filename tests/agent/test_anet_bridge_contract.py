"""M14-AC-037：ANet bridge 接口桩（FakeDaemon 契约：register、find、delegate、results、end、evidence 映射；wire 不一致 484；
生成的 daemon 配置不含 auto-reply 与 shell、入站为白名单、hub 只允许回环或局域网）。"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from fakes.fake_daemon import FakeDaemon, FakeHub
from fakes.stub_provider import StubProvider

from awr.agent.anet_bridge.config_gen import ConfigError, check_config, daemon_config, hub_url_ok, load_versions
from awr.agent.anet_bridge.network import AnetDaemonNetwork, AnetError, HttpControlPlane
from awr.agent.anet_mock import identity as ID
from awr.agent.anet_mock.network import MockNetwork
from awr.agent.runtime.effect import completed_ok
from awr.agent.runtime.network import AgentNetwork
from awr.agent.runtime.types import EffectStatus


def _net(wire: str = "1"):
    hub = FakeHub()
    coord = ID.coordinator_aid("w")
    b1 = StubProvider("b1", 2, ["thermal.imaging"])
    planes = {coord: FakeDaemon(hub, coord, wire=wire), b1.aid: FakeDaemon(hub, b1.aid, wire=wire)}
    return AnetDaemonNetwork(planes, coordinator_aid=coord), planes, coord, b1


def test_mapping_register_find_delegate_results_end_evidence() -> None:
    net, planes, coord, b1 = _net()

    async def main():
        await net.check_wire()
        await net.register(b1)
        found = await net.find(coord, "thermal.*")
        ix = await net.delegate(coord, b1.aid, "thermal.imaging", {"target_enu_m": [0, 0, None]}, task_id="T-0001")
        res = await net.result(coord, ix, 1.0)
        await net.cancel(coord, ix)
        await net.evidence(coord, {"type": "agent.task.accepted"})
        miss = await net.result(coord, await net.delegate(coord, b1.aid, "lidar.mapping", {}, task_id="T-2"), 1.0)
        return found, res, miss

    found, res, miss = asyncio.run(main())
    assert [v.aid for v in found] == [b1.aid]
    # service 模块效果封顶 V1、非仿真：不能 completed（FR-071：V1.0 需请求方读回证据）
    assert res.effect.status is EffectStatus.OK and res.effect.verify_trust == 1 and not res.effect.simulated
    assert not completed_ok(res.effect, True, True)
    assert miss.effect.status is EffectStatus.UNAVAILABLE
    routes = planes[coord].routes + planes[b1.aid].routes
    for r in ("/version", "/hub-register", "/find", "/delegate", "/results", "/end", "/evidence"):
        assert r in routes, r


def test_wire_mismatch_484() -> None:
    net, *_ = _net(wire="2")
    with pytest.raises(AnetError) as ei:
        asyncio.run(net.check_wire())
    assert ei.value.code == 484


def test_interface_signatures_match_mock() -> None:
    """AgentNetwork 在 Mock 与真 ANet 两个实现中签名一致（NFR-014：换接时 TaskManager、Allocator 零改动）。"""
    for name in ("register", "unregister", "find", "view", "delegate", "updates", "result", "cancel"):
        want = inspect.signature(getattr(AgentNetwork, name))
        for impl in (MockNetwork, AnetDaemonNetwork):
            got = inspect.signature(getattr(impl, name))
            assert list(got.parameters) == list(want.parameters), (impl.__name__, name)


def test_daemon_config_safety() -> None:
    cfg = daemon_config(world_id="newyork", vehicle_id="p600-b1", aid_hint=ID.aid("newyork", "p600-b1"),
                        capabilities=["thermal.imaging", "task.quote"], index=1, fleet_aids=[ID.coordinator_aid("newyork")],
                        long_running=["thermal.imaging"])
    assert check_config(cfg) == []
    assert cfg["auto_reply"]["enabled"] is False and "shell" not in cfg["build"]["tags"]
    assert cfg["inbound"]["policy"] == "closed" and cfg["inbound"]["allow"]
    assert cfg["data_dir"].endswith("runs/.anet/newyork/p600-b1") and cfg["control"]["listen"] == "127.0.0.1:39812"
    bad = {**cfg, "auto_reply": {"enabled": True}, "build": {"tags": ["shell"]}, "inbound": {"policy": "open", "allow": []}}
    assert len(check_config(bad)) == 3
    with pytest.raises(ConfigError):
        daemon_config(world_id="w", vehicle_id="v", aid_hint="a", capabilities=[], index=0, fleet_aids=[],
                      hub_url="http://8.8.8.8:18088")
    assert hub_url_ok("http://192.168.1.5:18088") and not hub_url_ok("http://example.com:18088")
    with pytest.raises(AnetError):
        HttpControlPlane("http://10.0.0.1:39811")


@pytest.mark.parametrize("base", [
    "http://127.0.0.1:1@evil.example:80",  # 前缀相符，主机是 evil.example
    "http://127.0.0.1:39811@evil.example/",
    "http://[::1]:1@evil.example",
    "http://127.0.0.1.evil.example:1",
    "http://user:pw@127.0.0.1:39811",
    "https://127.0.0.1:39811",
    "http://127.0.0.1",
    "http://127.0.0.1:39811/x",
    "http://127.0.0.1:39811?u=1",
    "http://127.0.0.1:99999",
])
def test_control_plane_rejects_non_loopback(base: str) -> None:
    with pytest.raises(AnetError):
        HttpControlPlane(base)


@pytest.mark.parametrize("base", ["http://127.0.0.1:39811", "http://127.0.0.1:39811/", "http://[::1]:39811"])
def test_control_plane_accepts_loopback(base: str) -> None:
    assert HttpControlPlane(base).base == base.rstrip("/")


def test_versions_lock() -> None:
    v = load_versions()
    assert v["wire"] == "1" and v["anetcore"] == "v0.14.0" and v["anet"] and v["anethub"]
