"""Env 物理档选择（ADR-226，5e）：`realistic` 是统一 L2 引擎（运行清单 `tier = l2`），`l2` 为其旧名，`gz` 是 gz 锁步交叉验证档，
未知档名 ValueError；`validate_clock` 对 realistic、gz、sitl 要求 d1_250。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from awr.agentsdk.env.core import BenchEnv
from awr.agentsdk.env.spec import EnvSpec, FleetEntry
from awr.phys import PhysError
from awr.phys.adapter import validate_clock
from awr.phys.fast import clock_of
from awr.phys.l2 import L2Tier

FLEET = (FleetEntry("uav-01", (-10.0, 50.0, 0.0)),)


def _env(w: Path, ev: Path, out: Path | None, **kw) -> BenchEnv:
    return BenchEnv(EnvSpec("t-tier", 5, w, ev, FLEET, duration_s=1.0, out_dir=out, **kw))


@pytest.mark.parametrize("physics", ["realistic", "l2"])
def test_realistic_runs_on_the_unified_l2_engine(synth_event, tmp_path: Path, physics: str) -> None:
    w, ev = synth_event
    env = _env(w, ev, tmp_path / physics, physics=physics)
    tier = env._make_tier()
    assert isinstance(tier, L2Tier) and tier.name == "l2"
    env.reset()
    while env.agents:
        env.step({})
    info = env.close()
    assert info["physics"] == physics and info["tier"] == "l2"
    man = json.loads((tmp_path / physics / "manifest.json").read_text())
    assert man["physics"] == physics and man["tier"] == "l2"


def test_gz_is_a_separate_tier_and_unknown_names_are_rejected(synth_event, tmp_path: Path) -> None:
    w, ev = synth_event
    with pytest.raises(ValueError, match="not available"):
        _env(w, ev, None, physics="nope")._make_tier()
    with pytest.raises(PhysError) as e:                                                   # gz keeps the formal clock rule (637)
        _env(w, ev, None, physics="gz", clock_profile="sandbox100")._make_tier()
    assert e.value.code == 637
    with pytest.raises(PhysError) as e:
        _env(w, ev, None, physics="realistic", clock_profile="sandbox100")._make_tier()
    assert e.value.code == 637
    assert isinstance(_env(w, ev, None, physics="l2", clock_profile="sandbox100")._make_tier(), L2Tier)   # the alias has no formal clock rule


def test_clock_rule_names_the_250_hz_tiers() -> None:
    cp = clock_of("sandbox100")
    for tier in ("realistic", "gz", "sitl"):
        with pytest.raises(PhysError) as e:
            validate_clock(tier, cp, 100)
        assert e.value.code == 637 and "realistic, gz and sitl" in str(e.value)
    validate_clock("fast", cp, 100)
