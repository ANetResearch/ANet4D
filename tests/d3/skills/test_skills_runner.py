"""单机技能执行器（自 CHIMERA `tests/test_skill_terrain.py` 移入，断言改为 `bench/act.v1` 的动作 dict）：贴地形航段、rtl 的
包络下限、follow 的 standoff、orbit 相位不重复积分；动作构造与研究者 SDK `anet4d.agent` 的构造函数输出相同。"""

from __future__ import annotations

import numpy as np

from awr.agentsdk.schemas import bench_validator
from awr.agentsdk.skills import actions
from awr.agentsdk.skills.runner import CRUISE_MPS, SkillRunner, footprint_w, lawnmower


def goto(x, y, z, v=CRUISE_MPS):
    return {"mode": "goto", "pos_enu_m": [float(x), float(y), float(z)], "speed_mps": float(v)}


def look(cam, x, y, z):
    return {"sensor_id": cam, "mode": "look_at", "target_enu_m": [float(x), float(y), float(z)]}


def obs(x, y, z):
    return {"nav": {"pos_enu_m": [x, y, z]}}


def test_goto_reaches_requested_xy_and_altitude_before_done():
    r = SkillRunner("u", "cam", (30, 120), (0, 0), terrain=lambda x, y: 10.0)
    assert r.set("goto", dict(x=100, y=0, agl=40), 0, (0, 0)) is None
    assert r.state.waypoints == [(100.0, 0.0)]
    nav, gim = r.tick(obs(100, 0, 70), 10, 5)
    assert nav == goto(100, 0, 50) and gim == look("cam", 100, 0, 10) and not r.state.done
    r.tick(obs(100, 0, 50), 10, 10)
    assert r.state.done
    nav, _ = r.tick(obs(100, 0, 50), 10, 11)
    assert nav == {"mode": "hover"}


def test_long_legs_follow_terrain_25m_ahead():
    # A leg over rising ground aims 25 m ahead at the terrain height there, so the vehicle climbs with the ground
    # instead of descending toward the far waypoint.
    r = SkillRunner("u", "cam", (30, 120), (0, 0), terrain=lambda x, y: 0.5 * x)
    r.set("goto", dict(x=200, y=0, agl=40), 0, (0, 0))
    nav, _ = r.tick(obs(0, 0, 40), 0, 0)
    assert nav == goto(25, 0, 12.5 + 40)
    r.set("orbit", dict(x=300, y=0, radius=20, period=100, agl=40), 0, (0, 0))
    nav, _ = r.tick(obs(0, 0, 40), 0, 0)
    assert abs(nav["pos_enu_m"][0] - 25.0) < 1e-2 and abs(nav["pos_enu_m"][2] - (0.5 * 25 + 40)) < 0.1


def test_rtl_leg_follows_terrain_at_the_envelope_floor():
    # The return leg is terrain-following too: aimed straight at the pad height, a drone 200 m out over ground 100 m
    # higher than the pad would sink below the 30 m floor long before the pad.
    r = SkillRunner("u", "cam", (30, 120), (0, 0), terrain=lambda x, y: 0.5 * x)
    r.set("rtl", {}, 0, (200, 0))
    nav, gim = r.tick(obs(200, 0, 140), 0, 0)
    assert nav == goto(175, 0, 0.5 * 175 + 32) and gim is None
    nav, _ = r.tick(obs(10, 0, 37), 0, 0)
    assert nav == goto(0, 0, 32) and not r.state.done
    nav, _ = r.tick(obs(1, 0, 32), 0, 0)
    assert nav == {"mode": "land"} and r.state.done
    assert r.tick(obs(1, 0, 32), 0, 0) == (None, None)        # landing in progress: the command is not re-issued


def test_aim_and_waypoint_use_destination_elevation():
    r = SkillRunner("u", "cam", (30, 120), (0, 0), terrain=lambda x, y: 0.2 * x + 0.1 * y)
    r.set("follow", dict(x=100, y=0, agl=40, standoff=20), 0, (0, 0))
    nav, gim = r.tick(obs(0, 0, 40), 0, 0)
    assert gim == look("cam", 100, 0, 20) and nav == goto(25, 0, 0.2 * 25 + 40)
    nav, _ = r.tick(obs(60, 0, 52), 0, 0)
    assert nav == goto(80, 0, 56)


def test_orbit_phase_is_not_integrated_twice():
    r = SkillRunner("u", "cam", (30, 120), (0, 0))
    r.set("orbit", dict(x=0, y=0, radius=20, period=100), 0, (20, 0))
    first = r.tick(obs(15, 5, 40), 0, 25)[0]
    second = r.tick(obs(5, 25, 40), 0, 25)[0]
    assert first == second


def test_set_rejections_and_speed_clamp():
    r = SkillRunner("u", "cam", (30, 120), (0, 0), terrain=lambda x, y: (_ for _ in ()).throw(ValueError("Destination is outside the public terrain map")))
    assert r.set("goto", dict(x=1, y=1), 0, (0, 0)) == "Destination is outside the public terrain map"
    r = SkillRunner("u", "cam", (30, 120), (0, 0))
    assert r.set("goto", dict(x=1), 0, (0, 0)) == "goto needs x, y"
    assert r.set("sweep", dict(agl=50), 0, (0, 0)) == "sweep needs polygon or x, y, size"
    assert r.set("fly", {}, 0, (0, 0)) == "unknown skill 'fly'"
    assert r.set("goto", dict(x=1, y=1, speed="x"), 0, (0, 0)) == "speed must be a number (m/s)"
    assert r.set("goto", dict(x=100, y=0, speed=30), 0, (0, 0)) is None and r.state.params["speed"] == CRUISE_MPS
    assert r.set("goto", dict(x=100, y=0, speed=0.2), 0, (0, 0)) is None and r.state.params["speed"] == 1.0
    r.set("look_at", dict(x=5, y=6), 0, (0, 0))                 # does not replace the motion skill
    assert r.state.kind == "goto" and r.look == (5.0, 6.0)
    assert r.status() == {"skill": "goto", "done": False, "progress": "1/1", "look_at": (5.0, 6.0)}


def test_sweep_lanes_cover_the_polygon_from_the_nearest_end():
    poly = np.array([[0, 0], [100, 0], [100, 60], [0, 60]], float)
    pts = lawnmower(poly, 20.0)
    assert len(pts) == 6 and all(0 <= x <= 100 and 0 <= y <= 60 for x, y in pts)
    r = SkillRunner("u", "cam", (30, 120), (0, 0))
    assert r.set("sweep", dict(polygon=poly.tolist(), lane=20, agl=50), 0, (120, 70)) is None
    assert r.state.waypoints[0] == pts[-1]                      # reversed: the drone starts at the far corner
    nav, gim = r.tick(obs(120, 70, 50), 3.0, 0)
    assert gim == look("cam", 120, 70, 3.0)                      # nadir while sweeping
    assert nav["mode"] == "goto" and abs(footprint_w(50) - 57.735) < 1e-2


def test_action_builders_match_the_researcher_sdk():
    from anet4d import agent as sdk

    assert actions.nav_goto((1, 2, 3), 4) == sdk.nav_goto((1, 2, 3), 4)
    assert actions.nav_hover() == sdk.nav_hover() and actions.nav_land() == sdk.nav_land()
    assert actions.nav_takeoff(20) == sdk.nav_takeoff(20)
    assert actions.gimbal_look_at("eo", (1, 2, 3)) == sdk.gimbal_look_at("eo", (1, 2, 3))
    a = actions.act(7, nav=actions.nav_goto((1, 2, 3)), gimbal=actions.gimbal_look_at("eo", (0, 0, 0)))
    assert a == sdk.act(7, nav=sdk.nav_goto((1, 2, 3)), gimbal=sdk.gimbal_look_at("eo", (0, 0, 0)))
    assert not list(bench_validator("act").iter_errors(a))
