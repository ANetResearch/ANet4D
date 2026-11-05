"""机队技能执行的阶段机（自 CHIMERA `Commander` 移入）：地面上的第一条运动技能触发起飞，hold / rtl 在地面被忽略，到达高度后安装
待执行技能，rtl 落地后结束该机，电量耗尽原地降落一次，被拒绝指令的文案不变。"""

from __future__ import annotations

from awr.agentsdk.skills.fleet import SkillFleet

UAVS = ["uav-01", "uav-02"]


def fleet(**kw):
    return SkillFleet(UAVS, dict.fromkeys(UAVS, "eo"), dict.fromkeys(UAVS, (0.0, 0.0)), (30, 120), terrain=lambda x, y: 10.0, **kw)


def ob(x, y, z, mode="airborne", soc=1.0, dtm=10.0):
    return {"nav": {"pos_enu_m": [x, y, z], "battery_soc": soc}, "status": {"mode": mode}, "ground_hint": {"dtm_m": dtm}}


def batch(**per_uav):
    return {u: {"obs": o} for u, o in per_uav.items()}


def test_takeoff_then_the_pending_skill_then_rtl_to_landed_final():
    f = fleet()
    o = ob(0, 0, 10, mode="landed")
    for u in UAVS:
        f.observe(u, o, 0.0)
    assert f.command("uav-01", "hold", {}, 0.0) is None and f.phase("uav-01") == "landed"      # ignored on the ground
    assert f.command("uav-01", "rtl", {}, 0.0) is None and f.phase("uav-01") == "landed"
    assert f.command("uav-01", "goto", {"x": 100, "y": 0, "agl": 40}, 0.0) is None
    assert f.phase("uav-01") == "takeoff" and f.state["uav-01"].pending == ("goto", {"x": 100, "y": 0, "agl": 40})
    a = f.actions(0, batch(**{"uav-01": o, "uav-02": o}), UAVS, 0.0)
    assert a["uav-01"] == {"schema": "drone4d.bench.act.v1", "k": 0, "nav": {"mode": "takeoff", "alt_agl_m": 40.0}}
    assert a["uav-02"] == {"schema": "drone4d.bench.act.v1", "k": 0}
    # climbing: no new command until 3 m below the target height
    a = f.actions(1, batch(**{"uav-01": ob(0, 0, 30)}), UAVS, 0.1)
    assert a["uav-01"] == {"schema": "drone4d.bench.act.v1", "k": 1} and f.phase("uav-01") == "takeoff"
    a = f.actions(2, batch(**{"uav-01": ob(0, 0, 47.5)}), UAVS, 0.2)
    assert f.phase("uav-01") == "airborne" and f.state["uav-01"].pending is None
    assert a["uav-01"]["nav"] == {"mode": "goto", "pos_enu_m": [25.0, 0.0, 50.0], "speed_mps": 8.0}
    assert a["uav-01"]["gimbal"] == {"sensor_id": "eo", "mode": "look_at", "target_enu_m": [100.0, 0.0, 10.0]}
    assert f.status("uav-01") == {"skill": "goto", "done": False, "progress": "1/1", "look_at": (100.0, 0.0)}
    # airborne commands go straight to the runner, malformed ones come back as text
    f.observe("uav-01", ob(100, 0, 50), 20.0)
    assert f.command("uav-01", "orbit", {"x": 100}, 20.0) == "orbit needs x, y"
    assert f.command("uav-01", "rtl", {}, 20.0) is None
    a = f.actions(200, batch(**{"uav-01": ob(100, 0, 50)}), UAVS, 20.0)
    assert a["uav-01"]["nav"]["mode"] == "goto" and f.phase("uav-01") == "airborne"
    a = f.actions(300, batch(**{"uav-01": ob(1, 0, 42)}), UAVS, 30.0)
    assert a["uav-01"]["nav"] == {"mode": "land"}
    f.actions(301, batch(**{"uav-01": ob(1, 0, 40, mode="airborne")}), UAVS, 30.1)
    assert f.phase("uav-01") == "airborne"
    a = f.actions(400, batch(**{"uav-01": ob(0, 0, 10, mode="landed")}), UAVS, 40.0)
    assert f.phase("uav-01") == "landed_final" and a["uav-01"] == {"schema": "drone4d.bench.act.v1", "k": 400}
    assert f.command("uav-01", "goto", {"x": 1, "y": 1}, 41.0) == "the drone has landed at the pad and ended its mission"
    assert f.actions(401, batch(**{"uav-01": ob(0, 0, 10, mode="landed")}), UAVS, 40.1)["uav-01"] == {"schema": "drone4d.bench.act.v1", "k": 401}


def test_battery_exhaustion_lands_once_and_refuses_further_commands():
    f = fleet()
    f.observe("uav-01", ob(0, 0, 10, mode="landed"), 0.0)
    f.command("uav-01", "goto", {"x": 50, "y": 50}, 0.0)
    f.actions(0, batch(**{"uav-01": ob(0, 0, 10, mode="landed")}), UAVS, 0.0)
    f.actions(1, batch(**{"uav-01": ob(0, 0, 68)}), UAVS, 0.1)
    assert f.phase("uav-01") == "airborne"
    assert f.observe("uav-01", ob(20, 20, 70, soc=0.5), 5.0) is None
    note = f.observe("uav-01", ob(30, 30, 70, soc=0.0), 6.0)
    assert note == "uav-01 battery exhausted at t=6s: emergency landing at (30, 30)" and f.phase("uav-01") == "exhausted"
    a = f.actions(60, batch(**{"uav-01": ob(30, 30, 70, soc=0.0)}), UAVS, 6.0)
    assert a["uav-01"] == {"schema": "drone4d.bench.act.v1", "k": 60, "nav": {"mode": "land"}}
    a = f.actions(61, batch(**{"uav-01": ob(30, 30, 69, soc=0.0)}), UAVS, 6.1)
    assert a["uav-01"] == {"schema": "drone4d.bench.act.v1", "k": 61}          # the land command once, then nothing
    assert f.command("uav-01", "goto", {"x": 1, "y": 1}, 7.0) == ("battery exhausted at t=6 s, the drone made an emergency landing at (30, 30) "
                                                                    "and is no longer available")


def test_rejections_of_the_interface():
    f = fleet()
    assert f.command("uav-09", "hold", {}, 0.0) == "unknown UAV"
    assert f.command("uav-01", "land", {}, 0.0) == "unknown skill 'land'"
    assert f.command("uav-01", "zoom", {"factor": 4}, 0.0) == "Optical lens unavailable in this run"
    seen = []
    f = fleet(zoom=lambda u, factor, t: seen.append((u, factor, t)) if factor <= 12 else (_ for _ in ()).throw(ValueError("zoom must be 1..12")))
    assert f.command("uav-01", "zoom", {"factor": 4}, 1.5) is None and seen == [("uav-01", 4, 1.5)]
    assert f.command("uav-01", "zoom", {"factor": 40}, 1.5) == "zoom must be 1..12"
    assert f.command("uav-01", "zoom", {}, 1.5) == "'factor'"
    # a UAV that is not in the episode any more gets no action
    assert f.actions(0, batch(**{"uav-01": ob(0, 0, 10, mode="landed")}), ["uav-02"], 0.0) == {}
