"""机队技能执行在真实 Env 上跑通（第 6 步，ADR-227）：合成世界、L1 观测、fast 物理、锁步。地面上的 goto 触发起飞，到高度后贴地形
飞到目标并把相机对着它；rtl 返回起飞点落地后该机结束；全程没有被动作守卫拒绝的动作，运行清单记到飞行时间。"""

from __future__ import annotations

from pathlib import Path

from awr.agentsdk.env.core import BenchEnv
from awr.agentsdk.env.spec import EnvSpec, FleetEntry
from awr.agentsdk.skills.fleet import SkillFleet

HOME = (-10.0, 50.0)
FLEET = (FleetEntry("uav-01", (*HOME, 0.0)),)


def test_goto_then_rtl_through_the_skill_fleet(synth_event, tmp_path: Path) -> None:
    w, ev = synth_event
    env = BenchEnv(EnvSpec("t-skills", 5, w, ev, FLEET, duration_s=90.0, out_dir=tmp_path / "run", physics="fast", noise="clean"))
    obs, infos = env.reset()
    cams = {u: next(s["sensor_id"] for s in infos[u]["config"]["sensors"] if s["kind"] == "camera") for u in env.possible_agents}
    fleet = SkillFleet(env.possible_agents, cams, {"uav-01": HOME}, (5.0, 120.0), takeoff_agl=20.0)
    phases, rejected, k_goto_done, looked = ["landed"], [], None, False
    while env.agents:
        k = env.k
        t_s = k * 0.1
        for u, o in obs.items():
            fleet.observe(u, o["obs"], t_s)
            rejected += [a for a in o["obs"].get("acks", []) if a.get("result") == "rejected"]
        if k == 0:
            assert fleet.command("uav-01", "goto", {"x": 30.0, "y": 50.0, "agl": 20.0}, t_s) is None
        st = fleet.status("uav-01")
        if st["skill"] == "goto" and st["done"] and k_goto_done is None:
            k_goto_done = k
            assert fleet.command("uav-01", "rtl", {}, t_s) is None
        if phases[-1] != fleet.phase("uav-01"):
            phases.append(fleet.phase("uav-01"))
        if fleet.phase("uav-01") == "landed_final":
            break
        actions = fleet.actions(k, obs, env.agents, t_s)
        looked |= actions["uav-01"].get("gimbal", {}).get("target_enu_m", [None])[:2] == [30.0, 50.0]
        obs, _r, _t, _tr, _i = env.step(actions)
    manifest = env.close()
    assert phases == ["landed", "takeoff", "airborne", "landed_final"], phases
    assert k_goto_done is not None and 50 < k_goto_done < 600, k_goto_done
    assert looked and rejected == [], rejected[:3]
    assert manifest["agents"]["uav-01"]["flight_s"] > 10.0
