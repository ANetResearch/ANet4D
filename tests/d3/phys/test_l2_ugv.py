"""Ground vehicle domain of the unified L2 physics engine: packages, the single-track model on the shared rigid body, the
tier's ground rules (terrain following, stopping, walls) and the comparison against the CommonRoad ST oracle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from awr.phys import PhysError
from awr.phys.adapter import OrbitSpec, TierCommand, VehicleStatus
from awr.phys.l2 import L2Tier
from awr.phys.l2 import rigid as RB
from awr.phys.l2.ground import GroundTable, UgvParams, load_ground_docs, schema_errors
from awr.phys.l2.st_oracle import STOracleTier
from awr.phys.l2.ugv import UgvFleet
from awr.phys.tier_compare import GROUND_CASES, compare_case, run_l2, run_st

G = 9.80665
CASE = {c.id: c for c in GROUND_CASES}


def _run(tier, script: dict[int, list[TierCommand]], n: int) -> list:
    tier.open()
    out = []
    for k in range(n):
        if k in script:
            assert all(r.ok for r in tier.submit(script[k]))
        out.append(tier.advance(k))
    return out


def _collapse(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


# ---------------------------------------------------------------------------------------------------------------- packages
def test_ground_packages_validate_and_load() -> None:
    docs = load_ground_docs()
    assert sorted(docs) == ["f1tenth", "osr", "podcar"]
    for doc, path in docs.values():
        assert schema_errors(doc) == [] and path.name == "params.yaml" and doc["kind"] == "ugv"
    T = GroundTable()
    p = T.get("f1tenth@1.0.0")
    assert isinstance(p, UgvParams) and p.mass_kg == 3.74 and p.izz == 0.04712 and abs(p.wheelbase - 0.3302) < 1e-9
    assert p.mu == 1.0489 and p.c_sf == 4.718 and p.c_sr == 5.4562 and p.steer_max == 0.4189 and p.v_switch == 7.319
    assert T.get("podcar").v_max == pytest.approx(15.0 / 3.6, abs=0.01) and T.get("osr").drive == "corner_steer"
    assert not T.has("x500")
    bad = dict(docs["f1tenth"][0])
    bad["tire"] = {**bad["tire"], "mu": {"value": -1.0, "conf": "A", "src": "x"}}
    assert any("mu" in e for e in schema_errors(bad))
    with pytest.raises(PhysError):
        T.get("rover_x")


# ---------------------------------------------------------------------------------------------------------------- model
def test_control_respects_the_actuation_limits() -> None:
    T = GroundTable()
    F = UgvFleet([T.get("f1tenth")])
    dt = 0.004
    a, sv = F.control(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([3.0]), np.array([1.0]), np.array([True]), dt)
    assert a[0] == pytest.approx(0.9 * 9.51 * (1 + 3.0 * dt)) and sv[0] == 3.2                   # reference ramp + P on its first step; steering rate limit
    F.v_ref[:] = 15.0
    a, _ = F.control(np.array([15.0]), np.array([0.0]), np.array([0.0]), np.array([20.0]), np.array([np.nan]), np.array([True]), dt)
    assert 0.0 < a[0] <= 9.51 * 7.319 / 15.0 + 1e-9                                               # power limit above v_switch
    F.v_ref[:] = 20.0
    a, _ = F.control(np.array([20.0]), np.array([0.0]), np.array([0.0]), np.array([25.0]), np.array([np.nan]), np.array([True]), dt)
    assert a[0] == 0.0                                                                            # no acceleration past v_max
    a, sv = F.control(np.array([1.0]), np.array([0.0]), np.array([0.4189]), np.array([0.0]), np.array([2.0]), np.array([False]), dt)
    assert a[0] == -9.51 and sv[0] == 0.0                                                        # kill: full brake; wheel at the stop


def test_low_speed_limit_is_the_kinematic_single_track() -> None:
    """With the implicit tyre solve, a slow car with a fixed steering angle yaws at v·tan δ / L (f1tenth_gym switches to the
    kinematic model below 0.5 m/s for the same reason): no low-speed blow-up."""
    T = GroundTable()
    p = T.get("f1tenth")
    F = UgvFleet([p])
    S = RB.RigidState(1)
    S.m[:] = p.mass_kg
    S.J[:] = p.izz
    S.p[0] = (0.0, 0.0, p.cg_h)
    S.v[0] = (0.3, 0.0, 0.0)
    F.delta[:] = 0.3
    F.sv_max[:] = 0.0                                                                             # hold the wheel
    for _ in range(250):
        F.step(S, np.array([0.3]), np.array([np.nan]), np.array([True]), np.zeros((1, 3)), np.array([1.225]), np.zeros((1, 2)), 0.004)
    v = float(np.hypot(S.v[0, 0], S.v[0, 1]))
    assert abs(v - 0.3) < 0.02 and abs(float(S.w[0, 2]) - v * 0.3 / p.wheelbase) < 0.03 and np.all(np.isfinite(S.v))


# ---------------------------------------------------------------------------------------------------------------- tier
def test_speed_step_stops_exactly_and_is_deterministic() -> None:
    def make():
        return L2Tier(["c"], {"c": ((0.0, 0.0, 0.0), 0.0)}, vehicles={"c": "f1tenth"}, history_s=12.0, retain_s=None)

    plan = {5: [TierCommand("c", "velocity", 5, vel_enu_mps=(3.0, 0.0, 0.0))], 60: [TierCommand("c", "velocity", 60, vel_enu_mps=(0.0, 0.0, 0.0))]}
    t = make()
    assert t.domains == {"c": "ugv"} and t.caps()["ground_commands"] == ["velocity", "goto", "follow_path", "orbit", "hover", "kill"]
    res = _run(t, plan, 100)
    st = _collapse([str(r.status["c"]) for r in res])
    assert st == ["GROUNDED", "MOVING", "GROUNDED"] and not any(r.contacts for r in res)
    sl = t.history.slice("c", 1, 10_000_000_000)
    assert np.allclose(sl.state[:, 2], 0.074) and np.allclose(sl.state[:, 9], 0.0, atol=1e-9)        # rides at cg height on z = 0
    vx = sl.state[:, 7]
    t_s = sl.t_ns / 1e9
    assert abs(float(np.mean(vx[(t_s > 4.0) & (t_s < 6.0)])) - 3.0) < 0.05 and vx.max() < 3.0 * 1.1
    assert np.all(sl.state[t_s > 8.0, 7:10] == 0.0)                                              # exactly at rest after the stop
    assert np.allclose(sl.state[-1, 13:16], [0.0, 0.0, G], atol=1e-9)                              # specific force at rest (0, 0, g)
    assert abs(float(RB.yaw_of(sl.state[-1:, 3:7])[0])) < 0.05 and abs(float(sl.state[-1, 1])) < 0.1   # drove straight
    t2 = make()
    _run(t2, plan, 100)
    assert t2.close()["pose_sha256"] == t.close()["pose_sha256"]
    info = t.close()
    assert info["domains"] == {"c": "ugv"} and info["profiles"] == {"c": "f1tenth"} and "ugv" in info["tracker"]


def test_slope_following_tilts_the_body_and_holds_speed() -> None:
    t = L2Tier(["c"], {"c": ((0.0, 0.0, 0.0), 0.0)}, vehicles={"c": "f1tenth"}, ground=lambda xy: 0.15 * float(xy[0]), history_s=14.0, retain_s=None)
    _run(t, {5: [TierCommand("c", "velocity", 5, vel_enu_mps=(2.0, 0.0, 0.0))]}, 120)
    s = t.history.last_sample("c")
    p = np.asarray(s.pos_enu_m)
    assert abs(p[2] - (0.15 * p[0] + 0.074)) < 1e-6
    tilt = math.degrees(RB.tilt_of(np.asarray(s.q_ws_xyzw)[None])[0])
    assert abs(tilt - math.degrees(math.atan(0.15))) < 0.05 and abs(float(RB.yaw_of(np.asarray(s.q_ws_xyzw)[None])[0])) < 0.05
    sl = t.history.slice("c", 8_000_000_000, 12_000_000_000)
    assert abs(float(np.mean(np.hypot(sl.state[:, 7], sl.state[:, 8]))) - 2.0) < 0.1               # speed held on the 15 % grade
    assert float(np.mean(sl.state[:, 9])) == pytest.approx(0.3, abs=0.05)                            # climbing at v·grade


def test_wall_collision_and_dsm_curb() -> None:
    run = run_l2(CASE["g_wall"])
    kinds = [c["kind"] for c in run.contacts]
    assert kinds[:1] == ["COLLISION_WORLD"] and run.contacts[0]["detector"] == "mesh"
    assert 15.3 < run.contacts[0]["pos_enu_m"][0] < 16.0 and run.status["c1"][-1] == "CRASHED"       # stopped at the wall face x = 16
    assert _collapse(run.status["c1"]) == ["GROUNDED", "MOVING", "CRASHED"]
    # a 0.5 m step in the DSM (a curb, a wall footprint) is a side hit, not a hill
    t = L2Tier(["c"], {"c": ((0.0, 0.0, 0.0), 0.0)}, vehicles={"c": "f1tenth"}, ground=lambda xy: 0.0 if xy[0] < 5.0 else 0.5, history_s=8.0,
               retain_s=None)
    res = _run(t, {2: [TierCommand("c", "velocity", 2, vel_enu_mps=(2.0, 0.0, 0.0))]}, 60)
    ev = [c for r in res for c in r.contacts]
    assert [e.kind for e in ev] == ["COLLISION_WORLD"] and ev[0].detector == "dsm" and 4.5 < ev[0].pos_enu_m[0] < 5.2
    assert res[-1].status["c"] == VehicleStatus.CRASHED and t.close()["ground_snaps"] == 1


def test_mixed_fleet_air_and_ground_in_one_tier() -> None:
    t = L2Tier(["q", "c"], {"q": ((0.0, 0.0, 10.0), 0.0), "c": ((2.0, 0.0, 0.0), 0.0)}, grounded=[], vehicles={"q": "x500", "c": "f1tenth"},
               history_s=10.0, retain_s=None)
    assert t.domains == {"q": "multirotor", "c": "ugv"}
    r = t.submit([TierCommand("c", "takeoff", 0, alt_m=5.0), TierCommand("c", "land", 0), TierCommand("q", "takeoff", 0, alt_m=5.0)])
    assert [x.code for x in r] == [630, 630, 0]
    res = _run(t, {0: [TierCommand("q", "hover", 0)], 5: [TierCommand("c", "goto", 5, target_enu_m=(10.0, 0.0, 0.0), speed_mps=2.0)]}, 80)
    assert res[-1].status == {"q": VehicleStatus.AIRBORNE, "c": VehicleStatus.GROUNDED}
    assert VehicleStatus.MOVING in [r.status["c"] for r in res]
    c = t.history.last_sample("c").pos_enu_m
    q = t.history.last_sample("q").pos_enu_m
    assert abs(c[0] - 10.0) < 1.0 and abs(c[2] - 0.074) < 1e-9 and abs(q[2] - 10.0) < 0.3
    assert t.tracker.kind("c") == "goto" and t.tracker.kind("q") == "hover"
    with pytest.raises(PhysError) as e:
        L2Tier(["f"], {"f": ((0.0, 0.0, 50.0), 0.0)}, grounded=[], vehicles={"f": "awr_v1"})
    assert e.value.code == 630 and "vtol" in str(e.value)


def test_orbit_and_waypoints_with_the_car() -> None:
    t = L2Tier(["c"], {"c": ((6.0, 0.0, 0.0), math.pi / 2)}, vehicles={"c": "f1tenth"}, history_s=32.0, retain_s=None)
    _run(t, {5: [TierCommand("c", "orbit", 5, orbit=OrbitSpec((0.0, 0.0, 0.0), 6.0, cw=False, turns=1.0), speed_mps=2.0)]}, 300)
    sl = t.history.slice("c", 5_000_000_000, 20_000_000_000)
    r = np.hypot(sl.state[:, 0], sl.state[:, 1])
    assert abs(float(np.mean(r)) - 6.0) < 0.3 and float(np.std(r)) < 0.3
    assert t.status()["c"] == VehicleStatus.GROUNDED                                               # one turn, then stopped


# ---------------------------------------------------------------------------------------------------------------- vs ST oracle
def test_oracle_is_the_commonroad_model() -> None:
    t = STOracleTier(["c"], {"c": ((0.0, 0.0, 0.0), 0.0)}, vehicles={"c": "f1tenth"}, history_s=8.0, retain_s=None)
    res = _run(t, {5: [TierCommand("c", "velocity", 5, vel_enu_mps=(3.0, 0.0, 0.0))], 40: [TierCommand("c", "hover", 40)]}, 70)
    assert _collapse([str(r.status["c"]) for r in res]) == ["GROUNDED", "MOVING", "GROUNDED"]
    assert t.close()["kernel"] == "commonroad-st-rk4" and t.close()["ks_steps"] > 0
    assert [x.code for x in t.submit([TierCommand("c", "takeoff", 0)])] == [630]


@pytest.mark.parametrize("cid", ["g_speed_step", "g_speed_limit", "g_goto_stop", "g_waypoints", "g_orbit", "g_slope", "g_speed_step_podcar",
                                 "g_waypoints_osr"])
def test_engine_agrees_with_the_st_oracle(cid: str) -> None:
    """Same commands, same tracker, same controller: receipts, status sequences, PoseHistory grid and command onset agree
    between the engine's ugv model and the CommonRoad ST equations; the dynamic differences are within the tolerances."""
    case = CASE[cid]
    rep = compare_case(case, run_l2(case), run_st(case), ("l2", "st"))
    assert rep["interface"]["pass"], rep["failures"]
    assert rep["tol_pass"], (rep["diff"], rep["tol"])
