"""Fixed-wing domain of the unified L2 physics engine: the Beard–McLain coefficient-table aeroplane on the shared rigid body,
the autopilot (course / altitude / airspeed with stall protection), catapult launch and parachute recovery, the mixed fleet,
and the comparison against the Beard–McLain 6-DoF oracle and the fixed-wing fast tier."""

from __future__ import annotations

import math

import numpy as np
import pytest

from awr.phys import PhysError
from awr.phys.adapter import TierCommand, VehicleStatus
from awr.phys.fwkin import FwKinTier
from awr.phys.l2 import L2Tier
from awr.phys.l2.bm_oracle import BMOracleTier
from awr.phys.l2.fixedwing import FLY, GROUND, LAUNCH, LAUNCH_MARGIN_MPS, RAIL_ELEV_RAD, WingParams, euler_flu
from awr.phys.tier_compare import WING_CASES, compare_case, run_bm, run_fwkin, run_l2
from awr.sim.fleet.profiles import ProfileTable

CASE = {c.id: c for c in WING_CASES}
A, G, L, C = VehicleStatus.AIRBORNE, VehicleStatus.GROUNDED, VehicleStatus.LANDING, VehicleStatus.CRASHED


def _collapse(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _fly(tier: L2Tier, script: dict[int, list[TierCommand]], n: int, *, wind=None) -> dict:
    """Run ``n`` ticks and log per tick the engine state of the wing group: position, velocity, Euler angles, mode."""
    tier.open()
    if wind is not None:
        tier.set_wind(np.array(wind, np.float64))
    log = {"t": [], "p": [], "v": [], "eul": [], "mode": [], "status": [], "events": []}
    for k in range(n):
        if k in script:
            assert all(r.ok for r in tier.submit(script[k])), script[k]
        res = tier.advance(k)
        S, F = tier.wing.S, tier.wing.fleet
        log["t"].append((k + 1) * tier.dt * tier.substeps if hasattr(tier, "substeps") else (k + 1) * 0.1)
        log["p"].append(S.p.copy())
        log["v"].append(S.v.copy())
        log["eul"].append(np.stack(euler_flu(S.q), axis=1))
        log["mode"].append(F.mode.copy())
        log["status"].append(dict(res.status))
        log["events"].extend(res.contacts)
    for k in ("t", "p", "v", "eul", "mode"):
        log[k] = np.array(log[k])
    return log


# ---------------------------------------------------------------------------------------------------------------- parameters
def test_wing_params_come_from_the_profile_and_trim() -> None:
    T = ProfileTable()
    p = WingParams.from_profile(T.get("awr_f1"))
    assert p.mass_kg > 0.0 and p.S > 0.0 and p.b > 0.0 and p.aspect_ratio == pytest.approx(p.b**2 / p.S)
    assert 0.0 < p.v_stall < p.v_min < p.v_cruise < p.v_max == 28.0 and p.v_cruise == 18.0
    assert 0.0 < math.degrees(p.alpha0) < 25.0                                                      # stall blend sits past the linear range
    alpha, de, dt = p.trim(p.v_cruise)
    assert 0.0 < alpha < p.alpha0 and abs(de) < math.radians(25.0) and 0.0 < dt < 1.0            # level trim at cruise is well inside the envelope
    alpha_slow, _de, dt_slow = p.trim(p.v_min)
    assert alpha_slow > alpha and 0.0 < dt_slow < 1.0                                               # slower: more angle of attack
    assert float(p.t_max(0.0)) >= float(p.t_max(p.v_cruise)) > 0.0                                 # power-limited propeller: thrust falls with airspeed
    with pytest.raises(PhysError) as e:
        WingParams.from_profile(T.get("awr_q1"))                                                     # a multirotor has no fw_dynamics
    assert e.value.code == 630


# ---------------------------------------------------------------------------------------------------------------- engine
def test_speed_step_and_climb_hold_are_deterministic() -> None:
    def mk():
        return L2Tier(["a"], {"a": ((0.0, 0.0, 100.0), 0.0)}, grounded=[], vehicles={"a": "awr_f1"}, history_s=42.0, retain_s=None)

    script = {5: [TierCommand("a", "velocity", 5, vel_enu_mps=(25.0, 0.0, 0.0))],
              200: [TierCommand("a", "velocity", 200, vel_enu_mps=(18.0, 0.0, 3.0))]}
    t = mk()
    assert t.domains == {"a": "fixed_wing"}
    assert {"velocity", "goto", "follow_path", "orbit", "hover", "takeoff", "land", "kill"} <= set(t.caps()["wing_commands"])
    lg = _fly(t, script, 400)
    assert _collapse([s["a"] for s in lg["status"]]) == [A] and not lg["events"]
    sp = np.hypot(lg["v"][:, 0, 0], lg["v"][:, 0, 1])
    assert np.all(np.abs(sp[170:200] - 25.0) < 0.5) and sp.max() < 27.5 and np.all(np.abs(lg["p"][:200, 0, 2] - 100.0) < 2.0)   # 25 m/s held, altitude kept
    assert np.all(np.abs(lg["v"][300:, 0, 2] - 3.0) < 0.15) and np.all(np.abs(sp[300:] - 18.0) < 1.0)       # 3 m/s climb while slowing 25 → 18
    assert np.all(np.abs(np.degrees(lg["eul"][50:, 0, 0])) < 5.0)                                             # wings level once the spawn loiter is left
    info = t.close()
    assert info["domains"] == {"a": "fixed_wing"} and info["profiles"] == {"a": "awr_f1"} and "fixed_wing" in info["tracker"]
    assert info["crashes"] == 0
    t2 = mk()
    _fly(t2, script, 400)
    assert t2.close()["pose_sha256"] == info["pose_sha256"]


def test_catapult_launch_climbs_to_the_target_altitude_and_loiters() -> None:
    t = L2Tier(["a"], {"a": ((0.0, 0.0, 0.0), 0.0)}, grounded=["a"], vehicles={"a": "awr_f1"}, history_s=62.0, retain_s=None)
    p = t.wing.tracker.params["a"]
    script = {1: [TierCommand("a", "velocity", 1, vel_enu_mps=(18.0, 0.0, 0.0))],                 # on the ground nothing moves until a takeoff
              5: [TierCommand("a", "takeoff", 5, alt_m=60.0)]}
    lg = _fly(t, script, 600)
    assert np.all(lg["p"][:5, 0, :2] == 0.0) and [int(m[0]) for m in lg["mode"][:5]] == [GROUND] * 5
    modes = _collapse([int(m[0]) for m in lg["mode"]])
    assert modes == [GROUND, LAUNCH, FLY]
    lift = [e for e in lg["events"] if e.kind == "LIFTOFF"]
    assert len(lift) == 1 and 1.3 <= lift[0].t_ns / 1e9 <= 1.7 and lift[0].speed_mps == pytest.approx(p.v_min + LAUNCH_MARGIN_MPS, abs=0.3)
    on_rail = lg["mode"][:, 0] == LAUNCH
    assert np.allclose(np.degrees(lg["eul"][on_rail, 0, 1]), math.degrees(RAIL_ELEV_RAD), atol=0.1)         # carried along the inclined rail
    assert np.all(lg["v"][on_rail, 0, 2] > 0.0)
    assert _collapse([s["a"] for s in lg["status"]]) == [G, A] and t.kind("a") == "takeoff"
    z = lg["p"][:, 0, 2]
    assert z[249] > 50.0 and np.all(np.abs(z[400:] - 60.0) < 3.0)                                             # 4 m/s climb, then holds 60 m
    assert np.all(np.hypot(lg["p"][400:, 0, 0], lg["p"][400:, 0, 1]) < 450.0)                                # loiters near the climb-out point
    assert t.close()["crashes"] == 0


def test_parachute_recovery_touches_down_and_kill_impacts() -> None:
    t = L2Tier(["a", "k"], {"a": ((0.0, 0.0, 80.0), 0.0), "k": ((0.0, 800.0, 100.0), 0.0)}, grounded=[],
               vehicles={"a": "awr_f1", "k": "awr_f1"}, history_s=42.0, retain_s=None)
    p = t.wing.tracker.params["a"]
    lg = _fly(t, {10: [TierCommand("k", "kill", 10)], 20: [TierCommand("a", "land", 20)]}, 400)
    assert _collapse([s["a"] for s in lg["status"]]) == [A, L, G]
    assert _collapse([s["k"] for s in lg["status"]]) == [A, C]
    down = [e for e in lg["events"] if e.uav == "a"]
    assert [e.kind for e in down] == ["TOUCHDOWN"] and 15.0 < down[0].t_ns / 1e9 < 22.0 and down[0].speed_mps == 0.0
    vz = lg["v"][80:140, 0, 2]
    assert np.all(np.abs(vz + p.chute_descent_mps) < 0.3)                                                    # hangs under the canopy at the rated descent
    assert np.all(np.abs(np.degrees(lg["eul"][120:140, 0, :2])) < 3.0)                                       # level under the chute
    assert np.all(lg["v"][-1] == 0.0) and lg["p"][-1, 0, 2] == pytest.approx(0.2, abs=0.05)
    hit = [e for e in lg["events"] if e.uav == "k"]
    assert [e.kind for e in hit] == ["IMPACT"] and hit[0].speed_mps > 15.0 and hit[0].t_ns / 1e9 < 40.0     # controls dead: spirals into the ground
    assert t.kind("k") == "kill" and t.close()["crashes"] == 1


def test_crosswind_track_holds_the_line_with_crab() -> None:
    t = L2Tier(["a"], {"a": ((0.0, 0.0, 100.0), 0.0)}, grounded=[], vehicles={"a": "awr_f1"}, history_s=42.0, retain_s=None)
    lg = _fly(t, {5: [TierCommand("a", "follow_path", 5, path_enu_m=np.array([[1200.0, 0.0, 100.0]]), speed_mps=18.0)]}, 400,
              wind=[[0.0, 8.0, 0.0]])
    late = slice(100, 400)
    assert np.all(np.abs(lg["p"][late, 0, 1]) < 3.0)                                                          # on the line in an 8 m/s crosswind
    yaw = np.degrees(lg["eul"][late, 0, 2])
    assert abs(float(np.mean(yaw)) + math.degrees(math.asin(8.0 / 18.0))) < 4.0                               # nose into the wind
    vg = np.hypot(lg["v"][late, 0, 0], lg["v"][late, 0, 1])
    assert abs(float(np.mean(vg)) - math.sqrt(18.0**2 - 8.0**2)) < 0.6                                        # airspeed 18: ground speed is the remainder
    assert t.close()["crashes"] == 0


def test_mixed_fleet_multirotor_aeroplane_and_car_in_one_tier() -> None:
    t = L2Tier(["q", "f", "c"], {"q": ((0.0, 0.0, 10.0), 0.0), "f": ((0.0, 50.0, 100.0), 0.0), "c": ((5.0, 0.0, 0.0), 0.0)}, grounded=[],
               vehicles={"q": "x500", "f": "awr_f1", "c": "f1tenth"}, history_s=12.0, retain_s=None)
    assert t.domains == {"q": "multirotor", "f": "fixed_wing", "c": "ugv"}
    t.open()
    r = t.submit([TierCommand("f", "takeoff", 0), TierCommand("f", "velocity", 0, vel_enu_mps=(18.0, 0.0, 0.0)),
                  TierCommand("c", "goto", 0, target_enu_m=(10.0, 0.0, 0.0), speed_mps=1.0), TierCommand("c", "takeoff", 0)])
    assert [x.code for x in r] == [0, 0, 0, 630]                                                              # takeoff in the air: accepted, no-op; car: 630
    res = [t.advance(k) for k in range(100)]
    assert res[-1].status["q"] == A and res[-1].status["f"] == A and res[-1].status["c"] in (G, VehicleStatus.MOVING)
    f = t.history.last_sample("f").pos_enu_m
    assert 150.0 < f[0] < 200.0 and abs(f[2] - 100.0) < 2.0
    assert abs(t.history.last_sample("q").pos_enu_m[2] - 10.0) < 0.3 and abs(t.history.last_sample("c").pos_enu_m[0] - 10.0) < 1.0
    info = t.close()
    assert set(info["tracker"]) >= {"multirotor", "fixed_wing", "ugv"} and info["crashes"] == 0


# ---------------------------------------------------------------------------------------------------------------- oracles
def test_oracles_are_the_beard_mclain_model_and_the_fast_tier() -> None:
    for cls, kernel in ((BMOracleTier, "beard-mclain-6dof-rk4"), (FwKinTier, "fw-kin")):
        t = cls(["a"], {"a": ((0.0, 0.0, 100.0), 0.0)}, grounded=[], vehicles={"a": "awr_f1"}, history_s=12.0, retain_s=None)
        t.open()
        assert [x.code for x in t.submit([TierCommand("a", "takeoff", 0)])] == [630]                            # airborne only
        assert all(x.ok for x in t.submit([TierCommand("a", "velocity", 0, vel_enu_mps=(25.0, 0.0, 0.0))]))
        res = [t.advance(k) for k in range(100)]
        assert _collapse([r.status["a"] for r in res]) == [A]
        s = t.history.last_sample("a")
        assert 180.0 < s.pos_enu_m[0] < 260.0 and abs(s.pos_enu_m[2] - 100.0) < 3.0
        assert t.close()["kernel"] == kernel


@pytest.mark.parametrize(("cid", "oracle"), [("f_climb", "bm"), ("f_orbit", "bm"), ("f_goto", "fw"), ("f_wind_track", "fw")])
def test_engine_agrees_with_the_oracles(cid: str, oracle: str) -> None:
    """Same commands, same reference tracker, same controller (vs the Beard–McLain oracle): receipts, status sequences,
    contact kinds and PoseHistory grid agree; the dynamic differences stay within the first-measurement tolerances. Two cases
    per oracle here (the full 7 × 2 set is ``tier_compare --set wing``); each case runs the engine for 40–60 s of sim time."""
    case = CASE[cid]
    other = run_bm(case) if oracle == "bm" else run_fwkin(case)
    rep = compare_case(case, run_l2(case), other, ("l2", oracle))
    assert rep["interface"]["pass"], rep["failures"]
    assert rep["tol_pass"], (rep["diff"], rep["tol"])


def test_engine_only_wall_case() -> None:
    run = run_l2(CASE["f_wall"])
    assert [c["kind"] for c in run.contacts][:1] == ["COLLISION_WORLD"] and any(s[-1] == "CRASHED" for s in run.status.values())
    assert run.info["crashes"] == 1
