"""Surface vessel domain of the unified L2 physics engine: packages, the Fossen 3-DoF model on the shared rigid body, the
tier's water rules (water level, station keeping, current, grounding) and the comparison against the Fossen oracle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from awr.phys import PhysError
from awr.phys.adapter import OrbitSpec, TierCommand, VehicleStatus
from awr.phys.l2 import L2Tier
from awr.phys.l2 import rigid as RB
from awr.phys.l2.fossen_oracle import FossenOracleTier, fossen_rhs
from awr.phys.l2.ground import USV, UsvParams, UsvTable, load_family_docs, schema_errors
from awr.phys.l2.usv import UsvFleet, crossflow
from awr.phys.tier_compare import WATER_CASES, compare_case, run_fossen, run_l2

CASE = {c.id: c for c in WATER_CASES}


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
def test_usv_packages_validate_and_load() -> None:
    docs = load_family_docs(USV)
    assert sorted(docs) == ["blueboat", "otter", "wamv"]
    for doc, path in docs.values():
        assert schema_errors(doc, USV) == [] and path.name == "params.yaml" and doc["kind"] == "usv"
    T = UsvTable()
    p = T.get("otter@1.0.0")
    assert isinstance(p, UsvParams) and p.mass_kg == 80.0 and p.length_m == 2.0 and p.thr_y == 0.395 and p.k_pos == 0.01108
    assert p.m_surge == pytest.approx(85.5) and p.thrust_max == pytest.approx(239.2, abs=0.5) and p.v_max == pytest.approx(3.086)
    assert T.get("wamv").cd_crossflow == 0.0 and T.get("wamv").d_rr == 800.0 and T.get("blueboat").propulsion == "twin_fixed"
    assert not T.has("x500") and not T.has("f1tenth")
    bad = dict(docs["otter"][0])
    bad["hydro"] = {**bad["hydro"], "d_u": {"value": -1.0, "conf": "A", "src": "x"}}
    assert any("d_u" in e for e in schema_errors(bad, USV))
    with pytest.raises(PhysError):
        T.get("nac_usv")


# ---------------------------------------------------------------------------------------------------------------- model
def test_control_allocates_within_the_shaft_limits() -> None:
    F = UsvFleet([UsvTable().get("otter")])
    dt = 0.004
    n = F.control(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([2.0]), np.array([0.0]), np.array([True]), dt)
    T = F.shaft_thrust(n)
    assert n.shape == (1, 2) and T[0, 0] == pytest.approx(T[0, 1]) and 0.0 < T[0, 0] <= F.T_fwd[0]       # straight: equal shafts
    F.v_ref[:] = 0.0
    n = F.control(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([math.pi / 2]), np.array([True]), dt)
    T = F.shaft_thrust(n)
    assert T[0, 1] > 0.0 > T[0, 0] and T[0, 0] >= F.T_rev[0] - 1e-9                                      # turn left: right ahead, left astern
    n = F.control(np.array([3.5]), np.array([0.0]), np.array([0.0]), np.array([3.5]), np.array([np.nan]), np.array([True]), dt)
    assert np.all(F.shaft_thrust(n) <= 0.0 + 1e-9)                                                        # no thrust past v_max
    n = F.control(np.array([1.0]), np.array([0.2]), np.array([0.0]), np.array([2.0]), np.array([1.0]), np.array([False]), dt)
    assert np.all(n == 0.0)                                                                               # kill: thrusters off
    Yh, Nh = crossflow(np.array([0.5]), np.array([0.0]), F.L, F.T, F.cd, F.rho)
    assert Yh[0] < 0.0 and abs(Nh[0]) < 1e-9                                                              # pure sway: force only


def test_rhs_is_the_fossen_model_at_rest_and_in_steady_surge() -> None:
    F = UsvFleet([UsvTable().get("otter")])
    x = np.zeros((1, 8))
    f = fossen_rhs(x, np.zeros((1, 2)), F, np.zeros((1, 2)))
    assert np.allclose(f, 0.0)
    # steady surge at U: both shafts at the speed whose thrust balances X_u·U
    U = 1.5
    X = F.d_u[0] * U
    n = math.sqrt(0.5 * X / F.k_pos[0])
    x[0, 3], x[0, 6:8] = U, n
    f = fossen_rhs(x, np.full((1, 2), n), F, np.zeros((1, 2)))
    assert abs(f[0, 3]) < 1e-9 and abs(f[0, 4]) < 1e-9 and abs(f[0, 5]) < 1e-9 and f[0, 0] == pytest.approx(U)


# ---------------------------------------------------------------------------------------------------------------- tier
def test_speed_step_stops_exactly_and_is_deterministic() -> None:
    def make():
        return L2Tier(["b"], {"b": ((0.0, 0.0, 0.0), 0.0)}, vehicles={"b": "otter"}, history_s=32.0, retain_s=None)

    plan = {5: [TierCommand("b", "velocity", 5, vel_enu_mps=(2.0, 0.0, 0.0))], 200: [TierCommand("b", "velocity", 200, vel_enu_mps=(0.0, 0.0, 0.0))]}
    t = make()
    assert t.domains == {"b": "usv"} and t.caps()["water_commands"] == ["velocity", "goto", "follow_path", "orbit", "hover", "kill"]
    res = _run(t, plan, 300)
    st = _collapse([str(r.status["b"]) for r in res])
    assert st == ["AFLOAT", "MOVING", "AFLOAT"] and not any(r.contacts for r in res)
    sl = t.history.slice("b", 1, 30_000_000_000)
    assert np.allclose(sl.state[:, 2], 0.2) and np.allclose(sl.state[:, 9], 0.0, atol=1e-9)             # rides on water_z = 0 + 0.2 m
    assert np.allclose(sl.state[:, 10:12], 0.0, atol=1e-9) and np.allclose(RB.tilt_of(sl.state[:, 3:7]), 0.0, atol=1e-9)   # 3-DoF: no roll / pitch
    vx = sl.state[:, 7]
    t_s = sl.t_ns / 1e9
    assert abs(float(np.mean(vx[(t_s > 10.0) & (t_s < 19.0)])) - 2.0) < 0.05 and vx.max() < 2.0 * 1.1
    assert np.all(sl.state[t_s > 27.0, 7:10] == 0.0)                                                     # exactly at rest after the stop
    assert np.allclose(sl.state[-1, 13:16], [0.0, 0.0, RB.G], atol=1e-9)
    assert abs(float(RB.yaw_of(sl.state[-1:, 3:7])[0])) < 0.05 and abs(float(sl.state[-1, 1])) < 0.1
    t2 = make()
    _run(t2, plan, 300)
    assert t2.close()["pose_sha256"] == t.close()["pose_sha256"]
    info = t.close()
    assert info["domains"] == {"b": "usv"} and info["profiles"] == {"b": "otter"} and "usv" in info["tracker"] and info["groundings"] == 0


def test_station_keeping_in_a_current_and_drift_when_killed() -> None:
    t = L2Tier(["b"], {"b": ((0.0, 0.0, 0.0), 0.0)}, vehicles={"b": "otter"}, history_s=62.0, retain_s=None)
    t.set_current(np.array([[0.5, 0.0]]))
    res = _run(t, {0: [TierCommand("b", "hover", 0)], 450: [TierCommand("b", "kill", 450)]}, 600)
    sl = t.history.slice("b", 30_000_000_000, 45_000_000_000)
    off = np.hypot(sl.state[:, 0], sl.state[:, 1])
    assert float(np.max(off)) < 1.0 and abs(abs(float(RB.yaw_of(sl.state[-1:, 3:7])[0])) - math.pi) < 0.1  # holds the spot, bow into the stream
    assert float(np.max(np.hypot(sl.state[:, 7], sl.state[:, 8]))) < 0.1                                  # ground speed ~0 while making way through the water
    assert res[449].status["b"] == VehicleStatus.MOVING
    p0 = np.asarray(t.history.slice("b", 45_000_000_000, 45_100_000_000).state[0, 0:2])
    p1 = np.asarray(t.history.last_sample("b").pos_enu_m)[:2]
    assert 0.5 * 14.0 * 0.7 < p1[0] - p0[0] < 0.5 * 15.0 + 0.5 and abs(p1[1] - p0[1]) < 1.0           # killed: drifts with the 0.5 m/s current
    assert res[-1].status["b"] == VehicleStatus.MOVING and t.kind("b") == "kill"


def test_grounding_on_a_rising_shore_and_water_level() -> None:
    run = run_l2(CASE["w_grounding"])
    ev = run.contacts
    assert [c["kind"] for c in ev] == ["COLLISION_WORLD"] and ev[0]["detector"] == "dsm"
    assert -4.5 < ev[0]["pos_enu_m"][0] < -3.5 and run.status["b1"][-1] == "CRASHED"                     # 0.05·x > −draft 0.195 → x > −3.9
    assert _collapse(run.status["b1"]) == ["AFLOAT", "MOVING", "CRASHED"] and run.info["groundings"] == 1
    # the water level is a tier parameter: the hull sits at water_z + cg height; deep water without a world
    t = L2Tier(["b"], {"b": ((0.0, 0.0, 0.0), 0.0)}, vehicles={"b": "otter"}, water_z=12.5, history_s=4.0, retain_s=None)
    _run(t, {0: [TierCommand("b", "velocity", 0, vel_enu_mps=(1.0, 0.0, 0.0))]}, 20)
    assert abs(t.history.last_sample("b").pos_enu_m[2] - 12.7) < 1e-9 and t.caps()["water_z_m"] == 12.5


def test_mixed_fleet_air_ground_and_sea_in_one_tier() -> None:
    t = L2Tier(["q", "c", "b"], {"q": ((0.0, 0.0, 10.0), 0.0), "c": ((2.0, 0.0, 0.0), 0.0), "b": ((0.0, 5.0, 0.0), 0.0)}, grounded=[],
               vehicles={"q": "x500", "c": "f1tenth", "b": "blueboat"}, history_s=12.0, retain_s=None)
    assert t.domains == {"q": "multirotor", "c": "ugv", "b": "usv"}
    r = t.submit([TierCommand("b", "takeoff", 0, alt_m=5.0), TierCommand("b", "land", 0), TierCommand("q", "takeoff", 0, alt_m=5.0)])
    assert [x.code for x in r] == [630, 630, 0]
    res = _run(t, {0: [TierCommand("q", "hover", 0)], 5: [TierCommand("c", "goto", 5, target_enu_m=(10.0, 0.0, 0.0), speed_mps=2.0),
                                                           TierCommand("b", "goto", 5, target_enu_m=(10.0, 5.0, 0.0), speed_mps=1.5)]}, 100)
    assert res[-1].status == {"q": VehicleStatus.AIRBORNE, "c": VehicleStatus.GROUNDED, "b": VehicleStatus.AFLOAT}
    b = t.history.last_sample("b").pos_enu_m
    assert abs(b[0] - 10.0) < 1.0 and abs(b[1] - 5.0) < 1.0 and abs(b[2] - 0.12) < 1e-9
    assert abs(t.history.last_sample("c").pos_enu_m[0] - 10.0) < 1.0 and abs(t.history.last_sample("q").pos_enu_m[2] - 10.0) < 0.3
    assert t.tracker.kind("b") == "goto" and t.tracker.stats.keys() == {"multirotor", "ugv", "usv"}


def test_orbit_settles_and_waypoints_arrive() -> None:
    t = L2Tier(["b"], {"b": ((10.0, 0.0, 0.0), math.pi / 2)}, vehicles={"b": "otter"}, history_s=82.0, retain_s=None)
    res = _run(t, {5: [TierCommand("b", "orbit", 5, orbit=OrbitSpec((0.0, 0.0, 0.0), 10.0, cw=False, turns=1.0), speed_mps=1.5)]}, 800)
    sl = t.history.slice("b", 10_000_000_000, 40_000_000_000)
    r = np.hypot(sl.state[:, 0], sl.state[:, 1])
    assert abs(float(np.mean(r)) - 10.0) < 0.6 and float(np.std(r)) < 0.6
    assert res[-1].status["b"] == VehicleStatus.AFLOAT                                                    # one turn, then stopped (no circling the end point)


# ---------------------------------------------------------------------------------------------------------------- vs Fossen oracle
def test_oracle_is_the_fossen_model() -> None:
    t = FossenOracleTier(["b"], {"b": ((0.0, 0.0, 0.0), 0.0)}, vehicles={"b": "otter"}, history_s=32.0, retain_s=None)
    res = _run(t, {5: [TierCommand("b", "velocity", 5, vel_enu_mps=(2.0, 0.0, 0.0))], 100: [TierCommand("b", "hover", 100)]}, 300)
    assert _collapse([str(r.status["b"]) for r in res]) == ["AFLOAT", "MOVING", "AFLOAT"]                 # hover under way: stops ahead, no U-turn
    x_hold, x_cmd = t.history.last_sample("b").pos_enu_m[0], float(t.history.slice("b", 10_000_000_000, 10_004_000_001).state[0, 0])
    assert 0.5 < x_hold - x_cmd < 4.0
    assert t.close()["kernel"] == "fossen-3dof-rk4" and t.close()["rest_steps"] > 0
    assert [x.code for x in t.submit([TierCommand("b", "takeoff", 0)])] == [630]


@pytest.mark.parametrize("cid", ["w_speed_step", "w_speed_limit", "w_goto_stop", "w_waypoints", "w_orbit", "w_current_hold", "w_speed_step_wamv",
                                 "w_waypoints_blueboat"])
def test_engine_agrees_with_the_fossen_oracle(cid: str) -> None:
    """Same commands, same tracker, same controller: receipts, status sequences, PoseHistory grid and command onset agree
    between the engine's usv model and the Fossen equations in matrix form; the dynamic differences are within the tolerances."""
    case = CASE[cid]
    rep = compare_case(case, run_l2(case), run_fossen(case), ("l2", "fossen"))
    assert rep["interface"]["pass"], rep["failures"]
    assert rep["tol_pass"], (rep["diff"], rep["tol"])
