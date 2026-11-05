"""Unified L2 physics engine (``awr.phys.l2``): the multirotor model on the shared rigid body, the tier's interface and
contact rules, and the comparison against Fast on the shared consistency cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from awr.phys import PhysError
from awr.phys.adapter import TierCommand, VehicleStatus
from awr.phys.kinematic import GEAR_Z_M
from awr.phys.l2 import L2Tier
from awr.phys.l2 import rigid as RB
from awr.phys.l2.multirotor import MultirotorFleet, MultirotorParams, rotor_layout
from awr.phys.tier_compare import CASES, compare_case, run_fast, run_l2

G = 9.80665
CASE = {c.id: c for c in CASES}


def _run(tier, script: dict[int, list[TierCommand]], n: int) -> list:
    tier.open()
    out = []
    for k in range(n):
        if k in script:
            assert all(r.ok for r in tier.submit(script[k]))
        out.append(tier.advance(k))
    return out


def _tilt(sample) -> float:
    q = sample.q_ws_xyzw
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2)))))


# ---------------------------------------------------------------------------------------------------------------- core
def test_rigid_body_helpers_are_consistent() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    assert np.allclose(RB.cross(a, b), np.cross(a, b))
    yaw = np.array([0.0, 0.7, -2.0])
    q = RB.q_from_yaw(yaw)
    e1 = np.tile([1.0, 0.0, 0.0], (3, 1))
    assert np.allclose(RB.q_rot(q, e1), np.stack([np.cos(yaw), np.sin(yaw), 0 * yaw], 1))
    assert np.allclose(RB.yaw_of(q), yaw)
    v = rng.normal(size=(3, 3))
    assert np.allclose(RB.q_rot_inv(q, RB.q_rot(q, v)), v)
    e3 = np.tile([0.0, 0.0, 1.0], (3, 1))
    qz = RB.q_from_z_yaw(e3, yaw)
    assert np.allclose(np.abs(np.sum(qz * q, axis=1)), 1.0)                       # same rotation (up to sign)
    b3 = np.array([[0.3, 0.0, 1.0]]) / math.sqrt(1.09)
    qt = RB.q_from_z_yaw(b3, np.array([0.0]))
    assert np.allclose(RB.q_rot(qt, e3[:1]), b3, atol=1e-6) and abs(float(RB.yaw_of(qt)[0])) < 1e-6
    qi = RB.q_integrate(q, np.tile([0.0, 0.0, 0.5], (3, 1)), 0.2)
    assert np.allclose(RB.yaw_of(qi), (yaw + 0.1 + np.pi) % (2 * np.pi) - np.pi)
    S = RB.RigidState(1)
    S.m[:] = 2.0
    S.J[:] = (0.02, 0.02, 0.04)
    a_w = S.integrate(np.array([[0.0, 0.0, 2.0 * G]]), np.array([[0.0, 0.0, 0.004]]), 0.004)
    assert np.allclose(a_w, [[0.0, 0.0, G]]) and np.allclose(S.w, [[0.0, 0.0, 0.0004]])


def test_quad_x_layout_allocation_and_authority() -> None:
    pos, spin = rotor_layout("quad_x", 4, 0.174)
    assert pos.shape == (4, 2) and list(spin) == [1.0, 1.0, -1.0, -1.0]
    from awr.sim.fleet.profiles import ProfileTable

    T = ProfileTable()
    prof = T.get("x500")
    p = MultirotorParams.from_profile(prof, T.limits[T.limit_ids[T.limits_index(None, "x500")]])
    assert p.n_rot == 4 and p.k_m == 0.016 and abs(p.k_f * p.omega_max ** 2 - p.t_max_rotor) < 1e-9
    assert abs(4 * p.k_f * p.hover_omega ** 2 - p.mass_kg * G) < 1e-9
    A = p.effectiveness()
    Ti = np.linalg.pinv(A) @ np.array([p.mass_kg * G, 0.0, 0.0, 0.0])
    assert np.allclose(Ti, p.mass_kg * G / 4)                                        # hover: equal thrusts
    roll = A @ np.array([1.0, 0.0, 1.0, 0.0]) if pos[0, 1] > 0 else A @ np.where(pos[:, 1] > 0, 1.0, 0.0)
    assert roll[1] > 0.0 and abs(roll[2]) < 1e-12                                     # more thrust on the left (+y) rolls positive
    yaw = A @ np.where(spin > 0, 1.0, 0.0)
    assert yaw[3] < 0.0                                                               # CCW rotors react with negative yaw torque
    auth = p.torque_authority()                                                       # PX4 normalisation of the mixer columns
    assert auth[0] == pytest.approx(2 * math.sqrt(2) * 0.174 * p.t_max_rotor) and auth[1] == pytest.approx(auth[0])
    assert auth[2] == pytest.approx(4 * 0.016 * p.t_max_rotor)
    f = MultirotorFleet([p])
    assert f.k_rate[0, 0] == pytest.approx(0.15 * auth[0]) and f.k_rate[0, 2] == pytest.approx(0.2 * auth[2])


# ---------------------------------------------------------------------------------------------------------------- tier
def test_hover_is_steady_and_samples_are_self_consistent() -> None:
    t = L2Tier(["a"], {"a": ((0.0, 0.0, 10.0), 0.3)}, grounded=[], history_s=12.0, retain_s=None)
    assert t.n_steps == 25 and t.step_ns == 4_000_000 and t.caps()["engine"] == "awr.phys.l2"
    res = _run(t, {0: [TierCommand("a", "hover", 0)]}, 100)
    assert all(r.status["a"] == VehicleStatus.AIRBORNE for r in res) and not any(r.contacts for r in res)
    s = t.history.last_sample("a")
    assert np.linalg.norm(np.asarray(s.pos_enu_m) - (0.0, 0.0, 10.0)) < 0.2 and _tilt(s) < 1.0
    assert np.allclose(t.fleet.omega_r[0], t.fleet.params[0].hover_omega, rtol=0.02)
    sl = t.history.slice("a", 9_000_000_000, 10_000_000_000)
    assert np.all(np.diff(sl.t_ns) == 4_000_000)
    assert np.allclose(sl.state[:, 13:16], [0.0, 0.0, G], atol=0.05)                 # specific force at rest in the air
    assert np.allclose(np.linalg.norm(sl.state[:, 3:7], axis=1), 1.0, atol=1e-9)
    assert abs(float(RB.yaw_of(sl.state[-1:, 3:7])[0]) - 0.3) < 0.02


def test_takeoff_goto_land_impact_and_determinism() -> None:
    def make():
        return L2Tier(["a", "b"], {"a": ((0.0, 0.0, 0.0), 0.0), "b": ((5.0, 0.0, 0.0), 0.5)}, ground=lambda xy: 2.0, history_s=52.0,
                      retain_s=None)

    plan = {5: [TierCommand("a", "takeoff", 5, alt_m=10.0), TierCommand("b", "takeoff", 5, alt_m=12.0)],
            150: [TierCommand("a", "goto", 150, target_enu_m=(30.0, 20.0, 15.0), speed_mps=5.0)],
            300: [TierCommand("a", "land", 300), TierCommand("b", "kill", 300)]}
    t = make()
    res = _run(t, plan, 500)
    ev = [(c.uav, c.kind) for r in res for c in r.contacts]
    assert res[3].status == {"a": VehicleStatus.GROUNDED, "b": VehicleStatus.GROUNDED}
    assert t.history.slice("a", 1, 100_000_000).state[:, 15] == pytest.approx(G)       # at rest on the ground: (0, 0, g)
    assert ev == [("a", "LIFTOFF"), ("b", "LIFTOFF"), ("b", "IMPACT"), ("a", "TOUCHDOWN")]
    a = t.history.last_sample("a").pos_enu_m
    assert np.hypot(a[0] - 30.0, a[1] - 20.0) < 1.0 and abs(a[2] - (2.0 + GEAR_Z_M)) < 1e-6
    assert res[-1].status == {"a": VehicleStatus.GROUNDED, "b": VehicleStatus.CRASHED}
    assert VehicleStatus.LANDING in [r.status["a"] for r in res[300:]]
    z = [t.history.slice("a", k * 100_000_000 + 1, (k + 1) * 100_000_000).state[-1, 2] for k in range(60, 140)]
    assert max(z) > 2.0 + 9.5 and max(z) < 2.0 + 10.0 + 1.0                                 # reached the takeoff altitude, no big overshoot
    t2 = make()
    _run(t2, plan, 500)
    assert t2.close()["pose_sha256"] == t.close()["pose_sha256"]


def test_velocity_and_yaw_rate_steps() -> None:
    t = L2Tier(["a"], {"a": ((0.0, 0.0, 10.0), 0.0)}, grounded=[], history_s=8.0, retain_s=None)
    _run(t, {0: [TierCommand("a", "hover", 0)], 10: [TierCommand("a", "velocity", 10, vel_enu_mps=(5.0, 0.0, 0.0), yaw_rate_rps=0.5)]}, 60)
    sl = t.history.slice("a", 1_000_000_000, 6_000_000_000)
    vx = sl.state[:, 7]
    assert vx.max() < 5.0 * 1.15 and abs(vx[-1] - 5.0) < 0.5
    first = np.flatnonzero(vx > 0.05)
    assert first.size and 0.0 < sl.t_ns[first[0]] / 1e9 - 1.0 <= 0.3                      # command onset within 3 ticks
    wz = sl.state[:, 12]
    assert abs(wz[-1] - 0.5) < 0.05 and float(np.max(np.abs(sl.state[:, 10:12]))) < math.radians(220.0)   # MC_*RATE_MAX
    assert float(np.max(sl.state[:, 2])) < 11.0 and float(np.min(sl.state[:, 2])) > 9.0  # altitude held through the manoeuvre


def test_wind_hover_tilts_like_the_l1_calibration() -> None:
    run = run_l2(CASE["wind_hover"])
    s = run.state["u1"]
    t = run.t_ns / 1e9
    tail = s[t >= 17.0]
    tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (tail[:, 3] ** 2 + tail[:, 4] ** 2), -1.0, 1.0)))
    assert abs(float(np.mean(tilt)) - 7.85) < 0.7                                          # x500 composite aero: SIH-calibrated 7.85°
    assert float(np.max(np.linalg.norm(s[:, 0:3] - s[0, 0:3], axis=1))) < 3.0
    assert run.status["u1"][-1] == "AIRBORNE"


def test_wall_collision_and_kill_impact_on_the_shared_case() -> None:
    run = run_l2(CASE["contact_crash"])
    kinds = {u: [c["kind"] for c in run.contacts if c["uav"] == u] for u in ("wall", "drop")}
    assert kinds["wall"][:1] == ["COLLISION_WORLD"] and kinds["drop"][:1] == ["IMPACT"]
    assert run.status["wall"][-1] == "CRASHED" and run.status["drop"][-1] == "CRASHED"
    hit = next(c for c in run.contacts if c["uav"] == "wall")
    assert 29.0 < hit["pos_enu_m"][0] < 30.5                                               # stopped at the wall face x = 30
    assert run.info["crashes"] == 2 and run.info["collision_queries"] > 0


def test_rejects_vehicles_without_a_domain_model_and_bad_commands() -> None:
    with pytest.raises(PhysError) as e:
        L2Tier(["f"], {"f": ((0.0, 0.0, 50.0), 0.0)}, grounded=[], vehicles={"f": "awr_v1"})                 # VTOL: no domain model yet
    assert e.value.code == 630 and "vtol" in str(e.value)
    t = L2Tier(["a"], {"a": ((0.0, 0.0, 10.0), 0.0)}, grounded=[])
    t.open()
    r = t.submit([TierCommand("zz", "hover", 0), TierCommand("a", "orbit", 0), TierCommand("a", "follow_path", 0)])
    assert [x.code for x in r] == [630, 630, 630]


# ---------------------------------------------------------------------------------------------------------------- vs Fast
@pytest.mark.parametrize("cid", ["hover", "velocity_step", "yaw_step", "takeoff_land"])
def test_interface_agrees_with_fast_on_the_shared_cases(cid: str) -> None:
    """Same commands, same reference tracker: receipts, status sequences, contact kinds, PoseHistory grid and command onset
    agree between Fast and the unified L2 engine; the dynamic differences are reported with the M24 tolerances."""
    case = CASE[cid]
    rep = compare_case(case, run_fast(case), run_l2(case), ("fast", "l2"))
    assert rep["interface"]["pass"], rep["failures"]
    assert rep["tiers"] == ["fast", "l2"] and "l2" in rep["metrics"]
    assert rep["diff"].get("pos_rms_m", 0.0) < 5.0
