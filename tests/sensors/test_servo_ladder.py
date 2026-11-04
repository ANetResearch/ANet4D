"""Gimbal servo tiers (one model for sim-core and the agent-env camera) and the device fidelity ladder."""

import math

import numpy as np
import pytest

from awr.agentsdk.env.camera import CameraModel, Gimbal
from awr.sim.device import DECLARED, TIERS, implemented, ladder, servo_responses
from awr.sim.sensors import servo as SV
from awr.sim.sensors.spec import SERVO_L2_DEFAULTS, GimbalSpec


def test_l1_is_the_legacy_rate_limited_step():
    for a, t, r, dt in ((0.0, 1.0, 1.5, 0.02), (0.5, -2.0, 0.3, 0.1), (1.0, 1.0001, 90.0, 0.02)):
        step = r * dt
        assert SV.step_l1(a, t, r, dt) == a + min(max(t - a, -step), step)
    a = np.array([0.0, 0.5, 1.0])
    t = np.array([1.0, -2.0, 1.0001])
    r = np.array([1.5, 0.3, 90.0])
    assert np.array_equal(SV.step_l1(a, t, r, 0.02), a + np.minimum(np.maximum(t - a, -r * 0.02), r * 0.02))


def test_l2_respects_rate_and_acceleration_limits_and_settles_without_overshoot():
    rate, acc, tau, dt = math.radians(90), math.radians(400), 0.08, 0.02
    a, w, prev = 0.0, 0.0, 0.0
    hist = []
    for _ in range(200):
        a, w = SV.step_l2(a, w, math.radians(60), rate, acc, tau, dt)
        assert abs(w) <= rate * (1 + 1e-12)
        # one acceleration limit per step; the landing step may stop the small residual velocity at once
        assert abs(w - prev) <= acc * dt * (1 + 1e-12) or (a == math.radians(60) and abs(prev) <= 2 * acc * dt)
        assert a <= math.radians(60) + 1e-12  # never beyond the target
        prev = w
        hist.append(a)
    assert a == pytest.approx(math.radians(60), abs=1e-9) and w == 0.0
    assert all(b >= c for b, c in zip(hist[1:], hist[:-1], strict=True))  # monotone
    # slower than l1 at the start (ramp-up), reaches the target later than the l1 slew time 60/90 = 0.667 s
    l1 = SV.response(math.radians(60), rate, dt, 200, "l1")
    assert hist[3] < l1[3]
    t2 = SV.settle_time_s(math.radians(60), rate, acc, tau, dt, math.radians(0.05))
    assert 0.667 < t2 < 1.2
    # negative direction is symmetric
    a, w = 0.0, 0.0
    for _ in range(200):
        a, w = SV.step_l2(a, w, -math.radians(60), rate, acc, tau, dt)
    assert a == pytest.approx(-math.radians(60), abs=1e-9)


def test_l2_scalar_and_vector_paths_agree_bitwise():
    rate, acc, tau, dt = math.radians(90), math.radians(400), 0.08, 0.02
    targets = np.array([0.8, -1.2, 0.05, 2.0])
    a = np.zeros(4)
    w = np.zeros(4)
    sa, sw = [0.0] * 4, [0.0] * 4
    for _ in range(120):
        a, w = SV.step_l2(a, w, targets, np.full(4, rate), np.full(4, acc), np.full(4, tau), dt)
        for i in range(4):
            sa[i], sw[i] = SV.step_l2(sa[i], sw[i], float(targets[i]), rate, acc, tau, dt)
        assert a.tolist() == sa and w.tolist() == sw


def test_l1_tends_to_l2_limit():
    rate, dt = math.radians(90), 0.02
    r1 = SV.response(1.0, rate, dt, 100, "l1")
    r2 = SV.response(1.0, rate, dt, 100, "l2", acc_max=1e9, tau=1e-6)
    assert np.allclose(r1, r2, atol=1e-9)


def test_env_camera_gimbal_tiers():
    g1 = Gimbal(pitch_deg=-30.0, yaw_deg=0.0)
    g1.command(-30.0, 60.0)
    for _ in range(10):
        g1.slew(0.1)
    assert g1.yaw_deg == pytest.approx(60.0) and g1.pitch_deg == -30.0  # l1: 90°/s, 0.667 s
    g2 = Gimbal(pitch_deg=-30.0, yaw_deg=0.0, cam=CameraModel(servo_tier="l2"))
    g2.command(-30.0, 60.0)
    g2.slew(0.1)
    assert 0.0 < g2.yaw_deg < 9.0  # ramping up, slower than l1's 9° in the first 0.1 s
    for _ in range(40):
        g2.slew(0.05)
    assert g2.yaw_deg == pytest.approx(60.0, abs=1e-6) and g2.omega_yaw_dps == 0.0
    assert g2.cam.config()["gimbal"]["rate_max_dps"] == 90.0


def test_gimbal_spec_parses_servo_tier():
    from awr.sim.sensors.spec import SensorSpecError, _gimbal

    base = {"pitch_min_deg": -90, "pitch_max_deg": 30, "yaw_min_deg": -150, "yaw_max_deg": 150, "rate_max_deg_s": 90}
    s = _gimbal({"gimbal": base})
    assert s.servo_tier == "l1" and s.acc_max_rad_s2 == math.inf and s.tau_s == 0.0
    s = _gimbal({"gimbal": {**base, "servo": {"tier": "l2"}}})
    assert s.servo_tier == "l2" and s.acc_max_rad_s2 == pytest.approx(math.radians(SERVO_L2_DEFAULTS["acc_max_deg_s2"]))
    assert s.tau_s == SERVO_L2_DEFAULTS["tau_s"]
    with pytest.raises(SensorSpecError):
        _gimbal({"gimbal": {**base, "servo": {"tier": "l9"}}})
    with pytest.raises(SensorSpecError):
        _gimbal({"gimbal": {**base, "servo": {"tier": "l2", "tau_s": 0}}})
    assert isinstance(s, GimbalSpec)


def test_gimbal_bank_l2_column_slews_with_second_order_servo(bench_factory, profiles):
    """An l2 gimbal in the bank: the axis ramps up (slower than l1 at first), never overshoots, and the vector path
    (dynamic set beyond SCALAR_MAX) agrees with the scalar path."""
    import dataclasses

    from awr.sim.sensors.gimbal import SCALAR_MAX

    def run(n_slots, tier):
        b = bench_factory()
        for s in range(n_slots):
            b.spawn(s, s + 1)
        b.run(0.02)  # rigs are instantiated at spawn (rt.rig_list); patch their gimbal specs afterwards
        for r_ in b.rt.rig_list:
            if r_ is None:
                continue
            specs = tuple(
                dataclasses.replace(
                    sp,
                    gimbal=dataclasses.replace(
                        sp.gimbal, servo_tier=tier, acc_max_rad_s2=math.radians(400), tau_s=0.08
                    ),
                )
                if sp.gimbal is not None
                else sp
                for sp in r_.specs
            )
            object.__setattr__(r_, "specs", specs)
            r_.__dict__.pop("col_specs", None)
        for s in range(n_slots):
            b.rt.set_mode(s, "camera", "fixed", {"az_rad": math.radians(60), "el_rad": math.radians(-15)})
        hist = []
        for _ in range(80):
            b.run(0.02)
            hist.append([float(b.S.blocks["sensors"]["g_az"][s, 0]) for s in range(n_slots)])
        return hist

    one = run(1, "l2")
    many = run(SCALAR_MAX + 2, "l2")
    l1 = run(1, "l1")
    assert one[2][0] < l1[2][0]  # ramp-up
    assert max(h[0] for h in one) <= math.radians(60) + 1e-12 and one[-1][0] == pytest.approx(
        math.radians(60), abs=1e-9
    )
    assert all(h[0] == m[0] for h, m in zip(one, many, strict=True))  # scalar path == vector path
    assert len({tuple(h) for h in many[-1:]}) == 1  # every slot identical


def test_encoder_noise_is_in_radians():
    from awr.sensorsvc.camera.effects import gimbal_reading

    g = (0.1, -0.5, 0.0)
    r = gimbal_reading(g, 2.0, seed=1, uav_no=0, sensor_no=0, frame_index=3)
    assert all(abs(a - b) < math.radians(2.0) * 6 for a, b in zip(r, g, strict=True))
    assert gimbal_reading(g, 0.0, seed=1, uav_no=0, sensor_no=0, frame_index=3) == g


def test_ladder_reads_the_catalog_and_states_each_tier():
    doc = ladder()
    impl = implemented()
    assert (
        impl["l1"]["implemented"]
        and impl["fw"]["implemented"]
        and impl["l2"]["implemented"]
        and not impl["l3"]["implemented"]
    )
    by_id = {d["id"]: d for d in doc["devices"]}
    q1 = by_id["awr_q1"]
    assert q1["domain"] == "air" and q1["fidelity_declared"] == ["l0", "l1"] and q1["tiers"]["l1"]["status"] == "可用"
    assert q1["tiers"]["l2"]["status"] == "无（gz 无该机型）" and q1["tiers"]["fw"]["status"] == "代码有、机型未声明"
    assert q1["sensors"]["eo"]["gimbal"]["servo_tier"] == "l1" and q1["sensors"]["eo"]["gimbal"]["rate_max_deg_s"] == 60
    assert by_id["awr_f1"]["tiers"]["fw"]["status"] == "可用"
    assert doc["tiers"]["l2"]["status"].startswith("可用") and doc["tiers"]["l3"]["status"] == "无"
    assert (
        by_id["x500"]["tiers"]["l2"]["status"] == "可用（gz）"
        and by_id["x500"]["tiers"]["l2"]["gz_model"] == "x500_d4d"
    )
    assert sum(d["tiers"]["l2"]["status"] == "可用（gz）" for d in doc["devices"]) == 2
    assert {d["domain"] for d in doc["devices"]} == {"air", "land", "sea"}
    assert all(d["status"] == "planned" for d in doc["devices"] if d["domain"] != "air")
    assert set(DECLARED.values()) <= set(TIERS)
    r = servo_responses()
    assert r["settle_s"]["l2"] > r["settle_s"]["l1"] > 0.6 and len(r["l1"]) == len(r["l2_response"])
