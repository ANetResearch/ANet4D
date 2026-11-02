import os
from pathlib import Path

import numpy as np
import pytest

from awr.life.motion.bullet import BulletWorld, Control, RouteFollower, VehicleProfile


@pytest.fixture
def build():
    path = os.environ.get("AWR_LIFE_BULLET_BUILD")
    if not path:
        pytest.skip("Set AWR_LIFE_BULLET_BUILD to a verified native build")
    return Path(path)


@pytest.fixture
def profile():
    return VehicleProfile(
        np.array([-2.0, -0.9, 0.17]),
        np.array([2.0, 0.9, 1.5]),
        np.array(
            [[-1.3, -0.77, 0.31, 0.31], [-1.3, 0.77, 0.31, 0.31], [1.25, -0.77, 0.31, 0.31], [1.25, 0.77, 0.31, 0.31]]
        ),
    )


def ground():
    p = np.array([[-100, -100, 0], [100, -100, 0], [100, 100, 0], [-100, 100, 0]], float)
    return p[[[0, 1, 2], [0, 2, 3]]]


def settle(world, seconds=3.0):
    for _ in range(round(seconds / 0.1)):
        world.step({n: Control() for n in world.names})


def test_shared_clock_batch_matches_individual_substeps(build, profile):
    def rollout(batch):
        with BulletWorld(build, ground()) as world:
            world.add_vehicle("a", profile, [0, 0, 0.15])
            world.add_vehicle("b", profile, [0, 8, 0.15])
            settle(world)
            controls = {"a": Control(0.2, 1000, 0), "b": Control(-0.15, 1500, 0)}
            rows = []
            for _ in range(240 // batch):
                rows.append(world.step(controls, substeps=batch))
            return np.concatenate(rows)

    a = rollout(1)
    b = rollout(24)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:, 0, 0], a[:, 1, 0])
    assert a[-1, 0, 0] == 4.0 and (a[-1, :, 1] > 0.2).all()


def test_invalid_commands_and_kinematic_teleports_are_atomic(build, profile):
    with BulletWorld(build, ground()) as world:
        world.add_vehicle("car", profile, [0, 0, 0.15])
        world.add_kinematic("person", [0.5, 0.4, 1.7], [5, 5, 0, 0])
        before = world.snapshot().states.copy()
        for controls, targets in [
            ({}, None),
            ({"car": Control(engine_force_n=1e8)}, None),
            ({"car": Control()}, {"person": [8, 5, 0, 0]}),
        ]:
            with pytest.raises(ValueError):
                world.step(controls, kinematic_targets=targets)
            np.testing.assert_array_equal(before, world.snapshot().states)
        world.step({"car": Control()}, kinematic_targets={"person": [5.1, 5, 0, 0]})
        assert world.snapshot().time_s == 0.1
        with pytest.raises(ValueError, match="frozen"):
            world.add_vehicle("new", profile, [0, 8, 0.15])


def test_vehicle_vehicle_contact_is_shared_not_independent_playback(build, profile):
    with BulletWorld(build, ground()) as world:
        world.add_vehicle("moving", profile, [0, 0, 0.15])
        world.add_vehicle("parked", profile, [8, 0, 0.15])
        settle(world)
        hit = False
        for _ in range(80):
            states = world.step({"moving": Control(0, 2500, 0), "parked": Control()}, substeps=24)
            contacts = world.contacts()
            hit |= any(set(row[:2]) == {1.0, 2.0} for row in contacts)
            if hit:
                break
        assert hit and states[:, :, 18].max() > 0.001


def test_kinematic_person_collision_is_reported(build, profile):
    with BulletWorld(build, ground()) as world:
        world.add_vehicle("car", profile, [0, 0, 0.15])
        world.add_kinematic("person", [0.5, 0.4, 1.7], [5, 0, 0, 0])
        settle(world)
        hit = False
        for _ in range(80):
            world.step({"car": Control(0, 2500, 0)})
            hit |= any(set(row[:2]) == {1.0, 1000000.0} for row in world.contacts())
            if hit:
                break
        assert hit


def test_route_replacement_preserves_actual_pose_velocity_and_wheels(build, profile):
    with BulletWorld(build, ground()) as world:
        world.add_vehicle("car", profile, [0, 0, 0.15])
        settle(world)
        start = world.snapshot().states[0]
        follower = RouteFollower(profile, [[*start[1:3]], [10, 0]], start)
        for _ in range(300):
            row = world.snapshot().states[0]
            c = follower.command(row, dt=world.dt)
            world.step({"car": c})
            if follower.arrived:
                break
        assert follower.arrived
        settle(world, 2.0)
        before = world.snapshot().states.copy()
        assert np.linalg.norm(before[0, 1:3] - [10, 0]) > 0.001  # not snapped to endpoint
        follower.set_route([before[0, 1:3], [20, 0]], before[0])
        np.testing.assert_array_equal(before, world.snapshot().states)
        for _ in range(300):
            c = follower.command(world.snapshot().states[0], dt=world.dt)
            world.step({"car": c})
            if follower.arrived:
                break
        assert follower.arrived and world.snapshot().states[0, 1] > 19.8
        assert (world.snapshot().states[0, 26::5] == 0).all()


def test_closed_world_and_wrong_thread_are_rejected(build, profile):
    import concurrent.futures

    world = BulletWorld(build, ground())
    world.add_vehicle("car", profile, [0, 0, 0.15])
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError, match="another thread"):
            pool.submit(world.snapshot).result()
    world.close()
    with pytest.raises(RuntimeError, match="closed"):
        world.snapshot()


def test_moving_kinematic_is_interpolated_and_contacts_are_recorded_between_ticks(build, profile):
    with BulletWorld(build, ground()) as world:
        world.add_vehicle("car", profile, [0, 0, 0.15])
        world.add_kinematic("person", [0.5, 0.4, 1.7], [0, -2, 0, 0], max_speed_mps=5.0)
        settle(world)
        samples = world.step({"car": Control()}, kinematic_targets={"person": [0, 2, 0, 0]}, substeps=240)
        assert (samples[:, 0, 19] > 0).any()  # crossing the car is a failed scenario, not a silent teleport
        assert samples[-1, 0, 0] == 4.0


def test_wheel_contacts_distinguish_ground_from_another_actor(build, profile):
    with BulletWorld(build, ground()) as world:
        world.add_vehicle("car", profile, [0, 0, 0.35])
        world.add_kinematic("platform", [20, 20, 0.2], [0, 0, 0, 0], max_speed_mps=0.0)
        settle(world)
        row = world.snapshot().states[0]
        assert row[20] == 4 and (row[26::5] == 1000000).all()


def test_rejects_changed_shared_library_before_loading(build, tmp_path):
    import json
    import shutil

    manifest = json.loads((build / "build.json").read_text())
    name = manifest["fleet_library"]["name"]
    shutil.copy2(build / "build.json", tmp_path / "build.json")
    (tmp_path / name).write_bytes((build / name).read_bytes() + b"changed")
    with pytest.raises(ValueError, match="library digest mismatch"):
        BulletWorld(tmp_path, ground())


def test_parked_body_sleeps_and_new_control_wakes_it(build, profile):
    with BulletWorld(build, ground()) as world:
        world.add_vehicle("car", profile, [0, 0, 0.15])
        settle(world, 5.0)
        before = world.snapshot().states[0]
        assert before[42] == 0
        for _ in range(50):
            world.step({"car": Control()})
        np.testing.assert_array_equal(world.snapshot().states[0, 1:], before[1:])
        world.step({"car": Control(0, 2000, 0)})
        assert world.snapshot().states[0, 42] == 1 and world.snapshot().states[0, 11] > 0.05


def test_brake_release_on_slope_wakes_and_rolls(build, profile):
    terrain = ground()
    terrain[:, :, 2] = 0.05 * terrain[:, :, 0]
    with BulletWorld(build, terrain) as world:
        world.add_vehicle("car", profile, [0, 0, 0.15])
        settle(world, 5.0)
        before = world.snapshot().states[0].copy()
        assert before[42] == 0
        for _ in range(30):
            world.step({"car": Control(0, 0, 0)})
        after = world.snapshot().states[0]
        assert after[42] == 1 and np.linalg.norm(after[1:3] - before[1:3]) > 0.1


def test_low_friction_sliding_is_not_falsely_put_to_sleep(build, profile):
    from dataclasses import replace

    profile = replace(profile, friction_slip=0.0001)
    terrain = ground()
    terrain[:, :, 2] = 0.2 * terrain[:, :, 0]
    with BulletWorld(build, terrain) as world:
        world.add_vehicle("car", profile, [0, 0, 0.15])
        settle(world, 10.0)
        row = world.snapshot().states[0]
        assert row[42] == 1 and abs(row[1]) > 1 and abs(row[11]) > 0.1


def test_sleeping_vehicle_still_senses_changed_wheel_support(build, profile):
    with BulletWorld(build, ground()) as world:
        world.add_vehicle("car", profile, [0, 0, 0.15])
        world.add_kinematic("low-obstacle", [4, 4, 0.1], [10, 0, 0, 0])
        settle(world, 5.0)
        before = world.snapshot().states[0].copy()
        assert before[42] == 0
        awake = False
        sensed = False
        for i in range(50):
            samples = world.step({"car": Control()}, kinematic_targets={"low-obstacle": [10 - 0.2 * (i + 1), 0, 0, 0]})
            awake |= bool((samples[:, 0, 42] == 1).any())
            sensed |= bool((samples[:, 0, 26:42:5] == 1000000).any())
        assert awake and sensed and world.snapshot().states[0, 3] > before[3] + 0.03


def test_solver_diagnostics_are_read_only_and_observe_normal_force(build, profile):
    def run(read):
        with BulletWorld(build, ground()) as w:
            w.add_vehicle("car", profile, [0, 0, 0.15])
            with pytest.raises(ValueError, match="physics step"):
                w.wheel_diagnostics()
            settle(w)
            xs = []
            for _ in range(20):
                xs.append(w.step({"car": Control(0, 1000, 0)}))
                if read:
                    before = w.snapshot().states.copy()
                    a = w.wheel_diagnostics()
                    b = w.wheel_diagnostics()
                    np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(before, w.snapshot().states)
                    assert a.shape == (1, 4, 22) and np.isfinite(a).all() and not a.flags.writeable
                    assert (a[0, :, 15] > 0).all() and (a[0, :, 18] == 250).all()
                    contacts = w.contact_diagnostics()
                    assert contacts.shape[1] == 13 and not contacts.flags.writeable
            return np.concatenate(xs)

    np.testing.assert_array_equal(run(False), run(True))
