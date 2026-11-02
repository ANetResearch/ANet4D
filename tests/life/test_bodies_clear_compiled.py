"""The compiled body clearance of WalkNetwork.clear_segment agrees with the NumPy reference on every case."""

import math

import numpy as np

from awr.life.civil.walking import ObstacleGrid, WalkNetwork


def grid(rng, n):
    free = np.ones((4, 4), bool).view(ObstacleGrid)
    free.positions = rng.uniform(-3, 3, (n, 2))
    free.sizes = np.c_[rng.uniform(0.2, 4.5, n), rng.uniform(0.2, 2.0, n), rng.uniform(0.5, 2.0, n)]
    free.yaws = rng.uniform(-math.pi, math.pi, n)
    free.circle_radii = np.where(rng.random(n) < 0.3, rng.uniform(0.1, 0.6, n), np.nan)
    free.radius = float(rng.uniform(0.25, 0.75))
    return free


def test_compiled_body_check_matches_numpy():
    net = WalkNetwork.__new__(WalkNetwork)
    rng = np.random.default_rng(7)
    seen = {True: 0, False: 0}
    for case in range(6000):
        free = grid(rng, int(rng.integers(1, 12)))
        a = rng.uniform(-4, 4, 2)
        if case % 3 == 0:  # starting inside a body's planning margin: the release rule
            j = int(rng.integers(len(free.positions)))
            a = free.positions[j] + rng.uniform(-1.0, 1.0, 2) * (free.sizes[j, :2] / 2 + free.radius + 0.2)
        b = a + rng.uniform(-6, 6, 2) * (rng.random() < 0.9)
        expect = WalkNetwork._bodies_clear_numpy(net, a, b, free)
        assert WalkNetwork._bodies_clear(net, a, b, free) == expect, case
        seen[bool(expect)] += 1
    assert min(seen.values()) > 1000


def test_compiled_body_check_follows_replaced_yaws():
    net = WalkNetwork.__new__(WalkNetwork)
    free = np.ones((4, 4), bool).view(ObstacleGrid)
    free.positions = np.array([[0.0, 2.3]])
    free.sizes = np.array([[4.0, 0.5, 1.0]])
    free.yaws = np.zeros(1)
    free.circle_radii = np.full(1, np.nan)
    free.radius = 0.3
    a, b = np.array([-3.0, 0.0]), np.array([3.0, 0.0])
    assert WalkNetwork._bodies_clear(net, a, b, free)  # a 4 m body lying along the segment, 2.05 m off it
    free.yaws = np.full(1, math.pi / 2)  # turned across, its near end 0.3 m off the segment (limit 0.44 m)
    assert not WalkNetwork._bodies_clear(net, a, b, free)
    assert not WalkNetwork._bodies_clear_numpy(net, a, b, free)


def test_indexed_place_availability_matches_the_full_test():
    from awr.life.civil.activity.director import Place
    from awr.life.civil.activity.grounded import GroundedActivities

    rng = np.random.default_rng(3)
    ga = GroundedActivities.__new__(GroundedActivities)
    ga.frames = {}
    ga.frame_index = None
    n = 60
    ga.indices = {f"p{i}": i for i in range(n)}
    seen = {True: 0, False: 0}
    for tick in range(40):
        xy = rng.uniform(0, 20, (n, 2))
        radii = rng.uniform(0.25, 2.5, n)
        riding = rng.random(n) < 0.2
        ga.update_frame(xy, radii, riding)
        for k in range(60):
            if k == 30:
                riding[:5] = ~riding[:5]
                xy[5:10] += 0.3  # changed in place within the frame
            place = Place("x", tuple(rng.uniform(-1, 21, 2)), "cafe", ("x",))
            actor = f"p{int(rng.integers(n))}"
            expect = ga.available_reference(place, actor)
            assert ga.available(place, actor) == expect
            seen[expect] += 1
    assert min(seen.values()) > 200


def test_berth_availability_does_not_apply_person_standing_radius():
    from awr.life.civil.activity.director import Place
    from awr.life.civil.activity.grounded import GroundedActivities

    ga = GroundedActivities.__new__(GroundedActivities)
    ga.frames = {}
    ga.frame_index = None
    ga.indices = {"driver": 0}
    ga.update_frame(np.array([[0.0, 0.0], [10.0, 1.0]]), np.array([0.5, 2.5]), np.array([False, False]))
    # Adjacent car's circumscribed radius overlaps a candidate berth centre.
    # This rejects an outdoor standing place but must delegate a car berth to
    # the exact vehicle backend. It does not grant a route or a reservation.
    for check in (ga.available, ga.available_reference):
        assert not check(Place("stay", (10.0, 0.0), "cafe", ("stay",)), "driver")
        assert check(Place("bay", (10.0, 0.0), "berth", ("bay",)), "driver")


def test_standing_room_beside_a_parked_car_uses_its_box():
    """A person's standing place beside a parked car is free when the car's box leaves room; the car's circumscribed
    disk is not the car (a resident's home 2.86 m from a curb car's centre was refused, town150 v5)."""
    from awr.life.civil.activity.director import Place
    from awr.life.civil.activity.grounded import GroundedActivities

    ga = GroundedActivities.__new__(GroundedActivities)
    ga.frames = {}
    ga.frame_index = None
    ga.indices = {"resident": 0}
    car = np.array([4.84, 2.18])
    xy = np.array([[50.0, 50.0], [0.0, 0.0]])
    radii = np.array([0.35, float(np.linalg.norm(car)) / 2])
    # Room: the person's radius and 0.15 m (0.5 m) from the car's side.
    for yaw, place, free in (
        (0.0, (0.0, 2.86), True),
        (math.pi / 2, (0.0, 2.86), False),
        (0.0, (0.0, 1.09 + 0.51), True),
        (0.0, (0.0, 1.09 + 0.49), False),
    ):
        ga.update_frame(xy, radii, np.array([False, False]), np.array([0.0, yaw]), np.array([[np.nan, np.nan], car]))
        for check in (ga.available, ga.available_reference):
            assert check(Place("home", place, "home", ("home",)), "resident") == free, (yaw, place)
    ga.update_frame(xy, radii, np.array([False, False]))  # without boxes: the disk test, as before
    assert not ga.available(Place("home", (0.0, 2.86), "home", ("home",)), "resident")


def test_indexed_place_availability_with_boxes_matches_the_full_test():
    from awr.life.civil.activity.director import Place
    from awr.life.civil.activity.grounded import GroundedActivities

    rng = np.random.default_rng(5)
    ga = GroundedActivities.__new__(GroundedActivities)
    ga.frames = {}
    ga.frame_index = None
    n = 60
    ga.indices = {f"p{i}": i for i in range(n)}
    seen = {True: 0, False: 0}
    for tick in range(30):
        xy = rng.uniform(0, 20, (n, 2))
        radii = rng.uniform(0.25, 2.5, n)
        riding = rng.random(n) < 0.2
        yaw = rng.uniform(-math.pi, math.pi, n)
        aspect = rng.uniform(0.2, 1.0, n)
        boxes = (
            2 * radii[:, None] * np.column_stack([np.ones(n), aspect]) / np.sqrt(1 + aspect**2)[:, None]
        )  # within the disk
        boxes[rng.random(n) < 0.5] = np.nan
        ga.update_frame(xy, radii, riding, yaw, boxes)
        for k in range(60):
            place = Place("x", tuple(rng.uniform(-1, 21, 2)), "cafe", ("x",))
            actor = f"p{int(rng.integers(n))}"
            expect = ga.available_reference(place, actor)
            assert ga.available(place, actor) == expect
            seen[expect] += 1
    assert min(seen.values()) > 150
