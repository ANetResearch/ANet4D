"""life_v1 vehicle execution: live stop lookahead and arc-based trip execution."""

import math

import numpy as np

from awr.life.civil.driving_motion import Limits
from awr.life.motion import kernels as K
from awr.life.motion.vehicles import (
    CRAWL_M,
    PERSON_ROOM_M,
    PERSON_ROOM_S,
    LiveDrive,
    predict_route,
    route_at,
    stop_arc,
    stop_blocker,
)

CAR = np.array([[-2.1, 2.1, -0.9, 0.9], [-2.1, 2.1, -0.9, 0.9]])
PERSON = np.array([[-0.12, 0.24, -0.29, 0.29], [-0.42, 0.59, -0.31, 0.31]])
TIMES = np.r_[np.arange(1, 11) * 0.1, 1.0 + np.arange(1, 7) * 0.25]


def straight(length=60.0, step=0.5):
    x = np.arange(0.0, length + 1e-9, step)
    return dict(points=np.stack([x, np.zeros_like(x)], 1), yaw=np.zeros_like(x))


def run(drive, ticks, stop=None):
    v = []
    for _ in range(ticks):
        drive.advance(0.1, stop_before_arc_m=stop(drive) if stop else None)
        v.append(drive.speed_mps)
    v = np.r_[0.0, v]
    return v, np.diff(v) / 0.1


def test_free_trip_respects_limits_and_arrives():
    d = LiveDrive(straight())
    v, a = run(d, 400)
    assert d.arrived and abs(d.arc_m - 60.0) < 1e-9 and v.max() <= Limits().forward_mps + 1e-9
    assert a.max() <= Limits().acceleration_mps2 + 1e-9 and a.min() >= -6.0


def test_gear_change_rests_then_reverses():
    fwd = straight(10.0)
    pts = np.r_[fwd["points"], fwd["points"][::-1][1:] + np.array([0.0, 1e-3])]
    gear = np.r_[np.ones(len(fwd["points"]) - 1), -np.ones(len(fwd["points"]) - 1)]
    # A reverse leg returns along the same line: arc keeps growing, the heading stays +x.
    pts[len(fwd["points"]) :, 1] = 0.0
    pts[len(fwd["points"]) :, 0] -= 1e-3
    d = LiveDrive(dict(points=pts, yaw=np.zeros(len(pts)), gear=gear))
    modes = []
    for _ in range(400):
        st = d.advance(0.1)
        modes.append(st["mode"])
        if d.arrived:
            break
    assert d.arrived and "gear_change" in modes and abs(st["xy"][0]) < 0.01


def test_stops_before_live_obstruction_and_resumes():
    d = LiveDrive(straight())
    block = {"on": True}
    v, a = run(d, 150, stop=lambda dd: 30.0 if block["on"] else None)
    assert d.speed_mps == 0.0 and 29.98 <= d.arc_m <= 30.0 and a.min() >= -Limits().braking_mps2 - 1e-6
    block["on"] = False
    run(d, 300)
    assert d.arrived


def test_late_obstruction_uses_emergency_braking_within_limits():
    d = LiveDrive(straight())
    run(d, 60)
    v = d.speed_mps
    s = d.arc_m
    stop = s + v * v / (2 * 3.5)
    vv, aa = run(d, 40, stop=lambda dd: stop)
    assert d.speed_mps == 0.0 and aa.min() >= -4.5 - 1e-6 and d.emergency_count > 0
    d2 = LiveDrive(straight())
    run(d2, 60)
    v0 = d2.speed_mps
    d2.advance(0.1, stop_before_arc_m=d2.arc_m + 0.5)
    assert d2.overruns == 1 and abs((v0 - d2.speed_mps) / 0.1 - 4.5) < 1e-6


def lookahead(bodies, speed=5.0, s0=0.0):
    r = straight()
    from awr.life.civil.driving_graph import route_arrays

    r = route_arrays(r["points"], r["yaw"])
    ext = np.stack([CAR] + [PERSON for _ in bodies])
    px = np.r_[r["points"][0, 0], [b[0] for b in bodies]]
    py = np.r_[0.0, [b[1] for b in bodies]]
    pyaw = np.r_[0.0, [b[2] for b in bodies]]
    ps = np.r_[speed, [b[3] for b in bodies]]
    n = len(px)
    pred = K.predict_bodies(px, py, pyaw, ps, pyaw, np.ones(n, bool), TIMES, 2.0)
    return stop_arc(
        0,
        r["points"],
        r["yaw"],
        r["arc"],
        s0,
        speed,
        30.0,
        ext,
        px,
        py,
        pyaw,
        ps,
        np.r_[False, np.ones(n - 1, bool)],
        np.ones(n, bool),
        np.zeros(n, bool),
        *pred,
        TIMES,
        PERSON_ROOM_M,
        PERSON_ROOM_S,
        0.3,
        0.2,
    )


def test_lookahead_sees_standing_and_crossing_people_only():
    s = lookahead([(20.0, 0.0, 1.57, 0.0)])
    room = PERSON_ROOM_M + PERSON_ROOM_S * 5.0
    assert 20.0 - 2.1 - 0.3 - room - 0.25 <= s <= 20.0 - 2.1 - 0.12 - room + 0.01
    assert lookahead([(20.0, 3.5, 0.0, 0.0)]) == math.inf
    # crossing from the kerb: at the vehicle's arrival time it is on the carriageway
    assert lookahead([(12.0, -2.5, math.pi / 2, 1.2)]) < 12.0
    # walking away from the carriageway
    assert lookahead([(12.0, -2.5, -math.pi / 2, 1.2)]) == math.inf


def test_a_body_already_within_its_room_stops_only_a_motion_that_comes_nearer():
    """cruise3-r11f v1, vehicle-045 at 815 s: a person stepped up beside the standing car, closer than the person room,
    and held it for 60 s although its straight route ahead never comes nearer. Such a body lets the car crawl on; it
    stops a route that turns into it, and a motion that stays within CLOSE_FLOOR_M of it, but not driving off from
    it (cq3p-d8-s4 v1: person-086, 1.3 cm behind vehicle-042, held it 60 s). stop_blocker names the body that sets
    the stop."""
    from awr.life.civil.driving_graph import route_arrays

    def look(route, bodies):
        ext = np.stack([CAR] + [PERSON for _ in bodies])
        px = np.r_[0.0, [b[0] for b in bodies]]
        py = np.r_[0.0, [b[1] for b in bodies]]
        pyaw = np.r_[0.0, [b[2] for b in bodies]]
        ps = np.zeros(len(px))
        n = len(px)
        pred = K.predict_bodies(px, py, pyaw, ps, pyaw, np.ones(n, bool), TIMES, 2.0)
        args = (
            0,
            route["points"],
            route["yaw"],
            route["arc"],
            0.0,
            0.0,
            4.0,
            ext,
            px,
            py,
            pyaw,
            ps,
            np.r_[False, np.ones(n - 1, bool)],
            np.ones(n, bool),
            np.zeros(n, bool),
            *pred,
            TIMES,
            PERSON_ROOM_M,
            PERSON_ROOM_S,
            0.3,
            0.2,
        )
        return stop_arc(*args), stop_blocker(*args)

    r = straight()
    line = route_arrays(r["points"], r["yaw"])
    beside = (0.0, -(0.9 + 0.29 + 0.15), 0.0, 0.0)  # 0.15 m from the car's side, within the 0.25 m person room
    assert look(line, [beside]) == (CRAWL_M, -1)
    assert look(line, [(0.0, -(0.9 + 0.29 + 0.03), 0.0, 0.0)]) == (0.0, 1)  # within the floor
    assert look(line, [(-(2.1 + 0.24 + 0.02), 0.0, 0.0, 0.0)]) == (CRAWL_M, -1)  # pressed against the back: drives off
    assert look(line, [(-(2.1 + 0.24 + 0.02), 0.0, 0.0, 0.0), (2.1 + 0.12 + 0.02, 0.0, 0.0, 0.0)]) == (
        0.0,
        2,
    )  # and against the front
    # cq3p-d12-s3 v4, person-128 behind vehicle-049: a person walking up to the back of the car does not hold it
    x0 = straight()["points"][0, 0]
    assert lookahead([(x0 - 2.54, 0.77, 0.46, 0.12)], speed=0.15) > 0.0
    assert lookahead([(x0 + 2.1 + 0.59 + 0.3, 0.0, math.pi, 1.0)], speed=0.0) == 0.0  # one walking into its front does
    assert look(line, [beside, (2.1 + 0.12 + 0.15, 0.0, 0.0, 0.0)]) == (0.0, 2)  # the person ahead comes nearer
    t = np.linspace(0.0, 1.0, 120)  # a right turn of radius 6 m: the car swings into the person beside it
    bend = route_arrays(np.stack([6 * np.sin(t), -6 + 6 * np.cos(t)], 1), -t)
    stop, who = look(bend, [beside])
    assert stop < CRAWL_M and who == 1
    assert look(line, [(0.0, -(0.9 + 0.29 + 0.4), 0.0, 0.0)]) == (math.inf, -1)  # beyond the room: unchanged


def test_route_prediction_follows_geometry():
    t = np.linspace(0.0, math.pi / 2, 200)
    pts = np.stack([10 * np.sin(t), 10 - 10 * np.cos(t)], 1)
    from awr.life.civil.driving_graph import route_arrays

    r = route_arrays(pts, t)
    qx, qy, qyaw = predict_route(r["points"], r["yaw"], r["arc"], 0.0, 4.0, TIMES)
    x, y, th = route_at(r["points"], r["yaw"], r["arc"], 10.0)
    assert abs(math.hypot(qx[-1], qy[-1] - 10.0) - 10.0) < 1e-2 and abs(qx[-1] - x) < 1e-9 and abs(th - 1.0) < 1e-3


def test_car_and_crossing_people_share_a_street_without_contact():
    """A car drives along a 7 m street while people cross it in both directions: the car stops for them,
    they keep clear of it while it moves, every tick passes the QA collision predicate."""
    from types import SimpleNamespace

    from shapely.geometry import box as rect

    from awr.life.civil import body as bodies
    from awr.life.civil.collision import collision_pairs
    from awr.life.civil.driving_graph import route_arrays
    from awr.life.motion.crowd import DT, TURN_RATE_RPS, Crowd, PathStore

    cell = 0.25
    L, W = 40.0, 7.0
    nx, ny = int(L / cell), int(W / cell)
    walk = np.ones((ny, nx), bool)
    surface = SimpleNamespace(
        x0=0.0,
        y0=-W / 2,
        cell_m=cell,
        nx=nx,
        ny=ny,
        width_m=L,
        height_m=W,
        z=np.zeros((ny, nx)),
        walk={"person": walk, "vehicle": walk},
    )
    walking = SimpleNamespace(free=None, origin=None, step=0.5, area=rect(0.0, -W / 2, L, W / 2))
    starts = [(14.0, -2.7), (16.0, 2.7), (18.5, -2.7), (21.0, 2.7)]
    goals = [(14.5, 2.7), (15.5, -2.7), (18.0, 2.7), (21.5, -2.7)]
    n = len(starts) + 1
    car = n - 1
    ext = np.stack([PERSON] * (n - 1) + [CAR])
    kinds = np.array(["person"] * (n - 1) + ["vehicle"])
    crowd = Crowd(ext, kinds, surface, walking)
    x = np.arange(3.0, 37.01, 0.5)
    route = route_arrays(np.stack([x, np.full_like(x, 1.2)], 1), np.zeros_like(x))
    drive = LiveDrive(route)
    xy = np.array(starts + [(3.0, 1.2)], float)
    yaw = np.r_[[math.atan2(g[1] - s[1], g[0] - s[0]) for s, g in zip(starts, goals, strict=False)], 0.0]
    speed = np.zeros(n)
    vmax = np.r_[np.full(n - 1, 1.3), 0.0]
    mover = kinds == "person"
    paths = PathStore(n)
    for i in range(n - 1):
        paths.set(i, np.array([starts[i], goals[i]], float))
    arrived = [None] * (n - 1)
    stops = 0
    prev = None
    for k in range(900):
        travel = [i for i in range(n - 1) if arrived[i] is None]
        des_yaw, des_speed, cap = yaw.copy(), np.zeros(n), np.full(n, np.inf)
        anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(n)
        if travel:
            led = np.array(travel, np.int64)
            h, want, c, an, tg, ln = paths.desire(led, xy, speed, vmax, yaw)
            cap[led] = c[led]
            ok = led[~np.isnan(h[led])]
            des_yaw[ok] = h[ok]
            anchor[ok] = an[ok]
            tangent[ok] = tg[ok]
            lane[ok] = ln[ok]
            des_speed[ok] = want[ok]  # a large heading change is made standing first (Crowd.step)
        s0, v0 = drive.arc_m, drive.speed_mps
        pred_car = predict_route(route["points"], route["yaw"], route["arc"], s0, v0, crowd.times)
        pp = K.predict_bodies(
            xy[:, 0].copy(), xy[:, 1].copy(), yaw, speed, crowd.aim, mover & (speed > 0), crowd.times, TURN_RATE_RPS
        )
        stop = stop_arc(
            car,
            route["points"],
            route["yaw"],
            route["arc"],
            s0,
            v0,
            v0 * v0 / 4.0 + v0 * 2.5 + 4.0,
            ext,
            xy[:, 0].copy(),
            xy[:, 1].copy(),
            yaw,
            speed,
            mover,
            np.ones(n, bool),
            np.zeros(n, bool),
            *pp,
            crowd.times,
            PERSON_ROOM_M,
            PERSON_ROOM_S,
            0.3,
            0.2,
        )
        if not drive.arrived:
            st = drive.advance(DT, stop_before_arc_m=None if math.isinf(stop) else stop)
        else:
            st = drive.state()
        stops += st["mode"] == "waiting"
        driven = (
            np.array([car]),
            st["xy"][None],
            np.array([st["yaw"]]),
            np.array([st["speed_mps"]]),
            pred_car[0][None],
            pred_car[1][None],
            pred_car[2][None],
        )
        new_xy, new_yaw, new_speed, held = crowd.step(
            xy, yaw, speed, np.ones(n, bool), mover, des_yaw, des_speed, vmax, cap, anchor, tangent, lane, driven=driven
        )
        assert not held
        v = (new_xy - xy) / DT
        sp = np.linalg.norm(v, axis=1)
        c, sz = bodies.boxes(new_xy, new_yaw, sp, ext, np.full(n, 1.7))
        assert collision_pairs(c, new_yaw, sz) == [], k
        if prev is not None:
            assert (np.linalg.norm(v - prev, axis=1) / DT)[:-1].max() <= 8.0
        prev = v
        xy, yaw, speed = new_xy, new_yaw, new_speed
        for i in travel:
            if np.linalg.norm(xy[i] - goals[i]) < 0.05 and speed[i] < 0.08:
                arrived[i] = k
                paths.pop(i)
        if drive.arrived and None not in arrived:
            break
    print(
        "final",
        xy.round(2).tolist(),
        yaw.round(2).tolist(),
        des_yaw.round(2).tolist(),
        des_speed.round(2).tolist(),
        arrived,
        stops,
        crowd.stats,
    )
    assert drive.arrived and None not in arrived and stops > 0


def test_a_trip_continues_on_a_new_route_from_a_splice_ahead():
    # The new route follows the present one for 30 m, then bends away; the car at speed carries over onto it.
    import pytest

    old = straight(60.0)
    x = np.arange(0.0, 30.0 + 1e-9, 0.5)
    u = np.linspace(0.0, np.pi / 2, 61)[1:]
    bend = np.r_[np.stack([x, np.zeros_like(x)], 1), np.stack([30.0 + 20.0 * np.sin(u), 20.0 - 20.0 * np.cos(u)], 1)]
    new = dict(points=bend, yaw=np.r_[np.zeros_like(x), u])
    d = LiveDrive(old)
    run(d, 40)
    before = d.state()
    assert 0.0 < before["arc_m"] < 20.0 and before["speed_mps"] > 0.0
    with pytest.raises(ValueError):
        d.reroute(new, before["arc_m"])  # a splice must lie ahead of the body
    with pytest.raises(ValueError):
        d.reroute(dict(points=bend + np.array([0.0, 0.01]), yaw=new["yaw"]), 30.0)  # and the way to it be the same
    d.reroute(new, 30.0)
    after = d.state()
    assert np.allclose(after["xy"], before["xy"]) and after["speed_mps"] == before["speed_mps"]
    assert after["elapsed_s"] == before["elapsed_s"] and d.length_m > 60.0
    v, a = run(d, 600)
    assert d.arrived and np.allclose(d.state()["xy"], bend[-1]) and a.min() >= -Limits().braking_mps2 - 1e-9


def test_walkers_keep_out_of_the_way_a_waiting_car_is_about_to_take():
    """cq3p v5, vehicle-042: in a dense crowd walkers stepped into the way of the standing car, whose lookahead then
    held it 60 s. A car that has stood ASK_AFTER_S in its trip is predicted setting off (car_intent_mps): a walker
    crossing just ahead of it waits outside its way instead of stepping into it."""
    from types import SimpleNamespace

    from shapely.geometry import box as rect

    from awr.life.civil.driving_graph import route_arrays
    from awr.life.motion.courtesy import ASK_AFTER_S
    from awr.life.motion.crowd import DT, Crowd, PathStore
    from awr.life.motion.engine import WAITING_CAR_MPS, car_intent_mps

    assert (
        car_intent_mps(0.0, round(ASK_AFTER_S / DT) - 1) == 0.0
        and car_intent_mps(0.0, round(ASK_AFTER_S / DT)) == WAITING_CAR_MPS
    )
    assert car_intent_mps(3.0, round(ASK_AFTER_S / DT)) == 3.0
    cell = 0.25
    L, W = 40.0, 7.0
    nx, ny = int(L / cell), int(W / cell)
    walk = np.ones((ny, nx), bool)
    surface = SimpleNamespace(
        x0=0.0,
        y0=-W / 2,
        cell_m=cell,
        nx=nx,
        ny=ny,
        width_m=L,
        height_m=W,
        z=np.zeros((ny, nx)),
        walk={"person": walk, "vehicle": walk},
    )
    walking = SimpleNamespace(free=None, origin=None, step=0.5, area=rect(0.0, -W / 2, L, W / 2))
    x = np.arange(3.0, 37.01, 0.5)
    route = route_arrays(np.stack([x, np.zeros_like(x)], 1), np.zeros_like(x))
    ext = np.stack([PERSON, CAR])
    kinds = np.array(["person", "vehicle"])

    def entered(intent):
        """Whether the walker crossing 1 m ahead of the standing car steps into its way (|y| < its half width)."""
        crowd = Crowd(ext, kinds, surface, walking)
        xy = np.array([(13.1, -3.0), (10.0, 0.0)])
        yaw = np.array([math.pi / 2, 0.0])
        speed = np.zeros(2)
        paths = PathStore(2)
        paths.set(0, np.array([(13.1, -3.0), (13.1, 3.0)]))
        mover = kinds == "person"
        vmax = np.r_[1.3, 0.0]
        for _ in range(150):
            led = np.array([0], np.int64)
            h, want, c, an, tg, ln = paths.desire(led, xy, speed, vmax, yaw)
            des_yaw, des_speed, cap = yaw.copy(), np.zeros(2), np.full(2, np.inf)
            des_yaw[0], des_speed[0], cap[0] = h[0], want[0], c[0]
            anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(2)
            anchor[0], tangent[0], lane[0] = an[0], tg[0], ln[0]
            pred = predict_route(route["points"], route["yaw"], route["arc"], 7.0, intent, crowd.times)
            driven = (
                np.array([1]),
                xy[1][None],
                np.array([0.0]),
                np.array([0.0]),
                pred[0][None],
                pred[1][None],
                pred[2][None],
            )
            xy, yaw, speed, _ = crowd.step(
                xy,
                yaw,
                speed,
                np.ones(2, bool),
                mover,
                des_yaw,
                des_speed,
                vmax,
                cap,
                anchor,
                tangent,
                lane,
                driven=driven,
            )
            if abs(xy[0, 1]) < 0.9:
                return True
        return False

    assert entered(0.0)  # seen standing, the car's way is walked into
    assert not entered(WAITING_CAR_MPS)


def test_a_walker_already_in_the_way_of_a_car_about_to_set_off_leaves_it():
    """cq3p-s3 v1, person-064 before vehicle-053: a walker standing just ahead of the waiting car, on its way past it,
    saw the car setting off (car_intent_mps) and every step it could take met it, so it froze there and the car
    waited 60 s. A walker already in the way (LifeEngine.in_way) sees the car where it stands and walks off its way
    beside it; a walker outside the way still keeps out of it."""
    from types import SimpleNamespace

    from shapely.geometry import box as rect

    from awr.life.civil.driving_graph import route_arrays
    from awr.life.motion.crowd import DT, Crowd, PathStore
    from awr.life.motion.engine import WAITING_CAR_MPS, LifeEngine

    cell = 0.25
    L, W = 40.0, 9.0
    nx, ny = int(L / cell), int(W / cell)
    walk = np.ones((ny, nx), bool)
    surface = SimpleNamespace(
        x0=0.0,
        y0=-W / 2,
        cell_m=cell,
        nx=nx,
        ny=ny,
        width_m=L,
        height_m=W,
        z=np.zeros((ny, nx)),
        walk={"person": walk, "vehicle": walk},
    )
    walking = SimpleNamespace(free=None, origin=None, step=0.5, area=rect(0.0, -W / 2, L, W / 2))
    x = np.arange(3.0, 37.01, 0.5)
    route = route_arrays(np.stack([x, np.zeros_like(x)], 1), np.zeros_like(x))
    ahead = np.stack([np.arange(10.0, 18.1 + 1e-9, 0.25), np.zeros(33)], 1)  # car centre to 6 m beyond its front
    ext = np.stack([PERSON, CAR])
    kinds = np.array(["person", "vehicle"])

    def walk_off(start, heading, goal, rule):
        """Seconds until the walker is clear of the car's way (|y| beyond its half width), None within 20 s."""
        crowd = Crowd(ext, kinds, surface, walking)
        engine = SimpleNamespace(kinds=kinds, active=np.ones(2, bool), ext=ext, crowd=crowd)
        xy = np.array([start, (10.0, 0.0)])
        yaw = np.array([heading, 0.0])
        speed = np.zeros(2)
        paths = PathStore(2)
        paths.set(0, np.array([start, goal]))
        mover = kinds == "person"
        vmax = np.r_[1.3, 0.0]
        for k in range(200):
            led = np.array([0], np.int64)
            h, want, c, an, tg, ln = paths.desire(led, xy, speed, vmax, yaw)
            des_yaw, des_speed, cap = yaw.copy(), np.zeros(2), np.full(2, np.inf)
            des_yaw[0], des_speed[0], cap[0] = h[0], want[0], c[0]
            anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(2)
            anchor[0], tangent[0], lane[0] = an[0], tg[0], ln[0]
            pred = predict_route(route["points"], route["yaw"], route["arc"], 7.0, WAITING_CAR_MPS, crowd.times)
            driven = (
                np.array([1]),
                xy[1][None],
                np.array([0.0]),
                np.array([0.0]),
                pred[0][None],
                pred[1][None],
                pred[2][None],
            )
            seen = LifeEngine.in_way(engine, {1: ahead}, xy, yaw, speed) if rule else None
            xy, yaw, speed, _ = crowd.step(
                xy,
                yaw,
                speed,
                np.ones(2, bool),
                mover,
                des_yaw,
                des_speed,
                vmax,
                cap,
                anchor,
                tangent,
                lane,
                driven=driven,
                seen_standing=seen,
            )
            box = K.box(xy[0, 0], xy[0, 1], yaw[0], speed[0], PERSON)
            if abs(box[1]) - max(box[2], box[3]) > 0.9:
                return k * DT
        return None

    # just ahead of the front bumper, on its way past the car's side
    assert walk_off((12.5, -0.9), math.radians(150), (5.0, -2.0), rule=False) is None
    assert walk_off((12.5, -0.9), math.radians(150), (5.0, -2.0), rule=True) is not None

    def entered():
        """Whether the walker crossing 1 m ahead of the standing car steps into its way (|y| < its half width)."""
        crowd = Crowd(ext, kinds, surface, walking)
        engine = SimpleNamespace(kinds=kinds, active=np.ones(2, bool), ext=ext, crowd=crowd)
        xy = np.array([(13.1, -3.0), (10.0, 0.0)])
        yaw = np.array([math.pi / 2, 0.0])
        speed = np.zeros(2)
        paths = PathStore(2)
        paths.set(0, np.array([(13.1, -3.0), (13.1, 3.0)]))
        mover = kinds == "person"
        vmax = np.r_[1.3, 0.0]
        for _ in range(150):
            led = np.array([0], np.int64)
            h, want, c, an, tg, ln = paths.desire(led, xy, speed, vmax, yaw)
            des_yaw, des_speed, cap = yaw.copy(), np.zeros(2), np.full(2, np.inf)
            des_yaw[0], des_speed[0], cap[0] = h[0], want[0], c[0]
            anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(2)
            anchor[0], tangent[0], lane[0] = an[0], tg[0], ln[0]
            pred = predict_route(route["points"], route["yaw"], route["arc"], 7.0, WAITING_CAR_MPS, crowd.times)
            driven = (
                np.array([1]),
                xy[1][None],
                np.array([0.0]),
                np.array([0.0]),
                pred[0][None],
                pred[1][None],
                pred[2][None],
            )
            xy, yaw, speed, _ = crowd.step(
                xy,
                yaw,
                speed,
                np.ones(2, bool),
                mover,
                des_yaw,
                des_speed,
                vmax,
                cap,
                anchor,
                tangent,
                lane,
                driven=driven,
                seen_standing=LifeEngine.in_way(engine, {1: ahead}, xy, yaw, speed),
            )
            if abs(xy[0, 1]) < 0.9:
                return True
        return False

    assert not entered()

    # cq3p-d8-s4 v1, person-022 beside vehicle-042: the walker whose body sets the car's stop sees the car standing
    # although outside its way, and crosses; the car waits for it, so it does not wait for the car
    walker = (12.1 + 0.24 + 0.1, -(0.9 + 0.29 + 0.15))
    engine = SimpleNamespace(kinds=kinds, active=np.ones(2, bool), ext=ext, crowd=Crowd(ext, kinds, surface, walking))
    engine.in_way = lambda cars, xy, yaw, speed: LifeEngine.in_way(engine, cars, xy, yaw, speed)
    xy, yaw, speed = np.array([walker, (10.0, 0.0)]), np.array([math.pi / 2, 0.0]), np.zeros(2)
    assert LifeEngine.in_way(engine, {1: ahead}, xy, yaw, speed)[0] == -1
    assert LifeEngine.seen_standing(engine, {1: ahead}, {1: dict(blocker=0)}, xy, yaw, speed)[0] == 1
    assert LifeEngine.seen_standing(engine, {1: ahead}, {1: dict(blocker=-1)}, xy, yaw, speed)[0] == -1


def test_a_walker_plans_round_a_car_standing_in_its_trip_and_behind_it():
    """cq3p-d8-s3 v2, person-009 before vehicle-043: the static snapshot holds only parked cars, so a walker that
    planned its way on beside a car standing in its trip was given a route through the car; it stood against the
    car's front corner, and the car waited for it 60 s. A route through the body or the way ahead of a standing car
    is planned again round them (LifeEngine.car_cells, _round_cars); a route clear of them is kept."""
    from types import SimpleNamespace

    from shapely.geometry import LineString
    from shapely.geometry import box as rect

    from awr.life.civil.walking import WalkNetwork
    from awr.life.motion.courtesy import CAR_AHEAD_M
    from awr.life.motion.engine import LifeEngine
    from awr.life.motion.route import LifeRouter

    net = WalkNetwork(np.ones((40, 80), bool), np.array([0.0, 0.0]), step=0.5)
    router = LifeRouter(net, net.free, landmarks=2)
    car, hl, hw = np.array([20.0, 10.0]), CAR[0, 1], CAR[0, 3]
    s = np.arange(0.0, hl + CAR_AHEAD_M + 1e-9, 1.0)
    ways = [(1, np.column_stack([car[0] + s, np.full(len(s), car[1])]), np.zeros(len(s)))]
    engine = SimpleNamespace(router=router, standing_ways=ways, ext=np.stack([PERSON, CAR]), _car_cells=None, stats={})
    cells = LifeEngine.car_cells(engine)
    body = rect(car[0] - hl, car[1] - hw, car[0] + hl, car[1] + hw)
    way = rect(car[0] + hl, car[1] - hw, car[0] + hl + CAR_AHEAD_M, car[1] + hw)
    start, goal = (
        car + np.array([hl + 0.4, hw + 0.3]),
        car + np.array([-hl - 3.0, -hw - 3.0]),
    )  # front left corner, behind on the right
    straight = router.plan(start, goal)
    assert LineString(straight["points"]).intersects(body)
    line = LineString(LifeEngine._round_cars(engine, start, goal, straight, cells)["points"])
    assert not line.intersects(body)
    assert line.intersection(way).length < 0.5  # by the car's side and behind it, not across its way
    clear = router.plan(car + np.array([0.0, 6.0]), car + np.array([-10.0, 6.0]))
    assert (
        LifeEngine._round_cars(engine, car + np.array([0.0, 6.0]), car + np.array([-10.0, 6.0]), clear, cells) is clear
    )
    # a walker whose present path runs through the car within its next JAM_AHEAD_M plans anew at once
    from awr.life.motion.crowd import PathStore
    from awr.life.motion.reroute import Reroutes

    engine.paths, engine.reroutes = PathStore(3), Reroutes(router)
    engine.paths.set(0, np.asarray(straight["points"]))
    engine.paths.set(1, np.asarray(clear["points"]))
    engine.paths.set(
        2, np.array([car + np.array([-30.0, 0.0]), car + np.array([-30.0, 9.0]), car + np.array([10.0, 9.0])])
    )
    avoided = np.zeros(len(router.cost), bool)
    avoided[cells] = True
    assert [LifeEngine._through_cars(engine, i, avoided) for i in range(3)] == [True, False, False]
