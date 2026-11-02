import math
from types import SimpleNamespace

import numpy as np
from shapely.geometry import box

from awr.life.civil import body as bodies
from awr.life.civil.collision import collision_pairs, obb_overlap
from awr.life.motion import kernels as K
from awr.life.motion.crowd import DT, Crowd, Path

EXT = np.array([[-0.12, 0.24, -0.29, 0.29], [-0.42, 0.59, -0.31, 0.31]])


def test_box_and_overlap_match_the_qa_predicates():
    rng = np.random.default_rng(1)
    for _ in range(400):
        p = rng.uniform(-1, 1, (2, 2))
        y = rng.uniform(-3, 3, 2)
        v = rng.uniform(0, 1.2, 2)
        c, s = bodies.boxes(p, y, v, np.stack([EXT, EXT]))
        mine = [K.box(p[i, 0], p[i, 1], y[i], v[i], EXT) for i in range(2)]
        for i in range(2):
            np.testing.assert_allclose(mine[i][:2], c[i], atol=1e-12)
            np.testing.assert_allclose([2 * mine[i][2], 2 * mine[i][3]], s[i, :2], atol=1e-12)
        a, b = mine
        expect = bool(obb_overlap(c[0], y[0], s[0], c[1], y[1], s[1]))
        assert K.overlap(a[0], a[1], y[0], a[2], a[3], b[0], b[1], y[1], b[2], b[3], 0.005) == expect


def corridor(width=3.2, length=30.0):
    cell = 0.25
    nx, ny = int(length / cell), int(12.0 / cell)
    walk = np.zeros((ny, nx), bool)
    y = (np.arange(ny) + 0.5) * cell - 6.0
    walk[np.abs(y) <= width / 2, :] = True
    surface = SimpleNamespace(
        x0=0.0,
        y0=-6.0,
        cell_m=cell,
        nx=nx,
        ny=ny,
        width_m=nx * cell,
        height_m=ny * cell,
        z=np.zeros((ny, nx)),
        walk={"person": walk, "vehicle": np.zeros_like(walk)},
    )
    step = 0.5
    fy = np.arange(int(12.0 / step)) * step - 6.0 + step / 2
    free = np.zeros((len(fy), int(length / step)), bool)
    free[np.abs(fy) <= width / 2 - 0.4, 1:-1] = True
    walking = SimpleNamespace(
        free=free, origin=np.array([step / 2, -6.0 + step / 2]), step=step, area=box(0.0, -width / 2, length, width / 2)
    )
    return surface, walking


def simulate(starts, goals, still=(), ticks=400):
    surface, walking = corridor()
    n = len(starts) + len(still)
    ext = np.stack([EXT] * n)
    crowd = Crowd(ext, np.array(["person"] * n), surface, walking)
    xy = np.array(list(starts) + list(still), float)
    yaw = np.array(
        [math.atan2(g[1] - s[1], g[0] - s[0]) for s, g in zip(starts, goals, strict=False)] + [0.0] * len(still)
    )
    speed = np.zeros(n)
    paths = [Path(np.array([s, g], float)) for s, g in zip(starts, goals, strict=False)]
    mover = np.ones(n, bool)
    vmax = np.full(n, 1.3)
    track, vel = [xy.copy()], []
    for _ in range(ticks):
        des_yaw, des_speed, cap = yaw.copy(), np.zeros(n), np.full(n, np.inf)
        anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(n)
        for i, p in enumerate(paths):
            h, want, c = p.desire(xy[i], speed[i], vmax[i], yaw[i])
            if h is not None:
                des_yaw[i] = h
                des_speed[i] = want
                anchor[i] = p.anchor
                tangent[i] = p.tangent
                lane[i] = p.lane
            cap[i] = c
        new_xy, new_yaw, new_speed, held = crowd.step(
            xy, yaw, speed, np.ones(n, bool), mover, des_yaw, des_speed, vmax, cap, anchor, tangent, lane
        )
        assert not held
        vel.append((new_xy - xy) / DT)
        c, s = bodies.boxes(new_xy, new_yaw, np.linalg.norm(vel[-1], axis=1), ext, np.full(n, 1.7))
        assert collision_pairs(c, new_yaw, s) == []
        assert np.all(np.abs((new_yaw - yaw + np.pi) % (2 * np.pi) - np.pi) <= 3.0 * DT + 1e-9)
        xy, yaw, speed = new_xy, new_yaw, new_speed
        track.append(xy.copy())
    v = np.array(vel)
    acc = np.linalg.norm(np.diff(np.round(v * 100) * 0.01, axis=0), axis=2) / DT
    assert acc.max() <= 8.0
    return np.array(track)


def test_two_walkers_pass_head_on_in_a_corridor():
    track = simulate([(3.0, 0.0), (27.0, 0.0)], [(27.0, 0.0), (3.0, 0.0)])
    assert np.linalg.norm(track[-1, 0] - (27.0, 0.0)) < 0.1 and np.linalg.norm(track[-1, 1] - (3.0, 0.0)) < 0.1


def test_walker_passes_a_standing_person():
    track = simulate([(3.0, 0.0)], [(27.0, 0.0)], still=[(15.0, 0.0)])
    assert np.linalg.norm(track[-1, 0] - (27.0, 0.0)) < 0.1
    assert np.allclose(track[:, 1], (15.0, 0.0))


def test_compiled_path_following_matches_path():
    from awr.life.motion.crowd import PathStore

    rng = np.random.default_rng(5)
    n = 6
    store = PathStore(n, capacity=64)
    refs = {}
    xy = rng.uniform(0, 20, (n, 2))
    for step in range(60):
        if step % 15 == 0:
            for i in range(n):
                pts = np.vstack([xy[i], np.cumsum(rng.uniform(-3, 3, (rng.integers(1, 7), 2)), axis=0) + xy[i]])
                refs[i] = Path(pts)
                store.set(i, pts)
        speed = rng.uniform(0, 1.3, n)
        yaw = rng.uniform(-np.pi, np.pi, n)
        h, w, c, an, tg, ln = store.desire(np.arange(n), xy, speed, np.full(n, 1.3), yaw)
        for i in range(n):
            rh, rw, rc = refs[i].desire(xy[i], speed[i], 1.3, yaw[i])
            assert (rh is None and np.isnan(h[i])) or abs(rh - h[i]) < 1e-12
            assert abs(rw - w[i]) < 1e-12 and abs(rc - c[i]) < 1e-9
            np.testing.assert_allclose(an[i], refs[i].anchor, atol=1e-12)
            assert (refs[i].tangent is None and np.isnan(tg[i])) or abs(refs[i].tangent - tg[i]) < 1e-12
            assert abs(refs[i].s - store.s[i]) < 1e-12 and abs(refs[i].lane - ln[i]) < 1e-12
        xy = xy + rng.normal(0, 0.3, (n, 2))


def walk_scene(surface, walking, starts, goals, ticks):
    """The runtime's motion loop (PathStore, Standoffs, Crowd) on a synthetic scene; checks every tick."""
    from awr.life.motion.crowd import PathStore
    from awr.life.motion.standoff import Standoffs

    class Agenda:
        def __init__(self):
            self.events = []

        def blocked(self, t, reason):
            self.events.append(reason)

    n = len(starts)
    ext = np.stack([EXT] * n)
    crowd = Crowd(ext, np.array(["person"] * n), surface, walking)
    standoffs = Standoffs(crowd)
    xy = np.array(starts, float)
    yaw = np.array([math.atan2(g[1] - a[1], g[0] - a[0]) for a, g in zip(starts, goals, strict=False)])
    speed = np.zeros(n)
    vmax = np.full(n, 1.3)
    stall = np.zeros(n, int)
    tangent = yaw.copy()
    paths = PathStore(n)
    for i in range(n):
        paths.set(i, np.array([starts[i], goals[i]], float))
    agendas = {i: Agenda() for i in range(n)}
    arrived = [None] * n
    prev = None
    for k in range(ticks):
        travel = [i for i in range(n) if arrived[i] is None]
        for i, pts in standoffs.update(k * DT, travel, xy, stall, paths, tangent, agendas, DT).items():
            paths.set(i, pts)
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
        new_xy, new_yaw, new_speed, held = crowd.step(
            xy, yaw, speed, np.ones(n, bool), np.ones(n, bool), des_yaw, des_speed, vmax, cap, anchor, tangent, lane
        )
        assert not held
        v = (new_xy - xy) / DT
        c, sz = bodies.boxes(new_xy, new_yaw, np.linalg.norm(v, axis=1), ext, np.full(n, 1.7))
        assert collision_pairs(c, new_yaw, sz) == []
        if prev is not None:
            assert (np.linalg.norm(np.round(v * 100) * 0.01 - np.round(prev * 100) * 0.01, axis=1) / DT).max() <= 8.0
        prev = v
        moving = np.zeros(n, bool)
        moving[travel] = True
        stall = np.where(moving & (np.linalg.norm(v, axis=1) < 0.2), stall + 1, 0)
        xy, yaw, speed = new_xy, new_yaw, new_speed
        for i in travel:
            if np.linalg.norm(xy[i] - goals[i]) < 0.05 and speed[i] < 0.08:
                arrived[i] = k
    return arrived, agendas, standoffs


def test_two_walkers_pass_each_other_in_a_1_6_m_passage():
    surface, walking = corridor(width=1.6)
    arrived, _, standoffs = walk_scene(surface, walking, [(3.0, 0.0), (27.0, 0.0)], [(27.0, 0.0), (3.0, 0.0)], 400)
    assert None not in arrived and standoffs.stats["gave_way"] == 0


def test_one_walker_gives_way_in_a_single_file_passage():
    from shapely.ops import unary_union

    surface, walking = corridor(width=3.2)
    cell = surface.cell_m
    xs = (np.arange(surface.nx) + 0.5) * cell
    ys = (np.arange(surface.ny) + 0.5) * cell - 6.0
    X, Y = np.meshgrid(xs, ys)
    surface.walk["person"][(X > 10.0) & (X < 20.0) & (np.abs(Y) > 0.6)] = False
    walking.area = unary_union([box(0, -1.6, 10, 1.6), box(10, -0.6, 20, 0.6), box(20, -1.6, 30, 1.6)])
    arrived, agendas, standoffs = walk_scene(
        surface, walking, [(3.0, 0.0), (24.0, 0.0)], [(27.0, 0.0), (3.0, 0.0)], 600
    )
    assert None not in arrived
    assert standoffs.stats["gave_way"] == 1 and standoffs.stats["released_passed"] == 1
    assert sum(a.events.count("gave_way") for a in agendas.values()) == 1


def test_two_people_locked_abeam_edge_apart():
    """Two stopped walkers about 3 cm apart, each needing to turn into the other (seed-2 proxy-town lock): any turn
    or start would swing a pose box into the other, so they edge aside at rest-pose speed, then walk off."""
    surface, walking = corridor(width=10.0, length=30.0)
    a, b = np.array([15.203, 0.315]), np.array([14.711, 0.619])
    ya, yb = 0.497, -0.31
    goals = [
        a + 10.0 * np.array([math.cos(2.17), math.sin(2.17)]) * 0.45,
        b + 10.0 * np.array([math.cos(0.35), math.sin(0.35)]),
    ]
    n = 2
    ext = np.stack([EXT] * n)
    crowd = Crowd(ext, np.array(["person"] * n), surface, walking)
    xy = np.array([a, b])
    yaw = np.array([ya, yb])
    speed = np.zeros(n)
    paths = [Path(np.array([xy[i], goals[i]])) for i in range(n)]
    vmax = np.full(n, 1.3)
    edged = 0
    arrived = [None, None]
    c, s = bodies.boxes(xy, yaw, speed, ext, np.full(n, 1.7))
    assert collision_pairs(c, yaw, s) == []
    for k in range(900):
        des_yaw, des_speed, cap = yaw.copy(), np.zeros(n), np.full(n, np.inf)
        anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(n)
        for i, p in enumerate(paths):
            if arrived[i] is not None:
                continue
            h, want, cc = p.desire(xy[i], speed[i], vmax[i], yaw[i])
            if h is not None:
                des_yaw[i] = h
                des_speed[i] = want
                anchor[i] = p.anchor
                tangent[i] = p.tangent
                lane[i] = p.lane
            cap[i] = cc
        new_xy, new_yaw, new_speed, held = crowd.step(
            xy, yaw, speed, np.ones(n, bool), np.ones(n, bool), des_yaw, des_speed, vmax, cap, anchor, tangent, lane
        )
        assert not held
        edged += int(np.isfinite(crowd.motion).sum())
        v = np.linalg.norm(new_xy - xy, axis=1) / DT
        c, s = bodies.boxes(new_xy, new_yaw, v, ext, np.full(n, 1.7))
        assert collision_pairs(c, new_yaw, s) == []
        xy, yaw, speed = new_xy, new_yaw, new_speed
        for i in range(n):
            if arrived[i] is None and np.linalg.norm(xy[i] - goals[i]) < 0.05 and speed[i] < 0.08:
                arrived[i] = k * DT
    assert None not in arrived and edged > 0, (arrived, edged)


def test_walker_leaving_beside_a_turning_walker_goes():
    """One walker arrives beside a person who sets off (seed-3 proxy-town lock): the arriving walker's turn towards
    its spot and the leaving walker's start conflict, and greedy yielding stopped both; the start, clear of the
    cancelled turn, is given back (Crowd._restore), so the leaving walker goes and the other then takes the spot."""
    surface, walking = corridor(width=10.0, length=30.0)
    a, b = np.array([14.68, -0.035]), np.array([15.32, 0.035])
    goals = [b.copy(), b + 4.0 * np.array([math.cos(1.526), math.sin(1.526)])]
    n = 2
    ext = np.stack([EXT] * n)
    crowd = Crowd(ext, np.array(["person"] * n), surface, walking)
    xy = np.array([a, b])
    yaw = np.array([-1.3, 1.77])
    speed = np.zeros(n)
    paths = [Path(np.array([xy[i], goals[i]])) for i in range(n)]
    vmax = np.full(n, 1.3)
    arrived = [None, None]
    for k in range(300):
        des_yaw, des_speed, cap = yaw.copy(), np.zeros(n), np.full(n, np.inf)
        anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(n)
        for i, p in enumerate(paths):
            if arrived[i] is not None:
                continue
            h, want, cc = p.desire(xy[i], speed[i], vmax[i], yaw[i])
            if h is not None:
                des_yaw[i] = h
                des_speed[i] = want
                anchor[i] = p.anchor
                tangent[i] = p.tangent
                lane[i] = p.lane
            cap[i] = cc
        new_xy, new_yaw, new_speed, held = crowd.step(
            xy, yaw, speed, np.ones(n, bool), np.ones(n, bool), des_yaw, des_speed, vmax, cap, anchor, tangent, lane
        )
        assert not held
        v = np.linalg.norm(new_xy - xy, axis=1) / DT
        c, s = bodies.boxes(new_xy, new_yaw, v, ext, np.full(n, 1.7))
        assert collision_pairs(c, new_yaw, s) == []
        xy, yaw, speed = new_xy, new_yaw, new_speed
        for i in range(n):
            if arrived[i] is None and np.linalg.norm(xy[i] - goals[i]) < 0.05 and speed[i] < 0.08:
                arrived[i] = k * DT
    assert None not in arrived and max(arrived) < 15.0, arrived
    assert crowd.stats["restored"] > 0


def test_walker_facing_a_parked_car_steps_aside_and_walks_round():
    """A walker stopped 0.26 m in front of a parked car, its kept lane through the car's side (seed 20261102 town
    run on dmax, person-029 beside vehicle-041): its walking pose reaches 0.32 m further forward than its rest pose,
    so it can neither set off towards the car nor turn away while walking. After waiting it turns on the spot and
    steps aside, then walks round the car to its goal."""
    surface, walking = corridor(width=10.0, length=40.0)
    car_ext = np.array([[-2.13, 2.13, -1.06, 1.06]] * 2)
    ext = np.stack([EXT, car_ext])
    crowd = Crowd(ext, np.array(["person", "vehicle"]), surface, walking)
    xy = np.array([[17.365, 1.0], [20.0, 1.87]])
    yaw = np.array([0.0, math.pi])
    speed = np.zeros(2)
    goal = np.array([32.0, 0.0])
    path = Path(np.array([[10.0, 0.0], goal]))
    mover = np.array([True, False])
    vmax = np.full(2, 1.3)
    arrived = None
    still = 0
    for k in range(900):
        des_yaw, des_speed, cap = yaw.copy(), np.zeros(2), np.full(2, np.inf)
        anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(2)
        if arrived is None:
            h, want, cc = path.desire(xy[0], speed[0], vmax[0], yaw[0])
            if h is not None:
                des_yaw[0] = h
                des_speed[0] = want
                anchor[0] = path.anchor
                tangent[0] = path.tangent
                lane[0] = path.lane
            cap[0] = cc
        new_xy, new_yaw, new_speed, held = crowd.step(
            xy, yaw, speed, np.ones(2, bool), mover, des_yaw, des_speed, vmax, cap, anchor, tangent, lane
        )
        assert not held
        v = np.linalg.norm(new_xy - xy, axis=1) / DT
        c, s = bodies.boxes(new_xy, new_yaw, v, ext, np.full(2, 1.7))
        assert collision_pairs(c, new_yaw, s) == []
        still = still + 1 if v[0] < 0.2 and arrived is None else 0
        assert still * DT <= 60.0, k  # the release stall gate
        xy, yaw, speed = new_xy, new_yaw, new_speed
        if arrived is None and np.linalg.norm(xy[0] - goal) < 0.05 and speed[0] < 0.08:
            arrived = k * DT
    assert arrived is not None
