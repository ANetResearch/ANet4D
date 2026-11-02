"""Congestion rerouting (awr.life.motion.reroute): a walker held up by standing people takes another way round."""

import math
from types import SimpleNamespace

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from awr.life.civil import body as bodies
from awr.life.civil.collision import collision_pairs
from awr.life.civil.walking import WalkNetwork
from awr.life.motion import reroute as RR
from awr.life.motion.crowd import DT, Crowd, Path, PathStore
from awr.life.motion.reroute import Reroutes
from awr.life.motion.route import LifeRouter
from awr.life.motion.standoff import Standoffs

EXT = np.array([[-0.12, 0.24, -0.29, 0.29], [-0.42, 0.59, -0.31, 0.31]])


class Agenda:
    def __init__(self):
        self.events = []

    def blocked(self, t, reason):
        self.events.append(reason)


def scene(bypass, parts=None, W=40.0, H=12.0):
    """A 2 m street from x=0 to 40 (y 2-4); with `bypass`, a parallel 2 m lane (y 7-9) joined to it at x 14-16 and
    24-26. `parts` replaces the street by other walkable shapes."""
    if parts is None:
        parts = [box(0.0, 2.0, 40.0, 4.0)]
        if bypass:
            parts += [box(14.0, 2.0, 16.0, 9.0), box(24.0, 2.0, 26.0, 9.0), box(14.0, 7.0, 26.0, 9.0)]
    area = unary_union(parts)
    cell = 0.25
    gx, gy = np.meshgrid((np.arange(int(W / cell)) + 0.5) * cell, (np.arange(int(H / cell)) + 0.5) * cell)
    import shapely

    walk = shapely.contains_xy(area, gx, gy)
    surface = SimpleNamespace(
        x0=0.0,
        y0=0.0,
        cell_m=cell,
        nx=walk.shape[1],
        ny=walk.shape[0],
        width_m=W,
        height_m=H,
        z=np.zeros(walk.shape),
        walk={"person": walk, "vehicle": np.zeros_like(walk)},
    )
    step = 0.5
    fx, fy = np.meshgrid((np.arange(int(W / step)) + 0.5) * step, (np.arange(int(H / step)) + 0.5) * step)
    free = shapely.contains_xy(area.buffer(-0.4), fx, fy)
    walking = WalkNetwork(free, np.array([step / 2, step / 2]), step=step)
    walking.area = area
    return surface, walking


def run(bypass, ticks=900):
    surface, walking = scene(bypass)
    router = LifeRouter(walking, walking.free, landmarks=2)
    start, goal = np.array([4.0, 3.0]), np.array([36.0, 3.0])
    # Six people stand across the street at x 20-21.5: nobody can pass them.
    still = [(20.0, 2.45), (20.0, 3.0), (20.0, 3.55), (21.4, 2.45), (21.4, 3.0), (21.4, 3.55)]
    n = 1 + len(still)
    ext = np.stack([EXT] * n)
    crowd = Crowd(ext, np.array(["person"] * n), surface, walking)
    standoffs, reroutes = Standoffs(crowd), Reroutes(router)
    xy = np.array([start] + still, float)
    yaw = np.zeros(n)
    yaw[1:] = math.pi / 2
    speed = np.zeros(n)
    vmax = np.full(n, 1.3)
    paths = PathStore(n)
    rr = router.plan(start, goal)
    paths.set(0, np.vstack([start, rr["points"][1:]]))
    agendas = {0: Agenda()}
    stall = np.zeros(n, int)
    track = [xy[0].copy()]
    arrived = None
    for k in range(ticks):
        t = k * DT
        travel = [0] if arrived is None else []
        for i, pts in standoffs.update(t, travel, xy, stall, paths, yaw, agendas, DT).items():
            paths.set(i, pts)
        for i, pts in reroutes.update(
            t, travel, xy, speed, np.ones(n, bool), stall, paths, lambda i: goal, set(standoffs.yielding), agendas, DT
        ).items():
            paths.set(i, pts)
        des_yaw, des_speed, cap = yaw.copy(), np.zeros(n), np.full(n, np.inf)
        anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(n)
        if travel:
            led = np.array(travel, np.int64)
            h, want, c, an, tg, ln = paths.desire(led, xy, speed, vmax, yaw)
            if not np.isnan(h[0]):
                des_yaw[0] = h[0]
                des_speed[0] = want[0]
                anchor[0] = an[0]
                tangent[0] = tg[0]
                lane[0] = ln[0]
            cap[0] = c[0]
        new_xy, new_yaw, new_speed, held = crowd.step(
            xy, yaw, speed, np.ones(n, bool), np.ones(n, bool), des_yaw, des_speed, vmax, cap, anchor, tangent, lane
        )
        assert not held
        v = np.linalg.norm(new_xy - xy, axis=1) / DT
        c_, s_ = bodies.boxes(new_xy, new_yaw, v, ext, np.full(n, 1.7))
        assert collision_pairs(c_, new_yaw, s_) == []
        stall = np.where((np.arange(n) == 0) & (arrived is None) & (v < 0.2), stall + 1, 0)
        xy, yaw, speed = new_xy, new_yaw, new_speed
        track.append(xy[0].copy())
        if arrived is None and np.linalg.norm(xy[0] - goal) < 0.05 and speed[0] < 0.08:
            arrived = t
    np.testing.assert_allclose(xy[1:], still)
    return arrived, np.array(track), reroutes.stats, agendas[0].events


def test_walker_held_up_by_standing_people_takes_the_bypass():
    arrived, track, stats, events = run(True)
    assert arrived is not None
    assert stats["rerouted"] == 1 and events == ["rerouted"]
    assert track[:, 1].max() > 7.0  # it walked round by the parallel lane
    assert track[:, 0].max() < 36.1


def test_walker_without_another_way_keeps_its_route():
    arrived, track, stats, events = run(False, ticks=400)
    assert arrived is None and stats["rerouted"] == 0 and stats["kept"] >= 1 and events == []
    assert track[:, 0].max() < 20.0


def test_detour_round_standing_people_on_open_ground_survives_string_pulling():
    """On an open square the search's way round two people standing on the straight line is kept: a shortcut may
    not run near them (the avoided cells) for longer than the grid path it replaces, so string pulling does not
    pull the detour back through them, and the new way passes fewer standing bodies than the present one."""
    _, walking = scene(False, parts=[box(0.0, 0.0, 12.0, 8.0)], W=12.0, H=8.0)
    router = LifeRouter(walking, walking.free, landmarks=2)
    start, goal = np.array([2.0, 4.0]), np.array([10.0, 4.0])
    people = np.array([[6.0, 3.8], [6.2, 4.3]])
    straight = np.asarray(router.plan(start, goal)["points"])
    assert len(straight) == 2
    cells = router.avoid_cells(people, RR.JAM_RADIUS_M)
    detour = np.asarray(router.plan(start, goal, avoid=(cells, RR.JAM_FACTOR))["points"])
    along = np.vstack(
        [a + u * (b - a) for a, b in zip(detour[:-1], detour[1:], strict=False) for u in np.linspace(0.0, 1.0, 50)]
    )
    assert np.linalg.norm(along[:, None] - people[None], axis=2).min() > RR.JAM_RADIUS_M - 0.25
    jam = np.zeros(len(router.cost), bool)
    jam[cells] = True
    rr = Reroutes(router)
    gained = rr._exposure(RR._samples(Path(straight), 0.0, RR.AHEAD_M), jam) - rr._exposure(
        RR._samples(Path(detour), 0.0, RR.AHEAD_M), jam
    )
    assert gained > RR.GAIN_M


def test_congestion_uses_full_rotated_vehicle_footprint_not_a_small_centre_disk():
    # Port of codex 8742f47.
    from shapely.geometry import Point, Polygon

    network = WalkNetwork(np.ones((40, 40), bool), np.array([0.25, 0.25]), step=0.5)
    router = LifeRouter(network, network.free, landmarks=2)
    xy = np.array([[10.0, 10.0]])
    ext = np.array([[[-3.0, 3.0, -1.0, 1.0]] * 2])
    centre_cells = set(router.avoid_cells(xy, 0.8))
    for angle in (0.0, math.pi / 2, 0.41):
        yaw = np.array([angle])
        speed = np.array([0.0])
        actual = set(router.avoid_bodies(xy, yaw, speed, ext, 0.45))
        c, s = math.cos(angle), math.sin(angle)
        poly = Polygon(
            [(10.0 + a * 3 * c - b * s, 10.0 + a * 3 * s + b * c) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        )
        expected = {
            int(router.index[tuple(rc)])
            for rc, p in zip(network.nodes, network.xy, strict=False)
            if poly.distance(Point(p)) <= 0.45 + 1e-10
        }
        assert actual == expected
        assert len(actual - centre_cells) > 20
    horizontal = set(router.avoid_bodies(xy, np.array([0.0]), np.zeros(1), ext))
    vertical = set(router.avoid_bodies(xy, np.array([math.pi / 2]), np.zeros(1), ext))
    cell = router.index[tuple(network.cell([12.25, 10.25]))]
    assert cell in horizontal and cell not in vertical
    assert not len(router.avoid_bodies(np.empty((0, 2)), np.empty(0), np.empty(0), np.empty((0, 2, 4))))


def test_a_walker_beside_a_stopped_car_goes_round_the_car_body():
    # cruise5 r11f v3: person-041 stood 110 s beside vehicle-047, which had stopped for it; its way ran along the car's
    # side, through the body but off its centre. Replanned against a disc round the centre, a way still runs through
    # the car's ends (or the present way is kept); against the body's footprint the walker goes round the car.
    _, walking = scene(False, parts=[box(0.0, 0.0, 40.0, 8.0)], W=40.0, H=8.0)
    router = LifeRouter(walking, walking.free, landmarks=2)
    start, goal = np.array([14.0, 4.6]), np.array([36.0, 4.6])
    car_ext = np.array([[-2.1, 2.2, -1.05, 1.05]] * 2)
    ext = np.stack([EXT, car_ext])
    xy = np.array([start, [20.0, 4.0]])
    yaw = np.zeros(2)
    speed = np.zeros(2)

    def through_body(way):
        along = np.vstack(
            [a + u * (b - a) for a, b in zip(way[:-1], way[1:], strict=False) for u in np.linspace(0.0, 1.0, 40)]
        )
        return bool(((np.abs(along[:, 0] - 20.05) <= 2.15) & (np.abs(along[:, 1] - 4.0) <= 1.05)).any())

    ways = []
    for with_body in (False, True):
        paths = PathStore(2)
        paths.set(0, np.array([start, goal]))
        rr = Reroutes(router, ext if with_body else None)
        stall = np.array([round(RR.AFTER_S / DT), 0])
        out = rr.update(
            0.0, [0], xy, speed, np.ones(2, bool), stall, paths, lambda i: goal, set(), {0: Agenda()}, DT, yaw=yaw
        )
        ways.append(out.get(0, np.array([start, goal])))
    assert through_body(np.array([start, goal]))
    assert through_body(ways[0])
    assert not through_body(ways[1])
