"""Courtesy (awr.life.motion.courtesy): a person at a stay steps aside for a walker held up by it, and returns."""

import math
from types import SimpleNamespace

import numpy as np
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

from awr.life.civil import body as bodies
from awr.life.civil.collision import collision_pairs
from awr.life.motion.courtesy import Courtesy
from awr.life.motion.crowd import DT, Crowd, PathStore
from awr.life.motion.standoff import Standoffs

EXT = np.array([[-0.12, 0.24, -0.29, 0.29], [-0.42, 0.59, -0.31, 0.31]])


class Agenda:
    def __init__(self):
        self.events = []
        self.paused_at = None

    def blocked(self, t, reason):
        self.events.append(reason)

    def pause(self, t, reason):
        if self.paused_at is not None:
            return False
        self.paused_at = t
        self.events.append("stepped_aside")
        return True

    def resume(self, t):
        self.paused_at = None
        self.events.append("returned_to_place")


def scene():
    """A 1.6 m passage from x=0 to 30 (y -0.8..0.8) with a 2 m alcove on its north side at x 13-17."""
    area = unary_union([box(0.0, -0.8, 30.0, 0.8), box(13.0, 0.7, 17.0, 2.8)])
    cell, W, H, y0 = 0.25, 30.0, 8.0, -4.0
    gx, gy = np.meshgrid((np.arange(int(W / cell)) + 0.5) * cell, (np.arange(int(H / cell)) + 0.5) * cell + y0)
    walk = shapely.contains_xy(area, gx, gy)
    surface = SimpleNamespace(
        x0=0.0,
        y0=y0,
        cell_m=cell,
        nx=walk.shape[1],
        ny=walk.shape[0],
        width_m=W,
        height_m=H,
        z=np.zeros(walk.shape),
        walk={"person": walk, "vehicle": np.zeros_like(walk)},
    )
    step = 0.5
    fx, fy = np.meshgrid((np.arange(int(W / step)) + 0.5) * step, (np.arange(int(H / step)) + 0.5) * step + y0)
    walking = SimpleNamespace(
        free=shapely.contains_xy(area.buffer(-0.3), fx, fy),
        origin=np.array([step / 2, y0 + step / 2]),
        step=step,
        area=area,
    )
    return surface, walking


def test_person_at_a_stay_steps_aside_and_returns():
    surface, walking = scene()
    n = 2
    ext = np.stack([EXT] * n)
    crowd = Crowd(ext, np.array(["person"] * n), surface, walking)
    standoffs, courtesy = Standoffs(crowd), Courtesy(crowd)
    start, goal, home = np.array([3.0, 0.0]), np.array([27.0, 0.0]), np.array([15.0, 0.0])
    xy = np.array([start, home])
    yaw = np.array([0.0, math.pi / 2])
    speed = np.zeros(n)
    vmax = np.full(n, 1.3)
    paths = PathStore(n)
    paths.set(0, np.array([start, goal]))
    agendas = {0: Agenda(), 1: Agenda()}
    stall = np.zeros(n, int)
    arrived = None
    aside_at = None
    for k in range(1200):
        t = k * DT
        travel = [0] if arrived is None else []
        standing = [1] if agendas[1].paused_at is None else []
        for i, pts in standoffs.update(t, travel, xy, stall, paths, yaw, agendas, DT).items():
            paths.set(i, pts)
        for i, pts in courtesy.update(
            t, travel, standing, xy, speed, np.ones(n, bool), stall, paths, agendas, DT
        ).items():
            paths.set(i, pts)
        for i in courtesy.aside:
            if i not in courtesy.walking:
                paths.pop(i)
        if 1 not in courtesy.aside:
            paths.pop(1)
        walkers = travel + courtesy.walking
        des_yaw, des_speed, cap = yaw.copy(), np.zeros(n), np.full(n, np.inf)
        anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(n)
        if walkers:
            led = np.array(walkers, np.int64)
            h, want, c, an, tg, ln = paths.desire(led, xy, speed, vmax, yaw)
            for i in walkers:
                if not np.isnan(h[i]):
                    des_yaw[i] = h[i]
                    des_speed[i] = want[i]
                    anchor[i] = an[i]
                    tangent[i] = tg[i]
                    lane[i] = ln[i]
                cap[i] = c[i]
        new_xy, new_yaw, new_speed, held = crowd.step(
            xy, yaw, speed, np.ones(n, bool), np.ones(n, bool), des_yaw, des_speed, vmax, cap, anchor, tangent, lane
        )
        assert not held
        v = np.linalg.norm(new_xy - xy, axis=1) / DT
        c_, s_ = bodies.boxes(new_xy, new_yaw, v, ext, np.full(n, 1.7))
        assert collision_pairs(c_, new_yaw, s_) == []
        moving = np.zeros(n, bool)
        moving[walkers] = True
        stall = np.where(moving & (v < 0.2), stall + 1, 0)
        xy, yaw, speed = new_xy, new_yaw, new_speed
        if aside_at is None and 1 in courtesy.aside:
            aside_at = t
        if arrived is None and np.linalg.norm(xy[0] - goal) < 0.05 and speed[0] < 0.08:
            arrived = t
    assert arrived is not None and aside_at is not None
    assert agendas[1].events == ["stepped_aside", "returned_to_place"] and agendas[1].paused_at is None
    assert np.linalg.norm(xy[1] - home) < 0.05 and courtesy.stats["stepped_aside"] == 1


def test_finished_path_outside_the_arrival_radius_is_planned_anew():
    # full300 v7, person-161: back from stepping aside, 1 cm short of the spot it had reached its stay from, 4.5 cm
    # from the stay itself: 5.45 cm from the target, outside the agendas' 5 cm, with the path finished.
    from awr.life.motion.engine import ARRIVAL_M, LifeEngine

    eng = LifeEngine.__new__(LifeEngine)
    eng.paths = PathStore(1)
    eng.paths.set(0, np.array([[0.0, 0.0], [1.0, 0.0]]))
    eng.paths.s[0] = 0.99
    eng.agendas = {0: SimpleNamespace(target_xy=(1.02, 0.045))}
    xy = np.array([[0.99, 0.0]])
    assert math.dist(xy[0], (1.02, 0.045)) > ARRIVAL_M and eng.ended_short(0, xy)
    eng.agendas[0].target_xy = (1.0, 0.03)  # within the arrival radius: the stay begins, nothing is planned
    assert not eng.ended_short(0, xy)
    eng.agendas[0].target_xy = (1.02, 0.045)
    eng.paths.s[0] = 0.5  # the path is not finished
    assert not eng.ended_short(0, np.array([[0.5, 0.0]]))


def test_a_waiting_car_asks_people_on_its_route_to_step_aside():
    """A car on a trip that has stood for ASK_AFTER_S asks the people within reach of its route ahead (a walker on its
    way and a person at a stay) to step aside to refuges clear of that route; they return once it has passed."""
    from awr.life.motion.courtesy import ASK_AFTER_S, CLEAR_GAP_M

    area = box(0.0, -3.0, 40.0, 3.0)  # a shared street 6 m wide
    cell, W, H, y0 = 0.25, 40.0, 8.0, -4.0
    gx, gy = np.meshgrid((np.arange(int(W / cell)) + 0.5) * cell, (np.arange(int(H / cell)) + 0.5) * cell + y0)
    walk = shapely.contains_xy(area, gx, gy)
    surface = SimpleNamespace(
        x0=0.0,
        y0=y0,
        cell_m=cell,
        nx=walk.shape[1],
        ny=walk.shape[0],
        width_m=W,
        height_m=H,
        z=np.zeros(walk.shape),
        walk={"person": walk, "vehicle": walk},
    )
    step = 0.5
    fx, fy = np.meshgrid((np.arange(int(W / step)) + 0.5) * step, (np.arange(int(H / step)) + 0.5) * step + y0)
    walking = SimpleNamespace(
        free=shapely.contains_xy(area.buffer(-0.3), fx, fy),
        origin=np.array([step / 2, y0 + step / 2]),
        step=step,
        area=area,
    )
    car_ext = np.array([[-2.1, 2.1, -1.05, 1.05]] * 2)
    ext = np.stack([car_ext, EXT, EXT])
    crowd = Crowd(ext, np.array(["vehicle", "person", "person"]), surface, walking)
    courtesy = Courtesy(crowd)
    xy = np.array([[6.0, 0.0], [10.5, 0.3], [13.0, -0.4]])
    speed = np.zeros(3)
    active = np.ones(3, bool)
    route = np.column_stack([np.arange(6.0, 40.0, 0.25), np.zeros(len(np.arange(6.0, 40.0, 0.25)))])
    agendas = {1: Agenda(), 2: Agenda()}
    paths = PathStore(3)

    def cars(waited, at=6.0):
        ahead = route[(route[:, 0] >= at) & (route[:, 0] <= at + 2.1 + 6.0)]
        longer = route[(route[:, 0] >= at) & (route[:, 0] <= at + 2.1 + 10.0)]
        return {0: dict(ahead=ahead, longer=longer, waited=waited)}

    need = round(ASK_AFTER_S / DT)
    assert (
        courtesy.update(0.0, [1], [2], xy, speed, active, np.zeros(3, int), paths, agendas, DT, cars=cars(need - 1))
        == {}
    )  # not yet waited long enough
    out = courtesy.update(0.1, [1], [2], xy, speed, active, np.zeros(3, int), paths, agendas, DT, cars=cars(need))
    assert set(out) == {1, 2} and courtesy.stats["car_asked"] == 2
    for j in (1, 2):
        spot = out[j][-1]
        assert abs(spot[1]) >= 1.05 + 0.31 + CLEAR_GAP_M - 1e-9 and agendas[j].events[-1] == "stepped_aside"
        xy[j] = spot  # walked there
    assert courtesy.update(0.2, [], [], xy, speed, active, np.zeros(3, int), paths, agendas, DT, cars=cars(need)) == {}
    assert all(courtesy.aside[j]["stage"] == "wait" for j in (1, 2))
    xy[0] = (22.0, 0.0)  # the car has passed both spots
    out = courtesy.update(5.0, [], [], xy, speed, active, np.zeros(3, int), paths, agendas, DT, cars=cars(0, at=22.0))
    # the person at a stay walks back to its spot; the walker goes on from its refuge (cq3p v6: walking back put it in
    # front of the car again)
    assert set(out) == {2} and courtesy.resumed == [1] and agendas[1].events[-1] == "returned_to_place"
    xy[2] = out[2][-1]
    courtesy.update(6.0, [], [], xy, speed, active, np.zeros(3, int), paths, agendas, DT, cars=cars(0, at=22.0))
    assert not courtesy.aside and all(agendas[j].events[-1] == "returned_to_place" for j in (1, 2))
    # A refuge not reached in OUT_TIMEOUT_S: the walker goes on from where it stands, the person at a stay walks back.
    from awr.life.motion.courtesy import OUT_TIMEOUT_S

    courtesy = Courtesy(crowd)
    xy = np.array([[6.0, 0.0], [10.5, 0.3], [13.0, -0.4]])
    agendas = {1: Agenda(), 2: Agenda()}
    out = courtesy.update(0.0, [1], [2], xy, speed, active, np.zeros(3, int), paths, agendas, DT, cars=cars(need))
    assert set(out) == {1, 2}
    out = courtesy.update(
        OUT_TIMEOUT_S + 0.1, [], [], xy, speed, active, np.zeros(3, int), paths, agendas, DT, cars=cars(need)
    )
    assert set(out) == {2} and courtesy.resumed == [1] and 1 not in courtesy.aside


def test_a_person_beside_a_waiting_car_finds_a_refuge_next_to_it():
    """cruise3-r11f v1 and cq-cruise5 v3: people standing beside a waiting car found no refuge (no_refuge) because
    every spot near the car lay within its 2.35 m bounding circle. Clearance is now taken from the car's pose box. The
    body that sets the car's stop is asked even when it is beyond the reach of the route's centre line."""
    from awr.life.motion.courtesy import ASK_AFTER_S, CLEAR_GAP_M

    area = box(0.0, -3.0, 40.0, 3.0)
    cell, W, H, y0 = 0.25, 40.0, 8.0, -4.0
    gx, gy = np.meshgrid((np.arange(int(W / cell)) + 0.5) * cell, (np.arange(int(H / cell)) + 0.5) * cell + y0)
    walk = shapely.contains_xy(area, gx, gy)
    surface = SimpleNamespace(
        x0=0.0,
        y0=y0,
        cell_m=cell,
        nx=walk.shape[1],
        ny=walk.shape[0],
        width_m=W,
        height_m=H,
        z=np.zeros(walk.shape),
        walk={"person": walk, "vehicle": walk},
    )
    step = 0.5
    fx, fy = np.meshgrid((np.arange(int(W / step)) + 0.5) * step, (np.arange(int(H / step)) + 0.5) * step + y0)
    walking = SimpleNamespace(
        free=shapely.contains_xy(area.buffer(-0.3), fx, fy),
        origin=np.array([step / 2, y0 + step / 2]),
        step=step,
        area=area,
    )
    car_ext = np.array([[-2.1, 2.1, -1.05, 1.05]] * 2)
    ext = np.stack([car_ext, EXT, EXT])
    crowd = Crowd(ext, np.array(["vehicle", "person", "person"]), surface, walking)
    xy = np.array([[6.0, 0.0], [6.5, -1.6], [13.0, 2.3]])
    speed = np.zeros(3)
    yaw = np.zeros(3)
    active = np.ones(3, bool)
    route = np.column_stack([np.arange(6.0, 40.0, 0.25), np.zeros(len(np.arange(6.0, 40.0, 0.25)))])
    ahead, longer = route[route[:, 0] <= 6.0 + 2.1 + 6.0], route[route[:, 0] <= 6.0 + 2.1 + 10.0]
    need = round(ASK_AFTER_S / DT)
    cars = {0: dict(ahead=ahead, longer=longer, waited=need, blocker=-1)}
    assert np.linalg.norm(xy[1] - xy[0]) < crowd.radius[0]  # inside the car's bounding circle
    circles = Courtesy(crowd)
    assert (
        circles.update(0.0, [1], [], xy, speed, active, np.zeros(3, int), PathStore(3), {1: Agenda()}, DT, cars=cars)
        == {}
        and circles.stats["no_refuge"] == 1
    )
    courtesy = Courtesy(crowd)
    agendas = {1: Agenda(), 2: Agenda()}
    out = courtesy.update(
        0.0, [1], [2], xy, speed, active, np.zeros(3, int), PathStore(3), agendas, DT, cars=cars, yaw=yaw
    )
    assert set(out) == {1} and abs(out[1][-1][1]) >= 1.05 + 0.31 + CLEAR_GAP_M - 1e-9
    assert 2 not in out  # beyond the reach of the route's centre line
    courtesy = Courtesy(crowd)
    agendas = {1: Agenda(), 2: Agenda()}
    cars[0]["blocker"] = 2
    out = courtesy.update(
        0.0, [1], [2], xy, speed, active, np.zeros(3, int), PathStore(3), agendas, DT, cars=cars, yaw=yaw
    )
    assert set(out) == {1, 2} and agendas[2].events[-1] == "stepped_aside"
