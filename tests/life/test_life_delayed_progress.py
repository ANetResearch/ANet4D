"""A low-speed shuffle must not permanently suppress useful safe detours."""

import numpy as np
import pytest
from test_life_crowd import EXT, corridor

from awr.life.civil import body
from awr.life.civil.collision import collision_pairs
from awr.life.motion.crowd import DT, Crowd, Path


@pytest.mark.xfail(
    strict=True,
    reason="known gap of the selected crowd: a 0.12 m/s shuffle beside the stopped car "
    "ends the wait, so the detour cost never builds up; the walker arrives after 16.9 s, not within "
    "12 s (docs/engine/merge-2026-10-25.zh-CN.md)",
)
@pytest.mark.parametrize("reverse", [False, True])
def test_delayed_walker_clears_stationary_car_front_without_perpetual_creeping(reverse):
    # Reduced from a mixed-traffic encounter: a stopped car is yielding while a
    # pedestrian's diagonal route passes its front corner. There is ample legal
    # space for a faster detour. The old selector needed 16.7 s to arrive because
    # every 0.12 m/s shuffle erased the delay; the same geometry is rotated too.
    surface, walking = corridor(width=11.0, length=30.0)
    ext = np.stack([EXT, np.array([[-2.13, 2.13, -1.06, 1.06]] * 2)])
    crowd = Crowd(ext, np.array(["person", "vehicle"]), surface, walking)
    xy = np.array([[10.0, 0.0], [12.785109, -0.648556]])
    yaw = np.array([0.705732, 2.918504])
    speed = np.array([0.12, 0.0])
    points = np.array([[8.0, -1.0239], [18.0, 4.5138]])
    if reverse:
        xy = [30.0, 0.0] - xy
        points = [30.0, 0.0] - points
        yaw += np.pi
    crowd.aim = yaw.copy()
    path = Path(points)
    stationary_car = xy[1].copy()
    previous_velocity = speed[:, None] * np.stack([np.cos(yaw), np.sin(yaw)], 1)
    arrived = False
    for tick in range(120):
        h, want, cap = path.desire(xy[0], speed[0], 1.25, yaw[0])
        dy, ds, gc = yaw.copy(), np.zeros(2), np.full(2, np.inf)
        anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(2)
        if h is not None:
            dy[0], ds[0], gc[0] = h, want, cap
            anchor[0], tangent[0], lane[0] = path.anchor, path.tangent, path.lane
        nx, ny, ns, held = crowd.step(
            xy,
            yaw,
            speed,
            np.ones(2, bool),
            np.array([True, False]),
            dy,
            ds,
            np.array([1.25, 0.0]),
            gc,
            anchor,
            tangent,
            lane,
        )
        assert not held
        np.testing.assert_array_equal(nx[1], stationary_car)
        cc, ss = body.boxes(nx, ny, ns, ext, np.full(2, 1.7))
        assert not collision_pairs(cc, ny, ss)
        assert crowd._supported(0, xy, yaw, nx, ny, np.maximum(speed, ns) + 0.01)
        velocity = (nx - xy) / DT
        acceleration = (
            np.linalg.norm(np.round(velocity * 100) * 0.01 - np.round(previous_velocity * 100) * 0.01, axis=1) / DT
        )
        assert acceleration.max() <= 8.0
        assert np.abs((ny - yaw + np.pi) % (2 * np.pi) - np.pi).max() <= 3.0 * DT + 1e-8
        xy, yaw, speed, previous_velocity = nx, ny, ns, velocity
        if np.linalg.norm(xy[0] - path.p[-1]) < 0.05:
            arrived = True
            break
    assert arrived  # bounded travel progress, not merely collision-free shuffling


def test_requested_slow_approach_does_not_accrue_congestion_delay():
    surface, walking = corridor()
    crowd = Crowd(np.stack([EXT]), np.array(["person"]), surface, walking)
    xy, yaw, speed = np.array([[5.0, 0.0]]), np.zeros(1), np.array([0.05])
    for _ in range(30):
        xy, yaw, speed, held = crowd.step(
            xy,
            yaw,
            speed,
            np.ones(1, bool),
            np.ones(1, bool),
            np.zeros(1),
            np.array([0.05]),
            np.array([1.25]),
            np.full(1, np.inf),
        )
        assert not held and crowd.waited[0] == 0.0
    assert xy[0, 0] > 5.1
