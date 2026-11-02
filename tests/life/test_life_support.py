"""Support of the pose box in the crowd kernel (awr.life.motion.kernels.body_supported): the tape QA's corner test plus
no overlapped support cell without walkable ground or height, which makes the predicate monotone in the box."""

import math
from types import SimpleNamespace

import numpy as np
from shapely.geometry import box

from awr.life.civil import body as bodies
from awr.life.motion import kernels as K
from awr.life.motion.crowd import DT, Crowd, Path

EXT = np.array(
    [[-0.135, 0.231, -0.307, 0.304], [-0.391, 0.532, -0.339, 0.336]]
)  # a town resident's rest and walking box


def corners_supported(x, y, yaw, speed, z_ok, walk, x0, y0, cell):
    """The tape QA's predicate (awr.life.civil.validate_tape): the four corners of the box at the recorded speed."""
    cx, cy, hl, hw = K.box(x, y, yaw, speed, EXT)
    c, s = math.cos(yaw), math.sin(yaw)
    return all(
        K.supported(cx + sx * hl * c - sy * hw * s, cy + sx * hl * s + sy * hw * c, z_ok, walk, x0, y0, cell)
        for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    )


def test_supported_walking_box_implies_supported_rest_box():
    rng = np.random.default_rng(11)
    cell, checked = 0.5, 0
    for _ in range(300):
        walk = rng.random((12, 12)) > 0.08
        z_ok = np.ones_like(walk)
        for _ in range(40):
            x, y, yaw = rng.uniform(1.5, 4.5), rng.uniform(1.5, 4.5), rng.uniform(-math.pi, math.pi)
            if K.body_supported(x, y, yaw, 1.2, EXT, z_ok, walk, 0.0, 0.0, cell):
                checked += 1
                assert K.body_supported(x, y, yaw, 0.0, EXT, z_ok, walk, 0.0, 0.0, cell)
                assert K.body_supported(x, y, yaw, 0.4, EXT, z_ok, walk, 0.0, 0.0, cell)
                assert corners_supported(x, y, yaw, 0.0, z_ok, walk, 0.0, 0.0, cell)
                assert corners_supported(x, y, yaw, 1.2, z_ok, walk, 0.0, 0.0, cell)
    assert checked > 1000


def test_box_straddling_a_hole_is_not_supported():
    walk = np.ones((10, 10), bool)
    walk[5, 5] = False  # the cell [2.5, 3] x [2.5, 3]
    z_ok = np.ones_like(walk)
    # heading +y: the rest box spans x 2.45..3.06 and y 2.62..2.98 around (2.75, 2.75); its corners lie outside the
    # hole (x), the box covers it
    assert corners_supported(2.75, 2.66, math.pi / 2, 1.2, z_ok, walk, 0.0, 0.0, 0.5)
    assert not K.body_supported(2.75, 2.66, math.pi / 2, 1.2, EXT, z_ok, walk, 0.0, 0.0, 0.5)
    assert K.body_supported(2.75, 4.2, math.pi / 2, 1.2, EXT, z_ok, walk, 0.0, 0.0, 0.5)


def test_walker_walks_round_a_missing_cell_on_its_line():
    from shapely.geometry import Polygon

    cell = 0.5
    walk = np.ones((24, 60), bool)
    walk[11, 30] = False  # the cell [15, 15.5] x [-0.5, 0] is missing
    z = np.zeros(walk.shape)
    surface = SimpleNamespace(
        x0=0.0,
        y0=-6.0,
        cell_m=cell,
        nx=60,
        ny=24,
        width_m=30.0,
        height_m=12.0,
        z=z,
        walk={"person": walk, "vehicle": np.zeros_like(walk)},
    )
    walking = SimpleNamespace(free=None, origin=None, step=0.25, area=box(0.0, -3.0, 30.0, 3.0))
    crowd = Crowd(np.stack([EXT]), np.array(["person"]), surface, walking)
    hole = box(15.0, -0.5, 15.5, 0.0)
    # the path runs over the middle of the missing cell: every corner of the body stays outside it, the body does not
    xy, yaw, speed = np.array([[5.0, -0.25]]), np.zeros(1), np.zeros(1)
    path = Path(np.array([[5.0, -0.25], [25.0, -0.25]]))
    z_ok = np.isfinite(z)
    for _ in range(400):
        h, want, cap = path.desire(xy[0], speed[0], 1.3, yaw[0])
        des_yaw, des_speed = np.array([yaw[0] if h is None else h]), np.array([0.0 if h is None else want])
        new_xy, new_yaw, new_speed, held = crowd.step(
            xy,
            yaw,
            speed,
            np.ones(1, bool),
            np.ones(1, bool),
            des_yaw,
            des_speed,
            np.full(1, 1.3),
            np.array([cap]),
            np.array([path.anchor]),
            np.array([path.tangent]),
            np.zeros(1),
        )
        assert not held
        v = float(np.linalg.norm(new_xy[0] - xy[0]) / DT)
        xy, yaw, speed = new_xy, new_yaw, new_speed
        assert corners_supported(xy[0, 0], xy[0, 1], yaw[0], v, z_ok, walk, 0.0, -6.0, cell)  # the recorded pose (QA)
        for sp in (v, 0.0):  # the body as recorded and where it stops: never over the missing cell
            c, size = bodies.boxes(xy, yaw, np.array([sp]), np.stack([EXT]), np.full(1, 1.7))
            u, w = np.array([math.cos(yaw[0]), math.sin(yaw[0])]), np.array([-math.sin(yaw[0]), math.cos(yaw[0])])
            body = Polygon(
                [
                    c[0] + a * u * size[0, 0] / 2 + b * w * size[0, 1] / 2
                    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                ]
            )
            assert body.intersection(hole).area < 1e-9
    assert np.linalg.norm(xy[0] - (25.0, -0.25)) < 0.1


def test_yielding_walkers_stay_supported_on_a_narrow_bent_pavement():
    """Walkers that give way to each other (a shorter step, a cancelled turn, the last-resort brake) keep their pose
    box on the support: every recorded pose passes the kernel's support test at the recorded speed (the tape QA's
    corner test included)."""
    cell = 0.25
    nx, ny = 120, 120
    walk = np.zeros((ny, nx), bool)
    walk[8:15, 8:100] = True  # an eastward pavement 1.75 m wide (y 2..3.75)
    walk[8:100, 92:99] = True  # turning north (x 23..24.75)
    z = np.where(walk, 0.0, np.nan)
    surface = SimpleNamespace(
        x0=0.0,
        y0=0.0,
        cell_m=cell,
        nx=nx,
        ny=ny,
        width_m=nx * cell,
        height_m=ny * cell,
        z=z,
        walk={"person": walk, "vehicle": np.zeros_like(walk)},
    )
    from shapely.geometry import Polygon

    area = Polygon([(2.0, 2.0), (24.75, 2.0), (24.75, 25.0), (23.0, 25.0), (23.0, 3.75), (2.0, 3.75)])
    walking = SimpleNamespace(free=None, origin=None, step=0.25, area=area)
    line = np.array([[3.0, 2.9], [23.9, 2.9], [23.9, 24.0]])
    rng = np.random.default_rng(4)
    n = 8
    crowd = Crowd(np.stack([EXT] * n), np.array(["person"] * n), surface, walking)
    paths, xy, yaw = [], [], []
    for k in range(n):
        forward = k % 2 == 0
        pts = line if forward else line[::-1]
        start = 2.0 + 2.5 * (k // 2)
        seg = np.diff(pts, axis=0)
        L = np.linalg.norm(seg, axis=1)
        s = start
        m = 0
        while s > L[m]:
            s -= L[m]
            m += 1
        xy.append(pts[m] + seg[m] / L[m] * s)
        yaw.append(math.atan2(seg[m, 1], seg[m, 0]))
        paths.append(Path(np.vstack([xy[-1], pts[m + 1 :]])))
    xy, yaw, speed = np.array(xy), np.array(yaw), np.zeros(n)
    z_ok = np.isfinite(z)
    vmax = rng.uniform(1.1, 1.4, n)
    for _ in range(500):
        des_yaw, des_speed, cap, anchor, tangent = (np.zeros(n) for _ in range(5))
        anchor = np.zeros((n, 2))
        for k, p in enumerate(paths):
            h, want, c = p.desire(xy[k], speed[k], vmax[k], yaw[k])
            des_yaw[k], des_speed[k], cap[k] = (yaw[k], 0.0, 0.0) if h is None else (h, want, c)
            anchor[k], tangent[k] = p.anchor, p.tangent
        new_xy, new_yaw, new_speed, held = crowd.step(
            xy,
            yaw,
            speed,
            np.ones(n, bool),
            np.ones(n, bool),
            des_yaw,
            des_speed,
            vmax,
            cap,
            anchor,
            tangent,
            np.zeros(n),
        )
        v = np.linalg.norm(new_xy - xy, axis=1) / DT
        xy, yaw, speed = new_xy, new_yaw, new_speed
        for k in range(n):
            assert K.body_supported(xy[k, 0], xy[k, 1], yaw[k], v[k], EXT, z_ok, walk, 0.0, 0.0, cell), (k, xy[k])
    s = crowd.stats
    assert s["shortened"] + s["turn_cancelled"] + s["last_resort"] > 0, s


def test_a_stopping_walker_is_supported_with_the_box_it_is_recorded_with():
    """The corner test is not monotone in the box: beside a hole the rest box of a walker that stops can fail where its
    walking box passes. The step is judged with the box of its recorded speed as well (an L-shaped pavement corner)."""
    from shapely.geometry import Polygon

    cell = 0.25
    walk = np.zeros((100, 60), bool)
    walk[0:7, 0:40] = True
    walk[0:100, 33:40] = True
    z = np.where(walk, 0.0, np.nan)
    surface = SimpleNamespace(
        x0=0.0,
        y0=0.0,
        cell_m=cell,
        nx=60,
        ny=100,
        width_m=15.0,
        height_m=25.0,
        z=z,
        walk={"person": walk, "vehicle": np.zeros_like(walk)},
    )
    walking = SimpleNamespace(
        free=None,
        origin=None,
        step=0.25,
        area=Polygon([(0, 0), (10, 0), (10, 25), (8.25, 25), (8.25, 1.75), (0, 1.75)]),
    )
    z_ok = np.isfinite(z)
    crowd = Crowd(np.stack([EXT, EXT]), np.array(["person"] * 2), surface, walking)
    xy = np.array([[9.289, 0.626], [8.659, 1.845]])
    yaw = np.array([-0.038, 1.367])
    speed = np.array([1.197, 0.347])
    new_xy, new_yaw, new_speed, held = crowd.step(
        xy,
        yaw,
        speed,
        np.ones(2, bool),
        np.ones(2, bool),
        np.array([1.571, -2.583]),
        np.array([1.2, 0.018]),
        np.full(2, 1.4),
        np.full(2, np.inf),
    )
    v = np.hypot(np.round((new_xy - xy)[:, 0] / DT * 100), np.round((new_xy - xy)[:, 1] / DT * 100)) * 0.01
    assert not held
    for k in range(2):
        assert K.body_supported(new_xy[k, 0], new_xy[k, 1], new_yaw[k], v[k], EXT, z_ok, walk, 0.0, 0.0, cell), k
