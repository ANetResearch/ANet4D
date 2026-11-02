"""A synthetic town for end-to-end tests of the life engine (not a release environment).

A one-way loop road (counter-clockwise, rounded corners) around a building block; kerbside berths outside the
loop with S-curve entry and exit edges; car-side access points beside each berth; homes along the south side and
facilities along the north side, each with its own berth. Flat support everywhere except the building block.
Everything an environment package provides to the engine is built here from geometry: surface and support,
walking network and router, Director places, activities and actors, driving graph, DrivingRoutes and
DriverExecution.
"""

import math
from types import SimpleNamespace

import numpy as np
from shapely.geometry import box as rect

from awr.life.civil.activity.director import Activity, Actor, Director, Place, Route, Vehicle
from awr.life.civil.activity.driver_execution import DriverExecution
from awr.life.civil.activity.driving import Body, DrivingRoutes
from awr.life.civil.agenda import Occupancy
from awr.life.civil.driving_graph import DrivingGraph, route_arrays
from awr.life.civil.walking import WalkNetwork
from awr.life.motion.agenda import LifeAgenda
from awr.life.motion.engine import LifeEngine
from awr.life.motion.route import LifeRouter

CAR = (4.26, 2.12, 1.58)
CAR_EXT = np.array([[-CAR[0] / 2, CAR[0] / 2, -CAR[1] / 2, CAR[1] / 2]] * 2)
PERSON_EXT = np.array([[-0.12, 0.24, -0.29, 0.29], [-0.42, 0.59, -0.31, 0.31]])
W, H = 150.0, 100.0
LOOP = (25.0, 25.0, 125.0, 75.0, 10.0)  # x0, y0, x1, y1, corner radius (lane centre line)
BLOCK = (40.0, 38.0, 110.0, 62.0)
BAY_M = 3.3  # berth centre to lane centre
SPACING = 0.3


class Surface(SimpleNamespace):
    def footprint_supported(self, footprint, profile):
        return rect(0.0, 0.0, W, H).difference(rect(*BLOCK)).covers(footprint)

    def vehicle_route_supported(self, points, yaw, size):
        """Candidate support backend for route admission: the body corners and centre on supported ground."""
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]]) * np.array(size[:2]) / 2
        q = points[:, None, :] + np.stack(
            [corners[:, 0] * c - corners[:, 1] * s, corners[:, 0] * s + corners[:, 1] * c], -1
        )
        return bool(self.supported(q.reshape(-1, 2), "vehicle").all())

    def supported(self, xy, profile):
        xy = np.atleast_2d(xy)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] <= W) & (xy[:, 1] >= 0) & (xy[:, 1] <= H)
        block = (xy[:, 0] > BLOCK[0]) & (xy[:, 0] < BLOCK[2]) & (xy[:, 1] > BLOCK[1]) & (xy[:, 1] < BLOCK[3])
        return inside & ~block


def surface_grid(cell=0.25):
    nx, ny = int(W / cell), int(H / cell)
    xs, ys = (np.arange(nx) + 0.5) * cell, (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    walk = ~((gx > BLOCK[0] - 0.1) & (gx < BLOCK[2] + 0.1) & (gy > BLOCK[1] - 0.1) & (gy < BLOCK[3] + 0.1))
    return Surface(
        x0=0.0,
        y0=0.0,
        cell_m=cell,
        nx=nx,
        ny=ny,
        width_m=W,
        height_m=H,
        z=np.zeros((ny, nx)),
        walk={"person": walk, "vehicle": walk},
    )


def loop_line():
    """Points (spacing SPACING) and headings of the counter-clockwise lane centre line, starting mid south side."""
    x0, y0, x1, y1, r = LOOP
    pieces = []

    def straight(a, b):
        n = max(int(math.ceil(math.dist(a, b) / SPACING)), 1)
        u = np.linspace(0, 1, n, endpoint=False)
        p = np.array(a) + u[:, None] * (np.array(b) - np.array(a))
        pieces.append((p, np.full(n, math.atan2(b[1] - a[1], b[0] - a[0]))))

    def arc(cx, cy, a0):
        n = max(int(math.ceil(r * math.pi / 2 / SPACING)), 1)
        th = a0 + np.linspace(0, math.pi / 2, n, endpoint=False)
        pieces.append((np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1), th + math.pi / 2))

    mid = (x0 + x1) / 2
    straight((mid, y0), (x1 - r, y0))
    arc(x1 - r, y0 + r, -math.pi / 2)
    straight((x1, y0 + r), (x1, y1 - r))
    arc(x1 - r, y1 - r, 0.0)
    straight((x1 - r, y1), (x0 + r, y1))
    arc(x0 + r, y1 - r, math.pi / 2)
    straight((x0, y1 - r), (x0, y0 + r))
    arc(x0 + r, y0 + r, math.pi)
    straight((x0 + r, y0), (mid, y0))
    pts = np.vstack([p for p, _ in pieces])
    yaw = np.unwrap(np.concatenate([h for _, h in pieces]))
    return pts, yaw


def s_curve(a, ya, b, yb, heading, n_per_m=1 / SPACING):
    """Lateral S-curve along `heading` from lateral offset ya at station a to yb at station b (smoothstep)."""
    L = b - a
    n = max(int(math.ceil(L * n_per_m)), 2)
    u = np.linspace(0.0, 1.0, n + 1)
    s = a + u * L
    lat = ya + (yb - ya) * (3 * u**2 - 2 * u**3)
    dlat = (yb - ya) * (6 * u - 6 * u**2) / L
    c, sn = math.cos(heading), math.sin(heading)
    # station along heading, lateral to the left of heading
    x = s * c - lat * sn
    y = s * sn + lat * c
    yaw = heading + np.arctan(dlat)
    return np.stack([x, y], 1), yaw


def build(
    walkers=0,
    drivers=2,
    seed=0,
    dwell=(40.0, 60.0),
    recurrence=240.0,
    reverse_exit_m=0.0,
    parking_legality=None,
    parked_walking=False,
    selection=None,
):
    surface = surface_grid()
    graph = DrivingGraph()
    pts, yaw = loop_line()
    n = len(pts)
    step_nodes = 13  # a lane node every 3.9 m
    lane_idx = list(range(0, n, step_nodes))
    for k in lane_idx:
        graph.add_node(f"lane:{k}", (pts[k, 0], pts[k, 1], yaw[k]), kind="lane")
    for a, b in zip(lane_idx, lane_idx[1:] + [lane_idx[0]], strict=False):
        idx = list(range(a, b + 1)) if b > a else list(range(a, n)) + [0]
        p = pts[idx]
        h = yaw[idx].copy()
        if b <= a:
            h[-1] = yaw[0] + 2 * math.pi * round((h[-2] - yaw[0]) / (2 * math.pi))
        r = route_arrays(p, h)
        graph.add_edge(f"travel:{a}", f"lane:{a}", f"lane:{b}", r, kind="travel")
    # Berths outside the loop on the straight south (heading +x) and north (heading -x) sides.
    berths = []
    x0, y0, x1, y1, _ = LOOP

    def bay(name, x_mid, south):
        wy = y0 if south else y1
        sign = 1.0 if south else -1.0
        heading = 0.0 if south else math.pi
        cand = [k for k in lane_idx if abs(pts[k, 1] - wy) < 1e-6]
        ka = min(cand, key=lambda k: abs(pts[k, 0] - (x_mid - sign * 14.0)))
        kb = min(cand, key=lambda k: abs(pts[k, 0] - (x_mid + sign * 14.0)))
        xa, xb = pts[ka, 0], pts[kb, 0]
        xm = (xa + xb) / 2
        # stations along the heading
        sa, sm, sb = sign * xa, sign * xm, sign * xb
        lat_lane = sign * wy  # lateral coordinate of the lane (left of heading)
        p1, h1 = s_curve(sa, lat_lane, sm, lat_lane - BAY_M, heading)
        # With reverse_exit_m, the car first reverses straight out of the berth (a gear change, as in a bay).
        back = reverse_exit_m  # station is already measured along heading
        p2, h2 = s_curve(sm - back, lat_lane - BAY_M, sb, lat_lane, heading)
        h1 = h1 + (yaw[ka] - h1[0])
        bid = f"berth:{name}"
        graph.add_node(bid, (p1[-1, 0], p1[-1, 1], h1[-1]), kind="berth")
        if parking_legality is not None:
            graph.nodes[bid]["parking_legality"] = dict(legal=parking_legality)
        graph.add_edge(f"entry:{name}", f"lane:{ka}", bid, route_arrays(p1, h1), kind="entry")
        if reverse_exit_m > 0:
            u = np.linspace(0.0, reverse_exit_m, int(math.ceil(reverse_exit_m / SPACING)) + 1)[1:]
            rp = p1[-1] - np.outer(u, (math.cos(heading), math.sin(heading)))
            pts_ = np.vstack([p1[-1:], rp, p2[1:]])
            yaw_ = np.r_[np.full(len(rp) + 1, h1[-1]), h2[1:] + (h1[-1] - h2[0])]
            gear = np.r_[np.full(len(rp), -1), np.ones(len(p2) - 1)]
            graph.add_edge(f"exit:{name}", bid, f"lane:{kb}", route_arrays(pts_, yaw_, gear), kind="exit")
        else:
            graph.add_edge(f"exit:{name}", bid, f"lane:{kb}", route_arrays(p2, h2 + (h1[-1] - h2[0])), kind="exit")
        access = (p1[-1, 0], p1[-1, 1] + (-1.0 if south else 1.0) * (CAR[1] / 2 + 0.75))
        berths.append(
            dict(id=bid, xy=(float(p1[-1, 0]), float(p1[-1, 1])), yaw=float(h1[-1]), access=access, south=south)
        )
        return bid

    homes, facilities = [], []
    for k, xm in enumerate((45.0, 75.0, 105.0)[: max(drivers, 1)]):
        b = bay(f"home{k}", xm, True)
        homes.append(dict(id=f"home{k}", xy=(xm, y0 - BAY_M - 6.0), berth=b))
    for k, (xm, kind) in enumerate(((55.0, "cafe"), (95.0, "post_office"))):
        b = bay(f"fac{k}", xm, False)
        facilities.append(dict(id=f"{kind}{k}", kind=kind, xy=(xm, y1 + BAY_M + 6.0), berth=b))
    # Walking: everywhere off the block; carriageway costs more.
    step = 0.5
    nx, ny = int(W / step), int(H / step)
    gx, gy = np.meshgrid((np.arange(nx)) * step + step / 2, (np.arange(ny)) * step + step / 2)
    free = ~((gx > BLOCK[0] - 0.6) & (gx < BLOCK[2] + 0.6) & (gy > BLOCK[1] - 0.6) & (gy < BLOCK[3] + 0.6))
    free &= (gx > 0.6) & (gx < W - 0.6) & (gy > 0.6) & (gy < H - 0.6)
    walking = WalkNetwork(free, np.array([step / 2, step / 2]), step=step)
    walking.area = rect(0.0, 0.0, W, H).difference(rect(*BLOCK))
    walking.surface = surface
    walking.radius = float(np.abs(PERSON_EXT[:, 2:]).max())
    router = LifeRouter(walking, walking.free)
    activity = None
    if parked_walking:
        # The resident cars standing parked are static walking geometry (awr.life.motion.parked_space); walks are
        # measured on the router of the current grid.
        from types import SimpleNamespace as NS

        from awr.life.civil.walking import ObstacleGrid

        grid = np.asarray(walking.free).copy().view(ObstacleGrid)
        for name, shape in (("positions", (0, 2)), ("sizes", (0, 3)), ("yaws", (0,)), ("circle_radii", (0,))):
            setattr(grid, name, np.zeros(shape))
        grid.radius, grid.static_free = 0.0, np.asarray(walking.free).copy()
        activity = NS(
            static_free=grid,
            network=walking,
            route_stats={},
            route_cache={},
            static_router=router,
            static_labels=None,
            source_components={},
            home_access_cache={},
        )
        activity.invalidate_routes = lambda: activity.route_cache.clear()
    places = []
    for k in range(walkers):
        places.append(Place(f"walker-home{k}", (32.0 + k * 7.0, 10.0 + (k % 2) * 3.0), "home", (f"walker-home{k}",)))
    for b in berths:
        places.append(Place(b["id"], b["xy"], "berth", (b["id"],), access_xy=b["access"]))
    for h in homes:
        places.append(Place(h["id"], h["xy"], "home", (h["id"],), parking=(h["berth"],)))
    for f in facilities:
        places.append(Place(f["id"], f["xy"], f["kind"], (f["id"],), parking=(f["berth"],), group=f["id"]))
        for j in range(walkers):  # standing places for walking visitors around the facility
            places.append(
                Place(
                    f"{f['id']}-stand{j}",
                    (f["xy"][0] - 6.0 + 2.0 * (j % 7), f["xy"][1] + 4.0 + 2.0 * (j // 7)),
                    "square",
                    (f"{f['id']}-stand{j}",),
                    group=f["id"] + "-square",
                )
            )
    owner_modes = ("walk", "car") if selection is not None else ("car",)
    activities = [
        Activity("coffee", ("cafe",), dwell, recurrence, modes=owner_modes),
        Activity("post", ("post_office",), dwell, recurrence, modes=owner_modes),
        Activity("stroll", ("square",), dwell, recurrence, roles=("walker",), modes=("walk",)),
    ]

    def walk_route(start, end):
        rr = (router if activity is None else activity.static_router).plan(
            np.asarray(start, float), np.asarray(end, float)
        )
        if rr is None:
            return None
        L = float(np.linalg.norm(np.diff(rr["points"], axis=0), axis=1).sum())
        return Route(L, L / 1.05, f"walk:{start}:{end}")

    driving_ref = {}

    def route(mode, start, end, entity):
        if mode == "walk":
            return walk_route(start, end)
        return driving_ref["routes"].route(mode, start, end, entity)

    director = Director(
        places,
        activities,
        route,
        seed=seed,
        home_dwell_s=(20.0, 40.0),
        retry_s=10.0,
        max_wait_s=600.0,
        max_speed_mps={"walk": 1.05, "drive": 6.0},
        selection=selection,
    )
    if activity is not None:
        activity.director, activity.life = director, {"planning_speed_mps": 1.05}
        activity.walk_allowed = lambda start, end: True
        activity.measured_route = lambda start, end: activity.static_router.plan(np.asarray(start), np.asarray(end))
    ids, kinds, ext, xy, yaw0, sizes, vehicle_ids = [], [], [], [], [], {}, {}
    agendas, vmax = {}, []
    for k in range(drivers):
        h = homes[k]
        aid, vid = f"person-{k}", f"vehicle-{k}"
        director.add_actor(Actor(aid, h["id"], vehicle=vid), t=0.0)
        ids.append(aid)
        kinds.append("person")
        ext.append(PERSON_EXT)
        xy.append(h["xy"])
        yaw0.append(math.pi / 2)
        vmax.append(1.3)
    for k in range(drivers):
        h = homes[k]
        b = next(x for x in berths if x["id"] == h["berth"])
        director.add_vehicle(Vehicle(f"vehicle-{k}", f"person-{k}", h["berth"]))
        vehicle_ids[len(ids)] = f"vehicle-{k}"
        sizes[len(ids)] = CAR
        ids.append(f"vehicle-{k}")
        kinds.append("vehicle")
        ext.append(CAR_EXT)
        xy.append(b["xy"])
        yaw0.append(b["yaw"])
        vmax.append(0.0)
    for k in range(walkers):
        aid = f"walker-{k}"
        director.add_actor(Actor(aid, f"walker-home{k}", role="walker"), t=0.0)
        agendas[len(ids)] = LifeAgenda(aid, director.places[f"walker-home{k}"], director, 0.0, 2.0 + 3.0 * k)
        ids.append(aid)
        kinds.append("person")
        ext.append(PERSON_EXT)
        xy.append(director.places[f"walker-home{k}"].xy)
        yaw0.append(math.pi / 2)
        vmax.append(1.25)
    for k in range(drivers):
        aid = f"person-{k}"
        agendas[k] = LifeAgenda(aid, director.places[homes[k]["id"]], director, 0.0, 5.0 + 5.0 * k)
    occupancy_bodies = {vid: Body((xy[i][0], xy[i][1], yaw0[i]), CAR) for i, vid in vehicle_ids.items()}
    areas = {e: walking.area for e in graph.edges}
    routes = DrivingRoutes(graph, surface, CAR, areas, occupancy_bodies)
    driving_ref["routes"] = routes
    drivers_exec = DriverExecution(director, routes)
    sizes_arr = {i: CAR for i in vehicle_ids}
    walk_space = None
    if activity is not None:
        from awr.life.motion.parked_space import ParkedSpace

        walk_space = ParkedSpace(activity, walking, 0.3)
    engine = LifeEngine(
        ids=ids,
        kinds=kinds,
        ext=np.stack(ext),
        xy=np.array(xy, float),
        yaw=np.array(yaw0, float),
        surface=surface,
        walking=walking,
        router=router,
        agendas=agendas,
        vmax=np.array(vmax),
        occupancy=Occupancy(),
        vehicle_ids=vehicle_ids,
        sizes=sizes_arr,
        driving=routes,
        driver_execution=drivers_exec,
        route_cache=None if activity is None else activity.route_cache,
        walk_space=walk_space,
    )
    return SimpleNamespace(
        engine=engine,
        director=director,
        graph=graph,
        routes=routes,
        places=places,
        walk_space=walk_space,
        berths=berths,
        homes=homes,
        facilities=facilities,
        surface=surface,
        walking=walking,
        router=router,
        ids=ids,
        kinds=kinds,
        ext=np.stack(ext),
        xy=np.array(xy, float),
        yaw=np.array(yaw0, float),
        vmax=np.array(vmax),
        agendas=agendas,
    )
