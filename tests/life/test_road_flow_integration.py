"""Independent checks for flow inheritance and preferred-lane routing."""

import math

import numpy as np
import pytest
from shapely.geometry import box
from test_activity_driving import SIZE, Ground, contacts  # noqa: F401 (contacts: autouse fixture)

from awr.life.civil.activity.driving import Body, DrivingRoutes
from awr.life.civil.driving_graph import DrivingGraph, route_arrays


def live_drive(routes, plans, starts, ticks=1200):
    # Unlike test_activity_driving.drive's reservation-only proposal loop,
    # supply the same live stopping input and LiveDrive backend as LifeEngine.
    from awr.life.civil.curb_allocation import footprint
    from awr.life.motion.vehicles import (
        LOOKAHEAD_S,
        LOOKAHEAD_STEP_M,
        PERSON_ROOM_M,
        PERSON_ROOM_S,
        VEHICLE_ROOM_M,
        LiveDrive,
        stop_arc,
    )

    names = sorted(starts)
    xy = np.array([starts[v][:2] for v in names])
    yaw = np.array([starts[v][2] for v in names])
    speeds = np.zeros(len(names))
    ext = np.tile(np.array([[-SIZE[0] / 2, SIZE[0] / 2, -SIZE[1] / 2, SIZE[1] / 2]] * 2), (len(names), 1, 1))
    active = np.ones(len(names), bool)
    ignore = np.zeros(len(names), bool)
    person = ignore.copy()
    times = np.array([0.1, 1.0, 2.5])
    pred = np.zeros((len(names), len(times)))
    frames = []
    for _ in range(ticks):
        observed_xy, observed_yaw, observed_speed = xy.copy(), yaw.copy(), speeds.copy()
        for i, v in enumerate(names):
            if v not in routes.active:
                if not routes.acquire(plans[v].evidence, (*xy[i], yaw[i]), speeds[i]):
                    continue
                routes.executions[v] = LiveDrive(routes.plans[plans[v].evidence]["schedule"].route, routes.limits)
            ex = routes.executions[v]
            if ex.arrived:
                continue
            route = ex.route
            length = speeds[i] ** 2 / (2 * routes.limits.braking_mps2) + speeds[i] * LOOKAHEAD_S + 4.0
            stop = stop_arc(
                i,
                route["points"],
                route["yaw"],
                route["arc"],
                ex.arc_m,
                speeds[i],
                length,
                ext,
                observed_xy[:, 0],
                observed_xy[:, 1],
                observed_yaw,
                observed_speed,
                person,
                active,
                ignore,
                pred,
                pred,
                pred,
                times,
                PERSON_ROOM_M,
                PERSON_ROOM_S,
                VEHICLE_ROOM_M,
                LOOKAHEAD_STEP_M,
            )
            proposal = routes.advance_motion(
                plans[v].evidence,
                np.r_[xy[i], yaw[i]],
                speeds[i],
                0.1,
                stop_before_arc_m=None if not math.isfinite(stop) else stop,
            )
            xy[i], yaw[i], speeds[i] = proposal["xy"], proposal["yaw"], proposal["speed_mps"]
        routes.update_occupancy({v: Body((*xy[i], yaw[i]), SIZE) for i, v in enumerate(names)})
        bodies = [routes.occupancy[v] for v in "ab"]
        a, b = [footprint(dict(xy=p.pose[:2], yaw=p.pose[2]), p.size, margin_m=0.0) for p in bodies]
        frame = dict(
            t_s=(_ + 1) * 0.1, gap=a.distance(b), planning_gap=bodies[0].polygon().distance(bodies[1].polygon())
        )
        if frame["gap"] == 0.0:
            frame.update(
                overlap_area_m2=a.intersection(b).area, xy=xy.tolist(), yaw=yaw.tolist(), speed_mps=speeds.tolist()
            )
        frames.append(frame)
        if all(v in routes.executions and routes.executions[v].arrived for v in names):
            break
    return frames


def merging(lead):
    g = DrivingGraph()
    nodes = dict(
        a0=(-40.0 + lead, 0.0, 0.0),
        b0=(-40.0, 3.2, 0.0),
        b1=(-30.0, 3.2, 0.0),
        m=(10.0, 0.0, 0.0),
        b_end=(30.0, 0.0, 0.0),
        a_end=(50.0, 0.0, 0.0),
        a_start=(-50.0 + lead, 0.0, 0.0),
        b_start=(-50.0, 3.2, 0.0),
        a_finish=(65.0, 0.0, 0.0),
        b_finish=(45.0, 4.0, 0.0),
    )
    for name, pose in nodes.items():
        g.add_node(
            name,
            pose,
            kind="berth" if name.endswith(("start", "finish")) else "lane",
            fragment=2 if name.startswith("b") else 1,
            traffic_resource="road:r",
            direction="forward",
        )
    for name, a, b, kind in [
        ("a", "a0", "m", "travel"),
        ("b", "b0", "b1", "travel"),
        ("merge", "b1", "m", "join"),
        ("common", "m", "b_end", "travel"),
        ("after", "b_end", "a_end", "travel"),
        ("start_a", "a_start", "a0", "exit"),
        ("start_b", "b_start", "b0", "exit"),
        ("finish_a", "a_end", "a_finish", "entry"),
        ("finish_b", "b_end", "b_finish", "entry"),
    ]:
        u = np.linspace(0.0, 1.0, 201)
        x0, y0, _ = nodes[a]
        x1, y1, _ = nodes[b]
        xy = np.column_stack([x0 + (x1 - x0) * u, y0 + (y1 - y0) * (3 * u * u - 2 * u * u * u)])
        yaw = np.arctan2((y1 - y0) * (6 * u - 6 * u * u), x1 - x0)
        g.add_edge(name, a, b, route_arrays(xy, yaw), kind=kind)
    starts = {v: nodes[v + "_start"] for v in "ab"}
    r = DrivingRoutes(
        g, Ground(), SIZE, {e: box(-60, -10, 75, 15) for e in g.edges}, {v: Body(p, SIZE) for v, p in starts.items()}
    )
    return r, starts


@pytest.mark.parametrize("lead", [0.0, 3.0, 8.0])
def test_same_direction_merge_retains_live_body_safety_and_progress(lead):
    r, starts = merging(lead)
    plans = {v: r.plan(v, v + "_start", v + "_finish") for v in "ab"}
    assert all(plans.values())
    frames = live_drive(r, plans, starts)
    assert min(f["gap"] for f in frames) > 0.0
    assert all(v in r.executions and r.executions[v].state()["arrived"] for v in "ab")


def parallel():
    g = DrivingGraph()
    g.add_node("s", (-30.0, 0.0, 0.0), kind="berth")
    g.add_node("t", (30.0, 0.0, 0.0), kind="berth")
    for name, y, mode in [("shared", 0.0, "exclusive_bidirectional"), ("own", -4.0, "separated_two_way")]:
        for suffix, x in [("0", -20.0), ("1", 20.0)]:
            g.add_node(name + suffix, (x, y, 0.0), kind="lane", traffic_mode=mode)
        for a, b, kind in [("s", name + "0", "join"), (name + "0", name + "1", "travel"), (name + "1", "t", "join")]:
            start, end = g.nodes[a]["pose"], g.nodes[b]["pose"]
            u = np.linspace(0.0, 1.0, 81)
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            xy = np.column_stack([start[0] + dx * u, start[1] + dy * (3 * u * u - 2 * u * u * u)])
            g.add_edge(a + "-" + b, a, b, route_arrays(xy, np.arctan2(dy * (6 * u - 6 * u * u), dx)), kind=kind)
    return g


def test_preferred_lane_blockage_falls_back_without_stale_snapshot():
    g = parallel()
    areas = {e: box(-40, -10, 40, 10) for e in g.edges}
    occupancy = {"car": Body((-30.0, 0.0, 0.0), SIZE), "blocker": Body((0.0, -4.0, 0.0), SIZE)}
    r = DrivingRoutes(g, Ground(), SIZE, areas, occupancy)
    p = r.plan("car", "s", "t")
    assert p is not None
    assert "shared0-shared1" in r.plans[p.evidence]["edges"]
    r.update_occupancy({**occupancy, "blocker": Body((0.0, 8.0, 0.0), SIZE)})
    p = r.plan("car", "s", "t")
    assert p is not None
    assert "own0-own1" in r.plans[p.evidence]["edges"]
    # Remaining route length is geometry, not weighted preference cost.
    assert np.isclose(p.length_m, sum(float(g.edges[e]["route"]["arc"][-1]) for e in r.plans[p.evidence]["edges"]))
