"""Activity/control integration fixture, not a town or transfer-animation proof.

The vehicle runs the physical route controller. Walk/transfer observations are
supplied by a test backend to isolate causal handoffs and reservation lifetime.
"""

import math
from dataclasses import asdict

import numpy as np
import pytest
from shapely.geometry import box

from awr.life.civil.activity import Activity, Actor, Director, Place, Route, Vehicle
from awr.life.civil.activity.driver_execution import DriverExecution
from awr.life.civil.activity.driving import Body, DrivingRoutes
from awr.life.civil.activity.travel_evidence import TravelEvidence
from awr.life.civil.driving_graph import DrivingGraph, route_arrays

SIZE = (4.2, 1.8, 1.5)


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(
        "awr.life.civil.contacts.vehicle_contacts",
        lambda size: np.array([[-1.4, -0.7], [-1.4, 0.7], [1.4, -0.7], [1.4, 0.7]]),
    )

    class Ground:
        def supported(self, xy, profile):
            return np.ones(len(xy), bool)

    graph = DrivingGraph()
    poses = {"hb": (0.0, -20.0, 0.0), "sb": (0.0, 20.0, math.pi)}
    for n, p in poses.items():
        graph.add_node(n, p, kind="berth")
    for a, b, start in [("hb", "sb", -math.pi / 2), ("sb", "hb", math.pi / 2)]:
        theta = np.linspace(start, start + math.pi, 300)
        graph.add_edge(
            a + "-" + b,
            a,
            b,
            route_arrays(np.column_stack([20 * np.cos(theta), 20 * np.sin(theta)]), theta + math.pi / 2),
            kind="travel",
        )
    driving = DrivingRoutes(
        graph,
        Ground(),
        SIZE,
        {k: box(-25, -25, 25, 25) for k in graph.edges},
        {"car": Body(poses["hb"], SIZE), "parked": None},
    )
    places = [
        Place("home", (0.0, -24.0), "home", ("home-slot",), parking=("hb",)),
        Place("shop", (0.0, 24.0), "shop", ("shop-slot",), parking=("sb",)),
        Place("hb", (0.0, -20.0), "berth", ("hb-slot",), access_xy=(0.0, -22.0)),
        Place("sb", (0.0, 20.0), "berth", ("sb-slot",), access_xy=(0.0, 22.0)),
    ]

    def route(mode, a, b, entity):
        if mode == "drive":
            return driving.route(mode, a, b, entity)
        return Route(math.dist(a, b), math.dist(a, b), "walk-observation-fixture")

    director = Director(
        places,
        [Activity("shopping", ("shop",), (2.0, 2.0), 200.0, modes=("car",))],
        route,
        home_dwell_s=(2.0, 2.0),
        errands_before_home=1,
    )
    director.add_actor(Actor("driver", "home", vehicle="car", completed={"shopping": -1000.0}))
    director.add_vehicle(Vehicle("car", "driver", "hb"))
    return director, driving, DriverExecution(director, driving), places


def ready(world):
    d, r, x, _ = world
    t = 1.0
    xy = (0.0, -24.0)
    intent = d.choose("driver", t, xy)
    assert intent.step.kind == "walk"
    t += 2.0
    xy = intent.step.xy
    assert d.acknowledge("driver", t, xy, 0.0)
    assert intent.step.kind == "board"
    assert not d.acknowledge("driver", t, xy, 0.0, vehicle_xy=(0.0, -20.0))
    t += 4.0
    assert d.acknowledge("driver", t, xy, 0.0, vehicle_xy=(0.0, -20.0))
    return t, intent


def test_revision_chain_keeps_the_owned_activity_and_arrival_binding(world):
    """Isolate the activity handoff; route geometry is exercised separately.

    Two congruent test plan IDs stand in for already-validated backend
    revisions. This must update the activity without changing its destination
    or authorizing a different owner, and must not consume an invalid chain.
    """
    t, intent = ready(world)
    director, routes, execution, _ = world
    old = intent.step.route.evidence
    assert execution.advance("driver", t, (0.0, -20.0), "car", (0.0, -20.0, 0.0), 0.0, 0.1)
    original = intent.step
    new1, new2 = old + ":revision-fixture-1", old + ":revision-fixture-2"
    routes.plans[new1] = dict(routes.plans[old])
    routes.plans[new2] = dict(routes.plans[old])
    queue = [(old, new1, 20.0, "parking:fixture"), (new1, new2, 30.0, "parking:fixture")]
    routes.revisions["car"] = list(queue)
    routes.active["car"] = new2
    routes.plans[new2]["target"] = "hb"  # wrong destination: reject the whole chain
    with pytest.raises(ValueError, match="destination berth"):
        execution.follow_revision("driver")
    assert intent.step is original and routes.revisions["car"] == queue
    assert execution.bindings["driver"] == (intent.id, old)
    routes.plans[new2]["target"] = "sb"
    assert execution.follow_revision("driver") == queue
    assert intent.step.route.evidence == new2 and intent.step.xy == original.xy
    assert execution.bindings["driver"] == (intent.id, new2)
    assert execution._step("driver")[2] == new2
    assert execution.follow_revision("driver") == []


def test_shared_execution_preserves_an_away_drivers_return_route(world):
    """The production adapter must connect parking promises to activity choice.

    The shop berth sits across a second driver's return road. Its own roundtrip
    is feasible, but accepting it would strand that driver. No per-actor rule or
    route-wide traffic lock is needed to reject that parking destination.
    """
    original, old, _, base_places = world
    graph = DrivingGraph()
    places = list(base_places)
    for name, node in old.graph.nodes.items():
        graph.add_node(name, **node)
    for name, edge in old.graph.edges.items():
        graph.add_edge(name, **edge)
    poses = {"away": (-25.0, 20.0, 0.0), "back": (25.0, 20.0, 0.0)}
    for name, pose in poses.items():
        graph.add_node(name, pose, kind="berth")
        places.append(Place(name, pose[:2], "berth", (name + "-slot",), access_xy=(pose[0], pose[1] + 2.0)))
    graph.add_edge(
        "return-road",
        "away",
        "back",
        route_arrays(np.linspace(poses["away"][:2], poses["back"][:2], 200), np.zeros(200)),
        kind="travel",
    )
    places.append(Place("second-home", (25.0, 24.0), "home", ("second-home-slot",), parking=("back",)))
    r = DrivingRoutes(
        graph,
        old.surface,
        SIZE,
        dict(old.areas, **{"return-road": box(-30, 15, 30, 25)}),
        dict(old.occupancy, **{"second-car": Body(poses["away"], SIZE)}),
    )

    def route(mode, a, b, entity):
        if mode == "drive":
            return r.route(mode, a, b, entity)
        return Route(math.dist(a, b), math.dist(a, b), "walk-observation-fixture")

    d = Director(places, original.activities, route)
    d.add_actor(Actor("driver", "home", vehicle="car", completed={"shopping": -1000.0}))
    d.add_vehicle(Vehicle("car", "driver", "hb"))
    d.add_actor(Actor("second-driver", "second-home", vehicle="second-car"))
    d.add_vehicle(Vehicle("second-car", "second-driver", "away"))
    DriverExecution(d, r)
    assert r.plan("second-car", "away", "back") is not None
    assert d.choose("driver", 1.0, (0.0, -24.0)) is None
    assert d.car_rejections["would_close_a_way_home"] > 0
    assert not r.active  # a rejected activity does not acquire traffic space

    # Once the second driver is home, its old return road is not permanently
    # excluded. The same shop trip becomes a valid activity again.
    d.vehicles["second-car"].berth = "back"
    r.update_occupancy(dict(r.occupancy, **{"second-car": Body(poses["back"], SIZE)}))
    intent = d.choose("driver", 100.0, (0.0, -24.0))
    assert intent is not None and intent.destination_berth == "sb"


def test_boarding_logic_does_not_substitute_for_actual_containment(world):
    t, intent = ready(world)
    d, r, x, _ = world
    assert d.vehicles["car"].occupant == "driver"
    with pytest.raises(ValueError, match="not contained"):
        x.advance("driver", t, (0.0, -22.0), None, (0.0, -20.0, 0.0), 0.0, 0.1)
    assert not r.active
    # Nominal elapsed time, even with a fabricated destination observation,
    # cannot finish a physical journey that never started.
    assert not x.finish("driver", t + 1000, (0.0, 20.0), "car", (0.0, 20.0, math.pi), 0.0)
    assert intent.step.kind == "drive"


def test_live_car_activity_roundtrip_waits_brakes_parks_and_serves(world):
    d, r, x, places = world
    t = 1.0
    xy = (0.0, -24.0)
    pose = np.array([0.0, -20.0, 0.0])
    speed = 0.0
    traces = []
    for journey in range(2):
        intent = d.choose("driver", t, xy)
        assert intent.destination_berth is not None
        while "driver" in d.active:
            step = intent.step
            if step.kind != "drive":
                # Only the observation handshake is exercised for these steps.
                t += step.route.duration_s if step.route else 0.1
                xy = step.xy
                kwargs = (
                    {"vehicle_xy": tuple(pose[:2]), "vehicle_speed": speed} if step.kind in ("board", "alight") else {}
                )
                if step.duration_s:
                    assert not d.acknowledge("driver", t, xy, 0.0, **kwargs)
                    t += step.duration_s + 0.1  # next observed frame after dwell
                assert d.acknowledge("driver", t, xy, 0.0, **kwargs)
                continue
            depart = t
            source = d.vehicles["car"].berth
            obstruction = None
            held_until = None
            for tick in range(2000):
                if journey == 0 and tick == 20:
                    e = r.executions["car"]
                    obstruction = e.arc_m + e.speed_mps**2 / (2 * r.limits.braking_mps2) + 0.2
                    held_until = t + 5.0
                result = x.advance(
                    "driver",
                    t,
                    pose[:2],
                    "car",
                    pose,
                    speed,
                    0.1,
                    stop_before_arc_m=obstruction if held_until is not None and t < held_until else None,
                )
                assert result is not None
                t += 0.1
                next_pose = np.r_[result["xy"], result["yaw"]]
                if result["arrived"]:
                    # A proposal is not the next actual frame.
                    assert not x.finish("driver", t, next_pose[:2], "car", next_pose, result["speed_mps"])
                pose = next_pose
                speed = result["speed_mps"]
                xy = tuple(pose[:2])
                r.update_occupancy(dict(r.occupancy, car=Body(pose, SIZE)))
                if result["arc_m"] > 8.0:
                    assert d.holders.get(source + "-slot") != "car"
                if x.finish("driver", t, xy, "car", pose, speed):
                    break
                assert intent.step.kind == "drive" and r.active
            else:
                pytest.fail("Physical drive did not finish")
            assert intent.step.kind == "alight" and not r.active
            assert d.vehicles["car"].berth == intent.destination_berth
            assert d.vehicles["car"].occupant == "driver"  # still inside until alight
            traces.append({"vehicle": "car", "t0_s": depart, "t1_s": t, "end_xy_m": list(xy)})
        t += 0.1
    assert len(r.completed_motion) == 2
    assert r.completed_motion[0]["brake_count"] == 1
    assert r.completed_motion[1]["brake_count"] == 0
    evidence = TravelEvidence(d.events, [asdict(p) for p in places], t + 1.0)
    chains = [evidence.match(trip) for trip in traces]
    assert all(chains)
    assert [c["purpose"] for c in chains] == ["shopping", "return_home"]
    assert len({c["intent_id"] for c in chains}) == 2
    assert not x.bindings and not x.last_proposal_time


def test_world_ticks_and_actual_pose_cannot_skip_or_replay(world):
    t, _ = ready(world)
    d, r, x, _ = world
    pose = np.array([0.0, -20.0, 0.0])
    result = x.advance("driver", t, pose[:2], "car", pose, 0.0, 0.1)
    with pytest.raises(ValueError, match="world progression"):
        x.advance("driver", t, pose[:2], "car", pose, 0.0, 0.1)
    with pytest.raises(ValueError, match="Measured motion differs"):
        x.advance("driver", t + 0.1, pose[:2], "car", pose, 0.0, 0.1)
    pose = np.r_[result["xy"], result["yaw"]]
    r.update_occupancy(dict(r.occupancy, car=Body(pose, SIZE)))
    with pytest.raises(ValueError, match="world progression"):
        x.advance("driver", t + 1.0, pose[:2], "car", pose, result["speed_mps"], 0.1)
    # Rejected calls do not consume a motion tick.
    assert x.advance("driver", t + 0.1, pose[:2], "car", pose, result["speed_mps"], 0.1)["arc_m"] > result["arc_m"]


def test_intent_cannot_reuse_another_activity_geometry_or_pause_a_moving_car(world):
    t, intent = ready(world)
    d, r, x, _ = world
    evidence = intent.step.route.evidence
    r.plans[evidence]["vehicle"] = "somebody-else"
    with pytest.raises(ValueError, match="does not belong"):
        x.advance("driver", t, (0.0, -20.0), "car", (0.0, -20.0, 0.0), 0.0, 0.1)
    r.plans[evidence]["vehicle"] = "car"
    d.pause("driver", t)
    with pytest.raises(ValueError, match="courtesy pause"):
        x.advance("driver", t, (0.0, -20.0), "car", (0.0, -20.0, 0.0), 0.0, 0.1)
    assert not r.active


def test_changed_occupancy_holds_departure_at_source_without_finishing_activity(world):
    t, intent = ready(world)
    d, r, x, _ = world
    r.update_occupancy(dict(r.occupancy, parked=Body((20.0, 0.0, 0.0), SIZE)))
    for elapsed in (0.0, 0.1, 10.0):
        assert x.advance("driver", t + elapsed, (0.0, -20.0), "car", (0.0, -20.0, 0.0), 0.0, 0.1) is None
    assert not r.active and not x.bindings and not r.executions
    assert intent.step.kind == "drive" and d.vehicles["car"].berth == "hb"
    assert d.holders["hb-slot"] == d.holders["sb-slot"] == "car"
    assert len([e for e in d.events if e["kind"] == "drive_departure_wait"]) == 1
    # A new complete observation explicitly clears the temporary obstruction.
    r.update_occupancy(dict(r.occupancy, parked=None))
    proposal = x.advance("driver", t + 10.1, (0.0, -20.0), "car", (0.0, -20.0, 0.0), 0.0, 0.1)
    assert 0 < proposal["arc_m"] < 0.01 and proposal["speed_mps"] > 0
    assert not x.waiting
