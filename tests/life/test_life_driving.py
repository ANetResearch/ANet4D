"""End-to-end driver chains on a synthetic town: the Director's car journeys executed by the life engine."""

import hashlib
import json

import life_toytown as toy
import numpy as np
import pytest

from awr.life.civil.activity.travel_evidence import TravelEvidence
from awr.life.civil.facts import write_facts
from awr.life.motion.crowd import DT


def run(world, seconds):
    """Step the engine; every tick: no hold, no collision (QA predicate), and the QA acceleration rule on recorded
    velocities of bodies visible in both ticks (people 8, vehicles 6 m/s2)."""
    e = world.engine
    prev_v = prev_state = None
    reversing_ticks = 0
    lim = np.where(e.kinds == "vehicle", 6.0, 8.0)
    for k in range(int(round(seconds / DT))):
        xy, yaw, speed, v, state, container, held = e.step(k * DT)
        reversing_ticks += int(np.any(speed[e.kinds == "vehicle"] < -1e-6))
        assert not held, (k * DT, held)
        assert e.collisions() == [], (k * DT, e.collisions())
        if prev_v is not None:
            both = np.isin(state, (1, 2)) & np.isin(prev_state, (1, 2))
            acc = np.linalg.norm(np.round(v * 100) * 0.01 - np.round(prev_v * 100) * 0.01, axis=1) / DT
            assert (acc[both] <= lim[both] + 1e-6).all(), (k * DT, acc[both].max())
        prev_v, prev_state = v.copy(), state.copy()
    return reversing_ticks


def test_two_drivers_complete_car_journeys():
    world = toy.build(drivers=2)
    run(world, 420.0)
    places = [dict(id=p.id, kind=p.kind, xy=list(p.xy), parking=list(p.parking), group=p.group) for p in world.places]
    evidence = TravelEvidence(world.director.events, places, 420.0)
    chains = {v: c for v, c in evidence.by_vehicle.items()}
    assert set(chains) == {"vehicle-0", "vehicle-1"}, world.engine.stats
    for rows in chains.values():
        assert any(r["place"].startswith(("cafe", "post_office")) for r in rows)
    s = world.engine.stats
    assert s["boardings"] >= 2 and s["alightings"] >= 2 and s["drives_completed"] >= 2


def test_shared_home_parking_executes_complete_return_chains():
    from dataclasses import replace

    from shapely.geometry import box

    from awr.life.civil.activity.driving_adapter import DrivingAdapter

    world = toy.build(drivers=3, walkers=0, parking_legality=True)
    adapter = DrivingAdapter(
        world.graph, toy.CAR, "toy", box(0, 0, 1, 1), world.walking, world.surface, 0.45, home_walk_m=40.0
    )
    usable = adapter.usable_berths(
        [], np.asarray(world.walking.free), None, lambda *args: True, lambda *args: dict(legal=True)
    )
    for home in world.homes:
        place = world.director.places[home["id"]]
        world.director.places[place.id] = replace(place, parking=adapter.home_parking(place, home["berth"], usable))
    # This public bay permits its existing occupant to leave but is unavailable
    # for new parking (e.g. a reservation). No obstacle/body is fabricated and no
    # motion state is changed; returning owners must use another admitted bay.
    world.director.available = lambda place, actor: place.id != "berth:home0"
    # Physical execution, not synthetic arrival acknowledgements: per-tick
    # collision/hold/acceleration checks are in run(). No routes or poses edited.
    # Three owners share two admitted home bays, so returns wait for a bay; with
    # concurrent traffic the order of those waits differs from serialized trips.
    run(world, 900.0)
    completed = {
        e["intent_id"]
        for e in world.director.events
        if e["kind"] == "activity_completed" and e["purpose"] == "return_home"
    }
    returns = [e for e in world.director.events if e["kind"] == "activity_selected" and e["intent_id"] in completed]
    assert {e["actor"] for e in returns} == {"person-0", "person-1", "person-2"}
    assert world.engine.stats["drives_completed"] >= 6
    alternatives = [e for e in returns if e["destination_berth"] and e["destination_berth"] != "berth:" + e["place"]]
    assert alternatives, returns
    for event in returns:
        if event["destination_berth"]:
            assert event["destination_berth"] in world.director.places[event["place"]].parking
            assert [s["kind"] for s in event["steps"]] == ["walk", "board", "drive", "alight", "walk", "activity"]


def test_static_tree_reuse_preserves_mixed_activity_trajectories_and_events():
    outcomes = []
    for capacity in (0, 16):
        world = toy.build(drivers=2, walkers=3, parking_legality=True)
        world.routes.static_tree_capacity = capacity
        digest = hashlib.sha256()
        for k in range(4200):
            *arrays, held = world.engine.step(k * DT)
            assert not held and not world.engine.collisions()
            for array in arrays:
                digest.update(np.ascontiguousarray(array).tobytes())
        assert world.engine.stats["drives_completed"] >= 2
        outcomes.append((digest.hexdigest(), world.engine.events, world.director.events))
    assert outcomes[0] == outcomes[1]


def test_indexed_occupancy_preserves_executed_mixed_activity():
    outcomes = []
    for indexed in (False, True):
        world = toy.build(drivers=2, walkers=3, parking_legality=True)
        world.routes.index_occupancy = indexed
        digest = hashlib.sha256()
        for k in range(4200):
            *arrays, held = world.engine.step(k * DT)
            assert not held and not world.engine.collisions()
            for array in arrays:
                digest.update(np.ascontiguousarray(array).tobytes())
        assert world.engine.stats["drives_completed"] >= 2
        outcomes.append((digest.hexdigest(), world.engine.events, world.director.events))
        if indexed:
            assert world.routes.stats["occupancy_index_builds"] > 0
    assert outcomes[0] == outcomes[1]


def test_drivers_and_walkers_share_the_loop_road():
    """Walkers cross the drivers' loop road to the squares north of it while the cars drive their errands."""
    world = toy.build(drivers=3, walkers=8)
    run(world, 420.0)
    places = [dict(id=p.id, kind=p.kind, xy=list(p.xy), parking=list(p.parking), group=p.group) for p in world.places]
    chains = TravelEvidence(world.director.events, places, 420.0).by_vehicle
    assert set(chains) == {"vehicle-0", "vehicle-1", "vehicle-2"}, world.engine.stats
    strolls = [e for e in world.director.events if e["kind"] == "activity_completed" and e["purpose"] == "stroll"]
    assert len(strolls) >= 8


def test_executed_transfers_export_observed_locations_and_trip_facts(tmp_path):
    """Exercise the production facts writer after actual walking and driving.

    A completed real-town journey exposed missing transfer coordinates only at
    export. Check door positions against the runtime, not against planned goals.
    """
    world = toy.build(drivers=1)
    e = world.engine
    positions, states = [], []
    transfers = []
    for k in range(1800):
        before = e.xy.copy()
        start = len(e.events)
        xy, yaw, speed, velocity, state, container, held = e.step(k * DT)
        positions.append(xy.copy())
        states.append(state.copy())
        assert not held and not e.collisions()
        for event in e.events[start:]:
            if event["kind"] not in ("board", "alight"):
                continue
            i = e.ids.index(event["actors"][0])
            v = e.vehicle_index[event["vehicle"]]
            observed = before[i] if event["kind"] == "board" else xy[i]
            np.testing.assert_allclose(event["xy"], observed, atol=1e-9)
            np.testing.assert_allclose(event["vehicle_xy"], xy[v], atol=1e-9)
            assert np.linalg.norm(observed - xy[v]) > 0.5  # door, not centre
            assert container[i] == (v if event["kind"] == "board" else -1)
            transfers.append(event)
    assert {r["kind"] for r in transfers} == {"board", "alight"}
    assert e.stats["drives_completed"] >= 1
    (tmp_path / "sim_events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in e.events))
    np.savez(tmp_path / "tracks.npz", xy=positions, state=states)
    actors = [
        dict(id=name, idx=i, **{"class": str(e.kinds[i])}, appearance={}, size_mm=[1000] * 3)
        for i, name in enumerate(e.ids)
    ]
    sem = dict(
        blocks=[dict(id="town", polygon=[[0, 0], [150, 0], [150, 100], [0, 100]])], curb_slots=[], parking_lots=[]
    )
    out = tmp_path / "facts.jsonl"
    write_facts(tmp_path, sem, {}, dict(persons=[], vehicles=[]), actors, np.arange(e.n), out, 180.0)
    facts = [json.loads(line) for line in out.read_text().splitlines()]
    exported = [r for r in facts if r["kind"] in ("board", "alight")]
    assert exported == [{**r, "block": "town"} for r in transfers]
    trips = [r for r in facts if r["kind"] == "trip"]
    assert len(trips) == e.stats["drives_completed"]
    assert all(r["t1_s"] > r["t0_s"] and r["to_block"] == "town" for r in trips)


@pytest.mark.parametrize("legal", [None, False])
def test_observed_parking_does_not_invent_legality(legal):
    world = toy.build(drivers=1, parking_legality=legal)
    run(world, 180.0)
    parked = [e for e in world.engine.events if e["kind"] == "park"]
    assert parked
    assert all(e["legal"] is legal for e in parked)


def test_reversing_vehicles_report_speed_magnitude_to_driving_backend():
    world = toy.build(drivers=2, reverse_exit_m=2.5)
    assert run(world, 420.0) > 0
    trips = [e for e in world.engine.events if e["kind"] == "park"]
    assert {e["actors"][0] for e in trips} == {"vehicle-0", "vehicle-1"}
    # Both sides of the loop have opposite headings. Reverse motion must
    # follow the body heading, not the sign of world x or y.
    for key, edge in world.routes.graph.edges.items():
        if key.startswith("exit:"):
            route = edge["route"]
            delta = route["points"][1] - route["points"][0]
            heading = route["yaw"][0]
            assert np.dot(delta, [np.cos(heading), np.sin(heading)]) < 0


def test_transfer_events_carry_their_position_with_reversing_exits():
    """Cars leave their berths in reverse first; the recorded speed is signed, the backend's observation is not."""
    world = toy.build(drivers=1, reverse_exit_m=2.5)
    run(world, 240.0)
    reversing = [
        k for k, e in world.routes.graph.edges.items() if k.startswith("exit:") and (e["route"]["gear"] < 0).any()
    ]
    assert reversing and world.engine.stats["drives_completed"] >= 1, world.engine.stats
    # Transfer events carry the place where they happen (the recorder's facts need it).
    transfers = [e for e in world.engine.events if e["kind"] in ("board", "alight")]
    assert transfers and all(len(e["xy"]) == 2 for e in transfers)


def test_a_parked_resident_car_is_walking_geometry_only_while_it_stands():
    """Paths are planned around a resident car while it stands parked, and through its berth once it has left."""
    from types import SimpleNamespace

    import numpy as np

    from awr.life.civil.walking import ObstacleGrid
    from awr.life.motion.parked_space import ParkedSpace

    world = toy.build(drivers=1)
    free = np.asarray(world.walking.free).copy().view(ObstacleGrid)
    for name, shape in (("positions", (0, 2)), ("sizes", (0, 3)), ("yaws", (0,)), ("circle_radii", (0,))):
        setattr(free, name, np.zeros(shape))
    free.radius, free.static_free = 0.0, np.asarray(world.walking.free).copy()
    activity = SimpleNamespace(
        static_free=free,
        network=world.walking,
        route_stats={},
        route_cache={},
        static_router=None,
        static_labels=None,
        source_components={},
        home_access_cache={},
    )
    space = ParkedSpace(activity, world.walking, 0.3)
    pose = world.graph.nodes["berth:home0"]["pose"]
    body = (float(pose[0]), float(pose[1]), float(pose[2]), *map(float, toy.CAR))
    far = dict(points=np.array([[60.0, 60.0], [61.0, 60.0]]))
    through = dict(points=np.array([pose[:2] - [3.0, 0.0], pose[:2] + [3.0, 0.0]]))
    activity.route_cache.update({"far": far, "through": through, "refused": None})
    router, closed = space.update({7: body})
    # Only the route through the parked car is dropped; the refusal stays while no cell opened.
    assert set(activity.route_cache) == {"far", "refused"}
    assert closed.any() and not np.asarray(activity.static_free)[closed].any()
    assert activity.static_router is router and len(activity.static_free.positions) == 1
    assert space.update({7: body}) is None  # unchanged: no rebuild
    router, reopened = space.update({})
    assert "refused" not in activity.route_cache and "far" in activity.route_cache  # cells opened: refusals dropped
    assert not reopened.any() and np.array_equal(np.asarray(activity.static_free), np.asarray(free.static_free))
    assert len(activity.static_free.positions) == 0 and space.stats["updates"] == 2


def test_drivers_and_walkers_with_parked_cars_as_walking_geometry():
    world = toy.build(walkers=4, drivers=2, parked_walking=True)
    assert world.walk_space.stats["updates"] == 1  # both cars stand at home from the start
    run(world, 420.0)
    assert world.engine.stats["drives_completed"] >= 2, world.engine.stats
    # Every departure and every parking changed the walking grid, and nobody walked through a standing car.
    assert world.walk_space.stats["updates"] >= 1 + world.engine.stats["drives_completed"]
    assert world.walk_space.future_stats["queries"] > 0
    selected = {
        r
        for event in world.director.events
        if event["kind"] == "activity_selected"
        for r in event["routes"]
        if r.startswith("measured-parked-walk:")
    }
    used = {
        event.get("activity_route_evidence")
        for event in world.engine.events
        if event["kind"] == "walking_route_assigned"
    }
    assert selected & used  # chosen parking proof actually reaches the walking executor
    assert world.engine.stats["walk_proofs_reused"] > 0
    assert any(event["kind"] == "activity_completed" for event in world.director.events)


def test_changed_parking_updates_router_cache_and_existing_walks():
    world = toy.build(drivers=1, parked_walking=True)
    e = world.engine

    def clear(a, b):
        return e.router.network.clear_segment(a, b, e.router.free)  # the engine's current router

    car = next(iter(e.vehicle_ids))
    vid = e.vehicle_ids[car]
    centre = e.xy[car].copy()
    a, b = centre + [-5.0, 0.0], centre + [5.0, 0.0]
    assert not clear(a, b)
    # Topology fixture only: no claim of an executed journey in this test.
    e.driving.active[vid] = "fixture-departure"
    e._update_walk_space()
    assert clear(a, b)  # the new snapshot
    assert e.reroutes.router is e.router
    e.xy[0] = a
    e.paths.set(0, np.array([a, b]))
    e.route_cache[(tuple(a), tuple(b))] = dict(points=np.array([a, b]))
    del e.driving.active[vid]
    e._update_walk_space(5.0)
    assert not clear(a, b)
    assert (tuple(a), tuple(b)) not in e.route_cache  # crosses the cells the parked car closed
    assert 0 not in e.paths
    assert e.stats["paths_closed_by_parking"] == 1
    closed = [x for x in e.events if x["kind"] == "walking_path_closed_by_parking"]
    assert [x["actors"] for x in closed] == [[e.ids[0]]] and closed[0]["parked_actors"] == [vid]
    count = world.walk_space.stats["updates"]
    e._update_walk_space()
    assert world.walk_space.stats["updates"] == count  # no per-tick rebuild


def test_departure_and_distant_parking_do_not_discard_an_observed_route():
    """Opening space cannot strand walkers whose observed connector is outside
    the planner's inset. This fixture tests invalidation, not physical motion.
    """
    world = toy.build(drivers=1, parked_walking=True)
    e = world.engine
    car = next(iter(e.vehicle_ids))
    vid = e.vehicle_ids[car]
    start = world.walking.origin - np.array([world.walking.step, 0.0])
    end = start + np.array([0.0, 3.0])
    e.xy[0] = start
    e.paths.set(0, np.array([start, end]))
    assert not e.router.network.clear_segment(start, end, e.router.free)
    assert np.linalg.norm(start - e.xy[car]) > 5.0
    e.driving.active[vid] = "fixture-departure"
    e._update_walk_space(10.0)
    assert 0 in e.paths
    assert not e.walk_space.changed_bodies
    # Returning far from this walker also cannot invalidate its connector.
    del e.driving.active[vid]
    e._update_walk_space(20.0)
    assert 0 in e.paths
    assert e.walk_space.changed_bodies == (car,)
    np.testing.assert_array_equal(e.paths.path(0).p, [start, end])
    assert e.stats.get("paths_closed_by_parking", 0) == 0


def test_moving_a_parked_body_invalidates_only_the_newly_obstructed_path():
    world = toy.build(drivers=1, parked_walking=True)
    e = world.engine
    car = next(iter(e.vehicle_ids))
    centre = e.xy[car].copy() + [0.0, 6.0]
    start, end = centre + [-5.0, 0.0], centre + [5.0, 0.0]
    e.xy[0] = start
    e.paths.set(0, np.array([start, end]))
    # Snapshot fixture: an observed parked pose changed; this is not a drive.
    e.xy[car] = centre
    e._update_walk_space(20.0)
    assert 0 not in e.paths
    event = e.events[-1]
    assert event["kind"] == "walking_path_closed_by_parking"
    assert event["actors"] == [e.ids[0]] and event["parked_actors"] == [e.ids[car]]
    assert event["t_s"] == 20.0
