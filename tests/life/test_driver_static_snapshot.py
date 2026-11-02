"""Owned-car admission must preserve permanent navigation geometry and live bodies."""

import json
from types import SimpleNamespace

import life_toytown as toy
import numpy as np
import pytest
from shapely.geometry import box

from awr.life.civil.activity.director import Activity, Actor, Director, Place
from awr.life.civil.grounded import admit_drivers, permanent_parking, resident_parking_evidence


def test_autonomous_stop_is_not_a_lifetime_static_obstacle():
    parked = dict(kind="vehicle", goal_s=0.0, speed=0.0)
    assert permanent_parking(parked)
    assert not permanent_parking(dict(parked, motion_policy="autonomous_activity"))
    assert not permanent_parking(dict(parked, goal_s=100.0))
    assert not permanent_parking(dict(parked, after="pending_departure"))


def test_admitting_owned_car_keeps_static_masks_and_registers_live_body(tmp_path, monkeypatch):
    w = toy.build(drivers=1, walkers=0)
    home = Place("home", tuple(w.homes[0]["xy"]), "home", ("home-slot",))
    facility = Place("cafe", tuple(w.facilities[0]["xy"]), "cafe", ("cafe-slot",))
    director = Director(
        [home, facility], [Activity("coffee", ("cafe",), (60.0, 60.0), 600.0, modes=("walk", "car"))], lambda *a: None
    )
    director.add_actor(Actor("person-0", "home"))
    runtime = SimpleNamespace(
        modes=("walk", "drive"), director=director, measured_walk=lambda a, b: float(np.linalg.norm(np.subtract(a, b)))
    )
    m = tmp_path / "map.json"
    m.write_text(
        json.dumps(
            dict(
                schema="awr.life.driving_map_candidate.v1",
                asset="toy",
                size_m=toy.CAR,
                nodes=w.graph.nodes,
                edges=w.graph.edges,
            ),
            default=lambda x: x.tolist(),
        )
    )
    source = tmp_path / "source.json"
    source.write_text("{}")
    monkeypatch.setattr("awr.life.civil.grounded.load", lambda _: ({"dimensions_m": toy.CAR}, None, None))
    monkeypatch.setattr("awr.life.civil.road_permission.driving_area", lambda *a: box(0, 0, toy.W, toy.H))
    walking = w.walking
    # Toy geometry is fully supported; production WalkNetwork.build supplies
    # both immutable turning masks alongside its clearance mask.
    walking.turn_free = np.asarray(walking.free).copy()
    walking.turn_clear = np.asarray(walking.free).copy()
    free = walking.obstacle_mask([], [], 0.45, [])
    before = np.asarray(free).copy()
    turns = walking.turn_free.copy() if walking.turn_free is not None else None
    original_clear = walking.turn_clear
    actors = [dict(id="person-0", **{"class": "person"})]
    jobs = [dict(kind="person", speed=0.0)]
    positions = [np.asarray(home.xy)]
    yaws = [0.0]
    sizes = [np.array([0.6, 0.6, 1.7])]
    permanent = set()
    meta = {}

    def add_vehicle(index, key, spec, size, xy, yaw, route, role):
        i = len(actors)
        actors.append(dict(id="vehicle-" + str(index), **{"class": "vehicle"}, size_mm=np.rint(size * 1000).tolist()))
        positions.append(xy)
        yaws.append(yaw)
        sizes.append(size)
        return i

    admit_drivers(
        dict(map=str(m), source_graph=str(source), drivers=1, home_walk_m=40.0),
        runtime,
        w.surface,
        {},
        walking,
        free,
        0.45,
        0.3,
        [lambda *a: True],
        actors,
        jobs,
        positions,
        yaws,
        sizes,
        add_vehicle,
        permanent,
        0.0,
        meta,
    )
    assert meta["driving"]["drivers"] == 1
    np.testing.assert_array_equal(free, before)
    if turns is not None:
        np.testing.assert_array_equal(walking.turn_free, turns)
    assert walking.turn_clear is original_clear
    assert not permanent
    assert jobs[-1]["motion_policy"] == "autonomous_activity" and not permanent_parking(jobs[-1])
    vid = actors[-1]["id"]
    assert runtime.driving_binding.routes.occupancy[vid] is not None
    assert director.vehicles[vid].owner == "person-0"


def test_activity_berth_parking_preserves_pose_and_requires_explicit_scene_rule_result():
    w = toy.build(drivers=1, walkers=0)
    berth = "berth:home0"
    pose = w.graph.nodes[berth]["pose"].copy()
    info = dict(legal=True, reasons=[], where="roadside")
    result = resident_parking_evidence(w.graph, berth, pose[:2], pose[2], info)
    assert result["slot"] == berth and result["legal"] is True
    assert "slot" not in info
    for legal in (False, None):
        with pytest.raises(ValueError):
            resident_parking_evidence(w.graph, berth, pose[:2], pose[2], dict(legal=legal))
    with pytest.raises(ValueError):
        resident_parking_evidence(w.graph, berth, pose[:2] + [1.0, 0.0], pose[2], info)
    np.testing.assert_array_equal(w.graph.nodes[berth]["pose"], pose)
