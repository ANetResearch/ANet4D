"""The builder entry of the life runtime (awr.life.motion.runtime.run) drives LifeEngine and records the tape arrays."""

import json
from pathlib import Path
from types import SimpleNamespace

import life_toytown as toy
import numpy as np
import pytest

from awr.life.civil.activity.grounded import ReactiveAgenda
from awr.life.civil.activity.travel_evidence import TravelEvidence
from awr.life.motion.binding import DrivingBinding
from awr.life.motion.capability import CapabilityRefused
from awr.life.motion.runtime import run


def context(world, seconds, tmp_path, agendas):
    n = len(world.ids)
    nt = int(round(seconds * 10))
    jobs = [dict(kind=k, max_speed=float(world.vmax[i])) for i, k in enumerate(world.kinds)]
    for i, ag in agendas.items():
        jobs[i]["agenda"] = ag
    activity = SimpleNamespace(
        network=world.walking,
        static_free=world.walking.free,
        static_router=world.router,
        route_stats={},
        route_cache={},
        update_frame=lambda *a: None,
        director=world.director,
    )
    actors = [
        dict(
            id=a,
            **(
                {"size_mm": np.rint(np.array(world.engine.sizes[i]) * 1000).astype(int).tolist()}
                if i in (world.engine.sizes or {})
                else {}
            ),
        )
        for i, a in enumerate(world.ids)
    ]
    return dict(
        cfg={"life_threads": 2},
        jobs=jobs,
        actors=actors,
        n=n,
        out=Path(tmp_path) / "tape",
        warm=0.0,
        nt=nt,
        tracks=np.zeros((nt, n, 2)),
        vel=np.zeros((nt, n, 2)),
        yaw_track=np.zeros((nt, n)),
        state=np.zeros((nt, n), np.int8),
        container_track=np.zeros((nt, n), int),
        events=[],
        occupancy=world.engine.occupancy,
        rest_r=np.full(n, 0.4),
        xy=world.xy,
        yaw=world.yaw,
        ext=world.ext,
        surface=world.surface,
        walking=world.walking,
        activity_runtime=activity,
    )


def test_runtime_records_walkers_with_reactive_agendas(tmp_path):
    world = toy.build(walkers=6, drivers=0)
    # The builder's town adapter gives ReactiveAgenda (no car journeys); the engine walks them unchanged.
    d = world.director
    agendas = {
        i: ReactiveAgenda(ag.actor, d.places[d.actors[ag.actor].home], d, 0.0, ag.visits[0].dwell_s)
        for i, ag in world.agendas.items()
    }
    ctx = context(world, 240.0, tmp_path, agendas)
    xy, yaw = run(ctx)
    profile = ctx["life_runtime_profile"]
    assert profile["ticks"] == 2400 and set(profile["stages_s"]) >= {"decisions", "motion", "record", "gates"}
    slow = profile["slowest_ticks"]
    assert len(slow) == 16 and slow[0]["wall_s"] >= slow[-1]["wall_s"]
    assert abs(slow[0]["wall_s"] * 1000 - profile["tick_ms"]["max"]) <= 0.0051
    assert all(r["phase"] == "recording" and 0 <= r["tick"] < 2400 for r in slow)
    assert (ctx["state"] == 1).any() and np.isin(ctx["state"], (1, 2)).all()
    moved = np.linalg.norm(ctx["tracks"][-1] - ctx["tracks"][0], axis=1)
    assert (moved > 5.0).sum() >= 3
    strolls = [e for e in world.director.events if e["kind"] == "activity_completed"]
    assert len(strolls) >= 3


def test_runtime_refuses_vehicle_trips(tmp_path):
    world = toy.build(walkers=1, drivers=0)
    ctx = context(world, 1.0, tmp_path, dict(world.agendas))
    ctx["jobs"].append(dict(kind="vehicle", goal_s=40.0))
    ctx["actors"].append(dict(id="car"))
    with pytest.raises(ValueError, match="maps no vehicle trips"):
        run(ctx)


def test_runtime_binds_backend_revision_before_motion_and_preserves_activity(tmp_path, monkeypatch):
    """A congruent plan ID isolates the real runtime/event adapter handoff.

    Geometric detours and collision checks are tested with the actual planner
    separately. Here no pose is replaced; the full walk/board/drive/park/alight/
    service chain still executes through LifeEngine and runtime recording.
    """
    w = toy.build(walkers=1, drivers=1, parking_legality=True)
    ctx = context(w, 420.0, tmp_path, dict(w.agendas))
    ctx["driving_binding"] = DrivingBinding(w.routes)
    ctx["cfg"].update(
        early_motion_gate=True,
        life_gates={"person_longest_stall_s_max": 60.0, "vehicle_longest_stop_in_trip_s_max": 60.0},
    )
    original = w.routes.prepare_revisions
    revised = []

    def prepare(t_s=None):
        original(t_s)
        if revised:
            return
        for vehicle, old in w.routes.active.items():
            controller = w.routes.executions.get(vehicle)
            if controller is None or controller.arc_m < 2.0:
                continue
            new = old + ":congruent-event-fixture"
            w.routes.plans[new] = dict(w.routes.plans[old])
            assert w.routes.traffic.revise(vehicle, new)
            w.routes.active[vehicle] = new
            w.routes.revisions[vehicle] = [(old, new, controller.arc_m + 1.0, "event-fixture")]
            revised.append((vehicle, old, new))
            break

    monkeypatch.setattr(w.routes, "prepare_revisions", prepare)
    run(ctx)
    assert len(revised) == 1
    vehicle, old, new = revised[0]
    rows = [e for e in ctx["events"] if e["kind"] == "reroute"]
    assert len(rows) == 1
    event = rows[0]
    assert event["actors"] == [vehicle] and event["replaces"] == old and event["evidence"] == new
    parks = [e for e in ctx["events"] if e["kind"] == "park" and e.get("intent_id") == event["intent_id"]]
    assert len(parks) == 1 and parks[0]["evidence"] == new
    completed = [
        e for e in w.director.events if e["kind"] == "activity_completed" and e["intent_id"] == event["intent_id"]
    ]
    assert len(completed) == 1 and completed[0]["t_s"] > parks[0]["t_s"]


def test_runtime_checks_the_environment_declaration(tmp_path):
    import hashlib

    data_root = tmp_path / "external-layers"
    data_root.mkdir()
    layer = data_root / "anchor.json"
    layer.write_text("{}")
    world = toy.build(walkers=2, drivers=0)
    manifest = dict(
        schema="awr.life.life.environment.v1",
        id="toy",
        layers={"anchor": {"path": "anchor.json", "sha256": hashlib.sha256(layer.read_bytes()).hexdigest()}},
        capabilities=dict(walk=dict(residents=2), drive=dict(drivers=1, destinations=2)),
    )
    path = Path(tmp_path) / "toy.json"
    path.write_text(json.dumps(manifest))
    ctx = context(world, 1.0, tmp_path, dict(world.agendas))
    ctx["cfg"] = dict(ctx["cfg"], environment=str(path), environment_root=str(data_root), people=2)
    with pytest.raises(CapabilityRefused, match="declared 1 drivers, measured 0"):
        run(ctx)
    manifest["capabilities"]["drive"] = dict(drivers=0)
    path.write_text(json.dumps(manifest))
    ctx = context(world, 1.0, tmp_path, dict(world.agendas))
    ctx["cfg"] = dict(ctx["cfg"], environment=str(path), environment_root=str(data_root), people=2)
    run(ctx)
    assert ctx["life_runtime_profile"]["capabilities"] == dict(
        environment="toy", residents=2, walk_kinds_min=1, vehicles=0
    )


def test_recording_mixed_population_has_real_vehicle_timeline_and_containment(tmp_path, monkeypatch):
    # This synthetic fixture declares its off-road berths legal. The engine
    # must preserve missing/false declarations instead of assuming this.
    w = toy.build(walkers=3, drivers=2, parking_legality=True)
    # As the scene configurations do, a destination nearer than an effective car trip (the audit's 50 m) is walked to.
    w.director.min_drive_m = 50.0
    ctx = context(w, 420.0, tmp_path, dict(w.agendas))
    ctx["driving_binding"] = DrivingBinding(w.routes)
    ctx["cfg"].update(
        early_motion_gate=True,
        life_gates={"person_longest_stall_s_max": 60.0, "vehicle_longest_stop_in_trip_s_max": 60.0},
    )
    run(ctx)
    assert ctx["life_runtime_profile"]["mixed_driving"]
    places = [dict(id=p.id, kind=p.kind, xy=list(p.xy), parking=list(p.parking), group=p.group) for p in w.places]
    evidence = TravelEvidence(w.director.events, places, 420.0)
    for i, vid in w.engine.vehicle_ids.items():
        owner = w.director.vehicles[vid].owner
        person = w.ids.index(owner)
        contained = ctx["state"][:, person] == 3
        assert contained.any() and (ctx["state"][:, person] == 1).any()
        assert np.all(ctx["container_track"][contained, person] == i)
        np.testing.assert_array_equal(ctx["tracks"][contained, person], ctx["tracks"][contained, i])
        timeline = [e for e in ctx["events"] if e["kind"] in ("depart", "park") and e["actors"] == [vid]]
        assert [e["kind"] for e in timeline] == ["depart", "park"] * (len(timeline) // 2) + (
            ["depart"] if len(timeline) % 2 else []
        )
        assert len(timeline) >= 4
        matches = []
        for dep, park in zip(timeline[::2], timeline[1::2], strict=False):
            k = round(dep["t_s"] * 10)
            assert np.linalg.norm(ctx["vel"][k, i]) > 0.0
            assert dep["evidence"] == park["evidence"] and dep["intent_id"] == park["intent_id"]
            trip = dict(vehicle=vid, t0_s=dep["t_s"], t1_s=park["t_s"], end_xy_m=park["xy"])
            matches.append(evidence.match(trip))
        assert any(matches), (vid, timeline)
    # Run the existing independent tape audit on quantized recorded arrays,
    # not on engine counters or nominal routes.
    import awr.life.civil.life_audit as audit_module

    dtype = [(name, "i4") for name in ("x_mm", "y_mm", "vx_cmps", "vy_cmps", "state")]
    frames = np.zeros(ctx["state"].shape, dtype=dtype)
    for j, name in enumerate(("x_mm", "y_mm")):
        frames[name] = np.rint(ctx["tracks"][:, :, j] * 1000.0)
    for j, name in enumerate(("vx_cmps", "vy_cmps")):
        frames[name] = np.rint(ctx["vel"][:, :, j] * 100.0)
    frames["state"] = ctx["state"]
    tape = SimpleNamespace(actors=[dict(id=aid, **{"class": kind}) for aid, kind in zip(w.ids, w.kinds, strict=False)])
    monkeypatch.setattr(audit_module, "_frames", lambda event: (tape, frames))
    for name, rows in [("activity-events.jsonl", ctx["events"]), ("activity-decisions.jsonl", w.director.events)]:
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "activity-places.json").write_text(json.dumps(places))
    (tmp_path / "agendas.json").write_text(json.dumps([ag.summary() for ag in w.agendas.values()]))
    report = audit_module.audit(
        tmp_path, {"life_gates": {"vehicle_recurring_vehicles_min": {"threshold": 2, "require_activity_chain": True}}}
    )
    assert report["vehicle"]["recurring_fleet"]["qualified_vehicles"] == 2
    assert report["vehicle"]["trips"] >= 4
    assert not report["vehicle"]["trips_ineffective"]


def test_binding_keeps_exact_dimensions_behind_quantized_actor_metadata(tmp_path):
    from awr.life.civil.activity.driving import Body

    w = toy.build(drivers=1)
    ctx = context(w, 1.0, tmp_path, dict(w.agendas))
    size = tuple(np.asarray(toy.CAR) + 0.0001)
    w.routes.size = np.array(size)
    w.routes.update_occupancy({aid: Body(body.pose, size) for aid, body in w.routes.occupancy.items()})
    binding = DrivingBinding(w.routes).prepare(ctx)
    assert binding["sizes"][1] == size


def test_mixed_runtime_refuses_insufficient_driver_capacity_before_recording(tmp_path):
    w = toy.build(drivers=1)
    ctx = context(w, 1.0, tmp_path, dict(w.agendas))
    ctx["driving_binding"] = DrivingBinding(w.routes)
    ctx["cfg"]["life_gates"] = {"vehicle_recurring_vehicles_min": {"threshold": 30, "min_distinct_destinations": 2}}
    with pytest.raises(CapabilityRefused, match="30 required"):
        run(ctx)
    assert not ctx["tracks"].any() and not ctx["events"] and not w.routes.active


def test_recording_stops_when_an_executing_vehicle_is_blocked_too_long(tmp_path, monkeypatch):
    import awr.life.motion.engine as engine_module

    original = engine_module.stop_arc

    def barrier(*args, **kwargs):
        # An external obstruction five metres into the trip, observed by the
        # normal speed controller. No body/route or arrival is rewritten.
        return min(5.0, original(*args, **kwargs))

    monkeypatch.setattr(engine_module, "stop_arc", barrier)
    w = toy.build(drivers=1)
    ctx = context(w, 120.0, tmp_path, dict(w.agendas))
    ctx["driving_binding"] = DrivingBinding(w.routes)
    ctx["cfg"]["life_gates"] = {"vehicle_longest_stop_in_trip_s_max": 2.0}
    with pytest.raises(ValueError, match="vehicle_longest_stop_in_trip_s_max"):
        run(ctx)
    failure = json.loads((tmp_path / "tape.life-failure.json").read_text())
    assert failure["failures"][0]["actor"] == "vehicle-0"
    assert any(e["kind"] == "depart" for e in ctx["events"])
    assert not any(e["kind"] == "park" for e in ctx["events"])
    assert not w.routes.completed_motion
    traffic = json.loads((tmp_path / failure["traffic_evidence"]).read_text())
    assert "vehicle-0" in traffic["trips"]
    assert traffic["courses"][traffic["trips"]["vehicle-0"]["evidence"]]["cached"]
    assert not traffic["admitted"]


@pytest.mark.parametrize(
    "fault, message",
    [
        ("no_binding", "explicit DrivingBinding"),
        ("car_pose", "differs from recorded body"),
        ("car_size", "differs from recorded body"),
        ("resident_pose", "actual home"),
        ("wrong_surface", "support surface"),
        ("omit_rigid", "all residents and rigid bodies"),
    ],
)
def test_recording_rejects_mismatched_mixed_population_before_any_frame(tmp_path, fault, message):
    w = toy.build(walkers=1, drivers=1)
    ctx = context(w, 1.0, tmp_path, dict(w.agendas))
    ctx["driving_binding"] = DrivingBinding(w.routes)
    car = next(iter(w.engine.vehicle_ids))
    if fault == "no_binding":
        del ctx["driving_binding"]
    elif fault == "car_pose":
        ctx["xy"][car, 0] += 1.0
    elif fault == "car_size":
        ctx["actors"][car]["size_mm"][0] += 500
    elif fault == "resident_pose":
        ctx["xy"][0, 0] += 1.0
    elif fault == "wrong_surface":
        ctx["surface"] = object()
    elif fault == "omit_rigid":
        w.routes.roster = frozenset()
    before = ctx["tracks"].copy()
    with pytest.raises(ValueError, match=message):
        run(ctx)
    np.testing.assert_array_equal(ctx["tracks"], before)
    assert not w.routes.active and not ctx["events"]


def test_runtime_saves_a_requested_scene_that_replays_the_run(tmp_path, monkeypatch):
    from awr.life.motion.scene import Scene

    world = toy.build(walkers=6, drivers=0)
    d = world.director
    agendas = {
        i: ReactiveAgenda(ag.actor, d.places[d.actors[ag.actor].home], d, 0.0, ag.visits[0].dwell_s)
        for i, ag in world.agendas.items()
    }
    ctx = context(world, 60.0, tmp_path, agendas)
    k = 300
    walker = next(iter(agendas))
    monkeypatch.setenv("AWR_LIFE_LIFE_SCENES", f"{k / 10:g}:{world.xy[walker][0]}:{world.xy[walker][1]}")
    run(ctx)
    sc = Scene(Path(tmp_path) / f"tape.scene-{k / 10:g}.npz")
    assert abs(sc.t - k / 10) < 1e-9
    rows = np.array([world.ids.index(s) for s in sc.ids])
    # the scene holds the state at the start of tick k: the pose recorded for tick k-1
    np.testing.assert_allclose(sc.xy, ctx["tracks"][k - 1, rows], atol=1e-9)
    np.testing.assert_allclose(np.degrees(sc.yaw), ctx["yaw_track"][k - 1, rows], atol=1e-9)
    # its walking network window answers as the run's router does there
    window = sc.router()
    for p in np.vstack([sc.xy, sc.xy + 0.6, sc.xy - 0.6]):
        assert window.on_snapshot(p) == world.router.on_snapshot(p)
