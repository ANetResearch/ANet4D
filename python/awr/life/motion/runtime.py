"""life_v1: advance an admitted grounded world with the life engine.

Called by awr.life.civil.grounded.build after admission when the config selects ``"runtime": "life_v1"``. Admission
(homes, facility slots, street objects, parked vehicles), the activity director, the tape assembly, the final QA,
the life audit and publishing are those of the builder. This module drives awr.life.motion.engine.LifeEngine (decisions,
routes, standoffs, congestion rerouting, crowd motion) tick by tick and records the result into the builder's
arrays, with the same early motion check and the same in-recording life gates.

The town adapter (awr.life.civil.activity.grounded) maps walking activities; with life.driving, resident drivers own
cars at home berths of a verified driving map (awr.life.civil.activity.driving_adapter), and the builder supplies an
explicit DrivingBinding (awr.life.motion.binding) that checks the backend and the recorded bodies are the same fresh
world. Legacy vehicle itineraries, delivery rounds and initial riders are refused explicitly. This recording bridge
does not manufacture missing town parking or access capacity.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from awr.life.civil import body as bodies
from awr.life.common.schema import schema_id

from .binding import DrivingBinding
from .capability import Demand, load_environment, measure, require
from .crowd import DT
from .engine import LifeEngine
from .route import LifeRouter
from .timing import SlowTicks


def search_profile(activity):
    """Cumulative route-search work behind the decisions: walking searches, LifeRouter builds, drive plans and the
    way-home checks (counts and wall seconds)."""
    rs = activity.route_stats
    out = dict(
        walk_n=rs.get("searches", 0),
        walk_s=round(rs.get("search_wall_s", 0.0), 1),
        router_builds=rs.get("life_router_builds", 0),
        router_build_s=round(rs.get("life_router_build_wall_s", 0.0), 1),
    )
    routes = getattr(activity, "driving_routes", None)
    if routes is not None:
        s = routes.stats
        out.update(
            drive_n=s.get("plans_computed", 0),
            drive_s=round(s.get("plan_wall_s", 0.0), 1),
            drive_searches=s.get("path_searches", 0),
            ways_s=round(s.get("keeps_ways_wall_s", 0.0), 1),
            tours=s.get("tours_computed", 0),
            traffic={k: round(v, 1) if isinstance(v, float) else v for k, v in routes.traffic.stats.items()},
        )
    return out


def run(ctx):
    cfg, jobs, actors = ctx["cfg"], ctx["jobs"], ctx["actors"]
    if any("delivery" in j for j in jobs):
        raise ValueError("life_v1 runtime: delivery rounds are not implemented; remove the delivery section")
    trips = [actors[i]["id"] for i, j in enumerate(jobs) if j["kind"] == "vehicle" and j.get("goal_s", 0.0) > 0.03]
    if trips:
        raise ValueError(
            f"life_v1 runtime: the town adapter maps no vehicle trips ({len(trips)} requested); car "
            "journeys need places with berths, access points and a driving backend"
        )
    if any(j.get("riding") for j in jobs):
        raise ValueError("life_v1 runtime: riders are not implemented in this stage")
    persons = [i for i, j in enumerate(jobs) if j["kind"] == "person"]
    if any(jobs[i].get("agenda") is None for i in persons):
        raise ValueError("life_v1 runtime: every person needs an activity agenda (life.behavior = activities_v1)")
    activity = ctx["activity_runtime"]
    if activity is None:
        raise ValueError("life_v1 runtime: the activity director is required")
    binding = ctx.get("driving_binding")
    if binding is not None and not isinstance(binding, DrivingBinding):
        raise ValueError("life_v1 runtime: explicit DrivingBinding required")
    if activity.director.vehicles and binding is None:
        raise ValueError("life_v1 runtime: resident cars require an explicit DrivingBinding")
    drive_args = {} if binding is None else binding.prepare(ctx)
    out, warm, nt = ctx["out"], float(ctx["warm"]), int(ctx["nt"])
    tracks, vel, yaw_track, state, container = (
        ctx["tracks"],
        ctx["vel"],
        ctx["yaw_track"],
        ctx["state"],
        ctx["container_track"],
    )
    events, occupancy, rest_r = ctx["events"], ctx["occupancy"], ctx["rest_r"]
    kinds = np.array([j["kind"] for j in jobs])
    import numba

    numba.set_num_threads(max(1, min(int(cfg.get("life_threads", 8)), numba.config.NUMBA_NUM_THREADS)))
    if not hasattr(activity.network, "finish_grid_path"):
        raise ValueError("life_v1 runtime: the activity director must plan on a WalkNetwork")
    walk_space = None
    if binding is not None and ctx.get("walk_radius") is not None:
        from .parked_space import ParkedSpace

        walk_space = ParkedSpace(activity, activity.network, float(ctx["walk_radius"]))
        # Capacity/access must be measured on the same parked geometry as execution.
        walk_space.update(
            {
                i: (
                    float(ctx["xy"][i, 0]),
                    float(ctx["xy"][i, 1]),
                    float(ctx["yaw"][i]),
                    *(float(c) for c in drive_args["sizes"][i]),
                )
                for i, vid in drive_args["vehicle_ids"].items()
                if vid not in drive_args["driving"].active
            }
        )
    if not isinstance(activity.static_router, LifeRouter):
        # The director's activity choices and the walking legs share one router on the static snapshot.
        activity.static_router = LifeRouter(activity.network, activity.static_free, stats=activity.route_stats)
    capabilities = None
    drive_capability = None
    if cfg.get("environment") or binding is not None:
        # The admitted roster, measured through the engine's own backends, must support the scenario and the
        # environment's declaration (awr.life.motion.capability).
        manifest = (
            load_environment(
                cfg["environment"],
                root=cfg.get("environment_root", "."),
                validation_only=ctx.get("dynamic_validation", False),
            )
            if cfg.get("environment")
            else None
        )
        environment_id = manifest["id"] if manifest else "admitted-population"
        began = time.perf_counter()
        print("measuring activity access", environment_id, flush=True)
        report = measure(activity.director, activity.static_router.components_near)
        (out.parent / (out.name + ".capabilities.json")).write_text(
            json.dumps(
                dict(
                    schema="awr.life.measured_activity_access.v1",
                    admitted=False,
                    environment=environment_id,
                    scope="individual static opportunities; not simultaneous execution or publication admission",
                    wall_s=time.perf_counter() - began,
                    report=report,
                ),
                indent=2,
            )
            + "\n"
        )
        print("activity access measured", round(time.perf_counter() - began, 2), "s", flush=True)
        report = require(
            report,
            Demand.from_config(cfg),
            environment_id,
            declared=manifest["capabilities"]
            if manifest and schema_id(manifest["schema"]) == "awr.life.life.environment.v1"
            else None,
            kind_minimum=getattr(activity, "kind_minimum", None),
        )
        kinds_of = list(report["walk"]["kinds"].values())
        capabilities = dict(
            environment=environment_id,
            residents=len(kinds_of),
            walk_kinds_min=min(kinds_of, default=0),
            vehicles=report["drive"]["vehicles"],
        )
        # Per owned car: facilities it can reach by car and come back from, or why it cannot (diagnostics).
        drive_capability = dict(
            destinations={r["vehicle"]: len(r["destinations"]) for r in report["drive"]["rows"]},
            refusals={r["vehicle"]: r["reason"] for r in report["drive"]["rows"] if r["reason"]},
        )
    vmax = np.array([float(j.get("max_speed", 1.2)) if j["kind"] == "person" else 0.0 for j in jobs])
    walk_space = None
    if binding is not None and ctx.get("walk_radius") is not None:
        from .parked_space import ParkedSpace

        walk_space = ParkedSpace(activity, activity.network, float(ctx["walk_radius"]))
    # Standing room at a place (GroundedActivities.available): people are disks, vehicles and objects their boxes.
    rest_boxes = np.where(
        (np.asarray(kinds) == "person")[:, None], np.nan, bodies.symmetric(np.asarray(ctx["ext"])[:, 0])
    )
    engine = LifeEngine(
        ids=[a["id"] for a in actors],
        kinds=kinds,
        ext=ctx["ext"],
        xy=ctx["xy"],
        yaw=ctx["yaw"],
        surface=ctx["surface"],
        walking=ctx["walking"],
        router=activity.static_router,
        agendas={i: jobs[i]["agenda"] for i in persons},
        vmax=vmax,
        occupancy=occupancy,
        events=events,
        route_cache=activity.route_cache,
        walk_space=walk_space,
        **drive_args,
    )
    gates = cfg.get("life_gates") or {}
    stall_gate = gates.get("person_longest_stall_s_max")
    stall_limit = (
        None if stall_gate is None else float(stall_gate["threshold"] if isinstance(stall_gate, dict) else stall_gate)
    )

    def gate_limit(name, key="threshold"):
        g = gates.get(name)
        if g is None:
            return None
        return (
            float(g[key])
            if isinstance(g, dict) and key in g
            else (float(g) if not isinstance(g, dict) and key == "threshold" else None)
        )

    # A car standing inside an executing trip longer than the trip-stop gates allows fails the run when it happens.
    stop_limits = [
        v
        for v in (
            gate_limit("vehicle_longest_stop_in_trip_s_max"),
            gate_limit("vehicle_ineffective_trips_max", "max_stop_s"),
        )
        if v is not None
    ]
    stop_limit = min(stop_limits) if stop_limits else None
    stop_run = np.zeros(engine.n, int)
    wall = {}
    tick_wall = []
    window = {}
    slow_ticks = SlowTicks()

    def walker(i):
        """Diagnostic state of person i for a failure report: agenda phase, place and target, path, standoff,
        courtesy and rerouting state, and the last agenda events."""
        ag = jobs[i].get("agenda")
        if ag is None:
            return {}
        try:
            return _walker(i, ag)
        except Exception as e:  # a diagnostic must never hide the failure being reported
            return dict(diagnostic_error=repr(e))

    def _walker(i, ag):
        row = dict(
            phase=ag.phase,
            place=getattr(getattr(ag, "destination", None), "place", None),
            target=[round(float(c), 2) for c in engine.target(i)],
            paused=getattr(ag, "paused_at", None) is not None,
            yielding=i in engine.standoffs.yielding,
            aside=(engine.courtesy.aside.get(i) or {}).get("stage"),
            next_reroute_s=engine.reroutes.next.get(i),
            events=ag.events[-6:],
            refuge_refusals=engine.courtesy.refusals.get(i),
        )
        if i in engine.paths:
            row["path"] = dict(
                remaining_m=round(float(engine.paths.remaining(i)), 2),
                end=[round(float(c), 2) for c in engine.paths.end(i)],
            )
        elif ag.phase == "travelling" and hasattr(engine.router, "probe"):
            row["route_probe"] = engine.router.probe(
                np.asarray(engine.xy[i], float), np.asarray(engine.target(i), float)
            )
        return row

    def snapshot():
        """The state at the start of a tick (arrays the step replaces are kept, those it changes in place copied;
        path points stay in the store until it is compacted, which allocates new arrays)."""
        c, p = engine.crowd, engine.paths
        return dict(
            xy=engine.xy,
            yaw=engine.yaw,
            speed=engine.speed,
            active=engine.active.copy(),
            waited=c.waited.copy(),
            motion=c.motion.copy(),
            aim=c.aim.copy(),
            pts=p.pts,
            first=p.first.copy(),
            count=p.count.copy(),
            s=p.s.copy(),
            k=p.k.copy(),
        )

    def scene(centres, state, half=20.0):
        """Local scene for offline replay (awr.life.motion.scene, scripts/engine_eval/replay_scene.py) at the start of a
        tick (`state`, from snapshot()): support and walking masks within `half` metres of the given centres, every
        body there, the paths and waiting state of its walkers, and the static walking network there (the planning
        snapshot's cells and costs, for offline route queries; its exact static bodies are not included)."""
        c = engine.crowd
        sxy = state["xy"]
        lo, hi = np.min(centres, 0) - half, np.max(centres, 0) + half

        def window(x0, y0, step, shape):
            c0, r0 = max(int((lo[0] - x0) // step), 0), max(int((lo[1] - y0) // step), 0)
            c1, r1 = min(int((hi[0] - x0) // step) + 1, shape[1]), min(int((hi[1] - y0) // step) + 1, shape[0])
            return (slice(r0, r1), slice(c0, c1)), (x0 + c0 * step, y0 + r0 * step)

        sw, sorig = window(c.sx0, c.sy0, c.scell, c.walk.shape)
        fw, forig = window(c.fx0, c.fy0, c.fstep, c.free.shape)
        network = engine.router.network
        # network cells are centred on origin + index*step
        nw, norig = window(
            network.origin[0] - network.step / 2,
            network.origin[1] - network.step / 2,
            network.step,
            engine.router.free.shape,
        )
        weight = network.weight[nw] if network.weight is not None else np.ones(engine.router.free[nw].shape, np.float32)
        inside = np.flatnonzero(((sxy >= lo) & (sxy <= hi)).all(1))
        pts, first, count, prog = [], [], [], []
        for i in inside:
            f, m = int(state["first"][i]), int(state["count"][i])
            first.append(sum(len(q) for q in pts))
            count.append(m if f >= 0 else 0)
            prog.append((float(state["s"][i]), int(state["k"][i])) if f >= 0 else (0.0, 0))
            if f >= 0:
                pts.append(state["pts"][f : f + m].copy())
        return dict(
            ids=np.array([actors[i]["id"] for i in inside]),
            index=inside,
            kinds=kinds[inside],
            ext=c.ext[inside],
            xy=sxy[inside],
            yaw=state["yaw"][inside],
            speed=state["speed"][inside],
            active=state["active"][inside],
            vmax=vmax[inside],
            waited=state["waited"][inside],
            motion=state["motion"][inside],
            aim=state["aim"][inside],
            path_points=np.concatenate(pts) if pts else np.zeros((0, 2)),
            path_first=np.array(first),
            path_count=np.array(count),
            path_progress=np.array(prog),
            walk=c.walk[sw],
            z_ok=c.z_ok[sw],
            support_origin=np.array(sorig),
            support_cell=c.scell,
            free=c.free[fw],
            free_origin=np.array(forig),
            free_step=c.fstep,
            net_free=np.asarray(engine.router.free[nw], bool),
            net_weight=weight,
            net_origin=np.array(norig) + network.step / 2,
            net_step=network.step,
            net_shortcut_tolerance=network.shortcut_tolerance,
        )

    # Diagnostic scenes at given times and places (AWR_LIFE_LIFE_SCENES="t:x:y;..."), saved at the start of the first
    # tick at or after t as <out>.scene-<t>.npz; they do not change the run.
    dumps = sorted(
        (float(a), float(b), float(c_))
        for a, b, c_ in (q.split(":") for q in os.environ.get("AWR_LIFE_LIFE_SCENES", "").split(";") if q.strip())
    )

    def fail(k, t, kind, rows, gate=None, limit=None):
        tw = np.asarray(tick_wall) if tick_wall else np.zeros(1)
        d = dict(
            tick=k,
            time_s=round(t, 1),
            kind=kind,
            gate=gate,
            threshold=limit,
            failures=rows,
            partial=out.name + ".partial.npz",
            crowd=engine.crowd.stats,
            courtesy=engine.courtesy.stats,
            standoffs=engine.standoffs.stats,
            reroutes=engine.reroutes.stats,
            tick_ms={q: round(1e3 * float(np.percentile(tw, p)), 2) for q, p in (("p50", 50), ("p95", 95))}
            | {"max": round(1e3 * float(tw.max()), 2)},
            stages_s={q: round(v, 2) for q, v in wall.items()},
            note="Recorded ticks 0..tick; float32 positions (m), velocities (m/s), yaw (deg), state.",
        )
        k1 = max(k, 0) + 1
        np.savez_compressed(
            out.parent / (out.name + ".partial.npz"),
            xy=tracks[:k1].astype(np.float32),
            v=vel[:k1].astype(np.float32),
            yaw=yaw_track[:k1].astype(np.float32),
            state=state[:k1],
            ids=np.array([a["id"] for a in actors]),
        )
        (out.parent / (out.name + ".partial-events.jsonl")).write_text("".join(json.dumps(e) + "\n" for e in events))
        try:
            centres = np.array([r["position"] for r in rows], float)
            np.savez_compressed(out.parent / (out.name + ".scene.npz"), **scene(centres, pre, half=20.0) | dict(t=t))
        except Exception as e:  # the scene is a diagnostic: never hide the failure being reported
            d["scene_error"] = repr(e)
        from .failure_evidence import record_activity, save_crowd

        d["activity_events"] = record_activity(out, engine)
        if engine.last_motion is not None:
            # The last crowd step, for an exact offline replay of the walkers' choices (failure_evidence.replay_crowd).
            try:
                save_crowd(out.parent / (out.name + ".life-crowd.npz"), engine.crowd, *engine.last_motion)
                d["crowd_replay"] = out.name + ".life-crowd.npz"
            except Exception as e:  # a diagnostic: never hide the failure being reported
                d["crowd_replay_error"] = repr(e)
        from .traffic_evidence import record_traffic

        d["traffic_evidence"] = record_traffic(out, engine)
        name = "motion-failure" if kind == "motion" else "life-failure"
        (out.parent / (out.name + f".{name}.json")).write_text(json.dumps(d, indent=2, default=float) + "\n")
        raise ValueError(f"life_v1 {kind} failure at {t:.1f} s: " + json.dumps(d, default=float)[:3000])

    def neighbours(i):
        d = np.linalg.norm(engine.xy - engine.xy[i], axis=1)
        return [
            dict(
                actor=actors[j]["id"],
                distance_m=round(float(d[j]), 2),
                speed_mps=round(float(speed[j]), 2),
                kind=jobs[j]["kind"],
                position=[round(float(c), 2) for c in engine.xy[j]],
                **walker(j),
            )
            for j in np.argsort(d)
            if j != i and d[j] < 6.0
        ][:8]

    prev_v = None
    prev_state = None
    began = time.perf_counter()
    for k in range(-round(warm / DT), nt):
        t = k * DT
        t0 = time.perf_counter()
        first_event = len(activity.director.events)
        xy, speed = engine.xy, engine.speed
        pre = snapshot()
        before = dict(wall)
        while dumps and t >= dumps[0][0] - 1e-9:
            dt_, dx, dy = dumps.pop(0)
            np.savez_compressed(
                out.parent / (out.name + f".scene-{dt_:g}.npz"),
                **scene(np.array([[dx, dy]]), pre, half=25.0) | dict(t=t),
            )
        activity.update_frame(xy, rest_r, engine.riding, engine.yaw, rest_boxes)
        try:
            new_xy, new_yaw, new_speed, v_now, st, cont, held = engine.step(t, wall)
        except Exception as error:
            from .failure_evidence import record

            try:
                record(out, engine, error, k, t, ctx)
            except Exception as diagnostic_error:
                error.add_note(f"Failure evidence could not be fully saved: {diagnostic_error!r}")
            raise
        e = time.perf_counter()
        if k >= 0:
            tracks[k] = new_xy
            vel[k] = v_now
            yaw_track[k] = np.degrees(new_yaw)
            state[k] = st
            container[k] = cont
        f = time.perf_counter()
        wall["record"] = wall.get("record", 0.0) + f - e
        if prev_v is not None and cfg.get("early_motion_gate", False):
            acc = np.linalg.norm((np.round(v_now * 100) - np.round(prev_v * 100)) * 0.01, axis=1) / DT
            # Only bodies visible (moving or standing) in both ticks have a motion; boarding is containment.
            both_visible = np.isin(st, (1, 2)) & np.isin(prev_state, (1, 2))
            bad = np.flatnonzero(both_visible & (acc > np.where(kinds == "vehicle", 6.0, 8.0) + 1e-7))
            if len(bad):
                fail(
                    k,
                    t,
                    "motion",
                    [
                        dict(
                            actor=actors[i]["id"],
                            acceleration_mps2=float(acc[i]),
                            position=xy[i].tolist(),
                            v_prev=np.round(prev_v[i], 3).tolist(),
                            v_now=np.round(v_now[i], 3).tolist(),
                            held=int(i) in held,
                            nearby=neighbours(i),
                        )
                        for i in bad
                    ],
                )
        prev_v = v_now
        prev_state = st
        stall_run = engine.stall_run
        if k >= 0 and stall_limit is not None and (stall_run * DT > stall_limit).any():
            fail(
                k,
                t,
                "life",
                [
                    dict(
                        actor=actors[i]["id"],
                        stalled_s=round(float(stall_run[i] * DT), 1),
                        position=xy[i].tolist(),
                        agenda_phase=jobs[i]["agenda"].phase,
                        yaw_deg=round(float(np.degrees(engine.yaw[i])), 1),
                        **walker(i),
                        nearby=neighbours(i),
                    )
                    for i in np.flatnonzero(stall_run * DT > stall_limit)
                ],
                "person_longest_stall_s_max",
                stall_limit,
            )
        in_trip = np.zeros(engine.n, bool)
        if engine.driving is not None:
            for i, vid in engine.vehicle_ids.items():
                execution = engine.driving.executions.get(vid)
                in_trip[i] = vid in engine.driving.active and execution is not None and not execution.arrived
        stop_run = np.where(in_trip & (np.linalg.norm(v_now, axis=1) < 0.2), stop_run + 1, 0)
        if k >= 0 and stop_limit is not None and (stop_run * DT > stop_limit + 1e-7).any():
            traffic = getattr(engine.driving, "traffic", None)
            fail(
                k,
                t,
                "life",
                [
                    dict(
                        actor=actors[i]["id"],
                        stopped_s=round(float(stop_run[i] * DT), 1),
                        position=engine.xy[i].tolist(),
                        nearby=neighbours(i),
                        traffic=None if traffic is None else traffic.describe(actors[i]["id"]),
                        traffic_others=None
                        if traffic is None
                        else {v: traffic.describe(v) for v in sorted(traffic.trips) if v != actors[i]["id"]},
                    )
                    for i in np.flatnonzero(stop_run * DT > stop_limit + 1e-7)
                ],
                "vehicle_longest_stop_in_trip_s_max",
                stop_limit,
            )
        wall["gates"] = wall.get("gates", 0.0) + time.perf_counter() - f
        for i in persons:
            jobs[i]["speed"] = float(new_speed[i])
        elapsed = time.perf_counter() - t0
        tick_wall.append(elapsed)
        for q, v in wall.items():
            window.setdefault(q, []).append(v - before.get(q, 0.0))
        slow_ticks.add(k, t, elapsed, before, wall, activity.director.events, first_event)
        if k % 500 == 0:
            tw = np.asarray(tick_wall[-500:])
            # per stage over the last 500 ticks: mean / p95 (ms)
            stages = {q: f"{1e3 * np.mean(v):.1f}/{1e3 * np.percentile(v, 95):.1f}" for q, v in window.items()}
            window.clear()
            print(
                f"tick {k} moving {int((np.linalg.norm(v_now, axis=1) > 0.1).sum())} paths {len(engine.paths)} "
                f"tick_ms p50 {1e3 * np.median(tw):.1f} p95 {1e3 * np.percentile(tw, 95):.1f} max {1e3 * tw.max():.1f} "
                f"stages_ms {stages} crowd {engine.crowd.stats} standoffs {engine.standoffs.stats} "
                f"reroutes {engine.reroutes.stats} courtesy {engine.courtesy.stats} "
                f"searches {search_profile(activity)}",
                flush=True,
            )
    tw = np.asarray(tick_wall)
    ctx["life_runtime_profile"] = dict(
        schema="awr.life.life_runtime_profile.v1",
        runtime="life_v1",
        ticks=len(tw),
        wall_s=round(time.perf_counter() - began, 2),
        tick_ms={q: round(1e3 * float(np.percentile(tw, p)), 2) for q, p in (("p50", 50), ("p95", 95), ("p99", 99))}
        | {"max": round(1e3 * float(tw.max()), 2)},
        stages_s={k: round(v, 2) for k, v in wall.items()},
        crowd=engine.crowd.stats,
        standoffs=engine.standoffs.stats,
        reroutes=engine.reroutes.stats,
        courtesy=engine.courtesy.stats,
        routes=engine.stats,
        capabilities=capabilities,
        static_routes=activity.route_stats,
        decision_wall_s=round(activity.director.decision_wall_s, 2),
        car_journey_rejections=dict(sorted(activity.director.car_rejections.items())),
        drive_capability=drive_capability,
        car_choice=dict(activity.director.car_choice),
        mixed_driving=binding is not None,
        parked_walk_space=None
        if walk_space is None
        else dict(walk_space.stats, future_planning=dict(walk_space.future_stats)),
        driving=None
        if binding is None
        else dict(
            binding.routes.stats,
            plans=len(binding.routes.plans),
            completed=len(binding.routes.completed_motion),
            traffic=dict(binding.routes.traffic.stats),
        ),
        slowest_ticks=slow_ticks.report(),
        note="Stage times are disjoint parts of the tick loop; decisions include the director and its static route queries.",
    )
    return engine.xy, engine.yaw
