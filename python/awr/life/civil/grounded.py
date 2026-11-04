"""Strict, persistent civil actors on measured support surfaces.

Explicit activities: walk to a waypoint, wait, return; drive a supported lane
segment, brake, remain stopped. No spontaneous disappearance or path fallback.
Only verified mesh types are admitted. Unknown building interiors, unverified
crossings and manoeuvres are rejected rather than represented as teleportation.
"""

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from awr.life.appearance.assets import load
from awr.life.civil import body as bodies
from awr.life.civil.collision import collision_pairs, rectangle
from awr.life.civil.local_navigation import distance_to_polyline, rolling_route, step_aside_route, yield_route
from awr.life.civil.surface import SurfaceMap
from awr.life.civil.walking import PLANNING_MARGIN_M, polyline
from awr.life.common.schema import schema_id

DT = 0.1


def line_samples(line, spacing=0.35):
    ds = np.arange(0, line.length, spacing)
    if len(ds) < 2:
        return None
    import shapely

    pp = shapely.get_coordinates(shapely.line_interpolate_point(line, ds))
    yaw = np.arctan2(np.gradient(pp[:, 1]), np.gradient(pp[:, 0]))
    return pp, yaw


def footprint_support(surface, pp, c, s, offsets, profile, *, chunk_poses=4096):
    """Batch the identical rotated probes; bound temporary memory on long paths.

    This changes query grouping only. Every original corner/contact/centre is
    retained, with the same floating-point expression and per-pose AND rule.
    SurfaceMap.supported is a pointwise predicate, not a batch-level gate.
    """
    valid = np.empty(len(pp), bool)
    dx, dy = offsets[:, 0], offsets[:, 1]
    for begin in range(0, len(pp), chunk_poses):
        end = min(begin + chunk_poses, len(pp))
        cc, ss = c[begin:end, None], s[begin:end, None]
        q = pp[begin:end, None, :] + np.stack([dx * cc - dy * ss, dx * ss + dy * cc], axis=-1)
        valid[begin:end] = surface.supported(q.reshape(-1, 2), profile).reshape(end - begin, len(offsets)).all(1)
    return valid


def probe_offsets(size, profile):
    """Body-frame points whose support support_mask requires: the corners of the padded body and its centre, and for a
    vehicle also its tyre contacts and the corners of the body itself."""
    # The vehicle expansion equals the per-tick lookahead (+.24 m); admitting a
    # route with a smaller margin produced vehicles that never moved.
    offset = (
        np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]])
        * (np.array(size[:2]) + (0.36 if profile == "person" else 0.24))
        / 2
    )
    if profile == "vehicle":
        from awr.life.civil.contacts import vehicle_contacts

        offset = np.vstack(
            [
                offset,
                vehicle_contacts(tuple(size)),
                np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.array(size[:2]) / 2,
            ]
        )
    return offset


def support_mask(surface, pp, yaw, size, profile):
    if profile == "vehicle" and hasattr(surface, "vehicle_support_mask"):
        # Explicit candidate backend only. Ordinary SurfaceMap construction,
        # live motion and tape QA retain their existing production contract.
        valid = surface.vehicle_support_mask(pp, yaw, size)
        curvature = np.abs(np.diff(np.unwrap(yaw), prepend=yaw[0])) / 0.35
        return valid & (curvature < min(0.26, math.tan(math.radians(35)) / (float(size[0]) * 0.65)))
    c, s = np.cos(yaw), np.sin(yaw)
    offset = probe_offsets(size, profile)
    valid = footprint_support(surface, pp, c, s, offset, profile)
    if profile == "vehicle" and len(pp) > 1:
        # The lookahead samples the route every .025 m; a .5 m support cell can
        # hide between .35 m admission samples. Validate the same footprint on
        # the interpolated poses between consecutive samples as well.
        u = np.linspace(0, 1, 8, endpoint=False)[1:]
        dyaw = (np.diff(np.unwrap(yaw)))[:, None]
        mid_yaw = (yaw[:-1, None] + u[None, :] * dyaw).ravel()
        mid = (pp[:-1, None, :] + u[None, :, None] * np.diff(pp, axis=0)[:, None, :]).reshape(-1, 2)
        cm, sm = np.cos(mid_yaw), np.sin(mid_yaw)
        ok = footprint_support(surface, mid, cm, sm, offset, profile)
        seg = ok.reshape(len(pp) - 1, len(u)).all(1)
        valid[:-1] &= seg
        valid[1:] &= seg
    curvature = np.abs(np.diff(np.unwrap(yaw), prepend=yaw[0])) / 0.35
    if profile == "vehicle":
        # Conservative wheelbase proxy, 35 degree steering limit; a fixed .13
        # curvature rejected normal small-car turns as if every vehicle were a bus.
        valid &= curvature < min(0.26, math.tan(math.radians(35)) / (float(size[0]) * 0.65))
    return valid


def segments(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return list(zip(edges[::2], edges[1::2], strict=False))


def routes(surface, sem, profile, size):
    result = []
    for road in sem["roads"]:
        center = LineString(road["line"])
        for side in (-1, 1):
            offset = (road["width_m"] / 4 if profile == "vehicle" else road["width_m"] / 2 + 1.1) * side
            geom = center.offset_curve(offset)
            if geom.geom_type != "LineString":
                continue
            if side > 0:
                geom = LineString(list(geom.coords)[::-1])
            data = line_samples(geom)
            if data is None:
                continue
            pp, yaw = data
            valid = support_mask(surface, pp, yaw, size, profile)
            for a, b in segments(valid):
                if b - a >= (80 if profile == "vehicle" else 35):
                    result.append(
                        {
                            "road": road.get("id", "road"),
                            "points": pp[a:b],
                            "yaw": yaw[a:b],
                            "dense_validated": profile == "vehicle",
                        }
                    )
    return sorted(result, key=lambda r: -len(r["points"]))


def at(route, distance):
    if "arc" in route:
        arc = route["arc"]
        pp = route["points"]
        d = float(np.clip(distance, 0, arc[-1]))
        if arc[-1] == 0:
            if np.any(np.linalg.norm(pp - pp[0], axis=1) > 1e-10):
                raise ValueError("A zero-length hold cannot change world position")
            return pp[0].copy(), float(route.get("yaw", [0.0])[0])
        k = min(int(np.searchsorted(arc, d, side="right")) - 1, len(pp) - 2)
        k = max(0, k)
        u = (d - arc[k]) / (arc[k + 1] - arc[k])
        delta = pp[k + 1] - pp[k]
        if route.get("smooth_vehicle_heading"):
            return pp[k] + u * delta, float(np.interp(d, arc, route["yaw"]))
        return pp[k] + u * delta, math.atan2(delta[1], delta[0])
    pp = route["points"]
    q = np.clip(distance / 0.35, 0, len(pp) - 1)
    k = min(int(q), len(pp) - 2)
    u = q - k
    p = pp[k] * (1 - u) + pp[k + 1] * u
    delta = pp[k + 1] - pp[k]
    angle = math.atan2(delta[1], delta[0])
    if "yaw" in route:
        a, b = route["yaw"][k : k + 2]
        angle = float(a + u * ((b - a + math.pi) % (2 * math.pi) - math.pi))
    return p, angle


def route_length(route):
    return float(route["arc"][-1]) if "arc" in route else (len(route["points"]) - 1) * 0.35


def permanent_parking(job):
    """A current stop is not permanent geometry when an activity owns future motion."""
    return (
        job["kind"] == "vehicle"
        and job.get("motion_policy") != "autonomous_activity"
        and job.get("goal_s", 0.0) <= 0.03
        and job.get("after") is None
        and job["speed"] == 0.0
    )


def resident_parking_evidence(graph, berth, xy, yaw, info):
    """Validate an activity-owned stop using its mapped berth and scene rules."""
    node = graph.nodes[berth]
    pose = node["pose"]
    if (
        node["kind"] != "berth"
        or np.linalg.norm(np.asarray(xy) - pose[:2]) > 0.1
        or abs(math.remainder(yaw - pose[2], 2 * math.pi)) > 0.05
        or info.get("legal") is not True
    ):
        raise ValueError("Unadmitted resident parking pose or parking rule result")
    return dict(info, slot=berth, parking_source="activity_driving_map")


def crossing_areas(sem):
    """The pedestrian crossings of the semantics as areas (their two ends, the mapped width)."""
    from shapely.geometry import LineString

    return [
        LineString(c["ends"]).buffer(float(c.get("width_m", 4.0)) / 2, cap_style="flat")
        for c in sem.get("crossings", [])
        if len(c.get("ends") or ()) == 2 and c["ends"][0] != c["ends"][1]
    ]


def choose_cruisers(homes, facilities, count, min_destinations, part_of):
    """Cruising owners and, for every owner, the facility berths other than the home berth in `part_of(owner)`: the
    strongly connected part of the driving map that holds the home berth while the other cars stand parked (the
    destinations a tour can reach and return from). `homes`: {owner: home berth}; the owners with the most
    destinations are chosen, ties by ID."""
    reach = {}
    for a, b in sorted(homes.items()):
        part = set(part_of(a))
        reach[a] = sum(1 for f in facilities if f != b and f in part)
    ranked = sorted((a for a in homes if reach[a] >= min_destinations), key=lambda a: (-reach[a], a))
    return sorted(ranked[:count]), reach


def admit_drivers(
    dcfg,
    runtime,
    surface,
    sem,
    walking,
    static_walk_free,
    turn_radius,
    walk_radius,
    passage_check,
    actors,
    jobs,
    positions,
    yaws,
    sizes,
    add_vehicle,
    permanent_walk_obstacles,
    t_start,
    life_meta,
    parking_check=None,
    overlay=None,
    driving_map=None,
):
    """Owned cars at home berths and the driving backend for the life runtime (no other body is moved)."""
    from awr.life.civil.activity.driving import Body, DrivingRoutes
    from awr.life.civil.activity.driving_adapter import DrivingAdapter
    from awr.life.civil.curb_allocation import footprint
    from awr.life.civil.driving_graph import load_candidate
    from awr.life.civil.road_permission import driving_area
    from awr.life.motion.binding import DrivingBinding

    if runtime is None or "drive" not in runtime.modes or not passage_check:
        raise ValueError(
            "life.driving needs activities_v1 with walk and drive modes and the curb_v1 parked-car admission"
        )
    path = Path(dcfg["map"])
    if driving_map is not None:
        doc, graph = driving_map
    else:
        doc = json.loads(path.read_text())
        graph = load_candidate(doc)
    key = doc["asset"]
    spec, _, _ = load(key)
    size = np.array(spec["dimensions_m"])
    if not np.allclose(size, doc["size_m"], rtol=0.0, atol=1e-8):
        raise ValueError("Driving map body differs from its asset")
    # The permitted area the map was compiled with (its recorded lane network included).
    permission = doc.get("permission") or {}
    lanes_net = None
    if permission.get("lane_network"):
        lane_path = Path(dcfg.get("data_root", ".")) / permission["lane_network"]
        if hashlib.sha256(lane_path.read_bytes()).hexdigest() != permission["lane_network_sha256"]:
            raise ValueError("Lane network differs from the one the driving map was compiled with")
        from awr.life.civil.lane_network import LaneNetwork

        lanes_net = LaneNetwork(str(lane_path))
    domain = driving_area(sem, json.loads(Path(dcfg["source_graph"]).read_text()), lanes_net)
    adapter = DrivingAdapter(
        graph,
        size,
        key,
        domain,
        walking,
        surface,
        turn_radius,
        home_walk_m=float(dcfg.get("home_walk_m", 150.0)),
        facility_walk_m=float(dcfg.get("facility_walk_m", 200.0)),
        facility_berths=int(dcfg.get("facility_berths", 4)),
    )
    usable = adapter.usable_berths(
        [footprint(dict(xy=p, yaw=y), z) for p, y, z in zip(positions, yaws, sizes, strict=False)],
        np.asarray(static_walk_free),
        walking.turn_clear,
        passage_check[0],
        parking_check,
    )
    if overlay is not None:
        # A frozen parked set (scenario overlay): no resident car parks in its region.
        frozen = [b for b in usable if overlay.frozen(usable[b][0][:2])]
        for b in frozen:
            del usable[b]
        adapter.report["overlay_frozen_berths"] = len(frozen)
    director = runtime.director
    from awr.life.civil.activity.director import supports_activity
    from awr.life.civil.activity.parking_admission import JointParkingAccess
    from awr.life.civil.vehicle_connections import PreparedPermission

    permitted = PreparedPermission(domain)
    facilities = {}
    for place in director.places.values():
        if not any("car" in a.modes and supports_activity(place, a) for a in director.activities):
            continue
        for berth in adapter.parking_candidates(place, usable):
            facilities.setdefault(berth, []).append(place)

    def destination_access(actor, berth):
        resident = director.actors[actor]
        activities = [
            a
            for a in director.activities
            if "car" in a.modes
            and resident.role in a.roles
            and (resident.activities is None or a.purpose in resident.activities)
        ]
        anchor = usable[berth][1]
        for place in facilities[berth]:
            if not any(supports_activity(place, a) for a in activities):
                continue
            outward = runtime.measured_walk(anchor, place.xy)
            back = runtime.measured_walk(place.xy, anchor) if outward is not None else None
            if (
                outward is not None
                and back is not None
                and max(outward, back) <= getattr(runtime, "walking_range_m", (0.0, 450.0))[1]
            ):
                return True
        return False

    standing = [
        footprint(dict(xy=p, yaw=y), z)
        for a, p, y, z in zip(actors, positions, yaws, sizes, strict=False)
        if a["class"] != "person"
    ]
    joint = JointParkingAccess(
        graph, surface, size, permitted, standing, set(usable), set(facilities), destination_access
    )
    # A resident's car parked at home leaves the drivers' roads passable, as every parked car of the scene must
    # (awr.life.civil.activity.connected_parking): any two lanes that reach each other still do with it in place.
    from awr.life.civil.activity.connected_parking import ConnectedRoads

    home_roads = ConnectedRoads(graph, size, standing) if dcfg.get("home_parking_keeps_roads", True) else None
    # Overlay targets stay pedestrians: their destinations and appearance are the query's.
    targets = {f"person-{k:03d}" for k in range(len(overlay.persons))} if overlay is not None else set()
    residents = sorted(
        ((a.id, director.places[a.home]) for a in director.actors.values() if a.id not in targets), key=lambda r: r[0]
    )
    first = 1 + max(
        [int(a["id"].split("-")[1]) for a in actors if a["class"] == "vehicle" and a["id"].startswith("vehicle-")],
        default=-1,
    )
    vids = []

    def admit(actor, berth):
        # The car must still fit beside the cars placed before it: bodies apart and walking passages open.
        pose = np.asarray(graph.nodes[berth]["pose"], float)
        xy = pose[:2].copy()
        body = footprint(dict(xy=xy, yaw=pose[2]), size)
        if any(
            body.intersects(footprint(dict(xy=positions[i], yaw=yaws[i]), sizes[i]))
            for i in range(len(actors) - len(vids), len(actors))
        ):
            return False
        if not passage_check[0](xy, float(pose[2]), size):
            return False
        if home_roads is not None and not home_roads.allows(body):
            return False
        if not joint.accept(actor, berth):
            return False
        if home_roads is not None:
            home_roads.reserve(body)
        rr = {"road": berth, "points": np.array([xy, xy])}
        idx = add_vehicle(first + len(vids), key, spec, size, xy, float(pose[2]), rr, "resident_driver")
        actors[idx].update(owner=actor, home_berth=berth)
        jobs.append(
            {
                "route": rr,
                "s": 0.0,
                "direction": 1,
                "speed": 0.0,
                "wait_until": 1e9,
                "kind": "vehicle",
                "status": "waiting",
                "goal_s": 0.0,
                "max_speed": 0.0,
                "stuck_s": 0.0,
                "replan_after": t_start,
                "agenda": None,
                "initial_purpose": "resident_home_parking",
            }
        )
        jobs[-1]["motion_policy"] = "autonomous_activity"
        vids.append(actors[idx]["id"])
        return True

    chosen = adapter.assign(residents, usable, runtime.measured_walk, int(dcfg.get("drivers", 0)), admit)
    # Owned cars are live bodies throughout the run, including while parked.
    # Do not burn their initial poses into immutable walk/turn masks: those
    # ghost obstacles would survive departures. Live crowd/transfer checks
    # retain all of them in the authoritative body roster.
    adapter.bind(director, usable, chosen, vids)
    # Cruising drivers (life.driving.cruise.drivers) drive tours (GroundedActivities) between the facility berths of
    # the strongly connected part of the map that holds their home berth with the other cars parked: the owners whose
    # part offers the most such destinations are chosen (ties by ID), none with fewer than cruise.min_destinations.
    cruise = dcfg.get("cruise") or {}

    def parked_part(owner):
        # The owner's car leaves; every other admitted car and the scene's bodies stand where they are.
        blocked = joint.base_blocked | set().union(
            *(row["blocked"] for who, row in joint.selected.items() if who != owner)
        )
        home = joint.selected[owner]["home"]
        return next(group for group in graph.components(blocked_edges=blocked) if home in group)

    cruisers, cruise_parts = (
        ([], {})
        if not int(cruise.get("drivers", 0))
        else choose_cruisers(
            {a: b for a, _, b in chosen},
            set(facilities),
            int(cruise.get("drivers", 0)),
            int(cruise.get("min_destinations", 4)),
            parked_part,
        )
    )
    for a in cruisers:
        director.assign_role(a, "cruiser", ("drive_around",), t=t_start)
    runtime.access_points = {usable[b][1] for b in usable}
    occupancy = {
        a["id"]: Body((float(positions[i][0]), float(positions[i][1]), float(yaws[i])), tuple(sizes[i]))
        for i, a in enumerate(actors)
        if a["class"] != "person"
    }
    routes = DrivingRoutes(
        graph,
        surface,
        size,
        {e: permitted for e in graph.edges},
        occupancy,
        static_tree_capacity=int(dcfg.get("static_tree_capacity", 16)),
        tour_legs=int(cruise.get("legs", 2)),
        tour_spacing_m=float(cruise.get("spacing_m", 120.0)),
        no_standing=crossing_areas(sem),
    )
    runtime.driving_routes = routes
    runtime.driving_binding = DrivingBinding(routes)
    # A car parks only where the other cars keep the drives they still need (Director.ways_home) ...
    director.keeps_ways = routes.keeps_ways
    # ... and the roads stay passable, as for the cars parked before the run (ConnectedRoads.keeps).
    if home_roads is not None and dcfg.get("parking_keeps_roads", True):
        routes.roads = home_roads
        director.keeps_roads = routes.keeps_roads
    # ... nor shut in a car parked at its owner's home (Director.way_out).
    director.way_out = routes.way_out_nodes()
    # Admitted journeys that have not departed occupy their destination berths in plans that must hold later.
    routes.pending = director.pending_destinations
    life_meta["driving"] = dict(
        map=str(path),
        map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        asset=key,
        requested_drivers=int(dcfg.get("drivers", 0)),
        cruisers=cruisers,
        cruise_parts=cruise_parts,
        home_parking_roads=None if home_roads is None else home_roads.report,
        joint_parking=joint.report,
        **adapter.report,
    )
    print("driving", json.dumps(life_meta["driving"]), flush=True)


def build(
    world,
    nav,
    sem_path,
    out,
    duration=180.0,
    people=24,
    vehicles=14,
    config=None,
    plan_only=False,
    dynamic_validation=False,
    principle=None,
):
    if not out.name.startswith("v") or not out.name[1:].isdigit():
        raise ValueError("Output version must be v followed by an integer")
    if out.exists():
        raise ValueError("Choose a fresh output version")
    from awr.life.civil.provenance import snapshot

    cfg = json.loads(config.read_text()) if config else {}
    principle_pkg = principle_report = None
    if principle is not None:
        # A scene principle package (awr.life.principle.v1) is the only scene input: the world reference, support
        # grids, semantics, lane network and places all come from it, and the recipe must stay within the
        # capabilities it declares. Legacy --world/--nav/--semantics are not accepted beside it.
        if any(v is not None for v in (world, nav, sem_path)):
            raise ValueError("--principle replaces --world, --nav and --semantics")
        from awr.life.principle import admit as admit_principle

        principle_pkg, principle_report = admit_principle(
            Path(principle), cfg, people=int(cfg.get("people", people)), vehicles=int(cfg.get("vehicles", vehicles))
        )
        world, nav, sem_path = principle_pkg.world, principle_pkg.support, principle_pkg.semantics
        print(
            "principle",
            principle_pkg.id,
            principle_pkg.content_version[:12],
            json.dumps(principle_report["stats"]),
            flush=True,
        )
    elif any(v is None for v in (world, nav, sem_path)):
        raise ValueError("--world, --nav and --semantics are required without --principle")
    if dynamic_validation and (plan_only or not cfg.get("environment") or cfg.get("runtime") != "life_v1"):
        raise ValueError("Dynamic validation requires an environment and life_v1 runtime, without plan-only")
    environment_binding = None
    if cfg.get("environment"):
        # The environment package's declared capabilities bound the scenario before anything is built.
        from awr.life.envpack.binding import bind_inputs
        from awr.life.motion.capability import Demand, check_declared, load_environment

        manifest = load_environment(
            cfg["environment"], root=cfg.get("environment_root", "."), validation_only=plan_only or dynamic_validation
        )
        environment_binding = bind_inputs(
            manifest, world=world, support=nav, semantics=sem_path, config=cfg, root=cfg.get("environment_root", ".")
        )
        if schema_id(manifest["schema"]) == "awr.life.life.environment.v1":
            check_declared(manifest, Demand.from_config(cfg))
    extra = [sem_path] + ([config] if config else []) + ([Path(cfg["environment"])] if cfg.get("environment") else [])
    if principle_pkg is not None:
        extra.append(principle_pkg.root / "principle.json")
    public_regions = None
    if cfg.get("public_regions"):
        from awr.life.civil.public_regions import load as load_public_regions

        public_regions = load_public_regions(cfg["public_regions"], world=world, semantics=sem_path)
        extra.append(Path(cfg["public_regions"]))
    overlay = None
    if cfg.get("scenario_overlay"):
        # Per-query adjustments of who lives where and what they wear (awr.life.civil.activity.overlay).
        if public_regions is None:
            raise ValueError("A scenario overlay names public regions: public_regions required")
        from awr.life.civil.activity.overlay import Overlay

        overlay = Overlay.load(cfg["scenario_overlay"], public_regions)
        overlay.validate_config(cfg, int(cfg.get("people", people)))
        extra.append(Path(cfg["scenario_overlay"]))
    if cfg.get("lane_network"):
        extra.append(Path(cfg["lane_network"]))
    if cfg.get("outdoor_places"):
        extra.append(Path(cfg["outdoor_places"]))
    driving_cfg = (cfg.get("life") or {}).get("driving") or {}
    extra.extend(Path(driving_cfg[k]) for k in ("map", "source_graph") if driving_cfg.get(k))
    provenance = snapshot(Path(__file__).resolve().parents[4], world, nav, extra)
    if environment_binding is not None:
        provenance["environment_binding"] = environment_binding
    if principle_pkg is not None:
        provenance["principle"] = {
            "id": principle_pkg.id,
            "content_version": principle_pkg.content_version,
            "root": str(principle_pkg.root),
        }
    if dynamic_validation:
        provenance["execution_mode"] = "dynamic_validation"
    # Persist before any long simulation; failed contract checks must not erase
    # the only evidence of which inputs produced a pending binary.
    preflight = out.parent / (out.name + ".initial-inputs.json")
    if preflight.exists():
        raise ValueError("Build attempt already exists; choose fresh version")
    preflight.write_text(json.dumps(provenance, indent=2) + "\n")
    duration = float(cfg.get("duration_s", duration))
    people = int(cfg.get("people", people))
    vehicles = int(cfg.get("vehicles", vehicles))
    seed = int(cfg.get("seed", 20261020))
    rng = np.random.default_rng(seed)
    lane_network = None
    if cfg.get("lane_network"):
        from awr.life.civil.lane_network import LaneNetwork

        lane_network = LaneNetwork(cfg["lane_network"])
    vehicle_keys = cfg.get("vehicle_assets", ["sedan", "hatchback", "suv", "van", "pickup"])
    # Life scenarios (config section `life`): daily chains for residents, purposeful
    # vehicle trips without queues, and a warm-up that is simulated but not recorded
    # so that the recording starts in an ongoing day rather than at a common start.
    life = cfg.get("life") or None
    from awr.life.civil.vehicle_execution import selected as select_vehicle_execution

    reactive_vehicles = select_vehicle_execution(cfg)
    if life and not cfg.get("life_gates"):
        raise ValueError("A life scenario config must declare life_gates; the life audit is a publish condition")
    warm = float(life.get("warmup_s", 0.0)) if life else 0.0
    t_start = -round(warm / DT) * DT
    life_meta = {"persons": {}, "vehicle_chains": [], "resident_parked": []}
    human_count = int(cfg.get("human_assets", 4))
    # Entity classes (awr.life.classes): what each kind of walker is to the engine. The built-in catalog reproduces the
    # legacy formulas; a recipe adds classes under `classes` and asks for members under `population`.
    from awr.life.classes import load as load_entity_classes
    from awr.life.classes import population as class_population

    entity_classes = load_entity_classes(cfg)
    person_classes = class_population(cfg, people, entity_classes)
    person_members = [(ec, j) for ec, n_ in person_classes for j in range(n_)]
    people = len(person_members)
    props = int(cfg.get("props", 0))
    surface = SurfaceMap(nav)
    sem = json.loads(sem_path.read_text())
    actors = []
    jobs = []
    positions = []
    yaws = []
    sizes = []
    rejected = []
    usage = {}
    cached_routes = {}
    walking = None
    from awr.life.civil.agenda import Occupancy

    occupancy = Occupancy()
    places = json.loads(Path(cfg["outdoor_places"]).read_text())["places"] if cfg.get("outdoor_places") else None
    if cfg.get("walking_network"):
        from awr.life.civil.walking import WalkNetwork

        # Common radius covers all twelve animated body envelopes in this pack.
        max_radius = 0.0
        half_width = 0.0
        person_sizes = []
        for ec, j in person_members:
            spec, mesh, _ = load(ec.asset(j))
            envelope = np.maximum(np.max(np.abs(mesh["poses"]), axis=(0, 1)), np.max(np.abs(mesh["p"]), axis=0))
            wh = 2 * envelope[:2] * ec.height(j) / spec["dimensions_m"][2] + 0.04
            max_radius = max(max_radius, float(np.linalg.norm(wh) / 2))
            half_width = max(half_width, float(wh[1] / 2))
            person_sizes.append(wh)
        if cfg.get("shared_streets"):
            # Shared streets: bodies occupy the footprint of the pose they are drawn
            # in (awr.life.civil.body). Walkers pass each other and walk along walls by
            # their walking boxes, so the walking cells keep half the walking width
            # (plus margins) from unsupported ground. A person rotates in place only
            # at rest, drawn in the rest pose: the turning disk is that of the rest
            # pose, required where people rotate (sharp corners, standing points,
            # refuges: walking.turn_free).
            ext_all = [bodies.person_extents(ec.asset(j), round(ec.height(j), 3)) for ec, j in person_members]
            half_width = max(float(np.abs(e[1, 2:]).max()) for e in ext_all)
            max_radius = max(float(bodies.turn_radius(e[0])) for e in ext_all)
            person_sizes = [bodies.symmetric(e[0]) for e in ext_all]
        walk_radius = half_width if cfg.get("shared_streets") else max_radius
        walking = WalkNetwork.from_scene(
            surface,
            sem,
            walk_radius,
            step=float(cfg.get("walking_step_m", 0.5)),
            lane_network=lane_network,
            shared_streets=cfg.get("shared_streets"),
            turn_radius=max_radius if cfg.get("shared_streets") else None,
        )
        print("walking network", walking.summary(), flush=True)
    # Street furniture is placed on supported roadside ground, outside vehicle
    # lanes, before moving actors are admitted. It participates in every collision
    # query and retains an ordinary instance ID in all camera outputs.
    for j in range(props):
        if j % 10 == 0:
            print("placing furniture", j, props, flush=True)
        key = cfg["prop_assets"][j % len(cfg["prop_assets"])]
        spec, mesh, _ = load(key)
        size = np.asarray(spec["dimensions_m"])
        options = []
        for road in sem["roads"]:
            center = LineString(road["line"])
            for side in (-1, 1):
                line = center.offset_curve(side * (road["width_m"] / 2 + 2.2))
                if line.geom_type != "LineString":
                    continue
                for distance in np.arange(3.0, line.length - 3, 9.0):
                    p = np.array(line.interpolate(distance).coords[0])
                    q = np.array(line.interpolate(min(distance + 0.1, line.length)).coords[0])
                    angle = math.atan2(q[1] - p[1], q[0] - p[0])
                    corners = rectangle(p, angle, size, margin=0.12)
                    if not surface.supported(np.vstack([corners, p]), "person").all():
                        continue
                    if surface.supported(p[None], "vehicle")[0]:
                        continue
                    if positions and np.linalg.norm(np.asarray(positions) - p, axis=1).min() < 3.0:
                        continue
                    if collision_pairs(np.array(positions + [p]), np.array(yaws + [angle]), np.array(sizes + [size])):
                        continue
                    options.append((road, p, angle))
                    break
        if not options:
            rejected.append({"kind": "object", "requested_index": j, "reason": "no_free_roadside_support"})
            continue
        options.sort(key=lambda x: usage.get(("object", x[0]["id"]), 0))
        road, p, angle = options[0]
        idx = len(actors)
        usage[("object", road["id"])] = usage.get(("object", road["id"]), 0) + 1
        actors.append(
            {
                "idx": idx,
                "id": f"{key}-{j:03d}",
                "class": "object",
                "scripted": True,
                "profile": None,
                "size_mm": np.round(size * 1000).astype(int).tolist(),
                "appearance": {"asset": key + "@1", "asset_sha256": spec["npz_sha256"]},
                "role": "street_furniture",
                "semantic_type": key,
                "description": key,
            }
        )
        positions.append(p)
        yaws.append(angle)
        sizes.append(size)
        jobs.append(
            {
                "route": {"road": road["id"], "points": np.array([p, p])},
                "s": 0.0,
                "direction": 1,
                "speed": 0.0,
                "wait_until": duration + 1,
                "kind": "object",
                "status": "static",
                "goal_s": 0.0,
                "max_speed": 0.0,
            }
        )
    static_walk_free = walking.obstacle_mask(positions, sizes, walk_radius, yaws) if walking else None
    permanent_walk_obstacles = set()
    shared_walk = bool(cfg.get("shared_streets")) and walking is not None
    if walking is not None and walking.turn_free is not None:
        walking.turn_body = tuple(np.max([np.asarray(sz)[:2] for sz in person_sizes], axis=0)) if person_sizes else None
        walking.turn_clear = (
            np.asarray(walking.obstacle_mask(positions, sizes, max_radius, yaws))
            if len(positions)
            else np.ones_like(walking.free)
        )
        walking.turn_free = walking.turn_free & walking.turn_clear
    pools = None
    site_slots = None
    slot_holder = {}
    activity_runtime = None
    if life and walking:
        from awr.life.civil.itineraries import life_pools

        pools = life_pools(walking, sem, places or [], static_walk_free, rng, life.get("places"))
        # Standing slots per site; a call at a household uses its single point.
        site_slots = {st["id"]: st["slots"] for st in pools["sites"]}
        site_slots.update(
            {
                sp["id"]: [dict(id=sp["id"] + "#slot0", xy=sp["xy"])]
                for sp in pools["street"]
                if sp["kind"] == "household"
            }
        )
        # Other approaches of the same facility in the same walking component,
        # nearest first: where a resident stands when its planned approach is full.
        site_alternatives = {}
        for st in pools["sites"]:
            near = [
                (float(np.linalg.norm(np.asarray(o["xy"]) - np.asarray(st["xy"]))), o["id"])
                for o in pools["sites"]
                if o["id"] != st["id"] and o.get("facility") == st.get("facility") and o["comp"] == st["comp"]
            ]
            site_alternatives[st["id"]] = [
                sid for d_, sid in sorted(near) if d_ <= float(life.get("alternative_approach_m", 120.0))
            ]
        print(
            "life pools",
            {k: len(v) for k, v in pools.items()},
            "slots",
            sum(len(v) for v in site_slots.values()),
            flush=True,
        )
    if life and life.get("behavior") is not None:
        if life["behavior"] != "activities_v1" or walking is None:
            raise ValueError("Unknown activity behavior or missing walking geometry")
        from awr.life.civil.activity.grounded import GroundedActivities

        activity_runtime = GroundedActivities(walking, pools, static_walk_free, life, seed)

    def with_travel_room(free):
        # Journeys were admitted through measured space with passing room
        # (itineraries.plan_outdoor_visits). Prefer that space at run time as
        # well: single-file shortcuts let two opposing streams lock each other.
        travel = getattr(walking, "travel_free", None)
        if travel is None:
            return None
        room = free.copy().view(type(free))
        room &= travel
        for name in ("positions", "sizes", "yaws", "radius", "circle_radii"):
            if hasattr(free, name):
                setattr(room, name, getattr(free, name))
        room.static_free = np.asarray(getattr(free, "static_free", free)) & travel
        return room

    rules = None
    if lane_network is not None:
        from awr.life.civil.parking_rules import ParkingRules

        rules = ParkingRules(sem, lane_network, **cfg.get("parking_rules", {}))
    curb_mode = bool(life and life.get("vehicles", {}).get("behavior") == "curb_v1")
    driving_map = None
    if life and life.get("driving"):
        # The resident drivers' map (admit_drivers), loaded once: the parked cars placed before them keep its roads.
        from awr.life.civil.driving_graph import load_candidate

        _doc = json.loads(Path(life["driving"]["map"]).read_text())
        driving_map = (_doc, load_candidate(_doc))
    if life and life.get("vehicles", {}).get("behavior") not in (None, "curb_v1"):
        raise ValueError("Unknown vehicle behavior; refusing legacy fallback")
    if curb_mode and (rules is None or walking is None or not cfg.get("shared_streets")):
        raise ValueError("curb_v1 requires road rules and the shared-street walking network")
    from awr.life.civil.fleet import composition as fleet_composition
    from awr.life.civil.fleet import requests as fleet_requests

    if life and life.get("vehicles", {}).get("population") is not None and not curb_mode:
        raise ValueError("Explicit mixed-fleet population requires curb_v1")
    if curb_mode:
        fleet_requests(vehicles, vehicle_keys, life["vehicles"])
    curb_cache = {}
    curb_reports = {}

    def curb_roads(size):
        from awr.life.civil.curb import candidates

        key = tuple(size)
        if key not in curb_cache:
            report = {}
            lv = life.get("vehicles", {})
            curb_cache[key] = candidates(
                surface,
                sem,
                size,
                rules,
                min_trip=float(lv.get("min_trip_m", 60.0)),
                merge_m=float(lv.get("merge_m", 18.0)),
                report=report,
            )
            curb_reports[str(key)] = report
        return curb_cache[key]

    delivery = cfg.get("delivery") if (walking and places is not None and lane_network is not None) else None
    delivery_plan = None
    if delivery:
        # One delivery round: a van drives a validated lane chain and stops where
        # the courier can alight onto supported sidewalk and walk to a mapped
        # outdoor receiving point. No boundary entry/exit: no supported lane
        # chain reaches the stage boundary in this support model.
        if not hasattr(walking, "assigned_waiting_places"):
            walking.assigned_waiting_places = set()
        if not hasattr(walking, "assigned_waiting_points"):
            walking.assigned_waiting_points = []
        van_key = delivery.get("van_asset", "van")
        vspec, _, _ = load(van_key)
        van_size = np.array(vspec["dimensions_m"])
        cache_key = ("vehicle", tuple(van_size))
        if cache_key not in cached_routes:
            cached_routes[cache_key] = routes(surface, sem, "vehicle", van_size) + lane_network.supported_routes(
                surface, van_size, min_length=35.0
            )
        if curb_mode:
            from awr.life.civil.curb import manoeuvre

            delivery_routes = []
            for road in curb_roads(van_size):
                for trip in road["trips"]:
                    start, end = trip["start"], trip["end"]
                    mids = [b for b in road["berths"] if start["s"] + 40.0 <= b["s"] <= end["s"] - 40.0]
                    if not mids:
                        continue
                    mid = min(mids, key=lambda b: abs(b["s"] - (start["s"] + end["s"]) / 2))
                    rr = manoeuvre(
                        road["geometry"],
                        start["s"],
                        end["s"],
                        road["layout"],
                        van_size,
                        surface,
                        road["road"],
                        stops=[mid["s"]],
                        merge_m=float(life["vehicles"].get("merge_m", 18.0)),
                    )
                    if rr is None:
                        continue
                    rr["curb_berths"] = [start, mid, end]
                    rr["street_layout"] = road["layout"]
                    delivery_routes.append(rr)
            cached_routes[cache_key] = delivery_routes
        courier_asset = "human-" + str(int(delivery.get("courier_asset", 7)) % human_count + 1)
        cspec, cmesh, _ = load(courier_asset)
        cheight = 1.72
        cenv = np.maximum(np.max(np.abs(cmesh["poses"]), axis=(0, 1)), np.max(np.abs(cmesh["p"]), axis=0))
        csize = np.r_[2 * cenv[:2] * cheight / cspec["dimensions_m"][2] + 0.04, cheight]
        cradius = float(np.linalg.norm(csize[:2]) / 2)
        radius_m = float(delivery.get("receiving_radius_m", 60.0))
        chosen = None
        probes = []
        # Receiving points: mapped outdoor approaches where they exist near the
        # drivable chains, otherwise walkable cells 2.5-8 m outside a building
        # footprint (an outdoor approach, not a verified entrance). The mapped
        # approaches of this town lie far from every drivable chain.
        import shapely as _sh
        from scipy.spatial import cKDTree as _KD
        from shapely.geometry import Polygon as _Polygon
        from shapely.ops import unary_union as _union

        building_polys = [
            (b.get("id", f"building-{k}"), _Polygon(b["polygon"]))
            for k, b in enumerate(sem.get("buildings", []))
            if len(b.get("polygon", [])) >= 3
        ]
        building_union = _union([g for _, g in building_polys]) if building_polys else None
        receivers = [dict(id=pl["id"], xy=np.asarray(pl["xy"], float), kind="mapped_outdoor_approach") for pl in places]
        if building_union is not None and walking.xy is not None and len(walking.xy):
            dist_b = _sh.distance(building_union, _sh.points(walking.xy))
            for k in np.flatnonzero((dist_b >= 2.5) & (dist_b <= 8.0)):
                receivers.append(
                    dict(id=f"building-approach-node-{int(k)}", xy=walking.xy[k], kind="building_outdoor_approach")
                )
        rxy = np.array([r["xy"] for r in receivers])
        rtree = _KD(rxy)
        for rr in sorted(cached_routes[cache_key], key=lambda r: -route_length(r)):
            L = route_length(rr)
            if float(delivery.get("min_route_m", 150.0)) > L:
                break
            xy0, yaw0 = at(rr, 0.0)
            if positions and collision_pairs(
                np.array(positions + [xy0]), np.array(yaws + [yaw0]), np.array(sizes + [van_size])
            ):
                continue
            stops = []
            last = -1e9
            reserved = []
            probe = dict(samples=0, kerb_found=0, place_found=0)
            for s_stop in rr["curb_stops_m"][:-1] if curb_mode else np.arange(30.0, L - 30.0, 5.0):
                if s_stop - last < float(delivery.get("stop_spacing_m", 40.0)):
                    continue
                p, a = at(rr, s_stop)
                normal = np.array([math.sin(a), -math.cos(a)])
                alight = None
                probe["samples"] += 1
                # The courier steps onto the nearest walkable kerb on the
                # vehicle's right side; the door-to-kerb distance is recorded.
                for extra_m in np.arange(0.0, float(delivery.get("max_door_to_kerb_m", 3.0)) + 1e-9, 0.25):
                    q = p + normal * (van_size[1] / 2 + cradius + 0.25 + extra_m)
                    cell = walking.cell(q)
                    if not (
                        (cell >= 0).all()
                        and (cell < np.array(walking.free.shape)).all()
                        and static_walk_free[tuple(cell)]
                    ):
                        continue
                    if not surface.supported(np.vstack([rectangle(q, a, csize, margin=0.12), q[None]]), "person").all():
                        continue
                    if lane_network.driving.contains(Point(q)):
                        continue
                    alight = q
                    door_to_kerb = float(extra_m)
                    break
                if alight is None:
                    continue
                probe["kerb_found"] += 1
                best = None
                near = sorted(rtree.query_ball_point(alight, radius_m), key=lambda k: np.linalg.norm(rxy[k] - alight))
                tried = 0
                for k in near:
                    pl = receivers[k]
                    if pl["id"] in walking.assigned_waiting_places or pl["id"] in reserved:
                        continue
                    d = np.linalg.norm(pl["xy"] - alight)
                    if d < 8.0:
                        continue
                    if any(np.linalg.norm(pl["xy"] - q) < 3.0 for q in walking.assigned_waiting_points):
                        continue
                    route = walking.plan(alight, pl["xy"], static_walk_free)
                    tried += 1
                    if route is not None and route_length(route) <= 1.6 * radius_m:
                        best = (route_length(route), pl)
                        break
                    if tried >= 8:
                        break
                if best is None:
                    continue
                probe["place_found"] += 1
                stops.append(
                    dict(
                        s=float(s_stop),
                        alight_xy=alight.tolist(),
                        yaw=float(a),
                        door_to_kerb_m=round(door_to_kerb, 2),
                        place=best[1]["id"],
                        place_kind=best[1]["kind"],
                        place_xy=[float(v) for v in best[1]["xy"]],
                        walk_m=round(best[0], 1),
                        verified_entrance=False,
                    )
                )
                reserved.append(best[1]["id"])
                last = s_stop
                if len(stops) >= int(delivery.get("max_stops", 2)):
                    break
            probes.append(dict(road=rr["road"], length_m=round(L, 1), **probe))
            if stops:
                chosen = (rr, stops)
                break
        if chosen is None:
            rejected.append(
                {
                    "kind": "delivery",
                    "reason": "no_supported_route_with_reachable_receiving_points",
                    "probes": probes[:12],
                }
            )
        else:
            rr, stops = chosen
            xy0, yaw0 = at(rr, 0.0)
            van_idx = len(actors)
            for st in stops:
                walking.assigned_waiting_places.add(st["place"])
                walking.assigned_waiting_points.append(np.asarray(st["place_xy"]))
            from awr.life.appearance.identity import clothing, plate_text
            from awr.life.civil.population import CAR_COLOURS, VEHICLE_TYPES

            rgb, _, cname = CAR_COLOURS[int(delivery.get("van_colour", 3)) % len(CAR_COLOURS)]
            actors.append(
                {
                    "idx": van_idx,
                    "id": "delivery-van-000",
                    "class": "vehicle",
                    "scripted": True,
                    "profile": "vehicle",
                    "size_mm": list(np.round(van_size * 1000).astype(int)),
                    "appearance": {
                        "asset": van_key + "@1",
                        "asset_sha256": vspec["npz_sha256"],
                        "body_rgb": list(rgb),
                        "colour_name": cname,
                        "plate_text": plate_text(900),
                        "plate_synthetic": True,
                    },
                    "role": "delivery_van",
                    "vtype": van_key,
                    "semantic_type": van_key,
                    "description": f"{cname}{VEHICLE_TYPES[van_key][2]}（送货）",
                }
            )
            positions.append(np.asarray(xy0))
            yaws.append(yaw0)
            sizes.append(van_size)
            usage[("vehicle", rr["road"])] = usage.get(("vehicle", rr["road"]), 0) + 1
            jobs.append(
                {
                    "route": rr,
                    "s": 0.0,
                    "direction": 1,
                    "speed": 0.0,
                    "wait_until": float(delivery.get("departure_s", 90.0)),
                    "kind": "vehicle",
                    "status": "waiting",
                    "goal_s": stops[0]["s"],
                    "max_speed": 3.0,
                    "stuck_s": 0.0,
                    "replan_after": t_start,
                    "agenda": None,
                    "depart_purpose": "delivery_round",
                    "curb_berths": rr.get("curb_berths"),
                    "initial_purpose": "delivery_depot_parking" if curb_mode else "initial_roadside",
                    "arrival_purpose": "destination_parking" if curb_mode else "end_of_trip",
                    "delivery": dict(
                        role="van", stops=stops, next=0, phase="driving", courier=None, route_m=route_length(rr)
                    ),
                }
            )
            capp = {
                "asset": courier_asset + "@1",
                "asset_sha256": cspec["npz_sha256"],
                **clothing(int(delivery.get("courier_clothing", 41))),
            }
            delivery_plan = dict(van_idx=van_idx, xy0=np.asarray(xy0), yaw0=yaw0, csize=csize, capp=capp, stops=stops)
    passage_check = []  # the parked-car passage rule of plan_life_vehicles, reused for resident drivers' cars

    def plan_life_vehicles(count):
        """Purposeful resident trips on validated finite chains, no queues.

        Each chain carries at most a few vehicles parked one behind the other at
        its start; they leave front first, each only after the one ahead has parked
        at its own destination further along the chain, so no vehicle ever waits
        behind a parked car for its trip. Chains used for trips are mutually
        separated. Vehicles without a trip slot are parked residents placed away
        from every trip corridor; they keep a recorded purpose and never move.
        A return trip or loop is not available: the support has no strongly
        connected lane core and no verified turning space.
        """
        lv = life.get("vehicles", {})
        planned_parks = []
        stand_cells = np.zeros(walking.free.shape, bool) if shared_walk else None
        slot_xy = (
            np.array([sl_["xy"] for st in pools["sites"] for sl_ in st["slots"]], float).reshape(-1, 2)
            if shared_walk and pools is not None
            else np.zeros((0, 2))
        )
        if shared_walk and pools is not None:
            # every standing point of daily life (site slots, street and household points)
            for q in [sl_["xy"] for st in pools["sites"] for sl_ in st["slots"]] + [sp["xy"] for sp in pools["street"]]:
                c_ = walking.cell(np.asarray(q, float))
                if (c_ >= 0).all() and (c_ < np.array(stand_cells.shape)).all():
                    stand_cells[tuple(c_)] = True

        def keeps_passage(xy, yaw, size):
            # A parked car must leave walkers a way past it: within 12 m of it, no
            # walking area may be cut into separate parts by its footprint (other
            # parked or planned cars and street objects included), and no standing
            # point may end up in a cut-off pocket however small. Drivers do not
            # park where they close the only passage of a pavement or a doorway.
            if not shared_walk:
                return True
            from scipy.ndimage import label as _label

            r = int(12.0 / walking.step)
            c = walking.cell(xy)
            lo = np.maximum(c - r, 0)
            hi = np.minimum(c + r + 1, np.array(walking.free.shape))
            if (hi <= lo).any():
                return True
            sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
            gy, gx = np.mgrid[sl[0], sl[1]]
            pts = walking.origin + np.stack([gx, gy], -1) * walking.step

            def clear_of(boxes):
                # walking cells clear of these bodies (criterion of WalkNetwork.obstacle_mask)
                m = np.ones(pts.shape[:2], bool)
                for p_, y_, z_ in boxes:
                    d = pts - np.asarray(p_, float)
                    c_, s_ = math.cos(y_), math.sin(y_)
                    loc = np.abs(np.stack([d[..., 0] * c_ + d[..., 1] * s_, -d[..., 0] * s_ + d[..., 1] * c_], -1))
                    m &= (
                        np.linalg.norm(np.maximum(loc - np.asarray(z_[:2], float) / 2, 0), axis=-1)
                        > walk_radius + 0.08 + PLANNING_MARGIN_M
                    )
                return m

            others = [
                (p_, y_, z_)
                for p_, y_, z_ in list(zip(positions, yaws, sizes, strict=False)) + planned_parks
                if np.linalg.norm(np.asarray(p_) - xy) < 20.0
            ]
            base = np.asarray(static_walk_free)[sl] & clear_of(others)
            after = base & clear_of([(xy, yaw, size)])
            lb, nb = _label(base)
            la, _ = _label(after)
            stand = stand_cells[sl]
            for comp in range(1, nb + 1):
                q = la[(lb == comp) & after]
                if not len(q):
                    continue
                parts = np.bincount(q[q > 0])
                if int((parts >= 8).sum()) > 1:
                    return False
                held = np.bincount(la[(lb == comp) & after & stand], minlength=len(parts))[: len(parts)]
                if (held[parts > 0] > 0).any() and int((parts > 0).sum()) > 1:
                    main = int(np.argmax(parts))
                    if any(held[k] > 0 for k in np.flatnonzero(parts) if k != main):
                        return False
            # A passage the car narrows to one or two cells (0.5-1 m of centre
            # line, often a diagonal staircase along a kerb or wall) cannot be
            # walked by an upright walking body; it counts as closed: the walking
            # area shrunk by one cell must not be split by the car either.
            from scipy.ndimage import binary_erosion as _erode

            sq = np.ones((3, 3), bool)
            be, ae = _erode(base, sq), _erode(after, sq)
            lb, nb = _label(be)
            la, _ = _label(ae)
            for comp in range(1, nb + 1):
                q = la[(lb == comp) & ae]
                if len(q) and int((np.bincount(q[q > 0]) >= 8).sum()) > 1:
                    return False
            # The passage must also stay open while people stand at the facility
            # slots nearby (each slot occupied by a standing body): a car that
            # leaves only the width of an occupied slot closes the pavement.
            near = slot_xy[np.linalg.norm(slot_xy - np.asarray(xy, float), axis=1) < 14.0] if len(slot_xy) else slot_xy
            if len(near):
                standing = [(q_, 0.0, np.array([0.7, 0.7, 1.7])) for q_ in near]
                base_s = base & clear_of(standing)
                after_s = after & clear_of(standing)
                lb, nb = _label(base_s)
                la, _ = _label(after_s)
                for comp in range(1, nb + 1):
                    q = la[(lb == comp) & after_s]
                    if len(q) and int((np.bincount(q[q > 0]) >= 8).sum()) > 1:
                        return False
            return True

        passage_check.append(keeps_passage)
        if curb_mode:
            from awr.life.civil.curb_allocation import Allocator, departure_times
            from awr.life.civil.curb_allocation import footprint as _footprint

            connected_roads = None
            if driving_map is not None:
                # A car parked for the whole recording cuts no road of the drivers' map
                # (awr.life.civil.activity.connected_parking).
                from awr.life.civil.activity.connected_parking import ConnectedRoads

                _spec, _, _ = load(driving_map[0]["asset"])
                connected_roads = ConnectedRoads(
                    driving_map[1],
                    np.array(_spec["dimensions_m"]),
                    [
                        _footprint(dict(xy=p_, yaw=y_), z_)
                        for a_, p_, y_, z_ in zip(actors, positions, yaws, sizes, strict=False)
                        if a_["class"] != "person"
                    ],
                )
            allocator = Allocator(
                surface, keeps_passage, list(zip(positions, yaws, sizes, strict=False)), roads=connected_roads
            )
            if delivery_plan is not None:
                vi = delivery_plan["van_idx"]
                vr = jobs[vi]["route"]
                # Initial van body is already in obstacles; reserve the other
                # stops as well so no resident ends up parked at its next stop.
                allocator.reserve(vr["curb_berths"][1:], sizes[vi], vr)
                planned_parks.extend((np.asarray(b["xy"]), b["yaw"], sizes[vi]) for b in vr["curb_berths"][1:])
            roster = fleet_requests(count, vehicle_keys, lv)
            parking_requests = {} if overlay is None else overlay.parking_requests(roster)
            n_trip = fleet_composition(roster)["curb_trip"]
            audit = {
                "schema": "awr.life.curb_admission.v1",
                "requested": count,
                "requested_trips": n_trip,
                "requested_population": fleet_composition(roster),
                "actors": [],
                "failures": [],
                "geometry": curb_reports,
                "parking_lots": {},
            }
            lot_cache = {}
            for request in roster:
                j, key = request.index, request.asset
                placement = parking_requests.get(j)
                region = None if placement is None else placement["polygon"]
                spec, _, _ = load(key)
                size = np.array(spec["dimensions_m"])
                is_trip = request.mode == "curb_trip"
                if request.mode == "lot_parked":
                    from awr.life.civil.parking_lots import allocate as allocate_lot
                    from awr.life.civil.parking_lots import candidates as lot_candidates

                    if tuple(size) not in lot_cache:
                        report = {}
                        lot_cache[tuple(size)] = lot_candidates(surface, sem, size, rules, report)
                        audit["parking_lots"][str(tuple(size))] = report
                    plan = allocate_lot(allocator, lot_cache[tuple(size)], size, region=region)
                else:
                    roads = curb_roads(size)
                    plan = (
                        allocator.trip(
                            roads,
                            size,
                            min_trip=float(lv.get("min_trip_m", 60.0)),
                            merge_m=float(lv.get("merge_m", 18.0)),
                        )
                        if is_trip
                        else allocator.stationary(roads, size, region=region)
                    )
                if plan is None:
                    failure = dict(
                        actor=f"vehicle-{j:03d}",
                        asset=key,
                        kind="vehicle",
                        requested_index=j,
                        requested_mode=request.mode,
                        reason="no_verified_" + request.mode,
                    )
                    if placement is not None:
                        failure.update(overlay=placement["key"], region=placement["region"])
                    rejected.append(failure)
                    audit["failures"].append(failure)
                    continue
                if is_trip:
                    rr = plan["route"]
                    start = plan["start"]
                    end = plan["end"]
                    goal = route_length(rr)
                    depart = (
                        t_start
                        if activity_runtime is not None
                        else float(rng.uniform(t_start, max(t_start, duration - goal / 3.0 - 10.0)))
                    )
                    role = "resident_trip"
                    max_speed = 3.0 + 0.2 * (j % 4)
                    planned_parks.append((np.asarray(end["xy"]), end["yaw"], size))
                    record = dict(
                        actor=f"vehicle-{j:03d}",
                        road=rr["road"],
                        start=start,
                        end=end,
                        trip_m=goal,
                        scheduled_s=depart,
                        layout=plan["layout"],
                    )
                    life_meta["vehicle_chains"].append(dict(road=rr["road"], vehicles=[record]))
                else:
                    start = plan["slot"]
                    p = np.asarray(start["xy"])
                    rr = {"road": plan["road"], "points": np.array([p, p]), "yaw": np.array([start["yaw"]] * 2)}
                    goal = 0.0
                    depart = 1e9
                    role = "resident_parked"
                    max_speed = 0.0
                    record = dict(
                        actor=f"vehicle-{j:03d}",
                        road=rr["road"],
                        start=start,
                        layout=plan["layout"],
                        purpose="resident_parking",
                    )
                idx = add_vehicle(j, key, spec, size, np.asarray(start["xy"]), start["yaw"], rr, role)
                actors[idx]["allocation_mode"] = request.mode
                if placement is not None:
                    actors[idx]["parking_overlay"] = placement["key"]
                    record["parking_overlay"] = placement["key"]
                if request.mode == "lot_parked":
                    actors[idx].update(parking_region=start["parking_lot"], parking_bay=start["bay"])
                record["allocation_mode"] = request.mode
                record["purpose"] = request.initial_purpose
                jobs.append(
                    dict(
                        route=rr,
                        s=0.0,
                        direction=1,
                        speed=0.0,
                        wait_until=depart,
                        kind="vehicle",
                        status="waiting",
                        goal_s=goal,
                        max_speed=max_speed,
                        stuck_s=0.0,
                        replan_after=t_start,
                        agenda=None,
                        curb_berths=[start, end] if is_trip else [start],
                        depart_purpose="resident_trip",
                        initial_purpose=request.initial_purpose,
                        arrival_purpose="destination_parking",
                    )
                )
                if not is_trip:
                    life_meta["resident_parked"].append(actors[idx]["id"])
                audit["actors"].append(record)
            if delivery and delivery_plan is None:
                audit["failures"].append(dict(actor="delivery-van-000", reason="no_verified_curb_delivery_round"))
            records = [a for a in audit["actors"] if "trip_m" in a]
            by_id = {a["id"]: i for i, a in enumerate(actors)}
            if activity_runtime is not None:
                from awr.life.civil.activity.traffic import parking_activity

                for record in records:
                    record["initial_activity"] = parking_activity(
                        record["actor"], seed, t_start, tuple(lv.get("initial_parking_activity_s", (120.0, 1800.0)))
                    )
                times = [a["initial_activity"]["departure_due_s"] for a in records]
            else:
                times = departure_times(
                    [a["trip_m"] / jobs[by_id[a["actor"]]]["max_speed"] for a in records], duration, warm, rng
                )
            for record, depart in zip(records, times, strict=False):
                record["scheduled_s"] = float(depart)
                jobs[by_id[record["actor"]]]["wait_until"] = float(depart)
            audit["all_requested_admitted"] = not audit["failures"]
            if connected_roads is not None:
                audit["connected_roads"] = connected_roads.report
            audit["admitted_population"] = {
                mode: sum(a["allocation_mode"] == mode for a in audit["actors"]) for mode in fleet_composition(roster)
            }
            (out.parent / (out.name + ".curb-admission.json")).write_text(json.dumps(audit, indent=2) + "\n")
            if audit["failures"]:
                raise ValueError(
                    f"Curb admission failed for {len(audit['failures'])} actors; see {out.name}.curb-admission.json. No static fallback or partial-population tape is published."
                )
            life_meta["curb_admission"] = audit
            return
        max_per_chain = int(lv.get("max_per_chain", 3))
        gap = float(lv.get("stack_gap_m", 4.0))
        min_trip = float(lv.get("min_trip_m", 60.0))
        sep = float(lv.get("chain_separation_m", 3.2))
        trip_share = float(lv.get("trip_share", 1.0))
        from scipy.spatial import cKDTree as _KD3

        reserved = []
        if delivery_plan is not None:
            reserved.append(np.asarray(jobs[delivery_plan["van_idx"]]["route"]["points"]))

        def near_reserved(pts, limit):
            if not reserved:
                return np.zeros(len(pts), bool)
            tree = _KD3(np.vstack(reserved))
            return tree.query(pts)[0] < limit

        chains = []
        static = []
        n_trip = int(round(count * trip_share))
        for j in range(count):
            key = vehicle_keys[j % len(vehicle_keys)]
            spec, _, _ = load(key)
            size = np.array(spec["dimensions_m"])
            ck = ("vehicle", tuple(size))
            if ck not in cached_routes:
                cached_routes[ck] = routes(surface, sem, "vehicle", size) + (
                    lane_network.supported_routes(surface, size, min_length=35.0) if lane_network else []
                )
            placed = False
            if j < n_trip:
                for ch in chains:
                    if ch["size_key"] == ck and len(ch["members"]) < ch["capacity"]:
                        ch["members"].append((j, key, spec, size))
                        placed = True
                        break
                if not placed:
                    for rr in sorted(cached_routes[ck], key=lambda r: -route_length(r)):
                        L = route_length(rr)
                        cap = min(max_per_chain, int((L - min_trip) // (size[0] + gap)) + 1)
                        if min_trip + size[0] > L:
                            break
                        if cap < 1:
                            continue
                        pts = np.asarray(rr["points"])
                        if near_reserved(pts[::3], sep + 0.5).mean() > 0.02:
                            continue
                        chains.append(dict(size_key=ck, route=rr, capacity=cap, members=[(j, key, spec, size)]))
                        reserved.append(pts)
                        placed = True
                        break
            if not placed:
                static.append((j, key, spec, size))
        span = (t_start, duration - 60.0)
        for c, ch in enumerate(chains):
            rr = ch["route"]
            L = route_length(rr)
            k = len(ch["members"])
            step = max(m[3][0] for m in ch["members"]) + gap
            times = np.sort(rng.uniform(*span, size=k))
            prev = None
            admitted = []
            last_dest = None
            for m, (j, key, spec, size) in enumerate(ch["members"]):
                start = (k - 1 - m) * step
                dest = L - 0.5 - m * step if last_dest is None else min(L - 0.5 - m * step, last_dest - step)
                # The destination moves back along the chain until the parked car
                # leaves walkers a way past it.
                while dest - start >= min_trip and not keeps_passage(*at(rr, dest), size):
                    dest -= 1.0
                if dest - start < min_trip:
                    static.append((j, key, spec, size))
                    continue
                xy, yaw = at(rr, start)
                if positions and collision_pairs(
                    np.array(positions + [xy]), np.array(yaws + [yaw]), np.array(sizes + [size])
                ):
                    static.append((j, key, spec, size))
                    continue
                if not keeps_passage(xy, yaw, size):
                    static.append((j, key, spec, size))
                    continue
                last_dest = dest
                planned_parks.append((*at(rr, dest), size))
                idx = add_vehicle(j, key, spec, size, xy, yaw, rr, "resident_trip")
                jobs.append(
                    {
                        "route": rr,
                        "s": start,
                        "direction": 1,
                        "speed": 0.0,
                        "wait_until": float(times[m]),
                        "kind": "vehicle",
                        "status": "waiting",
                        "goal_s": dest,
                        "max_speed": 3.0 + 0.2 * (j % 4),
                        "stuck_s": 0.0,
                        "replan_after": t_start,
                        "agenda": None,
                        "depart_purpose": "resident_trip",
                        "initial_purpose": "resident_parking",
                        "arrival_purpose": "destination_parking",
                        "after": prev,
                        "after_gap": float(rng.uniform(*lv.get("follow_gap_s", [20.0, 90.0]))),
                    }
                )
                admitted.append(
                    dict(
                        actor=actors[idx]["id"],
                        start_m=round(start, 1),
                        destination_m=round(dest, 1),
                        trip_m=round(dest - start, 1),
                        scheduled_s=round(float(times[m]), 1),
                        after=actors[prev]["id"] if prev is not None else None,
                    )
                )
                prev = idx
            life_meta["vehicle_chains"].append(dict(chain=c, road=rr["road"], length_m=round(L, 1), vehicles=admitted))
        corridor = _KD3(np.vstack(reserved)) if reserved else None
        for j, key, spec, size in static:
            ck = ("vehicle", tuple(size))
            chosen = None
            cands = sorted(cached_routes[ck], key=lambda r: (usage.get(("vehicle", r["road"]), 0), -len(r["points"])))
            for rr in cands:
                for start in rng.permutation(np.arange(0.0, max(0.1, route_length(rr) - 2.0), 6.0)):
                    xy, yaw = at(rr, start)
                    if corridor is not None and corridor.query(xy)[0] < sep:
                        continue
                    if positions and collision_pairs(
                        np.array(positions + [xy]), np.array(yaws + [yaw]), np.array(sizes + [size])
                    ):
                        continue
                    others = [k2 for k2, a in enumerate(actors) if a["class"] == "vehicle"]
                    if others and np.min(np.linalg.norm(np.asarray(positions)[others] - xy, axis=1)) < size[0] + 1.0:
                        continue
                    if not keeps_passage(xy, yaw, size):
                        continue
                    chosen = (rr, start, xy, yaw)
                    break
                if chosen:
                    break
            if chosen is None:
                rejected.append(
                    {"kind": "vehicle", "requested_index": j, "reason": "no_parking_space_outside_trip_corridors"}
                )
                continue
            rr, start, xy, yaw = chosen
            usage[("vehicle", rr["road"])] = usage.get(("vehicle", rr["road"]), 0) + 1
            idx = add_vehicle(j, key, spec, size, xy, yaw, rr, "resident_parked")
            jobs.append(
                {
                    "route": {"road": rr["road"], "points": np.array([xy, xy])},
                    "s": 0.0,
                    "direction": 1,
                    "speed": 0.0,
                    "wait_until": 1e9,
                    "kind": "vehicle",
                    "status": "waiting",
                    "goal_s": 0.0,
                    "max_speed": 0.0,
                    "stuck_s": 0.0,
                    "replan_after": t_start,
                    "agenda": None,
                    "initial_purpose": "resident_parking",
                }
            )
            life_meta["resident_parked"].append(actors[idx]["id"])

    def add_vehicle(j, key, spec, size, xy, yaw, rr, role):
        idx = len(actors)
        actor = {
            "idx": idx,
            "id": f"vehicle-{j:03d}",
            "class": "vehicle",
            "scripted": True,
            "profile": "vehicle",
            "size_mm": list(np.round(size * 1000).astype(int)),
            "appearance": {"asset": key + "@1", "asset_sha256": spec["npz_sha256"]},
            "role": role,
            "vtype": key,
            "semantic_type": key,
            "description": key,
        }
        if cfg.get("visual_identity"):
            from awr.life.appearance.identity import plate_text
            from awr.life.civil.population import CAR_COLOURS, VEHICLE_TYPES

            rgb, _, name = CAR_COLOURS[j % len(CAR_COLOURS)]
            actor["appearance"].update(
                body_rgb=list(rgb), colour_name=name, plate_text=plate_text(j), plate_synthetic=True
            )
            actor["description"] = f"{name}{VEHICLE_TYPES[key][2]}"
        actors.append(actor)
        positions.append(xy)
        yaws.append(yaw)
        sizes.append(size)
        usage[("vehicle", rr["road"])] = usage.get(("vehicle", rr["road"]), 0) + 1
        return idx

    for kind, count, ec in [
        ("vehicle", vehicles, entity_classes["local_traffic"]),
        *[("person", n_, c_) for c_, n_ in person_classes],
    ]:
        if life and kind == "vehicle":
            plan_life_vehicles(count)
            if overlay is not None:
                overlay.paint_actors(actors, positions)
            if activity_runtime is not None:
                # Prove residents' activity access against the long-stay fleet
                # now, not only after all residents have already been admitted.
                from awr.life.civil.walking import merge_obstacles

                ids_ = [i for i, j in enumerate(jobs) if permanent_parking(j)]
                if ids_:
                    pp = np.asarray(positions)[ids_]
                    ss = np.asarray(sizes)[ids_]
                    aa = np.asarray(yaws)[ids_]
                    merge_obstacles(static_walk_free, walking.obstacle_mask(pp, ss, walk_radius, aa))
                    permanent_walk_obstacles.update(ids_)
                    if walking.turn_free is not None:
                        m = np.asarray(walking.obstacle_mask(pp, ss, max_radius, aa))
                        walking.turn_clear = walking.turn_clear & m
                        walking.turn_free = walking.turn_free & m
                    activity_runtime.invalidate_routes()
            continue
        for j in range(count):
            if j % 10 == 0:
                print("planning", kind, j, count, "admitted", len(actors), flush=True)
            key = vehicle_keys[j % len(vehicle_keys)] if kind == "vehicle" else ec.asset(j)
            pid = ec.actor_id(j) if kind == "person" else f"{kind}-{j:03d}"
            spec, mesh, _ = load(key)
            if kind == "vehicle":
                size = np.array(spec["dimensions_m"])
            else:
                height = ec.height(j)
                envelope = np.maximum(np.max(np.abs(mesh["poses"]), axis=(0, 1)), np.max(np.abs(mesh["p"]), axis=0))
                size = np.r_[2 * envelope[:2] * height / spec["dimensions_m"][2] + 0.04, height]
            cache_key = (kind, tuple(size))
            agenda = None
            departure = float((j * 73 + (13 if kind == "person" else 0)) % int(cfg.get("departure_window_s", 6)))
            if walking and kind == "person":
                if life:
                    if activity_runtime is not None:
                        departure = t_start
                        entry = overlay.person(j) if overlay is not None and ec.id == "resident" else None
                        rr, agenda, meta = activity_runtime.place_person(
                            pid,
                            positions,
                            sizes,
                            yaws,
                            np.linalg.norm(size[:2]) / 2,
                            t_start,
                            region=entry["polygon"] if entry else None,
                            places_region=entry["places_polygon"] if entry and entry["places"] == "region" else None,
                            min_kinds=entry["min_activity_kinds"] if entry else ec.min_activity_kinds,
                            role=ec.role,
                            only_purposes=ec.activities,
                        )
                        if entry and rr is None:
                            raise ValueError(
                                f"Scenario overlay {entry['key']}: no home in {entry['region']} with the required activity access"
                            )
                        if entry and meta:
                            meta["overlay"] = entry["key"]
                    elif overlay is not None and ec.id == "resident" and overlay.person(j) is not None:
                        raise ValueError("Scenario overlay persons need the activity runtime")
                    elif ec.id != "resident":
                        raise ValueError(
                            f"Entity class {ec.id} needs the activity runtime (life.behavior = activities_v1)"
                        )
                    else:
                        from awr.life.civil.itineraries import plan_daily_life

                        departure = float(t_start + rng.uniform(0.0, float(life.get("departure_spread_s", 300.0))))
                        rr, agenda, meta = plan_daily_life(
                            walking,
                            pools,
                            rng,
                            pid,
                            positions,
                            sizes,
                            yaws,
                            np.linalg.norm(size[:2]) / 2,
                            departure,
                            duration,
                            static_walk_free,
                            life,
                        )
                    if meta:
                        life_meta["persons"][pid] = meta
                elif places is not None:
                    from awr.life.civil.itineraries import plan_outdoor_visits

                    rr, agenda = plan_outdoor_visits(
                        walking,
                        places,
                        rng,
                        pid,
                        positions,
                        sizes,
                        yaws,
                        np.linalg.norm(size[:2]) / 2,
                        departure,
                        static_walk_free,
                    )
                else:
                    rr = walking.trip(
                        rng, positions=positions, sizes=sizes, radius=np.linalg.norm(size[:2]) / 2, yaws=yaws
                    )
                candidates = [rr] if rr is not None else []
            else:
                if cache_key not in cached_routes:
                    cached_routes[cache_key] = routes(surface, sem, kind, size)
                    if lane_network and kind == "vehicle":
                        cached_routes[cache_key] += lane_network.supported_routes(surface, size, min_length=35.0)
                candidates = list(cached_routes[cache_key])
                if delivery_plan is not None and kind == "vehicle":
                    # The delivery chain is reserved: a resident car parked on it
                    # would turn the delivery round into a queue.
                    from scipy.spatial import cKDTree as _KD2

                    van_tree = _KD2(jobs[delivery_plan["van_idx"]]["route"]["points"])
                    candidates = [r for r in candidates if (van_tree.query(r["points"][::4])[0] < 2.5).mean() < 0.1]
            if kind == "vehicle" and config:
                # Finite one-way trips are the only honest traffic on this support
                # (no turning space, no boundary chains). Long chains carry
                # several staggered vehicles; short fragments only when needed.
                candidates.sort(
                    key=lambda r: (
                        usage.get((kind, r["road"]), 0) // 3,
                        -len(r["points"]),
                        usage.get((kind, r["road"]), 0),
                    )
                )
            else:
                candidates.sort(key=lambda r: (usage.get((kind, r["road"]), 0), -len(r["points"])))
            chosen = None
            for rr in candidates:
                starts = (
                    [0.0]
                    if "arc" in rr
                    else np.arange(0.0, max(0.1, route_length(rr) - 15), 12.0 if kind == "vehicle" else 2.4)
                    if config
                    else (0.0, 8.0, 16.0, 24.0, 32.0)
                )
                # Distribute along the entire supported route, not just its mouth.
                if config:
                    starts = rng.permutation(starts)
                for start in starts:
                    stationary_home = activity_runtime is not None and kind == "person"
                    if not stationary_home and start > route_length(rr) - (8 if "arc" in rr else 15):
                        continue
                    xy, yaw = at(rr, start)
                    testpos = np.array(positions + [xy])
                    testyaw = np.array(yaws + [yaw])
                    testsize = np.array(sizes + [size])
                    if collision_pairs(testpos, testyaw, testsize):
                        continue
                    # Do not start in another actor's braking zone.
                    clearance = [k for k, a in enumerate(actors) if a["class"] == kind]
                    if clearance and np.min(np.linalg.norm(np.asarray(positions)[clearance] - xy, axis=1)) < (
                        12.0 if kind == "vehicle" else 1.8
                    ):
                        continue
                    chosen = (rr, start, xy, yaw)
                    break
                if chosen:
                    break
            if chosen is None:
                rejected.append({"kind": kind, "requested_index": j, "reason": "no_supported_free_route"})
                continue
            rr, start, xy, yaw = chosen
            idx = len(actors)
            usage[(kind, rr["road"])] = usage.get((kind, rr["road"]), 0) + 1
            actor = {
                "idx": idx,
                "id": pid,
                "class": kind,
                "scripted": True,
                "profile": kind,
                "size_mm": list(np.round(size * 1000).astype(int)),
                "appearance": {"asset": key + "@1", "asset_sha256": spec["npz_sha256"]},
                "role": ec.role,
                "vtype": key if kind == "vehicle" else "",
                "semantic_type": key if kind == "vehicle" else (ec.semantic_type or "pedestrian"),
                "description": key,
            }
            if cfg.get("visual_identity"):
                from awr.life.appearance.identity import clothing, plate_text

                if kind == "person":
                    actor["appearance"].update(
                        clothing(overlay.outfit(j, j) if overlay is not None and ec.id == "resident" else j)
                    )
                    actor["description"] = (
                        f"{ec.description or '成年人'}，{actor['appearance']['top_name']}上衣、{actor['appearance']['bottom_name']}裤子"
                    )
                else:
                    from awr.life.civil.population import CAR_COLOURS, VEHICLE_TYPES

                    rgb, _, name = CAR_COLOURS[j % len(CAR_COLOURS)]
                    actor["appearance"].update(
                        body_rgb=list(rgb), colour_name=name, plate_text=plate_text(j), plate_synthetic=True
                    )
                    actor["description"] = f"{name}{VEHICLE_TYPES[key][2]}"
            actors.append(actor)
            positions.append(xy)
            yaws.append(yaw)
            sizes.append(size)
            if activity_runtime is not None and kind == "person":
                activity_runtime.indices[actor["id"]] = idx
            jobs.append(
                {
                    "route": rr,
                    "s": start,
                    "direction": 1,
                    "speed": 0.0,
                    "wait_until": departure,
                    "kind": kind,
                    "status": "waiting",
                    "goal_s": route_length(rr),
                    "max_speed": ec.max_speed(j),
                    "stuck_s": 0.0,
                    "replan_after": t_start,
                    "agenda": agenda,
                }
            )
    if delivery_plan is not None:
        dp = delivery_plan
        courier_idx = len(actors)
        van_idx = dp["van_idx"]
        actors.append(
            {
                "idx": courier_idx,
                "id": "courier-000",
                "class": "person",
                "scripted": True,
                "profile": "person",
                "size_mm": list(np.round(dp["csize"] * 1000).astype(int)),
                "appearance": dp["capp"],
                "role": "courier",
                "semantic_type": "pedestrian",
                "description": f"快递员，{dp['capp']['top_name']}上衣、{dp['capp']['bottom_name']}裤子",
            }
        )
        positions.append(dp["xy0"])
        yaws.append(dp["yaw0"])
        sizes.append(dp["csize"])
        jobs.append(
            {
                "route": {"road": "delivery-van-000", "points": np.array([dp["xy0"], dp["xy0"]])},
                "s": 0.0,
                "direction": 1,
                "speed": 0.0,
                "wait_until": duration + 1,
                "kind": "person",
                "status": "riding",
                "goal_s": 0.0,
                "max_speed": 1.2,
                "stuck_s": 0.0,
                "replan_after": t_start,
                "agenda": None,
                "riding": True,
                "container": van_idx,
                "delivery": dict(
                    role="courier",
                    van=van_idx,
                    phase="riding",
                    parcels=[f"parcel-{k:02d}" for k in range(len(dp["stops"]))],
                ),
            }
        )
        jobs[van_idx]["delivery"]["courier"] = courier_idx
    if not actors:
        raise ValueError("No valid actors")
    if life and life.get("driving"):
        # Resident drivers (awr.life.civil.activity.driving_adapter): owned cars at home berths of a driving map
        # compiled on this scene's layers, once every other body is admitted.
        admit_drivers(
            life["driving"],
            activity_runtime,
            surface,
            sem,
            walking,
            static_walk_free,
            max_radius,
            walk_radius,
            passage_check,
            actors,
            jobs,
            positions,
            yaws,
            sizes,
            add_vehicle,
            permanent_walk_obstacles,
            t_start,
            life_meta,
            parking_check=rules.evaluate if rules else lambda *a: dict(legal=None),
            overlay=overlay,
            driving_map=driving_map,
        )
        (out.parent / (out.name + ".initial-driving.json")).write_text(
            json.dumps(
                dict(
                    schema="awr.life.driver_admission_candidate.v1",
                    admitted=False,
                    scope="initial placement and home access; not a completed driver activity chain",
                    **life_meta["driving"],
                ),
                indent=2,
            )
            + "\n"
        )
    vehicle_ids = [i for i, j in enumerate(jobs) if j["kind"] == "vehicle" and "delivery" not in j]
    if vehicle_ids and config and not life:
        # Vehicles leave front to back: a vehicle never waits for its own
        # scheduled departure while blocking a vehicle behind it on the same
        # carriageway. Departures are spread over the whole window.
        window = float(cfg.get("vehicle_departure_window_s", cfg.get("departure_window_s", 6)))
        pos_arr = np.asarray(positions)
        ahead = {i: set() for i in vehicle_ids}
        for i in vehicle_ids:
            pts = jobs[i]["route"]["points"]
            si = jobs[i]["s"]
            for j in vehicle_ids:
                if j == i:
                    continue
                d = np.linalg.norm(pts - pos_arr[j], axis=1)
                k = int(np.argmin(d))
                if d[k] < 2.5 and k * 0.35 > si + 1.0:
                    ahead[i].add(j)
        order = []
        pending = set(vehicle_ids)
        while pending:
            ready = sorted(i for i in pending if not (ahead[i] & pending))
            chosen_i = ready[0] if ready else min(pending)
            order.append(chosen_i)
            pending.discard(chosen_i)
        for rank, i in enumerate(order):
            jobs[i]["wait_until"] = rank * window / len(order)
    if cfg.get("visual_overlay") is not None:
        from awr.life.civil.visual_overlay import apply as apply_visual_overlay

        actors[:] = apply_visual_overlay(actors, cfg["visual_overlay"])
    overlay_bindings = None if overlay is None else overlay.bindings(actors, activity_runtime)
    if overlay_bindings is not None:
        overlay_bindings["parked_sets_initial"] = overlay.parked_bindings(actors, positions)
        (out.parent / (out.name + ".initial-scenario-overlay.json")).write_text(
            json.dumps(overlay_bindings, ensure_ascii=False, indent=2) + "\n"
        )
    from dataclasses import asdict

    (out.parent / (out.name + ".initial-plans.json")).write_text(
        json.dumps(
            {
                "actors": actors,
                "positions": np.asarray(positions).tolist(),
                "rejected": rejected,
                "activity_admission": activity_runtime.admission_stats if activity_runtime is not None else None,
                "agendas": [asdict(j["agenda"]) for j in jobs if j.get("agenda") is not None],
            },
            indent=2,
            default=lambda x: int(x),
        )
        + "\n"
    )
    if (curb_mode or activity_runtime is not None) and rejected:
        raise ValueError(
            "Requested actors were not all admitted; inspect initial-plans.json. No partial-population tape is published."
        )
    if activity_runtime is not None:
        # Preserve the exact admitted homes and facility slots before recording.
        # Access/capacity probes must not silently reuse another population's
        # home layout when a long recording later fails a continuity gate.
        (out.parent / (out.name + ".initial-activity-places.json")).write_text(
            json.dumps([asdict(p) for p in activity_runtime.director.places.values()], ensure_ascii=False) + "\n"
        )
    if plan_only:
        if environment_binding is not None:
            after_binding = bind_inputs(
                manifest,
                world=world,
                support=nav,
                semantics=sem_path,
                config=cfg,
                root=cfg.get("environment_root", "."),
            )
            if after_binding != environment_binding:
                raise RuntimeError("Environment inputs changed during planning")
        print("Plan-only preflight complete; no tape simulated or published", flush=True)
        return
    xy = np.array(positions)
    yaw = np.array(yaws)
    sizes = np.array(sizes)
    nt = round(duration / DT)
    n = len(actors)
    tracks = np.zeros((nt, n, 2), "f4")
    vel = np.zeros_like(tracks)
    yaw_track = np.zeros((nt, n), "f4")
    state = np.full((nt, n), 2, "u1")
    events = []
    container_track = np.full((nt, n), -1, "i4")
    riding = np.array([bool(j.get("riding")) for j in jobs])
    courier_agendas = []
    vehicle_in_trip = np.zeros(len(actors), bool)

    def log(kind, i, t, **kw):
        events.append({"kind": kind, "actors": [actors[i]["id"]], "t_s": round(t, 1), "xy": xy[i].tolist(), **kw})
        if kind in ("depart", "park"):
            vehicle_in_trip[i] = kind == "depart"

    def park(i, t, purpose):
        info = rules.evaluate(xy[i], yaw[i], sizes[i]) if rules else dict(legal=None, reasons=[], where=None, slot=None)
        if jobs[i].get("motion_policy") == "autonomous_activity":
            # The activity backend owns these berths; curb_v1's per-car finite
            # route slots do not. Still require the declared pose and the same
            # scene parking rules, without snapping a body into position.
            vehicle = activity_runtime.director.vehicles[actors[i]["id"]]
            info = resident_parking_evidence(activity_runtime.driving_routes.graph, vehicle.berth, xy[i], yaw[i], info)
        elif curb_mode:
            slots = jobs[i].get("curb_berths") or []
            slot = next(
                (
                    s
                    for s in slots
                    if np.linalg.norm(xy[i] - s["xy"]) < 0.1
                    and abs((yaw[i] - s["yaw"] + math.pi) % (2 * math.pi) - math.pi) < 0.05
                ),
                None,
            )
            if slot is None or not info.get("legal"):
                raise ValueError(f"Unadmitted parking pose for {actors[i]['id']} at {t}: {info}")
            if "parking_lot" in slot:
                info.update(
                    where="parking_lot",
                    slot=slot["bay"],
                    verified_parking_lot_bay=dict(
                        xy=slot["xy"], yaw=slot["yaw"], access_evidence=slot["access_evidence"]
                    ),
                )
            else:
                info["verified_curb_berth"] = dict(xy=slot["xy"], yaw=slot["yaw"])
        log("park", i, t, purpose=purpose, **info)
        if purpose == "destination_parking" and t > t_start:
            settle_parked(np.array([i]))

    for i, j in enumerate(jobs):
        if j["kind"] == "vehicle":
            park(i, t_start, j.get("initial_purpose", "initial_roadside"))

    def delivery_step(i, job, t):
        """Delivery chain state machine. Returns True when generic motion is skipped this tick."""
        d = job["delivery"]
        if d["role"] == "van":
            if d["phase"] == "driving":
                if d["next"] < len(d["stops"]) and abs(job["goal_s"] - job["s"]) < 0.03:
                    d["phase"] = "stopped"
                    job["speed"] = 0.0
                    job["status"] = "arrived"
                    park(i, t, "delivery_stop")
                    cj = jobs[d["courier"]]
                    cj["delivery"]["phase"] = "alight_pending"
                    cj["replan_after"] = t + 1.0
                    return True
                return False
            job["speed"] = 0.0
            return True
        van = jobs[d["van"]]
        dv = van["delivery"]
        if d["phase"] == "riding":
            return True
        if d["phase"] == "alight_pending":
            if t < job["replan_after"]:
                return True
            job["replan_after"] = t + 1.0
            stop = dv["stops"][dv["next"]]
            q = np.asarray(stop["alight_xy"])
            a = float(stop["yaw"])
            active = np.flatnonzero(~riding)
            cell = walking.cell(q)
            free = bool(static_walk_free[tuple(cell)]) and bool(
                surface.supported(np.vstack([rectangle(q, a, sizes[i], margin=0.12), q[None]]), "person").all()
            )
            if not free or collision_pairs(
                np.vstack([old[active], q[None]]), np.r_[yaw[active], a], np.vstack([sizes[active], sizes[i][None]])
            ):
                if not d.get("alight_blocked_logged"):
                    log("alight_blocked", i, t, stop=dv["next"])
                    d["alight_blocked_logged"] = True
                return True
            # Declared physical transition: the body appears beside the van door.
            d.pop("alight_blocked_logged", None)
            new[i] = q
            new_yaw[i] = a
            old[i] = q
            job["riding"] = False
            riding[i] = False
            transition.add(i)
            from awr.life.civil.agenda import Agenda, Visit

            parcel = d["parcels"][dv["next"]]
            job["agenda"] = Agenda(
                actors[i]["id"],
                "courier",
                [
                    Visit(
                        stop["place"],
                        tuple(stop["place_xy"]),
                        "deliver_" + parcel,
                        float(cfg["delivery"].get("stop_dwell_s", 20.0)),
                    ),
                    Visit(f"delivery-van-000-door-{dv['next']}", tuple(q.tolist()), "return_to_van", 0.0),
                ],
                departure_s=t,
            )
            courier_agendas.append(job["agenda"])
            job.update(
                route={"road": "courier", "points": np.array([q, q]), "arc": np.array([0.0, 0.0])},
                s=0.0,
                goal_s=0.0,
                needs_route=True,
                replan_after=t,
                wait_until=t,
                status="waiting",
                speed=0.0,
                stuck_s=0.0,
            )
            log(
                "alight",
                i,
                t,
                vehicle=actors[d["van"]]["id"],
                stop=dv["next"],
                place=stop["place"],
                parcel=parcel,
                door_to_kerb_m=stop.get("door_to_kerb_m"),
            )
            d["phase"] = "delivering"
            d["parcel"] = parcel
            return True
        if d["phase"] == "delivering":
            ag = job["agenda"]
            if not d.get("delivered_logged") and any(
                e["kind"] == "visit_completed" and e["visit_index"] == 0 for e in ag.events
            ):
                log(
                    "delivered",
                    i,
                    t,
                    parcel=d["parcel"],
                    place=dv["stops"][dv["next"]]["place"],
                    vehicle=actors[d["van"]]["id"],
                )
                d["delivered_logged"] = True
            if ag.phase != "finished":
                return False
            d["phase"] = "board_pending"
            job["agenda"] = None
            d["board_pending_since"] = t
        if d["phase"] == "board_pending":
            if np.linalg.norm(old[i] - np.asarray(dv["stops"][dv["next"]]["alight_xy"])) > 0.3:
                return True
            d.pop("delivered_logged", None)
            d.pop("board_pending_since", None)
            job["riding"] = True
            riding[i] = True
            new[i] = new[d["van"]]
            new_yaw[i] = new_yaw[d["van"]]
            old[i] = new[i]
            transition.add(i)
            log("board", i, t, vehicle=actors[d["van"]]["id"], stop=dv["next"])
            d["phase"] = "riding"
            dv["next"] += 1
            if dv["next"] < len(dv["stops"]):
                van["goal_s"] = dv["stops"][dv["next"]]["s"]
                van["depart_purpose"] = "next_delivery_stop"
            else:
                van["goal_s"] = dv["route_m"]
                van["depart_purpose"] = "end_of_round"
            dv["phase"] = "driving"
            van["status"] = "waiting"
            van["wait_until"] = t + 3.0
            return True
        return True

    radii = np.linalg.norm(sizes[:, :2], axis=1) / 2
    profiles = np.array(["vehicle" if j["kind"] == "vehicle" else "person" for j in jobs])
    person_mask = np.array([j["kind"] == "person" for j in jobs])
    circle_radii = np.where(person_mask, radii, np.nan)
    # Shared streets: walkers pass each other as the oriented bodies of the
    # collision check (half body width plus the planning margins), not as
    # turning disks; against static geometry the turning disk is kept.
    shared = bool(cfg.get("shared_streets")) and walking is not None
    plan_radius = np.where(person_mask & shared, sizes[:, 1] / 2, radii)
    # Body footprints. Shared streets: people occupy the footprint of the pose they
    # are drawn in at the speed of the tick (rest pose when standing, blended walk
    # cycle when walking; awr.life.civil.body), vehicles and objects their catalogue
    # boxes. `sizes` stays the outer bound of every pose (catalogue, admission).
    # Otherwise every body keeps its outer box.
    ext = bodies.actor_extents(actors, sizes) if shared else np.stack([bodies.centred(sz) for sz in sizes])
    rest_r = np.where(person_mask, bodies.turn_radius(ext[:, 0]), radii)
    rest_box = np.column_stack([bodies.symmetric(ext[:, 0]), sizes[:, 2]])
    walk_ext = ext[:, 1]
    prev_speed = np.zeros(n)
    route_history = [
        {"actor": actors[i]["id"], "t_s": 0.0, "reason": "initial_trip", "waypoints": j["route"]["points"].tolist()}
        for i, j in enumerate(jobs)
    ]
    corridor_cache = {"t": None}

    def vehicle_corridors():
        # The stretch a vehicle in its trip drives through next: from its front,
        # its stopping distance plus 6 m (6-12 m, at most to its goal), cut at
        # the first vehicle or street object ahead, VEHICLE_PERSON_CLEARANCE_M
        # wider than the vehicle on each side; as 1 m boxes along its route.
        # Walkers plan around it: they pass beside or behind a vehicle that is
        # about to drive on and do not walk or stand in front of it. Someone
        # already inside a vehicle's stretch plans without it (walks on ahead of
        # the vehicle or past it; standing there, the vehicle asks to give way).
        if corridor_cache["t"] == t:
            return corridor_cache["boxes"]
        from awr.life.civil.collision import obb_overlap
        from awr.life.civil.prediction import route_at_many

        cc, cs, cy, co = [], [], [], []
        rigid = np.flatnonzero(~person_mask & ~riding)
        for v in vehicle_ids:
            vj = jobs[v]
            if vj["status"] not in ("moving", "blocked") or t < vj["wait_until"] or riding[v]:
                continue
            ahead = min(min(12.0, vj["speed"] ** 2 / (2 * 1.8) + 6.0), abs(vj["goal_s"] - vj["s"]))
            if ahead < 0.5:
                continue
            d = np.arange(0.5, ahead + 1e-6, 1.0)
            pts, angs = route_at_many(vj["route"], vj["s"] + vj["direction"] * (sizes[v, 0] / 2 + d))
            if vj["direction"] < 0:
                angs = angs + math.pi
            near = rigid[(rigid != v) & (np.linalg.norm(old[rigid] - old[v], axis=1) < ahead + sizes[v, 0] + 12.0)]
            if len(near):
                hit = obb_overlap(
                    pts[:, None, :],
                    angs[:, None],
                    np.array([1.0, sizes[v, 1], 0.0]),
                    old[near][None],
                    yaw[near][None],
                    sizes[near][None],
                ).any(1)
                if hit.any():
                    pts, angs = pts[: int(np.argmax(hit))], angs[: int(np.argmax(hit))]
            cc += list(pts)
            cy += list(angs)
            cs += [[1.1, sizes[v, 1] + 2 * VEHICLE_PERSON_CLEARANCE_M, 0.0]] * len(pts)
            co += [v] * len(pts)
        boxes = (
            np.asarray(cc, float).reshape(-1, 2),
            np.asarray(cs, float).reshape(-1, 3),
            np.asarray(cy, float),
            np.asarray(co, int),
        )
        corridor_cache.update(t=t, boxes=boxes)
        return boxes

    def dynamic_free_for(i):
        ids = np.arange(n) != i
        if shared and person_mask[i]:
            sz = cur_sizes.copy()
            for b, until in list(jobs[i].get("avoid", {}).items()):
                if until < t:
                    del jobs[i]["avoid"][b]
                else:
                    sz[b, :2] += 0.6
            cc, cs, cy, co = vehicle_corridors()
            near = np.linalg.norm(cc - old[i], axis=1) < 40.0
            if near.any():
                # stretches this walker is already inside (within its planning clearance)
                d = old[i] - cc
                c_, s_ = np.cos(cy), np.sin(cy)
                loc = np.abs(np.stack([d[:, 0] * c_ + d[:, 1] * s_, -d[:, 0] * s_ + d[:, 1] * c_], -1))
                inside = (
                    np.linalg.norm(np.maximum(loc - cs[:, :2] / 2, 0), axis=1)
                    <= plan_radius[i] + 0.08 + PLANNING_MARGIN_M
                )
                if inside.any():
                    near &= ~np.isin(co, co[inside])
            free = walking.obstacle_mask(
                np.vstack([cur_centres[ids], cc[near]]),
                np.vstack([sz[ids], cs[near]]),
                plan_radius[i],
                np.r_[yaw[ids], cy[near]],
            )
        else:
            free = walking.obstacle_mask(old[ids], sizes[ids], radii[i], yaw[ids], circle_radii=circle_radii[ids])
        free &= static_walk_free
        free.static_free = np.asarray(static_walk_free)
        return free

    def oncoming_conflict(i):
        # A walking person meets another one coming the other way on nearly the
        # same line (within 0.9 m sideways, 0.8-6 m ahead).
        h = yaw[i]
        fwd = np.array([math.cos(h), math.sin(h)])
        right = np.array([math.sin(h), -math.cos(h)])
        for j in tree.query_ball_point(old[i], 6.5):
            if j == i or not person_mask[j] or riding[j] or speeds_now[j] <= 0.3 or math.cos(yaw[j] - h) >= -0.7:
                continue
            rel = old[j] - old[i]
            a = float(rel @ fwd)
            if 0.8 < a < 6.0 and abs(float(rel @ right)) < 0.9:
                return True
        return False

    def keep_right_route(i, free):
        # Both walkers move to the right of their direction of travel and pass
        # left shoulder to left shoulder, as people do on a pavement: a shift of
        # 0.6-0.8 m over the next 3-3.5 m, then 4 m straight on. The bends stay
        # below the 0.25 rad that would make a walker stop and turn on the spot.
        h = yaw[i]
        fwd = np.array([math.cos(h), math.sin(h)])
        right = np.array([math.sin(h), -math.cos(h)])
        for ahead, lat in ((3.5, 0.6), (3.5, 0.8), (3.0, 0.6)):
            q1 = old[i] + fwd * ahead + right * lat
            q2 = q1 + fwd * 4.0
            if walking.clear_segment(old[i], q1, free) and walking.clear_segment(q1, q2, free):
                return polyline([old[i], q1, q2], "keep_right")
        return None

    def settle_parked(ids_):
        # A vehicle with no further registered motion is static street geometry
        # for walkers. Future departure eligibility cannot depend on tape length.
        if not shared or not len(ids_):
            return
        ids_ = np.asarray(
            [
                i
                for i in ids_
                if i not in permanent_walk_obstacles and jobs[i].get("motion_policy") != "autonomous_activity"
            ],
            int,
        )
        if not len(ids_):
            return
        from awr.life.civil.walking import merge_obstacles

        m = walking.obstacle_mask(
            xy[ids_], sizes[ids_], float(plan_radius[person_mask].max()) if person_mask.any() else max_radius, yaw[ids_]
        )
        merge_obstacles(static_walk_free, m)
        permanent_walk_obstacles.update(ids_)
        if activity_runtime is not None:
            activity_runtime.invalidate_routes()
        if walking.turn_free is not None:
            m = np.asarray(walking.obstacle_mask(xy[ids_], sizes[ids_], max_radius, yaw[ids_]))
            walking.turn_clear = walking.turn_clear & m
            walking.turn_free = walking.turn_free & m

    if shared:
        # A late departure is still a future departure even if this particular
        # recording ends first. Treating it as permanent changes walking routes
        # when only the requested recording duration changes.
        settle_parked(
            np.array(
                [
                    i
                    for i, j in enumerate(jobs)
                    if (
                        permanent_parking(j)
                        if activity_runtime is not None
                        else j["kind"] == "vehicle"
                        and j.get("wait_until", 0.0) >= duration
                        and j.get("after") is None
                        and j["speed"] == 0.0
                    )
                ],
                int,
            )
        )
    still_cache = {"key": None, "free": None}

    def static_now():
        # Walking space without every vehicle standing still now (parked, waiting
        # to depart or arrived): guides go around a stationary car, as people do,
        # instead of through a gap beside it that the car leaves too narrow.
        if not shared:
            return static_walk_free
        ids_ = np.array([v for v in vehicle_ids if jobs[v]["speed"] == 0.0 and jobs[v]["status"] != "moving"], int)
        key = tuple(ids_.tolist())
        if still_cache["key"] != key:
            # One cell (0.5 m) more than the passability clearance: a gap of one or
            # two cells beside a car (often a diagonal staircase along a kerb) is
            # passable for the planning disk but not for an upright walking body.
            m = walking.obstacle_mask(old[ids_], sizes[ids_], walk_radius + STATIONARY_CAR_ROOM_M, yaw[ids_])
            m &= static_walk_free
            m.static_free = np.asarray(static_walk_free)
            still_cache.update(key=key, free=m)
        return still_cache["free"]

    navigation_profile = {"guided_calls": 0, "guided_wall_s": 0.0}

    def guided_route(i, dest, free):
        if activity_runtime is None:
            return _guided_route(i, dest, free)
        began = time.perf_counter()
        try:
            return _guided_route(i, dest, free)
        finally:
            navigation_profile["guided_calls"] += 1
            navigation_profile["guided_wall_s"] += time.perf_counter() - began

    def _guided_route(i, dest, free):
        # Destination-preserving local route; guides first around stationary
        # vehicles, then over the static walking space alone.
        for base in (static_now(), static_walk_free):
            room = with_travel_room(base)
            if room is not None:
                rr, temporary = rolling_route(walking, old[i], dest, room, with_travel_room(free))
                if rr is not None:
                    return rr, temporary
            rr, temporary = rolling_route(walking, old[i], dest, base, free)
            if rr is not None:
                return rr, temporary
            if not shared:
                break
        return None, False

    def request_courtesy(i, t):
        # A traveller that has not progressed for several seconds asks the
        # standing residents on its intended path (waiting to depart, queueing,
        # servicing or finished) to step aside. Standing is not a right of way over a
        # corridor; a visit is paused honestly, never faked or shortened.
        guide = None
        for base in (static_now(), static_walk_free):
            room = with_travel_room(base)
            guide = walking.plan(old[i], jobs[i]["agenda"].destination.xy, room) if room is not None else None
            if guide is None:
                guide = walking.plan(old[i], jobs[i]["agenda"].destination.xy, base)
            if guide is not None:
                break
        if guide is None:
            return
        pts = guide["points"][: max(2, int((guide["arc"] <= 10.0).sum()) + 1)]
        asked = 0
        for j in np.argsort(np.linalg.norm(old - old[i], axis=1)):
            if j == i or not person_mask[j] or jobs[j].get("agenda") is None:
                continue
            if np.linalg.norm(old[j] - old[i]) > 10.0:
                break
            aj = jobs[j]["agenda"]
            if (
                aj.paused_at is not None
                or aj.phase not in ("scheduled", "queueing", "servicing", "finished")
                or t < jobs[j].get("courtesy_cooldown", -math.inf)
            ):
                continue
            need = plan_radius[i] + (sizes[j, 1] / 2 if shared else radii[j]) + 0.08 + PLANNING_MARGIN_M + 0.3
            if distance_to_polyline(pts, old[j]) > need:
                continue
            rr = step_aside_route(walking, old[j], pts, dynamic_free_for(j), need)
            jobs[j]["courtesy_cooldown"] = t + 20.0
            if rr is None:
                log("step_aside_unavailable", j, t, requester=actors[i]["id"])
                continue
            if not aj.pause(t, "obstructs_" + actors[i]["id"]):
                continue
            jobs[j].update(
                route=rr,
                s=0.0,
                goal_s=route_length(rr),
                direction=1,
                speed=0.0,
                stuck_s=0.0,
                temporary_endpoint=False,
                needs_route=False,
                courtesy=dict(stage="aside", requester=i, return_xy=old[j].copy(), asked_t=t),
                wait_until=t,
                replan_after=t,
            )
            if j < len(next_s):
                next_s[j] = 0.0
            log("stepped_aside", j, t, requester=actors[i]["id"], refuge=rr["points"][-1].tolist())
            route_history.append(
                {
                    "actor": actors[j]["id"],
                    "t_s": round(t, 1),
                    "reason": "step_aside",
                    "requester": actors[i]["id"],
                    "waypoints": rr["points"].tolist(),
                }
            )
            asked += 1
            if asked >= 3:
                break

    def request_way(i, t):
        # A vehicle held up inside its trip asks the people standing in its swept
        # corridor (next 15 m of its route) to give way: each steps aside, or
        # back to the nearest passing place within 20 m, waits until the vehicle
        # has passed and then goes on. Only people at walking pace below 0.3 m/s
        # are asked (no abrupt stop); a visit is paused honestly, never shortened.
        job = jobs[i]
        job["ask_after"] = t + 5.0
        ss = job["s"]
        ahead = min(15.0, abs(job["goal_s"] - ss))
        if ahead < 1.0:
            return
        d = np.arange(0.0, ahead + 0.25, 0.5)
        pts = np.array([at(job["route"], ss + job["direction"] * x)[0] for x in d])
        from awr.life.civil.local_navigation import distance_to_polyline, retreat_route

        for j in np.argsort(np.linalg.norm(old - old[i], axis=1)):
            if np.linalg.norm(old[j] - old[i]) > 18.0:
                break
            if j == i or not person_mask[j] or riding[j] or jobs[j].get("agenda") is None:
                continue
            aj = jobs[j]["agenda"]
            if (
                aj.paused_at is not None
                or "courtesy" in jobs[j]
                or jobs[j]["speed"] > 0.3
                or t < jobs[j].get("courtesy_cooldown", -math.inf)
            ):
                continue
            need = sizes[i, 1] / 2 + sizes[j, 1] / 2 + 0.08 + PLANNING_MARGIN_M + 0.3
            if distance_to_polyline(pts, old[j]) > need:
                continue
            free = dynamic_free_for(j)
            rr = step_aside_route(walking, old[j], pts, free, need) or retreat_route(
                walking, old[j], pts, free, need, max_distance=20.0
            )
            jobs[j]["courtesy_cooldown"] = t + 10.0
            if rr is None:
                log("give_way_unavailable", j, t, vehicle=actors[i]["id"])
                continue
            if not aj.pause(t, "gives_way_to_" + actors[i]["id"]):
                continue
            along = float(d[int(np.argmin(np.linalg.norm(pts - old[j], axis=1)))])
            jobs[j].update(
                route=rr,
                s=0.0,
                goal_s=route_length(rr),
                direction=1,
                speed=0.0,
                stuck_s=0.0,
                temporary_endpoint=False,
                needs_route=False,
                yielding=False,
                extending=False,
                wait_until=t,
                replan_after=t,
                courtesy=dict(
                    stage="aside",
                    requester=i,
                    return_xy=old[j].copy(),
                    asked_t=t,
                    resume_from_refuge=aj.phase == "travelling",
                    pass_s=ss + job["direction"] * (along + sizes[i, 0] / 2 + 1.0),
                ),
            )
            log("gave_way", j, t, vehicle=actors[i]["id"], refuge=rr["points"][-1].tolist())
            route_history.append(
                {
                    "actor": actors[j]["id"],
                    "t_s": round(t, 1),
                    "reason": "give_way_to_vehicle",
                    "requester": actors[i]["id"],
                    "waypoints": rr["points"].tolist(),
                }
            )

    def courtesy_step(i, job, t):
        # Returns True when the body should follow its current route this tick.
        c = job["courtesy"]
        agenda = job["agenda"]
        remaining = abs(job["goal_s"] - job["s"])
        if c["stage"] == "aside":
            if remaining < 0.03:
                if "refuge_t" not in c:
                    c["refuge_t"] = t
                    job["stuck_s"] = 0.0
                r = c["requester"]
                ar = jobs[r].get("agenda")
                if ar is None:
                    # a vehicle: it has passed once its centre is beyond the walker's
                    # former spot by half its length plus a metre, or left its trip
                    passed = (jobs[r]["s"] - c["pass_s"]) * jobs[r]["direction"] >= 0.0
                    gone = jobs[r]["status"] != "moving"
                    wait = 30.0
                else:
                    passed = np.linalg.norm(old[r] - c["return_xy"]) > 2.5 and np.linalg.norm(old[r] - old[i]) > 2.0
                    gone = ar.phase != "travelling"
                    wait = 15.0
                if not (passed or t >= c["refuge_t"] + wait or gone):
                    job["speed"] = 0.0
                    return False
                if c.get("resume_from_refuge"):
                    # A traveller continues its own journey from the refuge.
                    agenda.resume(t)
                    log("resumed_after_giving_way", i, t, aside_s=round(t - c["asked_t"], 1))
                    del job["courtesy"]
                    job.update(speed=0.0, stuck_s=0.0, needs_route=True, replan_after=t, courtesy_cooldown=t + 10.0)
                    return False
                c["stage"] = "return"
                job["replan_after"] = t
            elif job["stuck_s"] >= 4.0 and t >= job["replan_after"]:
                c["stage"] = "return"
                job["replan_after"] = t
            else:
                return True
        if c["stage"] == "return":
            if t < job["replan_after"]:
                job["speed"] = 0.0
                return False
            job["replan_after"] = t + 2.0 + (i % 7) * 0.15
            if np.linalg.norm(old[i] - c["return_xy"]) < 0.005:
                rr = None
            else:
                rr = walking.plan(old[i], c["return_xy"], dynamic_free_for(i))
                if rr is None:
                    if not c.get("return_blocked"):
                        log("return_blocked", i, t)
                        c["return_blocked"] = True
                    job["speed"] = 0.0
                    return False
                job.update(
                    route=rr,
                    s=0.0,
                    goal_s=route_length(rr),
                    direction=1,
                    speed=0.0,
                    stuck_s=0.0,
                    temporary_endpoint=False,
                )
                next_s[i] = 0.0
                c["stage"] = "returning"
                route_history.append(
                    {
                        "actor": actors[i]["id"],
                        "t_s": round(t, 1),
                        "reason": "return_to_place",
                        "waypoints": rr["points"].tolist(),
                    }
                )
                return True
        if c["stage"] == "returning" and abs(job["goal_s"] - job["s"]) >= 0.03:
            if job["stuck_s"] >= 4.0 and t >= job["replan_after"]:
                c["stage"] = "return"
                job["speed"] = 0.0
                return False
            return True
        agenda.resume(t)
        log("returned_to_place", i, t, aside_s=round(t - c["asked_t"], 1))
        del job["courtesy"]
        job["speed"] = 0.0
        job["stuck_s"] = 0.0
        job["courtesy_cooldown"] = t + 20.0
        return False

    # Life gates that can be decided while recording (awr.life.civil.life_audit,
    # same definitions): a build that has already failed one stops here with a
    # diagnostic and its partial tracks instead of running to the end. The full
    # audit, including group blockages and trip lengths, runs before publishing.
    gates = cfg.get("life_gates") or {}

    def gate_value(name, key="threshold"):
        g = gates.get(name)
        if g is None:
            return None
        if not isinstance(g, dict):
            return float(g) if key == "threshold" else None
        return float(g[key]) if key in g else None

    stall_limit = gate_value("person_longest_stall_s_max")
    stop_limits = [
        v
        for v in (
            gate_value("vehicle_longest_stop_in_trip_s_max"),
            gate_value("vehicle_ineffective_trips_max", "max_stop_s"),
        )
        if v is not None
    ]
    stop_limit = min(stop_limits) if stop_limits else None
    board_limit = gate_value("courier_board_after_stop_s_max")
    stall_run = np.zeros(n, int)
    stop_run = np.zeros(n, int)

    def life_gate_failure(k, t, gate, limit, rows):
        near = []
        for r in rows:
            i = r.pop("_i")
            d = np.linalg.norm(xy - xy[i], axis=1)
            close = [j for j in np.argsort(d)[1:9] if d[j] < 6.0]
            r.update(
                position=np.round(xy[i], 2).tolist(),
                status=jobs[i]["status"],
                last_mode=jobs[i].get("_mode"),
                last_freeze=jobs[i].get("_freeze"),
                agenda_phase=jobs[i]["agenda"].phase if jobs[i].get("agenda") is not None else None,
                nearby=[
                    dict(
                        actor=actors[j]["id"],
                        distance_m=round(float(d[j]), 2),
                        speed_mps=round(float(jobs[j]["speed"]), 2),
                        status=jobs[j]["status"],
                        last_mode=jobs[j].get("_mode"),
                    )
                    for j in close
                ],
            )
            near.append(r)
        diagnostic = {
            "tick": k,
            "time_s": round(t, 1),
            "gate": gate,
            "threshold": limit,
            "failures": near,
            "partial": out.name + ".partial.npz",
            "note": "Recorded ticks 0..tick; float32 positions (m), velocities (m/s), yaw (deg), state.",
        }
        np.savez_compressed(
            out.parent / (out.name + ".partial.npz"),
            xy=tracks[: k + 1].astype(np.float32),
            v=vel[: k + 1].astype(np.float32),
            yaw=yaw_track[: k + 1].astype(np.float32),
            state=state[: k + 1],
            ids=np.array([a["id"] for a in actors]),
        )
        (out.parent / (out.name + ".partial-events.jsonl")).write_text("".join(json.dumps(e) + "\n" for e in events))
        (out.parent / (out.name + ".life-failure.json")).write_text(
            json.dumps(diagnostic, indent=2, default=float) + "\n"
        )
        error = ValueError(
            "Life gate failed during recording (" + gate + "): " + json.dumps(diagnostic, default=float)[:2000]
        )
        # Preserve the failing actors for the diagnostic runner. `bad` in the
        # outer build frame can still refer to an earlier, empty motion check.
        error.diagnostic = diagnostic
        raise error

    warm_prev_v = None
    warm_prev_skip = np.zeros(n, bool)
    simulation_began = time.perf_counter()
    life_ctx = None
    from awr.life.runtimes import resolve as resolve_runtime

    runtime_name, run_life = resolve_runtime(cfg)
    if run_life is not None:
        # A registered motion backend (awr.life.runtimes; life_v1 is awr.life.motion.runtime) advances the admitted
        # world; the loop below is skipped. Assembly, final QA, life audit and publishing follow.
        life_ctx = dict(
            cfg=cfg,
            jobs=jobs,
            actors=actors,
            n=n,
            out=out,
            warm=warm,
            nt=nt,
            tracks=tracks,
            vel=vel,
            yaw_track=yaw_track,
            state=state,
            container_track=container_track,
            events=events,
            occupancy=occupancy,
            rest_r=rest_r,
            xy=xy,
            yaw=yaw,
            ext=ext,
            surface=surface,
            walking=walking,
            activity_runtime=activity_runtime,
            driving_binding=getattr(activity_runtime, "driving_binding", None),
            dynamic_validation=dynamic_validation,
            walk_radius=walk_radius if walking is not None else None,
        )
        xy, yaw = run_life(life_ctx)
    departure_reservations = None
    if reactive_vehicles:
        from awr.life.civil.traffic_admission import DepartureReservations

        departure_reservations = DepartureReservations(
            {
                actors[i]["id"]: (job["route"], sizes[i])
                for i, job in enumerate(jobs)
                if job["kind"] == "vehicle" and job["goal_s"] > 0.03
            }
        )
    for k in range(-round(warm / DT), nt) if life_ctx is None else ():
        t = k * DT
        old = xy.copy()
        new = xy.copy()
        next_s = []
        new_yaw = yaw.copy()
        transition = set()
        # Reserve each moving body's stopping corridor. Using only its present
        # rectangle reacts too late when pedestrians and cars converge.
        speeds_now = np.array([j["speed"] for j in jobs])
        stopping = speeds_now**2 / (2 * 1.8) + np.where(speeds_now > 0.01, 0.35, 0.0)
        at_rest = speeds_now <= 0.01
        # Present footprints: the pose drawn at the larger of the last recorded
        # speed and the current speed.
        cur_centres, cur_sizes = bodies.boxes(old, yaw, np.maximum(prev_speed, speeds_now) + 0.01, ext, sizes[:, 2])
        if reactive_vehicles:
            from awr.life.civil.vehicle_execution import braking_direction

            motion_sign = np.array([braking_direction(job) for job in jobs])
        else:
            motion_sign = np.ones(n)
        obstacles = cur_centres + np.column_stack([np.cos(yaw), np.sin(yaw)]) * (motion_sign * stopping)[:, None] / 2
        obstacle_sizes = cur_sizes.copy()
        # Pedestrians can rotate while another actor is approaching. Reserve
        # their complete turning envelope before adding their braking corridor;
        # reserving only the present narrow torso reacts too late to that turn.
        pedestrian_ids = np.array([i for i, j in enumerate(jobs) if j["kind"] == "person"], int)
        if shared:
            # Vehicles (the only users of these boxes on shared streets) also keep
            # VEHICLE_PERSON_CLEARANCE_M from every side of a person.
            obstacle_sizes[pedestrian_ids, :2] = (
                np.maximum(cur_sizes[pedestrian_ids, :2], 2 * rest_r[pedestrian_ids, None])
                + 2 * VEHICLE_PERSON_CLEARANCE_M
            )
        elif walking:
            obstacle_sizes[pedestrian_ids, :2] = 2 * radii[pedestrian_ids, None]
        obstacle_sizes[:, 0] += stopping
        obstacle_radii = np.linalg.norm(obstacle_sizes[:, :2], axis=1) / 2
        if shared:
            body_sizes = cur_sizes.copy()
            body_sizes[:, 0] += stopping
            body_radii = np.linalg.norm(body_sizes[:, :2], axis=1) / 2
        # Riders are inside their vehicle: not obstacles, not checked as bodies.
        obstacles[riding] = 1e6
        tree = cKDTree(obstacles)
        if shared:
            for i in vehicle_ids:
                vj = jobs[i]
                if (
                    vj["status"] == "moving"
                    and vj.get("stuck_s", 0.0) >= 2.0
                    and t >= vj.get("ask_after", -math.inf)
                    and t >= vj["wait_until"]
                ):
                    request_way(i, t)
        if activity_runtime is not None:
            activity_runtime.update_frame(old, rest_r, riding)
        if shared:
            # Select retreats against one common body frame. Fixed ID priority
            # alone traps a pair when the preferred yielder has no space (e.g.
            # against a wall), and static-only selection repeatedly walks into
            # a third person even when another retreat direction is clear.
            eligible = np.array(
                [
                    j["kind"] == "person"
                    and j.get("agenda") is not None
                    and j["agenda"].phase == "travelling"
                    and j["agenda"].paused_at is None
                    and j.get("backstep") is None
                    and j.get("stuck_s", 0.0) >= BACKSTEP_AFTER_S
                    and t >= max(j["wait_until"], j.get("yield_until", -math.inf))
                    and speeds_now[i] <= 0.05
                    and prev_speed[i] <= 0.05
                    and not riding[i]
                    for i, j in enumerate(jobs)
                ],
                bool,
            )
            if eligible.any():
                from awr.life.civil.crowd_recovery import choose_backsteps

                began = time.perf_counter()
                active = {
                    i: (j["backstep"][0], max(0.0, j["backstep"][1]))
                    for i, j in enumerate(jobs)
                    if j.get("backstep") is not None
                }
                selected = choose_backsteps(
                    old,
                    yaw,
                    np.maximum(speeds_now, prev_speed),
                    ext,
                    cur_centres,
                    cur_sizes,
                    person_mask,
                    eligible,
                    lambda i, end: walking.clear_segment(old[i], end, static_now()),
                    active=active,
                    near_m=BACKSTEP_NEAR_M,
                    distance=BACKSTEP_M,
                    max_speed=BACKSTEP_MPS,
                    dt=DT,
                )
                navigation_profile["recovery_calls"] = navigation_profile.get("recovery_calls", 0) + 1
                navigation_profile["recovery_wall_s"] = (
                    navigation_profile.get("recovery_wall_s", 0.0) + time.perf_counter() - began
                )
                for i, choice in selected.items():
                    jobs[i]["backstep"] = (
                        choice["direction"],
                        choice["distance_m"],
                        t + choice["distance_m"] / BACKSTEP_MPS + 2.0,
                    )
                    jobs[i]["stuck_s"] = 0.0
                    log(
                        "step_back",
                        i,
                        t,
                        other=actors[choice["other"]]["id"],
                        distance_m=round(float(np.linalg.norm(old[i] - old[choice["other"]])), 2),
                        selection="available_space",
                        candidate_angle=choice["candidate_angle"],
                        priority_fallback=choice["priority_fallback"],
                    )
        for i, job in enumerate(jobs):
            ss = job["s"]
            next_s.append(ss)
            if "delivery" in job and delivery_step(i, job, t):
                continue
            agenda = job.get("agenda")
            if (
                agenda is not None
                and site_slots is not None
                and agenda.paused_at is None
                and agenda.phase == "scheduled"
                and t >= agenda.departure_s
            ):
                # Setting off: a site visit takes a free standing slot now; the
                # slot held at the previous stay is released when the body leaves it.
                v = agenda.visits[agenda.index]
                bound = None
                if v.place.startswith("site:"):
                    from awr.life.civil.itineraries import bind_site_visit

                    others = np.flatnonzero((np.arange(n) != i) & ~riding)
                    bound, slot = bind_site_visit(
                        v, site_slots[v.place[5:]], slot_holder, old[i], old[others], rest_r[others], rest_r[i]
                    )
                    for alt in [] if bound is not None else site_alternatives.get(v.place[5:], []):
                        # Same facility and purpose, another approach: people stand at the
                        # other side of a full forecourt or park rather than wait at home.
                        bound, slot = bind_site_visit(
                            v, site_slots[alt], slot_holder, old[i], old[others], rest_r[others], rest_r[i]
                        )
                        if bound is not None:
                            log("alternative_approach", i, t, planned=v.place[5:], site=alt)
                            break
                    if bound is None:
                        agenda.departure_s = t + float(life.get("full_site_retry_s", 15.0))
                        if not job.get("site_full_logged"):
                            log("site_full", i, t, site=v.place[5:])
                            job["site_full_logged"] = True
                if bound is not None or not v.place.startswith("site:"):
                    for held in job.pop("held_slots", []):
                        slot_holder.pop(held, None)
                if bound is not None:
                    agenda.visits[agenda.index] = bound
                    slot_holder[slot] = i
                    job["held_slots"] = [slot]
                    job["needs_route"] = True
                    job["replan_after"] = t
                    job.pop("site_full_logged", None)
                    log("slot_bound", i, t, site=v.place[5:], slot=slot)
            if agenda is not None:
                try:
                    phase = agenda.tick(t, old[i], job["speed"], occupancy, arrival_radius=0.05)
                except RuntimeError as exc:
                    if activity_runtime is not None:
                        diagnostic = dict(
                            t_s=t,
                            actor=actors[i]["id"],
                            position=old[i].tolist(),
                            error=str(exc),
                            events=activity_runtime.director.events[-100:],
                            resources=activity_runtime.director.holders,
                        )
                        (out.parent / (out.name + ".activity-failure.json")).write_text(
                            json.dumps(diagnostic, indent=2) + "\n"
                        )
                    raise
                if phase == "paused":
                    if not courtesy_step(i, job, t):
                        continue
                    ss = job["s"]
                    next_s[i] = ss
                    phase = "courtesy"
                if phase in ("wait", "finished"):
                    job["speed"] = 0.0
                    job["stuck_s"] = 0.0
                    continue
                if phase == "next_destination":
                    job["needs_route"] = True
                # In shared-street mode a long temporary leg is extended before its
                # end, so that walking does not stop at every local horizon.
                near = (
                    2.5
                    if (
                        cfg.get("shared_streets")
                        and not job.get("yielding")
                        and route_length(job["route"]) >= 6.0
                        and t >= job.get("extend_after", -math.inf)
                    )
                    else 0.03
                )
                if phase != "courtesy" and job.get("temporary_endpoint") and abs(job["goal_s"] - ss) < near:
                    job["needs_route"] = True
                    job["temporary_endpoint"] = False
                    job["extending"] = abs(job["goal_s"] - ss) >= 0.03
                    if not job["extending"]:
                        job["speed"] = 0.0
                    job["replan_after"] = t
                    if job.pop("yielding", False):
                        job["yield_until"] = t + job.pop("yield_hold", 2.0)
                        log("yield_refuge_reached", i, t)
                if t < job.get("yield_until", -math.inf):
                    continue
                if job.get("needs_route") and t < job["replan_after"]:
                    continue
                if (
                    shared
                    and job["kind"] == "person"
                    and phase == "travel"
                    and job["speed"] > 0.3
                    and not job.get("yielding")
                    and not job.get("needs_route")
                    and not job.get("extending")
                    and "arc" in job["route"]
                    and t >= job.get("keep_right_after", -math.inf)
                    and oncoming_conflict(i)
                ):
                    job["keep_right_after"] = t + 3.0
                    rr = keep_right_route(i, dynamic_free_for(i))
                    if rr is not None:
                        keep = min(
                            job["speed"], math.sqrt(2 * 1.2 * float(np.linalg.norm(rr["points"][1] - rr["points"][0])))
                        )
                        job["_mode"] = (round(t, 1), f"keep_right:keep={keep:.2f}")
                        job.update(
                            route=rr,
                            s=0.0,
                            goal_s=route_length(rr),
                            direction=1,
                            speed=keep,
                            stuck_s=0.0,
                            needs_route=False,
                            temporary_endpoint=True,
                            yielding=False,
                        )
                        next_s[i] = 0.0
                        ss = 0.0
                        route_history.append(
                            {
                                "actor": actors[i]["id"],
                                "t_s": round(t, 1),
                                "reason": "keep_right",
                                "place": agenda.destination.place,
                                "waypoints": rr["points"].tolist(),
                            }
                        )
                # Shared streets: a walker held up by a body ahead looks for a way
                # around after one second (people sidestep, they do not queue);
                # courtesy requests and yielding still wait four seconds.
                detour_after = 1.0 if cfg.get("shared_streets") and job["kind"] == "person" else 4.0
                if job.get("backstep") is not None:
                    d_, left, until = job["backstep"]
                    if left <= 0.005 or t >= until:
                        from awr.life.civil.route_handoff import finish_backstep

                        finish_backstep(job, old[i], yaw[i], t, max_speed=BACKSTEP_MPS)
                        ss = 0.0
                        next_s[i] = 0.0
                        log("step_back_completed", i, t, position=old[i].tolist(), destination=agenda.destination.place)
                    else:
                        v = min(BACKSTEP_MPS, left / DT, job["speed"] + 1.0 * DT)
                        new[i] = old[i] + d_ * v * DT
                        new_yaw[i] = yaw[i]
                        job["speed"] = v
                        job["_mode"] = (round(t, 1), "step_back")
                        job["backstep"] = (d_, left - v * DT, until)
                        continue
                if phase != "courtesy" and (
                    job.get("needs_route") or (job["stuck_s"] >= detour_after and t >= job["replan_after"])
                ):
                    # Global intent uses static geometry. Only local progress
                    # needs to be free of temporary people right now.
                    free = dynamic_free_for(i)
                    stalled = job["stuck_s"] >= 4.0
                    if stalled and t >= job.get("ask_after", -math.inf):
                        job["ask_after"] = t + 6.0
                        request_courtesy(i, t)
                    rr, temporary = guided_route(i, agenda.destination.xy, free)
                    yielding = False
                    if rr is None and stalled:
                        rr = yield_route(walking, old[i], yaw[i], free, i, old, rest_r, person_mask)
                        temporary = yielding = rr is not None
                        if rr is None and shared and job["stuck_s"] >= 6.0:
                            # Head-on in a passage too narrow for both: the walker
                            # with lower priority (as in yield_route) goes back to the
                            # nearest passing place off the other's intended path.
                            near_ids = [
                                o
                                for o in np.argsort(np.linalg.norm(old - old[i], axis=1))[1:4]
                                if person_mask[o]
                                and np.linalg.norm(old[o] - old[i]) < 3.5
                                and jobs[o].get("agenda") is not None
                                and jobs[o]["agenda"].phase == "travelling"
                            ]
                            if near_ids and i > near_ids[0]:
                                o = near_ids[0]
                                guide_o = walking.plan(old[o], jobs[o]["agenda"].destination.xy, static_walk_free)
                                if guide_o is not None:
                                    from awr.life.civil.local_navigation import retreat_route

                                    pts = guide_o["points"][: max(2, int((guide_o["arc"] <= 30.0).sum()) + 1)]
                                    rr = retreat_route(
                                        walking,
                                        old[i],
                                        pts,
                                        free,
                                        plan_radius[i] + sizes[o, 1] / 2 + 0.08 + PLANNING_MARGIN_M + 0.3,
                                    )
                                    if rr is not None:
                                        temporary = yielding = True
                                        job["yield_hold"] = 6.0
                                        log(
                                            "retreat_to_passing_place",
                                            i,
                                            t,
                                            other=actors[o]["id"],
                                            route_length_m=round(route_length(rr), 1),
                                        )
                    job["replan_after"] = t + 2.0 + (i % 7) * 0.15

                    def continues(rr):
                        # The new route continues the current direction of travel:
                        # its first segment is at least 0.5 m long and within 0.2 rad
                        # of both the current route heading and the body heading
                        # (the body may still lag a bend of the current route; the
                        # in-place rotation rule compares the route with the body).
                        if len(rr["points"]) < 2:
                            return False
                        _, heading = at(job["route"], ss)
                        d0 = rr["points"][1] - rr["points"][0]
                        h = math.atan2(d0[1], d0[0])
                        return float(np.linalg.norm(d0)) >= 0.5 and all(
                            abs((h - a + math.pi) % (2 * math.pi) - math.pi) <= 0.2 for a in (heading, yaw[i])
                        )

                    if rr is not None and job.get("extending"):
                        # Extend only onto a route that continues the current
                        # direction of travel; otherwise finish the current leg,
                        # braking to its end, and turn there as usual.
                        if (
                            yielding
                            or not continues(rr)
                            or math.sqrt(2 * 1.2 * float(np.linalg.norm(rr["points"][1] - rr["points"][0])))
                            < job["speed"] - 1.8 * DT
                        ):
                            rr = None
                    if rr is None and job.pop("extending", False):
                        # The extension failed: keep walking the current temporary
                        # leg (it brakes to its end) and try again in a second.
                        job.update(needs_route=False, temporary_endpoint=True, extend_after=t + 1.0, replan_after=t)
                    elif rr is None:
                        agenda.blocked(t, "destination_temporarily_unreachable")
                        if job["speed"] > 0.7:
                            job["_freeze"] = (round(t, 1), "route_unavailable_at_speed")
                        job["needs_route"] = True
                        job["speed"] = 0.0
                        continue
                    else:
                        job.pop("extending", None)
                        keep = 0.0
                        reason_hint = "yield" if yielding else "temporary" if temporary else "destination"
                        if (
                            cfg.get("shared_streets")
                            and not yielding
                            and len(rr["points"]) > 1
                            and job["speed"] > 0
                            and "arc" in job["route"]
                        ):
                            # A walking person who replans onto a route that continues in
                            # the current direction of travel keeps walking, no faster
                            # than allows stopping at the end of the first segment.
                            if continues(rr):
                                keep = min(
                                    job["speed"],
                                    math.sqrt(2 * 1.2 * float(np.linalg.norm(rr["points"][1] - rr["points"][0]))),
                                )
                        if job["speed"] - keep > 0.7:
                            job["_freeze"] = (round(t, 1), "replan_turn_at_speed:" + reason_hint)
                        if job["speed"] > 0.05:
                            job["_mode"] = (round(t, 1), f"replan:{reason_hint}:keep={keep:.2f}")
                        job.update(
                            route=rr,
                            s=0.0,
                            goal_s=route_length(rr),
                            direction=1,
                            speed=keep,
                            stuck_s=0.0,
                            needs_route=False,
                            temporary_endpoint=temporary,
                            yielding=yielding,
                        )
                        next_s[i] = 0.0
                        ss = 0.0
                        reason = (
                            "yield_to_neighbour"
                            if yielding
                            else "local_progress"
                            if temporary
                            else "agenda_destination"
                        )
                        if yielding:
                            log(reason, i, t, destination=agenda.destination.place)
                        route_history.append(
                            {
                                "actor": actors[i]["id"],
                                "t_s": round(t, 1),
                                "reason": reason,
                                "place": agenda.destination.place,
                                "waypoints": rr["points"].tolist(),
                            }
                        )
            if job.get("after") is not None:
                ahead = jobs[job["after"]]
                if ahead.get("arrived_t") is None:
                    continue
                job["wait_until"] = max(job["wait_until"], ahead["arrived_t"] + job.get("after_gap", 20.0))
                job["after"] = None
            if t < job["wait_until"]:
                continue
            remaining = abs(job["goal_s"] - ss)
            if (
                walking
                and agenda is None
                and job["kind"] == "person"
                and (remaining < 0.03 or job["stuck_s"] >= 4.0)
                and t >= job["replan_after"]
            ):
                arrived = remaining < 0.03
                if arrived and job["status"] != "arrived":
                    log("goal_reached", i, t, road=job["route"]["road"])
                    job["status"] = "arrived"
                ids = np.arange(n) != i
                rr = None
                if not arrived:
                    free = walking.obstacle_mask(old[ids], sizes[ids], radii[i], yaw[ids])
                    rr = walking.plan(old[i], job["route"]["points"][-1], free)
                if arrived or rr is None:
                    rr = walking.trip(
                        rng, start=old[i], positions=old[ids], sizes=sizes[ids], radius=radii[i], yaws=yaw[ids]
                    )
                job["replan_after"] = t + 6.0 + (i % 7) * 0.4
                if rr is None:
                    if arrived:
                        job["speed"] = 0.0
                    log("route_unavailable", i, t, reason="no_reachable_clear_destination")
                    continue
                job.update(
                    route=rr,
                    s=0.0,
                    goal_s=route_length(rr),
                    direction=1,
                    speed=0.0,
                    stuck_s=0.0,
                    wait_until=t + (float(rng.uniform(*cfg.get("walking_dwell_s", [8.0, 8.0]))) if arrived else 0.5),
                    status="waiting",
                )
                next_s[i] = 0.0
                reason = "next_outdoor_waypoint" if arrived else "blocked_route_replan"
                route_history.append(
                    {"actor": actors[i]["id"], "t_s": round(t, 1), "reason": reason, "waypoints": rr["points"].tolist()}
                )
                log(reason, i, t, goal_xy=rr["points"][-1].tolist(), route_length_m=round(route_length(rr), 2))
                continue
            if remaining < 0.03 and (job.get("_vehicle_execution") is None or job["_vehicle_execution"].arrived):
                if job["status"] != "arrived":
                    log("goal_reached", i, t, road=job["route"]["road"])
                    job["status"] = "arrived"
                    job["arrived_t"] = t
                    if job["kind"] == "vehicle":
                        park(i, t, job.get("arrival_purpose", "end_of_trip"))
                        if departure_reservations is not None and actors[i]["id"] in departure_reservations.active:
                            departure_reservations.release(actors[i]["id"])
                            log("traffic_reservation_released", i, t, reason="measured_destination_parking")
                job["speed"] = 0.0
                if job["kind"] == "person" and not walking:
                    job["direction"] *= -1
                    job["goal_s"] = 0.0 if job["direction"] < 0 else (len(job["route"]["points"]) - 1) * 0.35
                    job["wait_until"] = t + 8.0
                    job["status"] = "waiting"
                    log("waiting", i, t, duration_s=8.0)
                continue
            if reactive_vehicles and job["kind"] == "vehicle" and job.get("_vehicle_execution") is None:
                if not departure_reservations.request(actors[i]["id"], t):
                    if not job.get("_traffic_waiting"):
                        log("traffic_departure_wait", i, t, reason="conflicting_admitted_journey", at="source_berth")
                        job["_traffic_waiting"] = True
                    continue
                job.pop("_traffic_waiting", None)
                log("traffic_reservation_acquired", i, t, policy="whole_journey_swept_corridor")
            desired_p, desired_yaw = at(job["route"], ss + job["direction"] * (1e-6 if "arc" in job["route"] else 0.1))
            if job["direction"] < 0:
                desired_yaw = (desired_yaw + math.pi) % (2 * math.pi)
            dy = (desired_yaw - yaw[i] + math.pi) % (2 * math.pi) - math.pi
            new_yaw[i] = yaw[i] + np.clip(dy, -1.5 * DT, 1.5 * DT)
            # A walking pedestrian whose route bends by a moderate angle (a replan
            # or bend the controller did not anticipate) slows into the turn instead
            # of stopping dead: speed is cut to its component along the new
            # direction and the body turns at most 0.25 rad per tick.
            walk_turn = shared and job["kind"] == "person" and job["speed"] > 0.05 and 0.25 < abs(dy) <= 0.6
            if abs(dy) > 0.25 and job["kind"] == "person" and not walk_turn:
                if job["speed"] > 0.7:
                    job["_freeze"] = (round(t, 1), "rotate_in_place_at_speed")
                if job["speed"] > 0.05:
                    job["_mode"] = (round(t, 1), f"rotate_in_place:{dy:.2f}")
                job["speed"] = 0.0
                if shared:
                    # Turn on the spot only if this turning step keeps the body clear
                    # of every walking or driving body and its braking corridor, so
                    # that the turn never forces a moving neighbour to an abrupt stop.
                    # Standing neighbours are left to the final collision check (both
                    # bodies are at rest, a refused turn is just a wait).
                    from awr.life.civil.collision import obb_overlap

                    speeds = np.array([jj["speed"] for jj in jobs])
                    mv = np.flatnonzero((speeds > 0.05) & (np.arange(n) != i) & ~riding)
                    mv = mv[np.linalg.norm(obstacles[mv] - old[i], axis=1) < rest_r[i] + body_radii[mv] + 0.1]
                    if len(mv):
                        turn = yaw[i] + np.linspace(0.0, 1.0, 4) * (new_yaw[i] - yaw[i])
                        if obb_overlap(
                            np.repeat(old[i][None], 4, 0)[:, None, :],
                            turn[:, None],
                            rest_box[i],
                            obstacles[mv][None],
                            yaw[mv][None],
                            body_sizes[mv][None],
                        ).any():
                            new_yaw[i] = yaw[i]
                continue
            maxv = min(job["max_speed"], math.sqrt(max(0.0, 2 * 1.5 * remaining)))
            if "arc" in job["route"]:
                # Brake before polyline corners, then rotate in place within the
                # admitted body disk. Do not change a walking velocity by 90°
                # in one tick merely because the route has a sharp vertex.
                arc = job["route"]["arc"]
                idx = int(np.searchsorted(arc, ss + 1e-6, side="right"))
                sharp = False
                if idx < len(arc) - 1 and job["kind"] == "person" and cfg.get("shared_streets"):
                    # Pedestrians walk through bends that the in-place rotation rule
                    # below would not stop for (heading error <= .25 rad), at most
                    # 1.4 rad/s of heading change (rotation limit 1.5 rad/s), and brake
                    # for every sharper corner within their stopping distance, not
                    # only for the next vertex.
                    p = job["route"]["points"]
                    reach = job["speed"] ** 2 / (2 * 1.2) + 0.6
                    j = idx
                    while j < len(arc) - 1 and arc[j] - ss <= reach:
                        a = p[j] - p[j - 1]
                        b = p[j + 1] - p[j]
                        turn = abs(math.atan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b)))
                        dist = max(0.0, arc[j] - ss)
                        if turn > 0.25:
                            maxv = min(maxv, math.sqrt(2 * 1.2 * dist))
                            sharp = j == idx
                            break
                        if turn > 0.02:
                            local = min(arc[j] - arc[j - 1], arc[j + 1] - arc[j]) / turn
                            maxv = min(maxv, math.sqrt((1.4 * local) ** 2 + 2 * 1.2 * dist))
                        j += 1
                elif idx < len(arc) - 1:
                    p = job["route"]["points"]
                    a = p[idx] - p[idx - 1]
                    b = p[idx + 1] - p[idx]
                    turn = abs(math.atan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b)))
                    sharp = turn > 0.15
                    if sharp:
                        maxv = min(maxv, math.sqrt(max(0.0, 2 * 1.2 * (arc[idx] - ss))))
            # Look ahead along the intended path, including full body extent,
            # to yield before contact instead of instant post-collision pushes.
            horizon = (
                max(1.5, job["speed"] ** 2 / (2 * 1.8) + 1.0)
                if shared and job["kind"] == "person"
                else max(1.0, job["speed"] ** 2 / (2 * 1.8) + 0.7)
            )
            from awr.life.civil.prediction import blocked_many, route_at_many

            distances = np.arange(0.025, horizon + 1, 0.025)
            points, angles = route_at_many(job["route"], ss + job["direction"] * distances)
            grounds = np.ones(len(distances), bool)
            if job["kind"] == "vehicle" and not job["route"].get("dense_validated"):
                # An expanded rectangle does not contain the *samples* of its
                # inner rectangle: a hole can lie beneath the actual corner
                # while all expanded corners remain on ground. Check both.
                # Routes validated densely at admission (support_mask) skip
                # this re-sampling: interpolated samples at arbitrary phases
                # can clip a raster cell corner that no fixed sampling sees,
                # and a vehicle refused that way never moves. The per-tick
                # swept-footprint check and the final QA still cover every
                # recorded pose.
                unit = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]])
                offset = np.vstack([unit * sizes[i, :2] / 2, unit * (sizes[i, :2] + 0.24) / 2])
                from awr.life.civil.contacts import vehicle_contacts

                offset = np.vstack([offset, vehicle_contacts(tuple(sizes[i]))])
                c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
                footprint = points[:, None, :] + np.stack(
                    [offset[:, 0] * c - offset[:, 1] * s, offset[:, 0] * s + offset[:, 1] * c], -1
                )
                grounds = surface.supported(footprint, "vehicle").reshape(-1, len(offset)).all(1)
            if shared and job["kind"] == "person":
                # Oriented bodies with their braking corridors; the own body is
                # 0.25 m wider and longer than the collision box, which covers a
                # neighbour's turning step (0.15 rad) during the tick.
                from awr.life.civil.prediction import blocked_ahead

                # The own body is predicted in the walking pose (its box is offset
                # forward of the body origin).
                we = walk_ext[i]
                off = (we[0] + we[1]) / 2
                wbox = np.array([we[1] - we[0], we[3] - we[2], sizes[i, 2]])
                ahead_pts = points + off * np.column_stack([np.cos(angles), np.sin(angles)])
                # The body reaches a sample with the heading of a step that began up
                # to two steps earlier (a step takes the route tangent at its start
                # and the body turns over the tick), not with the tangent there; on
                # a bend that box reaches further to the outside of the turn. Test
                # the route headings up to two steps back as well (behind the body:
                # its current heading).
                reach_step = (job["speed"] + 1.0 * DT) * DT
                nl = max(1, int(math.ceil(2 * reach_step / 0.05)))
                lag_s = distances[None, :] - np.linspace(2 * reach_step / nl, 2 * reach_step, nl)[:, None]
                _, lag_yaw = route_at_many(job["route"], (ss + job["direction"] * lag_s).ravel())
                lag_yaw = np.where(
                    lag_s < 0, yaw[i] + (0.0 if job["direction"] > 0 else math.pi), lag_yaw.reshape(lag_s.shape)
                )

                def turn_of(a, b):
                    return np.abs((a - b + math.pi) % (2 * math.pi) - math.pi)

                same = turn_of(lag_yaw, angles[None, :]) <= 1e-3
                same[1:] |= turn_of(lag_yaw[1:], lag_yaw[:-1]) <= 1e-3
                lag_yaw[same] = np.nan
                lag_pts = points[None] + off * np.stack([np.cos(lag_yaw), np.sin(lag_yaw)], -1)
                blocked, blocker = blocked_ahead(
                    i,
                    ahead_pts,
                    angles,
                    wbox + np.array([0.25, 0.25, 0.0]),
                    obstacles,
                    yaw,
                    body_sizes,
                    tree,
                    body_radii,
                    still=at_rest,
                    still_body=wbox + np.array([0.1, 0.1, 0.0]),
                    return_blocker=True,
                    headings=(lag_pts, lag_yaw),
                    starting=bool(at_rest[i]),
                )
                if blocker >= 0 and at_rest[blocker] and job["stuck_s"] >= 0.5:
                    # Planning treats the walker as a disk of half its width; its
                    # 1.2 m long box can still clip a resting body's corner on a
                    # bend. Keep further from that body when replanning (10 s).
                    job.setdefault("avoid", {})[int(blocker)] = t + 10.0
                # Ground ahead. Walking cells guarantee support for half the body
                # width only; the walking pose reaches further forward. Where only
                # a slow pose fits on supported ground the walker slows to
                # CREEP_MPS in time (people slow down along an edge), and it stops
                # before ground where even that pose does not fit, instead of being
                # stopped dead at walking pace by the per-tick support check.
                pp, aa, dd = points[::4], angles[::4], distances[::4]

                def fits(v):
                    bc, bs = bodies.boxes(pp, aa, np.full(len(pp), v), np.repeat(ext[i][None], len(pp), 0))
                    off = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])[None] * bs[:, None, :2] / 2
                    c_, s_ = np.cos(aa)[:, None], np.sin(aa)[:, None]
                    cr = bc[:, None, :] + np.stack(
                        [off[..., 0] * c_ - off[..., 1] * s_, off[..., 0] * s_ + off[..., 1] * c_], -1
                    )
                    return surface.supported(cr.reshape(-1, 2), "person").reshape(-1, 4).all(1)

                ground_cap = math.inf
                full = fits(bodies.FULL_WALK_MPS + 0.01)
                if not full.all():
                    ground_cap = math.sqrt(CREEP_MPS**2 + 2 * 1.8 * max(0.0, dd[int(np.argmin(full))] - 0.3))
                    slow = fits(2 * CREEP_MPS + 0.02)
                    if not slow.all():
                        ground_cap = min(ground_cap, math.sqrt(2 * 1.8 * max(0.0, dd[int(np.argmin(slow))] - 0.3)))
            elif walking and job["kind"] == "person":
                from awr.life.civil.prediction import blocked_disks

                future_circles = np.where(person_mask, radii + stopping / 2, np.nan)
                blocked = blocked_disks(
                    i, points, radii[i], obstacles, yaw, obstacle_sizes, tree, obstacle_radii, future_circles
                )
            else:
                blocked = ~grounds | blocked_many(
                    i, points, angles, sizes[i], obstacles, yaw, obstacle_sizes, tree, obstacle_radii
                )
            if blocked.any():
                dist = distances[np.flatnonzero(blocked)[0]]
                maxv = min(maxv, math.sqrt(max(0.0, 2 * 1.8 * (dist - 0.65))))
            if curb_mode and job["kind"] == "vehicle":
                cause = None
                if maxv < 0.2 and blocked.any():
                    from awr.life.civil.collision import obb_overlap

                    b = int(np.flatnonzero(blocked)[0])
                    hit = obb_overlap(points[b], angles[b], sizes[i], obstacles, yaw, obstacle_sizes)
                    ids = [int(q) for q in np.flatnonzero(hit) if q != i and not riding[q]]
                    cause = ("unsupported_ahead" if not grounds[b] else "give_way", tuple(actors[q]["id"] for q in ids))
                previous = job.get("yield_cause")
                if cause != previous:
                    if previous is not None:
                        log(
                            "yield_end",
                            i,
                            t,
                            reason=previous[0],
                            blockers=list(previous[1]),
                            duration_s=round(t - job["yield_since"], 1),
                        )
                    if cause is not None:
                        log("yield_start", i, t, reason=cause[0], blockers=list(cause[1]))
                        job["yield_since"] = t
                    job["yield_cause"] = cause
            if shared and job["kind"] == "person":
                maxv = min(maxv, ground_cap)
            if walk_turn:
                maxv = min(maxv, job["speed"] * math.cos(dy))
            if reactive_vehicles and job["kind"] == "vehicle":
                from awr.life.civil.vehicle_execution import advance_job

                stopping_arc = (
                    max(ss, ss + float(distances[np.flatnonzero(blocked)[0]]) - 0.65) if blocked.any() else None
                )
                proposal = advance_job(job, old[i], yaw[i], DT, stop_before_arc_m=stopping_arc)
                next_s[i] = proposal["arc_m"]
                new[i] = proposal["xy"]
                new_yaw[i] = proposal["yaw"]
                speed = proposal["speed_mps"]
                job["speed"] = speed
                job["_mode"] = (round(t, 1), "vehicle_" + proposal["mode"])
                if job["status"] != "moving" and speed > 0.05:
                    log("trip_started", i, t, goal_distance_m=round(remaining, 2))
                    job["status"] = "moving"
                    log("depart", i, t, purpose=job.get("depart_purpose", "local_trip"))
                continue
            # Comfortable braking is 1.8 m/s2. When a blocker is found too late for
            # it (a body entering the path at close range), brake harder, up to
            # EMERGENCY_DECEL (below the motion limits), so that the body stops in
            # time instead of being held dead at walking pace by the swept check.
            floor = job["speed"] - 1.8 * DT
            if maxv < floor:
                floor = max(maxv, job["speed"] - EMERGENCY_DECEL[job["kind"]] * DT)
            speed = np.clip(maxv, floor, job["speed"] + 1.0 * DT)
            if walk_turn:
                speed = min(speed, job["speed"] * math.cos(dy))
            ds = min(remaining, speed * DT)
            if "arc" in job["route"] and sharp and idx < len(arc):
                ds = min(ds, max(0.0, arc[idx] - ss))
            next_s[i] = ss + job["direction"] * ds
            new[i], aa = at(job["route"], next_s[i])
            new_yaw[i] = (
                aa
                if job["route"].get("smooth_vehicle_heading")
                else desired_yaw
                if "arc" in job["route"]
                else aa + (math.pi if job["direction"] < 0 else 0.0)
            )
            if walk_turn:
                new_yaw[i] = yaw[i] + np.clip(dy, -0.25, 0.25)
                job["_mode"] = (round(t, 1), f"walk_turn:{dy:.2f}")
            if job["status"] != "moving" and speed > 0.05:
                log("trip_started", i, t, goal_distance_m=round(remaining, 2))
                job["status"] = "moving"
                if job["kind"] == "vehicle":
                    log("depart", i, t, purpose=job.get("depart_purpose", "local_trip"))
            job["speed"] = speed

        # Check support during turns as well as translation. A square body has
        # a larger diagonal sweep than its forward footprint. Footprints over the
        # tick: the pose drawn at the larger of the previous and the new recorded
        # speed (footprints grow with speed, so this contains both).
        def swept_speed():
            return np.maximum(prev_speed, np.linalg.norm(new - old, axis=1) / DT) + 0.01

        support_ok = np.ones(n, bool)
        sw = swept_speed()
        for u in np.linspace(0, 1, 6):
            p = old + (new - old) * u
            aa = yaw + ((new_yaw - yaw + math.pi) % (2 * math.pi) - math.pi) * u
            bc, bs = bodies.boxes(p, aa, sw, ext)
            offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])[None, :, :] * bs[:, None, :2] / 2
            c, s = np.cos(aa)[:, None], np.sin(aa)[:, None]
            corners = bc[:, None, :] + np.stack(
                [offsets[:, :, 0] * c - offsets[:, :, 1] * s, offsets[:, :, 0] * s + offsets[:, :, 1] * c], -1
            )
            for profile in ("person", "vehicle"):
                ids = np.flatnonzero(profiles == profile)
                support_ok[ids] &= surface.supported(corners[ids], profile).reshape(-1, 4).all(1) & surface.supported(
                    p[ids], profile
                )
            if walking:
                ids = np.array([i for i, j in enumerate(jobs) if j["kind"] == "person"], int)
                rc = np.floor((p[ids] - walking.origin) / walking.step + 0.5).astype(int)[:, ::-1]
                support_ok[ids] &= walking.free[rc[:, 0], rc[:, 1]]
        support_ok[riding] = True

        def swept_supported(i, q1, a1, v):
            # support test of one body over the tick (same samples as above)
            for u in np.linspace(0, 1, 6):
                p = old[i] + (q1 - old[i]) * u
                aa = yaw[i] + ((a1 - yaw[i] + math.pi) % (2 * math.pi) - math.pi) * u
                bc, bs = bodies.boxes(p[None], np.array([aa]), np.array([max(prev_speed[i], v) + 0.01]), ext[i][None])
                off = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * bs[0, :2] / 2
                cr = bc[0] + np.stack(
                    [
                        off[:, 0] * math.cos(aa) - off[:, 1] * math.sin(aa),
                        off[:, 0] * math.sin(aa) + off[:, 1] * math.cos(aa),
                    ],
                    -1,
                )
                if not (surface.supported(cr, profiles[i]).all() and surface.supported(p[None], profiles[i]).all()):
                    return False
                if walking and jobs[i]["kind"] == "person":
                    rc = np.floor((p - walking.origin) / walking.step + 0.5).astype(int)[::-1]
                    if not walking.free[rc[0], rc[1]]:
                        return False
            return True

        for i in np.flatnonzero(~support_ok):
            job = jobs[i]
            if shared and job["kind"] == "person" and "arc" in job["route"] and job.get("backstep") is None:
                # A shorter step along the same route first (as for a collision):
                # the end of the full step may reach past supported ground while
                # a shorter one stays on it, within the motion limits.
                v0 = abs(next_s[i] - job["s"]) / DT
                stepped = False
                for decel in PARTIAL_DECEL:
                    v = v0 - decel * DT
                    if v * DT < 0.002:
                        break
                    s1 = job["s"] + job["direction"] * v * DT
                    q1 = at(job["route"], s1)[0]
                    if swept_supported(i, q1, yaw[i], v):
                        new[i] = q1
                        new_yaw[i] = yaw[i]
                        next_s[i] = s1
                        job["speed"] = v
                        job["_mode"] = (round(t, 1), f"partial_step_support:{decel:g}")
                        stepped = True
                        break
                if stepped:
                    continue
            if job["speed"] > 0.7:
                job["_freeze"] = (round(t, 1), "unsupported_swept_footprint")
            new[i] = old[i]
            new_yaw[i] = yaw[i]
            next_s[i] = job["s"]
            job["speed"] = 0.0
            if job["status"] != "blocked":
                log("action_failed", i, t, reason="unsupported_swept_footprint")
            job["status"] = "blocked"
        # Conservative swept-body checks at 5 substeps. A bad initial state is
        # an error; a blocked proposal waits in place, retaining its identity.
        active = np.flatnonzero(~riding)
        for _ in range(4 * n + 4):
            pairs = set()
            sw = swept_speed()
            for u in np.linspace(0, 1, 6):
                q = old + (new - old) * u
                angles = yaw + ((new_yaw - yaw + math.pi) % (2 * math.pi) - math.pi) * u
                bc, bs = bodies.boxes(q, angles, sw, ext, sizes[:, 2])
                for i, j in collision_pairs(bc[active], angles[active], bs[active]):
                    pairs.add((int(active[i]), int(active[j])))
            if not pairs:
                break
            # One side of each colliding pair gives way per pass. A body that can
            # stop within the motion limits yields when the other cannot (a walker
            # starting off beside a passer-by waits). Otherwise the body that
            # closes the gap (its step towards the other is larger) yields: a
            # walker approached from behind or from the side is not slowed into
            # the approaching body. When both close alike (head-on), the body
            # that moves less shortens or holds its step first; the other is held
            # back only if the pair still collides on the next pass. A pair where
            # only one body moves resolves with that body; a pair where neither
            # moves is an overlap.
            step_m = np.linalg.norm(new - old, axis=1)
            turned = np.abs((new_yaw - yaw + math.pi) % (2 * math.pi) - math.pi) > 1e-9

            def moves(a):
                return bool(step_m[a] > 0 or turned[a])

            def closing(a, b):
                d = old[b] - old[a]
                dn = float(np.linalg.norm(d))
                return float(np.dot(new[a] - old[a], d / dn)) if dn > 1e-9 else float(step_m[a])

            def yielder(a, b):
                sa, sb = (step_m[i] / DT <= MOTION_LIMIT[jobs[i]["kind"]] for i in (a, b))
                if sa != sb:
                    return a if sa else b
                ca, cb = closing(a, b), closing(b, a)
                if abs(ca - cb) > CLOSING_TIE_M:
                    return a if ca > cb else b
                return a if step_m[a] <= step_m[b] else b

            def partial_step(i):
                # A shorter step along the same route, mildest deceleration first
                # (all inside the motion limit together with a bend of the route).
                job = jobs[i]
                step = abs(next_s[i] - job["s"]) if i < len(next_s) else 0.0
                part = job.get("_partial")
                if part is None or part[0] != k:
                    part = (k, 0, step / DT)
                if not (
                    shared
                    and job["kind"] == "person"
                    and "arc" in job["route"]
                    and job.get("backstep") is None
                    and part[1] < len(PARTIAL_DECEL)
                    and part[2] - PARTIAL_DECEL[part[1]] * DT > 0.02
                ):
                    return False
                v = part[2] - PARTIAL_DECEL[part[1]] * DT
                job["_partial"] = (k, part[1] + 1, part[2])
                next_s[i] = job["s"] + job["direction"] * v * DT
                new[i] = at(job["route"], next_s[i])[0]
                new_yaw[i] = yaw[i]
                job["speed"] = v
                job["_mode"] = (round(t, 1), f"partial_step:{PARTIAL_DECEL[part[1]]:g}")
                return True

            blocked = set()
            partners = {}
            for a, b in pairs:
                partners.setdefault(a, set()).add(b)
                partners.setdefault(b, set()).add(a)
                if moves(a) and moves(b):
                    blocked.add(yielder(a, b))
                elif moves(a) or moves(b):
                    blocked.add(a if moves(a) else b)
                else:
                    blocked.update((a, b))
            changed = False
            for i in blocked:
                if np.any(new[i] != old[i]) or new_yaw[i] != yaw[i]:
                    changed = True
                job = jobs[i]
                step = abs(next_s[i] - job["s"]) if i < len(next_s) else 0.0
                part = job.get("_partial")
                if part is None or part[0] != k:
                    part = (k, 0, step / DT)
                if partial_step(i):
                    continue
                if part[2] / DT > MOTION_LIMIT[job["kind"]]:
                    # Holding would exceed the motion limit: a moving partner that
                    # can still shorten its step does so first.
                    if any(moves(b) and partial_step(b) for b in sorted(partners[i])):
                        changed = True
                        continue
                job["_mode"] = (round(t, 1), "collision_hold")
                if max(job["speed"], part[2]) > 0.7:
                    job["_freeze"] = (
                        round(t, 1),
                        "collision_hold:" + ",".join(sorted(actors[b]["id"] for b in partners[i]))[:120],
                    )
                new[i] = old[i]
                new_yaw[i] = yaw[i]
                next_s[i] = jobs[i]["s"]
                jobs[i]["speed"] = 0.0
            if not changed:
                raise ValueError("Initial body overlap cannot be repaired by teleportation")
        for i, job in enumerate(jobs):
            if reactive_vehicles:
                from awr.life.civil.vehicle_execution import confirm_job

                confirm_job(job, new[i], new_yaw[i], next_s[i])
            job["s"] = next_s[i]
            if "stuck_s" in job:
                job["stuck_s"] = (
                    job["stuck_s"] + DT if t >= job["wait_until"] and np.linalg.norm(new[i] - old[i]) < 0.002 else 0.0
                )
        for i in np.flatnonzero(riding):
            c = jobs[i]["container"]
            new[i] = new[c]
            new_yaw[i] = new_yaw[c]
        prev_speed = np.linalg.norm(new - old, axis=1) / DT
        prev_speed[riding] = 0.0
        prev_speed[list(transition)] = 0.0
        xy = new
        yaw = new_yaw
        if k < 0:
            if k % 500 == 0:
                print(
                    "warm-up tick", k, "moving", int((np.linalg.norm(new - old, axis=1) / DT > 0.1).sum()), flush=True
                )
            if cfg.get("early_motion_gate", False) and shared:
                # The warm-up is not recorded but must obey the same motion limits.
                v_now = (new - old) / DT
                v_now[list(transition)] = 0.0
                skip = riding | warm_prev_skip
                if warm_prev_v is not None:
                    acceleration = (
                        np.linalg.norm((np.round(v_now * 100) - np.round(warm_prev_v * 100)) * 0.01, axis=1) / DT
                    )
                    limits = np.array([6.0 if j["kind"] == "vehicle" else 8.0 for j in jobs])
                    bad = np.flatnonzero((acceleration > limits) & ~skip)
                    if len(bad):
                        diagnostic = {
                            "tick": k,
                            "time_s": t,
                            "phase": "warm-up",
                            "failures": [
                                {
                                    "actor": actors[i]["id"],
                                    "acceleration_mps2": float(acceleration[i]),
                                    "position": xy[i].tolist(),
                                    "status": jobs[i]["status"],
                                    "last_freeze": jobs[i].get("_freeze"),
                                    "last_mode": jobs[i].get("_mode"),
                                    "v_prev": np.round(warm_prev_v[i], 3).tolist(),
                                    "v_now": np.round(v_now[i], 3).tolist(),
                                }
                                for i in bad
                            ],
                        }
                        (out.parent / (out.name + ".motion-failure.json")).write_text(
                            json.dumps(diagnostic, indent=2) + "\n"
                        )
                        raise ValueError("Early motion QA failed: " + json.dumps(diagnostic))
                warm_prev_v = v_now
                warm_prev_skip = riding.copy()
                warm_prev_skip[list(transition)] = True
            continue
        tracks[k] = xy
        vel[k] = (new - old) / DT
        yaw_track[k] = np.degrees(yaw)
        for i in transition:
            vel[k, i] = 0.0
        state[k, np.linalg.norm(vel[k], axis=1) > 0.025] = 1
        state[k, riding] = 3
        for i in np.flatnonzero(riding):
            container_track[k, i] = jobs[i]["container"]
        if k > 1 and cfg.get("early_motion_gate", False):
            # Same velocity quantisation as tape encoding; fail early on a long
            # build instead of spending another twenty simulated minutes on an
            # already-invalid prefix. Final independent QA still checks all frames.
            acceleration = np.linalg.norm((np.round(vel[k] * 100) - np.round(vel[k - 1] * 100)) * 0.01, axis=1) / DT
            limits = np.array([6.0 if j["kind"] == "vehicle" else 8.0 for j in jobs])
            both = np.isin(state[k], [1, 2]) & np.isin(state[k - 1], [1, 2])
            bad = np.flatnonzero((acceleration > limits) & both)
            if len(bad):

                def route_near(j):
                    # route waypoints from 3 m behind to 8 m ahead of the body (polyline routes)
                    r = jobs[j].get("route") or {}
                    if "arc" not in r:
                        return None
                    keep = (r["arc"] >= jobs[j]["s"] - 3.0) & (r["arc"] <= jobs[j]["s"] + 8.0)
                    return dict(
                        s=round(float(jobs[j]["s"]), 3),
                        arc=np.round(r["arc"][keep], 3).tolist(),
                        points=np.round(r["points"][keep], 3).tolist(),
                    )

                def neighbours(i):
                    d = np.linalg.norm(xy - xy[i], axis=1)
                    return [
                        dict(
                            actor=actors[j]["id"],
                            distance_m=round(float(d[j]), 2),
                            v_prev=np.round(vel[k - 1, j], 3).tolist(),
                            v_now=np.round(vel[k, j], 3).tolist(),
                            status=jobs[j]["status"],
                            last_mode=jobs[j].get("_mode"),
                            route=route_near(j) if d[j] < 3.0 else None,
                        )
                        for j in np.argsort(d)[1:9]
                        if d[j] < 6.0
                    ]

                diagnostic = {
                    "tick": k,
                    "time_s": t,
                    "failures": [
                        {
                            "actor": actors[i]["id"],
                            "acceleration_mps2": float(acceleration[i]),
                            "position": xy[i].tolist(),
                            "status": jobs[i]["status"],
                            "last_freeze": jobs[i].get("_freeze"),
                            "last_mode": jobs[i].get("_mode"),
                            "v_prev": np.round(vel[k - 1, i], 3).tolist(),
                            "v_now": np.round(vel[k, i], 3).tolist(),
                            "route": route_near(i),
                            "nearby": neighbours(i),
                        }
                        for i in bad
                    ],
                    "partial": out.name + ".partial.npz",
                    "note": "Recorded ticks 0..tick; float32 positions (m), velocities (m/s), yaw (deg), state.",
                }
                np.savez_compressed(
                    out.parent / (out.name + ".partial.npz"),
                    xy=tracks[: k + 1].astype(np.float32),
                    v=vel[: k + 1].astype(np.float32),
                    yaw=yaw_track[: k + 1].astype(np.float32),
                    state=state[: k + 1],
                    ids=np.array([a["id"] for a in actors]),
                )
                (out.parent / (out.name + ".motion-failure.json")).write_text(
                    json.dumps(diagnostic, indent=2, default=float) + "\n"
                )
                raise ValueError("Early motion QA failed: " + json.dumps(diagnostic, default=float)[:4000])
        if gates:
            spd = np.linalg.norm(vel[k], axis=1)
            travelling = np.array(
                [
                    j.get("agenda") is not None and j["agenda"].phase == "travelling" and j["agenda"].paused_at is None
                    for j in jobs
                ]
            )
            stall_run = np.where(travelling & ~riding & (spd < 0.2), stall_run + 1, 0)
            stop_run = np.where(vehicle_in_trip & (spd < 0.2), stop_run + 1, 0)
            if stall_limit is not None and (stall_run * DT > stall_limit).any():
                life_gate_failure(
                    k,
                    t,
                    "person_longest_stall_s_max",
                    stall_limit,
                    [
                        dict(_i=int(i), actor=actors[i]["id"], stalled_s=round(float(stall_run[i] * DT), 1))
                        for i in np.flatnonzero(stall_run * DT > stall_limit)
                    ],
                )
            if stop_limit is not None and (stop_run * DT > stop_limit).any():
                life_gate_failure(
                    k,
                    t,
                    "vehicle_longest_stop_in_trip_s_max",
                    stop_limit,
                    [
                        dict(_i=int(i), actor=actors[i]["id"], stopped_s=round(float(stop_run[i] * DT), 1))
                        for i in np.flatnonzero(stop_run * DT > stop_limit)
                    ],
                )
            if board_limit is not None:
                late = [
                    i
                    for i, j in enumerate(jobs)
                    if "delivery" in j
                    and j["delivery"].get("phase") == "board_pending"
                    and t - j["delivery"]["board_pending_since"] > board_limit
                ]
                if late:
                    life_gate_failure(
                        k,
                        t,
                        "courier_board_after_stop_s_max",
                        board_limit,
                        [
                            dict(
                                _i=i,
                                actor=actors[i]["id"],
                                waiting_s=round(t - jobs[i]["delivery"]["board_pending_since"], 1),
                            )
                            for i in late
                        ],
                    )
        if k % 500 == 0:
            active_agendas = [j["agenda"] for j in jobs if j.get("agenda") is not None]
            print(
                "tick",
                k,
                "actors",
                n,
                "moving",
                int((np.linalg.norm(vel[k], axis=1) > 0.1).sum()),
                "arrived",
                sum(any(e["kind"] == "arrived" for e in a.events) for a in active_agendas),
                "finished",
                sum(a.phase == "finished" for a in active_agendas),
                flush=True,
            )
    # Start recording at the actual first state; finite difference is the canonical velocity.
    # With a warm-up, tick 0 already has a true backward difference.
    if not warm:
        vel[0] = vel[1]
    work = out.parent / (out.name + ".work")
    work.mkdir(parents=True, exist_ok=False)
    sim = work / "sim"
    sim.mkdir()
    np.savez_compressed(sim / "tracks.npz", xy=tracks, v=vel, yaw=yaw_track, state=state, container=container_track)
    (sim / "actors.json").write_text(json.dumps(actors, indent=2, default=lambda x: int(x)) + "\n")
    (sim / "sim_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    if life_ctx is not None:
        (work / "life-runtime-profile.json").write_text(json.dumps(life_ctx["life_runtime_profile"], indent=2) + "\n")
    if reactive_vehicles:
        rows = []
        for i, job in enumerate(jobs):
            execution = job.get("_vehicle_execution")
            if execution is None:
                continue
            rows.append(
                dict(
                    actor=actors[i]["id"],
                    elapsed_s=execution.time_s,
                    arc_m=execution.arc_m,
                    length_m=float(execution.route["arc"][-1]),
                    arrived=execution.arrived,
                    brake_count=execution.brake_count,
                    gear_stops_m=execution.route["required_stops_m"],
                )
            )
        (work / "vehicle-execution.json").write_text(
            json.dumps(
                dict(
                    schema="awr.life.grounded_vehicle_execution.v1",
                    model="reactive_schedule_v1",
                    note="Physical leg execution only; does not prove resident driver activity chains.",
                    vehicles=rows,
                ),
                indent=2,
            )
            + "\n"
        )
    if activity_runtime is not None:
        (work / "activity-decisions.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in activity_runtime.director.events)
        )
        (work / "activity-places.json").write_text(
            json.dumps(
                [
                    dict(id=p.id, kind=p.kind, xy=p.xy, group=p.group, parking=p.parking, access_xy=p.access_xy)
                    for p in activity_runtime.director.places.values()
                ],
                ensure_ascii=False,
            )
            + "\n"
        )
        profile = dict(
            static_routes=activity_runtime.route_stats,
            activity_decision_wall_s=activity_runtime.director.decision_wall_s,
            driving_routes=dict(activity_runtime.driving_routes.stats) if activity_runtime.driving_routes else None,
            local_guided_navigation=navigation_profile,
            simulation_wall_s=time.perf_counter() - simulation_began,
            note="Decision time includes static-route queries; guided-route timing is a subset of all navigation. Do not add nested timings. Simulation includes warm-up, excludes admission and final QA.",
        )
        (work / "activity-profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    if walking:
        (work / "walking-network.json").write_text(json.dumps(walking.summary(), indent=2) + "\n")
        (work / "route-history.json").write_text(json.dumps(route_history, indent=2) + "\n")
    agendas = [j["agenda"] for j in jobs if j.get("agenda") is not None and "delivery" not in j] + courier_agendas
    if agendas:
        from dataclasses import asdict

        (work / "agendas.json").write_text(
            json.dumps(
                [dict(**a.summary(), visits=[asdict(v) for v in a.visits], departure_s=a.departure_s) for a in agendas],
                indent=2,
            )
            + "\n"
        )
    (work / "semantics.json").write_text(json.dumps(sem))
    if life:
        life_meta["warmup_s"] = warm
        life_meta["unavailable"] = life.get("unavailable", [])
        if activity_runtime is not None:
            mixed = bool(life_ctx and life_ctx["life_runtime_profile"].get("mixed_driving"))
            life_meta["activity_engine"] = dict(
                schema="awr.life.activities.v1",
                backend="life-mixed" if mixed else "grounded-walk",
                decisions=sum(a.decisions for a in activity_runtime.director.actors.values()),
                vehicles="resident driver backend enabled; actual chains require life audit evidence"
                if mixed
                else "curb_v1; resident driver adapter unavailable",
            )
        life_meta["planning_failures"] = getattr(walking, "life_failures", {}) if walking else {}
        if pools:
            life_meta["pools"] = dict(
                sites=len(pools["sites"]),
                site_slots=sum(len(st["slots"]) for st in pools["sites"]),
                street_points=len(pools["street"]),
                household_points=sum(sp["kind"] == "household" for sp in pools["street"]),
            )
        (work / "life-plans.json").write_text(
            json.dumps(
                life_meta,
                ensure_ascii=False,
                indent=1,
                default=lambda x: float(x) if isinstance(x, np.floating) else int(x),
            )
            + "\n"
        )
    (work / "route-plans.json").write_text(
        json.dumps(
            {
                "schema": "awr.life.grounded_plans.v1",
                "life": bool(life),
                "warmup_s": warm,
                "unavailable": (life or {}).get("unavailable", []),
                "activities": (
                    [
                        "daily_chain_home_facility_household",
                        "dwell_at_measured_outdoor_waiting_point",
                        "pause_at_home_door",
                        "step_aside_and_return",
                        "resident_trip_to_destination_parking",
                        "resident_parking",
                        "park_with_intent",
                        "static_roadside_object",
                    ]
                    + (["delivery_stop", "alight", "deliver_parcel", "board"] if delivery_plan else [])
                )
                if life
                else (
                    [
                        "travel_to_mapped_outdoor_approach",
                        "wait_for_space",
                        "dwell_after_actual_arrival",
                        "travel_to_next_facility",
                        "return_and_finish",
                        "step_aside_and_return",
                    ]
                    if places
                    else ["walk_to_waypoint", "wait", "return"]
                )
                + ["drive_lane_segment", "stop", "park_with_intent", "static_roadside_object"]
                + (["delivery_stop", "alight", "deliver_parcel", "board"] if delivery_plan else []),
                "requested": {"people": people, "vehicles": vehicles, "props": props},
                "admitted": n,
                "rejected": rejected,
                "parking_rules": rules.describe() if rules else None,
                "delivery": None
                if delivery_plan is None
                else dict(
                    van="delivery-van-000",
                    courier="courier-000",
                    stops=delivery_plan["stops"],
                    boundary_exit=False,
                    boundary_note="No supported lane chain reaches the stage boundary; the round starts and ends inside the town.",
                ),
                "plans": [
                    {"actor": actors[i]["id"], "road": j["route"]["road"], "waypoints": j["route"]["points"].tolist()}
                    for i, j in enumerate(jobs)
                ],
            },
            indent=2,
        )
    )
    if public_regions is not None:
        (work / "anchors.overlay.json").write_text(json.dumps(public_regions, ensure_ascii=False, indent=2) + "\n")
    if overlay is not None:
        (work / "scenario-overlay.json").write_text(json.dumps(overlay_bindings, ensure_ascii=False, indent=2) + "\n")
    elif config and public_regions is None:
        # Public search zones: a 3x3 grid over the recorded pedestrian activity
        # extent (20 m margin), not the stage quadrants. A quadrant of the
        # 2.5 km stage cannot be swept in a 4 minute acquisition window by
        # three drones at an altitude where a person is still identifiable;
        # a public zone must be searchable for the task to be solvable.
        pm = np.array([a["class"] == "person" for a in actors])
        px = tracks[:, pm, :].reshape(-1, 2)
        x0, y0 = px.min(0) - 20.0
        x1, y1 = px.max(0) + 20.0
        xs = np.linspace(x0, x1, 4)
        ys = np.linspace(y0, y1, 4)
        names = {
            (0, 2): "西北街区",
            (1, 2): "北街区",
            (2, 2): "东北街区",
            (0, 1): "西街区",
            (1, 1): "中心街区",
            (2, 1): "东街区",
            (0, 0): "西南街区",
            (1, 0): "南街区",
            (2, 0): "东南街区",
        }
        zones = [
            (f"r{row}c{col}", names[(col, row)], float(xs[col]), float(ys[row]), float(xs[col + 1]), float(ys[row + 1]))
            for row in range(3)
            for col in range(3)
        ]
        anchors = [
            {
                "id": f"ev_{key}",
                "kind": "zone",
                "category": "manual",
                "profile": "person",
                "label_zh": label,
                "polygon": [[a, b], [c, b], [c, d], [a, d]],
                "grid": "3x3 over recorded pedestrian extent, 20 m margin",
            }
            for key, label, a, b, c, d in zones
        ]
        (work / "anchors.overlay.json").write_text(json.dumps({"anchors": anchors}, ensure_ascii=False))
    from awr.life.civil.world_reference import navigation_hashes
    from awr.sim.events.compile import raycast_sha256

    wdoc = json.loads((world / "world.json").read_text())
    ref = {
        "id": wdoc["id"],
        "content_version": wdoc["contentVersion"],
        "raycast_sha256": raycast_sha256(world),
        "navmesh_sha256": navigation_hashes(world, require_vehicle=any(a["class"] == "vehicle" for a in actors)),
    }
    from awr.life.civil.assemble import build as assemble

    pending = Path(tempfile.mkdtemp(prefix=out.name + ".pending-", dir=out.parent))
    from awr.life.envpack.execution import finish_output, mark_validation

    if dynamic_validation:
        mark_validation(pending, manifest["id"])
    assemble(
        sim,
        sem,
        {"seed": seed, "warmup_s": warm},
        {"persons": [], "vehicles": []},
        ref,
        out.parent.name,
        int(out.name.removeprefix("v")),
        pending,
        ground="grid",
        nav_dir=nav,
        world_dir=world,
        env=None,
        time_origin="2026-06-15T05:30:00Z",
        brief={
            "scope": "Validated outdoor lane/walking activities and street furniture; no unverified indoor or boarding transitions."
        },
    )
    from awr.life.civil.validate_tape import validate

    result = validate(pending, nav)
    (pending / "final_qa.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["pass"]:
        raise ValueError("Final QA failed; pending event not published")
    shutil.copy(work / "route-plans.json", pending / "route-plans.json")
    if overlay is not None:
        shutil.copy(work / "scenario-overlay.json", pending / "scenario-overlay.json")
    if reactive_vehicles:
        shutil.copy(work / "vehicle-execution.json", pending / "vehicle-execution.json")
    if activity_runtime is not None:
        for name in ("activity-decisions.jsonl", "activity-profile.json", "activity-places.json"):
            shutil.copy(work / name, pending / name)
    if (work / "agendas.json").exists():
        shutil.copy(work / "agendas.json", pending / "agendas.json")
    if (work / "life-plans.json").exists():
        shutil.copy(work / "life-plans.json", pending / "life-plans.json")
    if walking:
        for name in ("walking-network.json", "route-history.json"):
            shutil.copy(work / name, pending / name)
        shutil.copy(sim / "sim_events.jsonl", pending / "activity-events.jsonl")
    shutil.copy(ROOT_ASSETS / "catalog.json", pending / "asset-catalog.json")
    if principle_pkg is not None:
        shutil.copy(principle_pkg.root / "principle.json", pending / "principle.json")
        (pending / "principle-check.json").write_text(json.dumps(principle_report, indent=2, ensure_ascii=False) + "\n")
    if config:
        registry = [
            {
                **a,
                "size_mm": [int(v) for v in a["size_mm"]],
                "spawn_xy_m": [float(v) for v in tracks[0, i]],
                "road": jobs[i]["route"]["road"],
            }
            for i, a in enumerate(actors)
        ]
        (pending / "entities.json").write_text(
            json.dumps({"schema": "awr.life.entities.v1", "entities": registry}, ensure_ascii=False, indent=2)
        )
        shutil.copy(config, pending / "testbed-config.json")
    if life:
        # The life audit is a publish condition: a tape whose people or vehicles
        # are stuck without colliding is rejected like one that collides. The
        # rejected event is kept beside the output for diagnosis.
        from awr.life.civil.life_audit import audit as life_audit
        from awr.life.civil.life_audit import markdown as life_markdown

        report = life_audit(pending, cfg)
        report["event"] = str(out)
        (pending / "life-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float) + "\n")
        (pending / "life-audit.md").write_text(life_markdown(report))
        failed = [name for name, g in report["gates"]["gates"].items() if g["pass_"] is False or g["pass_"] is None]
        print(
            json.dumps(
                {
                    "life_gates": {
                        name: dict(value=g["value"], threshold=g["threshold"], pass_=g["pass_"])
                        for name, g in report["gates"]["gates"].items()
                    }
                },
                default=float,
            ),
            flush=True,
        )
        if failed:
            rejected = out.parent / (out.name + ".rejected")
            pending.rename(rejected)
            raise ValueError(f"Life gates failed ({', '.join(failed)}); event not published, kept at {rejected}")
    after = snapshot(Path(__file__).resolve().parents[4], world, nav, extra)
    if dynamic_validation:
        after["execution_mode"] = "dynamic_validation"
    if principle_pkg is not None:
        after["principle"] = provenance["principle"]
    if environment_binding is not None:
        after["environment_binding"] = bind_inputs(
            manifest, world=world, support=nav, semantics=sem_path, config=cfg, root=cfg.get("environment_root", ".")
        )
    if after != provenance:
        raise RuntimeError("Build inputs changed during generation; pending event not published")
    (pending / "build-inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
    target = finish_output(pending, out, dynamic_validation=dynamic_validation)
    print("diagnostic complete; not published" if dynamic_validation else "published", target, flush=True)


ROOT_ASSETS = Path(__file__).resolve().parents[1] / "appearance/catalog"
# Decelerations (m/s^2) of the shorter steps a walker tries before a collision hold.
PARTIAL_DECEL = (4.0, 6.0, 7.5)
# Guides keep this much more room (m) from stationary cars than passability needs.
STATIONARY_CAR_ROOM_M = 0.5
# Step back of a walker stalled close to another stalled person (s, m, m, m/s).
BACKSTEP_AFTER_S, BACKSTEP_NEAR_M, BACKSTEP_M, BACKSTEP_MPS = 3.0, 1.0, 0.6, 0.2
# Strongest planned braking (m/s^2), used only when comfortable braking (1.8)
# cannot stop before a blocker; motion limits are 8 (person) and 6 (vehicle),
# leaving room for the turn component of the acceleration.
EMERGENCY_DECEL = {"person": 5.5, "vehicle": 4.5, "object": 0.0}
# Recorded acceleration limits (tape QA) and the closing-speed tie of a colliding
# pair (m per tick) below which the slower body yields.
MOTION_LIMIT = {"person": 8.0, "vehicle": 6.0, "object": 0.0}
CLOSING_TIE_M = 0.005
# Lateral room a vehicle keeps from a person on a shared street (each side of the
# person's footprint or rest turning disk), as in the lookahead before this model.
VEHICLE_PERSON_CLEARANCE_M = 0.25
# Speed of a walker passing where only a slow pose fits on supported ground.
CREEP_MPS = 0.15
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--principle",
        type=Path,
        help="Scene principle package (awr.life.principle.v1); replaces --world/--nav/--semantics",
    )
    ap.add_argument("--world", type=Path)
    ap.add_argument("--nav", type=Path)
    ap.add_argument("--semantics", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=180)
    ap.add_argument("--people", type=int, default=24)
    ap.add_argument("--vehicles", type=int, default=14)
    ap.add_argument("--config", type=Path)
    ap.add_argument(
        "--plan-only", action="store_true", help="Write admission evidence without recording or publishing a tape"
    )
    ap.add_argument(
        "--dynamic-validation",
        action="store_true",
        help="Run candidate environment with all motion/life checks; retain diagnostic output only",
    )
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    build(
        a.world,
        a.nav,
        a.semantics,
        a.out,
        a.duration,
        a.people,
        a.vehicles,
        a.config,
        a.plan_only,
        a.dynamic_validation,
        principle=a.principle,
    )
