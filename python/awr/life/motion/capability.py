"""Capabilities of an environment and the explicit refusal of scenario demands it cannot meet.

The engine runs any environment that provides its objects: a support surface, a walking network and router,
Director places and activities, and for car journeys berths with access points and a driving backend. What an
environment supports is stated twice and checked against the scenario's demand at both points:

1. Declared: the environment package's manifest (configs/environments/*.json, `load_environment`) states its
   capabilities with the evidence they rest on. `check_declared` refuses a demand beyond the declaration before
   anything is built.
2. Measured: `measure` derives the capabilities from the admitted objects through the interfaces the engine uses
   at run time (the Director's route backends and the walking router's static components), and `require` refuses
   a demand beyond the measurement, or a declaration the measurement does not support.

Capabilities:

- walk: residents whose static walking component holds places of at least `walk_kinds` activity kinds of their
  role (a necessary condition: the Director still proves each walk by an actual route);
- drive: vehicles of the roster whose owner can make car journeys to at least `destinations` distinct places
  (at least `separation_m` apart): walk from home to the access point of the car's berth, drive to a berth of the
  place, walk from that berth's access point to the place, and drive back to a home berth. These are the
  conditions of Director._journey that do not depend on time or occupancy.

A refusal names each unmet demand with the declared or measured number. Nothing is substituted: no straight-line or
shortened routes, no relabelled berths, no fewer trips.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from awr.life.civil.activity.director import supports_activity
from awr.life.common.schema import schema_id

ENVIRONMENT_SCHEMA = "awr.life.life.environment.v1"


class CapabilityRefused(ValueError):
    def __init__(self, environment, reasons, report=None):
        self.environment, self.reasons, self.report = environment, list(reasons), report
        super().__init__(f"Environment {environment} cannot support this scenario: " + "; ".join(self.reasons))


@dataclass(frozen=True)
class Demand:
    walkers: int = 0
    walk_kinds: int = 1
    drivers: int = 0
    destinations: int = 2
    separation_m: float = 0.0

    def __post_init__(self):
        for key, minimum in (("walkers", 0), ("drivers", 0), ("walk_kinds", 1), ("destinations", 1)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if not math.isfinite(self.separation_m) or self.separation_m < 0:
            raise ValueError("separation_m must be finite and nonnegative")

    @classmethod
    def from_config(cls, cfg):
        """The demand of a scenario config: its people, the recurring-vehicle gate and the activity kinds."""
        gate = (cfg.get("life_gates") or {}).get("vehicle_recurring_vehicles_min") or {}
        drivers = int(gate.get("threshold", 0))
        engine = (cfg.get("life") or {}).get("activity_engine") or {}
        return cls(
            walkers=max(int(cfg.get("people", 0)) - drivers, 0),
            walk_kinds=int(engine.get("min_activity_kinds", 1)),
            drivers=drivers,
            destinations=int(gate.get("min_distinct_destinations", 2)),
            separation_m=float(gate.get("destination_separation_m", 0.0)),
        )


def _distinct(points, separation_m, groups=None):
    """Greedy lower bound: separated destinations, at most one per facility."""
    kept, used = [], set()
    groups = range(len(points)) if groups is None else groups
    for p, group in zip(points, groups, strict=False):
        if group not in used and all(math.dist(p, q) >= separation_m for q in kept):
            kept.append(p)
            used.add(group)
    return len(kept)


def measure(director, components, *, cache_routes=True):
    """Measured walk and drive capabilities of the admitted roster (see the module docstring): for each resident the
    number of activity kinds within walking reach, for each vehicle the places its owner can reach by car.
    `components(xy)` is the set of static walking components a route from or to xy may use (the walking router's
    source convention, e.g. LifeRouter.components_near)."""
    places = director.places
    near = {}
    routes = {}
    route_stats = dict(queries=0, cache_hits=0)

    def route(mode, start, end, entity):
        # This measurement runs synchronously on one static occupancy snapshot.
        # Many facility slots share a berth, so the same drive/return was being
        # planned repeatedly. Preserve entity, direction and exact coordinates;
        # discard the cache at return, including all negative results.
        route_stats["queries"] += 1
        key = (mode, tuple(start), tuple(end), entity)
        if cache_routes and key in routes:
            route_stats["cache_hits"] += 1
            return routes[key]
        value = director._route(mode, start, end, entity)
        if cache_routes:
            routes[key] = value
        return value

    def comps(xy):
        key = (float(xy[0]), float(xy[1]))
        if key not in near:
            near[key] = frozenset(components(key))
        return near[key]

    places_in = {}
    for p in places.values():
        for c in comps(p.xy):
            places_in.setdefault(c, set()).add(p.id)
    walk_rows = []
    car_only = 0  # residents whose role has no walking activity (cruising drivers)
    for aid, actor in sorted(director.actors.items()):
        rules = [
            a
            for a in director.activities
            if "walk" in a.modes
            and actor.role in a.roles
            and (actor.activities is None or a.purpose in actor.activities)
        ]
        if not rules and actor.vehicle is not None and actor.activities is not None:
            car_only += 1
            continue
        reachable = set().union(*(places_in.get(c, set()) for c in comps(places[actor.home].xy)))
        if actor.places is not None:  # destinations restricted by a scenario overlay
            reachable &= actor.places
        reach = {places[pid].kind for pid in reachable if any(supports_activity(places[pid], a) for a in rules)}
        walk_rows.append(dict(actor=aid, kinds=len(reach)))
    drive_rows = []
    for vid, v in sorted(director.vehicles.items()):
        actor = director.actors[v.owner]
        home, source = places[actor.home], places[v.berth]
        rules = [
            a
            for a in director.activities
            if "car" in a.modes
            and actor.role in a.roles
            and (actor.activities is None or a.purpose in actor.activities)
        ]
        row = dict(vehicle=vid, owner=actor.id, destinations=[], reason=None)  # destinations: [(place, x, y)]
        drive_rows.append(row)
        if v.berth not in home.parking:
            row["reason"] = "car_not_at_a_home_berth"
            continue
        if source.access_xy is None or route("walk", home.xy, source.access_xy, actor.id) is None:
            row["reason"] = "no_walk_to_the_car"
            continue
        reached = []
        for p in sorted(places.values(), key=lambda q: q.id):
            if not any(supports_activity(p, a) for a in rules) or not p.parking:
                continue
            for bid in p.parking:
                berth = places[bid]
                if berth.access_xy is None or bid == v.berth:
                    continue
                if (
                    route("drive", source.xy, berth.xy, vid) is not None
                    and route("walk", berth.access_xy, p.xy, actor.id) is not None
                    and (
                        bid in home.parking
                        or any(route("drive", berth.xy, places[b].xy, vid) is not None for b in home.parking)
                    )
                ):
                    reached.append(p)
                    break
        row["destinations"] = [(p.id, float(p.xy[0]), float(p.xy[1])) for p in reached]
        row["destination_groups"] = {p.id: p.group or p.id for p in reached}
    return dict(
        walk=dict(residents=len(walk_rows), kinds={r["actor"]: r["kinds"] for r in walk_rows}, car_only=car_only),
        drive=dict(vehicles=len(drive_rows), rows=drive_rows),
        measurement=route_stats,
    )


def drivers(report, destinations, separation_m=0.0):
    """Vehicles of a measurement that reach at least `destinations` places pairwise separation_m apart."""
    return [
        r["vehicle"]
        for r in report["drive"]["rows"]
        if r["reason"] is None
        and _distinct(
            [d[1:] for d in r["destinations"]],
            separation_m,
            [r.get("destination_groups", {}).get(d[0], d[0]) for d in r["destinations"]],
        )
        >= destinations
    ]


def require(report, demand, environment, declared=None, kind_minimum=None):
    """Refuse (CapabilityRefused) a demand beyond the measurement, or a declaration the measurement contradicts.

    `kind_minimum`: {actor: activity kinds} for residents admitted with their own minimum (a scenario overlay
    target whose destinations are restricted to one area); every other resident needs `demand.walk_kinds`."""
    reasons = []
    kind_minimum = kind_minimum or {}
    walkers = sum(k >= kind_minimum.get(a, demand.walk_kinds) for a, k in report["walk"]["kinds"].items())
    required = demand.walkers - report["walk"].get("car_only", 0)  # cruising drivers walk only to and from their car
    if walkers < required:
        reasons.append(
            f"walk: {walkers} residents reach places of at least {demand.walk_kinds} activity kinds, "
            f"{required} required"
        )
    able = len(drivers(report, demand.destinations, demand.separation_m))
    if able < demand.drivers:
        reasons.append(
            f"drive: {able} of {report['drive']['vehicles']} vehicles can make car journeys to "
            f"{demand.destinations} distinct destinations, {demand.drivers} required"
        )
    if declared is not None:
        claim = declared.get("drive") or {}
        measured = len(drivers(report, int(claim.get("destinations", 0)), demand.separation_m))
        if measured < int(claim.get("drivers", 0)):
            reasons.append(f"drive: declared {int(claim['drivers'])} drivers, measured {measured}")
    if reasons:
        raise CapabilityRefused(environment, reasons, report)
    return report


def load_environment(path, root=".", *, validation_only=False):
    """An environment package manifest: its layers (with SHA-256 where given) and declared capabilities."""
    path = Path(path)
    m = json.loads(path.read_text())
    candidate = schema_id(m.get("schema")) == "awr.life.environment.v2"
    if candidate and not validation_only:
        raise ValueError(f"{path}: v2 candidate requires validation_only; no runtime admission yet")
    if not candidate and schema_id(m.get("schema")) != ENVIRONMENT_SCHEMA:
        raise ValueError(f"{path}: schema {ENVIRONMENT_SCHEMA} required")
    for key in ("id", "layers") if candidate else ("id", "layers", "capabilities"):
        if key not in m:
            raise ValueError(f"{path}: {key} missing")
    for name, layer in m["layers"].items():
        if layer.get("sha256") and layer.get("local", True):
            data = (Path(root) / layer["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != layer["sha256"]:
                raise ValueError(f"{path}: layer {name} differs from its recorded hash")
    return m


def check_declared(manifest, demand):
    """Refuse a demand beyond the manifest's declared capabilities, before anything is built."""
    caps, reasons = manifest["capabilities"], []
    walk = caps.get("walk")
    if demand.walkers and not walk:
        reasons.append("walk: not declared")
    elif demand.walkers > int((walk or {}).get("residents", 0)):
        reasons.append(f"walk: {walk.get('residents', 0)} residents declared, {demand.walkers} required")
    drive = caps.get("drive") or {}
    drivers = int(drive.get("drivers", 0))
    if demand.drivers > drivers:
        reasons.append(
            f"drive: {drivers} drivers declared ({drive.get('basis', 'no evidence')}), {demand.drivers} required"
        )
    if demand.drivers and int(drive.get("destinations", 0)) < demand.destinations:
        reasons.append(
            f"drive: {drive.get('destinations', 0)} destinations per driver declared, {demand.destinations} required"
        )
    if reasons:
        raise CapabilityRefused(manifest["id"], reasons)
    return manifest
