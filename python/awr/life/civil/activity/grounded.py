"""Scene adapter for measured walking and optional owned-car activity chains.

Driving requires explicit mapped berths, access and a driving backend. Neither
indoor visits nor driving connections are invented by this adapter.
"""

import hashlib
import math
import random
import time

import numpy as np

from awr.life.civil.agenda import Agenda, Visit
from awr.life.civil.itineraries import LIFE_FACILITY_PURPOSES

from .director import Activity, Actor, Director, Place, Route, supports_activity

REGION_ROUTE_TOLERANCE_M = 1.0  # how far an overlay resident's access routes may stray outside its public region


class ReactiveAgenda(Agenda):
    """An Agenda-compatible executor with one currently selected activity.

    Completed visits remain in the audit history; future visits do not exist
    until the preceding activity completes. The inherited physical arrival,
    dwell interruption and courtesy behaviour are retained.
    """

    def __init__(self, actor, home, director, start, initial_dwell):
        super().__init__(actor, "resident", [Visit(home.id, home.xy, "pause_at_home_street", initial_dwell)], start)
        self.director = director
        self.initial_departure = start
        self.wait_logged = False

    @property
    def destination(self):
        return self.visits[min(self.index, len(self.visits) - 1)]

    def emit(self, kind, t, **kw):
        if kind != "itinerary_completed":
            super().emit(kind, t, **kw)

    def pause(self, t, reason):
        changed = super().pause(t, reason)
        if changed:
            self.director.pause(self.actor, t)
        return changed

    def resume(self, t):
        super().resume(t)
        self.director.resume(self.actor, t)

    def _choose(self, t, xy):
        began = time.perf_counter()
        try:
            intent = self.director.choose(self.actor, t, tuple(xy))
        finally:
            self.director.decision_wall_s += time.perf_counter() - began
        if intent is None:
            self.phase = "deciding"
            if not self.wait_logged:
                # Audit as unplanned between-legs time, never hide it as service.
                self.emit("activity_decision_wait", t, reason="no_due_feasible_activity")
                self.wait_logged = True
            return "wait"
        if any(s.kind not in ("walk", "activity") for s in intent.steps):
            raise ValueError("Grounded adapter has no verified resident-driver transfers")
        self.visits.append(Visit(intent.slot, intent.step.xy, intent.purpose, intent.steps[-1].duration_s))
        self.index = len(self.visits) - 1
        self.phase = "scheduled"
        self.departure_s = t
        self.wait_logged = False
        self.emit("activity_decided", t, place=intent.place, purpose=intent.purpose)
        return "next_destination" if any(s.kind == "walk" for s in intent.steps) else "wait"

    def tick(self, t, xy, speed, occupancy, arrival_radius=0.6):
        if self.paused_at is not None:
            return "paused"
        self.director.observe_departure(self.actor, tuple(xy))
        if self.phase == "deciding":
            return self._choose(t, xy)
        result = super().tick(t, xy, speed, occupancy, arrival_radius)
        intent = self.director.active.get(self.actor)
        if intent is not None:
            if intent.step.kind == "walk" and self.phase in ("servicing", "finished"):
                self.director.acknowledge(self.actor, t, tuple(xy), speed, radius=arrival_radius)
            if intent.step.kind == "activity":
                self.director.acknowledge(self.actor, t, tuple(xy), speed, radius=arrival_radius)
        if result == "finished":
            if self.actor in self.director.active:
                raise RuntimeError("Motion agenda and activity clock disagree")
            return self._choose(t, xy)
        return result

    def summary(self):
        value = super().summary()
        value.update(
            behavior="activities_v1",
            initial_departure_s=self.initial_departure,
            decisions=self.director.actors[self.actor].decisions,
        )
        return value


class GroundedActivities:
    def __init__(self, network, pools, static_free, life, seed):
        self.network = network
        self.pools = pools
        self.static_free = static_free
        self.life = life
        self.seed = seed
        self.cfg = life.get("activity_engine", {})
        # Car journeys need a verified driving map with berths, car-side access and owned cars
        # (life.driving, awr.life.civil.activity.driving_adapter); without one the town is walk only.
        self.modes = tuple(self.cfg.get("modes", ["walk"]))
        if self.modes not in (("walk",), ("walk", "drive")) or ("drive" in self.modes) != bool(life.get("driving")):
            raise ValueError("Town adapter modes: walk, or walk and drive with a life.driving map")
        self.driving_routes = None  # DrivingRoutes, set once the cars are admitted
        self.access_points = set()  # car-side anchors of the berths: walks to and from them are necessary legs
        self.walking_range_m = tuple(self.cfg.get("walking_range_m", life.get("leg_distance_m", (40.0, 450.0))))
        if (
            len(self.walking_range_m) != 2
            or not all(math.isfinite(x) for x in self.walking_range_m)
            or not 0 <= self.walking_range_m[0] < self.walking_range_m[1]
        ):
            raise ValueError("Activity walking range must have finite ordered bounds")
        self.min_activity_kinds = self.cfg.get("min_activity_kinds", 3)
        if not isinstance(self.min_activity_kinds, int) or self.min_activity_kinds < 1:
            raise ValueError("min_activity_kinds must be a positive integer")
        self.frames = {}
        self.frame_index = None
        self.indices = {}
        self.route_cache = {}
        self.static_router = None
        self.static_labels = None
        self.source_components = {}
        self.home_access_cache = {}
        self.route_stats = dict(queries=0, cache_hits=0, searches=0, search_wall_s=0.0)
        self.homes = []
        self.admission_stats = {}
        self.kind_minimum = {}  # resident -> activity kinds required instead of min_activity_kinds (overlay targets)
        places = []
        for site in pools["sites"]:
            for slot in site["slots"]:
                places.append(
                    Place(
                        slot["id"],
                        tuple(slot["xy"]),
                        site["kind"],
                        (slot["id"],),
                        group=str(site.get("facility") or site["id"]),
                        affordances=site.get("affordances", ()),
                        semantic_source=site.get("semantic_source"),
                    )
                )
        rules = []
        for kind in sorted({p.kind for p in places}):
            purpose, dwell = LIFE_FACILITY_PURPOSES[kind]
            rules.append(
                Activity(
                    purpose,
                    (kind,),
                    tuple(life.get("dwell_s", {}).get(purpose, dwell)),
                    float(self.cfg.get("recurrence_s", {}).get(purpose, 600.0)),
                    modes=tuple("car" if m == "drive" else m for m in self.modes),
                    requires=("outdoor_stay",),
                )
            )
        cruise = (life.get("driving") or {}).get("cruise") if "drive" in self.modes else None
        if cruise:
            # Cruising drivers (role 'cruiser', chosen among the car owners when the cars are admitted) drive tours
            # to any outdoor place and make a short stop there; the drive is the activity.
            rules.append(
                Activity(
                    "drive_around",
                    tuple(sorted({p.kind for p in places})),
                    tuple(cruise.get("dwell_s", (20.0, 90.0))),
                    float(cruise.get("recurrence_s", 60.0)),
                    roles=("cruiser",),
                    modes=("car",),
                    requires=("outdoor_stay",),
                    tour=True,
                )
            )
        speeds = {"walk": float(life.get("planning_speed_mps", 1.05))}
        if "drive" in self.modes:
            speeds["drive"] = float(life["driving"].get("planning_speed_mps", 6.0))
        self.director = Director(
            places,
            rules,
            self.route,
            seed=seed,
            retry_s=float(self.cfg.get("retry_s", 15.0)),
            max_wait_s=float(self.cfg.get("max_wait_s", 120.0)),
            home_dwell_s=tuple(life.get("dwell_s", {}).get("pause_at_home_street", (60.0, 180.0))),
            errands_before_home=int(self.cfg.get("errands_before_home", 2)),
            available=self.available,
            max_speed_mps=speeds,
            in_vehicle_time_factor=float((life.get("driving") or {}).get("in_vehicle_time_factor", 1.0))
            if "drive" in self.modes
            else 1.0,
            min_drive_m=float((life.get("driving") or {}).get("min_drive_m", 0.0)),
            selection=self.cfg.get("selection"),
        )

    def invalidate_routes(self):
        # Called when parked bodies change the static walkable surface. Dynamic
        # bodies never enter this cache; the motion controller checks them live.
        self.route_cache.clear()
        self.static_router = None
        self.static_labels = None
        self.source_components.clear()
        self.home_access_cache.clear()
        self.route_stats["invalidations"] = self.route_stats.get("invalidations", 0) + 1

    def update_frame(self, xy, radii, riding, yaw=None, boxes=None):
        # Snapshot of the existing controller's authoritative bodies; no pose edits. `boxes`: (n, 2) length and
        # width of the bodies that are tested as oriented boxes (vehicles, objects; NaN for people, which are
        # disks of `radii`), with headings `yaw`. A box lies within the disk of its radius.
        self.frames = dict(xy=xy, radii=radii, riding=riding, yaw=yaw, boxes=boxes)
        self.frame_index = None

    @staticmethod
    def _clearance(f, ids, xy):
        """Distance from `xy` to each body `ids`: to its oriented box, or to its disk."""
        delta = np.asarray(f["xy"])[ids] - np.asarray(xy, float)
        out = np.linalg.norm(delta, axis=1) - np.asarray(f["radii"])[ids]
        if f.get("boxes") is not None:
            half = np.asarray(f["boxes"], float)[ids] / 2
            box = np.isfinite(half).all(1)
            if box.any():
                yaw = np.asarray(f["yaw"], float)[ids][box]
                c, s = np.cos(yaw), np.sin(yaw)
                d = delta[box]
                local = np.abs(np.column_stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c]))
                out[box] = np.linalg.norm(np.maximum(local - half[box], 0.0), axis=1)
        return out

    def _frame_index(self):
        # Bodies that are not riding, in a k-d tree, with copies of the arrays it was built from: the frame's arrays
        # may be changed in place, and the index is then rebuilt.
        f, idx = self.frames, self.frame_index
        if idx is None or not all(np.array_equal(f[k], idx[k]) for k in ("xy", "radii", "riding")):
            from scipy.spatial import cKDTree

            ids = np.flatnonzero(~np.asarray(f["riding"], bool))
            idx = self.frame_index = dict(
                xy=np.array(f["xy"]),
                radii=np.array(f["radii"]),
                riding=np.array(f["riding"]),
                ids=ids,
                tree=cKDTree(np.asarray(f["xy"])[ids]) if len(ids) else None,
                reach=float(np.max(np.asarray(f["radii"])[ids])) if len(ids) else 0.0,
            )
        return idx

    def available(self, place, actor):
        if place.kind == "berth":
            # This predicate tests standing-person room. A berth is instead
            # checked by DrivingRoutes against actual car bodies and corridors;
            # the Director also checks berth reservations before planning.
            return True
        if not self.frames or actor not in self.indices:
            return True
        i = self.indices[actor]
        f = self.frames
        # The same test as on every other body that does not ride, restricted to those the k-d tree finds within the
        # largest radius (plus 1e-6 m): every other body passes it.
        idx = self._frame_index()
        if idx["tree"] is None:
            return True
        near = idx["ids"][
            idx["tree"].query_ball_point(np.asarray(place.xy, float), idx["reach"] + f["radii"][i] + 0.15 + 1e-6)
        ]
        near = near[near != i]
        if not len(near):
            return True
        return bool(np.all(self._clearance(f, near, place.xy) > f["radii"][i] + 0.15))

    def available_reference(self, place, actor):
        """The test of `available` on every body (reference for the tests)."""
        if place.kind == "berth":
            return True
        if not self.frames or actor not in self.indices:
            return True
        i = self.indices[actor]
        f = self.frames
        other = np.flatnonzero((np.arange(len(f["xy"])) != i) & ~f["riding"])
        return bool(np.all(self._clearance(f, other, place.xy) > f["radii"][i] + 0.15))

    def route_possible(self, start, end):
        """Necessary connectivity check using the planner's exact node convention.

        All <=24 candidate source connectors are retained here, including ones
        that the exact body/segment check will reject later. Thus this is a
        superset of the planner's possible sources, never a nearest-cell snap.
        """
        if not hasattr(self.network, "tree"):
            return True  # Other backends own their topology contract.
        if self.static_labels is None:
            from scipy.ndimage import label

            began = time.perf_counter()
            self.static_labels, _ = label(np.asarray(self.static_free, bool))
            self.route_stats["component_rebuilds"] = self.route_stats.get("component_rebuilds", 0) + 1
            self.route_stats["component_wall_s"] = (
                self.route_stats.get("component_wall_s", 0.0) + time.perf_counter() - began
            )
        # The planner's goal convention (WalkNetwork.goal_cell), not only the goal's own cell.
        target = (
            self.network.goal_cell(end, self.static_free)
            if hasattr(self.network, "goal_cell")
            else tuple(self.network.cell(end))
        )
        if target is None or any(q < 0 or q >= n for q, n in zip(target, self.static_labels.shape, strict=False)):
            return False
        component = int(self.static_labels[target])
        if component == 0:
            return False
        key = tuple(start)
        if key not in self.source_components:
            dist, ids = self.network.tree.query(start, k=min(24, len(self.network.xy)))
            components = {
                int(self.static_labels[tuple(self.network.nodes[int(i)])])
                for d, i in zip(np.atleast_1d(dist), np.atleast_1d(ids), strict=False)
                if d <= 1.5
            }
            components.discard(0)
            if len(self.source_components) >= 4096:
                self.source_components.clear()
            self.source_components[key] = components
        return component in self.source_components[key]

    def route(self, mode, start, end, entity):
        if mode in ("drive", "tour"):
            return None if self.driving_routes is None else self.driving_routes.route(mode, start, end, entity)
        if mode != "walk":
            return None
        if not self.walk_allowed(start, end):
            return None
        rr = self.measured_route(start, end)
        if rr is None:
            return None
        length = float(np.linalg.norm(np.diff(rr["points"], axis=0), axis=1).sum())
        return Route(
            length,
            length / float(self.life.get("planning_speed_mps", 1.05)),
            "measured-walk:" + hashlib.sha256(np.asarray(rr["points"]).tobytes()).hexdigest(),
            points=rr["points"],
        )

    def walk_allowed(self, start, end):
        """Whether a walk from start to end is within the declared walking range (for present and future geometry)."""
        lo, hi = self.walking_range_m
        d = math.dist(start, end)
        # Returning home is a necessary continuation, not a discretionary leg;
        # it may be shorter or longer than the preferred normal activity range.
        # So are the walks to and from a car (the berth's car-side anchor), up to the longest normal leg.
        is_home = any(p.kind == "home" and p.xy == end for p in self.director.places.values())
        is_access = tuple(start) in self.access_points or tuple(end) in self.access_points
        return is_home or not (d > hi or (d < lo and not is_access))

    def measured_walk(self, start, end):
        """Length of the measured static walking route from start to end (m), or None."""
        rr = self.measured_route(start, end)
        return None if rr is None else float(np.linalg.norm(np.diff(rr["points"], axis=0), axis=1).sum())

    def measured_route(self, start, end):
        self.route_stats["queries"] += 1
        if not self.route_possible(start, end):
            self.route_stats["component_rejections"] = self.route_stats.get("component_rejections", 0) + 1
            return None
        key = (tuple(start), tuple(end))
        if key not in self.route_cache:
            began = time.perf_counter()
            if hasattr(self.network, "finish_grid_path"):
                if self.static_router is None:
                    from awr.life.civil.static_walk import StaticWalkRouter

                    self.static_router = StaticWalkRouter(self.network, self.static_free, stats=self.route_stats)
                rr = self.static_router.plan(np.asarray(start), np.asarray(end))
            else:
                rr = self.network.plan(np.asarray(start), np.asarray(end), self.static_free)
            self.route_stats["search_wall_s"] += time.perf_counter() - began
            self.route_stats["searches"] += 1
            if len(self.route_cache) >= 65536:
                self.route_cache.clear()
            self.route_cache[key] = rr
        else:
            self.route_stats["cache_hits"] += 1
        return self.route_cache[key]

    def place_person(
        self, actor, positions, sizes, yaws, radius, start, region=None, places_region=None, min_kinds=None
    ):
        """Only place a home body and prove activity access; do not prewrite a day.

        From a scenario overlay: `region` (a polygon) holds the home, and the routes proving its activity access stay
        in it (within REGION_ROUTE_TOLERANCE_M): a home reached only from outside its public region puts the resident
        outside the region whenever it walks (cq3p-d8-s3 v4, T2 target person-003 32.6 m outside its region at task
        start); `places_region` (a polygon) restricts the resident's destinations to the activity places in it, and
        its activity access is proved on those alone; `min_kinds` replaces min_activity_kinds for this resident
        (recorded in `kind_minimum`)."""
        free = self.network.obstacle_mask(positions, sizes, radius, yaws, min_separation=3.0)
        candidates = list(self.pools["street"])
        allowed = None
        if region is not None:
            from shapely.geometry import Point

            candidates = [c for c in candidates if region.covers(Point(*map(float, c["xy"])))]
        if places_region is not None:
            from shapely.geometry import Point

            allowed = frozenset(
                p.id
                for p in self.director.places.values()
                if p.kind not in ("home", "berth") and places_region.covers(Point(*map(float, p.xy)))
            )
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{self.seed}:home:{actor}".encode()).digest(), "big"))
        rng.shuffle(candidates)
        candidates.sort(key=lambda p: p["kind"] != "household")
        audit = dict(
            candidates=len(candidates),
            occupied=0,
            accessibility_attempts=0,
            insufficient_connected_kinds=0,
            insufficient_verified_kinds=0,
            routes_leaving_region=0,
            budget_exhausted=False,
            admitted=False,
        )
        if region is not None:
            from shapely.geometry import LineString

            kept = region.buffer(REGION_ROUTE_TOLERANCE_M)

            def in_region(route):
                return len(route.points) >= 2 and kept.covers(LineString(route.points))

        self.admission_stats[actor] = audit
        # Bound expensive accessibility proofs, not raw shuffled map entries.
        # At high density the old first-200 slice was mostly occupied homes;
        # it rejected later residents without testing any free candidate.
        attempted = 0
        for home in candidates:
            xy = np.asarray(home["xy"])
            if not free[tuple(self.network.cell(xy))] or any(np.linalg.norm(xy - p) < 3.0 for p in self.homes):
                audit["occupied"] += 1
                continue
            # Population admission follows the map's actual affordances. A home
            # connected only to one cafe must not get fictitious needs at seven
            # disconnected facilities, nor produce a cafe/home ping-pong loop.
            key = tuple(xy) if allowed is None else (tuple(xy), allowed)
            if key not in self.home_access_cache:
                lo, hi = self.walking_range_m
                self.home_access_cache[key] = sorted(
                    (
                        p
                        for p in self.director.places.values()
                        if p.kind != "home"
                        and (allowed is None or p.id in allowed)
                        and any(supports_activity(p, a) for a in self.director.activities)
                        and lo <= math.dist(xy, p.xy) <= hi
                        and self.route_possible(tuple(xy), p.xy)
                    ),
                    key=lambda p: (math.dist(xy, p.xy), p.id),
                )
            nearby = self.home_access_cache[key]
            minimum = self.min_activity_kinds if min_kinds is None else int(min_kinds)
            if len({p.kind for p in nearby}) < minimum:
                audit["insufficient_connected_kinds"] += 1
                continue
            # Cheap topology/distance exclusions must not spend the finite
            # exact-route budget. Otherwise the first 200 disconnected homes
            # hide accessible candidates later in the same measured map.
            if attempted >= 200:
                audit["budget_exhausted"] = True
                break
            attempted += 1
            audit["accessibility_attempts"] = attempted
            witnesses = {}
            for place in nearby:
                if place.kind in witnesses:
                    continue
                outward = self.route("walk", tuple(xy), place.xy, actor)
                back = self.route("walk", place.xy, tuple(xy), actor) if outward is not None else None
                if (
                    outward is not None
                    and back is not None
                    and region is not None
                    and not (in_region(outward) and in_region(back))
                ):
                    audit["routes_leaving_region"] += 1
                    continue
                if outward is not None and back is not None:
                    witnesses[place.kind] = dict(
                        place=place.id,
                        outward=outward.evidence,
                        return_route=back.evidence,
                        distance_m=outward.length_m,
                    )
            if len(witnesses) < minimum:
                audit["insufficient_verified_kinds"] += 1
                continue
            purposes = tuple(
                a.purpose
                for a in self.director.activities
                if "resident" in a.roles and any(k in witnesses for k in a.kinds)
            )
            site = Place(actor + "-home", tuple(xy), "home", (actor + "-home",))
            self.director.places[site.id] = site
            self.director.add_actor(Actor(actor, site.id, activities=purposes, places=allowed), t=start)
            if min_kinds is not None:
                self.kind_minimum[actor] = int(min_kinds)
            # Initial state is a sampled ongoing outdoor home activity, not a
            # departure schedule stretched to fit the recording window.
            initial = rng.uniform(*self.director.home_dwell_s)
            if "drive" in self.modes:
                from awr.life.motion.agenda import LifeAgenda  # also executes the Director's car journeys

                agenda = LifeAgenda(actor, site, self.director, start, initial)
            else:
                agenda = ReactiveAgenda(actor, site, self.director, start, initial)
            self.homes.append(xy.copy())
            audit.update(admitted=True, home_kind=home["kind"], verified_kinds=sorted(witnesses))
            route = dict(road="home:" + actor, points=np.array([xy, xy]))
            return (
                route,
                agenda,
                dict(
                    home=site.id,
                    home_xy=xy.tolist(),
                    behavior="activities_v1",
                    home_kind="outdoor_approach",
                    planned_visits=1,
                    walking_range_m=list(self.walking_range_m),
                    activity_access=witnesses,
                    activity_purposes=list(purposes),
                    unavailable_purposes=[a.purpose for a in self.director.activities if a.purpose not in purposes],
                ),
            )
        return None, None, None
